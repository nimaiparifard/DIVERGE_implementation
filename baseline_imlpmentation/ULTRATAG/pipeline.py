"""
End-to-end UltraTAG-S pipeline orchestration: builds the LLM-augmented graph once
per (dataset, sparsity_ratio) -- caching the expensive LLM-generation steps -- then
runs the (cheap, stochastic) LM finetuning + dual-GNN training per seed on top of it.
"""
import json
import os

import torch

from common import load_graph_dataset_for_tape, set_seed
from common.dataloader import re_split_data

from ULTRATAG.sparsity import simulate_sparsity
from ULTRATAG.text_augmentation import propagate_texts, augment_texts
from ULTRATAG.structure_augmentation import add_virtual_edges, pagerank_important_nodes, reconfigure_edges
from ULTRATAG.lm_finetune import finetune_lm_and_extract_embeddings
from ULTRATAG.dual_gnn import DualGNNTrainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

VIRTUAL_EDGE_PERCENTILE = 1.0  # see structure_augmentation.build_virtual_edges docstring for why
                                # this replaces the paper's fixed tau_1=0.8 threshold
EDGE_RECONFIG_THRESHOLD = 0.5  # tau_2, paper Section 5.4
IMPORTANT_NODE_RATIO = 0.10    # paper Section 5.4
TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT = 0.6, 0.2, 0.2


def resolve_path_prefix():
    cwd = os.path.abspath(os.getcwd())
    if cwd == os.path.abspath(REPO_ROOT):
        return "."
    return os.path.normpath(os.path.relpath(REPO_ROOT, start=cwd))


def load_split_graph(dataset_name, device, seed):
    path_prefix = resolve_path_prefix()
    data, num_classes, raw_texts = load_graph_dataset_for_tape(
        dataset_name, device, re_split=False, path_prefix=path_prefix, seed=seed,
    )
    data.y = data.y.squeeze()
    data.train_mask, data.val_mask, data.test_mask = re_split_data(
        data.num_nodes, TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT, device=device, seed=seed,
    )
    return data, num_classes, raw_texts


def get_augmentation_cache_path(dataset_name, ratio, augmentation_seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{dataset_name}_ratio{int(ratio * 100)}_augseed{augmentation_seed}"
    return os.path.join(CACHE_DIR, f"{tag}_augmented_graph.pt")


def build_augmented_graph(dataset_name, ratio, augmentation_seed, generator, embedder, device, force_rebuild=False):
    """
    Modules 1 (LLM-based Robustness Enhancement): sparsify -> propagate -> LLM
    text-augment -> virtual edges -> PageRank node selection -> LLM edge
    reconfiguration. Cached per (dataset, ratio, augmentation_seed) since every
    step here involves LLM generation calls and is far more expensive than the
    downstream LM-finetuning/dual-GNN training that varies per seed.
    """
    cache_path = get_augmentation_cache_path(dataset_name, ratio, augmentation_seed)
    if os.path.exists(cache_path) and not force_rebuild:
        return torch.load(cache_path, weights_only=False)

    set_seed(augmentation_seed)
    data, num_classes, raw_texts = load_split_graph(dataset_name, device, augmentation_seed)
    label_names = list(data.label_name)

    sparse_texts, sparse_edge_index, deleted_mask = simulate_sparsity(
        data.edge_index.cpu(), raw_texts, data.num_nodes, ratio, augmentation_seed,
    )

    propagated_texts = propagate_texts(sparse_texts, sparse_edge_index, data.num_nodes)

    # Separate checkpoint for the O(N) text-generation step specifically -- by far
    # the longest-running part of augmentation -- so a crash mid-way (see
    # ULTRATAG/generation.py's `generate()` docstring) resumes instead of restarting.
    text_checkpoint_path = cache_path.replace("_augmented_graph.pt", "_text_augmentation_checkpoint.pt")
    augmented_texts, soft_labels = augment_texts(
        propagated_texts, dataset_name, label_names, generator, checkpoint_path=text_checkpoint_path,
    )

    text_embeddings = embedder.encode(augmented_texts, show_progress=True)
    edge_index_virtual = add_virtual_edges(
        sparse_edge_index, text_embeddings, soft_labels, top_percentile=VIRTUAL_EDGE_PERCENTILE,
    )

    num_train_nodes = int(data.train_mask.sum().item())
    important_nodes = pagerank_important_nodes(
        edge_index_virtual, data.num_nodes, num_train_nodes, top_ratio=IMPORTANT_NODE_RATIO,
    )

    final_edge_index = reconfigure_edges(
        edge_index_virtual, important_nodes, augmented_texts, generator, dataset_name,
        threshold=EDGE_RECONFIG_THRESHOLD,
    )

    bundle = {
        "dataset_name": dataset_name,
        "ratio": ratio,
        "augmentation_seed": augmentation_seed,
        "augmented_texts": augmented_texts,
        "soft_labels": soft_labels,
        "final_edge_index": final_edge_index.cpu(),
        "important_nodes": sorted(important_nodes),
        "train_mask": data.train_mask.cpu(),
        "val_mask": data.val_mask.cpu(),
        "test_mask": data.test_mask.cpu(),
        "y": data.y.cpu(),
        "num_classes": num_classes,
        "deleted_text_mask": deleted_mask,
    }
    torch.save(bundle, cache_path)
    if os.path.exists(text_checkpoint_path):
        os.remove(text_checkpoint_path)
    return bundle


class SimpleData:
    """Minimal PyG-Data-like container (mask/label tensors + a real .to())."""

    def __init__(self, train_mask, val_mask, test_mask, y):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.y = y

    def to(self, device):
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.y = self.y.to(device)
        return self


def run_downstream(bundle, llm_name, seed, device, hp):
    """Modules 2+3 (LM finetuning + dual-GNN), the stochastic part varied per seed."""
    set_seed(seed)
    dataset_name = bundle["dataset_name"]
    augmented_texts = bundle["augmented_texts"]
    labels = bundle["y"]

    _, _, embeddings = finetune_lm_and_extract_embeddings(
        dataset_name, augmented_texts, labels,
        bundle["train_mask"], bundle["val_mask"],
        llm_name=llm_name,
        lr=hp["lm_lr"], epochs=hp["lm_epochs"], batch_size=hp["lm_batch_size"], dropout=hp["lm_dropout"],
        run_tag=f"ratio{int(bundle['ratio'] * 100)}_seed{seed}",
    )

    num_nodes = embeddings.shape[0]
    important_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if bundle["important_nodes"]:
        important_mask[torch.tensor(bundle["important_nodes"], dtype=torch.long)] = True

    data = SimpleData(bundle["train_mask"], bundle["val_mask"], bundle["test_mask"], labels)

    trainer = DualGNNTrainer(
        num_node_features=embeddings.shape[1],
        num_classes=bundle["num_classes"],
        augmented_edge_index=bundle["final_edge_index"],
        important_mask=important_mask,
        hidden_dim=hp["gnn_hidden_dim"],
        dropout=hp["gnn_dropout"],
        lr=hp["gnn_lr"],
        weight_decay=hp["gnn_weight_decay"],
        epochs=hp["gnn_epochs"],
        topk=hp["gnn_topk"],
        device=device,
    )
    results = trainer.train(embeddings, data, patience=hp["gnn_patience"])
    return results
