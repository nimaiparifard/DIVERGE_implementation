import torch 
import os
import json
from collections import defaultdict
from torch_geometric.utils import to_undirected


def re_split_data(num_node, train_percent=0.6, val_percent=0.2, test_percent=0.2, device="cuda:0", seed=None):
    import numpy as np
    
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
    
    node_ids = np.arange(num_node)
    np.random.shuffle(node_ids)
    
    train_ids = np.sort(node_ids[:int(num_node * train_percent)])
    val_ids = np.sort(node_ids[int(num_node * train_percent): int(num_node * (train_percent + val_percent))])
    test_ids = np.sort(node_ids[int(num_node * (train_percent + val_percent)): int(num_node * (train_percent + val_percent + test_percent))])
    
    def to_mask(ids):
        mask = np.zeros(num_node, dtype=bool)
        mask[ids] = True
        return torch.from_numpy(mask)

    return to_mask(train_ids).to(device), to_mask(val_ids).to(device), to_mask(test_ids).to(device)


def is_seeded_split(dataset_name, re_split):
    """True when the train/val/test masks are drawn from `seed` (False = the dataset's own fixed split)."""
    return bool(re_split) or dataset_name in ["computer", "history", "photo"]


def compute_split(dataset_name, num_nodes, re_split, device, seed=None):
    if dataset_name in ["computer", "history", "photo"] and not re_split:
        # Semi-supervised setting for Computer, History, and Photo (10%:10%:60% for train:val:test)
        return re_split_data(num_nodes, train_percent=0.1, val_percent=0.1, test_percent=0.6, device=device, seed=seed)
    return re_split_data(num_nodes, device=device, seed=seed)


# Splits exported by scripts/export_seeds.py; copied to another machine they pin the exact
# train/val/test nodes of every (dataset, split, seed). Set DIVERGE_IGNORE_SAVED_SPLITS=1 to bypass.
SAVED_SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seed_snapshot", "splits")


def saved_split_path(dataset_name, re_split, seed):
    return os.path.join(SAVED_SPLITS_DIR, f"{dataset_name}_split{int(bool(re_split))}_seed{int(seed)}.npz")


def load_saved_split(dataset_name, num_nodes, re_split, device, seed):
    if seed is None or os.environ.get("DIVERGE_IGNORE_SAVED_SPLITS") == "1":
        return None
    path = saved_split_path(dataset_name, re_split, seed)
    if not os.path.exists(path):
        return None
    import numpy as np
    saved = np.load(path)
    if int(saved["num_nodes"]) != num_nodes:
        raise ValueError(f"Saved split {path} is for {int(saved['num_nodes'])} nodes, dataset has {num_nodes}")
    # Leave numpy's global RNG exactly where re_split_data would have left it
    np.random.seed(seed)
    np.random.shuffle(np.arange(num_nodes))
    masks = []
    for key in ("train_idx", "val_idx", "test_idx"):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(saved[key].astype(np.int64))] = True
        masks.append(mask.to(device))
    return tuple(masks)


def load_graph_dataset(dataset_name, device, re_split=False, path_prefix="../..", seed=None, modified_dataset=None):
    if modified_dataset:
        graph_data = modified_dataset
    else:
        graph_data = torch.load(f"{path_prefix}/datasets/{dataset_name}.pt", weights_only=False).to(device)
    # Alternative
    # graph_data.edge_index = to_undirected(graph_data.edge_index) if dataset_name in ["citeseer", "arxiv"] else graph_data.edge_index
    graph_data.edge_index = to_undirected(graph_data.edge_index)

    # Skip re-splitting if we have a modified dataset (it already has the correct masks)
    # if not modified_dataset:
    if is_seeded_split(dataset_name, re_split):
        # Augmented graphs append synthetic nodes after the original ones (num_base_nodes). Only the
        # original nodes are split - exactly as for the un-augmented graph, so the val/test nodes stay
        # the ones the LLM never trained on - and the appended nodes are train-only (never val/test).
        num_base = int(getattr(graph_data, 'num_base_nodes', None) or graph_data.num_nodes)
        masks = load_saved_split(dataset_name, num_base, re_split, device, seed)
        if masks is None:
            masks = compute_split(dataset_name, num_base, re_split, device, seed=seed)
        n_extra = graph_data.num_nodes - num_base
        if n_extra > 0:
            train, val, test = masks
            ones = torch.ones(n_extra, dtype=torch.bool, device=train.device)
            zeros = torch.zeros(n_extra, dtype=torch.bool, device=train.device)
            masks = (torch.cat([train, ones]), torch.cat([val, zeros]), torch.cat([test, zeros]))
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = masks

    return graph_data


def load_graph_dataset_for_gnn(dataset_name, device, re_split=False, path_prefix="../..", emb_model="shallow"):
    graph_data = load_graph_dataset(dataset_name, device, re_split, path_prefix)

    if emb_model != "shallow":
        assert os.path.exists(f"{path_prefix}/datasets/{emb_model}/{dataset_name}.pt")
        node_feat = torch.load(f"{path_prefix}/datasets/{emb_model}/{dataset_name}.pt", map_location=device, weights_only=False).to(device).type(torch.float)
        graph_data.x = node_feat
    
    # TODO: check datasets that need shallow embedding
    # Apply Node2Vec for datasets without shallow embeddings
    if emb_model == "shallow" and dataset_name in ["reddit", "instagram", "computer", "photo", "history"]:
        if os.path.exists(f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt"):
            node_feat = torch.load(f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt", map_location=device).to(device)
        else:
            from node2vec import Node2Vec
            from torch_geometric.utils.convert import to_networkx
        
            nx_graph = to_networkx(graph_data)
            node2vec = Node2Vec(nx_graph, dimensions=300, walk_length=30, num_walks=10, workers=4)
            node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
            print(node2vec_model.wv.vectors.shape, type(node2vec_model.wv.vectors))
            node_feat = torch.FloatTensor(node2vec_model.wv.vectors).to(device)
            os.makedirs(f"{path_prefix}/datasets/Node2Vec", exist_ok=True)
            torch.save(node_feat, f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt")
        graph_data.x = node_feat
    
    return graph_data


def get_gpt_location(dataset_name, gpt_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_path = os.path.join(repo_root, "datasets")
    got_responses_path = os.path.join(data_path, gpt_name)
    gpt_dataset_path = os.path.join(got_responses_path, f'{dataset_name}.json')
    return gpt_dataset_path


def load_graph_dataset_for_tape(dataset_name, device, re_split=False, use_gpt=False, gpt_name="Mistral-7B", path_prefix='../..', seed=None, modified_dataset=None):
    graph_data = load_graph_dataset(dataset_name, device, re_split, path_prefix=path_prefix, seed=seed, modified_dataset=modified_dataset)

    if use_gpt:
        prediction_file = get_gpt_location(dataset_name, "gpt_4o")
        if not os.path.exists(prediction_file):
            print("LLM generated explanations do not exist, use original text instead.")
        else:
            file_reader = open(prediction_file, 'r')
            id2exp = defaultdict(str)
            for line in file_reader:
                content = json.loads(line)
                answer = content["answer"]
                # TODO: check whether it needs to replace prediction part
                # if "Explanation:" in content["answer"]:
                #     answer = content["answer"][content["answer"].index("Explanation:"):]
                # else:
                #     answer = content["answer"]
                id2exp[content["idx"]] = answer
            
            raw_texts = [id2exp.get(idx, graph_data.raw_texts[idx]) for idx in range(graph_data.num_nodes)]
            return graph_data, len(graph_data.label_name), raw_texts

    return graph_data, len(graph_data.label_name), graph_data.raw_texts


def load_graph_dataset_for_llaga(dataset_name, device, re_split=False, encoder="roberta"):
    graph_data = load_graph_dataset_for_gnn(dataset_name, device, re_split, path_prefix="../..", emb_model=encoder)

    return graph_data
