# CKA between GNN node embeddings (penultimate layer, before the classification logits)
# of GNNs trained on each LoRA-init LLM embedding, for one dataset / llm / seed.
r"""
Counterpart of experiments/cka.py, one level further down the pipeline.

experiments/cka.py compares the *LLM* embeddings produced by each LoRA-init
approach. Here we first train one GNN per approach on those embeddings (same
tuned hyperparameters, same seed and same train/val/test split for all of them),
then take each GNN's node representation right before the last graph conv (the
layer that outputs the class logits), and compute the 5x5 linear-CKA matrix
between those representations.

This tells whether the diversity seen in the LLM embeddings survives GNN
training: if the GNN embeddings of different approaches still have low CKA, the
GNN ensemble members really do encode the graph differently.

Additionally, for every approach, CKA(LLM embedding, GNN embedding) is reported:
how much the GNN changed the geometry of its input features.
"""
import os
from datetime import datetime

import torch

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import set_mgnn_cfg, GNNTrainer, append_gnn_result_csv
from report.reporter import ReportResults
from experiments.cka import APPROACHES, TITLES, linear_cka, compute_cka_matrix, save_cka_heatmap

NODE_SUBSETS = ['all', 'train', 'val', 'test']


@torch.no_grad()
def extract_gnn_penultimate_embedding(model, features, edge_index) -> torch.Tensor:
    """
    Node representation fed into the last graph conv of GNNEncoder, i.e. the
    output of the last hidden layer (after batch-norm/activation, and after
    jump-knowledge aggregation if enabled). The last conv maps it to the class
    logits, so this is the GNN's "embedding before logits". Captured with a
    forward pre-hook, so it stays correct for any GNNEncoder configuration.
    """
    captured = {}

    def hook(_module, inputs):
        captured['x'] = inputs[0].detach()

    handle = model.conv_layers[-1].register_forward_pre_hook(hook)
    try:
        model.eval()
        model(features, edge_index)
    finally:
        handle.remove()
    return captured['x']


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         approaches=None, gnn_model_name=None, node_subset='all', reporter_index=0,
         csv_path='results/cka_gnn/cka_gnn_pairwise_results.csv',
         summary_csv_path='results/cka_gnn/cka_gnn_summary_results.csv',
         per_approach_csv_path='results/cka_gnn/cka_gnn_per_approach_results.csv',
         save_heatmap=True, heatmap_dir='results/cka_gnn', save_embeddings=False,
         embeddings_dir='results/cka_gnn/embeddings', does_print_training_process=False):
    """
    Train one GNN per LoRA-init approach on its cached LLM embeddings, extract
    the penultimate-layer GNN node embeddings, and compute the pairwise
    linear-CKA matrix between them, for one (dataset, llm, seed) combination.

    Args:
        seed: Seed of the LoRA adapters / embedding caches, also used as the GNN
              training seed (so every approach gets the same split and init RNG).
              If None, uses configs/dataset/<name>.json's seed.
        supervised: True for the supervised embedding cache / GNN split, False
              for the semi-supervised (seed-tagged) one.
        gnn_model_name: GNN encoder; None keeps the one from the tuned
              hyperparameter JSON (SAGE by default).
        node_subset: which nodes' embeddings enter the CKA ('all', 'train',
              'val' or 'test'). The split is identical across approaches.
        csv_path: long-format CSV, one row per ordered approach pair.
        summary_csv_path: one row per run with the mean off-diagonal CKA and the
              most/least similar pair.
        per_approach_csv_path: one row per approach with its GNN accuracy and
              CKA(LLM embedding, GNN embedding).
        save_embeddings: also save the extracted GNN embeddings as .pt files.

    Returns:
        dict with the CKA matrix (keyed by (approach_i, approach_j)), per-approach
        GNN results and the diversity summary.
    """
    approaches = approaches or APPROACHES
    if len(approaches) < 2:
        raise ValueError("Need at least 2 approaches to compute pairwise CKA")
    if node_subset not in NODE_SUBSETS:
        raise ValueError(f"node_subset must be one of {NODE_SUBSETS}, got '{node_subset}'")

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    gnn_cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=resolved_seed, llm_name=llm_name,
                           gnn_model_name=gnn_model_name)
    gnn_name = gnn_cfg.gnn_model_name

    print("=" * 80)
    print(f"GNN-embedding CKA | dataset={dataset_name} llm={llm_name} peft={peft_type} seed={resolved_seed} "
          f"supervised={supervised} gnn={gnn_name} nodes={node_subset} approaches={approaches}")
    print("=" * 80)

    reporter = ReportResults(
        cfg,
        save_dir=f"results/train_info/{dataset_name}_cka_gnn_seed{resolved_seed}",
        index_run=reporter_index,
    )
    reporter.report_title("CKA (Centered Kernel Alignment) - GNN Embedding Diversity Analysis")
    reporter.report_txt(f"Dataset: {dataset_name}")
    reporter.report_txt(f"LLM: {llm_name}")
    reporter.report_txt(f"PEFT Type: {peft_type}")
    reporter.report_txt(f"Seed: {resolved_seed}")
    reporter.report_txt(f"Supervised: {supervised}")
    reporter.report_txt(f"GNN: {gnn_name} (layers={gnn_cfg.num_layers}, hidden_dim={gnn_cfg.hidden_dim}, "
                        f"hyperparameters from: {gnn_cfg.hp_source})")
    reporter.report_txt(f"Embedding layer: input of the last graph conv (before the classification logits)")
    reporter.report_txt(f"Nodes used for CKA: {node_subset}")
    reporter.report_txt(f"Approaches: {[TITLES[a] for a in approaches]}")
    reporter.report_txt("")
    if gnn_cfg.num_layers < 2:
        msg = ("WARNING: num_layers=1, so the layer before the logits is the raw LLM input embedding; "
               "the GNN CKA will equal the LLM CKA.")
        print(msg)
        reporter.report_txt(msg)

    # Load the seed-tagged LLM embeddings for every requested init approach
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    llm_emb = {name: get_embedding_from_data(all_data[name]) for name in approaches}
    print("Loaded LLM embeddings: " + ", ".join(f"{name}={tuple(llm_emb[name].shape)}" for name in approaches))

    # Train one GNN per approach and extract its penultimate-layer node embeddings
    gnn_emb, gnn_results, node_mask = {}, {}, None
    for name in approaches:
        print(f"\n--- Training {gnn_name} on {TITLES[name]} embeddings ---")
        trainer = GNNTrainer(gnn_cfg, llm_emb[name], does_print_training_process=does_print_training_process)
        results, _, _ = trainer.train()
        gnn_results[name] = results
        emb = extract_gnn_penultimate_embedding(trainer.model, trainer.features, trainer.graph)
        if node_subset != 'all':
            mask = getattr(trainer.data, f"{node_subset}_mask").detach().cpu()
            if node_mask is not None and not torch.equal(mask, node_mask):
                raise RuntimeError(f"{node_subset} split differs across approaches; CKA needs the same nodes")
            node_mask = mask
            emb = emb[mask.to(emb.device)]
        gnn_emb[name] = emb.cpu()
        print(f"{TITLES[name]}: test_acc={results['test_acc']:.2f} val_acc={results['val_acc']:.2f} "
              f"gnn_embedding={tuple(gnn_emb[name].shape)}")

    # CKA between each approach's LLM input and its GNN embedding (same nodes)
    llm_vs_gnn = {}
    for name in approaches:
        x = llm_emb[name].detach().cpu()
        if node_mask is not None:
            x = x[node_mask]
        llm_vs_gnn[name] = linear_cka(x, gnn_emb[name])

    if save_embeddings:
        os.makedirs(embeddings_dir, exist_ok=True)
        for name in approaches:
            path = os.path.join(embeddings_dir, f"{dataset_name}_{llm_name}_{peft_type}_{gnn_name}_"
                                                f"init-{name}_seed{resolved_seed}_{node_subset}.pt")
            torch.save(gnn_emb[name], path)
        print(f"[OK] GNN embeddings saved to: {embeddings_dir}")

    # Pairwise CKA matrix between GNN embeddings
    cka_matrix = compute_cka_matrix(gnn_emb, approaches)

    # ---- report per-approach GNN results ----
    reporter.report_title("Per-approach GNN results")
    header = f"{'Approach':<12}{'Test Acc':>10}{'Val Acc':>10}{'Test F1':>10}{'CKA(LLM,GNN)':>14}"
    print("\n" + header)
    reporter.report_txt(header)
    for name in approaches:
        r = gnn_results[name]
        line = (f"{TITLES[name]:<12}{r['test_acc']:>10.2f}{r['val_acc']:>10.2f}{r['test_f1']:>10.2f}"
                f"{llm_vs_gnn[name]:>14.4f}")
        print(line)
        reporter.report_txt(line)
    reporter.report_txt("")

    # ---- report as a formatted matrix ----
    reporter.report_title("GNN Embedding CKA Matrix")
    header = " " * 12 + "".join(f"{TITLES[a]:>12}" for a in approaches)
    print("\n" + header)
    reporter.report_txt(header)
    for a in approaches:
        row = f"{TITLES[a]:<12}" + "".join(f"{cka_matrix[(a, b)]:>12.4f}" for b in approaches)
        print(row)
        reporter.report_txt(row)
    reporter.report_txt("")

    # ---- diversity summary: mean off-diagonal CKA (lower = more diverse ensemble) ----
    off_diag_pairs = [(a, b) for a in approaches for b in approaches if a != b]
    off_diag_values = [cka_matrix[p] for p in off_diag_pairs]
    mean_off_diag = sum(off_diag_values) / len(off_diag_values)
    most_similar_pair = max(off_diag_pairs, key=lambda p: cka_matrix[p])
    most_diverse_pair = min(off_diag_pairs, key=lambda p: cka_matrix[p])

    summary_lines = [
        f"Mean off-diagonal CKA (lower = more diverse ensemble members): {mean_off_diag:.4f}",
        f"Most similar pair (highest CKA):  {TITLES[most_similar_pair[0]]} vs {TITLES[most_similar_pair[1]]} "
        f"= {cka_matrix[most_similar_pair]:.4f}",
        f"Most diverse pair (lowest CKA):   {TITLES[most_diverse_pair[0]]} vs {TITLES[most_diverse_pair[1]]} "
        f"= {cka_matrix[most_diverse_pair]:.4f}",
    ]
    reporter.report_title("Diversity Summary")
    for line in summary_lines:
        print(line)
        reporter.report_txt(line)

    # ---- heatmap visualization (paper figure) ----
    heatmap_path = None
    if save_heatmap:
        heatmap_path = os.path.join(
            heatmap_dir, f"{dataset_name}_{llm_name}_{peft_type}_{gnn_name}_seed{resolved_seed}_{node_subset}"
                         f"_cka_gnn_heatmap.png")
        save_cka_heatmap(cka_matrix, approaches, heatmap_path,
                         title=f"Linear CKA between {gnn_name} embeddings (before logits)")
        reporter.report_txt(f"\nCKA heatmap saved to: {heatmap_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_info = {
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "llm_name": llm_name,
        "peft_type": peft_type,
        "seed": resolved_seed,
        "supervised": supervised,
        "gnn_model_name": gnn_name,
        "num_layers": gnn_cfg.num_layers,
        "hidden_dim": gnn_cfg.hidden_dim,
        "node_subset": node_subset,
        "hp_source": os.path.basename(gnn_cfg.hp_source),
    }

    # ---- CSV: one row per ordered pair (long format, easy to pivot/aggregate) ----
    if csv_path:
        for a in approaches:
            for b in approaches:
                row = dict(run_info, approach_i=TITLES[a], approach_j=TITLES[b], cka=cka_matrix[(a, b)])
                append_gnn_result_csv(row, csv_path)
        print(f"[OK] GNN CKA pairwise results appended to: {csv_path}")

    # ---- CSV: one row per approach (GNN accuracy + CKA between its LLM and GNN embeddings) ----
    if per_approach_csv_path:
        for name in approaches:
            r = gnn_results[name]
            row = dict(run_info, approach=TITLES[name],
                       test_accuracy=r["test_acc"], test_macro_f1=r["test_f1"],
                       val_accuracy=r["val_acc"], val_macro_f1=r["val_f1"],
                       epochs_run=r["epochs_run"], llm_vs_gnn_cka=llm_vs_gnn[name])
            append_gnn_result_csv(row, per_approach_csv_path)
        print(f"[OK] GNN per-approach results appended to: {per_approach_csv_path}")

    # ---- CSV: one summary row per run ----
    if summary_csv_path:
        summary_row = dict(
            run_info,
            approaches=",".join(TITLES[a] for a in approaches),
            mean_off_diagonal_cka=mean_off_diag,
            most_similar_pair=f"{TITLES[most_similar_pair[0]]}-{TITLES[most_similar_pair[1]]}",
            most_similar_cka=cka_matrix[most_similar_pair],
            most_diverse_pair=f"{TITLES[most_diverse_pair[0]]}-{TITLES[most_diverse_pair[1]]}",
            most_diverse_cka=cka_matrix[most_diverse_pair],
            mean_test_accuracy=sum(gnn_results[a]["test_acc"] for a in approaches) / len(approaches),
            heatmap_path=heatmap_path or "",
        )
        append_gnn_result_csv(summary_row, summary_csv_path)
        print(f"[OK] GNN CKA summary appended to: {summary_csv_path}")

    return {
        "cka_matrix": cka_matrix,
        "approaches": approaches,
        "gnn_results": gnn_results,
        "llm_vs_gnn_cka": llm_vs_gnn,
        "mean_off_diagonal_cka": mean_off_diag,
        "most_similar_pair": most_similar_pair,
        "most_diverse_pair": most_diverse_pair,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train one GNN per LoRA-init embedding set and compute the pairwise linear-CKA matrix "
                    "between their penultimate-layer (before logits) node embeddings for one dataset/llm/seed")
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches, also used as the GNN seed')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embedding cache/GNN split, 0 = semi-supervised (seed-tagged) one')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                         help=f"Comma-separated subset of {APPROACHES} to include")
    parser.add_argument('--gnn_model_name', type=str, default=None,
                         help='GNN encoder (default: the one in the tuned hyperparameter JSON, SAGE otherwise)')
    parser.add_argument('--node_subset', type=str, default='all', choices=NODE_SUBSETS,
                         help='Which nodes enter the CKA computation')
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/cka_gnn/cka_gnn_pairwise_results.csv')
    parser.add_argument('--summary_csv_path', type=str, default='results/cka_gnn/cka_gnn_summary_results.csv')
    parser.add_argument('--per_approach_csv_path', type=str,
                         default='results/cka_gnn/cka_gnn_per_approach_results.csv')
    parser.add_argument('--no_heatmap', action='store_true', help='Skip saving the CKA heatmap PNG')
    parser.add_argument('--save_embeddings', action='store_true', help='Also save the extracted GNN embeddings')
    parser.add_argument('--print_training', action='store_true', help='Print GNN training progress')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        approaches=[a.strip() for a in args.approaches.split(',') if a.strip()],
        gnn_model_name=args.gnn_model_name,
        node_subset=args.node_subset,
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
        summary_csv_path=args.summary_csv_path,
        per_approach_csv_path=args.per_approach_csv_path,
        save_heatmap=not args.no_heatmap,
        save_embeddings=args.save_embeddings,
        does_print_training_process=args.print_training,
    )
