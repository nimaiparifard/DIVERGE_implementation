"""
TAPE end-to-end experiment runner.

Reproduces the paper's core pipeline (Section 4, Eq. 4-7): fine-tune LM_orig and
LM_expl (Eq. 4-5) via this repo's existing LoRA infrastructure, build ranked-
prediction features h_pred (Eq. 5), then for each of {MLP, GCN, SAGE, RevGAT}
independently train that architecture on each of {h_orig, h_expl, h_pred} and
mean-pool their softmax predictions into that architecture's h_TAPE score
(Eq. 6-7) -- matching the paper's Table 1 (h_TAPE column) / Table 3 (ablation) layout.

This repo already has full LLM-response coverage cached for both Cora
(`datasets/gpt_4o/cora.json`) and PubMed (`datasets/PubMed/{i}.json`), so unlike
STAGE/ULTRATAG this pipeline makes no new LLM API/generation calls at all -- it
reuses those existing outputs (see TAPE/prediction_parser.py for why PubMed's
older-format cache needs a slightly different parse than Cora's).

Usage (from the repo root):
    python -m TAPE.run_tape --dataset cora   --seeds 42 43 44 45
    python -m TAPE.run_tape --dataset pubmed --seeds 42 43 44 45
"""
import argparse
import json
import os
import statistics
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common import load_graph_dataset_for_tape, set_seed
from common.dataloader import re_split_data
from common.metrics import compute_acc_and_f1
from config import setup_finetuning_cfg
from dataset.dataset_loader import load_gpt_enhanced_explanations

from STAGE.ensemble import ensemble_mean_probs
from STAGE.models import MODEL_BUILDERS
from STAGE.trainer import StageTrainer

from TAPE.finetune import finetune_and_extract
from TAPE.pred_features import build_h_pred
from TAPE.prediction_parser import parse_llm_response

TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT = 0.6, 0.2, 0.2

# Per-architecture GNN hyperparameters, taken directly from the paper's Table 10
# (GCN/SAGE/RevGAT); MLP is not covered by the paper's table, so it reuses GCN/SAGE's
# values as a reasonable default.
GNN_HYPERPARAMS = {
    "MLP":    dict(hidden_dim=256, n_layers=3, dropout=0.5, lr=0.01,  weight_decay=0.0, epochs=1000, patience=50),
    "GCN":    dict(hidden_dim=256, n_layers=3, dropout=0.5, lr=0.01,  weight_decay=0.0, epochs=1000, patience=50),
    "SAGE":   dict(hidden_dim=256, n_layers=3, dropout=0.5, lr=0.01,  weight_decay=0.0, epochs=1000, patience=50),
    "RevGAT": dict(hidden_dim=256, n_layers=3, dropout=0.75, lr=0.002, weight_decay=0.0, epochs=1000, patience=50),
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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


def build_h_pred_for_dataset(dataset_name, label_names, seed, d_pred=128):
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name="llama_3.2_1B", peft_type="lora")
    raw_responses = load_gpt_enhanced_explanations(cfg)
    predictions_per_node = [parse_llm_response(r, label_names, dataset_name=dataset_name)[0] for r in raw_responses]
    return build_h_pred(predictions_per_node, label_names, d_pred=d_pred, seed=seed)


def run_one_seed(dataset_name, llm_name, seed, device):
    set_seed(seed)
    data, num_classes, _ = load_split_graph(dataset_name, device, seed)
    label_names = list(data.label_name)

    print(f"  [seed={seed}] fine-tuning LM_orig...")
    h_orig = finetune_and_extract(dataset_name, llm_name, used_llm_responses=False, seed=seed, run_tag="orig")
    print(f"  [seed={seed}] fine-tuning LM_expl...")
    h_expl = finetune_and_extract(dataset_name, llm_name, used_llm_responses=True, seed=seed, run_tag="expl")
    print(f"  [seed={seed}] building h_pred...")
    h_pred = build_h_pred_for_dataset(dataset_name, label_names, seed=seed)

    feature_sources = {"h_orig": h_orig, "h_expl": h_expl, "h_pred": h_pred}

    arch_results = {}
    for arch_name, builder in MODEL_BUILDERS.items():
        hp = GNN_HYPERPARAMS[arch_name]
        per_source_metrics = {}
        per_source_probs = {}
        for src_name, features in feature_sources.items():
            model = builder(
                input_dim=features.shape[1], output_dim=num_classes,
                hidden_dim=hp["hidden_dim"], n_layers=hp["n_layers"], dropout=hp["dropout"],
            )
            trainer = StageTrainer(
                model, features, data,
                lr=hp["lr"], weight_decay=hp["weight_decay"],
                epochs=hp["epochs"], patience=hp["patience"], device=device,
            )
            results, _, probs = trainer.train()
            per_source_metrics[src_name] = results
            per_source_probs[src_name] = probs

        ensemble_probs = ensemble_mean_probs(list(per_source_probs.values()))
        pred = ensemble_probs.argmax(dim=-1)
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(
            pred[data.test_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy()
        )
        print(f"  [seed={seed}] {arch_name:6s} h_orig={per_source_metrics['h_orig']['test_acc']:.2f} "
              f"h_expl={per_source_metrics['h_expl']['test_acc']:.2f} h_pred={per_source_metrics['h_pred']['test_acc']:.2f} "
              f"h_TAPE={acc:.2f}")

        arch_results[arch_name] = {
            "per_source": per_source_metrics,
            "h_TAPE": {"test_acc": acc, "test_macro_f1": macro_f1, "test_weighted_f1": weighted_f1},
        }

    return arch_results


def aggregate(all_seed_results):
    arch_names = list(all_seed_results[0].keys())
    summary = {}
    for arch_name in arch_names:
        source_names = list(all_seed_results[0][arch_name]["per_source"].keys()) + ["h_TAPE"]
        summary[arch_name] = {}
        for src_name in source_names:
            if src_name == "h_TAPE":
                values = [r[arch_name]["h_TAPE"]["test_acc"] for r in all_seed_results]
            else:
                values = [r[arch_name]["per_source"][src_name]["test_acc"] for r in all_seed_results]
            summary[arch_name][src_name] = {
                "test_acc_mean": round(statistics.mean(values), 2),
                "test_acc_std": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
            }
    return summary


def print_summary_table(dataset_name, llm_name, summary):
    print(f"\n{'=' * 78}\nTAPE results -- dataset={dataset_name} llm={llm_name}\n{'=' * 78}")
    print(f"{'Arch':8s} {'h_orig':>16s} {'h_expl':>16s} {'h_pred':>16s} {'h_TAPE':>16s}")
    for arch_name, sources in summary.items():
        row = [arch_name]
        for src in ["h_orig", "h_expl", "h_pred", "h_TAPE"]:
            m = sources[src]
            row.append(f"{m['test_acc_mean']:.2f}+/-{m['test_acc_std']:.2f}")
        print(f"{row[0]:8s} {row[1]:>16s} {row[2]:>16s} {row[3]:>16s} {row[4]:>16s}")
    print(f"{'=' * 78}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the TAPE pipeline on a text-attributed graph dataset.")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed"])
    parser.add_argument("--llm_name", type=str, default="llama_3.2_1B", help="LM fine-tuned as LM_orig/LM_expl.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_seed_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        all_seed_results.append(run_one_seed(args.dataset, args.llm_name, seed, device))

    summary = aggregate(all_seed_results)
    print_summary_table(args.dataset, args.llm_name, summary)

    output_path = os.path.join(RESULTS_DIR, f"{args.dataset}_{args.llm_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "llm_name": args.llm_name,
                "seeds": args.seeds,
                "gnn_hyperparameters": GNN_HYPERPARAMS,
                "per_seed_results": all_seed_results,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
