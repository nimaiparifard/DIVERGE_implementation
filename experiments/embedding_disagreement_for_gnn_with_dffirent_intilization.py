# in this file we want to design experiment
# get llm_name and seed and dataset name
# run for the embedding of the same dataset
# run gnn for each embedding of diffrent initilization lora
# then report label disagreement for each pair embedding
# save it in csv
# and report with reporter
"""
Experiment: pairwise label-disagreement between GNNs trained on embeddings
coming from the same LLM/seed but with different LoRA initialization
approaches (PISSA, ORTHOGONAL, GAUSSIAN, LOFTQ, EVA).

For one (dataset_name, llm_name, seed):
  1) Load the 5 cached embeddings for this dataset/llm/seed (one per LoRA
     init approach) via dataset.data_utils.get_init_dataset_for_gnn.
  2) Train a GNN on each embedding (5 GNNs total), via
     gnns.gnn_mtrainer.gnn_train_and_report.
  3) Predict labels for every node with each of the 5 trained GNNs.
  4) For every pair of the 5 GNNs, compute the label-disagreement rate: the
     fraction of nodes (on the test split, and over all nodes) where the two
     GNNs' predicted labels differ.
  5) Append one row per pair to a CSV report (for aggregation across seeds/
     datasets/LLMs) and report the full pairwise matrix via the shared
     reporter.
"""
import itertools
import os

import torch

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report, set_mgnn_cfg, append_gnn_result_csv
from report.reporter import ReportResults
from common import load_graph_dataset_for_tape

# Order must match dataset.data_utils.get_init_dataset_for_gnn's return tuple
APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
TITLES = {'pissa': 'PISSA', 'orthogonal': 'ORTHOGONAL', 'gaussian': 'GAUSSIAN', 'loftq': 'LOFTQ', 'eva': 'EVA'}


def _resolve_path_prefix():
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            return '.'
        return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'


def get_model_predictions(model, embedding, edge_index):
    """Run a trained GNN on its embedding and return CPU label predictions for every node."""
    model_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(embedding.to(model_device), edge_index.to(model_device))
        pred = logits.argmax(dim=1).cpu()
    return pred


def compute_pairwise_disagreement(predictions, test_mask):
    """
    Compute label disagreement between every pair of models' predictions.

    Args:
        predictions (dict): {approach_name: torch.Tensor[num_nodes]} predicted labels (CPU).
        test_mask (torch.Tensor[bool]): mask selecting the test split.

    Returns:
        dict: {(name_a, name_b): {"disagreement_rate_all", "disagreement_rate_test",
                                    "agreement_rate_test", "num_disagree_test",
                                    "num_test_nodes"}}
    """
    test_mask = test_mask.cpu()
    num_test_nodes = int(test_mask.sum().item())
    pairwise = {}
    for name_a, name_b in itertools.combinations(predictions.keys(), 2):
        pred_a, pred_b = predictions[name_a], predictions[name_b]
        disagree_all = (pred_a != pred_b)
        disagree_test = disagree_all[test_mask]
        num_disagree_test = int(disagree_test.sum().item())
        pairwise[(name_a, name_b)] = {
            "disagreement_rate_all": float(disagree_all.float().mean().item()),
            "disagreement_rate_test": float(num_disagree_test) / num_test_nodes if num_test_nodes else 0.0,
            "agreement_rate_test": 1.0 - (float(num_disagree_test) / num_test_nodes if num_test_nodes else 0.0),
            "num_disagree_test": num_disagree_test,
            "num_test_nodes": num_test_nodes,
        }
    return pairwise


def save_disagreement_matrix_csv(pairwise, approaches, save_path):
    """Save the (symmetric) test-split disagreement-rate matrix as a wide CSV table."""
    import csv

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [TITLES[a] for a in approaches])
        for name_a in approaches:
            row = [TITLES[name_a]]
            for name_b in approaches:
                if name_a == name_b:
                    row.append(0.0)
                else:
                    key = (name_a, name_b) if (name_a, name_b) in pairwise else (name_b, name_a)
                    row.append(round(pairwise[key]["disagreement_rate_test"], 6))
            writer.writerow(row)

    print(f"Saved pairwise disagreement matrix CSV to: {save_path}")


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         reporter_index=0, csv_path='results/disagreement/embedding_disagreement_results.csv',
         matrix_csv_dir='results/disagreement'):
    """
    Run the embedding-disagreement experiment for one (dataset, llm, seed) combination.

    Args:
        dataset_name: Dataset to run on (e.g., 'cora', 'wikics').
        llm_name: LLM whose LoRA-initialized embeddings/hyperparameters to use.
        peft_type: PEFT type of the cached embeddings (e.g., 'lora').
        seed: Seed the LoRA adapters/embedding caches were trained with (must match
              --seed used in train_llm/train_other_lora_init_apporaches.py and
              cache/cache_embedding_with_diffrent_init_weights.py); also used as the
              GNN's own training seed. If None, uses configs/dataset/<name>.json's seed.
        supervised: True for the supervised embedding cache/GNN split, False for the
              semi-supervised (seed-tagged) one.
        csv_path: where to append the long-format (one row per pair) CSV result
              ("" / None disables it).
        matrix_csv_dir: directory to save the per-run wide disagreement matrix CSV
              ("" / None disables it).

    Returns:
        dict with the 5 individual GNN results and the pairwise disagreement stats.
    """
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    print("=" * 80)
    print(f"EMBEDDING DISAGREEMENT | dataset={dataset_name} llm={llm_name} peft={peft_type} "
          f"seed={resolved_seed} supervised={supervised}")
    print("=" * 80)

    reporter = ReportResults(
        cfg,
        save_dir=f"results/train_info/{dataset_name}_embedding_disagreement_seed{resolved_seed}",
        index_run=reporter_index,
    )
    reporter.report_title("Label Disagreement Between GNNs Trained on Different LoRA-Init Embeddings")
    reporter.report_txt(f"Dataset: {dataset_name}")
    reporter.report_txt(f"LLM: {llm_name}")
    reporter.report_txt(f"PEFT Type: {peft_type}")
    reporter.report_txt(f"Seed: {resolved_seed}")
    reporter.report_txt(f"Supervised: {supervised}")
    reporter.report_txt("")

    # 1) Load the seed-tagged embeddings for the 5 LoRA init approaches
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    emb = {name: get_embedding_from_data(all_data[name]) for name in APPROACHES}
    print("Loaded embeddings: " + ", ".join(f"{name}={tuple(emb[name].shape)}" for name in APPROACHES))

    # 2) Train one GNN per embedding
    gnn_kwargs = dict(does_print_training_process=False, reporter=reporter, supervised=supervised,
                       seed=resolved_seed, llm_name=llm_name, peft_type=peft_type)
    results, models = {}, {}
    for name in APPROACHES:
        results[name], models[name] = gnn_train_and_report(dataset_name, emb[name], title=TITLES[name], **gnn_kwargs)

    reporter.report_title("Individual Model Accuracy")
    for name in APPROACHES:
        line = f"  {TITLES[name]}: val_acc={results[name]['val_acc']:.4f}, test_acc={results[name]['test_acc']:.4f}"
        print(line)
        reporter.report_txt(line)

    # 3) Predict labels for every node with each trained GNN
    gnn_cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=resolved_seed, llm_name=llm_name)
    device = torch.device("cuda:0" if gnn_cfg.device > 0 else "cpu")
    path_prefix = _resolve_path_prefix()
    data, _, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=gnn_cfg.re_split,
                                              path_prefix=path_prefix, seed=gnn_cfg.split_seed)

    predictions = {name: get_model_predictions(models[name], emb[name], data.edge_index) for name in APPROACHES}

    # 4) Pairwise label disagreement
    pairwise = compute_pairwise_disagreement(predictions, data.test_mask)

    reporter.report_title("Pairwise Label Disagreement (Test Split)")
    for (name_a, name_b), stats in pairwise.items():
        line = (f"  {TITLES[name_a]} vs {TITLES[name_b]}: "
                f"disagreement={stats['disagreement_rate_test']:.4f} "
                f"({stats['num_disagree_test']}/{stats['num_test_nodes']}), "
                f"disagreement_all_nodes={stats['disagreement_rate_all']:.4f}")
        print(line)
        reporter.report_txt(line)

    # 5a) Append long-format CSV rows (one per pair), for aggregation across runs
    if csv_path:
        for (name_a, name_b), stats in pairwise.items():
            row = {
                "dataset_name": dataset_name,
                "llm_name": llm_name,
                "peft_type": peft_type,
                "seed": resolved_seed,
                "supervised": supervised,
                "approach_a": TITLES[name_a],
                "approach_b": TITLES[name_b],
                "test_acc_a": results[name_a]["test_acc"],
                "test_acc_b": results[name_b]["test_acc"],
                "disagreement_rate_test": stats["disagreement_rate_test"],
                "agreement_rate_test": stats["agreement_rate_test"],
                "num_disagree_test": stats["num_disagree_test"],
                "num_test_nodes": stats["num_test_nodes"],
                "disagreement_rate_all": stats["disagreement_rate_all"],
            }
            append_gnn_result_csv(row, csv_path)
        print(f"[OK] Pairwise disagreement rows appended to: {csv_path}")

    # 5b) Save a wide disagreement-matrix CSV for this run (handy for a paper table)
    if matrix_csv_dir:
        matrix_path = os.path.join(
            matrix_csv_dir, f"{dataset_name}_{llm_name}_seed{resolved_seed}_disagreement_matrix.csv")
        save_disagreement_matrix_csv(pairwise, APPROACHES, matrix_path)

    return {
        "individual_results": results,
        "pairwise_disagreement": pairwise,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairwise label disagreement between GNNs trained on differently LoRA-initialized embeddings")
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches (must match --seed used in '
                              'train_llm/train_other_lora_init_apporaches.py and the caching step)')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--csv_path', type=str, default='results/disagreement/embedding_disagreement_results.csv')
    parser.add_argument('--matrix_csv_dir', type=str, default='results/disagreement')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        reporter_index=args.reporter_index,
        csv_path=args.csv_path,
        matrix_csv_dir=args.matrix_csv_dir,
    )
