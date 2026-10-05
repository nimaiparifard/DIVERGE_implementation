# Response to journal review comment (diverge_paper/review.txt,
# Reviewer 2 - Comment 5, also raised as Reviewer 1 - Comment 6):
#
#   "The proposed adaptive ensemble module introduces additional learnable
#    parameters, but the paper does not provide sufficient analysis on how
#    the learned weights behave or why they improve over simple averaging."
#
# idea to Response (from review.txt):
#   - Table/plot of learned ensemble weights per dataset; compare adaptive
#     vs uniform average.
#   - Show weights adapt when one view is weak; link to why MLP helps over
#     simple averaging.
"""
Experiment: behavior of the learned adaptive-ensemble weights.

For each dataset:
  1) Train the 5 per-init GNNs (PISSA, ORTHOGONAL, GAUSSIAN, LOFTQ, EVA) on
     their cached embeddings.
  2) Combine them three ways, holding everything else fixed:
       - 'uniform'   : simple average (1/5 each) -> baseline from the paper.
       - 'scalar'    : WeightLearner with num_layers=1 -> one global learned
                       scalar weight per view (softmax-normalized). This is
                       the most interpretable "adaptive" setting and is what
                       we tabulate/plot per dataset.
       - 'mlp'       : WeightLearner with num_layers>1 -> node-conditioned
                       MLP weights (mean weight per view, averaged over all
                       nodes, reported for the same table).
  3) Record, per (dataset, view): the view's own standalone test accuracy
     alongside its learned scalar/MLP weight, plus the dataset-level test
     accuracy of all three combination strategies.
  4) Correlate each view's learned weight with its standalone accuracy (and
     with its error rate) to test whether the ensemble down-weights weak
     views and up-weights strong ones, rather than weighting them uniformly.
  5) Save a long-format per-view CSV, a per-dataset summary CSV, a per-dataset
     bar plot of the three weighting schemes, and one combined
     weight-vs-accuracy scatter plot across all datasets/views. Report
     everything via the shared reporter.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report, append_gnn_result_csv
from ensemble.ensemble_gnns_learning import ensemble_gnn_learning
from report.reporter import ReportResults

# Order must match dataset.data_utils.get_init_dataset_for_gnn's return tuple
APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
TITLES = {'pissa': 'PISSA', 'orthogonal': 'ORTHOGONAL', 'gaussian': 'GAUSSIAN', 'loftq': 'LOFTQ', 'eva': 'EVA'}


def _repo_root():
    datasets_path = get_datasets_path()
    return os.path.dirname(datasets_path)


def plot_dataset_weights(dataset_name, scalar_weights, mlp_weights, save_path):
    """Grouped bar chart of uniform / scalar-learned / MLP-learned weights for one dataset."""
    titles = [TITLES[a] for a in APPROACHES]
    x = np.arange(len(titles))
    width = 0.25
    uniform_value = 1.0 / len(APPROACHES)

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, [uniform_value] * len(titles), width, label='Uniform', color='lightgray', edgecolor='black')
    plt.bar(x, [scalar_weights[a] for a in APPROACHES], width, label='Scalar-Learned', color='steelblue', edgecolor='black')
    plt.bar(x + width, [mlp_weights[a] for a in APPROACHES], width, label='MLP-Learned (mean)', color='darkorange', edgecolor='black')
    plt.axhline(uniform_value, color='gray', linestyle='--', linewidth=1)
    plt.xticks(x, titles, rotation=30, ha="right")
    plt.ylabel("Ensemble weight")
    plt.title(f"Learned Ensemble Weights per View — {dataset_name.upper()}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved per-dataset weight plot to: {save_path}")


def plot_weight_vs_accuracy(rows, save_path):
    """
    Scatter of standalone view accuracy vs. learned scalar weight, across all
    datasets/views, with a linear trend line — shows whether weak views get
    down-weighted (and strong views up-weighted) by the adaptive ensemble.
    """
    accs = np.array([r["individual_test_acc"] for r in rows])
    weights = np.array([r["scalar_weight"] for r in rows])

    plt.figure(figsize=(7, 6))
    plt.scatter(accs, weights, s=60, alpha=0.8, edgecolor='black')
    for r in rows:
        plt.annotate(f"{r['dataset_name']}-{TITLES[r['approach']]}", (r["individual_test_acc"], r["scalar_weight"]),
                     fontsize=6, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    if len(accs) >= 2 and np.std(accs) > 0:
        slope, intercept = np.polyfit(accs, weights, 1)
        xs = np.linspace(accs.min(), accs.max(), 50)
        plt.plot(xs, slope * xs + intercept, color='red', linestyle='--',
                 label=f"trend (slope={slope:.3f})")
        corr = float(np.corrcoef(accs, weights)[0, 1])
        plt.title(f"Learned Weight vs. Standalone View Accuracy (Pearson r={corr:.3f})")
        plt.legend()
    else:
        plt.title("Learned Weight vs. Standalone View Accuracy")

    plt.axhline(1.0 / len(APPROACHES), color='gray', linestyle=':', linewidth=1, label='Uniform weight')
    plt.xlabel("Standalone view test accuracy")
    plt.ylabel("Learned scalar ensemble weight")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined weight-vs-accuracy plot to: {save_path}")


def run_dataset_weight_analysis(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None,
                                 supervised=True, reporter=None, mlp_num_layers=3, mlp_hidden_dim=128,
                                 mlp_dropout=0.3, mlp_lr=0.01, mlp_epochs=200):
    """
    Train the 5 per-init GNNs for one dataset, combine them with uniform /
    scalar-learned / MLP-learned weights, and return per-view + dataset-level
    stats for the weight-behavior analysis.
    """
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed

    print("=" * 80)
    print(f"ADAPTIVE ENSEMBLE WEIGHT ANALYSIS | dataset={dataset_name} llm={llm_name} "
          f"peft={peft_type} seed={resolved_seed} supervised={supervised}")
    print("=" * 80)

    # 1) Load embeddings and train the 5 per-init GNNs
    data_by_approach = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=resolved_seed)
    all_data = dict(zip(APPROACHES, data_by_approach))
    emb = {name: get_embedding_from_data(all_data[name]) for name in APPROACHES}
    emb_list = [emb[name] for name in APPROACHES]

    gnn_kwargs = dict(does_print_training_process=False, reporter=reporter, supervised=supervised,
                       seed=resolved_seed, llm_name=llm_name, peft_type=peft_type)
    results, models = {}, {}
    for name in APPROACHES:
        results[name], models[name] = gnn_train_and_report(dataset_name, emb[name], title=TITLES[name], **gnn_kwargs)
    model_list = [models[name] for name in APPROACHES]

    # 2) Combine with the three weighting strategies (same models/embeddings, only
    #    the fusion rule changes) so the comparison isolates the ensemble's effect.
    ensemble_common = dict(model_list=model_list, dataset_name=dataset_name, device='cuda:0',
                            custom_embeddings=emb_list, supervised=supervised, seed=resolved_seed,
                            llm_name=llm_name)

    _, uniform_results, uniform_weights = ensemble_gnn_learning(weight_approach='uniform', **ensemble_common)

    _, scalar_results, scalar_weights_list = ensemble_gnn_learning(
        weight_approach='learnable', num_layers=1, **ensemble_common)
    scalar_weights = dict(zip(APPROACHES, scalar_weights_list))

    _, mlp_results, mlp_weights_list = ensemble_gnn_learning(
        weight_approach='learnable', num_layers=mlp_num_layers, hidden_dim=mlp_hidden_dim,
        dropout=mlp_dropout, learning_rate=mlp_lr, epochs=mlp_epochs, **ensemble_common)
    mlp_weights = dict(zip(APPROACHES, [float(w) for w in mlp_weights_list]))

    # 3) Per-view rows: standalone accuracy vs. the weight each strategy assigned it
    per_view_rows = []
    for name in APPROACHES:
        per_view_rows.append({
            "dataset_name": dataset_name,
            "llm_name": llm_name,
            "seed": resolved_seed,
            "approach": name,
            "individual_val_acc": results[name]["val_acc"],
            "individual_test_acc": results[name]["test_acc"],
            "individual_error_rate_pct": (1.0 - results[name]["test_acc"]) * 100.0,
            "uniform_weight": uniform_weights[APPROACHES.index(name)],
            "scalar_weight": scalar_weights[name],
            "mlp_mean_weight": mlp_weights[name],
        })

    # 4) Does the ensemble down-weight weak views? Correlate weight with accuracy.
    accs = np.array([r["individual_test_acc"] for r in per_view_rows])
    scalar_w = np.array([r["scalar_weight"] for r in per_view_rows])
    mlp_w = np.array([r["mlp_mean_weight"] for r in per_view_rows])
    corr_scalar = float(np.corrcoef(accs, scalar_w)[0, 1]) if np.std(accs) > 0 else float('nan')
    corr_mlp = float(np.corrcoef(accs, mlp_w)[0, 1]) if np.std(accs) > 0 else float('nan')

    dataset_summary = {
        "dataset_name": dataset_name,
        "llm_name": llm_name,
        "seed": resolved_seed,
        "uniform_test_acc": uniform_results['test']['accuracy'],
        "uniform_test_macro_f1": uniform_results['test']['macro_f1'],
        "scalar_test_acc": scalar_results['test']['accuracy'],
        "scalar_test_macro_f1": scalar_results['test']['macro_f1'],
        "mlp_test_acc": mlp_results['test']['accuracy'],
        "mlp_test_macro_f1": mlp_results['test']['macro_f1'],
        "scalar_gain_over_uniform": scalar_results['test']['accuracy'] - uniform_results['test']['accuracy'],
        "mlp_gain_over_uniform": mlp_results['test']['accuracy'] - uniform_results['test']['accuracy'],
        "mlp_gain_over_scalar": mlp_results['test']['accuracy'] - scalar_results['test']['accuracy'],
        "weight_accuracy_corr_scalar": corr_scalar,
        "weight_accuracy_corr_mlp": corr_mlp,
    }

    if reporter is not None:
        reporter.report_title(f"Adaptive Ensemble Weight Analysis - {dataset_name.upper()}")
        reporter.report_txt("Per-view: standalone accuracy vs. learned weight")
        for r in per_view_rows:
            reporter.report_txt(
                f"  {TITLES[r['approach']]}: test_acc={r['individual_test_acc']:.4f} | "
                f"uniform_w={r['uniform_weight']:.4f} scalar_w={r['scalar_weight']:.4f} "
                f"mlp_w={r['mlp_mean_weight']:.4f}")
        reporter.report_txt("")
        reporter.report_txt(f"Uniform  test acc: {dataset_summary['uniform_test_acc']:.4f}")
        reporter.report_txt(f"Scalar-learned test acc: {dataset_summary['scalar_test_acc']:.4f} "
                             f"(gain over uniform: {dataset_summary['scalar_gain_over_uniform']:+.4f})")
        reporter.report_txt(f"MLP-learned test acc: {dataset_summary['mlp_test_acc']:.4f} "
                             f"(gain over uniform: {dataset_summary['mlp_gain_over_uniform']:+.4f}, "
                             f"gain over scalar: {dataset_summary['mlp_gain_over_scalar']:+.4f})")
        reporter.report_txt(f"Pearson corr(weight, standalone accuracy): scalar={corr_scalar:.4f}, mlp={corr_mlp:.4f}")
        reporter.report_txt("A positive correlation means the ensemble assigns higher weight to stronger, "
                             "i.e. it down-weights weak views instead of averaging them uniformly.")
        reporter.report_txt("")

    return {
        "per_view_rows": per_view_rows,
        "dataset_summary": dataset_summary,
        "scalar_weights": scalar_weights,
        "mlp_weights": mlp_weights,
    }


def main(dataset_names, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         reporter_index=0,
         per_view_csv_path='results/ensemble/weight_analysis/per_view_weights.csv',
         summary_csv_path='results/ensemble/weight_analysis/dataset_summary.csv',
         plot_dir='results/ensemble/weight_analysis/plots',
         mlp_num_layers=3, mlp_hidden_dim=128, mlp_dropout=0.3, mlp_lr=0.01, mlp_epochs=200):
    """
    Run the adaptive-ensemble weight-behavior analysis over one or more datasets
    and produce the CSV tables + plots requested in the review response.
    """
    cfg0 = setup_finetuning_cfg(dataset_names[0], llm_name, peft_type)
    reporter = ReportResults(cfg0, save_dir="results/train_info/ensemble_weight_analysis", index_run=reporter_index)
    reporter.report_title("Adaptive Ensemble Weight Behavior Analysis (Review Response)")
    reporter.report_txt(f"Datasets: {dataset_names}")
    reporter.report_txt(f"LLM: {llm_name}, PEFT: {peft_type}, Supervised: {supervised}")
    reporter.report_txt("")

    all_per_view_rows = []
    all_dataset_summaries = []

    for dataset_name in dataset_names:
        out = run_dataset_weight_analysis(
            dataset_name, llm_name=llm_name, peft_type=peft_type, seed=seed, supervised=supervised,
            reporter=reporter, mlp_num_layers=mlp_num_layers, mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout, mlp_lr=mlp_lr, mlp_epochs=mlp_epochs)

        all_per_view_rows.extend(out["per_view_rows"])
        all_dataset_summaries.append(out["dataset_summary"])

        plot_path = os.path.join(plot_dir, f"{dataset_name}_ensemble_weights.png")
        plot_dataset_weights(dataset_name, out["scalar_weights"], out["mlp_weights"], plot_path)

    # Long-format per-view CSV (one row per dataset x view)
    if per_view_csv_path:
        for row in all_per_view_rows:
            append_gnn_result_csv(row, per_view_csv_path)
        print(f"[OK] Per-view weight rows appended to: {per_view_csv_path}")

    # Per-dataset summary CSV (uniform vs scalar vs mlp test accuracy + correlations)
    if summary_csv_path:
        for row in all_dataset_summaries:
            append_gnn_result_csv(row, summary_csv_path)
        print(f"[OK] Dataset summary rows appended to: {summary_csv_path}")

    # Combined weight-vs-accuracy scatter across every dataset/view
    combined_plot_path = os.path.join(plot_dir, "weight_vs_accuracy_all_datasets.png")
    plot_weight_vs_accuracy(all_per_view_rows, combined_plot_path)

    overall_corr = float(np.corrcoef(
        [r["individual_test_acc"] for r in all_per_view_rows],
        [r["scalar_weight"] for r in all_per_view_rows])[0, 1])

    reporter.report_title("Overall Summary Across Datasets")
    reporter.report_txt(f"Overall Pearson corr(weight, standalone accuracy) across all datasets/views: {overall_corr:.4f}")
    for s in all_dataset_summaries:
        reporter.report_txt(
            f"  {s['dataset_name']}: uniform={s['uniform_test_acc']:.4f}, "
            f"scalar={s['scalar_test_acc']:.4f} ({s['scalar_gain_over_uniform']:+.4f}), "
            f"mlp={s['mlp_test_acc']:.4f} ({s['mlp_gain_over_uniform']:+.4f})")
    print(f"\nOverall weight-accuracy correlation across all datasets/views: {overall_corr:.4f}")

    return {
        "per_view_rows": all_per_view_rows,
        "dataset_summaries": all_dataset_summaries,
        "overall_weight_accuracy_corr": overall_corr,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Review-response experiment: analyze how the adaptive ensemble's learned weights "
                    "behave per dataset/view, and compare against simple averaging.")
    parser.add_argument('--dataset_names', type=str, default='cora,citeseer,pubmed,wikics,arxiv',
                         help='Comma-separated list of datasets to analyze')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches (must match --seed used in '
                              'train_llm/train_other_lora_init_apporaches.py and the caching step)')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--reporter_index', type=int, default=0)
    parser.add_argument('--per_view_csv_path', type=str,
                         default='results/ensemble/weight_analysis/per_view_weights.csv')
    parser.add_argument('--summary_csv_path', type=str,
                         default='results/ensemble/weight_analysis/dataset_summary.csv')
    parser.add_argument('--plot_dir', type=str, default='results/ensemble/weight_analysis/plots')
    parser.add_argument('--mlp_num_layers', type=int, default=3)
    parser.add_argument('--mlp_hidden_dim', type=int, default=128)
    parser.add_argument('--mlp_dropout', type=float, default=0.3)
    parser.add_argument('--mlp_lr', type=float, default=0.01)
    parser.add_argument('--mlp_epochs', type=int, default=200)
    args = parser.parse_args()

    main(
        dataset_names=[d.strip() for d in args.dataset_names.split(',') if d.strip()],
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        reporter_index=args.reporter_index,
        per_view_csv_path=args.per_view_csv_path,
        summary_csv_path=args.summary_csv_path,
        plot_dir=args.plot_dir,
        mlp_num_layers=args.mlp_num_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        mlp_lr=args.mlp_lr,
        mlp_epochs=args.mlp_epochs,
    )
