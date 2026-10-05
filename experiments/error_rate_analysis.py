import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report
from report.reporter import ReportResults
from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter, ensemble_gnn_learning
from common import load_graph_dataset_for_tape


def save_result_in_csv(results, titles, dataset_name):
    """
    Save overall test metrics (accuracy, macro F1, weighted F1) for each model
    into a single CSV file. Each row corresponds to a model in `titles`.

    Args:
        results (dict): keys 'accuracy', 'macro_f1', 'weighted_f1', each a list
                        of floats of the same length as `titles`.
        titles (list[str]): model names in the same order as metrics lists.
    """
    assert len(results["accuracy"]) == len(titles), "results and titles length mismatch"
    assert len(results["macro_f1"]) == len(titles), "results and titles length mismatch"
    assert len(results["weighted_f1"]) == len(titles), "results and titles length mismatch"

    # Save under repo_root/results/train_info
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    save_dir = os.path.join(repo_root, "results", "train_info")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"gnn_results_summary_{dataset_name}.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header (include dataset name)
        writer.writerow(["Dataset", "Model", "Accuracy", "Macro_F1", "Weighted_F1"])
        # Rows
        for i, title in enumerate(titles):
            writer.writerow([
                dataset_name,
                title,
                results["accuracy"][i],
                results["macro_f1"][i],
                results["weighted_f1"][i],
            ])

    print(f"Saved results summary CSV to: {csv_path}")


def save_error_rate_in_csv(error_rates, titles, dataset_name):
    """
    Save per-class error rate for each model into a single CSV file.

    Each element of `error_rates` is expected to be a dict:
        {class_id: error_rate_percentage}
    for one model. One CSV row per class id, with columns being models.

    Args:
        error_rates (list[dict]): list of per-model error-rate dicts.
        titles (list[str]): model names, same order as `error_rates`.
    """
    assert len(error_rates) == len(titles), "error_rates and titles length mismatch"

    # Collect all class ids that appear in any dictionary
    class_ids = set()
    for er in error_rates:
        class_ids.update(er.keys())
    class_ids = sorted(class_ids)

    # Save under repo_root/results/train_info
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    save_dir = os.path.join(repo_root, "results", "train_info")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"gnn_error_rates_by_class_{dataset_name}.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header: dataset + class_id + model names
        writer.writerow(["Dataset", "class_id"] + titles)

        # One row per class id
        for cid in class_ids:
            row = [dataset_name, cid]
            for er in error_rates:
                # Use 0.0 when a model has no entries for this class
                row.append(er.get(cid, 0.0))
            writer.writerow(row)

    print(f"Saved per-class error-rate CSV to: {csv_path}")


def plot_sum_error_number(error_numbers, titles, dataset_name):
    """
    Plot and save a histogram (bar chart) of the total error counts per model.

    Args:
        error_numbers (list[dict]): list of per-model error-number dicts
                                    {class_id: misclassified_count}.
        titles (list[str]): model names, same order as `error_numbers`.
    """
    assert len(error_numbers) == len(titles), "error_numbers and titles length mismatch"

    # Total error count per model
    total_errors = [int(np.sum(list(en.values()))) for en in error_numbers]

    # Save under repo_root/results/train_info
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    save_dir = os.path.join(repo_root, "results", "train_info")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"gnn_total_error_numbers_{dataset_name}.png")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(titles))
    plt.bar(x, total_errors, color="skyblue", edgecolor="black")
    plt.xticks(x, titles, rotation=45, ha="right")
    plt.ylabel("Total Misclassified Samples")
    plt.xlabel("Model")
    plt.title("Total Error Number per Model")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved total error-number histogram to: {fig_path}")

if __name__ == "__main__":
    dataset_name = 'wikics'

    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    reporter_index = dataset_name + "_" + "error_rate"
    reporter = ReportResults(cfg, index_run=reporter_index)
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        # If running from project root, use current directory
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            path_prefix = '.'
        else:
            # Otherwise, use relative path from current directory to repo root
            path_prefix = os.path.relpath(repo_root, start=os.getcwd())
            # Normalize path to avoid double slashes
            path_prefix = os.path.normpath(path_prefix)
    else:
        # Fallback to default
        path_prefix = '../..'
    dataset, _, _ = load_graph_dataset_for_tape(dataset_name, 'cuda:0', re_split=1, path_prefix=path_prefix,
                                                seed=cfg.dataset.seed)
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian = get_embedding_from_data(
        data_pissa), get_embedding_from_data(data_orthogonal), get_embedding_from_data(
        data_loftq), get_embedding_from_data(data_eva), get_embedding_from_data(data_guassian)
    results_pissa, model_pissa = gnn_train_and_report(dataset_name, emb_pissa, title="PISSA",
                                                      does_print_training_process=False, reporter=reporter)
    results_orthogonal, model_orthogonal = gnn_train_and_report(dataset_name, emb_orthogonal, title="ORTHOGONAL",
                                                                does_print_training_process=False, reporter=reporter)
    results_loftq, model_loftq = gnn_train_and_report(dataset_name, emb_loftq, title="LOFTQ",
                                                      does_print_training_process=False, reporter=reporter)
    results_eva, model_eva = gnn_train_and_report(dataset_name, emb_eva, title="EVA", does_print_training_process=False,
                                                  reporter=reporter)
    results_guassian, model_guassian = gnn_train_and_report(dataset_name, emb_guassian, title="GUASSIAN",
                                                            does_print_training_process=False, reporter=reporter)

    model_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_guassian]
    emb_list = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, results_uniform, _=  ensemble_gnn_learning(model_list, dataset_name, 'uniform', device, custom_embeddings=emb_list)
    # results_ensemble1,_, _ = tune_ensemble_hyperparameter(dataset_name=dataset_name,
    #     model_list=model_list,
    #     custom_embeddings=emb_list,
    #     visualize_best=False,
    #     does_report_training_process=False,
    #     ensemble_approaches='learnable',)
    # results_ensemble2, _,_ = tune_ensemble_hyperparameter(
    #     dataset_name=dataset_name,
    #     model_list=model_list,
    #     custom_embeddings=emb_list,
    #     visualize_best=False,
    #     does_report_training_process=False,
    #     ensemble_approaches='learnable_classes')
    # results_ensemble3, _ , _= tune_ensemble_hyperparameter(
    #     dataset_name=dataset_name,
    #     model_list=model_list,
    #     custom_embeddings=emb_list,
    #     visualize_best=False,
    #     does_report_training_process=False,
    #     ensemble_approaches='learnable_per_classes')
    error_rates = [
        results_pissa['error_rate'],
        results_orthogonal['error_rate'],
        results_loftq['error_rate'],
        results_eva['error_rate'],
        results_guassian['error_rate'],
        results_uniform['test']['error_rate'],
        # results_ensemble1['test']['error_rate'],
        # results_ensemble2['test']['error_rate'],
        # results_ensemble3['test']['error_rate'],
    ]
    error_numbers = [
        results_pissa['error_number'],
        results_orthogonal['error_number'],
        results_loftq['error_number'],
        results_eva['error_number'],
        results_guassian['error_number'],
        results_uniform['test']['error_number'],
        # results_ensemble1['test']['error_number'],
        # results_ensemble2['test']['error_number'],
        # results_ensemble3['test']['error_number'],
    ]
    results = {
        "accuracy": [
            results_pissa['test_acc'],
            results_orthogonal['test_acc'],
            results_loftq['test_acc'],
            results_eva['test_acc'],
            results_guassian['test_acc'],
            results_uniform['test']['accuracy'],
            # results_ensemble1['test']['accuracy'],
            # results_ensemble2['test']['accuracy'],
            # results_ensemble3['test']['accuracy'],
        ],
        "macro_f1": [
            results_pissa['test_f1'],
            results_orthogonal['test_f1'],
            results_loftq['test_f1'],
            results_eva['test_f1'],
            results_guassian['test_f1'],
            results_uniform['test']['macro_f1'],
            # results_ensemble1['test']['macro_f1'],
            # results_ensemble2['test']['macro_f1'],
            # results_ensemble3['test']['macro_f1'],
        ],
        "weighted_f1": [
            results_pissa['test_weight_f1'],
            results_orthogonal['test_weight_f1'],
            results_loftq['test_weight_f1'],
            results_eva['test_weight_f1'],
            results_guassian['test_weight_f1'],
            results_uniform['test']['weighted_f1'],
            # results_ensemble1['test']['weighted_f1'],
            # results_ensemble2['test']['weighted_f1'],
            # results_ensemble3['test']['weighted_f1'],
        ],
    }

    titles = ['Pissa', 'Orthogonal', 'Loftq', 'Eva', 'Guassian', 'Uniform',
              # 'Ensemble1', 'Ensemble2', 'Ensemble3'
              ]
    save_result_in_csv(results, titles,dataset_name)
    save_error_rate_in_csv(error_rates, titles, dataset_name)
    plot_sum_error_number(error_numbers, titles, dataset_name)

    # write code to create 2 csv file
    # accuracy,f1,weighted f1 in one csv  of the diffirent model pissa, eva, loftq, orthogonal, guassian, orthogonal, uniform_ensemble, method1 ensemble, method2 ensemble, method 3 ensemble
    # error rate for each class in one csv file for diffirent model model pissa, eva, loftq, orthogonal, guassian, orthogonal, uniform_ensemble, method1 ensemble, method2 ensemble, method 3 ensemble
    # also plot hidtogram for total error number for diffirent model
    # save all of this with two factor

