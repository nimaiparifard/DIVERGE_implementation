import os

import torch

from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from ensemble.ensemble_gnns_learning import ensemble_gnn_learning, \
    tune_ensemble_hyperparameter
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report
from report.reporter import ReportResults
from common import load_graph_dataset_for_tape

if __name__ == "__main__":
    import argparse
    
    ## argument definition
    parser = argparse.ArgumentParser(description='Run semi-supervised learning with ensemble GNNs')
    parser.add_argument('--dataset_name', type=str, default='pubmed',
                        help='Dataset name (e.g., cora, citeseer, pubmed, wikics)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='LLM model name')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='PEFT type (e.g., lora)')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    llm_name = args.llm_name
    peft_type = args.peft_type

    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    reporter_index = dataset_name + "_"  + "semi_supervised"
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
    dataset, _, _ = load_graph_dataset_for_tape(dataset_name, 'cuda:0', re_split=0, path_prefix=path_prefix,
                                                seed=cfg.dataset.seed)
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg, supervised=False)
    emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian = get_embedding_from_data(
        data_pissa), get_embedding_from_data(data_orthogonal), get_embedding_from_data(
        data_loftq), get_embedding_from_data(data_eva), get_embedding_from_data(data_guassian)

    results_pissa, model_pissa = gnn_train_and_report(dataset_name, emb_pissa, title="PISSA",
                                                      does_print_training_process=True, reporter=reporter, supervised=False)
    results_orthogonal, model_orthogonal = gnn_train_and_report(dataset_name, emb_orthogonal, title="ORTHOGONAL",
                                                                does_print_training_process=False, reporter=reporter, supervised=False)
    results_loftq, model_loftq = gnn_train_and_report(dataset_name, emb_loftq, title="LOFTQ",
                                                      does_print_training_process=False, reporter=reporter, supervised=False)
    results_eva, model_eva = gnn_train_and_report(dataset_name, emb_eva, title="EVA", does_print_training_process=False,
                                                  reporter=reporter, supervised=False)
    results_guassian, model_guassian = gnn_train_and_report(dataset_name, emb_guassian, title="GUASSIAN",
                                                            does_print_training_process=False, reporter=reporter, supervised=False)
    model_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_guassian]
    emb_list = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    supervised = False
    _, results_uniform, _=  ensemble_gnn_learning(model_list, dataset_name, 'uniform', device, custom_embeddings=emb_list, supervised=supervised)
    reporter.report("Uniform", results_uniform)
    reporter.report("Ensemble Method 1 Learnable", "")
    results_ensemble1,_, _ = tune_ensemble_hyperparameter(dataset_name=dataset_name,
        model_list=model_list,
        custom_embeddings=emb_list,
        visualize_best=False,
        does_report_training_process=False,
        ensemble_approaches='learnable',
        reporter=reporter,
        supervised=supervised,)

    reporter.report("Ensemble Method 2 Learnable Classes", "")
    results_ensemble2, _,_ = tune_ensemble_hyperparameter(
        dataset_name=dataset_name,
        model_list=model_list,
        custom_embeddings=emb_list,
        visualize_best=False,
        does_report_training_process=False,
        ensemble_approaches='learnable_classes',
        supervised=supervised,
        reporter=reporter,)

    reporter.report("Ensemble Method 3 Learnable Per Classes", "")
    results_ensemble3, _ , _= tune_ensemble_hyperparameter(
        dataset_name=dataset_name,
        model_list=model_list,
        custom_embeddings=emb_list,
        visualize_best=False,
        does_report_training_process=False,
        ensemble_approaches='learnable_per_classes',
        supervised=supervised,
    reporter=reporter,)

