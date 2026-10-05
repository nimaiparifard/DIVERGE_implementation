from config import setup_finetuning_cfg
from gnns.gnn_mtrainer import get_datasets_path, gnn_train_and_report
import os
import torch
import argparse

from report.reporter import ReportResults
from common import load_graph_dataset_for_tape


def get_init_dataset_for_gnn_specific_init_type(cfg, init_type_approaches='pissa'):
    """
    Load initial datasets for GNN training with different initialization methods.

    Args:
        cfg: Configuration object with dataset and model settings

    Returns:
        tuple: (data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva)
    """
    from gnns.gnn_mtrainer import get_datasets_path

    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)

    if os.path.exists(os.path.join(repo_root, 'artifacts', 'cache')):
        cache_dir = os.path.join(repo_root, 'artifacts', 'cache')
    elif os.path.exists(os.path.join(repo_root, 'datasets', 'cache')):
        cache_dir = os.path.join(repo_root, 'datasets', 'cache')
    elif os.path.exists('artifacts/cache'):
        cache_dir = 'artifacts/cache'
    else:
        raise FileNotFoundError(
            "Cannot find cache directory. Expected either:\n"
            "  - artifacts/cache\n"
            "  - datasets/cache"
        )

    # Build file paths
    data_path = os.path.join(cache_dir,
                                   f'{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_{cfg.peft.type}_init-{init_type_approaches}_pool-mean.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load all datasets
    data = torch.load(data_path, map_location=device)
    return data['embeddings']

def save_ogbn_gnn_trained_model_logits(dataset, model, emb, init_approaches, dataset_name):
    """
    Run inference on the trained model and save logits to a file.
    
    Args:
        dataset: Graph dataset object
        model: Trained GNN model
        emb: Embeddings used for training
        init_approaches: Initialization approach name for naming the output file
        dataset_name: Name of the dataset for naming the output file
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get logits from the model
    with torch.no_grad():
        logits = model(emb.to(device), dataset.edge_index.to(device))
    
    # Create output directory if it doesn't exist
    output_dir = 'artifacts/logits'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save logits with descriptive filename
    output_path = os.path.join(output_dir, f'{dataset_name}_{init_approaches}_logits.pt')
    torch.save({
        'logits': logits.cpu(),
        'init_approach': init_approaches,
    }, output_path)
    
    print(f"Logits saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN training with different LLM initialization approaches")
    parser.add_argument('--dataset_name', type=str, default='arxiv', 
                        help='Name of the dataset (default: arxiv)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='Name of the LLM model (default: llama_3.2_1B)')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='Type of PEFT method (default: lora)')
    parser.add_argument('--init_approaches', type=str, default='pissa',
                        help='Initialization approach (default: pissa)')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    llm_name = args.llm_name
    peft_type = args.peft_type
    init_approaches = args.init_approaches
    
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
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
    dataset, _, _ = load_graph_dataset_for_tape(dataset_name, 'cuda:0', re_split=1, path_prefix=path_prefix, seed=cfg.dataset.seed)

    reporter = ReportResults(cfg, index_run=init_approaches)
    emb = get_init_dataset_for_gnn_specific_init_type(cfg, init_type_approaches=init_approaches)

    results, model = gnn_train_and_report(dataset_name, emb, title=init_approaches,
                                                      does_print_training_process=True, reporter=reporter)

    save_ogbn_gnn_trained_model_logits(dataset, model, emb, init_approaches, dataset_name)