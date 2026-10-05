from sentence_transformers.util import normalize_embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import os

from ensemble.ensemble_gnns_learning import tune_ensemble_hyperparameter
from dataset.dataset_loader import load_dataset
from config import setup_finetuning_cfg
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data, get_init_dataset_for_gnn_with_retrained_gnn_mistake
from common.dataloader import load_graph_dataset_for_tape
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer, util
from gnns.gnn_mtrainer import gnn_train_and_report_with_modified_dataset, \
    get_datasets_path
from gnns.gnn_mtrainer import GNNTrainer, set_mgnn_cfg, gnn_train_and_report
from visualize.visualize_gnn_mistakes import get_gnn_embedding_mistakes
from report.reporter import ReportResults

def load_sentence_transformer_model(model_name = "sentence-transformers/all-mpnet-base-v2"):
    print("Loading sentence transformer model...")
    model = SentenceTransformer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

def load_encoder_model(model_name = "intfloat/e5-large"):
    print("Loading e5-large model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model

def sample_edges(n, dataset):
    num_samples = min(n, dataset.edge_index.shape[1])
    edge_indices = torch.randperm(dataset.edge_index.shape[1])[:num_samples]
    sampled_edges = dataset.edge_index[:, edge_indices]
    return sampled_edges

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding_sentence_transformer(text, model):
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

def get_embedding(text, tokenizer, model, device):
    """Get embedding for a single text using e5-large model"""
    # e5 models expect the prefix "query: " or "passage: " for optimal performance
    prefixed_text = f"passage: {text}"
    encoded_input = tokenizer(prefixed_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.squeeze(0).cpu()

def calculate_edge_embeddings(dataset, sampled_edges, tokenizer, model, device, sentence_transformer_used=False):
    unique_nodes = torch.unique(sampled_edges.flatten()).tolist()
    node_embeddings = {}

    print(f"Getting embeddings for {len(unique_nodes)} unique nodes...")
    for idx, node_id in enumerate(unique_nodes):
        if idx % 10 == 0:
            print(f"Processing node {idx + 1}/{len(unique_nodes)}")
        node_text = dataset.raw_texts[node_id]
        if sentence_transformer_used:
            node_embeddings[node_id] = get_embedding_sentence_transformer(node_text, model)
        else:
            node_embeddings[node_id] = get_embedding(node_text, tokenizer, model, device)
    return node_embeddings

def get_precomputed_node_embeddings(dataset, emb):
    node_embeddings = {}
    print(f"Loading precomputed embeddings for {dataset.num_nodes} nodes...")
    for node_id in range(dataset.num_nodes):
        node_embeddings[node_id] = emb[node_id]
    return node_embeddings

def calculate_cosine_similarities(sampled_edges, node_embeddings, dataset, sentence_transformer_used=False):
    print("\nCalculating cosine similarities...")
    similarities = []
    edge_info = []

    for i in range(sampled_edges.shape[1]):
        node1_id = sampled_edges[0, i].item()
        node2_id = sampled_edges[1, i].item()

        emb1 = node_embeddings[node1_id]
        emb2 = node_embeddings[node2_id]

        # Calculate cosine similarity
        if sentence_transformer_used:
            cosine_sim = float(util.cos_sim(emb1, emb2))
        else:
            cosine_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        similarities.append(cosine_sim)

        node1_label = dataset.y[node1_id].item()
        node2_label = dataset.y[node2_id].item()

        edge_info.append({
            'edge_idx': i,
            'node1_id': node1_id,
            'node2_id': node2_id,
            'node1_label': node1_label,
            'node2_label': node2_label,
            'similarity': cosine_sim,
            'same_label': node1_label == node2_label
        })
    return edge_info, similarities

def visualize_edges_info(similarities, edge_info, cfg, title, show=False):
    ## show the distibution of this similarity scores using a histogram also some charts for give me info
    print("\nGenerating visualizations...")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    ax1.hist(similarities, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    density = gaussian_kde(similarities)
    xs = np.linspace(min(similarities), max(similarities), 200)
    ax1.plot(xs, density(xs), 'r-', linewidth=2, label='KDE')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Edge Cosine Similarities', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Box plot comparing same label vs different label
    ax2 = axes[0, 1]
    same_label_sims = [info['similarity'] for info in edge_info if info['same_label']]
    diff_label_sims = [info['similarity'] for info in edge_info if not info['same_label']]
    box_data = [same_label_sims, diff_label_sims]
    bp = ax2.boxplot(box_data, labels=['Same Label', 'Different Label'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Similarity: Same vs Different Labels', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # 3. Scatter plot of similarities
    ax3 = axes[1, 0]
    colors = ['green' if info['same_label'] else 'red' for info in edge_info]
    ax3.scatter(range(len(similarities)), similarities, c=colors, alpha=0.6, s=50)
    ax3.axhline(y=np.mean(similarities), color='blue', linestyle='--', label=f'Mean: {np.mean(similarities):.4f}')
    ax3.set_xlabel('Edge Sample Index', fontsize=12)
    ax3.set_ylabel('Cosine Similarity', fontsize=12)
    ax3.set_title('Edge Similarity Scores (Green=Same Label, Red=Different)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Statistics Summary:
    {'=' * 50}

    Overall Statistics:
      • Total Edges: {len(similarities)}
      • Mean Similarity: {np.mean(similarities):.4f}
      • Median Similarity: {np.median(similarities):.4f}
      • Std Deviation: {np.std(similarities):.4f}
      • Min Similarity: {np.min(similarities):.4f}
      • Max Similarity: {np.max(similarities):.4f}

    Same Label Edges:
      • Count: {len(same_label_sims)}
      • Mean Similarity: {np.mean(same_label_sims):.4f}
      • Std Deviation: {np.std(same_label_sims):.4f}

    Different Label Edges:
      • Count: {len(diff_label_sims)}
      • Mean Similarity: {np.mean(diff_label_sims):.4f}
      • Std Deviation: {np.std(diff_label_sims):.4f}

    Quartiles:
      • Q1 (25%): {np.percentile(similarities, 25):.4f}
      • Q2 (50%): {np.percentile(similarities, 50):.4f}
      • Q3 (75%): {np.percentile(similarities, 75):.4f}
    """
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    try:
        plt.savefig(f'results/edge_analysis/edge_similarity_analysis_{cfg.dataset.name}_{title}.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'edge_similarity_analysis.png'")
    except:
        print("Can not Saved in jupyter notebook environment.Just i show this.")
    if show:
        plt.show()


def create_modified_dataset_for_mistakes(mistakes, dataset, node_embeddings, tokenizer, model, device, similarity_threshold=0.75, sentence_transformer_used=False):
    """
    Create a modified dataset by removing edges with similarity < threshold from mistake nodes.

    Args:
        mistakes: List of misclassified node IDs
        dataset: Original dataset
        node_embeddings: Dictionary of node embeddings
        similarity_threshold: Threshold for edge removal

    Returns:
        modified_dataset: Dataset with filtered edges
        removal_stats: Statistics about edge removal
    """
    print(f"\n  Number of misclassified nodes: {len(mistakes)}")

    # Get all edges involving misclassified nodes
    mistake_node_ids = set(mistakes)
    edge_index = dataset.edge_index
    edges_to_check = []

    for i in range(edge_index.shape[1]):
        node1 = edge_index[0, i].item()
        node2 = edge_index[1, i].item()
        if node1 in mistake_node_ids or node2 in mistake_node_ids:
            edges_to_check.append((i, node1, node2))

    print(f"  Edges connected to misclassified nodes: {len(edges_to_check)}")

    # Get embeddings for nodes involved in these edges
    nodes_to_embed = set()
    for _, node1, node2 in edges_to_check:
        nodes_to_embed.add(node1)
        nodes_to_embed.add(node2)

    # Compute embeddings for new nodes if needed
    nodes_to_compute = nodes_to_embed - set(node_embeddings.keys())
    if nodes_to_compute:
        print(f"  Computing embeddings for {len(nodes_to_compute)} additional nodes...")
        for idx, node_id in enumerate(nodes_to_compute):
            if idx % 100 == 0 and idx > 0:
                print(f"    Processed {idx}/{len(nodes_to_compute)} nodes")
            node_text = dataset.raw_texts[node_id]
            if sentence_transformer_used:
                node_embeddings[node_id] = get_embedding_sentence_transformer(node_text, model)
            else:
                node_embeddings[node_id] = get_embedding(node_text, tokenizer, model, device)

    # Calculate similarities and decide which edges to remove
    edges_to_remove = []
    edge_similarities = []

    for edge_idx, node1, node2 in edges_to_check:
        emb1 = node_embeddings[node1]
        emb2 = node_embeddings[node2]
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        edge_similarities.append(sim)

        if sim < similarity_threshold:
            edges_to_remove.append(edge_idx)

    # Create modified edge_index
    all_edge_indices = set(range(edge_index.shape[1]))
    edges_to_remove_set = set(edges_to_remove)
    edges_to_keep = all_edge_indices - edges_to_remove_set

    modified_edge_list = []
    for i in edges_to_keep:
        modified_edge_list.append([edge_index[0, i].item(), edge_index[1, i].item()])

    modified_edge_index = torch.tensor(modified_edge_list, dtype=torch.long).t()

    # Create modified dataset
    modified_dataset = dataset.clone()
    modified_dataset.edge_index = modified_edge_index

    # Calculate statistics
    removed_sims = [edge_similarities[i] for i, (idx, _, _) in enumerate(edges_to_check) if idx in edges_to_remove]
    kept_sims = [edge_similarities[i] for i, (idx, _, _) in enumerate(edges_to_check) if idx not in edges_to_remove]

    removal_stats = {
        'total_checked': len(edges_to_check),
        'removed': len(edges_to_remove),
        'kept': len(edges_to_check) - len(edges_to_remove),
        'removed_sims': removed_sims,
        'kept_sims': kept_sims,
        'original_edges': edge_index.shape[1],
        'modified_edges': modified_edge_index.shape[1]
    }

    print(f"  Edges removed: {len(edges_to_remove)} (similarity < {similarity_threshold})")
    print(f"  Original edges: {edge_index.shape[1]} → Modified edges: {modified_edge_index.shape[1]}")

    return modified_dataset, removal_stats

def visualize_edge_editing_results(all_original_results, modified_results, all_mistakes, removal_stats_all, cfg, show=False):
    # Create comprehensive comparison visualization
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    model_names = list(all_mistakes.keys())
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

    # Plot 1: Test Accuracy Comparison
    ax = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.35
    original_acc = [all_original_results[m]['test_acc'] for m in model_names]
    modified_acc = [modified_results[m]['test_acc'] for m in model_names]
    bars1 = ax.bar(x - width / 2, original_acc, width, label='Original', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width / 2, modified_acc, width, label='Modified', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Test Accuracy: Original vs Modified', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Plot 2: Test F1 Comparison
    ax = axes[0, 1]
    original_f1 = [all_original_results[m]['test_f1'] for m in model_names]
    modified_f1 = [modified_results[m]['test_f1'] for m in model_names]
    bars1 = ax.bar(x - width / 2, original_f1, width, label='Original', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width / 2, modified_f1, width, label='Modified', color='lightcoral', edgecolor='black')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Test F1 (Macro)', fontsize=11)
    ax.set_title('Test F1: Original vs Modified', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Plot 3: Improvement Heatmap
    ax = axes[0, 2]
    acc_improvements = [(modified_results[m]['test_acc'] - all_original_results[m]['test_acc']) * 100 for m in
                        model_names]
    f1_improvements = [(modified_results[m]['test_f1'] - all_original_results[m]['test_f1']) * 100 for m in model_names]
    bars1 = ax.barh(x - width / 2, acc_improvements, width, label='Accuracy Δ%', color='steelblue', edgecolor='black')
    bars2 = ax.barh(x + width / 2, f1_improvements, width, label='F1 Δ%', color='coral', edgecolor='black')
    ax.set_ylabel('Model', fontsize=11)
    ax.set_xlabel('Improvement (%)', fontsize=11)
    ax.set_title('Performance Improvements', fontsize=13, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(model_names)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # Plot 4: Number of Mistakes
    ax = axes[1, 0]
    num_mistakes = [len(all_mistakes[m]) for m in model_names]
    bars = ax.bar(model_names, num_mistakes, color=colors[:len(model_names)], edgecolor='black')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Number of Mistakes', fontsize=11)
    ax.set_title('Misclassified Nodes per Model', fontsize=13, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # Plot 5: Edges Removed
    ax = axes[1, 1]
    edges_removed = [removal_stats_all[m]['removed'] for m in model_names]
    bars = ax.bar(model_names, edges_removed, color=colors[:len(model_names)], edgecolor='black')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Edges Removed', fontsize=11)
    ax.set_title('Number of Edges Removed per Model', fontsize=13, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # Plot 6: Summary Statistics Table
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "SUMMARY STATISTICS\n" + "=" * 50 + "\n\n"
    for m in model_names:
        summary_text += f"{m.upper()}:\n"
        summary_text += f"  Mistakes: {len(all_mistakes[m])}\n"
        summary_text += f"  Edges Removed: {removal_stats_all[m]['removed']}\n"
        summary_text += f"  Acc Δ: {(modified_results[m]['test_acc'] - all_original_results[m]['test_acc']) * 100:+.2f}%\n"
        summary_text += f"  F1 Δ: {(modified_results[m]['test_f1'] - all_original_results[m]['test_f1']) * 100:+.2f}%\n\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    try:
        plt.savefig(f'comprehensive_edge_editing_analysis/{cfg.dataset.name}_{cfg.llm.model_name}_{cfg.peft.type}.png', dpi=300, bbox_inches='tight')
        print("Comprehensive analysis saved as 'comprehensive_edge_editing_analysis.png'")
    except:
        print("Can not Saved in jupyter notebook environment.Just i can show this only.")
    if show:
        plt.show()

    print("\n" + "=" * 80)
    print("EDGE EDITING ANALYSIS COMPLETE!")
    print("=" * 80)

def gnn_split_train_nodes(dataset_name, supervised=True, seed=None, llm_name=None):
    """Train node ids of the split the GNNs are trained/evaluated on (same split seed as GNNTrainer).
    load_dataset(cfg) keeps the dataset's original masks, which are NOT this split."""
    from gnns.gnn_mtrainer import resolve_path_prefix
    split_cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name)
    data, _, _ = load_graph_dataset_for_tape(dataset_name, 'cpu', re_split=1 if supervised else 0,
                                             path_prefix=resolve_path_prefix(), seed=split_cfg.split_seed)
    return set(torch.where(data.train_mask.cpu())[0].tolist())


def keep_train_mistakes(mistakes, dataset_name, supervised=True, seed=None, llm_name=None):
    """Mistakes are found with ground-truth labels, so only train-node mistakes may drive a graph edit;
    val/test mistakes would leak their labels into the modified graph the GNN is evaluated on."""
    train_nodes = gnn_split_train_nodes(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name)
    kept = [int(n) for n in mistakes if int(n) in train_nodes]
    print(f"  Using {len(kept)}/{len(mistakes)} mistakes (train nodes only; val/test mistakes dropped)")
    return kept


def get_best_similarity_threshold(mistakes, dataset, node_embeddings, data_emb, tokenizer, model, device, cfg,  sentence_transformer_used=False, reporter=None, title="", supervised=True, used_mistakes=False,
                                   seed=None, llm_name=None, threshold_range=None):
    similarities_th_range = threshold_range if threshold_range is not None else np.arange(0.1, 0.95, 0.02)
    best_acc, best_th, best_modified_dataset, best_stats, best_results, best_model = 0, 0, None, None, None, None
    best_val_acc = -1.0
    print(f"Searching for best similarity threshold (testing {len(similarities_th_range)} thresholds)...")
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    node_list = list(range(dataset.num_nodes))
    if used_mistakes:
        mistakes = keep_train_mistakes(mistakes, cfg.dataset.name, supervised=supervised, seed=seed,
                                       llm_name=llm_name)
    else:
        mistakes = node_list
    for th in similarities_th_range:
        modified_dataset, stats = create_modified_dataset_for_mistakes(
                mistakes,
                dataset,
                node_embeddings,
                tokenizer,
                model,
                device,
                th,
            sentence_transformer_used=sentence_transformer_used
        )
        results_mod, model_mod = gnn_train_and_report_with_modified_dataset(
            modified_dataset=modified_dataset,
            embedding=data_emb,
            dataset_name=cfg.dataset.name,
            supervised=supervised,
            seed=seed,
            llm_name=llm_name,
        )
        if results_mod['val_acc'] > best_val_acc:  # select the threshold on validation accuracy only
            best_val_acc = results_mod['val_acc']
            best_acc = results_mod['test_acc']
            best_th = th
            best_modified_dataset = modified_dataset
            best_stats = stats
            best_results = results_mod
            best_model = model_mod
    print('Best test accuracy: ', best_acc)
    print('Best test threshold: ', best_th)
    
    # Report results using reporter
    if reporter is not None:
        report_title = f"Best Similarity Threshold Search - {title}" if title else "Best Similarity Threshold Search"
        report_text = f"""
Best Similarity Threshold Search Results:
  • Best Threshold: {best_th:.4f}
  • Best Test Accuracy: {best_acc:.4f} ({best_acc:.2%})
  • Best Validation Accuracy: {best_results['val_acc']:.4f} ({best_results['val_acc']:.2%})
  • Best Test F1 (Macro): {best_results['test_f1']:.4f}
  • Best Test F1 (Weighted): {best_results['test_weight_f1']:.4f}
  • Best Validation F1 (Macro): {best_results['val_f1']:.4f}
  • Best Validation F1 (Weighted): {best_results['val_weight_f1']:.4f}

Edge Removal Statistics:
  • Total Edges Checked: {best_stats['total_checked']}
  • Edges Removed: {best_stats['removed']}
  • Edges Kept: {best_stats['kept']}
  • Original Edges: {best_stats['original_edges']}
  • Modified Edges: {best_stats['modified_edges']}
  • Edge Reduction: {((best_stats['original_edges'] - best_stats['modified_edges']) / best_stats['original_edges'] * 100):.2f}%
"""
        reporter.report(report_title, report_text)
    
    return best_acc, best_th, best_modified_dataset, best_stats, best_results, best_model

def visualize_edges_info_list(similarities_list, edge_info_list, titles_list, cfg, reporter=None, show=False):
    """
        similarities_list: list of similarities
        edge_info_list: list of edge_info
        titles_list: list of titles for each subplot
    """
    print("\nGenerating comparative visualizations...")
    
    n_methods = len(similarities_list)
    fig, axes = plt.subplots(n_methods, 3, figsize=(18, 5 * n_methods))
    
    # Handle case where there's only one method
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    
    for idx, (similarities, edge_info, title) in enumerate(zip(similarities_list, edge_info_list, titles_list)):
        # Get same/different label similarities
        same_label_sims = [info['similarity'] for info in edge_info if info['same_label']]
        diff_label_sims = [info['similarity'] for info in edge_info if not info['same_label']]
        
        # Plot 1: Histogram with KDE
        ax1 = axes[idx, 0]
        ax1.hist(similarities, bins=30, alpha=0.7, color=colors[idx % len(colors)], edgecolor='black', density=True)
        density = gaussian_kde(similarities)
        xs = np.linspace(min(similarities), max(similarities), 200)
        ax1.plot(xs, density(xs), 'r-', linewidth=2, label='KDE')
        ax1.set_xlabel('Cosine Similarity', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.set_title(f'{title}: Distribution of Edge Cosine Similarities', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Box plot comparing same label vs different label
        ax2 = axes[idx, 1]
        box_data = [same_label_sims, diff_label_sims]
        bp = ax2.boxplot(box_data, labels=['Same Label', 'Different Label'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Cosine Similarity', fontsize=10)
        ax2.set_title(f'{title}: Same vs Different Labels', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: Statistics table
        ax3 = axes[idx, 2]
        ax3.axis('off')
        stats_text = f"""
{title} Statistics:
{'=' * 40}

Overall:
  • Total Edges: {len(similarities)}
  • Mean Similarity: {np.mean(similarities):.4f}
  • Median: {np.median(similarities):.4f}
  • Std Dev: {np.std(similarities):.4f}
  • Min: {np.min(similarities):.4f}
  • Max: {np.max(similarities):.4f}

Same Label Edges:
  • Count: {len(same_label_sims)}
  • Mean: {np.mean(same_label_sims):.4f}
  • Std Dev: {np.std(same_label_sims):.4f}

Different Label Edges:
  • Count: {len(diff_label_sims)}
  • Mean: {np.mean(diff_label_sims):.4f}
  • Std Dev: {np.std(diff_label_sims):.4f}

Quartiles:
  • Q1 (25%): {np.percentile(similarities, 25):.4f}
  • Q2 (50%): {np.percentile(similarities, 50):.4f}
  • Q3 (75%): {np.percentile(similarities, 75):.4f}
        """
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Report statistics using reporter
        if reporter is not None:
            report_text = f"""
                            Overall Statistics:
                            • Total Edges: {len(similarities)}
                            • Mean Similarity: {np.mean(similarities):.4f}
                            • Median Similarity: {np.median(similarities):.4f}
                            • Std Deviation: {np.std(similarities):.4f}
                            • Min Similarity: {np.min(similarities):.4f}
                            • Max Similarity: {np.max(similarities):.4f}

                            Same Label Edges:
                            • Count: {len(same_label_sims)}
                            • Mean Similarity: {np.mean(same_label_sims):.4f}
                            • Std Deviation: {np.std(same_label_sims):.4f}

                            Different Label Edges:
                            • Count: {len(diff_label_sims)}
                            • Mean Similarity: {np.mean(diff_label_sims):.4f}
                            • Std Deviation: {np.std(diff_label_sims):.4f}

                            Quartiles:
                            • Q1 (25%): {np.percentile(similarities, 25):.4f}
                            • Q2 (50%): {np.percentile(similarities, 50):.4f}
                            • Q3 (75%): {np.percentile(similarities, 75):.4f}
                            """
            reporter.report(f"Edge Similarity Analysis - {title}", report_text)
    
    plt.tight_layout()
    os.makedirs('results/edge_analysis', exist_ok=True)
    try:
        filename = f'edge_similarity_comparison_list_{cfg.dataset.name}.png'
        plt.savefig(f'results/edge_analysis/{filename}', dpi=300, bbox_inches='tight')
        print(f"Comparative visualization saved as '{filename}'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    if show:
        plt.show()

def print_modified_edge_result(apporach_name, before_results, after_results, reporter=None):
    print("\n" + "=" * 80)
    print(f"{apporach_name} test data acc before editing:", before_results['test_acc'])
    print(f"{apporach_name} test data acc after editing:", after_results['test_acc'])
    print(f"{apporach_name} test data f1 before editing:", before_results['test_f1'])
    print(f"{apporach_name} test data f1 after editing:", after_results['test_f1'])
    print(f"{apporach_name} val data acc before editing:", before_results['val_acc'])
    print(f"{apporach_name} val data acc after editing:", after_results['val_acc'])
    print(f"{apporach_name} val data f1 before editing:", before_results['val_f1'])
    print(f"{apporach_name} val data f1 after editing:", after_results['val_f1'])
    print("=" * 80)
    
    acc_improvement = after_results['test_acc'] - before_results['test_acc']
    f1_improvement = after_results['test_f1'] - before_results['test_f1']
    val_acc_improvement = after_results['val_acc'] - before_results['val_acc']
    val_f1_improvement = after_results['val_f1'] - before_results['val_f1']
    
    print(f"{apporach_name} Improvement in Accuracy: ", acc_improvement)
    print(f"{apporach_name} Improvement in F1 Score: ", f1_improvement)
    
    # Report results using reporter
    if reporter is not None:
        report_text = f"""
Edge Editing Results Comparison:

Before Edge Editing:
  • Test Accuracy: {before_results['test_acc']:.4f} ({before_results['test_acc']:.2%})
  • Test F1 (Macro): {before_results['test_f1']:.4f}
  • Test F1 (Weighted): {before_results['test_weight_f1']:.4f}
  • Validation Accuracy: {before_results['val_acc']:.4f} ({before_results['val_acc']:.2%})
  • Validation F1 (Macro): {before_results['val_f1']:.4f}
  • Validation F1 (Weighted): {before_results['val_weight_f1']:.4f}

After Edge Editing:
  • Test Accuracy: {after_results['test_acc']:.4f} ({after_results['test_acc']:.2%})
  • Test F1 (Macro): {after_results['test_f1']:.4f}
  • Test F1 (Weighted): {after_results['test_weight_f1']:.4f}
  • Validation Accuracy: {after_results['val_acc']:.4f} ({after_results['val_acc']:.2%})
  • Validation F1 (Macro): {after_results['val_f1']:.4f}
  • Validation F1 (Weighted): {after_results['val_weight_f1']:.4f}

Improvements:
  • Test Accuracy Improvement: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)
  • Test F1 Improvement: {f1_improvement:+.4f}
  • Validation Accuracy Improvement: {val_acc_improvement:+.4f} ({val_acc_improvement*100:+.2f}%)
  • Validation F1 Improvement: {val_f1_improvement:+.4f}
"""
        reporter.report(f"Edge Editing Results - {apporach_name}", report_text)


def main(dataset_name, llm_name, peft_type, retrained_with_gnn_mistakes=True, reporter_index=0, ensemble_approaches='learnable'):
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
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
    dataset, _, _ = load_graph_dataset_for_tape(dataset_name, 'cuda:0', re_split=1, path_prefix=path_prefix, seed=cfg.dataset.seed)
    labels = dataset.y
    n = 2708
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not retrained_with_gnn_mistakes:
        data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
        print("Getting Data Without retrained gnn mistakes")
    else:
        data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn_with_retrained_gnn_mistake(cfg)
        print("Getting Data With retrained gnn mistakes")
    emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian = get_embedding_from_data(
        data_pissa), get_embedding_from_data(data_orthogonal), get_embedding_from_data(
        data_loftq), get_embedding_from_data(data_eva), get_embedding_from_data(data_guassian)
    node_pissa_embeddings = get_precomputed_node_embeddings(dataset, emb_pissa)
    node_orthogonal_embeddings = get_precomputed_node_embeddings(dataset, emb_orthogonal)
    node_loftq_embeddings = get_precomputed_node_embeddings(dataset, emb_loftq)
    node_eva_embeddings = get_precomputed_node_embeddings(dataset, emb_eva)
    node_guassian_embeddings = get_precomputed_node_embeddings(dataset, emb_guassian)
    sampled_edges = sample_edges(5429, dataset)

    tokenizer, model = load_encoder_model("intfloat/e5-large")
    edge_pissa_info, simiarities_pissa = calculate_cosine_similarities(sampled_edges, node_pissa_embeddings, dataset,
                                                                       sentence_transformer_used=False)
    edge_orthogonal_info, simiarities_orthogonal = calculate_cosine_similarities(sampled_edges,
                                                                                 node_orthogonal_embeddings, dataset,
                                                                                 sentence_transformer_used=False)
    edge_loftq_info, simiarities_loftq = calculate_cosine_similarities(sampled_edges, node_loftq_embeddings, dataset,
                                                                       sentence_transformer_used=False)
    edge_eva_info, simiarities_eva = calculate_cosine_similarities(sampled_edges, node_eva_embeddings, dataset,
                                                                   sentence_transformer_used=False)
    edge_guassian_info, simiarities_guassian = calculate_cosine_similarities(sampled_edges, node_guassian_embeddings,
                                                                             dataset, sentence_transformer_used=False)
    # visualize_edges_info(simiarities_pissa, edge_pissa_info, cfg, 'PISSA')
    # visualize_edges_info(simiarities_orthogonal, edge_orthogonal_info, cfg, 'ORTHOGONAL')
    # visualize_edges_info(simiarities_loftq, edge_loftq_info, cfg, 'LOFTQ')
    # visualize_edges_info(simiarities_eva, edge_eva_info, cfg, 'EVA')
    # visualize_edges_info(simiarities_guassian, edge_guassian_info, cfg, 'GUASSIAN')
    # simiarities_list = [simiarities_pissa, simiarities_orthogonal, simiarities_loftq, simiarities_eva,
    #                     simiarities_guassian]
    # edge_info_list = [edge_pissa_info, edge_orthogonal_info, edge_loftq_info, edge_eva_info, edge_guassian_info]
    # titles_list = ['PISSA', 'ORTHOGONAL', 'LOFTQ', 'EVA', 'GUASSIAN']
    # visualize_edges_info_list(simiarities_list, edge_info_list, titles_list, cfg, reporter=reporter)

    from visualize.visualize_embeddings import visualize_pca_list, visualize_tsne_list, visualize_umap_list, \
        visualize_classification_metrics_list, visualize_clustering_metrics_list
    emb_list = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
    # visualize_tsne_list(cfg, dataset, emb_list, labels, titles_list, n=n, reporter=None)
    # visualize_umap_list(cfg, dataset, emb_list, labels, titles_list, n=n, reporter=None)
    # visualize_pca_list(cfg, dataset, emb_list, labels, titles_list, n=n, reporter=None)
    # visualize_classification_metrics_list(cfg, dataset, emb_list, labels, titles_list, reporter=reporter)
    # visualize_clustering_metrics_list(cfg, dataset, emb_list, labels, titles_list, reporter=reporter)

    ## train gnn and get mistakes
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
    model_before_edge_editing_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_guassian]
    custom_before_edge_editing_embeddings = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    cora_dataset = load_dataset(cfg)
    gnn_train_pissa_mistakes, gnn_val_pissa_mistakes, gnn_test_pissa_mistakes, gnn_pissa_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_pissa, emb_pissa, output_mask='all')
    gnn_train_orthogonal_mistakes, gnn_val_orthogonal_mistakes, gnn_train_test_mistakes, gnn_orthogonal_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_orthogonal, emb_orthogonal, output_mask='all')
    gnn_train_loftq_mistakes, gnn_val_loftq_mistakes, gnn_test_loftq_mistakes,gnn_loftq_mistakes  = get_gnn_embedding_mistakes(cora_dataset, model_loftq, emb_loftq, output_mask='all')
    gnn_train_eva_mistakes, gnn_val_eva_mistakes, gnn_test_eva_mistakes, gnn_eva_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_eva, emb_eva, output_mask='all')
    gnn_train_guassian_mistakes, gnn_val_guassian_mistakes, gnn_test_guassian_mistakes, gnn_guassian_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_guassian, emb_guassian, output_mask='all')
    # from visualize_gnn_embedding import visualize_gnn_embedding_list
    # from visualize_gnn_mistakes import visualize_gnn_embedding_mistakes_list, \
    #     visualize_init_weights_embedding_mistakes_list, comprehensive_gnn_mistakes_analysis_visualization, \
    #     comprehensive_gnn_mistakes_analysis_visualization_list
    # visualize_gnn_embedding_list(cfg,
    #                              dataset,
    #                              emb_list,
    #                              model_before_edge_editing_list,
    #                              titles_list,
    #                              device,
    #                              dataset_name, n)
    # visualize_gnn_embedding_mistakes_list(cfg,
    #                                       dataset,
    #                                       model_before_edge_editing_list,
    #                                       emb_list,
    #                                       titles_list, )
    # visualize_init_weights_embedding_mistakes_list(cfg,
    #                                                dataset,
    #                                                model_before_edge_editing_list,
    #                                                emb_list,
    #                                                titles_list
    #                                                )
    # comprehensive_gnn_mistakes_analysis_visualization(cfg, dataset, model_pissa, emb_pissa, "PISSA", )
    # comprehensive_gnn_mistakes_analysis_visualization(cfg, dataset, model_orthogonal, emb_orthogonal, "ORTHOGONAL")
    # comprehensive_gnn_mistakes_analysis_visualization(cfg, dataset, model_loftq, emb_loftq, "LOFTQ")
    # comprehensive_gnn_mistakes_analysis_visualization(cfg, dataset, model_eva, emb_eva, "EVA")
    # comprehensive_gnn_mistakes_analysis_visualization(cfg, dataset, model_guassian, emb_guassian, "GUASSIAN")
    # comprehensive_gnn_mistakes_analysis_visualization_list(cfg, dataset, model_before_edge_editing_list,
    #                                                        custom_before_edge_editing_embeddings, titles_list,
    #                                                        reporter=reporter)
    pissa_acc, pissa_best_th, pissa_modified_dataset, pissa_stats, piss_modified_results, pissa_modified_model = get_best_similarity_threshold(
        gnn_pissa_mistakes, cora_dataset, node_pissa_embeddings, emb_pissa, tokenizer, model, device, cfg,
        sentence_transformer_used=False, reporter=reporter, title="PISSA")
    # orthogonal_acc, orthogonal_best_th, orthogonal_modified_dataset, orthogonal_stats, orthogonal_modified_results, orthogonal_modified_model = get_best_similarity_threshold(
    #     gnn_orthogonal_mistakes, cora_dataset, node_orthogonal_embeddings, emb_orthogonal, tokenizer, model, device,
    #     cfg,
    #     sentence_transformer_used=False, reporter=reporter, title="ORTHOGONAL")
    # loftq_acc, loftq_best_th, loftq_modified_dataset, loftq_stats, loftq_modified_results, loftq_modified_model = get_best_similarity_threshold(
    #     gnn_loftq_mistakes, cora_dataset, node_loftq_embeddings, emb_loftq, tokenizer, model, device, cfg,
    #     sentence_transformer_used=False, reporter=reporter, title="LOFTQ")
    # eva_acc, eva_best_th, eva_modified_dataset, eva_stats, eva_modified_results, eva_modified_model = get_best_similarity_threshold(
    #     gnn_eva_mistakes, cora_dataset, node_eva_embeddings, emb_eva, tokenizer, model, device, cfg,
    #     sentence_transformer_used=False, reporter=reporter, title="EVA")
    guassian_acc, guassian_best_th, guassian_modified_dataset, guassian_stats, guassian_modified_results, guassian_modified_model = get_best_similarity_threshold(
        gnn_guassian_mistakes, cora_dataset, node_guassian_embeddings, emb_guassian, tokenizer, model, device, cfg,
        sentence_transformer_used=False, reporter=reporter, title="GUASSIAN")
    # print_modified_edge_result("PISSA", results_pissa, piss_modified_results, reporter=reporter)
    # print_modified_edge_result("ORTHOGONAL", results_orthogonal, orthogonal_modified_results, reporter=reporter)
    # print_modified_edge_result("LOFTQ", results_loftq, loftq_modified_results, reporter=reporter)
    # print_modified_edge_result("EVA", results_eva, eva_modified_results, reporter=reporter)
    # print_modified_edge_result("GUASSIAN", results_guassian, guassian_modified_results, reporter=reporter)
    # all_original_results = {
    #     'pissa': results_pissa,
    #     'orthogonal': results_orthogonal,
    #     'loftq': results_loftq,
    #     'eva': results_eva,
    #     'guassian': results_guassian
    # }
    # modified_results = {
    #     'pissa': piss_modified_results,
    #     'orthogonal': orthogonal_modified_results,
    #     'loftq': loftq_modified_results,
    #     'eva': eva_modified_results,
    #     'guassian': guassian_modified_results
    # }
    # all_mistakes = {
    #     'pissa': gnn_pissa_mistakes,
    #     'orthogonal': gnn_orthogonal_mistakes,
    #     'loftq': gnn_loftq_mistakes,
    #     'eva': gnn_eva_mistakes,
    #     'guassian': gnn_guassian_mistakes
    # }
    # removal_stats_all = {
    #     'pissa': pissa_stats,
    #     'orthogonal': orthogonal_stats,
    #     'loftq': loftq_stats,
    #     'eva': eva_stats,
    #     'guassian': guassian_stats
    # }
    # visualize_edge_editing_results(all_original_results, modified_results, all_mistakes, removal_stats_all, cfg,
    #                                show=False)
    ensemble_results_before, best_params_before, _ = tune_ensemble_hyperparameter(
        dataset_name,
        model_before_edge_editing_list,
        custom_before_edge_editing_embeddings,
        does_report_training_process=False,
        ensemble_approaches=ensemble_approaches,
        title="Before Edge Editing",
        reporter=reporter
    )
    print("\nBest ensemble parameters (before edge editing):")
    print(f"{best_params_before}")
    print(f"\nBest ensemble results (before edge editing):")
    print(f"  Test Accuracy: {ensemble_results_before['test']['accuracy']:.4f}")
    print(f"  Test F1: {ensemble_results_before['test']['macro_f1']:.4f}")

    # model_after_edge_editing_list = [
    #     pissa_modified_model,
    #     orthogonal_modified_model,
    #     loftq_modified_model,
    #     eva_modified_model,
    #     guassian_modified_model]
    # custom_after_edge_editing_embeddings = [
    #     emb_pissa,
    #     emb_orthogonal,
    #     emb_loftq,
    #     emb_eva,
    #     emb_guassian
    # ]
    # modified_datasets_list = [
    #     pissa_modified_dataset,
    #     orthogonal_modified_dataset,
    #     loftq_modified_dataset,
    #     eva_modified_dataset,
    #     guassian_modified_dataset
    # ]
    # ensemble_results_after, best_params_after, _ = tune_ensemble_hyperparameter(
    #     dataset_name,
    #     model_after_edge_editing_list,
    #     custom_after_edge_editing_embeddings,
    #     modified_datasets_list=modified_datasets_list,
    #     does_report_training_process=False,
    #     ensemble_approaches=ensemble_approaches,
    #     title="After Edge Editing",
    #     reporter=reporter
    # )
    # print("\nBest ensemble parameters (after edge editing):")
    # print(f"  {best_params_after}")
    # print(f"\nBest ensemble results (after edge editing):")
    # print(f"  Test Accuracy: {ensemble_results_after['test']['accuracy']:.4f}")
    # print(f"  Test F1: {ensemble_results_after['test']['macro_f1']:.4f}")
    # # Compare ensemble results
    # print("\n" + "=" * 80)
    # print("ENSEMBLE LEARNING COMPARISON")
    # print("=" * 80)
    #
    # print("\nBEFORE EDGE EDITING:")
    # print(f"  Test Accuracy: {ensemble_results_before['test']['accuracy']:.4f}")
    # print(f"  Test F1: {ensemble_results_before['test']['macro_f1']:.4f}")
    #
    # print("\nAFTER EDGE EDITING:")
    # print(f"  Test Accuracy: {ensemble_results_after['test']['accuracy']:.4f}")
    # print(f"  Test F1: {ensemble_results_after['test']['macro_f1']:.4f}")
    #
    # print("\nENSEMBLE IMPROVEMENT:")
    # ensemble_acc_improvement = ensemble_results_after['test']['accuracy'] - ensemble_results_before['test']['accuracy']
    # ensemble_f1_improvement = ensemble_results_after['test']['macro_f1'] - ensemble_results_before['test']['macro_f1']
    # print(f"  Test Accuracy: {ensemble_acc_improvement:+.4f} ({ensemble_acc_improvement * 100:+.2f}%)")
    # print(f"  Test F1: {ensemble_f1_improvement:+.4f} ({ensemble_f1_improvement * 100:+.2f}%)")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Edge Editing Analysis')
    parser.add_argument('--dataset_name', type=str, default='arxiv',
                        help='Name of the dataset (default: wikics)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B', 
                        help='Name of the LLM model (default: llama_3.2_1B)')
    parser.add_argument('--peft_type', type=str, default='lora', 
                        help='Type of PEFT (default: lora)')
    parser.add_argument('--retrained_with_gnn_mistakes', type=bool, default=True,
                        help='Whether to retrain the GNN with mistakes (default: True)')
    parser.add_argument('--reporter_index', type=int, default=0,
                        help='Index of the reporter (default: 0)')
    parser.add_argument('--ensemble_approaches', type=str, default='learnable',
                        help='Approaches for ensemble (default: learnable)')
    args = parser.parse_args()
    main(args.dataset_name, args.llm_name, args.peft_type, args.retrained_with_gnn_mistakes, args.reporter_index, args.ensemble_approaches)