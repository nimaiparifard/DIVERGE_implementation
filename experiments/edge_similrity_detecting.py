from dataset.dataset_loader import load_dataset
from config import setup_finetuning_cfg
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


## bring e5-large model

print("Loading e5-large model and tokenizer...")
model_name = "intfloat/e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model loaded on device: {device}")

## load cora dataset
print("\nLoading Cora dataset...")
cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
cora_dataset = load_dataset(cfg)
print(f"Dataset loaded: {cora_dataset.num_nodes} nodes, {cora_dataset.edge_index.shape[1]} edges")

## sample 100 edges from dataset
num_samples = min(2000, cora_dataset.edge_index.shape[1])
edge_indices = torch.randperm(cora_dataset.edge_index.shape[1])[:num_samples]
sampled_edges = cora_dataset.edge_index[:, edge_indices]
print(f"\nSampled {num_samples} edges from dataset")

## bring the text text of 100 edges nodes and tokenize it using e5-large tokenizer and get embeddings and do mean pooling to get node embeddings
print("\nProcessing node texts and generating embeddings...")

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
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

# Get embeddings for all unique nodes in sampled edges
unique_nodes = torch.unique(sampled_edges.flatten()).tolist()
node_embeddings = {}

print(f"Getting embeddings for {len(unique_nodes)} unique nodes...")
for idx, node_id in enumerate(unique_nodes):
    if idx % 10 == 0:
        print(f"Processing node {idx+1}/{len(unique_nodes)}")
    node_text = cora_dataset.raw_texts[node_id]
    node_embeddings[node_id] = get_embedding(node_text)

## calculate cosine similarity between the embeddings of the nodes in each edge and print the similarity score along with the edge information
print("\nCalculating cosine similarities...")
similarities = []
edge_info = []

for i in range(sampled_edges.shape[1]):
    node1_id = sampled_edges[0, i].item()
    node2_id = sampled_edges[1, i].item()

    emb1 = node_embeddings[node1_id]
    emb2 = node_embeddings[node2_id]

    # Calculate cosine similarity
    cosine_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    similarities.append(cosine_sim)

    node1_label = cora_dataset.y[node1_id].item()
    node2_label = cora_dataset.y[node2_id].item()

    edge_info.append({
        'edge_idx': i,
        'node1_id': node1_id,
        'node2_id': node2_id,
        'node1_label': node1_label,
        'node2_label': node2_label,
        'similarity': cosine_sim,
        'same_label': node1_label == node2_label
    })

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
{'='*50}

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
plt.savefig('edge_similarity_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'edge_similarity_analysis.png'")
plt.show()

## also print the result like cosine similirity edge sample i: {} node1 label: {} node2 label: {}
print("\n" + "="*80)
print("DETAILED EDGE SIMILARITY RESULTS")
print("="*80)

# Sort by similarity for better viewing
sorted_edge_info = sorted(edge_info, key=lambda x: x['similarity'], reverse=True)

for info in sorted_edge_info:
    same_label_indicator = "✓" if info['same_label'] else "✗"
    print(f"Edge Sample {info['edge_idx']:3d}: Cosine Similarity = {info['similarity']:.4f} | "
          f"Node1 (ID:{info['node1_id']:4d}, Label:{info['node1_label']}) <-> "
          f"Node2 (ID:{info['node2_id']:4d}, Label:{info['node2_label']}) | "
          f"Same Label: {same_label_indicator}")

print("\n" + "="*80)
print(f"Summary: {len(same_label_sims)}/{len(similarities)} edges connect nodes with the same label")
print("="*80)

# run gnn and got mistake
from gnns.gnn_mtrainer import GNNTrainer, set_mgnn_cfg, gnn_train_and_report
from visualize.visualize_gnn_mistakes import get_gnn_embedding_mistakes
data_path_pissa = 'artifacts/cache/llama_3.2_1B_cora_seqcls_lora_init-pissa_pool-mean.pt'
data_path_orthogonal = 'artifacts/cache/llama_3.2_1B_cora_seqcls_lora_init-orthogonal_pool-mean.pt'
data_path_loftq = 'artifacts/cache/llama_3.2_1B_cora_seqcls_lora_init-loftq_pool-mean.pt'
data_path_eva = 'artifacts/cache/llama_3.2_1B_cora_seqcls_lora_init-eva_pool-mean.pt'
data_path_guassian = 'artifacts/cache/llama_3.2_1B_cora_seqcls_lora_init-gaussian_pool-mean.pt'
data_pissa = torch.load(data_path_pissa, map_location=device)
data_orthogonal = torch.load(data_path_orthogonal, map_location=device)
data_eva = torch.load(data_path_eva, map_location=device)
data_loftq = torch.load(data_path_loftq, map_location=device)
data_guassian = torch.load(data_path_guassian, map_location=device)
emb_pissa = data_pissa['embeddings']
emb_orthogonal = data_orthogonal['embeddings']
emb_loftq = data_loftq['embeddings']
emb_eva = data_eva['embeddings']
emb_guassian = data_guassian['embeddings']
results_pissa, model_pissa = gnn_train_and_report('cora', emb_pissa)
results_orthogonal, model_orthogonal  = gnn_train_and_report('cora', emb_orthogonal)
results_loftq, model_loftq = gnn_train_and_report('cora', emb_loftq)
results_eva, model_eva  = gnn_train_and_report('cora', emb_eva)
results_guassian, model_guassian  = gnn_train_and_report('cora', emb_guassian)
model_before_edge_editing_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_guassian]
custom_before_edge_editing_embeddings = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
cora_dataset = load_dataset(cfg)
gnn_pissa_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_pissa, emb_pissa)
gnn_orthogonal_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_orthogonal, emb_orthogonal)
gnn_loftq_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_loftq, emb_loftq)
gnn_eva_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_eva, emb_eva)
gnn_guassian_mistakes = get_gnn_embedding_mistakes(cora_dataset, model_guassian, emb_guassian)


# for each model mistake create modified dataset by removing edges with similarity less than 0.75
# also after reaching modified dataset run gnn and report the result and diffrence between them

# Store all mistakes
all_mistakes = {
    'pissa': gnn_pissa_mistakes,
    'orthogonal': gnn_orthogonal_mistakes,
    'loftq': gnn_loftq_mistakes,
    'eva': gnn_eva_mistakes,
    'guassian': gnn_guassian_mistakes
}

# Store all embeddings
all_embeddings = {
    'pissa': emb_pissa,
    'orthogonal': emb_orthogonal,
    'loftq': emb_loftq,
    'eva': emb_eva,
    'guassian': emb_guassian
}

# Store original results
all_original_results = {
    'pissa': results_pissa,
    'orthogonal': results_orthogonal,
    'loftq': results_loftq,
    'eva': results_eva,
    'guassian': results_guassian
}

# Function to create modified dataset for a specific set of mistakes
def create_modified_dataset_for_mistakes(mistakes, dataset, node_embeddings, similarity_threshold=0.75):
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
            node_embeddings[node_id] = get_embedding(node_text)
    
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

print("\n" + "="*80)
print("CREATING MODIFIED DATASETS BY REMOVING LOW-SIMILARITY EDGES")
print("="*80)

# Process each model and create modified datasets
similarity_threshold = 0.80
modified_datasets = {}
removal_stats_all = {}
modified_results = {}
model_after_edge_editing_list = []
custom_after_edge_editing_embeddings = []

from gnns.gnn_mtrainer import gnn_train_and_report_with_modified_dataset

for model_name in all_mistakes.keys():
    print(f"\n{'='*80}")
    print(f"Processing {model_name.upper()} model")
    print(f"{'='*80}")
    if model_name in ['eva', 'pissa']:
        similarity_threshold = 0.80
    if model_name in ['loftq', 'orthogonal']:
        similarity_threshold = 0.80
    else:
        similarity_threshold = 0.70
    # Create modified dataset for this model's mistakes
    modified_dataset, stats = create_modified_dataset_for_mistakes(
        all_mistakes[model_name],
        cora_dataset,
        node_embeddings,
        similarity_threshold
    )
    
    modified_datasets[model_name] = modified_dataset
    removal_stats_all[model_name] = stats
    
    # Train GNN on modified dataset
    print(f"\n  Training GNN on modified dataset...")
    results_mod, model_mod = gnn_train_and_report_with_modified_dataset(
        modified_dataset=modified_dataset,
        embedding=all_embeddings[model_name],
        dataset_name='cora'
    )
    
    modified_results[model_name] = results_mod
    model_after_edge_editing_list.append(model_mod)
    custom_after_edge_editing_embeddings.append(all_embeddings[model_name])
    
    # Print comparison
    print(f"\n  {'='*60}")
    print(f"  COMPARISON FOR {model_name.upper()}")
    print(f"  {'='*60}")
    print(f"  Original → Modified")
    print(f"    Test Accuracy:   {all_original_results[model_name]['test_acc']:.4f} → {results_mod['test_acc']:.4f} "
          f"({(results_mod['test_acc'] - all_original_results[model_name]['test_acc'])*100:+.2f}%)")
    print(f"    Test F1 (Macro): {all_original_results[model_name]['test_f1']:.4f} → {results_mod['test_f1']:.4f} "
          f"({(results_mod['test_f1'] - all_original_results[model_name]['test_f1'])*100:+.2f}%)")

# Create comprehensive comparison visualization
print("\n" + "="*80)
print("CREATING COMPREHENSIVE VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

model_names = list(all_mistakes.keys())
colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

# Plot 1: Test Accuracy Comparison
ax = axes[0, 0]
x = np.arange(len(model_names))
width = 0.35
original_acc = [all_original_results[m]['test_acc'] for m in model_names]
modified_acc = [modified_results[m]['test_acc'] for m in model_names]
bars1 = ax.bar(x - width/2, original_acc, width, label='Original', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, modified_acc, width, label='Modified', color='lightcoral', edgecolor='black')
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
bars1 = ax.bar(x - width/2, original_f1, width, label='Original', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, modified_f1, width, label='Modified', color='lightcoral', edgecolor='black')
ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Test F1 (Macro)', fontsize=11)
ax.set_title('Test F1: Original vs Modified', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Improvement Heatmap
ax = axes[0, 2]
acc_improvements = [(modified_results[m]['test_acc'] - all_original_results[m]['test_acc'])*100 for m in model_names]
f1_improvements = [(modified_results[m]['test_f1'] - all_original_results[m]['test_f1'])*100 for m in model_names]
bars1 = ax.barh(x - width/2, acc_improvements, width, label='Accuracy Δ%', color='steelblue', edgecolor='black')
bars2 = ax.barh(x + width/2, f1_improvements, width, label='F1 Δ%', color='coral', edgecolor='black')
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
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
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
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
            ha='center', va='bottom', fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 6: Summary Statistics Table
ax = axes[1, 2]
ax.axis('off')
summary_text = "SUMMARY STATISTICS\n" + "="*50 + "\n\n"
for m in model_names:
    summary_text += f"{m.upper()}:\n"
    summary_text += f"  Mistakes: {len(all_mistakes[m])}\n"
    summary_text += f"  Edges Removed: {removal_stats_all[m]['removed']}\n"
    summary_text += f"  Acc Δ: {(modified_results[m]['test_acc'] - all_original_results[m]['test_acc'])*100:+.2f}%\n"
    summary_text += f"  F1 Δ: {(modified_results[m]['test_f1'] - all_original_results[m]['test_f1'])*100:+.2f}%\n\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('comprehensive_edge_editing_analysis.png', dpi=300, bbox_inches='tight')
print("Comprehensive analysis saved as 'comprehensive_edge_editing_analysis.png'")
plt.show()

print("\n" + "="*80)
print("EDGE EDITING ANALYSIS COMPLETE!")
print("="*80)


## run tune_ensemble for before modified and after modified dataset,gnn_models
print("\n" + "="*80)
print("ENSEMBLE LEARNING: BEFORE vs AFTER EDGE EDITING")
print("="*80)

from ensemble.ensemble_gnns_learning import ensemble_gnn_learning, tune_ensemble_hyperparameter

# run ensemble for before edge editing
print("\n" + "-"*80)
print("TUNING ENSEMBLE HYPERPARAMETERS - BEFORE EDGE EDITING")
print("-"*80)

ensemble_results_before, best_params_before = tune_ensemble_hyperparameter(
    'cora',
    model_before_edge_editing_list,
    custom_before_edge_editing_embeddings
)

print("\nBest ensemble parameters (before edge editing):")
print(f"{best_params_before}")
print(f"\nBest ensemble results (before edge editing):")
print(f"  Test Accuracy: {ensemble_results_before['test']['accuracy']:.4f}")
print(f"  Test F1: {ensemble_results_before['test']['macro_f1']:.4f}")

# run ensemble after edge editing
print("\n" + "-"*80)
print("TUNING ENSEMBLE HYPERPARAMETERS - AFTER EDGE EDITING")
print("-"*80)

# Create list of modified datasets in the same order as the models
modified_datasets_list = [modified_datasets[m] for m in model_names]

ensemble_results_after, best_params_after = tune_ensemble_hyperparameter(
    'cora',
    model_after_edge_editing_list,
    custom_after_edge_editing_embeddings,
    modified_datasets_list=modified_datasets_list
)

print("\nBest ensemble parameters (after edge editing):")
print(f"  {best_params_after}")
print(f"\nBest ensemble results (after edge editing):")
print(f"  Test Accuracy: {ensemble_results_after['test']['accuracy']:.4f}")
print(f"  Test F1: {ensemble_results_after['test']['macro_f1']:.4f}")

# Compare ensemble results
print("\n" + "="*80)
print("ENSEMBLE LEARNING COMPARISON")
print("="*80)

print("\nBEFORE EDGE EDITING:")
print(f"  Test Accuracy: {ensemble_results_before['test']['accuracy']:.4f}")
print(f"  Test F1: {ensemble_results_before['test']['macro_f1']:.4f}")

print("\nAFTER EDGE EDITING:")
print(f"  Test Accuracy: {ensemble_results_after['test']['accuracy']:.4f}")
print(f"  Test F1: {ensemble_results_after['test']['macro_f1']:.4f}")

print("\nENSEMBLE IMPROVEMENT:")
ensemble_acc_improvement = ensemble_results_after['test']['accuracy'] - ensemble_results_before['test']['accuracy']
ensemble_f1_improvement = ensemble_results_after['test']['macro_f1'] - ensemble_results_before['test']['macro_f1']
print(f"  Test Accuracy: {ensemble_acc_improvement:+.4f} ({ensemble_acc_improvement*100:+.2f}%)")
print(f"  Test F1: {ensemble_f1_improvement:+.4f} ({ensemble_f1_improvement*100:+.2f}%)")

# Create final comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Ensemble Accuracy Comparison
ax = axes[0]
categories = ['Before Edge Editing', 'After Edge Editing']
ensemble_accs = [ensemble_results_before['test']['accuracy'], ensemble_results_after['test']['accuracy']]
bars = ax.bar(categories, ensemble_accs, color=['steelblue', 'coral'], edgecolor='black', width=0.6)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Ensemble Test Accuracy: Before vs After Edge Editing', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim([min(ensemble_accs) - 0.02, max(ensemble_accs) + 0.02])

# Plot 2: Ensemble F1 Comparison
ax = axes[1]
ensemble_f1s = [ensemble_results_before['test']['macro_f1'], ensemble_results_after['test']['macro_f1']]
bars = ax.bar(categories, ensemble_f1s, color=['steelblue', 'coral'], edgecolor='black', width=0.6)
ax.set_ylabel('Test F1 (Macro)', fontsize=12)
ax.set_title('Ensemble Test F1: Before vs After Edge Editing', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim([min(ensemble_f1s) - 0.02, max(ensemble_f1s) + 0.02])

plt.tight_layout()
plt.savefig('ensemble_comparison_before_after_edge_editing.png', dpi=300, bbox_inches='tight')
print("\nEnsemble comparison saved as 'ensemble_comparison_before_after_edge_editing.png'")
plt.show()

# Final summary
print("\n" + "="*80)
print("COMPLETE EXPERIMENT SUMMARY")
print("="*80)

print("\nINDIVIDUAL MODEL IMPROVEMENTS (Average):")
avg_acc_improvement = np.mean([modified_results[m]['test_acc'] - all_original_results[m]['test_acc']
                               for m in model_names])
avg_f1_improvement = np.mean([modified_results[m]['test_f1'] - all_original_results[m]['test_f1']
                              for m in model_names])
print(f"  Average Accuracy Improvement: {avg_acc_improvement:+.4f} ({avg_acc_improvement*100:+.2f}%)")
print(f"  Average F1 Improvement: {avg_f1_improvement:+.4f} ({avg_f1_improvement*100:+.2f}%)")

print("\nENSEMBLE MODEL IMPROVEMENT:")
print(f"  Accuracy Improvement: {ensemble_acc_improvement:+.4f} ({ensemble_acc_improvement*100:+.2f}%)")
print(f"  F1 Improvement: {ensemble_f1_improvement:+.4f} ({ensemble_f1_improvement*100:+.2f}%)")

print("\nTOTAL EDGES REMOVED ACROSS ALL MODELS:")
total_edges_removed = sum([removal_stats_all[m]['removed'] for m in model_names])
print(f"  {total_edges_removed} edges removed (cumulative across all models)")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE! ALL RESULTS SAVED.")
print("="*80)
