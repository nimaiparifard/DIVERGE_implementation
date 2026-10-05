"""
Comprehensive comparison of embeddings with and without fine-tuning.
Analyzes differences to understand impact as initial features for GNNs.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    pairwise_distances
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from scipy.special import kl_div, rel_entr
import json
from config import setup_finetuning_cfg

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")




def compute_basic_statistics(embeddings, name=""):
    """Compute basic statistical properties of embeddings."""
    stats = {
        'mean': embeddings.mean().item(),
        'std': embeddings.std().item(),
        'min': embeddings.min().item(),
        'max': embeddings.max().item(),
        'l2_norm_mean': torch.norm(embeddings, dim=1).mean().item(),
        'l2_norm_std': torch.norm(embeddings, dim=1).std().item(),
        'sparsity': (embeddings == 0).float().mean().item(),
    }
    
    print(f"\n{'='*60}")
    print(f"Basic Statistics for {name}")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key:20s}: {value:.6f}")
    
    return stats


def compute_pairwise_similarities(emb1, emb2):
    """Compute various similarity/distance metrics between corresponding embeddings."""
    n_samples = emb1.shape[0]
    
    # Normalize for cosine similarity
    emb1_norm = emb1 / (torch.norm(emb1, dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (torch.norm(emb2, dim=1, keepdim=True) + 1e-8)
    
    # Cosine similarity
    cosine_sims = (emb1_norm * emb2_norm).sum(dim=1)
    
    # Euclidean distance
    euclidean_dists = torch.norm(emb1 - emb2, dim=1)
    
    # Manhattan distance
    manhattan_dists = torch.abs(emb1 - emb2).sum(dim=1)
    
    # Relative change
    relative_change = euclidean_dists / (torch.norm(emb1, dim=1) + 1e-8)
    
    metrics = {
        'cosine_similarity_mean': cosine_sims.mean().item(),
        'cosine_similarity_std': cosine_sims.std().item(),
        'cosine_similarity_min': cosine_sims.min().item(),
        'euclidean_distance_mean': euclidean_dists.mean().item(),
        'euclidean_distance_std': euclidean_dists.std().item(),
        'manhattan_distance_mean': manhattan_dists.mean().item(),
        'relative_change_mean': relative_change.mean().item(),
        'relative_change_std': relative_change.std().item(),
    }
    
    print(f"\n{'='*60}")
    print(f"Pairwise Similarity Metrics")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:30s}: {value:.6f}")
    
    return metrics, cosine_sims, euclidean_dists


def compute_distribution_divergence(emb1, emb2):
    """Compute distribution-level divergence metrics."""
    emb1_np = emb1.cpu().numpy().flatten()
    emb2_np = emb2.cpu().numpy().flatten()
    
    # Sample for computational efficiency if too large
    if len(emb1_np) > 100000:
        indices = np.random.choice(len(emb1_np), 100000, replace=False)
        emb1_np = emb1_np[indices]
        emb2_np = emb2_np[indices]
    
    # Wasserstein distance
    wasserstein = wasserstein_distance(emb1_np, emb2_np)
    
    # KL divergence (need to convert to probability distributions)
    # Use histograms
    bins = 100
    hist1, bin_edges = np.histogram(emb1_np, bins=bins, density=True)
    hist2, _ = np.histogram(emb2_np, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    kl_div_value = np.sum(rel_entr(hist1, hist2))
    js_div = 0.5 * (np.sum(rel_entr(hist1, (hist1 + hist2) / 2)) + 
                    np.sum(rel_entr(hist2, (hist1 + hist2) / 2)))
    
    metrics = {
        'wasserstein_distance': wasserstein,
        'kl_divergence': kl_div_value,
        'js_divergence': js_div,
    }
    
    print(f"\n{'='*60}")
    print(f"Distribution Divergence Metrics")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.6f}")
    
    return metrics


def compute_clustering_metrics(embeddings, labels, name=""):
    """Compute clustering quality metrics."""
    embeddings_np = embeddings.cpu().numpy()
    
    if len(np.unique(labels)) < 2:
        print(f"Warning: Less than 2 unique labels for {name}, skipping clustering metrics")
        return {}
    
    silhouette = silhouette_score(embeddings_np, labels, sample_size=min(5000, len(labels)))
    davies_bouldin = davies_bouldin_score(embeddings_np, labels)
    calinski = calinski_harabasz_score(embeddings_np, labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski,
    }
    
    print(f"\n{'='*60}")
    print(f"Clustering Metrics for {name}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.6f}")
    
    return metrics


def evaluate_classification(embeddings, labels, max_iter=1000,  name=""):
    """Evaluate classification performance using simple classifiers."""
    embeddings_np = embeddings.cpu().numpy()
    
    if len(np.unique(labels)) < 2:
        print(f"Warning: Less than 2 unique labels for {name}, skipping classification")
        return {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_np, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    
    # Linear classifier
    lr = LogisticRegression(max_iter=max_iter, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    
    metrics = {
        'knn_accuracy': knn_acc,
        'linear_accuracy': lr_acc,
    }
    
    print(f"\n{'='*60}")
    print(f"Classification Performance for {name}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f}")
    
    return metrics


def compute_structural_properties(embeddings, name=""):
    """Compute structural properties of embedding matrix."""
    embeddings_np = embeddings.cpu().numpy()
    
    # Compute covariance matrix
    cov_matrix = np.cov(embeddings_np.T)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    # Condition number
    cond_number = np.linalg.cond(embeddings_np)
    
    # Effective rank
    eigenvalues_normalized = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(eigenvalues_normalized * np.log(eigenvalues_normalized + 1e-10))
    effective_rank = np.exp(entropy)
    
    # Matrix rank
    matrix_rank = np.linalg.matrix_rank(embeddings_np)
    
    metrics = {
        'condition_number': cond_number,
        'matrix_rank': matrix_rank,
        'effective_rank': effective_rank,
        'top_10_eigenvalues_sum': eigenvalues[:10].sum() / eigenvalues.sum(),
        'eigenvalue_variance': eigenvalues.std(),
    }
    
    print(f"\n{'='*60}")
    print(f"Structural Properties for {name}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:30s}: {value:.6f}")
    
    return metrics, eigenvalues

def visualize_tsne(emb, labels, title="t-SNE Visualization"):
    """Visualize embeddings using t-SNE."""
    print("Generating t-SNE visualization...")
    emb_np = emb.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_np) - 1))
    emb1_tsne = tsne.fit_transform(emb_np)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter1 = axes[0].scatter(emb1_tsne[:, 0], emb1_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title(f'{title} (t-SNE)')
    plt.colorbar(scatter1, ax=axes[0])
    plt.tight_layout()
    plt.close()

def visualize_pca(emb, labels, save_path, title="PCA Visualization"):
    """Visualize embeddings using PCA."""
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb.cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_umap(emb, labels, save_path, title="UMAP Visualization"):
    """Visualize embeddings using UMAP."""
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping UMAP visualization.")
        return

    reducer = umap.UMAP(random_state=42)
    emb_umap = reducer.fit_transform(emb.cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_umap[:, 0], emb_umap[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def visualize_embeddings(emb1, emb2, labels, save_dir, title1, title2, prefix="last_layer"):
    """Create comprehensive visualizations comparing embeddings."""
    os.makedirs(save_dir, exist_ok=True)
    
    emb1_np = emb1.cpu().numpy()
    emb2_np = emb2.cpu().numpy()
    
    # 1. PCA visualization
    print("\nGenerating PCA visualization...")
    pca = PCA(n_components=2)
    emb1_pca = pca.fit_transform(emb1_np)
    emb2_pca = pca.transform(emb2_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    scatter1 = axes[0].scatter(emb1_pca[:, 0], emb1_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title(f'{title1} (PCA)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(emb2_pca[:, 0], emb2_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title(f'{title2} (PCA)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Plot movement vectors
    for i in range(0, len(emb1_pca), max(1, len(emb1_pca) // 100)):  # Subsample for clarity
        axes[2].arrow(emb1_pca[i, 0], emb1_pca[i, 1],
                     emb2_pca[i, 0] - emb1_pca[i, 0],
                     emb2_pca[i, 1] - emb1_pca[i, 1],
                     alpha=0.3, head_width=0.1, color='gray', length_includes_head=True)
    axes[2].scatter(emb1_pca[:, 0], emb1_pca[:, 1], c='blue', alpha=0.3, s=10, label='Before')
    axes[2].scatter(emb2_pca[:, 0], emb2_pca[:, 1], c='red', alpha=0.3, s=10, label='After')
    axes[2].set_title('Embedding Movement (PCA)')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_pca_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE visualization
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb1_np) - 1))
    emb1_tsne = tsne.fit_transform(emb1_np)
    emb2_tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb2_np) - 1)).fit_transform(emb2_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scatter1 = axes[0].scatter(emb1_tsne[:, 0], emb1_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title(f'{title1} (t-SNE)')
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(emb2_tsne[:, 0], emb2_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title('With Fine-tuning (t-SNE)')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_tsne_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. UMAP if available
    if UMAP_AVAILABLE:
        print("Generating UMAP visualization...")
        reducer = umap.UMAP(random_state=42)
        emb1_umap = reducer.fit_transform(emb1_np)
        emb2_umap = umap.UMAP(random_state=42).fit_transform(emb2_np)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        scatter1 = axes[0].scatter(emb1_umap[:, 0], emb1_umap[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
        axes[0].set_title('Without Fine-tuning (UMAP)')
        plt.colorbar(scatter1, ax=axes[0])
        
        scatter2 = axes[1].scatter(emb2_umap[:, 0], emb2_umap[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
        axes[1].set_title('With Fine-tuning (UMAP)')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_umap_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def visualize_distance_distributions(cosine_sims, euclidean_dists, save_dir, prefix="last_layer"):
    """Visualize distributions of similarity/distance metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(cosine_sims.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Cosine Similarities')
    axes[0].axvline(cosine_sims.mean().item(), color='red', linestyle='--', label=f'Mean: {cosine_sims.mean().item():.3f}')
    axes[0].legend()
    
    axes[1].hist(euclidean_dists.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Euclidean Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Euclidean Distances')
    axes[1].axvline(euclidean_dists.mean().item(), color='red', linestyle='--', label=f'Mean: {euclidean_dists.mean().item():.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_distance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compare_layer_embeddings(emb_layers_ft, emb_layers_base, labels, save_dir):
    """Compare embeddings across all layers."""
    n_layers = len(emb_layers_ft)
    
    print(f"\n{'='*80}")
    print(f"LAYER-WISE COMPARISON ({n_layers} layers)")
    print(f"{'='*80}")
    
    layer_metrics = []
    
    for layer_idx in range(n_layers):
        print(f"\n{'*'*60}")
        print(f"Layer {layer_idx + 1}/{n_layers}")
        print(f"{'*'*60}")
        
        emb_ft = emb_layers_ft[layer_idx]
        emb_base = emb_layers_base[layer_idx]
        
        metrics = {}
        
        # Basic stats
        stats_ft = compute_basic_statistics(emb_ft, f"Layer {layer_idx} - Fine-tuned")
        stats_base = compute_basic_statistics(emb_base, f"Layer {layer_idx} - Base")
        
        # Pairwise similarities
        sim_metrics, cosine_sims, euclidean_dists = compute_pairwise_similarities(emb_base, emb_ft)
        metrics.update(sim_metrics)
        
        # Clustering metrics
        cluster_ft = compute_clustering_metrics(emb_ft, labels, f"Layer {layer_idx} - Fine-tuned")
        cluster_base = compute_clustering_metrics(emb_base, labels, f"Layer {layer_idx} - Base")
        
        metrics['silhouette_improvement'] = cluster_ft.get('silhouette_score', 0) - cluster_base.get('silhouette_score', 0)
        
        layer_metrics.append({
            'layer': layer_idx,
            **metrics
        })
    
    # Visualize layer-wise trends
    plot_layer_trends(layer_metrics, save_dir)
    
    return layer_metrics


def plot_layer_trends(layer_metrics, save_dir):
    """Plot how metrics change across layers."""
    layers = [m['layer'] for m in layer_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cosine similarity
    cosine_means = [m['cosine_similarity_mean'] for m in layer_metrics]
    axes[0, 0].plot(layers, cosine_means, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Mean Cosine Similarity')
    axes[0, 0].set_title('Cosine Similarity Across Layers')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Euclidean distance
    euclidean_means = [m['euclidean_distance_mean'] for m in layer_metrics]
    axes[0, 1].plot(layers, euclidean_means, marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Mean Euclidean Distance')
    axes[0, 1].set_title('Euclidean Distance Across Layers')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative change
    relative_changes = [m['relative_change_mean'] for m in layer_metrics]
    axes[1, 0].plot(layers, relative_changes, marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Mean Relative Change')
    axes[1, 0].set_title('Relative Change Across Layers')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Silhouette improvement
    silhouette_impr = [m.get('silhouette_improvement', 0) for m in layer_metrics]
    axes[1, 1].plot(layers, silhouette_impr, marker='o', linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Silhouette Score Improvement')
    axes[1, 1].set_title('Clustering Quality Improvement Across Layers')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'layer_wise_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_to_json(metrics, filepath):
    """Save metrics dictionary to JSON file."""
    # Convert numpy/torch types to Python native types
    def convert_value(v):
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif torch.is_tensor(v):
            return v.item() if v.numel() == 1 else v.tolist()
        return v
    
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_serializable[key] = {k: convert_value(v) for k, v in value.items()}
        else:
            metrics_serializable[key] = convert_value(value)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\nMetrics saved to: {filepath}")


def compare_single_layer_embeddings(emb_ft, emb_base, labels, save_dir, prefix="last_layer"):
    """Comprehensive comparison of single layer embeddings."""
    print(f"\n{'#'*80}")
    print(f"COMPREHENSIVE COMPARISON - {prefix.upper()}")
    print(f"{'#'*80}")
    
    all_metrics = {}
    
    # 1. Basic statistics
    stats_ft = compute_basic_statistics(emb_ft, "Fine-tuned")
    stats_base = compute_basic_statistics(emb_base, "Base Model")
    all_metrics['stats_finetuned'] = stats_ft
    all_metrics['stats_base'] = stats_base
    
    # 2. Pairwise similarities
    sim_metrics, cosine_sims, euclidean_dists = compute_pairwise_similarities(emb_base, emb_ft)
    all_metrics['pairwise_similarities'] = sim_metrics
    
    # 3. Distribution divergence
    div_metrics = compute_distribution_divergence(emb_base, emb_ft)
    all_metrics['distribution_divergence'] = div_metrics
    
    # 4. Clustering metrics
    cluster_ft = compute_clustering_metrics(emb_ft, labels, "Fine-tuned")
    cluster_base = compute_clustering_metrics(emb_base, labels, "Base Model")
    all_metrics['clustering_finetuned'] = cluster_ft
    all_metrics['clustering_base'] = cluster_base
    
    # 5. Classification performance
    class_ft = evaluate_classification(emb_ft, labels, "Fine-tuned")
    class_base = evaluate_classification(emb_base, labels, "Base Model")
    all_metrics['classification_finetuned'] = class_ft
    all_metrics['classification_base'] = class_base
    
    # 6. Structural properties
    struct_ft, eigen_ft = compute_structural_properties(emb_ft, "Fine-tuned")
    struct_base, eigen_base = compute_structural_properties(emb_base, "Base Model")
    all_metrics['structural_finetuned'] = struct_ft
    all_metrics['structural_base'] = struct_base
    
    # 7. Visualizations
    print("\nGenerating visualizations...")
    visualize_embeddings(emb_base, emb_ft, labels, save_dir, prefix)
    visualize_distance_distributions(cosine_sims, euclidean_dists, save_dir, prefix)
    
    # 8. Save metrics
    save_metrics_to_json(all_metrics, os.path.join(save_dir, f'{prefix}_comparison_metrics.json'))
    
    return all_metrics


if __name__ == '__main__':
    cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
    
    # Load labels
    print("Loading dataset labels...")
    from dataset.dataset_loader import load_dataset
    data = load_dataset(cfg)
    data.to('cpu')
    labels = data.y.numpy()
    
    save_dir = f"artifacts/embedding_comparison/{cfg.llm.model_name}_{cfg.dataset.name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"STARTING EMBEDDING COMPARISON")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Model: {cfg.llm.model_name}")
    print(f"Number of samples: {len(labels)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"{'='*80}")
    
    # ========================================================================
    # PART 1: Compare Last Layer Embeddings
    # ========================================================================
    print("\n\n" + "="*80)
    print("PART 1: LAST LAYER EMBEDDINGS COMPARISON")
    print("="*80)
    
    # Load fine-tuned embeddings
    ftp = f'artifacts/cache/{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_lora_pool-mean.pt'
    if os.path.exists(ftp):
        finetuned_embeddings = torch.load(ftp)['embeddings']
        print(f"✓ Loaded fine-tuned embeddings: {finetuned_embeddings.shape}")
    else:
        print(f"✗ Fine-tuned embeddings not found at: {ftp}")
        finetuned_embeddings = None
    
    # Load base model embeddings
    wftp = f'artifacts/cache/{cfg.llm.model_name}_{cfg.dataset.name}_seqcls_pool-mean_based_model.pt'
    if os.path.exists(wftp):
        base_embeddings = torch.load(wftp)['embeddings']
        print(f"✓ Loaded base embeddings: {base_embeddings.shape}")
    else:
        print(f"✗ Base embeddings not found at: {wftp}")
        base_embeddings = None
    
    if finetuned_embeddings is not None and base_embeddings is not None:
        last_layer_metrics = compare_single_layer_embeddings(
            finetuned_embeddings, base_embeddings, labels, save_dir, prefix="last_layer"
        )
    
    # ========================================================================
    # PART 2: Compare All Layer Embeddings
    # ========================================================================
    print("\n\n" + "="*80)
    print("PART 2: ALL LAYERS EMBEDDINGS COMPARISON")
    print("="*80)
    
    # Load fine-tuned all layers
    ftp_all = f'artifacts/engine_cache/{cfg.llm.model_name}_{cfg.dataset.name}_engine_pool-mean.pt'
    if os.path.exists(ftp_all):
        finetuned_all_layers = torch.load(ftp_all)['layer_embeddings']
        print(f"✓ Loaded fine-tuned all layers: {len(finetuned_all_layers)} layers, shape: {finetuned_all_layers[0].shape}")
    else:
        print(f"✗ Fine-tuned all layers not found at: {ftp_all}")
        finetuned_all_layers = None
    
    # Load base all layers
    wftp_all = f'artifacts/engine_cache/{cfg.llm.model_name}_{cfg.dataset.name}_engine_pool-mean_based_model.pt'
    if os.path.exists(wftp_all):
        base_all_layers = torch.load(wftp_all)['layer_embeddings']
        print(f"✓ Loaded base all layers: {len(base_all_layers)} layers, shape: {base_all_layers[0].shape}")
    else:
        print(f"✗ Base all layers not found at: {wftp_all}")
        base_all_layers = None
    
    if finetuned_all_layers is not None and base_all_layers is not None:
        layer_metrics = compare_layer_embeddings(
            finetuned_all_layers, base_all_layers, labels, save_dir
        )
        save_metrics_to_json(
            {'layer_wise_metrics': layer_metrics},
            os.path.join(save_dir, 'all_layers_comparison_metrics.json')
        )
    
    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE!")
    print(f"All results saved to: {save_dir}")
    print(f"{'='*80}")



