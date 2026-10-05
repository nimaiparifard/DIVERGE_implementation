"""
    Comprehensive visualization functions for embeddings using various dimensionality reduction techniques and evaluation metrics.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")


def visualize_tsne_list(cfg, dataset, embedding_list, labels, title_list, n=2000, save_dir="results/tsne_embeddings/", show=False, reporter=None):
    """
        Visualize a list of embeddings using t-SNE beside each other with subplots.
        Args:
            cfg: Configuration object.
            dataset: Dataset object.
            embedding_list: List of embeddings to visualize.
            labels: Corresponding labels for the embeddings.
            title_list: List of titles for each subplot
            n: Number of samples to visualize.
            save_dir: Directory to save the visualizations.
        bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving the plots.
    """
    print("\nGenerating t-SNE visualizations for multiple embeddings...")
    os.makedirs(save_dir, exist_ok=True)
    
    n_methods = len(embedding_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    # Handle case where there's only one method
    if n_methods == 1:
        axes = [axes]

    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    for idx, (emb, title) in enumerate(zip(embedding_list, title_list)):
        # Subsample if needed
        emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        if len(emb_np) > n:
            indices = np.random.choice(len(emb_np), n, replace=False)
            emb_np = emb_np[indices]
            labels_subset = labels_np[indices]
        else:
            labels_subset = labels_np
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_np) - 1))
        emb_tsne = tsne.fit_transform(emb_np)
        
        # Plot
        scatter = axes[idx].scatter(emb_tsne[:, 0], emb_tsne[:, 1], 
                                   c=labels_subset, cmap='tab10', alpha=0.6, s=20)
        axes[idx].set_title(f'{title}\n(t-SNE)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('t-SNE Dimension 1')
        axes[idx].set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=axes[idx])
        axes[idx].grid(alpha=0.3)
        
        # Report t-SNE info using reporter
        if reporter is not None:
            report_text = f"""
t-SNE Visualization:
  • Embedding Dimension: {emb_np.shape[1]}
  • Samples Visualized: {len(emb_np)}
  • Perplexity: {min(30, len(emb_np) - 1)}
  • Number of Classes: {len(np.unique(labels_subset))}
"""
            reporter.report(f"t-SNE Visualization - {title}", report_text)
    
    plt.tight_layout()
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_tsne_comparison.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()

def visualize_pca_list(cfg, dataset, embedding_list, labels,title_list, n=2000, save_dir="results/pca_embeddings/", show=False, reporter=None):
    """
        Visualize a list of embeddings using PCA beside each other with axes.
        Args:
            cfg: Configuration object.
            dataset: Dataset object.
            embedding_list: List of embeddings to visualize.
            labels: Corresponding labels for the embeddings.
            title_list: List of titles for each subplot
            n: Number of samples to visualize.
            save_dir: Directory to save the visualizations.
        bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving
    """
    print("\nGenerating PCA visualizations for multiple embeddings...")
    os.makedirs(save_dir, exist_ok=True)
    
    n_methods = len(embedding_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    # Handle case where there's only one method
    if n_methods == 1:
        axes = [axes]
    
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    for idx, (emb, title) in enumerate(zip(embedding_list, title_list)):
        # Subsample if needed
        emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        if len(emb_np) > n:
            indices = np.random.choice(len(emb_np), n, replace=False)
            emb_np = emb_np[indices]
            labels_subset = labels_np[indices]
        else:
            labels_subset = labels_np
        
        # Apply PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb_np)
        
        # Plot
        scatter = axes[idx].scatter(emb_pca[:, 0], emb_pca[:, 1], 
                                   c=labels_subset, cmap='tab10', alpha=0.6, s=20)
        axes[idx].set_title(f'{title}\n(PCA)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.colorbar(scatter, ax=axes[idx])
        axes[idx].grid(alpha=0.3)
        
        # Report PCA info using reporter
        if reporter is not None:
            total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
            report_text = f"""
PCA Visualization:
  • Embedding Dimension: {emb_np.shape[1]}
  • Samples Visualized: {len(emb_np)}
  • PC1 Explained Variance: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]:.2%})
  • PC2 Explained Variance: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]:.2%})
  • Total Variance Explained (PC1+PC2): {total_variance:.4f} ({total_variance:.2%})
  • Number of Classes: {len(np.unique(labels_subset))}
"""
            reporter.report(f"PCA Visualization - {title}", report_text)
    
    plt.tight_layout()
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_pca_comparison.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()

def visualize_umap_list(cfg, dataset, embedding_list, labels, title_list, n=2000, save_dir="results/umap_embeddings/", show=False, reporter=None):
    """
        Visualize a list of embeddings using UMAP beside each other with axes.
        Args:
            cfg: Configuration object.
            dataset: Dataset object.
            embedding_list: List of embeddings to visualize.
            labels: Corresponding labels for the embeddings.
            title_list: List of titles for each subplot
            n: Number of samples to visualize.
            save_dir: Directory to save the visualizations.
        bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving the plots.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping UMAP visualization.")
        return
    
    print("\nGenerating UMAP visualizations for multiple embeddings...")
    os.makedirs(save_dir, exist_ok=True)
    
    n_methods = len(embedding_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    # Handle case where there's only one method
    if n_methods == 1:
        axes = [axes]
    
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    for idx, (emb, title) in enumerate(zip(embedding_list, title_list)):
        # Subsample if needed
        emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        if len(emb_np) > n:
            indices = np.random.choice(len(emb_np), n, replace=False)
            emb_np = emb_np[indices]
            labels_subset = labels_np[indices]
        else:
            labels_subset = labels_np
        
        # Apply UMAP
        reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(emb_np) - 1))
        emb_umap = reducer.fit_transform(emb_np)
        
        # Plot
        scatter = axes[idx].scatter(emb_umap[:, 0], emb_umap[:, 1], 
                                   c=labels_subset, cmap='tab10', alpha=0.6, s=20)
        axes[idx].set_title(f'{title}\n(UMAP)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('UMAP Dimension 1')
        axes[idx].set_ylabel('UMAP Dimension 2')
        plt.colorbar(scatter, ax=axes[idx])
        axes[idx].grid(alpha=0.3)
        
        # Report UMAP info using reporter
        if reporter is not None:
            report_text = f"""
UMAP Visualization:
  • Embedding Dimension: {emb_np.shape[1]}
  • Samples Visualized: {len(emb_np)}
  • N Neighbors: {min(15, len(emb_np) - 1)}
  • Number of Classes: {len(np.unique(labels_subset))}
"""
            reporter.report(f"UMAP Visualization - {title}", report_text)
    
    plt.tight_layout()
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_umap_comparison.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"UMAP visualization saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()

def visualize_clustering_metrics_list(cfg, dataset, embedding_list, labels,title_list, save_dir="results/clustering/", show=False, reporter=None):
    """
        Visualize clustering metrics for a list of embeddings.
        Args:
            cfg: Configuration object.
            dataset: Dataset object.
            embedding_list: List of embeddings to evaluate.
            labels: Corresponding labels for the embeddings.
            title_list: List of titles for each subplot
            save_dir: Directory to save the visualizations.
        bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving the plots.
    """
    print("\nComputing clustering metrics for multiple embeddings...")
    os.makedirs(save_dir, exist_ok=True)
    
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    if len(np.unique(labels_np)) < 2:
        print("Warning: Less than 2 unique labels, skipping clustering metrics")
        return
    
    # Compute metrics for each embedding
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    
    for emb, title in zip(embedding_list, title_list):
        emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        
        # Compute clustering metrics
        silhouette = silhouette_score(emb_np, labels_np, sample_size=min(5000, len(labels_np)))
        davies_bouldin = davies_bouldin_score(emb_np, labels_np)
        calinski = calinski_harabasz_score(emb_np, labels_np)
        
        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        calinski_harabasz_scores.append(calinski)
        
        print(f"{title}: Silhouette={silhouette:.4f}, Davies-Bouldin={davies_bouldin:.4f}, Calinski-Harabasz={calinski:.2f}")
        
        # Report clustering metrics using reporter
        if reporter is not None:
            report_text = f"""
                    Clustering Metrics:
                    • Silhouette Score: {silhouette:.4f} (Higher is better, range: -1 to 1)
                    • Davies-Bouldin Score: {davies_bouldin:.4f} (Lower is better)
                    • Calinski-Harabasz Score: {calinski:.2f} (Higher is better)
                    • Embedding Dimension: {emb_np.shape[1]}
                    • Number of Samples: {len(emb_np)}
                    • Number of Classes: {len(np.unique(labels_np))}
                    """
            reporter.report(f"Clustering Metrics - {title}", report_text)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(title_list))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    
    # Plot 1: Silhouette Score (higher is better)
    bars1 = axes[0].bar(x, silhouette_scores, color=[colors[i % len(colors)] for i in range(len(title_list))], 
                       edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Method', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(title_list, rotation=45, ha='right')
    axes[0].grid(alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Davies-Bouldin Score (lower is better)
    bars2 = axes[1].bar(x, davies_bouldin_scores, color=[colors[i % len(colors)] for i in range(len(title_list))], 
                       edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Method', fontsize=11)
    axes[1].set_ylabel('Davies-Bouldin Score', fontsize=11)
    axes[1].set_title('Davies-Bouldin Score\n(Lower is Better)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(title_list, rotation=45, ha='right')
    axes[1].grid(alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Calinski-Harabasz Score (higher is better)
    bars3 = axes[2].bar(x, calinski_harabasz_scores, color=[colors[i % len(colors)] for i in range(len(title_list))], 
                       edgecolor='black', alpha=0.8)
    axes[2].set_xlabel('Method', fontsize=11)
    axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=11)
    axes[2].set_title('Calinski-Harabasz Score\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(title_list, rotation=45, ha='right')
    axes[2].grid(alpha=0.3, axis='y')
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_clustering_metrics.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Clustering metrics visualization saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()

def visualize_classification_metrics_list(cfg, dataset, embedding_list, labels,title_list, save_dir="results/classification/", show=False, reporter=None):
    """
        Visualize classification metrics for a list of embeddings.
        Args:
            cfg: Configuration object.
            dataset: Dataset object.
            embedding_list: List of embeddings to evaluate.
            labels: Corresponding labels for the embeddings.
            title_list: List of titles for each subplot
            save_dir: Directory to save the visualizations.
        bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving
    """
    print("\nComputing classification metrics for multiple embeddings...")
    os.makedirs(save_dir, exist_ok=True)
    
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    if len(np.unique(labels_np)) < 2:
        print("Warning: Less than 2 unique labels, skipping classification metrics")
        return
    
    # Compute metrics for each embedding
    knn_accuracies = []
    linear_accuracies = []
    
    for emb, title in zip(embedding_list, title_list):
        emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            emb_np, labels_np, test_size=0.3, random_state=42, stratify=labels_np
        )
        
        # KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_acc = knn.score(X_test, y_test)
        
        # Linear classifier
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_acc = lr.score(X_test, y_test)
        
        knn_accuracies.append(knn_acc)
        linear_accuracies.append(lr_acc)
        
        print(f"{title}: KNN={knn_acc:.4f}, Linear={lr_acc:.4f}")
        
        # Report classification metrics using reporter
        if reporter is not None:
            report_text = f"""
                        Classification Metrics:
                        • KNN Accuracy (k=5): {knn_acc:.4f} ({knn_acc:.2%})
                        • Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc:.2%})
                        • Embedding Dimension: {emb_np.shape[1]}
                        • Training Samples: {len(X_train)}
                        • Test Samples: {len(X_test)}
                        • Number of Classes: {len(np.unique(labels_np))}
                        """
            reporter.report(f"Classification Metrics - {title}", report_text)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(title_list))
    width = 0.35
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    
    # Plot 1: Bar chart comparing KNN and Linear
    bars1 = axes[0].bar(x - width/2, knn_accuracies, width, label='KNN (k=5)', 
                       color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, linear_accuracies, width, label='Logistic Regression',
                       color='coral', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Method', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Classification Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(title_list, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Individual method comparison
    for i, (knn_acc, lr_acc, title) in enumerate(zip(knn_accuracies, linear_accuracies, title_list)):
        axes[1].scatter([i, i], [knn_acc, lr_acc], s=100, 
                       color=[colors[i % len(colors)], colors[i % len(colors)]], 
                       alpha=0.7, edgecolors='black', linewidths=2)
        axes[1].plot([i, i], [knn_acc, lr_acc], color='gray', linewidth=1.5, alpha=0.5)
    
    axes[1].set_xlabel('Method', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Classification Performance per Method', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(title_list, rotation=45, ha='right')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.0])
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='KNN',
                             markerfacecolor='steelblue', markersize=10, markeredgecolor='black'),
                      Line2D([0], [0], marker='o', color='w', label='Linear',
                             markerfacecolor='coral', markersize=10, markeredgecolor='black')]
    axes[1].legend(handles=legend_elements)
    
    plt.tight_layout()
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_classification_metrics.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Classification metrics visualization saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()