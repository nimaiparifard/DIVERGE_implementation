import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")
import os
from common import GNNEncoder, load_graph_dataset_for_tape, set_seed

def extract_gnn_embedding(data, features, model, device):
    """
    Extract node embeddings from a trained GNN Encoder model on the given dataset.
    
    Args:
        data: PyG Data object with edge_index, y, masks, etc. (or tuple from load_graph_dataset_for_tape)
        features: Node feature tensor (num_nodes, feature_dim) or dict with 'embeddings' key
        model: Trained GNNEncoder model
        device: torch device (cuda or cpu) or str
    
    Returns:
        embeddings: Node embeddings from the second-to-last layer (num_nodes, hidden_dim)
        labels: Node labels (num_nodes,)
    """
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(data, tuple):
        data = data[0]  # Extract the graph data object
    
    # Handle dict with embeddings
    if isinstance(features, dict):
        if 'embeddings' in features:
            features = features['embeddings']
        else:
            raise ValueError("features dict must contain 'embeddings' key")
    
    # Handle device as string
    if isinstance(device, str):
        device = torch.device(device)
    
    # Ensure data is on the correct device
    if hasattr(data, 'to'):
        data = data.to(device)
    
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = features.to(device) if isinstance(features, torch.Tensor) else torch.tensor(features).to(device)
        edge_index = data.edge_index.to(device)
        
        # Forward pass through all layers except the last one to get embeddings
        for i, graph_conv in enumerate(model.conv_layers[:-1]):
            if model.residual_conn and i > 0:
                x = graph_conv(x, edge_index) + x
            else:
                x = graph_conv(x, edge_index)
            
            if model.batch_norm:
                x = model.bns[i](x)
            x = model.act(x)
            x = F.dropout(x, p=model.dropout, training=False)
        
        embeddings = x.cpu().numpy()
        labels = data.y.cpu().numpy()
    
    return embeddings, labels

def visualize_gnn_embedding(data, features, model, device, dataset_name, save_dir="../../results/TAPE/gnn_embeddings", show=False):
    """
    Visualize node embeddings from a trained GNN Encoder model using t-SNE, UMAP and PCA.
    
    Args:
        data: PyG Data object with edge_index, y, masks, etc. (or tuple from load_graph_dataset_for_tape)
        features: Node feature tensor (num_nodes, feature_dim) or dict with 'embeddings' key
        model: Trained GNNEncoder model
        device: torch device (cuda or cpu) or str
        dataset_name: Name of the dataset for saving plots
        save_dir: Directory to save visualization plots
    """
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(data, tuple):
        data = data[0]  # Extract the graph data object
    
    # Handle device as string
    if isinstance(device, str):
        device = torch.device(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract embeddings
    print("Extracting node embeddings...")
    embeddings, labels = extract_gnn_embedding(data, features, model, device)
    
    # Get masks for different splits
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    # Create splits indicator for visualization
    splits = np.array(['train'] * len(labels), dtype=object)
    splits[val_mask] = 'val'
    splits[test_mask] = 'test'
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Apply dimensionality reduction techniques
    print("\nApplying dimensionality reduction...")
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # UMAP
    if UMAP_AVAILABLE:
        print("Computing UMAP...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_umap = umap_reducer.fit_transform(embeddings)
        methods = [
            ('PCA', embeddings_pca),
            ('t-SNE', embeddings_tsne),
            ('UMAP', embeddings_umap)
        ]
        fig_cols = 3
    else:
        print("Skipping UMAP (not installed)...")
        methods = [
            ('PCA', embeddings_pca),
            ('t-SNE', embeddings_tsne)
        ]
        fig_cols = 2
    
    # Visualization
    print("\nCreating visualizations...")
    
    # Create figure with subplots (2 or 3 methods x 2 colorings: by label and by split)
    fig, axes = plt.subplots(2, fig_cols, figsize=(6*fig_cols, 12))
    
    # Row 1: Color by label
    for idx, (method_name, coords) in enumerate(methods):
        ax = axes[0, idx]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=labels, cmap='tab10', 
                           s=20, alpha=0.6, edgecolors='none')
        ax.set_title(f'{method_name} (colored by label)', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Label', fontsize=10)
    
    # Row 2: Color by split (train/val/test)
    split_colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    for idx, (method_name, coords) in enumerate(methods):
        ax = axes[1, idx]
        for split_name, color in split_colors.items():
            mask = splits == split_name
            ax.scatter(coords[mask, 0], coords[mask, 1], 
                      c=color, label=split_name,
                      s=20, alpha=0.6, edgecolors='none')
        ax.set_title(f'{method_name} (colored by split)', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{dataset_name}_embeddings.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    if show:
        plt.show()
    plt.close()
    
    # Print explained variance for PCA
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

def visualize_gnn_embedding_list(cfg, data, features_list, model_list, title_list, device, dataset_name, n=2000, save_dir="results/gnn_embeddings", show=False):
    """
        plot the bnn embeddings with different feature extraction methods and models. your visulization should have diffrent colors for different classes and different train/val/test splits.
        labels: dataset.y
        train_mask: dataset.train_mask
        val_mask: dataset.val_mask
        test_mask: dataset.test_mask
    1. data: the graph data object
    2. features_list: list of features to be used for embedding extraction
    3. model_list: list of trained GNN models
    4. title_list: list of titles for each subplot
    5. device: torch device
    6. dataset_name: name of the dataset
    7. n: number of nodes to visualize
    8. save_dir: directory to save the plots
    bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving the plots.
    """
    print("\nGenerating GNN embedding visualizations for multiple methods...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(data, tuple):
        data = data[0]
    
    # Handle device as string
    if isinstance(device, str):
        device = torch.device(device)
    
    # Get masks and labels
    labels = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    # Create splits indicator
    splits = np.array(['train'] * len(labels), dtype=object)
    splits[val_mask] = 'val'
    splits[test_mask] = 'test'
    
    n_methods = len(features_list)
    
    # Extract embeddings for all methods
    embeddings_list = []
    for idx, (features, model, title) in enumerate(zip(features_list, model_list, title_list)):
        print(f"Extracting embeddings for {title}...")
        embeddings, _ = extract_gnn_embedding(data, features, model, device)
        embeddings_list.append(embeddings)
    
    # Subsample if needed
    if len(labels) > n:
        indices = np.random.choice(len(labels), n, replace=False)
        labels_subset = labels[indices]
        splits_subset = splits[indices]
        embeddings_list = [emb[indices] for emb in embeddings_list]
    else:
        labels_subset = labels
        splits_subset = splits
    
    # Apply t-SNE to all embeddings
    print("\nApplying t-SNE dimensionality reduction...")
    embeddings_tsne_list = []
    for idx, (emb, title) in enumerate(zip(embeddings_list, title_list)):
        print(f"Computing t-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
        emb_tsne = tsne.fit_transform(emb)
        embeddings_tsne_list.append(emb_tsne)
    
    # Create visualizations
    print("\nCreating comparative visualizations...")
    
    # Figure 1: Colored by class label
    fig1, axes1 = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes1 = [axes1]
    
    for idx, (coords, title) in enumerate(zip(embeddings_tsne_list, title_list)):
        scatter = axes1[idx].scatter(coords[:, 0], coords[:, 1], 
                                    c=labels_subset, cmap='tab10', 
                                    s=30, alpha=0.6, edgecolors='none')
        axes1[idx].set_title(f'{title}\n(colored by class)', fontsize=12, fontweight='bold')
        axes1[idx].set_xlabel('t-SNE Dimension 1', fontsize=10)
        axes1[idx].set_ylabel('t-SNE Dimension 2', fontsize=10)
        axes1[idx].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes1[idx])
        cbar.set_label('Class', fontsize=9)
    
    plt.tight_layout()
    filename1 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_gnn_embeddings_by_class.png'
    filepath1 = os.path.join(save_dir, filename1)
    try:
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        print(f"Class-colored visualization saved as '{filepath1}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    # Figure 2: Colored by split (train/val/test)
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes2 = [axes2]
    
    split_colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    for idx, (coords, title) in enumerate(zip(embeddings_tsne_list, title_list)):
        for split_name, color in split_colors.items():
            mask = splits_subset == split_name
            if mask.sum() > 0:  # Only plot if there are samples
                axes2[idx].scatter(coords[mask, 0], coords[mask, 1], 
                                  c=color, label=split_name,
                                  s=30, alpha=0.6, edgecolors='none')
        axes2[idx].set_title(f'{title}\n(colored by split)', fontsize=12, fontweight='bold')
        axes2[idx].set_xlabel('t-SNE Dimension 1', fontsize=10)
        axes2[idx].set_ylabel('t-SNE Dimension 2', fontsize=10)
        axes2[idx].legend(fontsize=9)
        axes2[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename2 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_gnn_embeddings_by_split.png'
    filepath2 = os.path.join(save_dir, filename2)
    try:
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"Split-colored visualization saved as '{filepath2}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    # Figure 3: Combined view (class on top, split on bottom)
    fig3, axes3 = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10))
    if n_methods == 1:
        axes3 = axes3.reshape(-1, 1)
    
    # Top row: colored by class
    for idx, (coords, title) in enumerate(zip(embeddings_tsne_list, title_list)):
        scatter = axes3[0, idx].scatter(coords[:, 0], coords[:, 1], 
                                       c=labels_subset, cmap='tab10', 
                                       s=30, alpha=0.6, edgecolors='none')
        axes3[0, idx].set_title(f'{title} - Class Labels', fontsize=12, fontweight='bold')
        axes3[0, idx].set_xlabel('t-SNE Dim 1', fontsize=10)
        axes3[0, idx].set_ylabel('t-SNE Dim 2', fontsize=10)
        axes3[0, idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes3[0, idx])
    
    # Bottom row: colored by split
    for idx, (coords, title) in enumerate(zip(embeddings_tsne_list, title_list)):
        for split_name, color in split_colors.items():
            mask = splits_subset == split_name
            if mask.sum() > 0:
                axes3[1, idx].scatter(coords[mask, 0], coords[mask, 1], 
                                     c=color, label=split_name,
                                     s=30, alpha=0.6, edgecolors='none')
        axes3[1, idx].set_title(f'{title} - Train/Val/Test', fontsize=12, fontweight='bold')
        axes3[1, idx].set_xlabel('t-SNE Dim 1', fontsize=10)
        axes3[1, idx].set_ylabel('t-SNE Dim 2', fontsize=10)
        axes3[1, idx].legend(fontsize=9)
        axes3[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename3 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_gnn_embeddings_combined.png'
    filepath3 = os.path.join(save_dir, filename3)
    try:
        plt.savefig(filepath3, dpi=300, bbox_inches='tight')
        print(f"Combined visualization saved as '{filepath3}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    print("\nGNN embedding visualizations complete!")


if __name__ == '__main__':
    # Example usage
    from gnns.gnn_mtrainer import set_mgnn_cfg, GNNTrainer
    
    # Set configuration
    dataset_name = 'cora'
    cfg = set_mgnn_cfg(dataset_name)
    
    print("=" * 80)
    print(f"Loading dataset: {dataset_name}")
    print("=" * 80)
    
    # Set seed
    set_seed(cfg.seed)
    
    # Load dataset
    device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
    data, num_classes, _ = load_graph_dataset_for_tape(cfg.dataset, device, re_split=cfg.re_split, path_prefix='../../../')
    
    # Load or create features (using Node2Vec embeddings as example)
    features_path = f"../../../datasets/sbert/{dataset_name}.pt"
    if os.path.exists(features_path):
        features = torch.load(features_path, map_location=device, weights_only=False).to(device).type(torch.float)
        print(f"Loaded features from: {features_path}")
    else:
        print(f"Features not found at {features_path}, using random features")
        features = torch.randn(data.y.shape[0], 768).to(device)
    
    # Initialize model
    model = GNNEncoder(
        input_dim=features.shape[1],
        hidden_dim=cfg.hidden_dim,
        output_dim=num_classes,
        n_layers=cfg.num_layers,
        gnn_type=cfg.gnn_model_name,
        dropout=cfg.dropout,
        batch_norm=cfg.batch_norm,
    ).to(device)
    
    # Load trained model if available
    model_path = f"../../results/TAPE/{dataset_name}/{cfg.gnn_model_name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from: {model_path}")
    else:
        print(f"No trained model found at {model_path}, using untrained model")
        print("Training a new model...")
        trainer = GNNTrainer(cfg, features)
        results, _ = trainer.train()
        print(f"Training complete. Test Acc: {results['test_acc']:.4f}, Test F1: {results['test_f1']:.4f}")
        model = trainer.model
    
    # Visualize embeddings
    print("\n" + "=" * 80)
    print("Visualizing GNN embeddings")
    print("=" * 80)
    visualize_gnn_embedding(data, features, model, device, dataset_name)