from visualize.visualize_gnn_embedding import extract_gnn_embedding
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

def get_gnn_embedding_mistakes(dataset, gnn_model, features, output_mask='test'):
    """
        gnn_model: trained gnn model GNNEncoder from common
        features: node features embeddings
        dataset: graph dataset with data.y, data.train_mask, data.val_mask, data.test_mask from load_graph_dataset_for_tape
        output_mask: 'test' or 'train' or 'val' to specify which mask to use for mistakes return mistakes
        also output mask can be 'all' to report all nodes mistakes return train_mistakes, val_mistakes, test_mistakes, all_mistakes
    Returns: 
        - If output_mask='train': list of node indices where gnn_model made mistakes on train set
        - If output_mask='val': list of node indices where gnn_model made mistakes on validation set
        - If output_mask='test': list of node indices where gnn_model made mistakes on test set
        - If output_mask='all': tuple of (train_mistakes, val_mistakes, test_mistakes, all_mistakes) lists
    """
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    # Get device from model
    device = next(gnn_model.parameters()).device
    
    # Set model to evaluation mode
    gnn_model.eval()
    with torch.no_grad():
        # Move data to correct device
        x = features.to(device) if isinstance(features, torch.Tensor) else torch.tensor(features).to(device)
        edge_index = dataset.edge_index.to(device)
        
        # Forward pass to get predictions
        logits = gnn_model(x, edge_index)
        pred = logits.argmax(dim=1)
        
        # Get masks and ground truth labels
        train_mask = dataset.train_mask.cpu().numpy()
        val_mask = dataset.val_mask.cpu().numpy()
        test_mask = dataset.test_mask.cpu().numpy()
        y_true = dataset.y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        # Find mistakes for each split: nodes where prediction != ground truth
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        train_mistakes = train_indices[y_true[train_indices] != y_pred[train_indices]]
        val_mistakes = val_indices[y_true[val_indices] != y_pred[val_indices]]
        test_mistakes = test_indices[y_true[test_indices] != y_pred[test_indices]]
        
        # Find all mistakes across all nodes
        all_indices = np.arange(len(y_true))
        all_mistakes = all_indices[y_true != y_pred]
        
    # Return based on output_mask parameter
    if output_mask == 'train':
        return train_mistakes.tolist()
    elif output_mask == 'val':
        return val_mistakes.tolist()
    elif output_mask == 'test':
        return test_mistakes.tolist()
    elif output_mask == 'all':
        return train_mistakes.tolist(), val_mistakes.tolist(), test_mistakes.tolist(), all_mistakes.tolist()
    else:
        raise ValueError(f"output_mask must be 'train', 'val', 'test', or 'all', got '{output_mask}'")

def get_embeddings_mistakes(dataset, ensemble_predictions):
    """
        dataset: graph dataset with data.y, data.test_mask from load_graph_dataset_for_tape
        ensemble_predictions: predictions from ensemble model (tensor or numpy array)
    Returns: list of node indices where ensemble made mistakes on test set
    """
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    # Convert ensemble predictions to numpy
    if isinstance(ensemble_predictions, torch.Tensor):
        ensemble_predictions = ensemble_predictions.cpu().numpy()
    
    # Get test mask and ground truth labels
    test_mask = dataset.test_mask.cpu().numpy()
    y_true = dataset.y.cpu().numpy()
    
    # Find mistakes: nodes in test set where prediction != ground truth
    test_indices = np.where(test_mask)[0]
    mistakes = test_indices[y_true[test_indices] != ensemble_predictions[test_indices]]
    
    return mistakes.tolist()


def visualize_init_weights_embedding_mistakes(dataset, gnn_model, features, ensemble_predictions=None, save_dir="../../results/TAPE/gnn_mistakes", show=False):
    """
        first got gnn mistake from get_gnn_embedding_mistakes
        then visualize those mistakes with t-SNE, UMAP, PCA that misktae have diffirent color
        if ensemble_predictions provided, also show ensemble mistakes
    """
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get mistake indices
    print("Finding GNN mistakes on test set...")
    mistake_indices = get_gnn_embedding_mistakes(dataset, gnn_model, features)
    print(f"Found {len(mistake_indices)} GNN mistakes out of {dataset.test_mask.sum()} test nodes")
    
    # Get ensemble mistakes if provided
    ensemble_mistake_indices = None
    if ensemble_predictions is not None:
        print("\nFinding Ensemble mistakes on test set...")
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
        print(f"Found {len(ensemble_mistake_indices)} Ensemble mistakes out of {dataset.test_mask.sum()} test nodes")
    
    # Prepare features for visualization
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features
    
    labels = dataset.y.cpu().numpy()
    test_mask = dataset.test_mask.cpu().numpy()
    
    # Create mistake masks
    mistake_mask = np.zeros(len(labels), dtype=bool)
    mistake_mask[mistake_indices] = True
    
    ensemble_mistake_mask = np.zeros(len(labels), dtype=bool)
    if ensemble_mistake_indices is not None:
        ensemble_mistake_mask[ensemble_mistake_indices] = True
    
    # Only visualize test nodes
    test_indices = np.where(test_mask)[0]
    test_features = features_np[test_indices]
    test_labels = labels[test_indices]
    test_mistakes = mistake_mask[test_indices]
    test_ensemble_mistakes = ensemble_mistake_mask[test_indices]
    
    print(f"Visualizing {len(test_indices)} test nodes (input features)")
    
    # Apply dimensionality reduction
    print("\nApplying dimensionality reduction...")
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(test_features)
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_features)//4), max_iter=1000)
    features_tsne = tsne.fit_transform(test_features)
    
    # UMAP
    methods = []
    if UMAP_AVAILABLE:
        print("Computing UMAP...")
        n_neighbors = min(15, len(test_features) - 1)
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        features_umap = umap_reducer.fit_transform(test_features)
        methods = [
            ('PCA', features_pca),
            ('t-SNE', features_tsne),
            ('UMAP', features_umap)
        ]
        fig_cols = 3
    else:
        methods = [
            ('PCA', features_pca),
            ('t-SNE', features_tsne)
        ]
        fig_cols = 2
    
    # Visualization
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(2, fig_cols, figsize=(6*fig_cols, 12))
    
    # Row 1: Color by label with GNN mistakes
    for idx, (method_name, coords) in enumerate(methods):
        ax = axes[0, idx]
        # Plot correct predictions
        correct_mask = ~test_mistakes
        scatter1 = ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                            c=test_labels[correct_mask], cmap='tab10',
                            s=20, alpha=0.6, edgecolors='none', label='Correct')
        # Plot GNN mistakes with red marker
        if test_mistakes.sum() > 0:
            ax.scatter(coords[test_mistakes, 0], coords[test_mistakes, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        # Plot ensemble mistakes with purple marker if available
        if ensemble_predictions is not None and test_ensemble_mistakes.sum() > 0:
            ax.scatter(coords[test_ensemble_mistakes, 0], coords[test_ensemble_mistakes, 1],
                      c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        ax.set_title(f'{method_name} (Initial Features, colored by label)', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        cbar = plt.colorbar(scatter1, ax=ax)
        cbar.set_label('Class Label', fontsize=10)
    
    # Row 2: Color by mistake vs correct
    for idx, (method_name, coords) in enumerate(methods):
        ax = axes[1, idx]
        # Plot correct predictions
        if ensemble_predictions is not None:
            # Both correct
            both_correct = (~test_mistakes) & (~test_ensemble_mistakes)
            ax.scatter(coords[both_correct, 0], coords[both_correct, 1],
                      c='green', s=20, alpha=0.6, edgecolors='none', label='Both Correct')
        else:
            ax.scatter(coords[~test_mistakes, 0], coords[~test_mistakes, 1],
                      c='blue', s=20, alpha=0.6, edgecolors='none', label='Correct')
        
        # Plot GNN mistakes
        if test_mistakes.sum() > 0:
            ax.scatter(coords[test_mistakes, 0], coords[test_mistakes, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        
        # Plot ensemble mistakes
        if ensemble_predictions is not None and test_ensemble_mistakes.sum() > 0:
            ax.scatter(coords[test_ensemble_mistakes, 0], coords[test_ensemble_mistakes, 1],
                      c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        
        title_suffix = ' (GNN & Ensemble)' if ensemble_predictions is not None else ''
        ax.set_title(f'{method_name} (Mistakes highlighted{title_suffix})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename_suffix = '_with_ensemble' if ensemble_predictions is not None else ''
    save_path = os.path.join(save_dir, f'init_features_mistakes{filename_suffix}.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    if show:
        plt.show()
    plt.close()
    
    return mistake_indices, ensemble_mistake_indices

def visualize_gnn_embedding_mistakes(dataset, gnn_model, features, ensemble_predictions=None, save_dir="../../results/TAPE/gnn_mistakes", show=False):
    """
        first get gnn mistake from get_gnn_embedding_mistakes
        then got gnn embedding for all nodes with extract_gnn_embedding
        then visualize those mistakes with t-SNE, UMAP, PCA that misktae have diffirent color
        if ensemble_predictions provided, also show ensemble mistakes
    """
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get device from model
    device = next(gnn_model.parameters()).device
    
    # Get mistake indices
    print("Finding GNN mistakes on test set...")
    mistake_indices = get_gnn_embedding_mistakes(dataset, gnn_model, features)
    print(f"Found {len(mistake_indices)} GNN mistakes out of {dataset.test_mask.sum()} test nodes")
    
    # Get ensemble mistakes if provided
    ensemble_mistake_indices = None
    if ensemble_predictions is not None:
        print("\nFinding Ensemble mistakes on test set...")
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
        print(f"Found {len(ensemble_mistake_indices)} Ensemble mistakes out of {dataset.test_mask.sum()} test nodes")
    
    # Extract GNN embeddings
    print("\nExtracting GNN embeddings...")
    embeddings, labels = extract_gnn_embedding(dataset, features, gnn_model, device)
    
    test_mask = dataset.test_mask.cpu().numpy()
    
    # Create mistake masks
    mistake_mask = np.zeros(len(labels), dtype=bool)
    mistake_mask[mistake_indices] = True
    
    ensemble_mistake_mask = np.zeros(len(labels), dtype=bool)
    if ensemble_mistake_indices is not None:
        ensemble_mistake_mask[ensemble_mistake_indices] = True
    
    # Only visualize test nodes
    test_indices = np.where(test_mask)[0]
    test_embeddings = embeddings[test_indices]
    test_labels = labels[test_indices]
    test_mistakes = mistake_mask[test_indices]
    test_ensemble_mistakes = ensemble_mistake_mask[test_indices]
    
    print(f"Visualizing {len(test_indices)} test nodes (GNN embeddings)")
    
    # Apply dimensionality reduction
    print("\nApplying dimensionality reduction...")
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(test_embeddings)
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_embeddings)//4), max_iter=1000)
    embeddings_tsne = tsne.fit_transform(test_embeddings)
    
    # UMAP
    methods = []
    if UMAP_AVAILABLE:
        print("Computing UMAP...")
        n_neighbors = min(15, len(test_embeddings) - 1)
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        embeddings_umap = umap_reducer.fit_transform(test_embeddings)
        methods = [
            ('PCA', embeddings_pca),
            ('t-SNE', embeddings_tsne),
            ('UMAP', embeddings_umap)
        ]
        fig_cols = 3
    else:
        methods = [
            ('PCA', embeddings_pca),
            ('t-SNE', embeddings_tsne)
        ]
        fig_cols = 2
    
    # Visualization
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(2, fig_cols, figsize=(6*fig_cols, 12))
    
    # Row 1: Color by label with GNN mistakes
    for idx, (method_name, coords) in enumerate(methods):
        ax = axes[0, idx]
        # Plot correct predictions
        correct_mask = ~test_mistakes
        scatter1 = ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                            c=test_labels[correct_mask], cmap='tab10',
                            s=20, alpha=0.6, edgecolors='none', label='Correct')
        # Plot GNN mistakes with red marker
        if test_mistakes.sum() > 0:
            ax.scatter(coords[test_mistakes, 0], coords[test_mistakes, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        # Plot ensemble mistakes with purple marker if available
        if ensemble_predictions is not None and test_ensemble_mistakes.sum() > 0:
            ax.scatter(coords[test_ensemble_mistakes, 0], coords[test_ensemble_mistakes, 1],
                      c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        ax.set_title(f'{method_name} (GNN Embeddings, colored by label)', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        cbar = plt.colorbar(scatter1, ax=ax)
        cbar.set_label('Class Label', fontsize=10)
    
    # Row 2: Color by mistake vs correct
    for idx, (method_name, coords) in enumerate(methods):
        ax = axes[1, idx]
        # Plot correct predictions
        if ensemble_predictions is not None:
            # Both correct
            both_correct = (~test_mistakes) & (~test_ensemble_mistakes)
            ax.scatter(coords[both_correct, 0], coords[both_correct, 1],
                      c='green', s=20, alpha=0.6, edgecolors='none', label='Both Correct')
        else:
            ax.scatter(coords[~test_mistakes, 0], coords[~test_mistakes, 1],
                      c='blue', s=20, alpha=0.6, edgecolors='none', label='Correct')
        
        # Plot GNN mistakes
        if test_mistakes.sum() > 0:
            ax.scatter(coords[test_mistakes, 0], coords[test_mistakes, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        
        # Plot ensemble mistakes
        if ensemble_predictions is not None and test_ensemble_mistakes.sum() > 0:
            ax.scatter(coords[test_ensemble_mistakes, 0], coords[test_ensemble_mistakes, 1],
                      c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        
        title_suffix = ' (GNN & Ensemble)' if ensemble_predictions is not None else ''
        ax.set_title(f'{method_name} (Mistakes highlighted{title_suffix})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename_suffix = '_with_ensemble' if ensemble_predictions is not None else ''
    save_path = os.path.join(save_dir, f'gnn_embeddings_mistakes{filename_suffix}.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    if show:
        plt.show()
    plt.close()
    
    # Print explained variance for PCA
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    return mistake_indices, ensemble_mistake_indices

def visualize_init_weights_embedding_mistakes_list(cfg, dataset, gnn_model_list, features_list, title_list, ensemble_predictions=None, n=2000, save_dir="results/gnn_mistakes_init_weights", show=False):
    """
        visualize_init_weights_embedding_mistakes for multiple gnn models and features with subplots
        Args:
            0. cfg: config
            1. dataset: graph dataset with data.y, data.test_mask from load_graph_dataset_for_tape
            2. gnn_model_list: list of trained gnn models GNNEncoder from
            3. features_list: list of node features embeddings
            4. ensemble_predictions: predictions from ensemble model (tensor or numpy array)
            5. n: number of nodes to visualize
            6. save_dir: directory to save the plots
        title_list: list of titles for each subplot
    bring the cfg.llm.model_name and cfg.dataset.name and cfg.peft.type for saving the plots.
    """
    print("\nGenerating initial features mistake visualizations for multiple methods...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    n_methods = len(gnn_model_list)
    
    # Get mistakes for all models
    print("Finding mistakes for all models...")
    all_mistake_indices = []
    for features, model, title in zip(features_list, gnn_model_list, title_list):
        mistakes = get_gnn_embedding_mistakes(dataset, model, features)
        all_mistake_indices.append(mistakes)
        print(f"{title}: {len(mistakes)} mistakes")
    
    # Get ensemble mistakes if provided
    ensemble_mistake_indices = None
    if ensemble_predictions is not None:
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
        print(f"Ensemble: {len(ensemble_mistake_indices)} mistakes")
    
    # Get test data
    test_mask = dataset.test_mask.cpu().numpy()
    labels = dataset.y.cpu().numpy()
    test_indices = np.where(test_mask)[0]
    
    # Subsample if needed
    if len(test_indices) > n:
        subsample_indices = np.random.choice(len(test_indices), n, replace=False)
        test_indices = test_indices[subsample_indices]
    
    # Apply t-SNE to each method's features
    print("\nApplying t-SNE to initial features...")
    coords_list = []
    for features, title in zip(features_list, title_list):
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
        
        test_features = features_np[test_indices]
        print(f"Computing t-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_features) - 1))
        coords = tsne.fit_transform(test_features)
        coords_list.append(coords)
    
    # Create visualizations
    print("\nCreating comparative visualizations...")
    
    # Figure 1: Colored by class with mistakes highlighted
    fig1, axes1 = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes1 = [axes1]
    
    for idx, (coords, mistakes, title) in enumerate(zip(coords_list, all_mistake_indices, title_list)):
        ax = axes1[idx]
        
        # Create mistake mask for subsampled data
        mistake_mask = np.isin(test_indices, mistakes)
        test_labels = labels[test_indices]
        
        # Plot correct predictions
        correct_mask = ~mistake_mask
        scatter = ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                           c=test_labels[correct_mask], cmap='tab10',
                           s=30, alpha=0.6, edgecolors='none')
        
        # Plot mistakes with red X
        if mistake_mask.sum() > 0:
            ax.scatter(coords[mistake_mask, 0], coords[mistake_mask, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        
        # Plot ensemble mistakes if provided
        if ensemble_predictions is not None and ensemble_mistake_indices is not None:
            ensemble_mask = np.isin(test_indices, ensemble_mistake_indices)
            if ensemble_mask.sum() > 0:
                ax.scatter(coords[ensemble_mask, 0], coords[ensemble_mask, 1],
                          c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        
        ax.set_title(f'{title}\n(Initial Features)', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    filename1 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_init_features_mistakes.png'
    filepath1 = os.path.join(save_dir, filename1)
    try:
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        print(f"Initial features mistakes visualization saved as '{filepath1}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    # Figure 2: Mistakes only (highlighting correct vs mistakes)
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes2 = [axes2]
    
    for idx, (coords, mistakes, title) in enumerate(zip(coords_list, all_mistake_indices, title_list)):
        ax = axes2[idx]
        
        mistake_mask = np.isin(test_indices, mistakes)
        
        # Plot correct
        correct_mask = ~mistake_mask
        ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                  c='green', s=30, alpha=0.5, edgecolors='none', label='Correct')
        
        # Plot mistakes
        if mistake_mask.sum() > 0:
            ax.scatter(coords[mistake_mask, 0], coords[mistake_mask, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        
        # Plot ensemble mistakes if provided
        if ensemble_predictions is not None and ensemble_mistake_indices is not None:
            ensemble_mask = np.isin(test_indices, ensemble_mistake_indices)
            if ensemble_mask.sum() > 0:
                ax.scatter(coords[ensemble_mask, 0], coords[ensemble_mask, 1],
                          c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        
        ax.set_title(f'{title}\n(Mistakes: {len(mistakes)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename2 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_init_features_mistakes_only.png'
    filepath2 = os.path.join(save_dir, filename2)
    try:
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"Mistakes-only visualization saved as '{filepath2}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    print("\nInitial features mistake visualizations complete!")
    return all_mistake_indices, ensemble_mistake_indices

def visualize_gnn_embedding_mistakes_list(cfg, dataset, gnn_model_list, features_list, title_list, ensemble_predictions=None, n=2000, save_dir="results/gnn_mistakes", show=False):
    """
        visualize_gnn_embedding_mistakes for multiple gnn models and features with subplots
        Args:
            0. cfg: config
            1. dataset: graph dataset with data.y, data.test_mask from load_graph_dataset_for_tape
            2. gnn_model_list: list of trained gnn models GNNEncoder from
            3. features_list: list of node features embeddings
            4. ensemble_predictions: predictions from ensemble model (tensor or numpy array)
            5. n: number of nodes to visualize
            6. save_dir: directory to save the plots
    """
    print("\nGenerating GNN embedding mistake visualizations for multiple methods...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    n_methods = len(gnn_model_list)
    
    # Get mistakes for all models
    print("Finding mistakes for all models...")
    all_mistake_indices = []
    for features, model, title in zip(features_list, gnn_model_list, title_list):
        mistakes = get_gnn_embedding_mistakes(dataset, model, features)
        all_mistake_indices.append(mistakes)
        print(f"{title}: {len(mistakes)} mistakes")
    
    # Get ensemble mistakes if provided
    ensemble_mistake_indices = None
    if ensemble_predictions is not None:
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
        print(f"Ensemble: {len(ensemble_mistake_indices)} mistakes")
    
    # Extract GNN embeddings for all models
    print("\nExtracting GNN embeddings for all models...")
    embeddings_list = []
    for features, model, title in zip(features_list, gnn_model_list, title_list):
        device = next(model.parameters()).device
        embeddings, _ = extract_gnn_embedding(dataset, features, model, device)
        embeddings_list.append(embeddings)
        print(f"Extracted embeddings for {title}")
    
    # Get test data
    test_mask = dataset.test_mask.cpu().numpy()
    labels = dataset.y.cpu().numpy()
    test_indices = np.where(test_mask)[0]
    
    # Subsample if needed
    if len(test_indices) > n:
        subsample_indices = np.random.choice(len(test_indices), n, replace=False)
        test_indices = test_indices[subsample_indices]
    
    # Apply t-SNE to each method's GNN embeddings
    print("\nApplying t-SNE to GNN embeddings...")
    coords_list = []
    for embeddings, title in zip(embeddings_list, title_list):
        test_embeddings = embeddings[test_indices]
        print(f"Computing t-SNE for {title}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_embeddings) - 1))
        coords = tsne.fit_transform(test_embeddings)
        coords_list.append(coords)
    
    # Create visualizations
    print("\nCreating comparative visualizations...")
    
    # Figure 1: Colored by class with mistakes highlighted
    fig1, axes1 = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes1 = [axes1]
    
    for idx, (coords, mistakes, title) in enumerate(zip(coords_list, all_mistake_indices, title_list)):
        ax = axes1[idx]
        
        # Create mistake mask for subsampled data
        mistake_mask = np.isin(test_indices, mistakes)
        test_labels = labels[test_indices]
        
        # Plot correct predictions
        correct_mask = ~mistake_mask
        scatter = ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                           c=test_labels[correct_mask], cmap='tab10',
                           s=30, alpha=0.6, edgecolors='none')
        
        # Plot mistakes with red X
        if mistake_mask.sum() > 0:
            ax.scatter(coords[mistake_mask, 0], coords[mistake_mask, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        
        # Plot ensemble mistakes if provided
        if ensemble_predictions is not None and ensemble_mistake_indices is not None:
            ensemble_mask = np.isin(test_indices, ensemble_mistake_indices)
            if ensemble_mask.sum() > 0:
                ax.scatter(coords[ensemble_mask, 0], coords[ensemble_mask, 1],
                          c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        
        ax.set_title(f'{title}\n(GNN Embeddings)', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    filename1 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_gnn_embeddings_mistakes.png'
    filepath1 = os.path.join(save_dir, filename1)
    try:
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        print(f"GNN embeddings mistakes visualization saved as '{filepath1}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    # Figure 2: Mistakes only (highlighting correct vs mistakes)
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes2 = [axes2]
    
    for idx, (coords, mistakes, title) in enumerate(zip(coords_list, all_mistake_indices, title_list)):
        ax = axes2[idx]
        
        mistake_mask = np.isin(test_indices, mistakes)
        
        # Plot correct
        correct_mask = ~mistake_mask
        ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1],
                  c='green', s=30, alpha=0.5, edgecolors='none', label='Correct')
        
        # Plot mistakes
        if mistake_mask.sum() > 0:
            ax.scatter(coords[mistake_mask, 0], coords[mistake_mask, 1],
                      c='red', s=100, alpha=0.8, marker='x', linewidths=2, label='GNN Mistakes')
        
        # Plot ensemble mistakes if provided
        if ensemble_predictions is not None and ensemble_mistake_indices is not None:
            ensemble_mask = np.isin(test_indices, ensemble_mistake_indices)
            if ensemble_mask.sum() > 0:
                ax.scatter(coords[ensemble_mask, 0], coords[ensemble_mask, 1],
                          c='purple', s=80, alpha=0.7, marker='+', linewidths=2, label='Ensemble Mistakes')
        
        ax.set_title(f'{title}\n(Mistakes: {len(mistakes)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename2 = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_gnn_embeddings_mistakes_only.png'
    filepath2 = os.path.join(save_dir, filename2)
    try:
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        print(f"Mistakes-only visualization saved as '{filepath2}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    print("\nGNN embedding mistake visualizations complete!")
    return all_mistake_indices, ensemble_mistake_indices

def comprehensive_gnn_mistakes_analysis_visualization(cfg, dataset, gnn_model, features, title, ensemble_predictions=None, save_dir="results/gnn_mistakes_comprehensive", show=False):
    """
        also compare class based miskaes. plot charts to help understand the mistakes.
        also charts for train, val, test based mistakes.  dataset.train_mask, dataset.val_mask, dataset.test_mask
        labels: dataset.y
    """
    print(f"\nGenerating comprehensive mistake analysis for {title}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    # Get mistakes
    train_mistake_indices, val_mistake_indices, test_mistake_indices, all_mistake_indices = get_gnn_embedding_mistakes(dataset, gnn_model, features, output_mask='all')
    ensemble_mistake_indices = None
    if ensemble_predictions is not None:
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
    
    # Get data
    labels = dataset.y.cpu().numpy()
    train_mask = dataset.train_mask.cpu().numpy()
    val_mask = dataset.val_mask.cpu().numpy()
    test_mask = dataset.test_mask.cpu().numpy()
    
    # Get predictions
    device = next(gnn_model.parameters()).device
    gnn_model.eval()
    with torch.no_grad():
        x = features.to(device) if isinstance(features, torch.Tensor) else torch.tensor(features).to(device)
        edge_index = dataset.edge_index.to(device)
        logits = gnn_model(x, edge_index)
        pred = logits.argmax(dim=1).cpu().numpy()
    
    # Analyze mistakes by class
    num_classes = len(np.unique(labels))
    mistakes_by_class = {i: [] for i in range(num_classes)}
    total_by_class = {i: 0 for i in range(num_classes)}
    
    for node_idx in range(len(labels)):
        if test_mask[node_idx]:  # Only consider test nodes
            true_label = labels[node_idx]
            total_by_class[true_label] += 1
            if node_idx in test_mistake_indices:
                mistakes_by_class[true_label].append(node_idx)
    
    # Analyze mistakes by split
    mistakes_by_split = {
        'train': len(train_mistake_indices),
        'val': len(val_mistake_indices),
        'test': len(test_mistake_indices)
    }
    total_by_split = {
        'train': train_mask.sum(),
        'val': val_mask.sum(),
        'test': test_mask.sum()
    }
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Mistakes by Class (Count)
    ax1 = fig.add_subplot(gs[0, 0])
    classes = list(range(num_classes))
    mistake_counts = [len(mistakes_by_class[i]) for i in classes]
    bars = ax1.bar(classes, mistake_counts, color='lightcoral', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Number of Mistakes', fontsize=11)
    ax1.set_title(f'{title}\nMistakes by Class', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    for bar, count in zip(bars, mistake_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Error Rate by Class
    ax2 = fig.add_subplot(gs[0, 1])
    error_rates = [len(mistakes_by_class[i]) / max(total_by_class[i], 1) * 100 for i in classes]
    bars = ax2.bar(classes, error_rates, color='steelblue', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Error Rate (%)', fontsize=11)
    ax2.set_title(f'{title}\nError Rate by Class', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    for bar, rate in zip(bars, error_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Mistakes by Split
    ax3 = fig.add_subplot(gs[0, 2])
    splits = ['Train', 'Val', 'Test']
    split_counts = [mistakes_by_split['train'], mistakes_by_split['val'], mistakes_by_split['test']]
    colors_split = ['skyblue', 'lightgreen', 'lightcoral']
    bars = ax3.bar(splits, split_counts, color=colors_split, edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Number of Mistakes', fontsize=11)
    ax3.set_title(f'{title}\nMistakes by Split', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    for bar, count in zip(bars, split_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Error Rate by Split
    ax4 = fig.add_subplot(gs[1, 0])
    split_error_rates = [
        mistakes_by_split['train'] / max(total_by_split['train'], 1) * 100,
        mistakes_by_split['val'] / max(total_by_split['val'], 1) * 100,
        mistakes_by_split['test'] / max(total_by_split['test'], 1) * 100
    ]
    bars = ax4.bar(splits, split_error_rates, color=colors_split, edgecolor='black', alpha=0.8)
    ax4.set_ylabel('Error Rate (%)', fontsize=11)
    ax4.set_title(f'{title}\nError Rate by Split', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    for bar, rate in zip(bars, split_error_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Confusion Matrix (Test Set Only)
    ax5 = fig.add_subplot(gs[1, 1])
    test_indices = np.where(test_mask)[0]
    test_labels = labels[test_indices]
    test_preds = pred[test_indices]
    
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(test_labels, test_preds):
        conf_matrix[true_label, pred_label] += 1
    
    im = ax5.imshow(conf_matrix, cmap='Blues', aspect='auto')
    ax5.set_xlabel('Predicted Class', fontsize=11)
    ax5.set_ylabel('True Class', fontsize=11)
    ax5.set_title(f'{title}\nConfusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(num_classes))
    ax5.set_yticks(range(num_classes))
    plt.colorbar(im, ax=ax5)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax5.text(j, i, conf_matrix[i, j],
                          ha="center", va="center", color="black" if conf_matrix[i, j] < conf_matrix.max()/2 else "white",
                          fontsize=9)
    
    # Plot 6: Class Distribution in Test Set
    ax6 = fig.add_subplot(gs[1, 2])
    class_counts = [np.sum(test_labels == i) for i in range(num_classes)]
    bars = ax6.bar(classes, class_counts, color='lightgreen', edgecolor='black', alpha=0.8)
    ax6.set_xlabel('Class', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title(f'{title}\nClass Distribution (Test Set)', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')
    
    # Plot 7: Statistics Table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    stats_text = f"""
{title} - MISTAKE ANALYSIS SUMMARY
{'=' * 100}

Overall Performance:
  • Total Test Nodes: {test_mask.sum()}
  • Total Mistakes: {len(test_mistake_indices)}
  • Overall Accuracy: {(1 - len(test_mistake_indices) / max(test_mask.sum(), 1)) * 100:.2f}%

Per-Class Statistics (Test Set):
"""
    for i in range(num_classes):
        correct = total_by_class[i] - len(mistakes_by_class[i])
        total = total_by_class[i]
        acc = correct / max(total, 1) * 100
        stats_text += f"  Class {i}: {len(mistakes_by_class[i])} mistakes / {total} samples = {100-acc:.2f}% error (Acc: {acc:.2f}%)\n"
    
    stats_text += f"""
Per-Split Statistics:
  Train: {mistakes_by_split['train']} mistakes / {total_by_split['train']} samples = {split_error_rates[0]:.2f}% error
  Val:   {mistakes_by_split['val']} mistakes / {total_by_split['val']} samples = {split_error_rates[1]:.2f}% error
  Test:  {mistakes_by_split['test']} mistakes / {total_by_split['test']} samples = {split_error_rates[2]:.2f}% error
"""
    
    if ensemble_predictions is not None:
        stats_text += f"""
Ensemble Comparison:
  Ensemble Mistakes: {len(ensemble_mistake_indices)}
  GNN Only Mistakes: {len(set(test_mistake_indices) - set(ensemble_mistake_indices))}
  Ensemble Only Mistakes: {len(set(ensemble_mistake_indices) - set(test_mistake_indices))}
  Both Wrong: {len(set(test_mistake_indices) & set(ensemble_mistake_indices))}
"""
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_{title}_comprehensive_analysis.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    print(f"\nComprehensive analysis complete for {title}!")
    return test_mistake_indices, ensemble_mistake_indices

def comprehensive_gnn_mistakes_analysis_visualization_list(cfg, dataset, gnn_model_list, features_list, title_list, ensemble_predictions=None, save_dir="results/gnn_mistakes_comprehensive", show=False, reporter=None):
    """
        Compare comprehensive mistake analysis across multiple models
    """
    print("\nGenerating comparative comprehensive mistake analysis...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tuple from load_graph_dataset_for_tape
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    
    n_methods = len(gnn_model_list)
    
    # Get data
    labels = dataset.y.cpu().numpy()
    train_mask = dataset.train_mask.cpu().numpy()
    val_mask = dataset.val_mask.cpu().numpy()
    test_mask = dataset.test_mask.cpu().numpy()
    num_classes = len(np.unique(labels))
    
    # Collect statistics for all models
    all_mistake_indices = []
    all_stats = []
    
    for features, model, title in zip(features_list, gnn_model_list, title_list):
        print(f"Analyzing {title}...")
        
        # Get mistakes
        train_mistake_indices, val_mistake_indices, test_mistake_indices, all_mistakes = get_gnn_embedding_mistakes(dataset, model, features, output_mask='all')
        all_mistake_indices.append(test_mistake_indices)
        
        # Get predictions
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            x = features.to(device) if isinstance(features, torch.Tensor) else torch.tensor(features).to(device)
            edge_index = dataset.edge_index.to(device)
            logits = model(x, edge_index)
            pred = logits.argmax(dim=1).cpu().numpy()
        
        # Analyze mistakes by class
        mistakes_by_class = {i: [] for i in range(num_classes)}
        total_by_class = {i: 0 for i in range(num_classes)}
        
        for node_idx in range(len(labels)):
            if test_mask[node_idx]:
                true_label = labels[node_idx]
                total_by_class[true_label] += 1
                if node_idx in test_mistake_indices:
                    mistakes_by_class[true_label].append(node_idx)
        
        # Analyze mistakes by split
        mistakes_by_split = {
            'train': len(train_mistake_indices),
            'val': len(val_mistake_indices),
            'test': len(test_mistake_indices)
        }
        
        all_stats.append({
            'mistakes_by_class': mistakes_by_class,
            'total_by_class': total_by_class,
            'mistakes_by_split': mistakes_by_split,
            'total_mistakes': len(test_mistake_indices)
        })
        
        # Report statistics using reporter
        if reporter is not None:
            test_acc = (1 - len(test_mistake_indices) / max(test_mask.sum(), 1)) * 100
            report_text = f"""
Mistake Analysis:
  • Total Mistakes: {len(test_mistake_indices)}
  • Test Accuracy: {test_acc:.2f}%
  • Test Nodes: {test_mask.sum()}
  
Mistakes by Split:
  • Train: {mistakes_by_split['train']}
  • Validation: {mistakes_by_split['val']}
  • Test: {mistakes_by_split['test']}

Mistakes by Class:
"""
            for j in range(num_classes):
                total = total_by_class[j]
                mistakes = len(mistakes_by_class[j])
                err_rate = mistakes / max(total, 1) * 100
                report_text += f"  • Class {j}: {mistakes}/{total} ({err_rate:.2f}%)\n"
            
            reporter.report(f"Comprehensive GNN Mistakes Analysis - {title}", report_text)
    
    # Create comparative visualizations
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Total Mistakes Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(n_methods)
    mistake_counts = [stats['total_mistakes'] for stats in all_stats]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    bars = ax1.bar(x, mistake_counts, color=[colors[i % len(colors)] for i in range(n_methods)],
                   edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Method', fontsize=11)
    ax1.set_ylabel('Number of Mistakes', fontsize=11)
    ax1.set_title('Total Mistakes Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(title_list, rotation=45, ha='right')
    ax1.grid(alpha=0.3, axis='y')
    for bar, count in zip(bars, mistake_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Error Rates by Split
    ax2 = fig.add_subplot(gs[0, 1])
    train_errors = [stats['mistakes_by_split']['train'] / max(train_mask.sum(), 1) * 100 for stats in all_stats]
    val_errors = [stats['mistakes_by_split']['val'] / max(val_mask.sum(), 1) * 100 for stats in all_stats]
    test_errors = [stats['mistakes_by_split']['test'] / max(test_mask.sum(), 1) * 100 for stats in all_stats]
    
    width = 0.25
    ax2.bar(x - width, train_errors, width, label='Train', color='skyblue', edgecolor='black', alpha=0.8)
    ax2.bar(x, val_errors, width, label='Val', color='lightgreen', edgecolor='black', alpha=0.8)
    ax2.bar(x + width, test_errors, width, label='Test', color='lightcoral', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Error Rate (%)', fontsize=11)
    ax2.set_title('Error Rates by Split', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(title_list, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Plot 3: Mistakes by Class (Heatmap)
    ax3 = fig.add_subplot(gs[1, :])
    mistake_matrix = np.zeros((n_methods, num_classes))
    for i, stats in enumerate(all_stats):
        for j in range(num_classes):
            mistake_matrix[i, j] = len(stats['mistakes_by_class'][j])
    
    im = ax3.imshow(mistake_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xlabel('Class', fontsize=11)
    ax3.set_ylabel('Method', fontsize=11)
    ax3.set_title('Mistakes by Class (Heatmap)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(num_classes))
    ax3.set_yticks(range(n_methods))
    ax3.set_yticklabels(title_list)
    plt.colorbar(im, ax=ax3, label='Number of Mistakes')
    
    # Add text annotations
    for i in range(n_methods):
        for j in range(num_classes):
            text = ax3.text(j, i, int(mistake_matrix[i, j]),
                          ha="center", va="center",
                          color="white" if mistake_matrix[i, j] > mistake_matrix.max()/2 else "black",
                          fontsize=8)
    
    # Plot 4: Error Rate by Class (Heatmap)
    ax4 = fig.add_subplot(gs[2, :])
    error_rate_matrix = np.zeros((n_methods, num_classes))
    for i, stats in enumerate(all_stats):
        for j in range(num_classes):
            total = stats['total_by_class'][j]
            mistakes = len(stats['mistakes_by_class'][j])
            error_rate_matrix[i, j] = mistakes / max(total, 1) * 100
    
    im = ax4.imshow(error_rate_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    ax4.set_xlabel('Class', fontsize=11)
    ax4.set_ylabel('Method', fontsize=11)
    ax4.set_title('Error Rate by Class (%)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(num_classes))
    ax4.set_yticks(range(n_methods))
    ax4.set_yticklabels(title_list)
    plt.colorbar(im, ax=ax4, label='Error Rate (%)')
    
    # Add text annotations
    for i in range(n_methods):
        for j in range(num_classes):
            text = ax4.text(j, i, f'{error_rate_matrix[i, j]:.1f}',
                          ha="center", va="center",
                          color="white" if error_rate_matrix[i, j] > 50 else "black",
                          fontsize=8)
    
    # Plot 5: Statistics Summary
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    stats_text = "COMPARATIVE MISTAKE ANALYSIS SUMMARY\n" + "=" * 120 + "\n\n"
    
    for i, (title, stats) in enumerate(zip(title_list, all_stats)):
        test_acc = (1 - stats['total_mistakes'] / max(test_mask.sum(), 1)) * 100
        stats_text += f"{title}:\n"
        stats_text += f"  Total Mistakes: {stats['total_mistakes']} | Test Accuracy: {test_acc:.2f}%\n"
        stats_text += f"  By Split - Train: {stats['mistakes_by_split']['train']}, Val: {stats['mistakes_by_split']['val']}, Test: {stats['mistakes_by_split']['test']}\n"
        
        # Per-class summary
        class_errors = []
        for j in range(num_classes):
            total = stats['total_by_class'][j]
            mistakes = len(stats['mistakes_by_class'][j])
            err_rate = mistakes / max(total, 1) * 100
            class_errors.append(f"C{j}:{err_rate:.1f}%")
        stats_text += f"  Per-Class Errors: {', '.join(class_errors)}\n\n"
    
    if ensemble_predictions is not None:
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
        stats_text += f"\nEnsemble Model: {len(ensemble_mistake_indices)} total mistakes\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    filename = f'{cfg.llm.model_name}_{cfg.dataset.name}_{cfg.peft.type}_comprehensive_comparison.png'
    filepath = os.path.join(save_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Comprehensive comparison saved as '{filepath}'")
    except:
        print("Cannot save in jupyter notebook environment. Just showing the plot.")
    if show:
        plt.show()
    
    # Report ensemble statistics if provided
    if ensemble_predictions is not None and reporter is not None:
        ensemble_mistake_indices = get_embeddings_mistakes(dataset, ensemble_predictions)
        ensemble_test_acc = (1 - len(ensemble_mistake_indices) / max(test_mask.sum(), 1)) * 100
        ensemble_report_text = f"""
Ensemble Model Mistake Analysis:
  • Total Mistakes: {len(ensemble_mistake_indices)}
  • Test Accuracy: {ensemble_test_acc:.2f}%
  • Test Nodes: {test_mask.sum()}
"""
        reporter.report("Ensemble Model Mistakes Analysis", ensemble_report_text)
    
    print("\nComprehensive comparative analysis complete!")
    return all_mistake_indices, all_stats