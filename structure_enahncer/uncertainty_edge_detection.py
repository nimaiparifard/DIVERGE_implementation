# K = 10 Model Cross Validation for detecting noisy edges
# here in this python file we want to detect noisy edges with GNN Edge Predictor
# for this we trained K Edge Predictor with diffirent seed with same split size implement that way i can set split size 60/20/20 is default
# after this we should have list of probability for each edges in the graph
# for detecting noisy edge we want to use standard deviation and entopy
# based on threshold we remove noisy edges
# contruct new graph dataset based on deleted edges
# prepare new dataset for node classifcation
# compare and report results before and after edge editing

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import os
import json
from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from LLMReasoner.TAPE.peft_tape_finetuning.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from LLMReasoner.TAPE.peft_tape_finetuning.gnn_mtrainer import *
from LLMReasoner.TAPE.tags_data_augmentation.edge_predictor import *
from LLMReasoner.TAPE.peft_tape_finetuning.reporter import ReportResults


class NoisyEdgeDetector:
    """
    Detect noisy edges using ensemble of edge predictors with different random seeds.
    Uses uncertainty metrics (standard deviation and entropy) to identify edges to remove.
    """
    
    def __init__(self, cfg, features, num_models=10, decoder_type='mlp', 
                 training_strategy='with_splitting', negative_sampling_ratio=1.0):
        """
        Initialize the noisy edge detector.

        Args:
            cfg: Configuration object
            features: Node embeddings [num_nodes, feature_dim]
            num_models: Number of edge predictor models to train with different seeds (default: 10)
            decoder_type: Type of decoder ('dot_product', 'mlp', 'bilinear')
            training_strategy: 'with_splitting' or 'without_splitting'
            negative_sampling_ratio: Ratio of negative to positive edges
        """
        self.cfg = cfg
        self.features = features
        self.num_models = num_models
        self.decoder_type = decoder_type
        self.training_strategy = training_strategy
        self.negative_sampling_ratio = negative_sampling_ratio
        self.models = []
        self.edge_probabilities = []  # List of probability tensors, one per model
        self.dataset = None
        self.original_edge_index = None  # Store original edge_index explicitly
        self.device = torch.device('cuda:0' if cfg.device > 0 else 'cpu')
        
        # Load dataset
        datasets_path = get_datasets_path()
        repo_root = os.path.dirname(datasets_path)
        if os.path.exists(os.path.join(repo_root, 'datasets')):
            if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
                path_prefix = '.'
            else:
                path_prefix = os.path.relpath(repo_root, start=os.getcwd())
                path_prefix = os.path.normpath(path_prefix)
        else:
            path_prefix = '../..'
        
        self.dataset, _, _ = load_graph_dataset_for_tape(
            cfg.dataset, 
            self.device, 
            re_split=cfg.re_split, 
            path_prefix=path_prefix, 
            seed=cfg.split_seed
        )

        self.original_edge_index = self.dataset.edge_index.clone()
        self.original_num_edges = self.original_edge_index.shape[1]  # Store the count for verification
        dataset_name = cfg.dataset if isinstance(cfg.dataset, str) else getattr(cfg.dataset, 'name', str(cfg.dataset))

    def train_edge_predictors(self):
        """
        Train K edge predictor models with different random seeds.
        Each model will predict probabilities for all edges.
        """
        
        original_seed = self.cfg.seed
        
        for i in range(self.num_models):
            # Set different seed for each model
            seed = original_seed + i * 100
            model_cfg = deepcopy(self.cfg)
            model_cfg.seed = seed
            set_seed(seed)
            
            print(f"Training model {i+1}/{self.num_models} (seed={seed})")
            
            # Verify original_edge_index hasn't been modified
            print(f"  [DEBUG] self.original_edge_index shape BEFORE EdgePredictor: {self.original_edge_index.shape}")
            
            # Create and train edge predictor
            # NOTE: We DON'T pass modified_dataset - let each EdgePredictor load its own dataset
            # We only care about computing probabilities for our original edges afterward
            edge_predictor = EdgePredictor(
                model_cfg,
                self.features,
                decoder_approach_type=self.decoder_type,
                training_strategy=self.training_strategy,
                negative_sampling_ratio=self.negative_sampling_ratio
                # No modified_dataset parameter - each model trains on its own dataset
            )
            
            # Verify original_edge_index hasn't been modified
            print(f"  [DEBUG] self.original_edge_index shape AFTER EdgePredictor: {self.original_edge_index.shape}")
            print(f"  [DEBUG] self.dataset.edge_index shape AFTER EdgePredictor: {self.dataset.edge_index.shape}")
            print(f"  [DEBUG] Are they the same? {torch.equal(self.original_edge_index, self.dataset.edge_index)}")
            
            # Train the model
            results, history = edge_predictor.train(show_plots=False)
            
            print(f"  Model {i+1} - Val Acc: {results['val_acc']:.4f}, Test Acc: {results['test_acc']:.4f}\n")
            
            # Store trained model
            self.models.append(edge_predictor.model)
            
            # Get probabilities for all edges in the original graph
            edge_probs = self._get_edge_probabilities(edge_predictor)
            
            # Verify the number of probabilities matches the original edge count
            expected_num_edges = self.original_edge_index.shape[1]
            if len(edge_probs) != expected_num_edges:
                raise ValueError(
                    f"Mismatch in number of edges! "
                    f"Expected {expected_num_edges} edges (from original_edge_index) "
                    f"but got {len(edge_probs)} probabilities. "
                    f"EdgePredictor data has {edge_predictor.data.edge_index.shape[1]} edges. "
                    f"This might be due to dataset modification during EdgePredictor initialization."
                )
            
            self.edge_probabilities.append(edge_probs)
        
        # Reset to original seed
        set_seed(original_seed)
        
        print(f"[OK] Trained {self.num_models} edge predictor models")
        print(f"     Each model predicted probabilities for {len(self.edge_probabilities[0])} edges")
        print(f"     Original edge count: {self.original_edge_index.shape[1]} edges\n")

    def _get_edge_probabilities(self, edge_predictor):
        """
        Get edge probabilities for all edges in the original graph using a trained edge predictor.
        
        This method manually computes probabilities for each edge by:
        1. Getting GNN embeddings for all nodes using the trained encoder
        2. For each edge in original_edge_index, computing the score using the decoder
        
        Args:
            edge_predictor: Trained EdgePredictor object
        
        Returns:
            edge_probs: Tensor of probabilities for each edge [num_edges]
        """
        edge_predictor.model.eval()
        device = edge_predictor.device
        
        print(f"    [DEBUG] self.original_edge_index id: {id(self.original_edge_index)}")
        print(f"    [DEBUG] self.original_edge_index shape: {self.original_edge_index.shape}")
        print(f"    [DEBUG] self.original_num_edges (stored at init): {self.original_num_edges}")
        print(f"    [DEBUG] Computing probabilities for {self.original_edge_index.shape[1]} original edges")
        
        # CRITICAL: Verify original_edge_index hasn't been modified
        if self.original_edge_index.shape[1] != self.original_num_edges:
            raise RuntimeError(
                f"CRITICAL ERROR: original_edge_index has been modified! "
                f"Expected {self.original_num_edges} edges but found {self.original_edge_index.shape[1]} edges. "
                f"This should never happen!"
            )
        
        with torch.no_grad():
            # Step 1: Get GNN embeddings for all nodes
            # Use the ORIGINAL graph structure for consistent node embeddings
            original_edge_index = self.original_edge_index.to(device)
            features = edge_predictor.features.to(device)
            
            # Get the encoder and decoder from the edge predictor model
            encoder = edge_predictor.model.encoder
            decoder = edge_predictor.model.decoder
            
            # Encode all nodes using the GNN encoder with original graph structure
            # This gives us node embeddings: [num_nodes, hidden_dim]
            node_embeddings = encoder(features, original_edge_index)
            
            # Step 2: For each edge in original_edge_index, compute the edge score
            num_edges = original_edge_index.shape[1]
            edge_scores = []
            
            # Process edges in batches to avoid memory issues
            batch_size = 10000
            for start_idx in range(0, num_edges, batch_size):
                end_idx = min(start_idx + batch_size, num_edges)
                batch_edges = original_edge_index[:, start_idx:end_idx]
                
                # Get source and destination node embeddings for this batch
                src_nodes = batch_edges[0]  # [batch_size]
                dst_nodes = batch_edges[1]  # [batch_size]
                
                src_embeddings = node_embeddings[src_nodes]  # [batch_size, hidden_dim]
                dst_embeddings = node_embeddings[dst_nodes]  # [batch_size, hidden_dim]
                
                # Compute edge scores using the decoder
                batch_scores = decoder(src_embeddings, dst_embeddings)  # [batch_size, 1] or [batch_size]
                
                # Ensure batch_scores is 1D
                if batch_scores.dim() == 2:
                    batch_scores = batch_scores.squeeze(1)
                
                edge_scores.append(batch_scores)
            
            # Concatenate all batch scores
            all_edge_scores = torch.cat(edge_scores, dim=0)  # [num_edges]
            
            print(f"    [DEBUG] Number of edge score batches: {len(edge_scores)}")
            print(f"    [DEBUG] Sizes of batches: {[s.shape for s in edge_scores]}")
            print(f"    [DEBUG] Computed edge scores shape: {all_edge_scores.shape}")
            print(f"    [DEBUG] Expected shape: torch.Size([{num_edges}])")
            print(f"    [DEBUG] Original edge_index used shape: {original_edge_index.shape}")
            
            # Verify we got the right number
            if all_edge_scores.shape[0] != num_edges:
                raise RuntimeError(
                    f"ERROR: Computed {all_edge_scores.shape[0]} edge scores but expected {num_edges}! "
                    f"Original edge_index shape: {original_edge_index.shape}, "
                    f"Batches processed: {len(edge_scores)}"
                )
            
            # Apply sigmoid to get probabilities
            edge_probs = torch.sigmoid(all_edge_scores)
            
            print(f"    [DEBUG] Final edge_probs shape: {edge_probs.shape}")
        
        return edge_probs.cpu()

    def calculate_standard_deviation(self):
        """
        Calculate standard deviation of edge probabilities across all models.
        Higher std indicates higher uncertainty.
        
        Returns:
            std_scores: Standard deviation for each edge [num_edges]
        """
        # Stack all probabilities [num_models, num_edges]
        probs_matrix = torch.stack(self.edge_probabilities, dim=0)
        
        # Calculate standard deviation across models
        std_scores = torch.std(probs_matrix, dim=0)
        
        print(f"[OK] Calculated standard deviation for {len(std_scores)} edges")
        print(f"     Mean std: {std_scores.mean():.4f}, Max std: {std_scores.max():.4f}, Min std: {std_scores.min():.4f}")
        
        return std_scores

    def calculate_entropy(self):
        """
        Calculate entropy of edge probabilities across all models.
        Higher entropy indicates higher uncertainty.
        
        For each edge, we compute the entropy of the probability distribution
        across all models using the formula: H = -sum(p * log(p))
        
        Returns:
            entropy_scores: Entropy for each edge [num_edges]
        """
        # Stack all probabilities [num_models, num_edges]
        probs_matrix = torch.stack(self.edge_probabilities, dim=0).numpy()
        
        # Calculate mean probability for each edge
        mean_probs = probs_matrix.mean(axis=0)
        
        # Calculate entropy for each edge
        # We treat each edge as having a binary outcome (exists or doesn't exist)
        # Entropy of a binary variable with probability p: H = -p*log(p) - (1-p)*log(1-p)
        entropy_scores = np.zeros(mean_probs.shape[0])
        for i in range(len(mean_probs)):
            p = mean_probs[i]
            # Avoid log(0) by adding small epsilon
            eps = 1e-10
            p = np.clip(p, eps, 1 - eps)
            entropy_scores[i] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        
        entropy_scores = torch.tensor(entropy_scores, dtype=torch.float32)
        
        print(f"[OK] Calculated entropy for {len(entropy_scores)} edges")
        print(f"     Mean entropy: {entropy_scores.mean():.4f}, Max entropy: {entropy_scores.max():.4f}, Min entropy: {entropy_scores.min():.4f}")
        
        return entropy_scores
    
    def get_uncertainty_scores(self, metric='std'):
        """
        Get uncertainty scores for all edges.
        
        Args:
            metric: 'std' for standard deviation, 'entropy' for entropy, 'both' for combined
        
        Returns:
            uncertainty_scores: Tensor of uncertainty scores [num_edges]
        """
        if metric == 'std':
            return self.calculate_standard_deviation()
        elif metric == 'entropy':
            return self.calculate_entropy()
        elif metric == 'both':
            std_scores = self.calculate_standard_deviation()
            entropy_scores = self.calculate_entropy()
            # Normalize both scores to [0, 1] and average
            std_norm = (std_scores - std_scores.min()) / (std_scores.max() - std_scores.min() + 1e-10)
            entropy_norm = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min() + 1e-10)
            return (std_norm + entropy_norm) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'std', 'entropy', or 'both'")
    
    def visualize_uncertainty(self, uncertainty_scores, metric='std', save_dir='results/edge_uncertainty'):
        """
        Visualize uncertainty scores distribution.
        
        Args:
            uncertainty_scores: Tensor of uncertainty scores
            metric: Name of the metric for plot title
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        uncertainty_np = uncertainty_scores.numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Histogram
        axes[0].hist(uncertainty_np, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Uncertainty Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Distribution of {metric.upper()} Scores', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Sorted uncertainty scores
        sorted_scores = np.sort(uncertainty_np)
        axes[1].plot(range(len(sorted_scores)), sorted_scores, linewidth=2, color='coral')
        axes[1].set_xlabel('Edge Index (sorted)', fontsize=12)
        axes[1].set_ylabel('Uncertainty Score', fontsize=12)
        axes[1].set_title(f'Sorted {metric.upper()} Scores', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # Plot 3: Statistics
        axes[2].axis('off')
        stats_text = f"""
Uncertainty Statistics ({metric.upper()}):
{'='*50}

Overall Statistics:
  • Total Edges: {len(uncertainty_np)}
  • Mean: {np.mean(uncertainty_np):.4f}
  • Median: {np.median(uncertainty_np):.4f}
  • Std Dev: {np.std(uncertainty_np):.4f}
  • Min: {np.min(uncertainty_np):.4f}
  • Max: {np.max(uncertainty_np):.4f}

Quartiles:
  • Q1 (25%): {np.percentile(uncertainty_np, 25):.4f}
  • Q2 (50%): {np.percentile(uncertainty_np, 50):.4f}
  • Q3 (75%): {np.percentile(uncertainty_np, 75):.4f}
  • Q4 (95%): {np.percentile(uncertainty_np, 95):.4f}
        """
        axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes, fontsize=10,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        # Fix: Use cfg.dataset if it's a string, otherwise use cfg.dataset.name
        dataset_name = self.cfg.dataset if isinstance(self.cfg.dataset, str) else 'unknown'
        save_path = os.path.join(save_dir, f'uncertainty_{metric}_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved uncertainty visualization to: {save_path}")
        plt.close()


def create_modified_dataset(dataset, uncertainty_scores, threshold, removal_strategy='top_k'):
    """
    Create a modified dataset by removing noisy edges based on uncertainty threshold.
    
    Args:
        dataset: Original dataset
        uncertainty_scores: Uncertainty scores for each edge [num_edges]
        threshold: Threshold for removing edges
        removal_strategy: 'threshold' (remove if score > threshold) or 'top_k' (remove top k% with highest uncertainty)
    
    Returns:
        modified_dataset: Dataset with filtered edges
        removal_stats: Statistics about edge removal
    """
    print(f"\n{'='*80}")
    print(f"Creating Modified Dataset")
    print(f"{'='*80}")
    print(f"Removal strategy: {removal_strategy}")
    print(f"Threshold/percentage: {threshold}")
    
    original_edge_index = dataset.edge_index
    num_original_edges = original_edge_index.shape[1]
    num_uncertainty_scores = len(uncertainty_scores)
    
    # Verify dimensions match
    if num_uncertainty_scores != num_original_edges:
        raise ValueError(
            f"Dimension mismatch! "
            f"Dataset has {num_original_edges} edges but uncertainty_scores has {num_uncertainty_scores} elements. "
            f"Edge index shape: {original_edge_index.shape}, uncertainty_scores shape: {uncertainty_scores.shape}"
        )
    
    print(f"Dataset edges: {num_original_edges}")
    print(f"Uncertainty scores: {num_uncertainty_scores}")
    
    # Determine which edges to remove
    if removal_strategy == 'threshold':
        # Remove edges with uncertainty > threshold
        edges_to_keep_mask = uncertainty_scores <= threshold
    elif removal_strategy == 'top_k':
        # Remove top k% of edges with highest uncertainty
        k_percent = threshold
        num_to_remove = int(num_original_edges * k_percent / 100)
        sorted_indices = torch.argsort(uncertainty_scores, descending=True)
        edges_to_remove_indices = sorted_indices[:num_to_remove]
        edges_to_keep_mask = torch.ones(num_original_edges, dtype=torch.bool)
        edges_to_keep_mask[edges_to_remove_indices] = False
    else:
        raise ValueError(f"Unknown removal strategy: {removal_strategy}")
    
    # Filter edges
    kept_edge_index = original_edge_index[:, edges_to_keep_mask]
    removed_edge_index = original_edge_index[:, ~edges_to_keep_mask]
    
    # Create modified dataset
    modified_dataset = dataset.clone()
    modified_dataset.edge_index = kept_edge_index
    
    # Calculate statistics
    num_removed = removed_edge_index.shape[1]
    num_kept = kept_edge_index.shape[1]
    
    removal_stats = {
        'original_edges': num_original_edges,
        'removed_edges': num_removed,
        'kept_edges': num_kept,
        'removal_percentage': (num_removed / num_original_edges) * 100,
        'removed_uncertainty_mean': uncertainty_scores[~edges_to_keep_mask].mean().item(),
        'kept_uncertainty_mean': uncertainty_scores[edges_to_keep_mask].mean().item(),
    }
    
    print(f"\nEdge Removal Statistics:")
    print(f"  Original edges: {num_original_edges}")
    print(f"  Removed edges: {num_removed} ({removal_stats['removal_percentage']:.2f}%)")
    print(f"  Kept edges: {num_kept}")
    print(f"  Removed edges avg uncertainty: {removal_stats['removed_uncertainty_mean']:.4f}")
    print(f"  Kept edges avg uncertainty: {removal_stats['kept_uncertainty_mean']:.4f}")
    print(f"{'='*80}\n")
    
    return modified_dataset, removal_stats


def compare_results(original_results, modified_results, removal_stats, cfg, save_dir='results/edge_uncertainty'):
    """
    Compare and visualize results before and after edge removal.
    
    Args:
        original_results: Results on original dataset
        modified_results: Results on modified dataset
        removal_stats: Statistics about edge removal
        cfg: Configuration object
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: Original vs Modified Dataset")
    print(f"{'='*80}")
    print(f"\nORIGINAL DATASET:")
    print(f"  Test Accuracy:       {original_results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {original_results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {original_results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {original_results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {original_results['val_f1']:.4f}")
    
    print(f"\nMODIFIED DATASET (After Edge Removal):")
    print(f"  Test Accuracy:       {modified_results['test_acc']:.4f}")
    print(f"  Test F1 (Macro):     {modified_results['test_f1']:.4f}")
    print(f"  Test F1 (Weighted):  {modified_results['test_weight_f1']:.4f}")
    print(f"  Val Accuracy:        {modified_results['val_acc']:.4f}")
    print(f"  Val F1 (Macro):      {modified_results['val_f1']:.4f}")
    
    acc_improvement = modified_results['test_acc'] - original_results['test_acc']
    f1_improvement = modified_results['test_f1'] - original_results['test_f1']
    
    print(f"\nIMPROVEMENT:")
    print(f"  Test Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
    print(f"  Test F1: {f1_improvement:+.4f}")
    print(f"{'='*80}\n")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy comparison
    metrics = ['Test Acc', 'Val Acc']
    original_accs = [original_results['test_acc'], original_results['val_acc']]
    modified_accs = [modified_results['test_acc'], modified_results['val_acc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, original_accs, width, label='Original', color='skyblue', edgecolor='black')
    axes[0].bar(x + width/2, modified_accs, width, label='Modified', color='lightcoral', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # Plot 2: F1 comparison
    f1_metrics = ['Test F1', 'Val F1']
    original_f1s = [original_results['test_f1'], original_results['val_f1']]
    modified_f1s = [modified_results['test_f1'], modified_results['val_f1']]
    
    axes[1].bar(x - width/2, original_f1s, width, label='Original', color='skyblue', edgecolor='black')
    axes[1].bar(x + width/2, modified_f1s, width, label='Modified', color='lightcoral', edgecolor='black')
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(f1_metrics)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    # Plot 3: Summary statistics
    axes[2].axis('off')
    summary_text = f"""
Comparison Summary:
{'='*50}

Edge Statistics:
  • Original edges: {removal_stats['original_edges']}
  • Removed edges: {removal_stats['removed_edges']}
  • Removed percentage: {removal_stats['removal_percentage']:.2f}%
  • Kept edges: {removal_stats['kept_edges']}

Performance Changes:
  • Test Acc Δ: {acc_improvement:+.4f}
  • Test F1 Δ: {f1_improvement:+.4f}
  • Val Acc Δ: {modified_results['val_acc'] - original_results['val_acc']:+.4f}
  • Val F1 Δ: {modified_results['val_f1'] - original_results['val_f1']:+.4f}

Uncertainty of Removed Edges:
  • Avg: {removal_stats['removed_uncertainty_mean']:.4f}
  • Kept edges avg: {removal_stats['kept_uncertainty_mean']:.4f}
    """
    axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    # Fix: Use cfg.dataset if it's a string, otherwise use cfg.dataset.name
    dataset_name = cfg.dataset if isinstance(cfg.dataset, str) else cfg.dataset.name
    save_path = os.path.join(save_dir, f'comparison_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparison visualization to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Noisy Edge Detection using Uncertainty')
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Name of the dataset (default: cora)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='Name of the LLM model (default: llama_3.2_1B)')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='Type of PEFT (default: lora)')
    parser.add_argument('--init_weight_approach', type=str, default='loftq',
                        choices=['pissa', 'orthogonal', 'guassian', 'loftq', 'pissa'],
                        help='Initialization weight approach (default: pissa)')
    parser.add_argument('--num_models', type=int, default=10,
                        help='Number of edge predictor models to train (default: 10)')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['dot_product', 'mlp', 'bilinear'],
                        help='Edge predictor decoder type (default: mlp)')
    parser.add_argument('--uncertainty_metric', type=str, default='std',
                        choices=['std', 'entropy', 'both'],
                        help='Uncertainty metric to use (default: std)')
    parser.add_argument('--removal_strategy', type=str, default='top_k',
                        choices=['threshold', 'top_k'],
                        help='Edge removal strategy (default: top_k)')
    parser.add_argument('--threshold', type=float, default=20,
                        help='Threshold for edge removal (for threshold: uncertainty value, for top_k: percentage, default: 10.0)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("NOISY EDGE DETECTION USING UNCERTAINTY")
    print("="*80)
    print(f"Dataset: {args.dataset_name}")
    print(f"LLM: {args.llm_name}")
    print(f"Init approach: {args.init_weight_approach}")
    print(f"Number of models: {args.num_models}")
    print(f"Decoder type: {args.decoder_type}")
    print(f"Uncertainty metric: {args.uncertainty_metric}")
    print(f"Removal strategy: {args.removal_strategy}")
    print(f"Threshold: {args.threshold}")
    print("="*80 + "\n")
    
    # Setup configuration
    cfg = setup_finetuning_cfg(args.dataset_name, args.llm_name, args.peft_type)
    gnn_cfg = set_mgnn_cfg(args.dataset_name)
    
    # Load dataset and features
    dataset = load_dataset(cfg)
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    
    embedding = None
    if args.init_weight_approach == "pissa":
        embedding = get_embedding_from_data(data_pissa)
    elif args.init_weight_approach == "orthogonal":
        embedding = get_embedding_from_data(data_orthogonal)
    elif args.init_weight_approach == "guassian":
        embedding = get_embedding_from_data(data_guassian)
    elif args.init_weight_approach == "loftq":
        embedding = get_embedding_from_data(data_loftq)
    elif args.init_weight_approach == "eva":
        embedding = get_embedding_from_data(data_eva)
    
    dataset.x = embedding
    
    # Create detector first so we use consistent dataset
    detector = NoisyEdgeDetector(
        cfg=gnn_cfg,
        features=embedding,
        num_models=args.num_models,
        decoder_type=args.decoder_type,
        training_strategy='with_splitting',
        negative_sampling_ratio=1.0
    )
    
    # Step 1: Train baseline GNN on detector's dataset (for consistency)
    print("\n" + "="*80)
    print("STEP 1: Training Baseline GNN (Original Dataset)")
    print("="*80)
    print(f"[INFO] Using detector's dataset with {detector.original_num_edges} edges")
    
    original_results, original_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=detector.dataset,
        embedding=detector.features,
        dataset_name=args.dataset_name
    )
    
    # Step 2: Train multiple edge predictors to detect noisy edges
    print("\n" + "="*80)
    print("STEP 2: Training Edge Predictors for Uncertainty Estimation")
    print("="*80)
    
    # Detector was already created above, just train it
    detector.train_edge_predictors()
    
    # Step 3: Calculate uncertainty scores
    print("\n" + "="*80)
    print("STEP 3: Calculating Uncertainty Scores")
    print("="*80)
    
    uncertainty_scores = detector.get_uncertainty_scores(metric=args.uncertainty_metric)
    detector.visualize_uncertainty(uncertainty_scores, metric=args.uncertainty_metric)
    
    # Step 4: Remove noisy edges based on uncertainty threshold
    print("\n" + "="*80)
    print("STEP 4: Removing Noisy Edges")
    print("="*80)
    
    # IMPORTANT: Use detector.dataset (not the main script's dataset)
    # because uncertainty_scores were computed for detector's dataset
    modified_dataset, removal_stats = create_modified_dataset(
        dataset=detector.dataset,  # ← Use detector's dataset!
        uncertainty_scores=uncertainty_scores,
        threshold=args.threshold,
        removal_strategy=args.removal_strategy
    )
    
    # Step 5: Train GNN on modified dataset
    print("\n" + "="*80)
    print("STEP 5: Training GNN on Modified Dataset")
    print("="*80)
    
    modified_results, modified_model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=modified_dataset,
        embedding=detector.features,  # Use detector's features for consistency
        dataset_name=args.dataset_name
    )
    
    # Step 6: Compare and report results
    print("\n" + "="*80)
    print("STEP 6: Comparing Results")
    print("="*80)
    
    compare_results(original_results, modified_results, removal_stats, cfg)
    
    # Save results to JSON
    results_dir = "./results/edge_uncertainty"
    os.makedirs(results_dir, exist_ok=True)
    
    results_summary = {
        'dataset': args.dataset_name,
        'llm_name': args.llm_name,
        'init_weight_approach': args.init_weight_approach,
        'num_models': args.num_models,
        'decoder_type': args.decoder_type,
        'uncertainty_metric': args.uncertainty_metric,
        'removal_strategy': args.removal_strategy,
        'threshold': args.threshold,
        'original_results': original_results,
        'modified_results': modified_results,
        'removal_stats': removal_stats,
        'improvement': {
            'test_acc': modified_results['test_acc'] - original_results['test_acc'],
            'test_f1': modified_results['test_f1'] - original_results['test_f1'],
            'val_acc': modified_results['val_acc'] - original_results['val_acc'],
            'val_f1': modified_results['val_f1'] - original_results['val_f1']
        }
    }
    
    results_path = os.path.join(results_dir, f"{args.dataset_name}_{args.llm_name}_{args.uncertainty_metric}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n[OK] Results saved to: {results_path}")
    print("\n" + "="*80)
    print("NOISY EDGE DETECTION COMPLETE!")
    print("="*80)
