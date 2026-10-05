# Implementation of  Edge Predictor
# Load Initial Features for Gnn Encoder
# Load Graph Dataset
# create Edge Dataset with negetive sampling
# create decoder for predicting with three diffrent way
## dot product
## mlp
## bilinear
# LinkPrediction Trainer
# i want to two type training
# training with train test split edges data mean i first apporach i perform splitting
# in second approaches i need to train in whole data all of the data contibute in trainign
# in training use Binary Cross Entropy for Loss function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os
import sys
import json
from pathlib import Path
from yacs.config import CfgNode as CN
from common import GNNEncoder, set_seed
from common import load_graph_dataset_for_tape
from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import GNNTrainer, set_mgnn_cfg, get_datasets_path
from copy import deepcopy

class EdgeDecoder(nn.Module):
    """Decoder for edge prediction with three different approaches."""
    def __init__(self, hidden_dim, decoder_type='dot_product'):
        super(EdgeDecoder, self).__init__()
        self.decoder_type = decoder_type
        self.hidden_dim = hidden_dim
        
        if decoder_type == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif decoder_type == 'bilinear':
            self.decoder = nn.Bilinear(hidden_dim, hidden_dim, 1)
        elif decoder_type == 'dot_product':
            # Dot product doesn't need parameters
            self.decoder = None
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. Use 'dot_product', 'mlp', or 'bilinear'")
    
    def forward(self, z_src, z_dst):
        """
        Predict edge probability between source and destination nodes.
        
        Args:
            z_src: Source node embeddings [num_edges, hidden_dim]
            z_dst: Destination node embeddings [num_edges, hidden_dim]
        
        Returns:
            edge_scores: Edge prediction scores [num_edges, 1]
        """
        if self.decoder_type == 'dot_product':
            return (z_src * z_dst).sum(dim=1, keepdim=True)
        elif self.decoder_type == 'mlp':
            z_concat = torch.cat([z_src, z_dst], dim=1)
            return self.decoder(z_concat)
        elif self.decoder_type == 'bilinear':
            return self.decoder(z_src, z_dst)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")


class EdgePredictorModel(nn.Module):
    """Complete model: GNN Encoder + Edge Decoder"""
    def __init__(self, encoder, decoder):
        super(EdgePredictorModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Forward pass for edge prediction.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Graph structure for GNN [2, num_edges]
            edge_label_index: Edges to predict [2, num_edges_to_predict]
        
        Returns:
            edge_scores: Prediction scores for edges [num_edges_to_predict]
        """
        # Get node embeddings from GNN encoder
        z = self.encoder(x, edge_index)  # [num_nodes, hidden_dim]
        
        # Get embeddings for source and destination nodes
        z_src = z[edge_label_index[0]]  # [num_edges_to_predict, hidden_dim]
        z_dst = z[edge_label_index[1]]  # [num_edges_to_predict, hidden_dim]
        
        # Predict edge scores
        edge_scores = self.decoder(z_src, z_dst)  # [num_edges_to_predict, 1]
        return edge_scores.squeeze()  # [num_edges_to_predict]


class EdgePredictor(GNNTrainer):
    def __init__(self, cfg, features, decoder_approach_type=None, training_strategy=None, 
                 negative_sampling_ratio=None, modified_dataset=None):
        """
        Initialize Edge Predictor.
        
        Args:
            cfg: Configuration object (can contain decoder_approach_type, training_strategy, negative_sampling_ratio)
            features: Node features/embeddings [num_nodes, feature_dim]
            decoder_approach_type: 'dot_product', 'mlp', or 'bilinear' (defaults to cfg.decoder_approach_type)
            training_strategy: 'with_splitting' or 'without_splitting' (defaults to cfg.training_strategy)
            negative_sampling_ratio: Ratio of negative to positive edges (defaults to cfg.negative_sampling_ratio)
            modified_dataset: Optional modified dataset with different edge_index
        """
        # Initialize parent class (GNNTrainer) but we'll override some attributes
        # We need to call super().__init__ but we'll modify the model creation
        # Use cfg values if parameters are not provided
        self.decoder_approach_type = decoder_approach_type if decoder_approach_type is not None else getattr(cfg, 'decoder_approach_type', 'dot_product')
        self.training_strategy = training_strategy if training_strategy is not None else getattr(cfg, 'training_strategy', 'with_splitting')
        self.negative_sampling_ratio = negative_sampling_ratio if negative_sampling_ratio is not None else getattr(cfg, 'negative_sampling_ratio', 1.0)
        
        # Initialize parent to get dataset and basic setup
        super().__init__(cfg, features, modified_dataset=modified_dataset, does_print_training_process=False)
        
        # Override model creation - we need encoder + decoder
        self.hidden_dim = cfg.hidden_dim
        self.encoder = GNNEncoder(
            input_dim=self.features.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,  # Output dimension for embeddings
            n_layers=self.num_layers,
            gnn_type=self.gnn_model_name,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self.device)
        
        # Create decoder
        self.decoder = EdgeDecoder(self.hidden_dim, decoder_type=decoder_approach_type).to(self.device)
        
        # Create complete model
        self.model = EdgePredictorModel(self.encoder, self.decoder).to(self.device)
        
        # Recreate optimizer with new model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Create edge dataset
        self.train_edge_index = None
        self.val_edge_index = None
        self.test_edge_index = None
        self.train_edge_label = None
        self.val_edge_label = None
        self.test_edge_label = None
        
        self.create_edge_predictor_dataset()
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of Edge Predictor parameters: {trainable_params}")
        print(f"Decoder type: {decoder_approach_type}")
        print(f"Training strategy: {training_strategy}")

    def create_edge_predictor_model(self):
        """Create edge predictor model (already done in __init__)."""
        return self.model

    def create_edge_predictor_dataset(self):
        """Create edge dataset with negative sampling."""
        edge_index = self.data.edge_index
        
        if self.training_strategy == 'with_splitting':
            # Split edges into train/val/test
            data = Data(edge_index=edge_index, num_nodes=self.data.num_nodes)
            data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
            
            # Get positive edges
            train_pos_edge_index = data.train_pos_edge_index
            val_pos_edge_index = data.val_pos_edge_index
            test_pos_edge_index = data.test_pos_edge_index
            
            # Generate negative edges
            train_neg_edge_index = negative_sampling(
                edge_index=train_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=train_pos_edge_index.size(1) * int(self.negative_sampling_ratio)
            )
            val_neg_edge_index = negative_sampling(
                edge_index=val_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=val_pos_edge_index.size(1) * int(self.negative_sampling_ratio)
            )
            test_neg_edge_index = negative_sampling(
                edge_index=test_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=test_pos_edge_index.size(1) * int(self.negative_sampling_ratio)
            )
            
            # Combine positive and negative edges
            self.train_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1)
            self.val_edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim=1)
            self.test_edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=1)
            
            # Create labels (1 for positive, 0 for negative)
            self.train_edge_label = torch.cat([
                torch.ones(train_pos_edge_index.size(1)),
                torch.zeros(train_neg_edge_index.size(1))
            ]).to(self.device)
            self.val_edge_label = torch.cat([
                torch.ones(val_pos_edge_index.size(1)),
                torch.zeros(val_neg_edge_index.size(1))
            ]).to(self.device)
            self.test_edge_label = torch.cat([
                torch.ones(test_pos_edge_index.size(1)),
                torch.zeros(test_neg_edge_index.size(1))
            ]).to(self.device)
            
            print(f"Edge dataset created with splitting:")
            print(f"  Train edges: {self.train_edge_index.size(1)} (pos: {train_pos_edge_index.size(1)}, neg: {train_neg_edge_index.size(1)})")
            print(f"  Val edges: {self.val_edge_index.size(1)} (pos: {val_pos_edge_index.size(1)}, neg: {val_neg_edge_index.size(1)})")
            print(f"  Test edges: {self.test_edge_index.size(1)} (pos: {test_pos_edge_index.size(1)}, neg: {test_neg_edge_index.size(1)})")
            
        else:  # without_splitting
            # Use all edges for training, generate negatives
            all_pos_edge_index = edge_index
            all_neg_edge_index = negative_sampling(
                edge_index=all_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=all_pos_edge_index.size(1) * int(self.negative_sampling_ratio)
            )
            
            # Combine positive and negative edges
            self.train_edge_index = torch.cat([all_pos_edge_index, all_neg_edge_index], dim=1)
            self.train_edge_label = torch.cat([
                torch.ones(all_pos_edge_index.size(1)),
                torch.zeros(all_neg_edge_index.size(1))
            ]).to(self.device)
            
            # For evaluation, we can use a subset
            # Create a small validation set from training data
            num_val_samples = int(self.train_edge_index.size(1) * 0.1)
            val_indices = torch.randperm(self.train_edge_index.size(1))[:num_val_samples]
            self.val_edge_index = self.train_edge_index[:, val_indices]
            self.val_edge_label = self.train_edge_label[val_indices]
            
            # Test set is same as validation for this strategy
            self.test_edge_index = self.val_edge_index.clone()
            self.test_edge_label = self.val_edge_label.clone()
            
            print(f"Edge dataset created without splitting:")
            print(f"  Train edges: {self.train_edge_index.size(1)} (pos: {all_pos_edge_index.size(1)}, neg: {all_neg_edge_index.size(1)})")
            print(f"  Val/Test edges: {self.val_edge_index.size(1)}")

    def train_edge_predictor_with_spliting(self):
        """Train edge predictor with train/val/test split."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Use training edges
        edge_scores = self.model(self.features, self.data.edge_index, self.train_edge_index)
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(edge_scores, self.train_edge_label.float())
        
        loss.backward()
        self.optimizer.step()
        
        return float(loss)

    def training_edge_predictor_without_spliting(self):
        """Train edge predictor on all data without splitting."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Use all edges for training
        edge_scores = self.model(self.features, self.data.edge_index, self.train_edge_index)
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(edge_scores, self.train_edge_label.float())
        
        loss.backward()
        self.optimizer.step()
        
        return float(loss)

    def _train(self):
        """Internal training method."""
        if self.training_strategy == 'with_splitting':
            return self.train_edge_predictor_with_spliting()
        else:
            return self.training_edge_predictor_without_spliting()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate edge predictor."""
        self.model.eval()
        
        # Evaluate on validation set
        val_scores = self.model(self.features, self.data.edge_index, self.val_edge_index)
        val_preds = (torch.sigmoid(val_scores) > 0.5).float()
        val_acc = (val_preds == self.val_edge_label.float()).float().mean()
        val_loss = F.binary_cross_entropy_with_logits(val_scores, self.val_edge_label.float())
        
        # Evaluate on test set
        test_scores = self.model(self.features, self.data.edge_index, self.test_edge_index)
        test_preds = (torch.sigmoid(test_scores) > 0.5).float()
        test_acc = (test_preds == self.test_edge_label.float()).float().mean()
        test_loss = F.binary_cross_entropy_with_logits(test_scores, self.test_edge_label.float())
        
        return {
            'val_acc': float(val_acc),
            'val_loss': float(val_loss),
            'test_acc': float(test_acc),
            'test_loss': float(test_loss)
        }

    def train(self, show_plots=False):
        """Train the edge predictor model."""
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_test_loss = float('inf')
        counter = 0
        timer = []
        
        # Initialize history tracking
        history = {
            'loss': [],
            'val_acc': [],
            'val_loss': [],
            'test_acc': [],
            'test_loss': []
        }
        
        for epoch in range(1, 1 + self.epochs):
            loss = self._train()
            eval_results = self.evaluate()
            
            # Track metrics
            history['loss'].append(loss)
            history['val_acc'].append(eval_results['val_acc'])
            history['val_loss'].append(eval_results['val_loss'])
            history['test_acc'].append(eval_results['test_acc'])
            history['test_loss'].append(eval_results['test_loss'])
            
            if eval_results['val_acc'] > best_val_acc:
                best_val_acc = eval_results['val_acc']
                best_test_acc = eval_results['test_acc']
                best_test_loss = eval_results['test_loss']
                counter = 0
                self.best_model = deepcopy(self.model)
            else:
                counter += 1
            
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} Loss {loss:.4f}  "
                    f"Val Acc {eval_results['val_acc']:.4f} Val Loss {eval_results['val_loss']:.4f}  "
                    f"Test Acc {eval_results['test_acc']:.4f} Test Loss {eval_results['test_loss']:.4f}"
                )
            
            # Early stopping
            if counter >= self.patience:
                break
        
        print(f'\nBest Val Acc {best_val_acc:.4f}  Best Test Acc {best_test_acc:.4f}  Best Test Loss {best_test_loss:.4f}\n')
        self.model = self.best_model
        
        # Plot learning curves
        self.plot_learning_curves(history, show=show_plots)
        
        return {
            "val_acc": best_val_acc,
            "test_acc": best_test_acc,
            "test_loss": best_test_loss
        }, history

    def plot_learning_curves(self, history, save_dir="../../results/TAPE/edge_predictor", show=True):
        """Plot and save learning curves."""
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Loss
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        axes[0].plot(epochs, history['test_loss'], 'g-', linewidth=2, label='Test Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[1].plot(epochs, history['val_acc'], 'b-', linewidth=2, label='Val Acc')
        axes[1].plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Test Acc')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Combined view
        ax3_twin = axes[2].twinx()
        axes[2].plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        ax3_twin.plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Test Acc')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12, color='b')
        ax3_twin.set_ylabel('Accuracy', fontsize=12, color='r')
        axes[2].set_title(f'Loss vs Accuracy', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='y', labelcolor='b')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        axes[2].legend(loc='upper left', fontsize=10)
        ax3_twin.legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{self.dataset_name}_seed{self.seed}_{self.decoder_approach_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        print(f"Learning curves saved to: {save_path}")
        plt.close()

def set_link_prediction_cfg(datset_name='cora'):
    """
    Build a yacs CfgNode for link prediction training. This function will try to load
    a JSON hyperparameter file from the `link_prediction_hyperparameters/` folder.

    The returned cfg contains fields consumed by `EdgePredictor`:
      - seed, device, dataset, gnn_model_name, llm_name
      - hidden_dim, num_layers, dropout, lr, epochs, early_stop
      - batch_norm, weight_decay, re_split
      - decoder_approach_type, training_strategy, negative_sampling_ratio (optional)

    If no JSON is found, sensible defaults are used.
    """
    
    # Find candidate paths for the hyperparameter JSON
    candidates = [
        Path(__file__).resolve().parents[0] / 'link_prediction_hyperparameters' / f"{datset_name}.json",
    ]
    
    hp = {}
    for c in candidates:
        if c.exists():
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    hp = json.load(f)
                print(f"Loaded link prediction hyperparameters from: {c}")
                break
            except Exception as e:
                print(f"Failed to load {c}: {e}")
    
    # Defaults
    defaults = {
        'seed': 42,
        'device': 0,
        'dataset': datset_name,
        'gnn_model_name': 'SAGE',
        'llm_name': 'llama_3.2_1B',
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'lr': 0.05,
        'epochs': 200,
        'early_stop': 20,
        'batch_norm': 1,
        'weight_decay': 0.1,
        're_split': 1,
        'decoder_approach_type': 'dot_product',  # 'dot_product', 'mlp', 'bilinear'
        'training_strategy': 'with_splitting',  # 'with_splitting', 'without_splitting'
        'negative_sampling_ratio': 1.0,  # Ratio of negative to positive edges
    }
    
    # Merge loaded hyperparams with defaults
    for k, v in defaults.items():
        defaults[k] = hp.get(k, v)
    
    cfg = CN()
    cfg.seed = int(defaults['seed'])
    cfg.device = int(defaults['device'])
    cfg.dataset = defaults['dataset']
    cfg.gnn_model_name = defaults['gnn_model_name']
    cfg.llm_name = defaults['llm_name']
    cfg.hidden_dim = int(defaults['hidden_dim'])
    cfg.num_layers = int(defaults['num_layers'])
    cfg.dropout = float(defaults['dropout'])
    cfg.lr = float(defaults['lr'])
    cfg.epochs = int(defaults['epochs'])
    cfg.early_stop = int(defaults['early_stop'])
    cfg.batch_norm = bool(int(defaults['batch_norm']))
    cfg.weight_decay = float(defaults['weight_decay'])
    cfg.re_split = bool(int(defaults.get('re_split', 1)))
    cfg.decoder_approach_type = defaults.get('decoder_approach_type', 'dot_product')
    cfg.training_strategy = defaults.get('training_strategy', 'with_splitting')
    cfg.negative_sampling_ratio = float(defaults.get('negative_sampling_ratio', 1.0))
    
    return cfg

def edge_predictor_train_and_report(dataset_name, features, decoder_approach_type='dot_product', 
                                     training_strategy='with_splitting', negative_sampling_ratio=1.0,
                                     title="", does_print_training_process=False, reporter=None):
    """
    Train an edge predictor model on the given dataset using provided features and report results.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'cora', 'pubmed')
        features (torch.Tensor): Node embeddings to use as features
        decoder_approach_type (str): 'dot_product', 'mlp', or 'bilinear'
        training_strategy (str): 'with_splitting' or 'without_splitting'
        negative_sampling_ratio (float): Ratio of negative to positive edges
        title (str): Title for reporting (e.g., 'PISSA', 'ORTHOGONAL')
        does_print_training_process (bool): Whether to print training progress
        reporter: Reporter object for writing results to file
    
    Returns:
        dict: Training results containing val_acc, test_acc, test_loss
        model: Trained edge predictor model
    """
    print(f"\n{'='*80}")
    print(f"Training Edge Predictor on {dataset_name.upper()} dataset")
    print(f"Embedding shape: {features.shape}")
    print(f"Decoder type: {decoder_approach_type}")
    print(f"Training strategy: {training_strategy}")
    print(f"{'='*80}\n")
    
    # Setup link prediction configuration for the dataset
    cfg = set_link_prediction_cfg(dataset_name)
    
    # Use cfg values if parameters are not provided
    decoder_approach_type = decoder_approach_type if decoder_approach_type is not None else getattr(cfg, 'decoder_approach_type', 'dot_product')
    training_strategy = training_strategy if training_strategy is not None else getattr(cfg, 'training_strategy', 'with_splitting')
    negative_sampling_ratio = negative_sampling_ratio if negative_sampling_ratio is not None else getattr(cfg, 'negative_sampling_ratio', 1.0)
    
    # Initialize Edge Predictor trainer with the embeddings
    trainer = EdgePredictor(
        cfg, 
        features, 
        decoder_approach_type=decoder_approach_type,
        training_strategy=training_strategy,
        negative_sampling_ratio=negative_sampling_ratio,
        modified_dataset=None
    )
    
    # Train the model and get results
    results, history = trainer.train(show_plots=False)
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"Final Results for {dataset_name.upper()}:")
    print(f"  Validation Accuracy: {results['val_acc']:.4f}")
    print(f"  Test Accuracy:       {results['test_acc']:.4f}")
    print(f"  Test Loss:           {results['test_loss']:.4f}")
    print(f"{'='*80}\n")
    
    # Report results using reporter
    if reporter is not None:
        report_title = f"Edge Predictor Training Results - {title}" if title else f"Edge Predictor Training Results - {dataset_name.upper()}"
        report_text = f"""
                        Final Results:
                        • Validation Accuracy: {results['val_acc']:.4f} ({results['val_acc']:.2%})
                        • Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']:.2%})
                        • Test Loss: {results['test_loss']:.4f}
                        • Decoder Type: {decoder_approach_type}
                        • Training Strategy: {training_strategy}
                        """
        reporter.report(report_title, report_text)
    
    return results, trainer.model


if __name__ == '__main__':
    from config import setup_finetuning_cfg
    
    dataset_name = 'cora'
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name='llama_3.2_1B', peft_type='lora')
    
    data_pissa, data_orthogonal, data_gaussian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    
    emb_pissa = get_embedding_from_data(data_pissa)
    # emb_orthogonal = get_embedding_from_data(data_orthogonal)
    # emb_loftq = get_embedding_from_data(data_loftq)
    # emb_eva = get_embedding_from_data(data_eva)
    # emb_gaussian = get_embedding_from_data(data_gaussian)
    
    results_pissa, model_pissa = edge_predictor_train_and_report(
        dataset_name, 
        emb_pissa, 
        decoder_approach_type='mlp',
        training_strategy='with_splitting',
        title="PISSA",
        does_print_training_process=True
    )
