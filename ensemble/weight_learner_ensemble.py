import torch
from torch import nn
import torch.nn.functional as F
import sys
import os

class WeightLearner(nn.Module):
    def __init__(self, num_models, num_classes=None, hidden_dim=64, num_layers=1, dropout=0.5):
        """
        Weight learner with configurable depth.

        Args:
            num_models: Number of models to ensemble
            num_classes: Number of output classes (if None, learns scalar weights)
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of layers (1 = simple scalar weights, >1 = deep MLP)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.num_layers = num_layers

        if num_layers == 1:
            # Simple scalar weights (original approach)
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
            self.use_mlp = False
        else:
            # Deep MLP to learn weights
            self.use_mlp = True

            if num_classes is None:
                raise ValueError("num_classes must be specified for multi-layer WeightLearner")

            # Input: concatenated predictions from all models
            input_dim = num_models * num_classes

            layers = []
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer - outputs num_models weights
            layers.append(nn.Linear(hidden_dim, num_models))

            self.mlp = nn.Sequential(*layers)

    def forward(self, *predictions):
        """
        Args:
            *predictions: Variable number of prediction tensors [batch_size, num_classes]

        Returns:
            ensemble_pred: Weighted ensemble prediction [batch_size, num_classes]
            normalized_weights: Learned weights (either scalar or per-sample)
        """
        if not self.use_mlp:
            # Simple scalar weights approach
            normalized_weights = F.softmax(self.weights, dim=0)

            # Weighted combination
            ensemble_pred = sum(w * pred for w, pred in zip(normalized_weights, predictions))
            return ensemble_pred, normalized_weights
        else:
            # MLP-based approach: learn weights conditioned on inputs
            batch_size = predictions[0].shape[0]

            # Concatenate all predictions
            concat_preds = torch.cat(predictions, dim=-1)  # [batch_size, num_models * num_classes]

            # Get weights from MLP
            weight_logits = self.mlp(concat_preds)  # [batch_size, num_models]
            normalized_weights = F.softmax(weight_logits, dim=-1)  # [batch_size, num_models]

            # Weighted combination (per-sample weights)
            ensemble_pred = torch.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += normalized_weights[:, i:i + 1] * pred

            # Return mean weights for logging
            mean_weights = normalized_weights.mean(dim=0)
            return ensemble_pred, mean_weights


class WeightLearnerPerClass(nn.Module):
    def __init__(self, num_models, num_classes, hidden_dim=64, num_layers=1, dropout=0.5):
        """
        Weight learner that learns class-specific weights for each model.

        Args:
            num_models: Number of models to ensemble
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of layers (1 = simple weight matrix, >1 = deep MLP)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.num_layers = num_layers

        if num_layers == 1:
            # Simple learnable weight matrix [num_models, num_classes]
            self.weights = nn.Parameter(torch.ones(num_models, num_classes) / num_models)
            self.use_mlp = False
        else:
            # Deep MLP to learn class-specific weights
            self.use_mlp = True

            # Input: concatenated predictions from all models
            input_dim = num_models * num_classes

            # Output: weight matrix [num_models * num_classes]
            output_dim = num_models * num_classes

            layers = []
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer - outputs num_models * num_classes weights
            layers.append(nn.Linear(hidden_dim, output_dim))

            self.mlp = nn.Sequential(*layers)

    def forward(self, *predictions):
        """
        Args:
            *predictions: Variable number of prediction tensors [batch_size, num_classes]

        Returns:
            ensemble_pred: Weighted ensemble prediction [batch_size, num_classes]
            normalized_weights: Learned weight matrix [num_models, num_classes]
        """
        batch_size = predictions[0].shape[0]

        if not self.use_mlp:
            # Simple weight matrix approach
            # Apply softmax over models dimension for each class
            normalized_weights = F.softmax(self.weights, dim=0)  # [num_models, num_classes]

            # Weighted combination
            ensemble_pred = torch.zeros_like(predictions[0])
            for model_idx, pred in enumerate(predictions):
                ensemble_pred += pred * normalized_weights[model_idx].unsqueeze(0)

            return ensemble_pred, normalized_weights
        else:
            # MLP-based approach: learn weights conditioned on inputs
            # Concatenate all predictions
            concat_preds = torch.cat(predictions, dim=-1)  # [batch_size, num_models * num_classes]

            # Get weight logits from MLP
            weight_logits = self.mlp(concat_preds)  # [batch_size, num_models * num_classes]

            # Reshape to [batch_size, num_models, num_classes]
            weight_logits = weight_logits.view(batch_size, self.num_models, self.num_classes)

            # Apply softmax over models dimension for each class
            normalized_weights = F.softmax(weight_logits, dim=1)  # [batch_size, num_models, num_classes]

            # Weighted combination (per-sample, per-class weights)
            ensemble_pred = torch.zeros_like(predictions[0])
            for model_idx, pred in enumerate(predictions):
                # pred: [batch_size, num_classes]
                # normalized_weights[:, model_idx, :]: [batch_size, num_classes]
                ensemble_pred += pred * normalized_weights[:, model_idx, :]

            # Return mean weights across batch for logging
            mean_weights = normalized_weights.mean(dim=0)  # [num_models, num_classes]
            return ensemble_pred, mean_weights


class PerClassMLPEnsemble(nn.Module):
    def __init__(self, num_models, num_classes, hidden_dim=64, num_layers=1, dropout=0.5):
        """
        Ensemble learner with separate MLP for each class.
        Each class has its own MLP that learns how to combine predictions from different models.

        Args:
            num_models: Number of models to ensemble
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of layers (1 = simple weights per class, >1 = deep MLP per class)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.num_layers = num_layers

        if num_layers == 1:
            # Simple learnable weights: one weight per model for each class
            # Shape: [num_classes, num_models]
            self.weights = nn.Parameter(torch.ones(num_classes, num_models) / num_models)
            self.use_mlp = False
        else:
            # Create separate MLP for each class
            self.use_mlp = True
            self.class_mlps = nn.ModuleList()

            for _ in range(num_classes):
                layers = []
                # Input: predictions from all models for this class [num_models]
                input_dim = num_models

                # First layer
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

                # Hidden layers
                for _ in range(num_layers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))

                # Output layer - outputs num_models weights for this class
                layers.append(nn.Linear(hidden_dim, num_models))

                self.class_mlps.append(nn.Sequential(*layers))

    def forward(self, *predictions):
        """
        Args:
            *predictions: Variable number of prediction tensors [batch_size, num_classes]

        Returns:
            ensemble_pred: Weighted ensemble prediction [batch_size, num_classes]
            normalized_weights: Learned weights [num_classes, num_models]
        """
        batch_size = predictions[0].shape[0]
        ensemble_pred = torch.zeros_like(predictions[0])

        if not self.use_mlp:
            # Simple weight approach
            # Apply softmax over models dimension for each class
            normalized_weights = F.softmax(self.weights, dim=1)  # [num_classes, num_models]

            # For each class, combine predictions using class-specific weights
            for class_idx in range(self.num_classes):
                class_weights = normalized_weights[class_idx]  # [num_models]
                for model_idx, pred in enumerate(predictions):
                    # pred[:, class_idx]: [batch_size] - predictions for this class
                    ensemble_pred[:, class_idx] += class_weights[model_idx] * pred[:, class_idx]

            return ensemble_pred, normalized_weights
        else:
            # MLP-based approach: each class has its own MLP
            # Store weights for all classes
            all_weights = []

            for class_idx in range(self.num_classes):
                # Collect predictions for this class from all models
                class_preds = torch.stack([pred[:, class_idx] for pred in predictions],
                                          dim=1)  # [batch_size, num_models]

                # Get weights from this class's MLP
                weight_logits = self.class_mlps[class_idx](class_preds)  # [batch_size, num_models]
                normalized_weights = F.softmax(weight_logits, dim=1)  # [batch_size, num_models]

                # Combine predictions for this class using learned weights
                for model_idx, pred in enumerate(predictions):
                    ensemble_pred[:, class_idx] += normalized_weights[:, model_idx] * pred[:, class_idx]

                # Store mean weights for logging
                all_weights.append(normalized_weights.mean(dim=0))

            # Stack to [num_classes, num_models]
            mean_weights = torch.stack(all_weights, dim=0)
            return ensemble_pred, mean_weights