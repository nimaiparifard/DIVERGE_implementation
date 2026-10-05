import torch
from torch import nn
import torch.nn.functional as F
import os

from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from ensemble.weight_learner_ensemble import WeightLearner, WeightLearnerPerClass, PerClassMLPEnsemble
from common import load_graph_dataset_for_tape, compute_acc_and_f1, set_seed
from config import setup_finetuning_cfg
from report.reporter import ReportResults
from visualize.visualize_gnn_mistakes import comprehensive_gnn_mistakes_analysis_visualization_list
from gnns.gnn_mtrainer import set_mgnn_cfg, gnn_train_and_report, get_datasets_path, append_gnn_result_csv
from train_llm.efficiency import ResourceMonitor
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier

# Weight approaches supported by ensemble_gnn_learning
ENSEMBLE_WEIGHT_APPROACHES = ['uniform', 'learnable', 'learnable_classes', 'learnable_per_classes']


def _get_path_prefix():
    """Resolve path prefix to repo root (same logic as gnns.gnn_mtrainer.GNNTrainer)."""
    datasets_path = get_datasets_path()
    repo_root = os.path.dirname(datasets_path)
    if os.path.exists(os.path.join(repo_root, 'datasets')):
        if os.path.abspath(os.getcwd()) == os.path.abspath(repo_root):
            return '.'
        return os.path.normpath(os.path.relpath(repo_root, start=os.getcwd()))
    return '../..'

def ensemble_gnn_learning(model_list: list, dataset_name, weight_approach='uniform', device='cuda:0',
                         num_layers=1, hidden_dim=64, dropout=0.5, learning_rate=0.01, epochs=200,
                         custom_embeddings: list =None, modified_datasets_list : list =None,
                          does_report_training_process: bool= False, predictions_list: list =None,
                          supervised: bool=True, seed=None, llm_name=None, select_split='val'):
    """
    Perform ensemble learning using multiple GNN models.
    
    Args:
        model_list: list of GNN encoder models (GNNEncoder from common); unused (may be None)
                    when predictions_list is given
        dataset_name: name of the dataset
        weight_approach: str
            'uniform' for equal weights
            'learnable' with learnable weights (supports multi-layer MLP)
            'learnable_classes' for class-specific weights per model (supports multi-layer MLP)
            'learnable_per_classes' for separate MLP per class (each class learns its own model weights)
        device: device to run on
        num_layers: number of layers for WeightLearner (1=scalar weights, >1=deep MLP)
        hidden_dim: hidden dimension for MLP layers (only used if num_layers > 1)
        dropout: dropout probability for MLP (only used if num_layers > 1)
        learning_rate: learning rate for weight optimization
        epochs: number of epochs to train the weight learner
        custom_embeddings: optional list of node embeddings to use for each model
                          (useful when models were trained on specific embeddings)
        modified_datasets_list: optional list of modified datasets (one per model)
                               If provided, uses each model's corresponding dataset for edge_index
        supervised: True Supervised setting False Semi Supervised Setting
        select_split: split whose accuracy picks the best training epoch of a learnable
                      head. 'val' (default) is the leakage-free protocol: test nodes are never
                      used for selection. 'test' leaks test labels into model selection and is
                      kept only to reproduce old runs; do not report numbers obtained with it.

        
    Returns:
        ensemble_predictions: Combined predictions from all models
        results: dict with accuracy, macro_f1, weighted_f1 for train/val/test
        weights: learned or uniform weights
        
    Note:
        - For 'learnable' with num_layers=1: learns simple scalar weights (one weight per model)
        - For 'learnable' with num_layers>1: learns per-sample weights using a deep MLP
        - For 'learnable_classes' with num_layers=1: learns weight matrix [num_models, num_classes]
        - For 'learnable_classes' with num_layers>1: learns per-sample class-specific weights using MLP
        - For 'learnable_per_classes' with num_layers=1: learns separate weights [num_classes, num_models]
        - For 'learnable_per_classes' with num_layers>1: each class has its own MLP to learn model weights
        - When modified_datasets_list is provided, each model uses its own edge_index
    """
    # Load dataset (for masks and labels)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name)
    set_seed(cfg.seed)
    path_prefix = _get_path_prefix()
    re_split = 1 if supervised else 0
    graph_data, num_classes, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=re_split, path_prefix=path_prefix, seed=cfg.split_seed)
    
    # Move all models to the correct device
    if predictions_list is None:
        for model in model_list:
            model.to(device)

        # Extract predictions from all models
        predictions_list = []
        with torch.no_grad():
            if custom_embeddings is not None and modified_datasets_list is not None:
                # Use different embeddings AND different datasets for each model
                print(f"Using custom embeddings and modified datasets for {len(model_list)} models")
                if isinstance(custom_embeddings, list) and isinstance(modified_datasets_list, list):
                    for idx, (model, emb, modified_data) in enumerate(zip(model_list, custom_embeddings, modified_datasets_list)):
                        model.eval()
                        emb = emb.to(device)
                        edge_index = modified_data.edge_index.to(device)
                        logits = model(emb, edge_index)
                        predictions_list.append(logits)
                        print(f"  Model {idx+1}: Using modified dataset with {edge_index.shape[1]} edges")
                else:
                    raise ValueError("Both custom_embeddings and modified_datasets_list must be lists of same length")

            elif custom_embeddings is not None:
                # Use different embeddings but same dataset for all models
                if isinstance(custom_embeddings, list):
                    print(f"Using custom embeddings for {len(model_list)} models (same edge_index)")
                    for model, emb in zip(model_list, custom_embeddings):
                        model.eval()
                        emb = emb.to(device)
                        logits = model(emb, graph_data.edge_index)
                        predictions_list.append(logits)
                # If custom_embeddings is a single tensor, use same embeddings for all models
                else:
                    embeddings = custom_embeddings.to(device)
                    for model in model_list:
                        model.eval()
                        logits = model(embeddings, graph_data.edge_index)
                        predictions_list.append(logits)
            else:
                # Use default graph_data.x for all models
                for model in model_list:
                    model.eval()
                    logits = model(graph_data.x, graph_data.edge_index)
                    predictions_list.append(logits)

    # When using modified_datasets_list with more nodes (e.g. augmented), use first modified dataset for masks/labels
    if predictions_list and modified_datasets_list and len(modified_datasets_list) > 0:
        n_pred = predictions_list[0].shape[0]
        n_graph = graph_data.y.shape[0]
        if n_pred != n_graph:
            first_modified = modified_datasets_list[0]
            graph_data.train_mask = first_modified.train_mask.to(device)
            graph_data.val_mask = first_modified.val_mask.to(device)
            graph_data.test_mask = first_modified.test_mask.to(device)
            graph_data.y = first_modified.y.to(device)
            num_classes = int(torch.unique(graph_data.y).numel())
            n_train = graph_data.train_mask.sum().item()
            n_val = graph_data.val_mask.sum().item()
            n_test = graph_data.test_mask.sum().item()
            print(f"Using masks/labels from modified dataset (n_nodes={n_pred}, original graph had n_nodes={n_graph})")
            print(f"  Modified dataset split — Train: {n_train}, Val: {n_val}, Test: {n_test}")

    num_models = len(predictions_list)  # model_list may be None when logits are precomputed
    if select_split not in ('val', 'test'):
        raise ValueError(f"select_split must be 'val' or 'test', got {select_split!r}")
    sel_mask = graph_data.val_mask if select_split == 'val' else graph_data.test_mask
    
    if weight_approach == 'uniform':
        # Uniform weights: simple average
        ensemble_logits = sum(predictions_list) / num_models
        weights = [1.0 / num_models] * num_models
        print(f"Using uniform weights: {weights}")
        
    elif weight_approach == 'learnable':
        # Learnable weights using MLP
        train_preds = [pred[graph_data.train_mask] for pred in predictions_list]
        target_labels = graph_data.y[graph_data.train_mask]
        
        # Initialize weight learner with configurable architecture
        weight_learner = WeightLearner(
            num_models=num_models,
            num_classes=num_classes if num_layers > 1 else None,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(weight_learner.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # For tracking best model
        best_sel_acc = 0.0
        best_ensemble_logits = None
        best_weights = None
        best_epoch = 0
        
        # Training loop
        # print(f"Training WeightLearner with {num_layers} layer(s)...")
        for epoch in range(epochs):
            # Train
            weight_learner.train()
            optimizer.zero_grad()
            
            # Get learned ensemble prediction
            ensemble_pred, current_weights = weight_learner(*train_preds)
            
            # Compute loss
            loss = criterion(ensemble_pred, target_labels)
            loss.backward()
            optimizer.step()
            
            # Evaluate on the selection split (validation by default)
            weight_learner.eval()
            with torch.no_grad():
                # Get predictions on full dataset
                eval_logits, eval_weights = weight_learner(*predictions_list)
                sel_preds = eval_logits[sel_mask].argmax(dim=1).cpu().numpy()
                sel_labels = graph_data.y[sel_mask].cpu().numpy()
                sel_acc, _, _ = compute_acc_and_f1(sel_preds, sel_labels)
                
                # Track the best epoch by accuracy on the selection split
                if sel_acc > best_sel_acc:
                    best_sel_acc = sel_acc
                    best_ensemble_logits = eval_logits.clone()
                    best_weights = eval_weights.clone()
                    best_epoch = epoch
            
            if epoch % max(1, epochs // 5) == 0 and does_report_training_process:  # Report 5 times during training
                print(f"Epoch {epoch}/{epochs}, loss: {loss.item():.4f}, {select_split}_acc: {sel_acc:.4f}, weights: {current_weights.detach().cpu().numpy()}")
        
        # Use best model
        ensemble_logits = best_ensemble_logits
        weights = best_weights.detach().cpu().numpy().tolist()
        
        print(f"\nBest model from epoch {best_epoch} with {select_split} accuracy: {best_sel_acc:.4f}")
        print(f"Best learned weights: {weights}")
    elif weight_approach == 'learnable_per_classes':
        # Separate MLP for each class that learns how to weight different models for that class
        train_preds = [pred[graph_data.train_mask] for pred in predictions_list]
        target_labels = graph_data.y[graph_data.train_mask]
        
        # Initialize per-class ensemble learner
        ensemble_learner = PerClassMLPEnsemble(
            num_models=num_models,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(ensemble_learner.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # For tracking best model
        best_sel_acc = 0.0
        best_ensemble_logits = None
        best_weights = None
        best_epoch = 0
        
        # Training loop
        architecture_info = f"{'Separate MLPs per class' if num_layers > 1 else 'Simple weights per class'} ({num_layers} layer(s))"
        if does_report_training_process:
            print(f"Training PerClassMLPEnsemble with {architecture_info}...")
        
        for epoch in range(epochs):
            # Train
            ensemble_learner.train()
            optimizer.zero_grad()
            
            # Get learned ensemble prediction
            ensemble_pred, current_weights = ensemble_learner(*train_preds)
            
            # Compute loss
            loss = criterion(ensemble_pred, target_labels)
            loss.backward()
            optimizer.step()
            
            # Evaluate on the selection split (validation by default)
            ensemble_learner.eval()
            with torch.no_grad():
                # Get predictions on full dataset
                eval_logits, eval_weights = ensemble_learner(*predictions_list)
                sel_preds = eval_logits[sel_mask].argmax(dim=1).cpu().numpy()
                sel_labels = graph_data.y[sel_mask].cpu().numpy()
                sel_acc, _, _ = compute_acc_and_f1(sel_preds, sel_labels)
                
                # Track the best epoch by accuracy on the selection split
                if sel_acc > best_sel_acc:
                    best_sel_acc = sel_acc
                    best_ensemble_logits = eval_logits.clone()
                    best_weights = eval_weights.clone()
                    best_epoch = epoch
            
            if epoch % max(1, epochs // 5) == 0 and does_report_training_process:
                print(f"Epoch {epoch}/{epochs}, loss: {loss.item():.4f}, {select_split}_acc: {sel_acc:.4f}")
                # Print weight statistics per class
                with torch.no_grad():
                    print(f"  Weight matrix shape: {current_weights.shape} [num_classes, num_models]")
                    for class_idx in range(min(3, num_classes)):  # Show first 3 classes
                        class_weights = current_weights[class_idx].cpu().numpy()
                        print(f"    Class {class_idx} weights: {class_weights}")
        
        # Use best model
        ensemble_logits = best_ensemble_logits
        weights = best_weights.detach().cpu().numpy()
        
        print(f"\nBest model from epoch {best_epoch} with {select_split} accuracy: {best_sel_acc:.4f}")
        print(f"Learned weight matrix shape: {weights.shape} [num_classes, num_models]")
        print(f"Each class learned separate weights for combining models")
        # Print weights for each class
        for class_idx in range(num_classes):
            print(f"  Class {class_idx} weights: {weights[class_idx]}")
        # Store as list for consistency with other approaches
        weights = weights.tolist()
    elif weight_approach == 'learnable_classes':
        # Learn a weight for each model and each class using MLP: num_models * num_classes weights
        train_preds = [pred[graph_data.train_mask] for pred in predictions_list]
        target_labels = graph_data.y[graph_data.train_mask]
        
        # Initialize weight learner with configurable MLP architecture
        weight_learner = WeightLearnerPerClass(
            num_models=num_models,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(weight_learner.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # For tracking best model
        best_sel_acc = 0.0
        best_ensemble_logits = None
        best_weights = None
        best_epoch = 0
        
        # Training loop
        architecture_info = f"{'MLP' if num_layers > 1 else 'Weight Matrix'} ({num_layers} layer(s))"
        if does_report_training_process:
            print(f"Training WeightLearnerPerClass with {architecture_info}...")
        
        for epoch in range(epochs):
            # Train
            weight_learner.train()
            optimizer.zero_grad()
            
            # Get learned ensemble prediction
            ensemble_pred, current_weights = weight_learner(*train_preds)
            
            # Compute loss
            loss = criterion(ensemble_pred, target_labels)
            loss.backward()
            optimizer.step()
            
            # Evaluate on the selection split (validation by default)
            weight_learner.eval()
            with torch.no_grad():
                # Get predictions on full dataset
                eval_logits, eval_weights = weight_learner(*predictions_list)
                sel_preds = eval_logits[sel_mask].argmax(dim=1).cpu().numpy()
                sel_labels = graph_data.y[sel_mask].cpu().numpy()
                sel_acc, _, _ = compute_acc_and_f1(sel_preds, sel_labels)
                
                # Track the best epoch by accuracy on the selection split
                if sel_acc > best_sel_acc:
                    best_sel_acc = sel_acc
                    best_ensemble_logits = eval_logits.clone()
                    best_weights = eval_weights.clone()
                    best_epoch = epoch
            
            if epoch % max(1, epochs // 5) == 0 and does_report_training_process:
                print(f"Epoch {epoch}/{epochs}, loss: {loss.item():.4f}, {select_split}_acc: {sel_acc:.4f}")
                # Print weight statistics
                with torch.no_grad():
                    weight_stats = current_weights.mean(dim=1).cpu().numpy()
                    print(f"  Avg weights per model: {weight_stats}")
        
        # Use best model
        ensemble_logits = best_ensemble_logits
        weights = best_weights.detach().cpu().numpy()
        
        print(f"\nBest model from epoch {best_epoch} with {select_split} accuracy: {best_sel_acc:.4f}")
        print(f"Learned weight matrix shape: {weights.shape} (num_models x num_classes)")
        print(f"Average weight per model: {weights.mean(axis=1)}")
        # Store as list for consistency with other approaches
        weights = weights.tolist() 
    else:
        raise ValueError(f"Unknown weight_approach: {weight_approach}. Use 'uniform', 'learnable', 'learnable_classes', or 'learnable_per_classes'.")
    
    # Get final predictions
    ensemble_predictions = ensemble_logits.argmax(dim=1)
    
    # Compute metrics for train, val, test
    results = {}
    masks = {
        'train': graph_data.train_mask,
        'val': graph_data.val_mask,
        'test': graph_data.test_mask
    }
    
    for split_name, mask in masks.items():
        pred = ensemble_predictions[mask].cpu().numpy()
        gt = graph_data.y[mask].cpu().numpy()
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(pred, gt)
        error_rate, error_number = compute_error_rate_and_number(pred, gt)
        results[split_name] = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'error_rate': error_rate,
            'error_number': error_number
        }
        print(f"[{split_name.upper()}] Acc: {acc:.2f}, Macro-F1: {macro_f1:.2f}, Weighted-F1: {weighted_f1:.2f}")
    
    return ensemble_predictions, results, weights

def compute_error_rate_and_number(pred, gt):
    """
    Compute error rate and error number for each class.
    
    Args:
        pred: array of predicted labels
        gt: array of ground truth labels
    
    Returns:
        error_rate: dict with error rate (percentage) per class
        error_number: dict with error count per class
    """
    import numpy as np
    
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    
    # Get unique classes from ground truth
    unique_classes = np.unique(gt)
    
    error_rate = {}
    error_number = {}
    
    for class_id in unique_classes:
        # Get mask for samples belonging to this class
        class_mask = gt == class_id
        
        # Count total samples in this class
        total_samples = class_mask.sum()
        
        if total_samples > 0:
            # Count mistakes in this class
            class_mistakes = ((pred[class_mask] != gt[class_mask]).sum())
            
            # Calculate error rate as percentage
            error_rate[class_id] = (class_mistakes / total_samples) * 100.0
            error_number[class_id] = int(class_mistakes)
        else:
            error_rate[class_id] = 0.0
            error_number[class_id] = 0
    
    return error_rate, error_number


def tune_ensemble_hyperparameter(dataset_name, model_list: list, custom_embeddings: list, modified_datasets_list: list =None,
                                 does_report_training_process: bool= False, title="", reporter=None,
                                 llm_name='llama_3.2_1B', peft_type='lora', title_list=None,
                                 visualize_best=True, ensemble_approaches = 'learnable',save_dir="results/gnn_mistakes_comprehensive", show=False
                                 , predictions_list: list =None, supervised=True, seed=None,
                                 select_split='val'):
    """
    Tune hyperparameters for ensemble GNN learning and report best results.
    After finding best hyperparameters, optionally visualize the best ensemble model comprehensively.
    
    Args:
        dataset_name: name of the dataset
        model_list: list of trained GNN models
        custom_embeddings: list of embeddings for each model
        modified_datasets_list: optional list of modified datasets (one per model)
        does_report_training_process: whether to print training progress
        title: title for reporting
        reporter: reporter object for saving results
        llm_name: name of LLM model (for visualization)
        peft_type: PEFT type (for visualization)
        title_list: list of titles for each model (for visualization)
        visualize_best: whether to generate comprehensive visualizations for best model
        save_dir: directory to save visualizations
        show: whether to display plots
        select_split: split whose accuracy selects the head configuration and its training
                      epoch. 'val' (default) is the leakage-free protocol: test nodes are never
                      used for selection. 'test' leaks test labels into model selection and is
                      kept only to reproduce old runs; do not report numbers obtained with it.
    
    Returns:
        best_results: dict with best performance metrics
        best_hyperparams: dict with best hyperparameters
        best_ensemble_predictions: predictions from best ensemble (if visualize_best=True)
    """
    best_results = {}
    best_hyperparams = {}
    best_ensemble_predictions = None

    # Define hyperparameter search space (REDUCED for stability and speed)
    # Start with most promising ranges based on typical ensemble learning
    num_layers_options = [2, 3]  # Reduced from [2, 3, 4, 5]
    hidden_dim_options = [64, 128, 256]  # Reduced from [16, 32, 64, 128, 256, 512]
    dropout_options = [0.3, 0.5]  # Reduced from [0.0, 0.3, 0.5, 0.7]
    learning_rate_options = [0.001, 0.01, 0.0001]  # Fixed duplicate: was [0.001, 0.01, 0.1, 0.0001, 0.0001]
    if ensemble_approaches == 'uniform':
        # Uniform averaging has no learnable weights: every combination gives the same result
        num_layers_options, hidden_dim_options, dropout_options, learning_rate_options = [2], [64], [0.3], [0.01]

    best_acc = -1.0
    total_combinations = len(num_layers_options) * len(hidden_dim_options) * len(dropout_options) * len(learning_rate_options)
    successful_runs = 0
    failed_runs = 0
    error_messages = []
    
    if modified_datasets_list is not None:
        print(f"Starting hyperparameter tuning with modified datasets ({len(modified_datasets_list)} models)...")
    else:
        print("Starting hyperparameter tuning with original dataset...")
    
    print(f"Total hyperparameter combinations to test: {total_combinations}")
    print(f"Hyperparameter search space:")
    print(f"  Layers: {num_layers_options}")
    print(f"  Hidden dims: {hidden_dim_options}")
    print(f"  Dropout: {dropout_options}")
    print(f"  Learning rates: {learning_rate_options}")
    print("="*80)

    # Cost of the whole search (every combination) and of fitting the selected config once
    tuning_monitor = ResourceMonitor().__enter__()
    best_fit_cost, config_times = None, []
    combination_idx = 0
    for num_layers in num_layers_options:
        for hidden_dim in hidden_dim_options:
            for dropout in dropout_options:
                for lr in learning_rate_options:
                    combination_idx += 1
                    print(f"\n[{combination_idx}/{total_combinations}] Testing: layers={num_layers}, hidden={hidden_dim}, dropout={dropout}, lr={lr}")

                    try:
                        with ResourceMonitor() as fit_monitor:
                            predictions, results, weights = ensemble_gnn_learning(
                                model_list=model_list,
                                dataset_name=dataset_name,
                                weight_approach=ensemble_approaches,
                                device='cuda:0',
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                                dropout=dropout,
                                learning_rate=lr,
                                epochs=200,  # Increased from 100 for better convergence
                                custom_embeddings=custom_embeddings,
                                modified_datasets_list=modified_datasets_list,
                                does_report_training_process=does_report_training_process,
                                predictions_list=predictions_list,
                                supervised=supervised,
                                seed=seed,
                                llm_name=llm_name,
                                select_split=select_split,
                            )

                        sel_acc = results[select_split]['accuracy']
                        successful_runs += 1
                        config_times.append(fit_monitor.time_sec)

                        print(f"  ✓ Success! {select_split} Acc: {sel_acc:.4f} | Test Acc: {results['test']['accuracy']:.4f}")
                        
                        if sel_acc > best_acc:
                            best_acc = sel_acc
                            best_results = results
                            best_hyperparams = {
                                'num_layers': num_layers,
                                'hidden_dim': hidden_dim,
                                'dropout': dropout,
                                'learning_rate': lr
                            }
                            best_fit_cost = fit_monitor.metrics()
                            print(f"  ★ New best {select_split} accuracy: {best_acc:.4f}")

                    except Exception as e:
                        failed_runs += 1
                        error_msg = f"Combination {combination_idx}: layers={num_layers}, hidden={hidden_dim}, dropout={dropout}, lr={lr} - Error: {str(e)}"
                        error_messages.append(error_msg)
                        print(f"  ✗ Error: {str(e)[:100]}")  # Print first 100 chars of error
                        
                        # Clear CUDA cache if CUDA error
                        if 'CUDA' in str(e) or 'out of memory' in str(e).lower():
                            print(f"  ⚠ CUDA memory issue detected. Clearing cache...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        continue

    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*80)
    print(f"Total combinations tested: {total_combinations}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/total_combinations*100:.1f}%")
    
    # Check if we have valid results
    if not best_results:
        print("\n" + "!"*80)
        print("WARNING: No valid results found during hyperparameter tuning!")
        print("!"*80)
        print("\nPossible causes:")
        print("  1. All hyperparameter combinations failed")
        print("  2. CUDA out of memory errors")
        print("  3. Model convergence issues")
        print("  4. Data quality problems")
        print("  5. Insufficient training epochs")
        
        if error_messages:
            print(f"\nError Summary (showing last {min(5, len(error_messages))} errors):")
            for err_msg in error_messages[-5:]:
                print(f"  - {err_msg}")
        
        print("\n" + "="*80)
        print("ATTEMPTING FALLBACK WITH SIMPLE DEFAULT HYPERPARAMETERS")
        print("="*80)
        print("Trying minimal configuration: layers=2, hidden=64, dropout=0.3, lr=0.01")
        
        try:
            # Try a simple, conservative configuration
            with ResourceMonitor() as fallback_monitor:
                predictions, results, weights = ensemble_gnn_learning(
                    model_list=model_list,
                    dataset_name=dataset_name,
                    weight_approach=ensemble_approaches,
                    device='cuda:0',
                    num_layers=2,
                    hidden_dim=64,
                    dropout=0.3,
                    learning_rate=0.01,
                    epochs=200,
                    custom_embeddings=custom_embeddings,
                    modified_datasets_list=modified_datasets_list,
                    does_report_training_process=True,
                    predictions_list=predictions_list,
                    supervised=supervised,
                    seed=seed,
                    llm_name=llm_name,
                    select_split=select_split,
                )

            best_fit_cost = fallback_monitor.metrics()
            config_times.append(fallback_monitor.time_sec)
            best_results = results
            best_hyperparams = {
                'num_layers': 2,
                'hidden_dim': 64,
                'dropout': 0.3,
                'learning_rate': 0.01
            }
            best_ensemble_predictions = predictions
            
            print(f"✓ Fallback successful! Test Accuracy: {results['test']['accuracy']:.4f}")
            print("="*80)
            
        except Exception as e:
            print(f"✗ Fallback also failed: {str(e)}")
            print("\nSuggestions:")
            print("  - Try reducing model complexity (fewer layers, smaller hidden_dim)")
            print("  - Increase training epochs")
            print("  - Check CUDA availability and memory")
            print("  - Verify input data quality")
            print("  - Try running on CPU if CUDA issues persist")
            print("  - Check if input models and embeddings are valid")
            print("!"*80 + "\n")
            tuning_monitor.__exit__(None, None, None)
            return {}, {}, None
    
    tuning_monitor.__exit__(None, None, None)
    tuning_cost = tuning_monitor.metrics()
    best_fit_cost = best_fit_cost or {}
    best_results = dict(best_results)
    best_results['cost'] = {
        'tuning_time_sec': tuning_cost['time_sec'],
        'tuning_peak_vram_mb': tuning_cost['peak_vram_mb'],
        'tuning_peak_vram_delta_mb': tuning_cost['peak_vram_delta_mb'],
        'n_configs': len(config_times),
        'avg_config_time_sec': round(sum(config_times) / len(config_times), 3) if config_times else None,
        'best_fit_time_sec': best_fit_cost.get('time_sec'),
        'best_fit_peak_vram_mb': best_fit_cost.get('peak_vram_mb'),
        'best_fit_peak_vram_delta_mb': best_fit_cost.get('peak_vram_delta_mb'),
    }
    cost = best_results['cost']

    print(f"\nBest Results Found:")
    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Test Accuracy: {best_results['test']['accuracy']:.4f}")
    print(f"Best Test Macro-F1: {best_results['test']['macro_f1']:.4f}")
    print(f"Best Test Weighted-F1: {best_results['test']['weighted_f1']:.4f}")
    print(f"Tuning cost: {cost['tuning_time_sec']:.2f}s for {cost['n_configs']} configs "
          f"(avg {cost['avg_config_time_sec']}s/config), peak VRAM {cost['tuning_peak_vram_mb']} MB")
    print(f"Selected ensemble fit: {cost['best_fit_time_sec']}s, peak VRAM {cost['best_fit_peak_vram_mb']} MB "
          f"(+{cost['best_fit_peak_vram_delta_mb']} MB)")
    print("="*80)

    # Report results using reporter
    if reporter is not None:
        report_title = f"Ensemble Hyperparameter Tuning - {title}" if title else "Ensemble Hyperparameter Tuning"
        report_text = f"""
Ensemble Hyperparameter Tuning Results:

Ensemble Weight Approach: {ensemble_approaches}{' (no learnable weights; hyperparameters below are unused)' if ensemble_approaches == 'uniform' else ''}

Best Hyperparameters:
  • Number of Layers: {best_hyperparams['num_layers']}
  • Hidden Dimension: {best_hyperparams['hidden_dim']}
  • Dropout: {best_hyperparams['dropout']}
  • Learning Rate: {best_hyperparams['learning_rate']}

Best Performance Metrics:
  • Test Accuracy: {best_results['test']['accuracy']:.4f} ({best_results['test']['accuracy']:.2%})
  • Test Macro-F1: {best_results['test']['macro_f1']:.4f}
  • Test Weighted-F1: {best_results['test']['weighted_f1']:.4f}
  • Validation Accuracy: {best_results['val']['accuracy']:.4f} ({best_results['val']['accuracy']:.2%})
  • Validation Macro-F1: {best_results['val']['macro_f1']:.4f}
  • Validation Weighted-F1: {best_results['val']['weighted_f1']:.4f}
  • Train Accuracy: {best_results['train']['accuracy']:.4f} ({best_results['train']['accuracy']:.2%})
  • Train Macro-F1: {best_results['train']['macro_f1']:.4f}
  • Train Weighted-F1: {best_results['train']['weighted_f1']:.4f}

Hyperparameter Search Space:
  • Number of Layers: {num_layers_options}
  • Hidden Dimensions: {hidden_dim_options}
  • Dropout Options: {dropout_options}
  • Learning Rates: {learning_rate_options}
  • Total Combinations Tested: {len(num_layers_options) * len(hidden_dim_options) * len(dropout_options) * len(learning_rate_options)}

Cost:
  • Tuning time (all configs): {cost['tuning_time_sec']:.2f}s ({cost['n_configs']} configs, avg {cost['avg_config_time_sec']}s/config)
  • Tuning peak VRAM: {cost['tuning_peak_vram_mb']} MB (+{cost['tuning_peak_vram_delta_mb']} MB during tuning)
  • Selected ensemble fit time: {cost['best_fit_time_sec']}s
  • Selected ensemble fit peak VRAM: {cost['best_fit_peak_vram_mb']} MB (+{cost['best_fit_peak_vram_delta_mb']} MB)
"""
        reporter.report(report_title, report_text)

    # Generate comprehensive visualizations for best model
    if visualize_best and best_hyperparams:
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR BEST ENSEMBLE")
        print("="*80)
        print(f"Training final ensemble with best hyperparameters and full epochs...")
        
        # Retrain with best hyperparameters and more epochs for better convergence
        best_ensemble_predictions, final_results, final_weights = ensemble_gnn_learning(
            model_list=model_list,
            dataset_name=dataset_name,
            weight_approach='learnable_per_classes',  # Use more advanced approach for final model
            device='cuda:0',
            num_layers=best_hyperparams['num_layers'],
            hidden_dim=best_hyperparams['hidden_dim'],
            dropout=best_hyperparams['dropout'],
            learning_rate=best_hyperparams['learning_rate'],
            epochs=200,  # More epochs for final model
            custom_embeddings=custom_embeddings,
            modified_datasets_list=modified_datasets_list,
            does_report_training_process=True,
            predictions_list=predictions_list,
            supervised=supervised,
            seed=seed,
            llm_name=llm_name,
            select_split=select_split,
        )

        # Generate comprehensive visualizations (only when graph has same size as original dataset)
        cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
        
        # Load dataset for visualization
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        path_prefix = _get_path_prefix()
        
        graph_data, num_classes, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=1, path_prefix=path_prefix)
        
        # Skip visualization when using modified/augmented datasets (different node count -> shape mismatch)
        n_original = graph_data.y.shape[0]
        n_modified = modified_datasets_list[0].y.shape[0] if modified_datasets_list and len(modified_datasets_list) > 0 else n_original
        if n_modified != n_original:
            print(f"\n[Skipping comprehensive visualization: modified dataset has {n_modified} nodes vs original {n_original}; visualization expects original graph.]")
        else:
            # Create title list if not provided
            if title_list is None:
                title_list = [f"Model_{i+1}" for i in range(len(model_list))]
            
            # Generate comprehensive visualizations
            os.makedirs(save_dir, exist_ok=True)
            viz_save_dir = os.path.join(save_dir, f"{dataset_name}_best_ensemble_tuned")
            
            all_mistake_indices, all_stats = comprehensive_gnn_mistakes_analysis_visualization_list(
                cfg=cfg,
                dataset=graph_data,
                gnn_model_list=model_list,
                features_list=custom_embeddings,
                title_list=title_list,
                ensemble_predictions=best_ensemble_predictions,
                save_dir=viz_save_dir,
                show=show,
                reporter=reporter
            )
            
            # Report final ensemble results
            if reporter is not None:
                reporter.report_title("Best Ensemble Model - Final Results with Visualizations")
                reporter.report_txt(f"Retrained with best hyperparameters for 200 epochs")
                reporter.report_txt(f"Weight Approach: learnable_per_classes")
                reporter.report_txt("")
                reporter.report_txt(f"Final Results:")
                reporter.report_txt(f"  Train - Acc: {final_results['train']['accuracy']:.4f}, "
                                  f"Macro-F1: {final_results['train']['macro_f1']:.4f}")
                reporter.report_txt(f"  Val   - Acc: {final_results['val']['accuracy']:.4f}, "
                                  f"Macro-F1: {final_results['val']['macro_f1']:.4f}")
                reporter.report_txt(f"  Test  - Acc: {final_results['test']['accuracy']:.4f}, "
                                  f"Macro-F1: {final_results['test']['macro_f1']:.4f}")
                reporter.report_txt(f"\nVisualizations saved to: {viz_save_dir}")
            
            print(f"\nComprehensive visualizations saved to: {viz_save_dir}")

    return best_results, best_hyperparams, best_ensemble_predictions


def ensemble_machine_learning_gnn(gnn_model_list: list, sklearn_model_list: list, dataset_name, 
                         weight_approach='uniform', device='cuda:0',
                         custom_embeddings: list =None, modified_datasets_list : list =None,
                          does_report_training_process: bool= False):
    """
    Perform meta-ensemble learning using sklearn models trained on GNN logits.
    First extracts logits from GNN models, then trains sklearn ensemble models on those logits.
    
    Args:
        gnn_model_list: list of trained GNN models to extract logits from
        sklearn_model_list: list of sklearn model types to use (e.g., ['rf', 'adaboost', 'bagging', 'histgb'])
        dataset_name: name of the dataset
        weight_approach: 'uniform' for equal weights (learnable approaches can be added later)
        device: device to run on
        custom_embeddings: optional list of node embeddings to use for each GNN model
        modified_datasets_list: optional list of modified datasets (mainly for compatibility)
        does_report_training_process: whether to print training progress
        
    Returns:
        ensemble_predictions: Combined predictions from all sklearn models
        results: dict with accuracy, macro_f1, weighted_f1 for train/val/test
        weights: learned or uniform weights
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier
    
    # Load dataset (for masks and labels)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    cfg = set_mgnn_cfg(dataset_name)
    set_seed(cfg.seed)
    
    path_prefix = _get_path_prefix()
    graph_data, num_classes, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=1, path_prefix=path_prefix, seed=cfg.split_seed)
    
    # Move all GNN models to the correct device
    for model in gnn_model_list:
        model.to(device)
    
    # Extract logits from all GNN models
    print(f"\nExtracting logits from {len(gnn_model_list)} GNN models...")
    gnn_logits_list = []
    with torch.no_grad():
        if custom_embeddings is not None and modified_datasets_list is not None:
            # Use different embeddings AND different datasets for each model
            print(f"Using custom embeddings and modified datasets for {len(gnn_model_list)} models")
            if isinstance(custom_embeddings, list) and isinstance(modified_datasets_list, list):
                for idx, (model, emb, modified_data) in enumerate(zip(gnn_model_list, custom_embeddings, modified_datasets_list)):
                    model.eval()
                    emb = emb.to(device)
                    edge_index = modified_data.edge_index.to(device)
                    logits = model(emb, edge_index)
                    gnn_logits_list.append(logits)
                    print(f"  GNN {idx+1}: Using modified dataset with {edge_index.shape[1]} edges")
            else:
                raise ValueError("Both custom_embeddings and modified_datasets_list must be lists of same length")
                
        elif custom_embeddings is not None:
            # Use different embeddings but same dataset for all models
            if isinstance(custom_embeddings, list):
                print(f"Using custom embeddings for {len(gnn_model_list)} models (same edge_index)")
                for idx, (model, emb) in enumerate(zip(gnn_model_list, custom_embeddings)):
                    model.eval()
                    emb = emb.to(device)
                    logits = model(emb, graph_data.edge_index)
                    gnn_logits_list.append(logits)
            else:
                embeddings = custom_embeddings.to(device)
                for model in gnn_model_list:
                    model.eval()
                    logits = model(embeddings, graph_data.edge_index)
                    gnn_logits_list.append(logits)
        else:
            # Use default graph_data.x for all models
            for model in gnn_model_list:
                model.eval()
                logits = model(graph_data.x, graph_data.edge_index)
                gnn_logits_list.append(logits)
    
    # Concatenate all GNN logits to create feature matrix for sklearn models
    # Shape: [num_nodes, num_gnns * num_classes]
    combined_gnn_logits = torch.cat(gnn_logits_list, dim=1).cpu().numpy()
    print(f"Combined GNN logits shape: {combined_gnn_logits.shape} [num_nodes, num_gnns * num_classes]")
    
    # Get labels and masks
    labels = graph_data.y.cpu().numpy()
    train_mask = graph_data.train_mask.cpu().numpy()
    val_mask = graph_data.val_mask.cpu().numpy()
    test_mask = graph_data.test_mask.cpu().numpy()
    
    # Train sklearn ensemble models on GNN logits
    sklearn_predictions_list = []
    trained_sklearn_models = []
    
    print(f"\nTraining {len(sklearn_model_list)} sklearn ensemble models on GNN logits...")
    
    for idx, model_name in enumerate(sklearn_model_list):
        # Select training data (GNN logits from training nodes)
        X_train = combined_gnn_logits[train_mask]
        y_train = labels[train_mask]
        
        # Initialize sklearn model based on name or type
        if isinstance(model_name, str):
            if model_name.lower() in ['rf', 'randomforest']:
                model = RandomForestClassifier(n_estimators=1000, random_state=cfg.seed, n_jobs=-1, verbose=1)
                model_type = 'RandomForest'
            elif model_name.lower() in ['adaboost', 'ada']:
                model = AdaBoostClassifier(n_estimators=1000, random_state=cfg.seed)
                model_type = 'AdaBoost'
            elif model_name.lower() in ['bagging', 'bag']:
                model = BaggingClassifier(n_estimators=100, random_state=cfg.seed, n_jobs=-1)
                model_type = 'Bagging'
            elif model_name.lower() in ['histgb', 'histgradient']:
                model = HistGradientBoostingClassifier(max_iter=1000, random_state=cfg.seed)
                model_type = 'HistGradientBoosting'
            else:
                raise ValueError(f"Unknown model type: {model_name}")
        else:
            # If model_name is already an instantiated model
            model = model_name
            model_type = type(model).__name__
        
        if does_report_training_process:
            print(f"  Training sklearn model {idx+1}/{len(sklearn_model_list)}: {model_type}")
        
        # Train the sklearn model on GNN logits
        model.fit(X_train, y_train)
        trained_sklearn_models.append(model)
        
        # Get predictions on all nodes
        if hasattr(model, 'predict_proba'):
            sklearn_logits = model.predict_proba(combined_gnn_logits)
        else:
            # If model doesn't support predict_proba, use one-hot encoding of predictions
            preds = model.predict(combined_gnn_logits)
            sklearn_logits = np.eye(num_classes)[preds]
        
        # Convert to torch tensor
        sklearn_logits_tensor = torch.FloatTensor(sklearn_logits).to(device)
        sklearn_predictions_list.append(sklearn_logits_tensor)
        
        if does_report_training_process:
            # Evaluate individual sklearn model
            train_preds = sklearn_logits_tensor[graph_data.train_mask].argmax(dim=1).cpu().numpy()
            train_labels = graph_data.y[graph_data.train_mask].cpu().numpy()
            train_acc, _, _ = compute_acc_and_f1(train_preds, train_labels)
            
            test_preds = sklearn_logits_tensor[graph_data.test_mask].argmax(dim=1).cpu().numpy()
            test_labels = graph_data.y[graph_data.test_mask].cpu().numpy()
            test_acc, _, _ = compute_acc_and_f1(test_preds, test_labels)
            
            print(f"    {model_type} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    num_sklearn_models = len(sklearn_model_list)
    
    # Combine sklearn predictions based on weight_approach
    if weight_approach == 'uniform':
        # Uniform weights: simple average
        ensemble_logits = sum(sklearn_predictions_list) / num_sklearn_models
        weights = [1.0 / num_sklearn_models] * num_sklearn_models
        print(f"\nUsing uniform weights for sklearn ensemble: {weights}")
    else:
        # For now, only uniform is supported, but learnable can be added later
        print(f"Warning: weight_approach '{weight_approach}' not implemented for ML ensemble, using uniform weights")
        ensemble_logits = sum(sklearn_predictions_list) / num_sklearn_models
        weights = [1.0 / num_sklearn_models] * num_sklearn_models
    
    # Get final predictions
    ensemble_predictions = ensemble_logits.argmax(dim=1)
    
    # Compute metrics for train, val, test
    results = {}
    masks = {
        'train': graph_data.train_mask,
        'val': graph_data.val_mask,
        'test': graph_data.test_mask
    }
    
    print("\n" + "="*80)
    print("Sklearn Meta-Ensemble Results:")
    print("="*80)
    for split_name, mask in masks.items():
        pred = ensemble_predictions[mask].cpu().numpy()
        gt = graph_data.y[mask].cpu().numpy()
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(pred, gt)
        results[split_name] = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
        print(f"[{split_name.upper()}] Acc: {acc:.2f}, Macro-F1: {macro_f1:.2f}, Weighted-F1: {weighted_f1:.2f}")
    print("="*80)

    return ensemble_predictions, results, weights

def comprehensive_ensemble_best_gnn_mistakes(dataset_name='cora', llm_name='llama_3.2_1B', peft_type='lora',
                                             weight_approach='learnable_per_classes', num_layers=2, hidden_dim=128,
                                             dropout=0.3, learning_rate=0.01, epochs=200,
                                             save_dir="results/gnn_mistakes_comprehensive", show=False,
                                             seed=None, supervised=True,
                                             csv_path="results/ensemble/ensemble_gnn_results.csv"):
    """
    Run comprehensive ensemble GNN learning on multiple embedding methods
    and report results for each method as well as the ensemble.
    like the comprehensive_gnn_mistakes_analysis_visualization function in visualize_gnn_mistakes.py

    Args:
        dataset_name: name of the dataset (e.g., 'cora', 'pubmed')
        llm_name: name of the LLM model
        peft_type: type of PEFT (e.g., 'lora')
        weight_approach: ensemble weight learning approach
        num_layers: number of layers for weight learner
        hidden_dim: hidden dimension for weight learner
        dropout: dropout probability
        learning_rate: learning rate for ensemble
        epochs: number of epochs for ensemble training
        save_dir: directory to save visualizations
        show: whether to display plots
        seed: Seed of the trained LoRA adapters/embedding caches to load (must match the
              --seed used in train_llm/train_other_lora_init_apporaches.py and
              cache/cache_embedding_with_diffrent_init_weights.py) and the seed used for
              the GNN's own training. If None, uses whatever seed is already in
              configs/dataset/<dataset_name>.json / gnn_hyperparameters/<dataset_name>.json.
        supervised: True for the supervised embedding cache, False for the semi-supervised
              (seed-tagged) one.
        csv_path: Where to append the complete-info CSV report ("" / None disables it).

    Returns:
        ensemble_predictions: predictions from ensemble
        results: performance metrics
        weights: learned weights
    """
    print("="*80)
    print("COMPREHENSIVE ENSEMBLE GNN MISTAKES ANALYSIS")
    print("="*80)

    # Setup config
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    if seed is not None:
        cfg.dataset.seed = seed  # must match the seed the LoRA adapter/embedding cache was trained with

    # Setup reporter
    report_save_dir = f"results/train_info/{dataset_name}_ensemble_mistakes"
    reporter = ReportResults(cfg, save_dir=report_save_dir)

    reporter.report_title("Comprehensive Ensemble GNN Mistakes Analysis")
    reporter.report_txt(f"Dataset: {dataset_name}")
    reporter.report_txt(f"LLM: {llm_name}")
    reporter.report_txt(f"PEFT Type: {peft_type}")
    reporter.report_txt(f"Seed: {cfg.dataset.seed}")
    reporter.report_txt(f"Ensemble Approach: {weight_approach}")
    reporter.report_txt(f"Num Layers: {num_layers}, Hidden Dim: {hidden_dim}")
    reporter.report_txt(f"Dropout: {dropout}, Learning Rate: {learning_rate}")
    reporter.report_txt(f"Epochs: {epochs}")
    reporter.report_txt("")

    # Load all embedding methods (seed-tagged cache when supervised=False)
    print("\nLoading embeddings from different initialization methods...")
    print(f"=== Using seed: {cfg.dataset.seed} | supervised: {supervised}")
    data_pissa, data_orthogonal, data_gaussian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg, supervised=supervised, seed=seed)

    emb_pissa = get_embedding_from_data(data_pissa)
    emb_orthogonal = get_embedding_from_data(data_orthogonal)
    emb_loftq = get_embedding_from_data(data_loftq)
    emb_eva = get_embedding_from_data(data_eva)
    emb_gaussian = get_embedding_from_data(data_gaussian)

    print(f"Loaded embeddings - PISSA: {emb_pissa.shape}, Orthogonal: {emb_orthogonal.shape}, "
          f"LoftQ: {emb_loftq.shape}, EVA: {emb_eva.shape}, Gaussian: {emb_gaussian.shape}")

    # Train individual GNN models
    print("\n" + "="*80)
    print("TRAINING INDIVIDUAL GNN MODELS")
    print("="*80)

    gnn_kwargs = dict(does_print_training_process=False, reporter=reporter, supervised=supervised,
                       seed=seed, llm_name=llm_name, peft_type=peft_type, csv_path=csv_path)
    results_pissa, model_pissa = gnn_train_and_report(dataset_name, emb_pissa, title="PISSA", **gnn_kwargs)
    results_orthogonal, model_orthogonal = gnn_train_and_report(dataset_name, emb_orthogonal, title="ORTHOGONAL", **gnn_kwargs)
    results_loftq, model_loftq = gnn_train_and_report(dataset_name, emb_loftq, title="LOFTQ", **gnn_kwargs)
    results_eva, model_eva = gnn_train_and_report(dataset_name, emb_eva, title="EVA", **gnn_kwargs)
    results_gaussian, model_gaussian = gnn_train_and_report(dataset_name, emb_gaussian, title="GAUSSIAN", **gnn_kwargs)

    # Create lists for ensemble
    gnn_model_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_gaussian]
    custom_embeddings = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_gaussian]
    title_list = ['PISSA', 'ORTHOGONAL', 'LOFTQ', 'EVA', 'GAUSSIAN']

    # Train ensemble
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE MODEL")
    print("="*80)

    ensemble_predictions, ensemble_results, weights = ensemble_gnn_learning(
        model_list=gnn_model_list,
        dataset_name=dataset_name,
        weight_approach=weight_approach,
        device='cuda:0',
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        custom_embeddings=custom_embeddings,
        does_report_training_process=True,
        supervised=supervised,
        seed=seed,
        llm_name=llm_name,
    )

    # Append complete-info CSV row for the final ensemble result
    if csv_path:
        from datetime import datetime
        ensemble_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "llm_name": llm_name,
            "peft_type": peft_type,
            "init_weight_approach": "ENSEMBLE",
            "embedding_seed": seed if seed is not None else cfg.dataset.seed,
            "gnn_seed": seed if seed is not None else "",
            "supervised": supervised,
            "embedding_shape": tuple(emb_pissa.shape),
            "gnn_model_name": "",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": learning_rate,
            "weight_decay": "",
            "batch_norm": "",
            "epochs": epochs,
            "early_stop": "",
            "re_split": 1 if supervised else 0,
            "test_accuracy": ensemble_results['test']['accuracy'],
            "test_macro_f1": ensemble_results['test']['macro_f1'],
            "test_weighted_f1": ensemble_results['test']['weighted_f1'],
            "val_accuracy": ensemble_results['val']['accuracy'],
            "val_macro_f1": ensemble_results['val']['macro_f1'],
            "val_weighted_f1": ensemble_results['val']['weighted_f1'],
            "weight_approach": weight_approach,
            "learned_weights": weights,
        }
        append_gnn_result_csv(ensemble_row, csv_path)
        print(f"[OK] Ensemble result appended to: {csv_path}")
    
    # Report ensemble results
    reporter.report_title("Ensemble Model Performance")
    reporter.report_txt(f"Weight Approach: {weight_approach}")
    reporter.report_txt(f"Architecture: {num_layers} layers, hidden_dim={hidden_dim}, dropout={dropout}")
    reporter.report_txt("")
    reporter.report_txt(f"Train - Acc: {ensemble_results['train']['accuracy']:.4f}, "
                       f"Macro-F1: {ensemble_results['train']['macro_f1']:.4f}, "
                       f"Weighted-F1: {ensemble_results['train']['weighted_f1']:.4f}")
    reporter.report_txt(f"Val   - Acc: {ensemble_results['val']['accuracy']:.4f}, "
                       f"Macro-F1: {ensemble_results['val']['macro_f1']:.4f}, "
                       f"Weighted-F1: {ensemble_results['val']['weighted_f1']:.4f}")
    reporter.report_txt(f"Test  - Acc: {ensemble_results['test']['accuracy']:.4f}, "
                       f"Macro-F1: {ensemble_results['test']['macro_f1']:.4f}, "
                       f"Weighted-F1: {ensemble_results['test']['weighted_f1']:.4f}")
    reporter.report_txt("")
    reporter.report_txt(f"Learned Weights: {weights}")
    
    # Load dataset for visualization
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path_prefix = _get_path_prefix()
    
    graph_data, num_classes, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=1, path_prefix=path_prefix)
    
    # Generate comprehensive visualizations
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE MISTAKE VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_mistake_indices, all_stats = comprehensive_gnn_mistakes_analysis_visualization_list(
        cfg=cfg,
        dataset=graph_data,
        gnn_model_list=gnn_model_list,
        features_list=custom_embeddings,
        title_list=title_list,
        ensemble_predictions=ensemble_predictions,
        save_dir=save_dir,
        show=show,
        reporter=reporter
    )
    
    # Create summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    summary_text = "\nIndividual Model Results:\n"
    individual_results = [
        ('PISSA', results_pissa),
        ('ORTHOGONAL', results_orthogonal),
        ('LOFTQ', results_loftq),
        ('EVA', results_eva),
        ('GAUSSIAN', results_gaussian)
    ]
    
    for name, result in individual_results:
        summary_text += f"  {name:15s} - Test Acc: {result['test_acc']:.4f}, Test F1: {result['test_f1']:.4f}\n"
    
    summary_text += f"\nEnsemble Result:\n"
    summary_text += f"  {'ENSEMBLE':15s} - Test Acc: {ensemble_results['test']['accuracy']:.4f}, "
    summary_text += f"Test F1: {ensemble_results['test']['macro_f1']:.4f}\n"
    
    # Best individual model, chosen on validation accuracy (choosing it on test would be an oracle baseline)
    best_individual_name, best_individual_result = max(individual_results, key=lambda r: r[1]['val_acc'])
    best_individual_acc = best_individual_result['test_acc']

    improvement = ensemble_results['test']['accuracy'] - best_individual_acc
    summary_text += f"\nImprovement over best individual by val ({best_individual_name}): {improvement:+.4f}\n"
    
    # Analyze ensemble mistakes per class
    print("\n" + "="*80)
    print("ENSEMBLE MISTAKE ANALYSIS PER CLASS")
    print("="*80)
    
    labels = graph_data.y.cpu().numpy()
    test_mask = graph_data.test_mask.cpu().numpy()
    ensemble_preds = ensemble_predictions.cpu().numpy() if isinstance(ensemble_predictions, torch.Tensor) else ensemble_predictions
    
    # Calculate per-class statistics for ensemble
    ensemble_mistakes_by_class = {}
    ensemble_total_by_class = {}
    ensemble_error_rates = {}
    
    for class_id in range(num_classes):
        # Get test nodes for this class
        class_test_mask = test_mask & (labels == class_id)
        total = class_test_mask.sum()
        
        if total > 0:
            # Count mistakes
            class_labels = labels[class_test_mask]
            class_preds = ensemble_preds[class_test_mask]
            mistakes = (class_labels != class_preds).sum()
            error_rate = mistakes / total * 100
            
            ensemble_mistakes_by_class[class_id] = mistakes
            ensemble_total_by_class[class_id] = total
            ensemble_error_rates[class_id] = error_rate
        else:
            ensemble_mistakes_by_class[class_id] = 0
            ensemble_total_by_class[class_id] = 0
            ensemble_error_rates[class_id] = 0.0
    
    summary_text += f"\nEnsemble Per-Class Mistake Rates (Test Set):\n"
    for class_id in range(num_classes):
        total = ensemble_total_by_class[class_id]
        mistakes = ensemble_mistakes_by_class[class_id]
        error_rate = ensemble_error_rates[class_id]
        correct = total - mistakes
        acc = (correct / total * 100) if total > 0 else 0.0
        
        summary_text += f"  Class {class_id}: {mistakes}/{total} mistakes ({error_rate:.2f}% error, {acc:.2f}% acc)\n"
    
    # Calculate average error rate
    avg_error_rate = sum(ensemble_error_rates.values()) / len(ensemble_error_rates)
    summary_text += f"\n  Average Error Rate: {avg_error_rate:.2f}%\n"
    summary_text += f"  Overall Test Accuracy: {ensemble_results['test']['accuracy']:.2f}%\n"
    
    # Find classes with highest and lowest error rates
    sorted_classes = sorted(ensemble_error_rates.items(), key=lambda x: x[1], reverse=True)
    worst_classes = sorted_classes[:3]
    best_classes = sorted_classes[-3:][::-1]
    
    summary_text += f"\n  Top 3 Most Difficult Classes (Highest Error):\n"
    for class_id, error_rate in worst_classes:
        total = ensemble_total_by_class[class_id]
        mistakes = ensemble_mistakes_by_class[class_id]
        summary_text += f"    Class {class_id}: {error_rate:.2f}% error ({mistakes}/{total})\n"
    
    summary_text += f"\n  Top 3 Easiest Classes (Lowest Error):\n"
    for class_id, error_rate in best_classes:
        total = ensemble_total_by_class[class_id]
        mistakes = ensemble_mistakes_by_class[class_id]
        summary_text += f"    Class {class_id}: {error_rate:.2f}% error ({mistakes}/{total})\n"
    
    print(summary_text)
    reporter.report_title("Final Summary")
    reporter.report_txt(summary_text)
    
    # Generate ensemble mistake visualizations
    print("\n" + "="*80)
    print("GENERATING ENSEMBLE MISTAKE VISUALIZATIONS")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Ensemble Mistakes Count per Class
    ax1 = fig.add_subplot(gs[0, 0])
    classes = list(range(num_classes))
    mistake_counts = [ensemble_mistakes_by_class[i] for i in classes]
    bars1 = ax1.bar(classes, mistake_counts, color='lightcoral', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Mistakes', fontsize=12)
    ax1.set_title('Ensemble: Mistakes by Class (Test Set)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_xticks(classes)
    for bar, count in zip(bars1, mistake_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Ensemble Error Rate per Class
    ax2 = fig.add_subplot(gs[0, 1])
    error_rates_list = [ensemble_error_rates[i] for i in classes]
    bars2 = ax2.bar(classes, error_rates_list, color='steelblue', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Ensemble: Error Rate by Class (Test Set)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_xticks(classes)
    ax2.axhline(y=avg_error_rate, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_error_rate:.2f}%')
    ax2.legend(fontsize=10)
    for bar, rate in zip(bars2, error_rates_list):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Ensemble Accuracy per Class
    ax3 = fig.add_subplot(gs[1, 0])
    accuracy_list = [(1 - ensemble_error_rates[i]/100)*100 for i in classes]
    bars3 = ax3.bar(classes, accuracy_list, color='lightgreen', edgecolor='black', alpha=0.8)
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Ensemble: Accuracy by Class (Test Set)', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_xticks(classes)
    ax3.axhline(y=ensemble_results['test']['accuracy']*100, color='green', linestyle='--', 
                linewidth=2, label=f"Overall: {ensemble_results['test']['accuracy']:.2f}%")
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 105])
    for bar, acc in zip(bars3, accuracy_list):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Class Distribution in Test Set
    ax4 = fig.add_subplot(gs[1, 1])
    class_counts = [ensemble_total_by_class[i] for i in classes]
    bars4 = ax4.bar(classes, class_counts, color='lightyellow', edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Class', fontsize=12)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('Ensemble: Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    ax4.set_xticks(classes)
    for bar, count in zip(bars4, class_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 5: Top 3 Most Difficult vs Easiest Classes
    ax5 = fig.add_subplot(gs[2, 0])
    worst_class_ids = [c[0] for c in worst_classes]
    worst_error_rates = [c[1] for c in worst_classes]
    best_class_ids = [c[0] for c in best_classes]
    best_error_rates = [c[1] for c in best_classes]
    
    x_pos = np.arange(3)
    width = 0.35
    bars_worst = ax5.bar(x_pos - width/2, worst_error_rates, width, 
                         label='Most Difficult', color='salmon', edgecolor='black', alpha=0.8)
    bars_best = ax5.bar(x_pos + width/2, best_error_rates, width, 
                        label='Easiest', color='lightgreen', edgecolor='black', alpha=0.8)
    
    ax5.set_xlabel('Rank', fontsize=12)
    ax5.set_ylabel('Error Rate (%)', fontsize=12)
    ax5.set_title('Ensemble: Top 3 Most Difficult vs Easiest Classes', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['1st', '2nd', '3rd'])
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3, axis='y')
    
    # Add class IDs as text
    for i, (bar, class_id) in enumerate(zip(bars_worst, worst_class_ids)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'C{class_id}\n{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    for i, (bar, class_id) in enumerate(zip(bars_best, best_class_ids)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'C{class_id}\n{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Summary Statistics Text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    stats_summary = f"""
ENSEMBLE MISTAKE ANALYSIS SUMMARY
{'='*60}

Overall Performance:
  • Total Test Samples: {test_mask.sum()}
  • Total Mistakes: {sum(ensemble_mistakes_by_class.values())}
  • Overall Test Accuracy: {ensemble_results['test']['accuracy']:.2f}%
  • Overall Test Error: {(1-ensemble_results['test']['accuracy']):.2f}%
  • Average Error Rate: {avg_error_rate:.2f}%

Most Difficult Classes:
"""
    for i, (class_id, error_rate) in enumerate(worst_classes, 1):
        total = ensemble_total_by_class[class_id]
        mistakes = ensemble_mistakes_by_class[class_id]
        stats_summary += f"  {i}. Class {class_id}: {error_rate:.2f}% error ({mistakes}/{total})\n"
    
    stats_summary += f"""
Easiest Classes:
"""
    for i, (class_id, error_rate) in enumerate(best_classes, 1):
        total = ensemble_total_by_class[class_id]
        mistakes = ensemble_mistakes_by_class[class_id]
        stats_summary += f"  {i}. Class {class_id}: {error_rate:.2f}% error ({mistakes}/{total})\n"
    
    stats_summary += f"""
Performance Metrics:
  • Test Macro-F1: {ensemble_results['test']['macro_f1']:.4f}
  • Test Weighted-F1: {ensemble_results['test']['weighted_f1']:.4f}
  • Improvement over best individual: {improvement:+.4f}
"""
    
    ax6.text(0.05, 0.95, stats_summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add main title
    fig.suptitle(f'Ensemble Model Mistake Analysis - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save the figure
    ensemble_viz_path = os.path.join(save_dir, f'{dataset_name}_ensemble_mistake_analysis_{weight_approach}.png')
    plt.savefig(ensemble_viz_path, dpi=300, bbox_inches='tight')
    print(f"Ensemble mistake visualization saved to: {ensemble_viz_path}")
    
    if show:
        plt.show()
    plt.close()
    
    # Report visualization path
    reporter.report_txt(f"\nEnsemble Mistake Visualization saved to: {ensemble_viz_path}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"Visualizations saved to: {save_dir}")
    print(f"Reports saved to: {report_save_dir}")
    print("="*80)
    
    return ensemble_predictions, ensemble_results, weights

# if __name__ == '__main__':
#     # Run comprehensive ensemble mistakes analysis
#     dataset_name = 'wikics'
#     llm_name = 'llama_3.2_1B'
#     peft_type = 'lora'
#
#     # Run comprehensive analysis with visualization
#     ensemble_predictions, results, weights = comprehensive_ensemble_best_gnn_mistakes(
#         dataset_name=dataset_name,
#         llm_name=llm_name,
#         peft_type=peft_type,
#         weight_approach='learnable',
#         num_layers=2,
#         hidden_dim=512,
#         dropout=0.0,
#         learning_rate=0.01,
#         epochs=200,
#         save_dir=f"results/gnn_mistakes_comprehensive/{dataset_name}",
#         show=False
#     )