## import necessary libraries and modules
import torch
import os
import json
import numpy as np
from collections import Counter

from gnns.gnn_mtrainer import gnn_train_and_report, get_datasets_path
from config import setup_finetuning_cfg
from train_llm.train import Train
from adapter_dir import get_adapter_dir_lora_init_weights
from report.reporter import ReportResults
from common import load_graph_dataset_for_tape, set_seed

# define init input
dataset_name = 'cora'
llm_name = 'llama_3.2_1B'
peft_type = 'lora'
init_weight_approaches = 'loftq'  # Options: 'pissa', 'gaussian', 'eva', 'loftq', 'orthogonal'
# Split seed the (un-seeded) LLM embedding cache was fine-tuned on, and the one the GNN uses by default.
# A different seed (e.g. a hard-coded 42) would mark nodes as "train" that are val/test nodes of that split.
seed = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type).dataset.seed

set_seed(seed)

def get_embedding(dataset_name, llm_name, peft_type, init_weight_approaches, save_root="./artifacts/cache"):
    fname = f"{llm_name}_{dataset_name}_seqcls_{peft_type}_init-{init_weight_approaches}_pool-mean.pt"
    load_path = os.path.join(save_root, fname)
    embedding = torch.load(load_path)
    print(f"[OK] Loaded embedding with init approach '{init_weight_approaches}' from: {load_path}")
    return embedding

# Load embeddings and train GNN
embeddings = get_embedding(dataset_name, llm_name, peft_type, init_weight_approaches)
emb = embeddings['embeddings']
results, model = gnn_train_and_report(dataset_name, emb, title=init_weight_approaches,
                                                      does_print_training_process=False, seed=seed)

# ==========================================
# Analyze GNN mistakes and identify top 3 classes with highest error rates
# ==========================================
print("\n" + "="*80)
print("Analyzing GNN Mistakes per Class")
print("="*80)

# Determine correct path_prefix for loading datasets
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
    path_prefix = '../../../'

print(f"Using path_prefix: {path_prefix}")

# Load dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset, num_classes, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=1, seed=seed, path_prefix=path_prefix)

# Get predictions from trained GNN
model.eval()
with torch.no_grad():
    logits = model(emb.to(device), dataset.edge_index.to(device))
    predictions = logits.argmax(dim=1)

# Calculate per-class error rates on training set
train_mask = dataset.train_mask
train_labels = dataset.y[train_mask]
train_preds = predictions[train_mask]

# Count mistakes per class
class_mistakes = {}
class_totals = {}
for i in range(num_classes):
    class_mask = (train_labels == i)
    class_totals[i] = class_mask.sum().item()
    class_mistakes[i] = ((train_preds[class_mask] != train_labels[class_mask]).sum().item())

# Calculate mistake rates
mistake_rates = {}
for i in range(num_classes):
    if class_totals[i] > 0:
        mistake_rates[i] = class_mistakes[i] / class_totals[i]
    else:
        mistake_rates[i] = 0.0

# Print mistake rates for all classes
print("\nPer-Class Mistake Rates (Training Set):")
for class_id in sorted(mistake_rates.keys()):
    print(f"  Class {class_id}: {mistake_rates[class_id]:.4f} ({class_mistakes[class_id]}/{class_totals[class_id]} mistakes)")

# Get top 3 classes with highest mistake rates
top_3_mistake_classes = sorted(mistake_rates.items(), key=lambda x: x[1], reverse=True)[:3]
top_3_class_ids = [cls_id for cls_id, _ in top_3_mistake_classes]

print(f"\nTop 3 Classes with Highest Mistake Rates:")
for cls_id, rate in top_3_mistake_classes:
    print(f"  Class {cls_id}: {rate:.4f} ({class_mistakes[cls_id]}/{class_totals[cls_id]} mistakes)")

# ==========================================
# Load enhanced texts for top 3 mistake classes
# ==========================================
print("\n" + "="*80)
print("Loading Enhanced Texts for Top 3 Mistake Classes")
print("="*80)

datasets_path = get_datasets_path()

# Load enhanced texts from datasets/enhanced_texts/
enhanced_texts_map = {}  # Map from node_idx to enhanced text
enhanced_texts_path = os.path.join(datasets_path, 'enhanced_texts', dataset_name)
if os.path.exists(enhanced_texts_path):
    print(f"\nLoading from enhanced_texts: {enhanced_texts_path}")
    for node_idx in range(len(dataset.y)):
        json_sample_path = os.path.join(enhanced_texts_path, f"{node_idx}.json")
        if os.path.exists(json_sample_path):
            with open(json_sample_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
                # Combine summary and paraphrased text as enhanced version
                enhanced_text = enhanced_data.get('summary', '') + " " + enhanced_data.get('paraphrased_text', '')
                enhanced_texts_map[node_idx] = enhanced_text.strip()

# Load GPT-4 responses from datasets/gpt_4o/
gpt_responses_path = os.path.join(datasets_path, 'gpt_4o', f"{dataset_name}.json")
gpt_responses_map = {}
if os.path.exists(gpt_responses_path):
    print(f"Loading from gpt_4o: {gpt_responses_path}")
    with open(gpt_responses_path, 'r', encoding='utf-8') as f:
        node_idx = 0
        for line in f:
            line = line.strip()
            if line:
                gpt_response = json.loads(line)
                gpt_responses_map[node_idx] = gpt_response.get('answer', '')
                node_idx += 1

print(f"\nLoaded enhanced texts: {len(enhanced_texts_map)} nodes")
print(f"Loaded GPT responses: {len(gpt_responses_map)} nodes")

# ==========================================
# Prepare augmented dataset for retraining
# ==========================================
print("\n" + "="*80)
print("Preparing Augmented Training Dataset")
print("="*80)

# Create a custom dataset preparation that prioritizes enhanced texts for top 3 mistake classes
def prepare_augmented_data_for_retraining(cfg, top_3_class_ids, enhanced_texts_map, gpt_responses_map, path_prefix):
    """
    Prepare augmented training data that uses enhanced texts for nodes in top 3 mistake classes.
    """
    from dataset.dataset_loader import get_tokenizer, encoding_texts_to_tokens, TagsDataset
    
    dataset, _, _ = load_graph_dataset_for_tape(cfg.dataset.name, cfg.device, re_split=1, seed=cfg.dataset.seed, path_prefix=path_prefix)
    tokenizer = get_tokenizer(cfg)
    
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    # Prepare texts with augmentation for top 3 mistake classes
    # For top 3 mistake classes in training set: create 3 samples (GPT + enhanced + original)
    # For other samples: use original text only
    
    train_texts = []
    train_labels = []
    augmented_count = 0
    
    for i in range(len(dataset.raw_texts)):
        label = int(dataset.y[i].item())
        
        if train_mask[i]:
            # If this node belongs to top 3 mistake classes, add all available versions
            if label in top_3_class_ids:
                # Add GPT response version if available
                if i in gpt_responses_map and gpt_responses_map[i]:
                    train_texts.append(gpt_responses_map[i])
                    train_labels.append(label)
                    augmented_count += 1
                
                # Add enhanced text version if available
                if i in enhanced_texts_map and enhanced_texts_map[i]:
                    train_texts.append(enhanced_texts_map[i])
                    train_labels.append(label)
                    augmented_count += 1
                
                # Always add original text
                train_texts.append(dataset.raw_texts[i])
                train_labels.append(label)
            else:
                # For non-mistake classes, use original text only
                train_texts.append(dataset.raw_texts[i])
                train_labels.append(label)
    
    # Val and test sets remain unchanged (no augmentation)
    val_texts = [dataset.raw_texts[i] for i in range(len(dataset.raw_texts)) if val_mask[i]]
    test_texts = [dataset.raw_texts[i] for i in range(len(dataset.raw_texts)) if test_mask[i]]
    
    val_labels = dataset.y[val_mask].tolist()
    test_labels = dataset.y[test_mask].tolist()
    
    print(f"\nAugmented Data Statistics:")
    print(f"  Train size: {len(train_labels)}, Val size: {len(val_labels)}, Test size: {len(test_labels)}")
    print(f"  Train label distribution: {Counter(train_labels)}")
    
    # Count original samples from top 3 mistake classes
    original_top3_count = sum(1 for i in range(len(dataset.y)) 
                              if train_mask[i] and int(dataset.y[i].item()) in top_3_class_ids)
    print(f"  Original samples from top 3 mistake classes: {original_top3_count}")
    print(f"  Added {augmented_count} augmented versions (GPT + enhanced texts)")
    print(f"  Total increase: {augmented_count} additional samples")
    
    # Tokenize
    train_encodings = encoding_texts_to_tokens(tokenizer, train_texts, cfg.dataset.max_length)
    val_encodings = encoding_texts_to_tokens(tokenizer, val_texts, cfg.dataset.max_length)
    test_encodings = encoding_texts_to_tokens(tokenizer, test_texts, cfg.dataset.max_length)
    
    # Create datasets
    train_dataset = TagsDataset(train_encodings, train_labels)
    val_dataset = TagsDataset(val_encodings, val_labels)
    test_dataset = TagsDataset(test_encodings, test_labels)
    
    return train_dataset, val_dataset, test_dataset

# ==========================================
# Retrain LLM with augmented data
# ==========================================
print("\n" + "="*80)
print("Retraining LLM with Augmented Data")
print("="*80)

# Setup configuration
cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
cfg.dataset.seed = seed

# Create save directory for retrained model
retrain_adapter_dir = os.path.join("./artifacts", 
                                    f"{llm_name}_{dataset_name}_seqcls_{peft_type}_init_{init_weight_approaches}_retrained_mistakes")
os.makedirs(retrain_adapter_dir, exist_ok=True)

# Setup reporter
reporter_save_dir = os.path.join("./results/train_info", 
                                 f"{dataset_name}_{llm_name}_{peft_type}_retrained_mistakes")
reporter = ReportResults(cfg, save_dir=reporter_save_dir)

# Report mistake analysis
reporter.report_title("GNN Mistake Analysis")
reporter.report_txt(f"Init Weight Approach: {init_weight_approaches}")
reporter.report_txt(f"Dataset: {dataset_name}")
reporter.report_txt(f"\nTop 3 Classes with Highest Mistake Rates:")
for cls_id, rate in top_3_mistake_classes:
    reporter.report_txt(f"  Class {cls_id}: {rate:.4f} ({class_mistakes[cls_id]}/{class_totals[cls_id]} mistakes)")

# Prepare augmented datasets
train_dataset, val_dataset, test_dataset = prepare_augmented_data_for_retraining(
    cfg, top_3_class_ids, enhanced_texts_map, gpt_responses_map, path_prefix
)

# Create custom trainer with prepared datasets
class CustomTrain(Train):
    """Custom trainer that uses pre-prepared datasets."""
    def __init__(self, cfg, save_dir, reporter, train_dataset, val_dataset, test_dataset):
        self.cfg = cfg
        self.reporter = reporter
        self.dataset_name = cfg.dataset.name
        self.seed = cfg.dataset.seed
        set_seed(cfg.dataset.seed)
        self.model_name = cfg.llm.model_name
        self.max_seq_length = cfg.tokenizer.max_length
        self.used_llm_responses = False
        
        from dataset.dataset_loader import get_tokenizer, load_dataset
        from train_llm.peft_model import load_llm_model, get_lora_model
        
        self.tokenizer = get_tokenizer(cfg)
        self.dataset = load_dataset(cfg)
        
        # Use provided datasets instead of loading new ones
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.num_classes = cfg.dataset.num_classes
        self.base_model = load_llm_model(cfg)
        self.save_dir = save_dir
        self.model = get_lora_model(self.base_model, cfg)

# Train with augmented data
trainer = CustomTrain(cfg, retrain_adapter_dir, reporter, train_dataset, val_dataset, test_dataset)
trainer.process()

print("\n" + "="*80)
print("Retraining Complete!")
print(f"Model saved to: {retrain_adapter_dir}")
print("="*80)

