from gnns.gnn_mtrainer import get_datasets_path
from common import set_seed
from dataset.dataset_loader import prepare_data_for_finetuning, get_tokenizer, load_dataset, prepare_data_for_keyword_finetuning, prepare_data_for_finetuning_with_augmented_nodes
from train_llm.peft_model import load_llm_model, get_lora_model, get_model_path
from train_llm.training_config import get_training_config
from train_llm.efficiency import (
    append_efficiency_csv,
    build_efficiency_row,
    count_parameters,
    format_efficiency_block,
    measure_inference_latency,
    peak_vram_bytes,
    reset_cuda_peak_stats,
)
from transformers import Trainer
import torch
import os
import time
from peft import PeftModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

class Train:
    def __init__(self, cfg, save_dir, reporter,  used_llm_responses=False, used_summary_texts=False, used_paraphrased_texts=False, train_with_keywords=False, split=1, 
                 use_augmented_nodes=False, augmented_dataset=None, split_seed=None, init_seed=None):
        """
        split_seed: seed of the train/val/test split. None (default) uses cfg.dataset.seed,
                    i.e. the split and the training randomness share one seed (the original
                    behaviour). Set it to train several LoRA seeds on one fixed split.
        init_seed:  seed of the LoRA adapter initialization (random A/B draw of gaussian,
                    orthogonal and default-LoRA inits). None (default) draws it from
                    cfg.dataset.seed. Set it to give several training seeds the identical
                    starting adapter; the classification head, data order and dropout still
                    follow cfg.dataset.seed.
        """
        self.cfg = cfg
        self.reporter= reporter
        self.split = split
        self.dataset_name = cfg.dataset.name
        self.seed = cfg.dataset.seed
        self.split_seed = split_seed if split_seed is not None else self.seed
        self.init_seed = init_seed
        set_seed(cfg.dataset.seed)
        self.model_name = cfg.llm.model_name
        self.lora_init = getattr(cfg.peft, "init_lora_weights", "N/A")
        self.max_seq_length = cfg.tokenizer.max_length
        self.used_llm_responses = used_llm_responses
        self.efficiency_metrics = None
        self.tokenizer = get_tokenizer(cfg)
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
            path_prefix = '../..'
        self.dataset = load_dataset(cfg)
        
        # Store augmented nodes parameters
        self.use_augmented_nodes = use_augmented_nodes
        self.augmented_dataset = augmented_dataset
        
        if train_with_keywords:
            self.train_dataset, self.val_dataset, self.test_dataset = prepare_data_for_keyword_finetuning(cfg)
        elif use_augmented_nodes and augmented_dataset is not None:
            # Use augmented dataset for training
            # Calculate original_num_nodes from the dataset (before augmentation)
            original_num_nodes = len(self.dataset.raw_texts)
            self.train_dataset, self.val_dataset, self.test_dataset = prepare_data_for_finetuning_with_augmented_nodes(
                cfg, augmented_dataset, 
                original_num_nodes=original_num_nodes,
                used_llm_responses=self.used_llm_responses, 
                used_summary_texts=used_summary_texts, 
                used_paraphrased_texts=used_paraphrased_texts, 
                seed=self.split_seed, 
                path_prefix=path_prefix, 
                split=self.split
            )
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = prepare_data_for_finetuning(cfg, used_llm_responses=self.used_llm_responses, used_summary_texts=used_summary_texts, used_paraphrased_texts=used_paraphrased_texts, seed=self.split_seed, path_prefix=path_prefix, split=self.split)
        self.num_classes = cfg.dataset.num_classes
        # Pass tokenizer so pad_token_id is set on the classification model
        self.base_model = load_llm_model(cfg, tokenizer=self.tokenizer)
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.save_dir = save_dir
        if self.init_seed is not None:
            set_seed(self.init_seed)
        self.model = get_lora_model(self.base_model, cfg)
        if self.init_seed is not None:
            set_seed(cfg.dataset.seed)

    def process(self):
        self.reporter.report_title("Training Process Started")
        self.reporter.report_txt(f"Dataset: {self.dataset_name}")
        self.reporter.report_txt(f"Model: {self.model_name}")
        self.reporter.report_txt(f"PEFT Type: {self.cfg.peft.type}")
        self.reporter.report_txt(f"LoRA init: {self.lora_init}")
        self.reporter.report_txt(f"Split: {self.split}")
        self.reporter.report_txt(f"Seed: {self.seed}")
        self.reporter.report_txt(f"Split seed: {self.split_seed}")
        if self.init_seed is not None:
            self.reporter.report_txt(f"LoRA init seed: {self.init_seed}")
        
        self._train()
        save_dir = self.save_model()
        print("saving model to {}".format(save_dir))
        self.reporter.report_txt(f"\nModel saved to: {save_dir}")

        # Cost / efficiency (wall-clock, peak VRAM, adapter size, inference latency)
        self.collect_and_report_efficiency(adapter_dir=save_dir)
        
        self.plot_learning_curve()
        self.save_best_model_result()

    @staticmethod
    def compute_metrics(p):
        from sklearn.metrics import accuracy_score
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    def _train(self):
        training_args = get_training_config(self.cfg)
        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.val_dataset,  # evaluation dataset
            compute_metrics=self.compute_metrics,
        )
        self._train_wall_clock_sec = None
        self._peak_vram_bytes = None
        self._last_test_results = None
        try:
            self.reporter.report_title("Training Started")
            self.reporter.report_txt(f"Number of training samples: {len(self.train_dataset)}")
            self.reporter.report_txt(f"Number of validation samples: {len(self.val_dataset)}")
            self.reporter.report_txt(f"Number of test samples: {len(self.test_dataset)}")
            self.reporter.report_txt(f"Number of epochs: {training_args.num_train_epochs}")
            self.reporter.report_txt(f"Learning rate: {training_args.learning_rate}")
            self.reporter.report_txt(f"LoRA init: {self.lora_init}")

            reset_cuda_peak_stats()
            t0 = time.perf_counter()
            self.trainer.train()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._train_wall_clock_sec = time.perf_counter() - t0
            self._peak_vram_bytes = peak_vram_bytes()

            test_results = self.trainer.evaluate(eval_dataset=self.test_dataset)
            self._last_test_results = test_results
            test_acc = test_results.get('eval_accuracy', 'N/A')
            print(f"Test Accuracy: {test_acc:.4f}")
            print(
                f"[Efficiency] train wall-clock={self._train_wall_clock_sec:.1f}s, "
                f"peak VRAM={(self._peak_vram_bytes or 0) / (1024**2):.1f} MB"
            )
            
            self.reporter.report_title("Training Completed")
            self.reporter.report_txt(f"Test Accuracy: {test_acc:.4f}")
            self.reporter.report_txt(f"Test Loss: {test_results.get('eval_loss', 'N/A'):.4f}")
            self.reporter.report_txt(f"Wall-clock train (s): {self._train_wall_clock_sec:.2f}")
            if self._peak_vram_bytes is not None:
                self.reporter.report_txt(
                    f"Peak VRAM (MB): {self._peak_vram_bytes / (1024 ** 2):.2f}"
                )
        except Exception as e:
            error_msg = f"Training failed with error: {str(e)}"
            print(error_msg)
            self.reporter.report_title("Training Failed")
            self.reporter.report_txt(error_msg)

    def collect_and_report_efficiency(self, adapter_dir: str):
        """
        Build paper efficiency-table row for this (model × LoRA-init) run:
        train time, GPU-hours, peak VRAM, adapter storage, inference latency.
        Writes to reporter + run CSV + global efficiency CSV.
        """
        wall = getattr(self, "_train_wall_clock_sec", None)
        if wall is None:
            wall = float("nan")
        peak = getattr(self, "_peak_vram_bytes", None)
        trainable, total = count_parameters(self.model)

        try:
            inference_stats = measure_inference_latency(
                self.model,
                self.test_dataset,
                batch_size=min(8, max(1, len(self.test_dataset))),
                num_warmup=2,
                num_batches=15,
            )
        except Exception as e:
            print(f"[Efficiency] Inference latency measurement failed: {e}")
            inference_stats = {
                "inference_ms_per_sample": float("nan"),
                "inference_samples_per_sec": float("nan"),
                "inference_num_samples": 0.0,
                "inference_batch_size": float("nan"),
            }

        test_results = getattr(self, "_last_test_results", None) or {}
        row = build_efficiency_row(
            cfg=self.cfg,
            split=self.split,
            wall_clock_train_sec=float(wall),
            peak_vram=peak,
            adapter_dir=adapter_dir,
            inference_stats=inference_stats,
            trainable_params=trainable,
            total_params=total,
            test_accuracy=test_results.get("eval_accuracy", ""),
            test_loss=test_results.get("eval_loss", ""),
        )
        self.efficiency_metrics = row

        # Per-run CSV next to adapter / train_info
        run_csv = os.path.join(self.save_dir, "efficiency_metrics.csv")
        append_efficiency_csv(row, run_csv)
        row["report_csv"] = run_csv

        # Global table for all models × LoRA inits (paper efficiency table)
        global_csv = os.path.join("results", "efficiency", "lora_finetune_efficiency.csv")
        append_efficiency_csv(row, global_csv)

        self.reporter.report_title("Efficiency / Cost Report")
        for line in format_efficiency_block(row):
            self.reporter.report_txt(line)
        self.reporter.report_txt("")
        self.reporter.report_txt(f"Per-run efficiency CSV: {run_csv}")
        self.reporter.report_txt(f"Global efficiency CSV: {global_csv}")
        print(f"[OK] Efficiency metrics saved to: {run_csv}")
        print(f"[OK] Appended to global efficiency table: {global_csv}")
        return row

    def plot_learning_curve(self):
        """
            plot the learning curve for train and validation data in same plot
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'trainer') or self.trainer is None:
            print("No trainer available. Please run training first.")
            return
        
        # Get training history from trainer's log history
        log_history = self.trainer.state.log_history
        
        train_losses = []
        eval_losses = []
        train_steps = []
        eval_steps = []
        
        for entry in log_history:
            if 'loss' in entry:
                train_losses.append(entry['loss'])
                train_steps.append(entry.get('step', len(train_losses)))
            if 'eval_loss' in entry:
                eval_losses.append(entry['eval_loss'])
                eval_steps.append(entry.get('step', len(eval_losses)))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_steps, train_losses, label='Train Loss', marker='o')
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='s')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'Learning Curve - {self.model_name} on {self.dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'learning_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Learning curve saved to: {plot_path}")
        plt.close()

    def _subsample_dataset(self, dataset, max_size: int):
        """Return dataset unchanged if it already fits max_size, else a deterministic random subset."""
        n = len(dataset)
        if max_size <= 0 or n <= max_size:
            return dataset
        idxs = random.Random(self.seed).sample(range(n), max_size)
        return Subset(dataset, idxs)

    def save_best_model_result(self):
        """
            save the final loss and accuracy for best model for train validation test data
            also save before fine tuning results 
            save the result in csv file 
        """
        import csv
        from datetime import datetime
        
        if not hasattr(self, 'trainer') or self.trainer is None:
            print("No trainer available. Please run training first.")
            return
        
        csv_path = os.path.join(self.save_dir, 'training_results.csv')

        # Evaluate on all splits.
        # Train split is usually by far the largest, so we cap it to the size
        # of the bigger of val/test instead of running a full extra pass over it.
        train_eval_cap = max(len(self.val_dataset), len(self.test_dataset))
        train_eval_dataset = self._subsample_dataset(self.train_dataset, train_eval_cap)
        train_results = self.trainer.evaluate(eval_dataset=train_eval_dataset)
        val_results = self.trainer.evaluate(eval_dataset=self.val_dataset)
        # Test was already evaluated at the end of _train(); reuse it instead of re-running.
        test_results = getattr(self, "_last_test_results", None)
        if not test_results:
            test_results = self.trainer.evaluate(eval_dataset=self.test_dataset)

        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'peft_type': self.cfg.peft.type,
            'lora_init': self.lora_init,
            'split': self.split,
            'num_classes': self.num_classes,
            'max_seq_length': self.max_seq_length,
            'train_eval_sample_size': len(train_eval_dataset),
            'train_loss': train_results.get('eval_loss', 'N/A'),
            'train_accuracy': train_results.get('eval_accuracy', 'N/A'),
            'val_loss': val_results.get('eval_loss', 'N/A'),
            'val_accuracy': val_results.get('eval_accuracy', 'N/A'),
            'test_loss': test_results.get('eval_loss', 'N/A'),
            'test_accuracy': test_results.get('eval_accuracy', 'N/A'),
        }
        if self.efficiency_metrics:
            for key in (
                'wall_clock_train_sec',
                'wall_clock_train_min',
                'gpu_hours',
                'peak_vram_mb',
                'peak_vram_gb',
                'adapter_dir_mb',
                'adapter_weights_mb',
                'inference_ms_per_sample',
                'inference_samples_per_sec',
                'trainable_params',
                'total_params',
                'trainable_pct',
            ):
                results[key] = self.efficiency_metrics.get(key, '')
        
        # Write to CSV
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        
        print(f"[OK] Training results saved to: {csv_path}")
        print(f"Train Acc: {results['train_accuracy']:.4f}, Val Acc: {results['val_accuracy']:.4f}, Test Acc: {results['test_accuracy']:.4f}")
        
        # Report final results using reporter
        self.reporter.report_title("Final Results Summary")
        self.reporter.report_txt(f"Timestamp: {results['timestamp']}")
        self.reporter.report_txt(f"Model: {results['model_name']}")
        self.reporter.report_txt(f"Dataset: {results['dataset_name']}")
        self.reporter.report_txt(f"PEFT Type: {results['peft_type']}")
        self.reporter.report_txt(f"LoRA init: {results['lora_init']}")
        self.reporter.report_txt(f"Split: {results['split']}")
        self.reporter.report_txt(f"Number of Classes: {results['num_classes']}")
        self.reporter.report_txt(f"Max Sequence Length: {results['max_seq_length']}")
        self.reporter.report_txt("")
        self.reporter.report_txt("--- Train Set ---")
        self.reporter.report_txt(f"Loss: {results['train_loss']:.4f}")
        self.reporter.report_txt(f"Accuracy: {results['train_accuracy']:.4f}")
        self.reporter.report_txt("")
        self.reporter.report_txt("--- Validation Set ---")
        self.reporter.report_txt(f"Loss: {results['val_loss']:.4f}")
        self.reporter.report_txt(f"Accuracy: {results['val_accuracy']:.4f}")
        self.reporter.report_txt("")
        self.reporter.report_txt("--- Test Set ---")
        self.reporter.report_txt(f"Loss: {results['test_loss']:.4f}")
        self.reporter.report_txt(f"Accuracy: {results['test_accuracy']:.4f}")
        if self.efficiency_metrics:
            self.reporter.report_txt("")
            self.reporter.report_txt("--- Efficiency (see Efficiency / Cost Report) ---")
            self.reporter.report_txt(
                f"GPU-hours: {self.efficiency_metrics.get('gpu_hours', float('nan')):.6f} | "
                f"Peak VRAM GB: {self.efficiency_metrics.get('peak_vram_gb', float('nan')):.3f} | "
                f"Adapter MB: {self.efficiency_metrics.get('adapter_weights_mb', float('nan')):.2f} | "
                f"Infer ms/sample: {self.efficiency_metrics.get('inference_ms_per_sample', float('nan')):.3f}"
            )
        self.reporter.report_txt("")
        self.reporter.report_txt(f"Results CSV saved to: {csv_path}")
        
        return results

    # =========================
    # SAVE: adapter + tokenizer
    # =========================
    def save_model(self):
        """
        Saves the LoRA adapter (including modules_to_save like 'score') and the tokenizer.
        For PEFT models, model.save_pretrained(save_dir) stores adapter weights + modules_to_save.
        """
        # Use self.save_dir for saving
        os.makedirs(self.save_dir, exist_ok=True)

        # Save adapter + modules_to_save (e.g., 'score')
        # IMPORTANT: call on the PEFT-wrapped model (self.model), not the base.
        self.model.save_pretrained(self.save_dir)

        # Save tokenizer (keep pad/eos alignment consistent)
        self.tokenizer.save_pretrained(self.save_dir)

        print(f"[OK] Saved adapter + modules_to_save to: {self.save_dir}")

        return self.save_dir



# ==========================================
# LOAD: base model + attach saved PEFT adapter
# ==========================================
def load_llm_with_lora_adapter(
    adapter_dir: str,
    cfg,
):
    # Load tokenizer from the base model, not the adapter directory
    base_model_path = get_model_path(cfg.llm.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        local_files_only=cfg.llm.local_files_only, 
        trust_remote_code=cfg.llm.trust_remote_code, 
        truncation=cfg.tokenizer.truncation, 
        padding=cfg.tokenizer.padding, 
        max_length=cfg.tokenizer.max_length
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = cfg.tokenizer.padding_side
    
    # Load base model
    base_model = load_llm_model(cfg, tokenizer=tokenizer)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load the LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.eval()
    print(f"[OK] Loaded base model '{cfg.llm.model_name}' and attached adapter from: {adapter_dir}")
    return peft_model, tokenizer

import random
import torch

def evaluate_sampled_examples(model, tokenizer, dataset_obj, cfg, k: int = 10, seed: int = 42):
    """
    dataset_obj is the *original* object with attributes:
      - raw_texts: list[str]
      - y: torch.LongTensor or list[int]
    """
    model.eval()
    device = next(model.parameters()).device

    n = len(dataset_obj.raw_texts)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:min(k, n)]

    correct = 0
    with torch.no_grad():
        for i in idxs:
            text = dataset_obj.raw_texts[i]
            label = int(dataset_obj.y[i].item() if torch.is_tensor(dataset_obj.y[i]) else dataset_obj.y[i])

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=cfg.tokenizer.truncation,
                max_length=cfg.tokenizer.max_length,
                padding=False,   # single example; no need to pad
            )
            enc = {k: v.to(device) for k, v in enc.items()}  # <-- NO unsqueeze

            out = model(**enc)
            logits = out.logits                    # [1, num_labels]
            pred = int(logits.argmax(dim=-1).item())

            correct += int(pred == label)

    acc = correct / max(len(idxs), 1)
    print(f"[Eval] Sampled {len(idxs)} examples — Accuracy: {acc:.4f}")
    return acc

def evaluate_batch_on_dataset(model, dataset, batch_size: int = 32):
    """
    dataset is your TagsDataset that yields:
      {'input_ids', 'attention_mask', 'labels'}
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits          # [B, num_labels]
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"[Eval] {total} examples — Accuracy: {acc:.4f}")
    return acc




if __name__ == "__main__":
    # Test with a fresh model instead of loading adapter
    from train_llm.peft_model import load_llm_model, get_lora_model
    from dataset.dataset_loader import load_dataset, get_tokenizer
    from config import setup_finetuning_cfg
    from adapter_dir import get_semi_supervised_adapter_dir
    from report.reporter import ReportResults
    
    cfg = setup_finetuning_cfg(dataset_name="cora", llm_name="llama_3.2_1B", peft_type="lora")
    adapter_dir = get_semi_supervised_adapter_dir(cfg)
    reporter = ReportResults(cfg, save_dir='results/train_info/test_run')
    train = Train(cfg, save_dir=adapter_dir, reporter=reporter, used_llm_responses=True, split=0)
    train.process()
    # Load dataset and test evaluation
    cora_dataset = load_dataset(cfg)
    model, tokenizer = load_llm_with_lora_adapter(adapter_dir, cfg)
    _ = evaluate_sampled_examples(model, tokenizer, cora_dataset, cfg, k=20, seed=42)