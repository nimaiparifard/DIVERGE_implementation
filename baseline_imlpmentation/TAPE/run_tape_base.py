"""
TAPE baseline on DIVERGE's setup (base LLM, no adapter):

  1) base-model embeddings of every node's raw text            -> h_text
  2) base-model embeddings of every node's GPT-4o response
     (artifacts/gpt_4o/<dataset>.json)                          -> h_aug
  3) per seed and GNN: train one GNN on h_text and one on h_aug
  4) TAPE prediction = mean of the two GNNs' logits

Embeddings are seed-independent (nothing is fine-tuned) and built once per (dataset, llm);
seeds only change the GNN init and the DIVERGE train/val/test split.

Per (dataset, llm, gnn, seed) one row is appended to --csv_path; --summary_path holds
mean/std over seeds per (dataset, llm, gnn). The per-node predictions of every run are saved to
--preds_dir/<dataset>_<llm>_<gnn>_seed<seed>.npz for the paired tests of
experiments/significance_vs_baselines.py.

The paper-faithful version (LoRA-finetuned LMs + prediction features) is TAPE/run_tape.py.

Usage:
    python -m TAPE.run_tape_base --dataset_name cora --llm_name llama_3.2_1B --gnn_models SAGE GCN --seeds 42 43 44 45
"""
import argparse
import os
from datetime import datetime

import numpy as np
import torch

from TAPE.base_pipeline import (DEFAULT_SEEDS, STAGE_GNN_MODELS, get_base_embeddings, load_split, train_gnn,
                                evaluate_mean, view_columns, mean_columns, append_row, write_summary)

DEFAULT_CSV = 'results/tape/tape_base_results.csv'
DEFAULT_SUMMARY = 'results/tape/tape_base_summary.csv'
DEFAULT_PREDS_DIR = 'results/tape/preds'


def preds_path(preds_dir, dataset_name, llm_name, gnn, seed):
    return os.path.join(preds_dir, f'{dataset_name}_{llm_name}_{gnn}_seed{seed}.npz')


def save_preds(path, data, logits):
    """Per-node argmax of each {name: logits [n, C]}; 'tape' is the mean of the two views' logits."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, y=data.y.view(-1).cpu().numpy(), test_idx=torch.where(data.test_mask)[0].cpu().numpy(),
             val_idx=torch.where(data.val_mask)[0].cpu().numpy(),
             **{f'pred_{k}': v.argmax(dim=-1).numpy() for k, v in logits.items()})


def main(dataset_name, llm_name, gnn_models, seeds, supervised=True, csv_path=DEFAULT_CSV,
         summary_path=DEFAULT_SUMMARY, batch_size=None, rebuild_embeddings=False, preds_dir=DEFAULT_PREDS_DIR):
    print("=" * 80)
    print(f"TAPE (base model) | dataset={dataset_name} llm={llm_name} gnns={gnn_models} seeds={seeds} "
          f"split={'supervised' if supervised else 'semi_supervised'}")
    print("=" * 80)

    emb_text = get_base_embeddings(dataset_name, llm_name, 'text', batch_size, rebuild_embeddings)
    emb_aug = get_base_embeddings(dataset_name, llm_name, 'augmented', batch_size, rebuild_embeddings)
    print(f"h_text={tuple(emb_text.shape)} h_aug={tuple(emb_aug.shape)}")

    for seed in seeds:
        data = load_split(dataset_name, seed, supervised)
        for gnn in gnn_models:
            print(f"\n--- seed={seed} gnn={gnn}")
            res_text, logits_text, cfg = train_gnn(dataset_name, emb_text, gnn, seed, llm_name, data, supervised,
                                                      label='text')
            res_aug, logits_aug, _ = train_gnn(dataset_name, emb_aug, gnn, seed, llm_name, data, supervised,
                                               label='aug')
            tape = evaluate_mean([logits_text, logits_aug], data)
            tape_probs = evaluate_mean([logits_text, logits_aug], data, use_probs=True)
            print(f"  {'TAPE':<12} test_acc={tape['test']['acc']:.2f} test_f1={tape['test']['macro_f1']:.2f} "
                  f"(prob-mean test_acc={tape_probs['test']['acc']:.2f})")

            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'TAPE_base',
                'dataset_name': dataset_name,
                'llm_name': llm_name,
                'gnn_model_name': gnn,
                'seed': seed,
                'supervised': supervised,
                'n_train': int(data.train_mask.sum()), 'n_val': int(data.val_mask.sum()),
                'n_test': int(data.test_mask.sum()),
                'embedding_dim': emb_text.shape[1],
                'hp_source': os.path.basename(str(cfg.hp_source)),
                'hidden_dim': cfg.hidden_dim, 'num_layers': cfg.num_layers, 'dropout': cfg.dropout,
                'lr': cfg.lr, 'weight_decay': cfg.weight_decay,
                **mean_columns('tape', tape),
                **mean_columns('tape_probmean', tape_probs),
                **view_columns('text', res_text),
                **view_columns('aug', res_aug),
            }
            append_row(row, csv_path)
            save_preds(preds_path(preds_dir, dataset_name, llm_name, gnn, seed), data,
                       {'tape': (logits_text + logits_aug) / 2, 'text': logits_text, 'aug': logits_aug})
        write_summary(csv_path, summary_path, ['method', 'dataset_name', 'llm_name', 'gnn_model_name', 'supervised'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TAPE with base-model embeddings of raw + GPT-4o-augmented text')
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--gnn_models', nargs='+', default=['SAGE'], choices=STAGE_GNN_MODELS + ('GAT',))
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--split', type=int, default=1, help='1 = supervised split (DIVERGE default), 0 = semi-supervised')
    parser.add_argument('--csv_path', type=str, default=DEFAULT_CSV)
    parser.add_argument('--summary_path', type=str, default=DEFAULT_SUMMARY)
    parser.add_argument('--preds_dir', type=str, default=DEFAULT_PREDS_DIR, help='Per-node predictions (.npz)')
    parser.add_argument('--batch_size', type=int, default=None, help='Embedding batch size (default: llm config)')
    parser.add_argument('--rebuild_embeddings', action='store_true', help='Ignore cached base-model embeddings')
    args = parser.parse_args()

    main(args.dataset_name, args.llm_name, args.gnn_models, args.seeds, supervised=args.split == 1,
         csv_path=args.csv_path, summary_path=args.summary_path, batch_size=args.batch_size,
         rebuild_embeddings=args.rebuild_embeddings, preds_dir=args.preds_dir)
