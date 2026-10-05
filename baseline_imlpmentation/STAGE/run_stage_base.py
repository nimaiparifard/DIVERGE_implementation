"""
STAGE baseline on DIVERGE's setup (base LLM, no adapter):

  1) base-model embeddings of every node's raw text (the same cache TAPE/run_tape_base.py
     builds; built here if missing)
  2) per seed: train SAGE, RevGAT, GCN and GIN on them
  3) STAGE prediction = mean of the GNNs' logits

Per (dataset, llm, seed) one row is appended to --csv_path; --summary_path holds
mean/std over seeds per (dataset, llm).

The original-paper version (60/20/20 split, MLP member) is STAGE/run_stage.py.

Usage:
    python -m STAGE.run_stage_base --dataset_name cora --llm_name llama_3.2_1B --seeds 42 43 44 45
"""
import argparse
from datetime import datetime

from TAPE.base_pipeline import (DEFAULT_SEEDS, STAGE_GNN_MODELS, get_base_embeddings, load_split, train_gnn,
                                evaluate_mean, view_columns, mean_columns, append_row, write_summary)

DEFAULT_CSV = 'results/stage/stage_base_results.csv'
DEFAULT_SUMMARY = 'results/stage/stage_base_summary.csv'


def main(dataset_name, llm_name, seeds, gnn_models=STAGE_GNN_MODELS, supervised=True, csv_path=DEFAULT_CSV,
         summary_path=DEFAULT_SUMMARY, batch_size=None, rebuild_embeddings=False):
    gnn_models = list(gnn_models)
    print("=" * 80)
    print(f"STAGE (base model) | dataset={dataset_name} llm={llm_name} gnns={gnn_models} seeds={seeds} "
          f"split={'supervised' if supervised else 'semi_supervised'}")
    print("=" * 80)

    emb = get_base_embeddings(dataset_name, llm_name, 'text', batch_size, rebuild_embeddings)
    print(f"h_text={tuple(emb.shape)}")

    for seed in seeds:
        print(f"\n--- seed={seed}")
        data = load_split(dataset_name, seed, supervised)
        results, logits = {}, []
        for gnn in gnn_models:
            results[gnn], gnn_logits, _ = train_gnn(dataset_name, emb, gnn, seed, llm_name, data, supervised)
            logits.append(gnn_logits)
        stage = evaluate_mean(logits, data)
        stage_probs = evaluate_mean(logits, data, use_probs=True)
        print(f"  {'STAGE':<12} test_acc={stage['test']['acc']:.2f} test_f1={stage['test']['macro_f1']:.2f} "
              f"(prob-mean test_acc={stage_probs['test']['acc']:.2f})")

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'STAGE_base',
            'dataset_name': dataset_name,
            'llm_name': llm_name,
            'gnn_models': '+'.join(gnn_models),
            'seed': seed,
            'supervised': supervised,
            'n_train': int(data.train_mask.sum()), 'n_val': int(data.val_mask.sum()),
            'n_test': int(data.test_mask.sum()),
            'embedding_dim': emb.shape[1],
            **mean_columns('stage', stage),
            **mean_columns('stage_probmean', stage_probs),
        }
        for gnn in gnn_models:
            row.update(view_columns(gnn, results[gnn]))
        append_row(row, csv_path)
        write_summary(csv_path, summary_path, ['method', 'dataset_name', 'llm_name', 'gnn_models', 'supervised'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STAGE: GNN ensemble on base-model text embeddings')
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--gnn_models', nargs='+', default=list(STAGE_GNN_MODELS), choices=STAGE_GNN_MODELS + ('GAT',))
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--split', type=int, default=1, help='1 = supervised split (DIVERGE default), 0 = semi-supervised')
    parser.add_argument('--csv_path', type=str, default=DEFAULT_CSV)
    parser.add_argument('--summary_path', type=str, default=DEFAULT_SUMMARY)
    parser.add_argument('--batch_size', type=int, default=None, help='Embedding batch size (default: llm config)')
    parser.add_argument('--rebuild_embeddings', action='store_true', help='Ignore cached base-model embeddings')
    args = parser.parse_args()

    main(args.dataset_name, args.llm_name, args.seeds, gnn_models=args.gnn_models, supervised=args.split == 1,
         csv_path=args.csv_path, summary_path=args.summary_path, batch_size=args.batch_size,
         rebuild_embeddings=args.rebuild_embeddings)
