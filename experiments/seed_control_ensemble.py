"""
Seed-control ensemble: is DIVERGE's gain due to initialization diversity or just to ensembling?
(reviewers R2-C2, R4-C1, R1-C7b)

For every split seed s, two pools of M=5 LoRA runs on the SAME train/val/test split:
  DIVERGE : the 5 initializations (PiSSA, Orthogonal, Gaussian, LoftQ, EVA), LoRA seed = s
            (the adapters/caches of the main pipeline: ..._seed{s}_semi_supervised.pt)
  control : one initialization (default Gaussian) x 5 LoRA seeds s+100, ..., s+500, fine-tuned
            on split s with --split_seed s (..._seed{s+o}_split{s}_semi_supervised.pt)
Both pools get the same GNN (hyperparameters, seed s, split s) and the same ensemble heads
(uniform + the 3 learned heads, configuration AND epoch selected on validation accuracy), so the
only difference is "5 initializations" vs "5 seeds of one initialization".

The control members also differ in the HF Trainer seed (data order, dropout), whereas the
existing DIVERGE members all used the Trainer default (42). The control therefore has at least
as much training randomness as DIVERGE, which makes the comparison conservative.

Outputs (results/seed_control/):
  seed_control_results.csv               one row per (dataset, llm, gnn, split, pool, method)
  <dataset>_<llm>_<gnn>_<ctrl>_report.txt  per-split table + paired comparison over splits
  <dataset>_<llm>_<gnn>_<ctrl>_preds.pt    test predictions of every method (for later tests)

Prerequisites: scripts/run_seed_control.bat (trains + caches the control members).

Example:
    python -m experiments.seed_control_ensemble --dataset_name cora --llm_name llama_3.2_1B
"""
import os
import argparse
from datetime import datetime

import numpy as np
import torch
from scipy import stats

from config import setup_finetuning_cfg
from dataset.data_utils import get_cached_dir
from adapter_dir import lora_run_tag
from gnns.gnn_mtrainer import GNNTrainer, set_mgnn_cfg, append_gnn_result_csv, DEFAULT_GNN_MODEL
from ensemble.ensemble_gnns_learning import ensemble_gnn_learning
from experiments.diversity_common import cka_distance_matrix

APPROACHES = ['pissa', 'orthogonal', 'gaussian', 'loftq', 'eva']
HEADS = ['uniform', 'learnable', 'learnable_classes', 'learnable_per_classes']
HEAD_TITLES = {'uniform': 'Uniform', 'learnable': 'Method 1', 'learnable_classes': 'Method 2',
               'learnable_per_classes': 'Method 3'}
# Same search space as ensemble.ensemble_gnns_learning.tune_ensemble_hyperparameter
HEAD_GRID = [(l, h, d, lr) for l in (2, 3) for h in (64, 128, 256) for d in (0.3, 0.5)
             for lr in (0.001, 0.01, 0.0001)]
OUT_DIR = 'results/seed_control'


def cache_path(cache_dir, model_name, dataset, peft, init, seed, split_seed):
    return os.path.join(cache_dir, f'{model_name}_{dataset}_seqcls_{peft}_init-{init}_pool-mean'
                                   f'{lora_run_tag(seed, split_seed)}_semi_supervised.pt')


def pool_members(args, split_seed):
    """[(label, init, lora_seed, cache path)] for both pools of one split."""
    cfg = setup_finetuning_cfg(args.dataset_name, args.llm_name, args.peft_type)
    cache_dir = get_cached_dir()
    mk = lambda init, seed: cache_path(cache_dir, cfg.llm.model_name, args.dataset_name, args.peft_type,
                                       init, seed, split_seed)
    diverge = [(f'{i}-{split_seed}', i, split_seed, mk(i, split_seed)) for i in APPROACHES]
    control = [(f'{args.control_init}-{split_seed + o}', args.control_init, split_seed + o,
                mk(args.control_init, split_seed + o)) for o in args.control_offsets]
    return {'diverge': diverge, 'control': control}


def train_members(args, members, split_seed):
    """Train one GNN per member on split `split_seed`. Returns logits [M, n, C], embeddings, results, data."""
    gnn_cfg = set_mgnn_cfg(args.dataset_name, supervised=True, seed=split_seed, llm_name=args.llm_name,
                           gnn_model_name=args.gnn_model_name)
    logits, embs, results, data = [], [], [], None
    for label, _, _, path in members:
        emb = torch.load(path, map_location='cpu', weights_only=False)['embeddings'].float()
        trainer = GNNTrainer(gnn_cfg, emb, verbose=False, data=data)
        res, best_logits, _ = trainer.train()
        data = trainer.data  # identical split for every member of this split seed
        logits.append(best_logits.detach().float())
        embs.append(emb)
        results.append(res)
        print(f'  [{label:<16}] val {res["val_acc"]:.2f}  test {res["test_acc"]:.2f}')
        del trainer
        torch.cuda.empty_cache()
    return torch.stack(logits), embs, results, data, gnn_cfg.gnn_model_name


def fit_head(args, head, logits, split_seed):
    """Fit one ensemble head; configuration and epoch selected on validation accuracy."""
    preds_list = [l for l in logits]
    grid = HEAD_GRID if head != 'uniform' else [(2, 64, 0.3, 0.01)]
    best = None
    for (nl, hd, dp, lr) in grid:
        preds, res, _ = ensemble_gnn_learning(
            model_list=preds_list, dataset_name=args.dataset_name, weight_approach=head,
            num_layers=nl, hidden_dim=hd, dropout=dp, learning_rate=lr, epochs=200,
            predictions_list=preds_list, supervised=True, seed=split_seed, llm_name=args.llm_name,
            select_split='val')
        if best is None or res['val']['accuracy'] > best[1]['val']['accuracy']:
            best = (preds.detach().cpu(), res, dict(num_layers=nl, hidden_dim=hd, dropout=dp, learning_rate=lr))
    return best


def pool_summary(logits, embs, results, data):
    y = data.y.cpu()
    test = data.test_mask.cpu()
    preds = logits.argmax(dim=2).cpu()                    # [M, n]
    correct = preds[:, test] == y[test]                   # [M, n_test]
    M = preds.shape[0]
    dis = [(preds[i, test] != preds[j, test]).float().mean().item() for i in range(M) for j in range(i + 1, M)]
    cka = cka_distance_matrix(embs)
    best_val = int(np.argmax([r['val_acc'] for r in results]))
    return {
        'member_test_acc': [r['test_acc'] for r in results],
        'mean_member_acc': float(np.mean([r['test_acc'] for r in results])),
        'best_member_by_val_acc': results[best_val]['test_acc'],
        'best_member_by_test_acc': float(max(r['test_acc'] for r in results)),
        'oracle_acc': 100.0 * correct.any(dim=0).float().mean().item(),
        'disagreement': float(np.mean(dis)),
        'llm_cka_distance': float(cka[np.triu_indices(M, 1)].mean()),
    }


def mcnemar_exact(a_correct, b_correct):
    """Two-sided exact McNemar test. Returns (b, c, p): b = only A correct, c = only B correct."""
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def main(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    gnn_tag = args.gnn_model_name or DEFAULT_GNN_MODEL
    tag = f'{args.dataset_name}_{args.llm_name}_{gnn_tag}_{args.control_init}'
    report_path = os.path.join(out_dir, f'{tag}_report.txt')
    csv_path = os.path.join(out_dir, 'seed_control_results.csv')
    lines = [f'Seed-control ensemble | {args.dataset_name} | {args.llm_name} | GNN {gnn_tag}',
             f'{datetime.now():%Y-%m-%d %H:%M:%S}',
             f'DIVERGE pool : {", ".join(APPROACHES)} (LoRA seed = split seed)',
             f'Control pool : {args.control_init} x LoRA seeds split+{args.control_offsets} (same split)',
             'Heads: configuration and epoch selected on validation accuracy', '']

    # check every cache exists before training anything
    all_members = {s: pool_members(args, s) for s in args.split_seeds}
    missing = [p for s in all_members for pool in all_members[s].values() for *_, p in pool if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('Missing embedding caches (run scripts/run_seed_control.bat):\n  ' + '\n  '.join(missing))

    per_split, saved = {}, {}
    for s in args.split_seeds:
        print(f'\n{"=" * 80}\nSplit seed {s}\n{"=" * 80}')
        per_split[s], saved[s] = {}, {}
        test_correct = {}
        for pool, members in all_members[s].items():
            print(f'-- pool {pool}')
            logits, embs, results, data, gnn_name = train_members(args, members, s)
            summ = pool_summary(logits, embs, results, data)
            y_test = data.y.cpu()[data.test_mask.cpu()]
            heads = {}
            for head in HEADS:
                preds, res, params = fit_head(args, head, logits, s)
                heads[head] = {'val_acc': res['val']['accuracy'], 'test_acc': res['test']['accuracy'],
                               'test_macro_f1': res['test']['macro_f1'], 'params': params}
                test_correct[(pool, head)] = (preds[data.test_mask.cpu()] == y_test).numpy()
                saved[s][(pool, head)] = preds[data.test_mask.cpu()].numpy()
                print(f'   {HEAD_TITLES[head]:<9} val {res["val"]["accuracy"]:.2f}  test {res["test"]["accuracy"]:.2f}  '
                      f'F1 {res["test"]["macro_f1"]:.2f}')
            per_split[s][pool] = {'summary': summ, 'heads': heads,
                                  'members': [m[0] for m in members]}
            saved[s]['y_test'] = y_test.numpy()
            for head, h in heads.items():
                append_gnn_result_csv({
                    'timestamp': f'{datetime.now():%Y-%m-%d %H:%M:%S}', 'dataset_name': args.dataset_name,
                    'llm_name': args.llm_name, 'gnn_model_name': gnn_name, 'split_seed': s, 'pool': pool,
                    'control_init': args.control_init, 'members': ';'.join(m[0] for m in members),
                    'method': head, 'val_accuracy': h['val_acc'], 'test_accuracy': h['test_acc'],
                    'test_macro_f1': h['test_macro_f1'], 'mean_member_acc': summ['mean_member_acc'],
                    'best_member_by_val_acc': summ['best_member_by_val_acc'], 'oracle_acc': summ['oracle_acc'],
                    'disagreement': summ['disagreement'], 'llm_cka_distance': summ['llm_cka_distance'],
                    'head_params': h['params'], 'select_split': 'val',
                }, csv_path)
        # per-split McNemar DIVERGE vs control
        per_split[s]['mcnemar'] = {h: mcnemar_exact(test_correct[('diverge', h)], test_correct[('control', h)])
                                   for h in HEADS}
        torch.save({'per_split': per_split, 'preds': saved, 'args': vars(args)},
                   os.path.join(out_dir, f'{tag}_preds.pt'))

    # ------------------------------------------------------------------ report
    def row(name, fn):
        vals = {p: [fn(per_split[s][p]) for s in args.split_seeds] for p in ('diverge', 'control')}
        d = np.array(vals['diverge']) - np.array(vals['control'])
        t = stats.ttest_rel(vals['diverge'], vals['control']).pvalue if len(d) > 1 and np.any(d != d[0]) else float('nan')
        return (f'{name:<28} {np.mean(vals["diverge"]):6.2f} +/- {np.std(vals["diverge"], ddof=1) if len(d) > 1 else 0:4.2f}   '
                f'{np.mean(vals["control"]):6.2f} +/- {np.std(vals["control"], ddof=1) if len(d) > 1 else 0:4.2f}   '
                f'{d.mean():+6.2f}  [{", ".join(f"{x:+.2f}" for x in d)}]  p={t:.3g}')

    lines.append(f'Mean over split seeds {args.split_seeds} (mean +/- std); diff = DIVERGE - control; '
                 f'p = paired t-test over splits')
    lines.append(f'{"":<28} {"DIVERGE":>15}   {"control":>15}   {"diff":>6}  per split')
    lines.append(row('Mean member (test acc)', lambda r: r['summary']['mean_member_acc']))
    lines.append(row('Best member by val', lambda r: r['summary']['best_member_by_val_acc']))
    lines.append(row('Oracle (any member)', lambda r: r['summary']['oracle_acc']))
    lines.append(row('Disagreement (%)', lambda r: 100 * r['summary']['disagreement']))
    lines.append(row('1 - CKA, LLM emb (x100)', lambda r: 100 * r['summary']['llm_cka_distance']))
    for h in HEADS:
        lines.append(row(f'{HEAD_TITLES[h]} acc', lambda r, h=h: r['heads'][h]['test_acc']))
        lines.append(row(f'{HEAD_TITLES[h]} macro-F1', lambda r, h=h: r['heads'][h]['test_macro_f1']))
        lines.append(row(f'{HEAD_TITLES[h]} gain over mean member',
                         lambda r, h=h: r['heads'][h]['test_acc'] - r['summary']['mean_member_acc']))
    lines.append('')
    lines.append('McNemar DIVERGE vs control per split (exact, two-sided): b = only DIVERGE correct, '
                 'c = only control correct; pooled = sum over splits')
    for h in HEADS:
        parts = [f's{s}: b={per_split[s]["mcnemar"][h][0]} c={per_split[s]["mcnemar"][h][1]} '
                 f'p={per_split[s]["mcnemar"][h][2]:.3g}' for s in args.split_seeds]
        b = sum(per_split[s]['mcnemar'][h][0] for s in args.split_seeds)
        c = sum(per_split[s]['mcnemar'][h][1] for s in args.split_seeds)
        p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        lines.append(f'  {HEAD_TITLES[h]:<9} ' + ' | '.join(parts) + f' || pooled b={b} c={c} p={p:.3g}')
    lines.append('')
    lines.append('Members (test acc):')
    for s in args.split_seeds:
        for p in ('diverge', 'control'):
            r = per_split[s][p]
            lines.append(f'  split {s} {p:<8} ' + ', '.join(
                f'{m}={a:.2f}' for m, a in zip(r['members'], r['summary']['member_test_acc'])))
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f'\nReport: {report_path}\nCSV   : {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--gnn_model_name', type=str, default=None,
                        help='GNN encoder (default: the one in the tuned hyperparameter JSON, SAGE)')
    parser.add_argument('--split_seeds', type=str, default='42,43,44,45')
    parser.add_argument('--control_init', type=str, default='gaussian', choices=APPROACHES)
    parser.add_argument('--control_offsets', type=str, default='100,200,300,400,500',
                        help='LoRA seed of control member k = split seed + offset k')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    a = parser.parse_args()
    a.split_seeds = [int(x) for x in a.split_seeds.split(',')]
    a.control_offsets = [int(x) for x in a.control_offsets.split(',')]
    main(a)
