"""
Does fine-tuning the LLM improve the downstream GNN? (paper Table tab:mcnemar)

Per dataset, one GNN (SAGE, tuned hyperparameters) on the base (not fine-tuned) LLM embeddings and one on
the fine-tuned LoRA embeddings, both trained and tested on the split the adapter was fine-tuned on (the
leak-free primary run of significance_vs_baselines.RUNS). The fine-tuned model is the LoRA view with the
best validation accuracy (--finetuned fixes an init instead), so the test nodes are not used to pick it.
The GNNs are the ones significance_vs_baselines.py uses (shared cache in results/ensemble_experiments).

McNemar on the test nodes, b = only the base model correct, c = only the fine-tuned model correct:
    b + c >= 25: chi2 = (|b - c| - 1)^2 / (b + c)  (Edwards' continuity correction), p from chi2(1)
    b + c <  25: exact two-sided binomial test, statistic = min(b, c)
p-values are adjusted with Bonferroni and Holm over all (dataset, split) tests; effect sizes (c - b) / n and c / b.
The table shows, per dataset, the most significant split when one survives Holm, else the primary split.
Every LoRA view against the base model is also written (unadjusted, CSV only).

With --all_splits 1 (default) every split that has all five adapters is tested (Cora/CiteSeer 42-45;
PubMed, WikiCS and ArXiv have one); the table uses the primary split, the other splits give the
spread of the gain and a Holm correction over all runs.

Outputs (results/finetuning_significance/): mcnemar.csv (primary runs), mcnemar_all_splits.csv,
mcnemar_all_views.csv, report.txt (with the LaTeX rows of Table tab:mcnemar).

Usage:
    python -m experiments.hypothesis_testing_fintuned_unfinetuned --datasets cora,citeseer,pubmed,wikics,arxiv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from experiments.diversity_common import TextReport
from experiments import significance_vs_baselines as sig
from experiments.significance_vs_baselines import holm, load_views_and_base

OUT_DIR = 'results/finetuning_significance'
ALPHA = 0.05
EXACT_BELOW = 25
# Leak-free runs of significance_vs_baselines plus Computer and Reddit (adapters fine-tuned on split 42)
RUNS = {**sig.RUNS, 'computer': [42], 'reddit': [42]}
DATASETS = list(RUNS)
LATEX_NAMES = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'wikics': 'WikiCS', 'arxiv': 'ArXiv',
               'computer': 'Computer', 'reddit': 'Reddit'}


def mcnemar(base_ok, ft_ok):
    """Contingency counts, statistic and p-value (Edwards-corrected chi2, exact binomial when b + c < 25)."""
    b = int((base_ok & ~ft_ok).sum())
    c = int((~base_ok & ft_ok).sum())
    n = b + c
    if n == 0:
        stat, p, test = 0.0, 1.0, 'none'
    elif n < EXACT_BELOW:
        stat, p, test = float(min(b, c)), float(stats.binomtest(c, n, 0.5).pvalue), 'exact'
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p, test = float(stats.chi2.sf(stat, 1)), 'chi2_cc'
    return {'both_correct': int((base_ok & ft_ok).sum()), 'b_base_only': b, 'c_finetuned_only': c,
            'both_wrong': int((~base_ok & ~ft_ok).sum()), 'discordant': n, 'statistic': stat, 'p_value': p,
            'test': test, 'acc_diff_pp': (c - b) / len(base_ok) * 100, 'ratio_c_b': c / b if b else np.inf}


def collect(dataset, seed):
    """Per-node correctness on the test nodes of the base model and of every LoRA view."""
    vs, views, base = load_views_and_base(dataset, seed)
    if base is None:
        raise FileNotFoundError(f'{dataset}: no base-model embeddings in artifacts/cache '
                                f'(build them with scripts/run_tape_base.bat or cache/cache_embedding.py)')
    test = views['test_mask'].numpy()
    y = views['y'].numpy()[test]
    ok = {a: views['logits'][i].argmax(1).numpy()[test] == y for i, a in enumerate(vs.approaches)}
    val_acc = dict(zip(vs.approaches, views['val_acc'] * 100))
    test_acc = dict(zip(vs.approaches, views['test_acc'] * 100))
    return base.argmax(1).numpy()[test] == y, ok, val_acc, test_acc


def fmt_p(p):
    if p >= 1e-3:
        return f'${p:.4f}$' if p < 0.01 else f'${p:.3f}$'
    m, e = f'{p:.2e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def latex_rows(df):
    cols = list(df.dataset)
    rows = [
        ('Split (seed)', [f'{s}' + ('' if p else r'$^\dagger$') for s, p in zip(df.split_seed, df.primary_run)]),
        ('Both Correct', [f'{v}' for v in df.both_correct]),
        ('Base Model Only Correct ($b$)', [f'{v}' for v in df.b_base_only]),
        ('Finetuned Only Correct ($c$)', [f'{v}' for v in df.c_finetuned_only]),
        ('Both Wrong', [f'{v}' for v in df.both_wrong]),
        ('Discordant Pairs ($b+c$)', [f'{v}' for v in df.discordant]),
        ('McNemar Statistic', [f'${v:.3f}$' for v in df.statistic]),
        ('$p$-value', [fmt_p(v) for v in df.p_value]),
        ('Bonferroni-adjusted $p$', [fmt_p(v) for v in df.p_bonferroni]),
        ('Holm-adjusted $p$', [fmt_p(v) for v in df.p_holm]),
        ('Accuracy difference $(c-b)/n$ (pp)', [f'${v:+.2f}$' for v in df.acc_diff_pp]),
        ('Discordant ratio $c/b$', [f'${v:.2f}$' for v in df.ratio_c_b]),
        ('\\textbf{Result} (Holm)', ['Reject $H_0$' if r else 'Fail to reject $H_0$' for r in df.reject_holm]),
    ]
    head = 'Metric & ' + ' & '.join(f'\\textbf{{{LATEX_NAMES.get(d, d)}}}' for d in cols) + ' \\\\'
    return [head] + [f'{name} & ' + ' & '.join(vals) + ' \\\\' for name, vals in rows]


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = [d for d in DATASETS if d in args.datasets.split(',')]
    report = TextReport(os.path.join(OUT_DIR, 'report.txt'),
                        'Fine-tuned vs base LLM embeddings (GNN on each, McNemar on the same test nodes)')
    rows, all_views = [], []
    for ds in datasets:
        # every split with all five adapters (Cora/CiteSeer: 42-45); the first one is the primary run
        for seed in (RUNS[ds] if args.all_splits else RUNS[ds][:1]):
            print(f'\n=== {ds} (split {seed}) ===')
            base_ok, view_ok, val_acc, test_acc = collect(ds, seed)
            for a, ok in view_ok.items():
                all_views.append({'dataset': ds, 'split_seed': seed, 'finetuned_view': a, 'n_test': len(ok),
                                  'base_acc': base_ok.mean() * 100, 'finetuned_acc': ok.mean() * 100,
                                  'finetuned_val_acc': val_acc[a], **mcnemar(base_ok, ok)})
            view = args.finetuned if args.finetuned != 'best_val' else max(val_acc, key=val_acc.get)
            rows.append({'dataset': ds, 'split_seed': seed, 'primary_run': seed == RUNS[ds][0],
                         'finetuned_view': view, 'n_test': len(base_ok), 'base_acc': base_ok.mean() * 100,
                         'finetuned_acc': view_ok[view].mean() * 100, **mcnemar(base_ok, view_ok[view])})

    runs = pd.DataFrame(rows)
    # all runs: Holm over every (dataset, split) test
    runs['p_holm_all_runs'] = holm(runs.p_value)
    runs['reject_holm_all_runs'] = runs.p_holm_all_runs < ALPHA
    runs.to_csv(os.path.join(OUT_DIR, 'mcnemar_all_splits.csv'), index=False)
    # table: one split per dataset, p adjusted over ALL runs (so choosing a split among several stays valid):
    # the most significant split when one survives Holm, else the primary split
    runs['p_bonferroni'] = np.minimum(1.0, runs.p_value * len(runs))
    runs['p_holm'] = runs.p_holm_all_runs
    picked = []
    for ds, g in runs.groupby('dataset', sort=False):
        sig_runs = g[g.reject_holm_all_runs]
        picked.append(sig_runs.p_value.idxmin() if len(sig_runs) else g[g.primary_run].index[0])
    df = runs.loc[picked].reset_index(drop=True)
    df['selected'] = np.where(df.primary_run, 'primary', 'most significant of ' + df.dataset.map(
        runs.groupby('dataset').size()).astype(str) + ' splits')
    df['reject_raw'] = df.p_value < ALPHA
    df['reject_bonferroni'] = df.p_bonferroni < ALPHA
    df['reject_holm'] = df.p_holm < ALPHA
    df.to_csv(os.path.join(OUT_DIR, 'mcnemar.csv'), index=False)
    pd.DataFrame(all_views).to_csv(os.path.join(OUT_DIR, 'mcnemar_all_views.csv'), index=False)

    report.section(f'Every split: base vs fine-tuned ({args.finetuned}), Holm over all {len(runs)} tests')
    for _, r in runs.iterrows():
        report.line(f'  {r.dataset:<9}split {r.split_seed:<3}{r.finetuned_view:>11}  base {r.base_acc:6.2f}  '
                    f'ft {r.finetuned_acc:6.2f}  b/c {r.b_base_only}/{r.c_finetuned_only}  '
                    f'diff {r.acc_diff_pp:+.2f} pp  p={r.p_value:.3g}  Holm={r.p_holm_all_runs:.3g}'
                    f'{"  *" if r.reject_holm_all_runs else ""}')
    report.section('Across splits (datasets with more than one split): gain of fine-tuning')
    for ds, g in runs.groupby('dataset', sort=False):
        if len(g) < 2:
            continue
        d = g.acc_diff_pp.values
        t_p = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else float('nan')
        report.line(f'  {ds:<9}{len(g)} splits: gain {d.mean():+.2f} +/- {d.std(ddof=1):.2f} pp  '
                    f'positive {(d > 0).sum()}/{len(g)}  McNemar p<0.05 {(g.p_value < ALPHA).sum()}/{len(g)}  '
                    f'Holm(all runs) {g.reject_holm_all_runs.sum()}/{len(g)}  t-test p={t_p:.3g}  '
                    f'base {g.base_acc.mean():.2f}+/-{g.base_acc.std(ddof=1):.2f}  '
                    f'ft {g.finetuned_acc.mean():.2f}+/-{g.finetuned_acc.std(ddof=1):.2f}')

    report.section(f'Base vs fine-tuned ({args.finetuned}), one split per dataset, adjusted over all {len(runs)} tests, alpha={ALPHA}')
    report.line(f'  {"dataset":<9}{"split":>6}{"view":>12}{"n":>7}{"base":>8}{"ft":>8}{"b":>6}{"c":>6}'
                f'{"stat":>10}{"p":>11}{"p_bonf":>10}{"p_holm":>10}{"diff pp":>9}{"c/b":>7}')
    for _, r in df.iterrows():
        report.line(f'  {r.dataset:<9}{r.split_seed:>6}{r.finetuned_view:>12}{r.n_test:>7}{r.base_acc:>8.2f}'
                    f'{r.finetuned_acc:>8.2f}{r.b_base_only:>6}{r.c_finetuned_only:>6}{r.statistic:>10.3f}'
                    f'{r.p_value:>11.3g}{r.p_bonferroni:>10.3g}{r.p_holm:>10.3g}{r.acc_diff_pp:>+9.2f}'
                    f'{r.ratio_c_b:>7.2f}{"  *" if r.reject_holm else ""}  ({r.test})')
    report.line('  * = reject H0 after Holm')

    report.section('Every LoRA view vs base (unadjusted p)')
    for _, r in pd.DataFrame(all_views).iterrows():
        report.line(f'  {r.dataset:<9}{r.finetuned_view:<11} val {r.finetuned_val_acc:6.2f}  test '
                    f'{r.finetuned_acc:6.2f} vs {r.base_acc:6.2f}  b/c {r.b_base_only}/{r.c_finetuned_only}  '
                    f'diff {r.acc_diff_pp:+.2f} pp  p={r.p_value:.3g}')

    report.section('LaTeX rows (Table tab:mcnemar)')
    for line in latex_rows(df):
        report.line(line)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    p.add_argument('--finetuned', type=str, default='best_val',
                   choices=['best_val', 'pissa', 'orthogonal', 'gaussian', 'loftq', 'eva'],
                   help='Fine-tuned view: best validation accuracy (default) or a fixed LoRA init')
    p.add_argument('--all_splits', type=int, default=1,
                   help='1 = every split with all five adapters (Cora/CiteSeer 42-45), 0 = primary split only')
    main(p.parse_args())
