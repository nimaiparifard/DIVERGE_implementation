# Shared helpers for the "initialization diversity vs. seed variance" experiments
# (reviewer R1-C2 / R2-C1):
#   experiments/seed_vs_init_diversity.py      - #1 within-init (seed) vs between-init diversity
#   experiments/error_consistency_analysis.py  - #2 are init differences systematic?
#   experiments/complementarity_analysis.py    - #3 does diversity turn into ensemble gain?
#   experiments/lora_subspace_analysis.py      - #4 mechanism: adapter subspaces + stage chain
#   experiments/knn_neighborhood_analysis.py   - #5 local (k-NN) neighbourhood agreement
"""
All five experiments work on the same *pool* of models for one (dataset, llm):
one member per (LoRA-init approach, LoRA seed), e.g. 5 inits x 4 seeds = 20.

- LLM embeddings are loaded strictly from the seed-tagged cache
  (<llm>_<dataset>_seqcls_<peft>_init-<init>_pool-mean_seed<seed>_semi_supervised.pt, for both splits).
  Unlike dataset.data_utils.get_init_dataset_for_gnn there is NO fallback to
  seed-less caches: a fallback would silently put the same embedding under
  several seeds and fake zero seed variance. Missing members are skipped and
  reported.
- One GNN is trained per member with a FIXED GNN seed / split (--gnn_seed), so
  the only thing that varies between two members is the LoRA init and/or the
  LoRA seed. Its test/val predictions, class probabilities and penultimate
  (before-logits) node embeddings are cached under
  results/diversity/gnn_pool/ and reused by every experiment.
- --gnn_seed -1 ("own split"): every member's GNN is trained and tested on the
  split of its OWN LoRA seed. With --split 1 the LoRA of seed s was fine-tuned on
  the seed-s split, so a fixed GNN split would test seed-t members (t != s) on
  nodes their LoRA was fine-tuned on. In own-split mode every node-level
  comparison uses only nodes that are test nodes of ALL compared members
  (pairs: test_i & test_j; groups: AND over the group), so no member has seen
  the labels of the nodes it is evaluated on. Group analyses whose common test
  set is smaller than --min_eval_nodes are skipped and reported as such.
- Leakage-free fixed split (--split_seed T --gnn_seed T): read members whose LoRA was
  fine-tuned on split T for every LoRA seed (trained with --split_seed T, optionally
  --init_seed), so every member is evaluated on the same, never-seen test nodes. The
  original members (LoRA seed s fine-tuned on split s) tested on one fixed split leak
  test labels; check_no_leakage refuses that combination unless --allow_leakage.
- Diversity matrices are always expressed as *distances* (larger = more
  diverse): 1 - CKA, prediction disagreement, 1 - subspace overlap, 1 - Jaccard.
"""
import itertools
import os
from datetime import datetime

import numpy as np
import torch

from config import setup_finetuning_cfg
from dataset.data_utils import get_cached_dir
from gnns.gnn_mtrainer import set_mgnn_cfg, GNNTrainer, append_gnn_result_csv
from experiments.cka import APPROACHES, TITLES
from experiments.cka_for_gnn_embeddings import extract_gnn_penultimate_embedding
from adapter_dir import lora_run_tag

DEFAULT_SEEDS = [42, 43, 44, 45]
DEFAULT_OUT_DIR = 'results/diversity'


# ----------------------------------------------------------------------------- CLI
def add_common_args(parser):
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, DEFAULT_SEEDS)),
                        help='Comma-separated LoRA seeds forming the pool (must match the trained/cached adapters)')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                        help=f"Comma-separated subset of {APPROACHES}")
    parser.add_argument('--split', type=int, default=0,
                        help='GNN split: 0 = semi-supervised (default), 1 = supervised; both read the seed-tagged _semi_supervised caches')
    parser.add_argument('--gnn_seed', type=int, default=42,
                        help='Fixed GNN seed/split shared by every pool member (isolates the LoRA seed). '
                             '-1 = own split: each member uses the split of its own LoRA seed (no leakage when '
                             'the LoRA was fine-tuned with --split 1)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Read members whose LoRA was fine-tuned on this split for every LoRA seed '
                             '(trained with --split_seed; ..._seed<s>_split<t>_... caches). Use it with '
                             '--gnn_seed equal to it: then no member has seen a test label. Default: the '
                             'original members, fine-tuned on the split of their own LoRA seed')
    parser.add_argument('--init_seed', type=int, default=None,
                        help='With --split_seed: read members trained with this --init_seed (same LoRA init '
                             'draw for every seed; ..._seed<s>_split<t>_init<i>_...)')
    parser.add_argument('--allow_leakage', action='store_true',
                        help='Supervised split only: run even if a member was fine-tuned on a different split '
                             'than the GNN split (its LoRA has seen labels of GNN test nodes). Only for '
                             'reproducing the old reports')
    parser.add_argument('--min_eval_nodes', type=int, default=30,
                        help='Own-split mode: skip group analyses whose common test set is smaller than this')
    parser.add_argument('--gnn_model_name', type=str, default=None,
                        help='GNN encoder (default: the one in the tuned hyperparameter JSON)')
    parser.add_argument('--retrain_gnn', action='store_true', help='Ignore the cached GNN pool and retrain')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--n_perm', type=int, default=2000, help='Permutations for the significance tests')
    return parser


def parse_list(text, cast=str):
    return [cast(t.strip()) for t in str(text).split(',') if t.strip()]


# ----------------------------------------------------------------------------- pool
def embedding_cache_path(cache_dir, llm_name, dataset_name, peft_type, init, seed, supervised,
                         split_seed=None, init_seed=None):
    # The seed-tagged semi-supervised cache serves both splits (same as
    # dataset.data_utils.get_init_dataset_for_gnn); `supervised` only affects the GNN split.
    return os.path.join(cache_dir, f'{llm_name}_{dataset_name}_seqcls_{peft_type}_init-{init}_pool-mean'
                                   f'{lora_run_tag(seed, split_seed, init_seed)}_semi_supervised.pt')


def adapter_path(llm_name, dataset_name, peft_type, init, seed, split_seed=None, init_seed=None):
    return os.path.join('artifacts', f'{llm_name}_{dataset_name}_seqcls_{peft_type}_semi_supervised_init_type_{init}'
                                     f'{lora_run_tag(seed, split_seed, init_seed)}')


class Pool:
    """Members of one (dataset, llm) pool, ordered by init (APPROACHES order) then seed."""

    def __init__(self, dataset_name, llm_name, peft_type, approaches, seeds, supervised, gnn_seed,
                 gnn_model_name=None, out_dir=DEFAULT_OUT_DIR, split_seed=None, init_seed=None,
                 allow_leakage=False):
        self.dataset_name, self.llm_name, self.peft_type = dataset_name, llm_name, peft_type
        self.supervised, self.gnn_seed, self.out_dir = supervised, gnn_seed, out_dir
        self.own_split = gnn_seed is not None and gnn_seed < 0
        if init_seed is not None and split_seed is None:
            raise ValueError('--init_seed needs --split_seed (the members were trained with both)')
        self.split_seed, self.init_seed = split_seed, init_seed
        check_no_leakage(seeds, split_seed, gnn_seed, supervised, allow_leakage)
        self.gnn_model_name_arg = gnn_model_name
        self.cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
        self.model_name = self.cfg.llm.model_name
        self.gnn_cfg = set_mgnn_cfg(dataset_name, supervised=supervised,
                                    seed=seeds[0] if self.own_split else gnn_seed, llm_name=llm_name,
                                    gnn_model_name=gnn_model_name)
        self.gnn_name = self.gnn_cfg.gnn_model_name

        cache_dir = get_cached_dir()
        approaches = [a for a in APPROACHES if a in approaches]
        self.members, self.missing = [], []
        for init in approaches:
            for seed in seeds:
                path = embedding_cache_path(cache_dir, self.model_name, dataset_name, peft_type, init, seed,
                                            supervised, split_seed, init_seed)
                m = {'init': init, 'seed': seed, 'label': f'{TITLES[init]}-{seed}', 'path': path,
                     'adapter': adapter_path(self.model_name, dataset_name, peft_type, init, seed,
                                             split_seed, init_seed)}
                (self.members if os.path.exists(path) else self.missing).append(m)
        if len(self.members) < 2:
            raise FileNotFoundError(
                f"Fewer than 2 seed-tagged embedding caches found for {dataset_name}/{llm_name}. Missing:\n" +
                "\n".join(f"  - {m['path']}" for m in self.missing))
        self.inits = np.array([m['init'] for m in self.members])
        self.seeds = np.array([m['seed'] for m in self.members])
        self.labels = [m['label'] for m in self.members]

    @property
    def tag(self):
        split = 'sup' if self.supervised else 'semi'
        gs = 'own' if self.own_split else self.gnn_seed
        # Fixed-split pools get their own outputs (GNN pool, matrices, reports) next to the old ones.
        ft = '' if self.split_seed is None else f'_ftsplit{self.split_seed}'
        ft += '' if self.init_seed is None else f'_init{self.init_seed}'
        return f'{self.dataset_name}_{self.model_name}_{self.peft_type}_{split}{ft}_{self.gnn_name}_gnnseed{gs}'

    def __len__(self):
        return len(self.members)

    def describe(self):
        lines = [f"Pool: {len(self.members)} members ({len(set(self.inits))} inits x "
                 f"{len(set(self.seeds))} seeds), GNN={self.gnn_name} " +
                 ("own split per member (GNN seed = LoRA seed; node metrics on shared test nodes)"
                  if self.own_split else f"fixed gnn_seed={self.gnn_seed}")]
        for init in dict.fromkeys(self.inits):
            lines.append(f"  {TITLES[init]}: seeds {sorted(int(s) for s in self.seeds[self.inits == init])}")
        if self.missing:
            lines.append(f"  Skipped (no seed-tagged cache): {', '.join(m['label'] for m in self.missing)}")
        return lines

    def balanced_subset(self):
        """Indices of the largest init x seed grid where every init has every seed (for ANOVA/matched sets)."""
        inits = list(dict.fromkeys(self.inits))
        common = set(self.seeds[self.inits == inits[0]])
        for init in inits[1:]:
            common &= set(self.seeds[self.inits == init])
        idx = [i for i, m in enumerate(self.members) if m['seed'] in common]
        return idx, inits, sorted(int(s) for s in common)

    # ---- LLM embeddings
    def load_llm_embedding(self, i):
        data = torch.load(self.members[i]['path'], map_location='cpu', weights_only=False)
        return data['embeddings'].float()

    def load_llm_embeddings(self):
        return [self.load_llm_embedding(i) for i in range(len(self))]

    # ---- GNN pool (trained once, cached)
    def gnn_pool_dir(self):
        return os.path.join(self.out_dir, 'gnn_pool', self.tag)

    def member_gnn_cfg(self, i):
        if not self.own_split:
            return self.gnn_cfg
        return set_mgnn_cfg(self.dataset_name, supervised=self.supervised, seed=int(self.members[i]['seed']),
                            llm_name=self.llm_name, gnn_model_name=self.gnn_model_name_arg or self.gnn_name)

    def load_gnn_pool(self, retrain=False, verbose=True):
        """
        Returns dict with
          probs [M, n, C] float32, preds [M, n] long, emb list of [n, h] float32,
          y [n], train/val/test masks [n] bool, test_acc/val_acc [M] (percent),
          test_masks [M, n] bool: each member's own test nodes (all equal to test_mask
          for a fixed GNN split). In own-split mode test_mask is the AND over members
          (nodes that are test nodes for every member).
        """
        pool_dir = self.gnn_pool_dir()
        os.makedirs(pool_dir, exist_ok=True)
        meta_path = os.path.join(pool_dir, 'meta.pt')
        meta = torch.load(meta_path, weights_only=False) if os.path.exists(meta_path) and not retrain else None

        records = []
        for i, m in enumerate(self.members):
            path = os.path.join(pool_dir, f"init-{m['init']}_seed{m['seed']}.pt")
            if os.path.exists(path) and meta is not None and not retrain:
                records.append(torch.load(path, weights_only=False))
                continue
            if verbose:
                print(f"\n--- [GNN pool] training {self.gnn_name} on {m['label']} ({i + 1}/{len(self)}) ---")
            emb = self.load_llm_embedding(i)
            trainer = GNNTrainer(self.member_gnn_cfg(i), emb, verbose=verbose)
            results, _, _ = trainer.train()
            with torch.no_grad():
                trainer.model.eval()
                logits = trainer.model(trainer.features, trainer.graph)
            penult = extract_gnn_penultimate_embedding(trainer.model, trainer.features, trainer.graph)
            rec = {
                'probs': torch.softmax(logits.float(), dim=1).cpu(),
                'emb': penult.cpu().half(),
                'test_acc': float(results['test_acc']),
                'val_acc': float(results['val_acc']),
                'test_mask': trainer.data.test_mask.detach().cpu().bool(),
            }
            torch.save(rec, path)
            records.append(rec)
            if meta is None:
                d = trainer.data
                meta = {'y': d.y.detach().cpu().long(), 'train_mask': d.train_mask.detach().cpu().bool(),
                        'val_mask': d.val_mask.detach().cpu().bool(), 'test_mask': d.test_mask.detach().cpu().bool(),
                        'hp_source': self.gnn_cfg.hp_source}
                torch.save(meta, meta_path)
            del trainer, emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        probs = torch.stack([r['probs'] for r in records])
        out = {
            'probs': probs,
            'preds': probs.argmax(dim=2),
            'emb': [r['emb'].float() for r in records],
            'test_acc': np.array([r['test_acc'] for r in records]),
            'val_acc': np.array([r['val_acc'] for r in records]),
            **meta,
        }
        if self.own_split:
            if any('test_mask' not in r for r in records):
                raise RuntimeError(f'GNN pool in {pool_dir} has no per-member test masks; rerun with --retrain_gnn')
            out['test_masks'] = torch.stack([r['test_mask'] for r in records])
            out['test_mask'] = out['test_masks'].all(dim=0)
        else:
            out['test_masks'] = out['test_mask'][None].expand(len(records), -1).clone()
        return out


def check_no_leakage(seeds, split_seed, gnn_seed, supervised, allow_leakage=False):
    """
    Supervised split: refuse a pool in which a member's LoRA was fine-tuned on a different 60/20/20
    split than the one its GNN is tested on. Such a LoRA has seen the text and labels of ~60% of the
    GNN test nodes (seed-43..45 members tested on the seed-42 split were 2-3 pp too accurate).
      fine-tuning split of a member = split_seed if given, else its own LoRA seed
      GNN split                     = gnn_seed, or the member's own LoRA seed if gnn_seed < 0
    """
    if not supervised or allow_leakage:
        return
    own_split = gnn_seed is not None and gnn_seed < 0
    if own_split and split_seed is not None:
        raise ValueError('--gnn_seed -1 (own split) contradicts --split_seed: with --split_seed every member '
                         f'was fine-tuned on split {split_seed}, so use --gnn_seed {split_seed}.')
    if own_split:
        return
    leaking = [s for s in seeds if (split_seed if split_seed is not None else s) != gnn_seed]
    if leaking:
        raise ValueError(
            f'Test-label leakage: LoRA seeds {leaking} were fine-tuned on split '
            f"{split_seed if split_seed is not None else 'of their own seed'}, but the GNN is tested on split "
            f'{gnn_seed}. Train the members with --split_seed {gnn_seed} and pass --split_seed {gnn_seed} '
            f'(clean), or use --gnn_seed -1 (own split, fewer shared test nodes), or --allow_leakage to '
            f'reproduce the old reports.')


def pool_from_args(args):
    return Pool(args.dataset_name, args.llm_name, args.peft_type,
                approaches=parse_list(args.approaches), seeds=parse_list(args.seeds, int),
                supervised=args.split == 1, gnn_seed=args.gnn_seed, gnn_model_name=args.gnn_model_name,
                out_dir=args.out_dir, split_seed=getattr(args, 'split_seed', None),
                init_seed=getattr(args, 'init_seed', None), allow_leakage=getattr(args, 'allow_leakage', False))


def common_mask(test_masks, members):
    """Nodes that are test nodes for every member in `members` ([n] bool)."""
    return test_masks[list(members)].all(dim=0)


# ----------------------------------------------------------------------------- similarity matrices
def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cka_distance_matrix(mats, device=None):
    """
    [M, M] matrix of 1 - linear CKA between every pair of representation
    matrices (same rows). Computed in feature space on the GPU in float32;
    each member's ||X^T X||_F is computed once. Members stay on the GPU when
    they fit, otherwise they are moved per pair.
    """
    device = device or _device()
    centered = [(x - x.mean(dim=0, keepdim=True)).float() for x in mats]
    total_bytes = sum(x.numel() * 4 for x in centered)
    on_device = device.type == 'cuda' and total_bytes < 0.5 * torch.cuda.mem_get_info()[0]
    if on_device:
        centered = [x.to(device) for x in centered]

    def dev(x):
        return x if on_device else x.to(device)

    M = len(centered)
    self_norm = []
    for x in centered:
        xd = dev(x)
        self_norm.append(torch.linalg.matrix_norm(xd.T @ xd).item())
    dist = np.zeros((M, M))
    for i in range(M):
        xi = dev(centered[i])
        for j in range(i + 1, M):
            xj = dev(centered[j])
            hsic = (torch.linalg.matrix_norm(xi.T @ xj) ** 2).item()
            denom = self_norm[i] * self_norm[j]
            cka = hsic / denom if denom > 0 else float('nan')
            dist[i, j] = dist[j, i] = 1.0 - cka
    return dist


def disagreement_matrix(preds, mask):
    """
    [M, M] fraction of masked nodes where two members predict different labels.
    `mask` is [n] (one node set for all pairs) or [M, n] (per-member test nodes;
    each pair is compared on the nodes in both members' test sets).
    """
    M = preds.shape[0]
    d = np.zeros((M, M))
    if mask.dim() == 1:
        p = preds[:, mask]
        for i in range(M):
            d[i] = (p[i:i + 1] != p).float().mean(dim=1).numpy()
        return d
    for i in range(M):
        for j in range(i + 1, M):
            m = mask[i] & mask[j]
            d[i, j] = d[j, i] = (preds[i, m] != preds[j, m]).float().mean().item() if m.any() else float('nan')
    return d


def pair_node_counts(test_masks):
    """[M, M] number of shared test nodes per pair (for reporting)."""
    t = test_masks.float()
    return (t @ t.T).long().numpy()


def cached_matrix(pool, name, fn, recompute=False):
    """Compute a pool-level matrix once and cache it (keyed by the exact member list)."""
    path = os.path.join(pool.out_dir, 'matrices', f'{pool.tag}_{name}.pt')
    if os.path.exists(path) and not recompute:
        saved = torch.load(path, weights_only=False)
        if saved.get('labels') == pool.labels:
            return saved['matrix']
    mat = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'labels': pool.labels, 'matrix': mat}, path)
    return mat


# ----------------------------------------------------------------------------- statistics
def pair_groups(inits, seeds):
    """Upper-triangle pair indices split by what differs between the two members:
       seed_only: same init, different seed   -> pure seed / training noise
       init_only: different init, same seed   -> pure initialization effect
                  (same seed = same head init, data order and dropout masks)
       both:      different init, different seed"""
    groups = {'seed_only': [], 'init_only': [], 'both': []}
    for i, j in itertools.combinations(range(len(inits)), 2):
        same_init, same_seed = inits[i] == inits[j], seeds[i] == seeds[j]
        if same_init and not same_seed:
            groups['seed_only'].append((i, j))
        elif not same_init and same_seed:
            groups['init_only'].append((i, j))
        elif not same_init:
            groups['both'].append((i, j))
    return groups


def _vals(D, pairs):
    return np.array([D[i, j] for i, j in pairs]) if pairs else np.array([])


def _mean(v):
    return float(v.mean()) if v.size else float('nan')


def within_between_test(D, inits, seeds, n_perm=2000, rng=None):
    """
    Two questions on a pool distance matrix D:

    1) Does the init have a consistent effect at all?  Among cross-seed pairs,
       are different-init pairs farther apart than same-init pairs?
       stat = mean(both) - mean(seed_only); permutation p-value from shuffling
       the init labels within each seed stratum. (Both groups differ in seed,
       so the comparison is not biased by seed variance.)

    2) How large is the init effect compared with seed noise?
       ratio = mean(init_only) / mean(seed_only): > 1 means changing only the
       init moves the representation more than changing only the seed.
       One-sided Mann-Whitney U (init_only > seed_only) on the pair values;
       approximate, because pairs sharing a member are not independent.
    """
    from scipy.stats import mannwhitneyu
    rng = rng or np.random.default_rng(0)
    g = pair_groups(inits, seeds)
    seed_only, init_only, both = (_vals(D, g[k]) for k in ('seed_only', 'init_only', 'both'))
    out = {
        'seed_only_mean': _mean(seed_only), 'seed_only_std': float(seed_only.std()) if seed_only.size else float('nan'),
        'n_seed_only': seed_only.size,
        'init_only_mean': _mean(init_only), 'init_only_std': float(init_only.std()) if init_only.size else float('nan'),
        'n_init_only': init_only.size,
        'both_mean': _mean(both), 'n_both': both.size,
    }
    out['init_over_seed_ratio'] = (out['init_only_mean'] / out['seed_only_mean']
                                   if out['seed_only_mean'] else float('nan'))
    out['init_vs_seed_mwu_p'] = (float(mannwhitneyu(init_only, seed_only, alternative='greater').pvalue)
                                 if init_only.size and seed_only.size else float('nan'))

    out['init_effect_diff'] = out['both_mean'] - out['seed_only_mean']
    if not seed_only.size or not both.size:
        out['init_effect_p'] = float('nan')
        return out
    strata = [np.where(seeds == s)[0] for s in np.unique(seeds)]
    count = 0
    for _ in range(n_perm):
        perm = inits.copy()
        for idx in strata:
            perm[idx] = rng.permutation(perm[idx])
        gp = pair_groups(perm, seeds)
        w, b = _vals(D, gp['seed_only']), _vals(D, gp['both'])
        if w.size and b.size and (b.mean() - w.mean()) >= out['init_effect_diff']:
            count += 1
    out['init_effect_p'] = (count + 1) / (n_perm + 1)
    return out


def permanova(D, groups, n_perm=2000, rng=None):
    """One-way PERMANOVA (Anderson, 2001) on a distance matrix.
    Returns pseudo-F, R^2 (share of total distance variance explained by the
    factor) and the permutation p-value."""
    rng = rng or np.random.default_rng(0)
    D2 = np.asarray(D, dtype=float) ** 2
    N = D2.shape[0]
    iu = np.triu_indices(N, 1)
    ss_total = D2[iu].sum() / N

    def ss_within(g):
        s = 0.0
        for lvl in np.unique(g):
            idx = np.where(g == lvl)[0]
            if len(idx) > 1:
                sub = D2[np.ix_(idx, idx)]
                s += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
        return s

    groups = np.asarray(groups)
    a = len(np.unique(groups))
    if a < 2 or a >= N:
        return {'F': float('nan'), 'R2': float('nan'), 'p_value': float('nan')}

    def stat(g):
        ssw = ss_within(g)
        ssa = ss_total - ssw
        return (ssa / (a - 1)) / (ssw / (N - a)) if ssw > 0 else float('inf'), ssa / ss_total

    F, R2 = stat(groups)
    count = sum(stat(rng.permutation(groups))[0] >= F for _ in range(n_perm))
    return {'F': F, 'R2': R2, 'p_value': (count + 1) / (n_perm + 1)}


def mantel_test(D1, D2, n_perm=2000, rng=None):
    """Spearman Mantel correlation between the upper triangles of two distance matrices."""
    from scipy.stats import spearmanr
    rng = rng or np.random.default_rng(0)
    N = D1.shape[0]
    iu = np.triu_indices(N, 1)
    r = spearmanr(D1[iu], D2[iu])[0]
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(N)
        if spearmanr(D1[iu], D2[np.ix_(p, p)][iu])[0] >= r:
            count += 1
    return {'r': r, 'p_value': (count + 1) / (n_perm + 1)}


# ----------------------------------------------------------------------------- output helpers
def save_pool_heatmap(D, labels, inits, save_path, title, cbar_label, vmin=None, vmax=None):
    """Heatmap of a pool matrix with lines separating init blocks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    M = len(labels)
    fig, ax = plt.subplots(figsize=(0.45 * M + 3, 0.42 * M + 2.5))
    im = ax.imshow(D, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(M))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(M))
    ax.set_yticklabels(labels, fontsize=7)
    for k in range(1, M):
        if inits[k] != inits[k - 1]:
            ax.axhline(k - 0.5, color='white', lw=1.2)
            ax.axvline(k - 0.5, color='white', lw=1.2)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] heatmap saved to: {save_path}")
    return save_path


def save_matrix_csv(D, labels, save_path):
    import csv
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([''] + labels)
        for lbl, row in zip(labels, D):
            w.writerow([lbl] + [round(float(v), 6) for v in row])
    return save_path


def run_info(pool):
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_name': pool.dataset_name,
        'llm_name': pool.llm_name,
        'peft_type': pool.peft_type,
        'supervised': pool.supervised,
        'gnn_model_name': pool.gnn_name,
        'gnn_seed': 'own' if pool.own_split else pool.gnn_seed,
        'n_members': len(pool),
        'inits': ','.join(TITLES[i] for i in dict.fromkeys(pool.inits)),
        'seeds': ','.join(str(s) for s in sorted(set(int(s) for s in pool.seeds))),
    }


def append_rows(rows, csv_path):
    for row in rows:
        append_gnn_result_csv(row, csv_path)
    print(f"[OK] {len(rows)} row(s) appended to: {csv_path}")


class TextReport:
    """Minimal report writer: prints and mirrors every line to a .txt file."""

    def __init__(self, path, title):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.path = path
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n{'=' * len(title)}\n{datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        print(title)

    def line(self, text=''):
        print(text)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(str(text) + '\n')

    def section(self, title):
        self.line('')
        self.line(title)
        self.line('-' * len(title))

    def within_between(self, name, r):
        self.line(f"{name}:")
        self.line(f"  seed only (same init, diff seed): {r['seed_only_mean']:.4f} +/- {r['seed_only_std']:.4f} "
                  f"[{r['n_seed_only']} pairs]")
        self.line(f"  init only (diff init, same seed): {r['init_only_mean']:.4f} +/- {r['init_only_std']:.4f} "
                  f"[{r['n_init_only']} pairs]")
        self.line(f"  init + seed (diff both)         : {r['both_mean']:.4f} [{r['n_both']} pairs]")
        self.line(f"  init effect exists (diff-both vs seed-only, permutation): "
                  f"diff={r['init_effect_diff']:+.4f} p={r['init_effect_p']:.4g}")
        self.line(f"  init-only / seed-only = {r['init_over_seed_ratio']:.2f}x "
                  f"(Mann-Whitney one-sided p={r['init_vs_seed_mwu_p']:.4g})")

    @staticmethod
    def within_between_row(r, prefix=''):
        keys = ['seed_only_mean', 'seed_only_std', 'n_seed_only', 'init_only_mean', 'init_only_std', 'n_init_only',
                'both_mean', 'n_both', 'init_over_seed_ratio', 'init_vs_seed_mwu_p', 'init_effect_diff',
                'init_effect_p']
        return {f'{prefix}{k}': r[k] for k in keys}
