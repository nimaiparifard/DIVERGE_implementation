"""
Shared pieces of the base-model TAPE / STAGE baselines (TAPE/run_tape_base.py,
STAGE/run_stage_base.py), run on DIVERGE's own setup so the numbers are comparable:

- Embeddings come from the *base* LLM (no LoRA adapter), mean-pooled last hidden layer,
  for two text sources:
      'text'      - the node's raw text
      'augmented' - the node's GPT-4o response in artifacts/gpt_4o/<dataset>.json
  They are seed-independent (nothing is fine-tuned) and cached once in artifacts/cache
  with the names cache.cache_embedding.CacheEmbedding already uses for base-model caches
  (so STAGE reuses the 'text' cache TAPE builds, and existing base caches are picked up).
- GNNs are trained with gnns.gnn_mtrainer.GNNTrainer and the tuned hyperparameters of
  gnn_hyperparameters/ (set_mgnn_cfg), on the same seed-dependent split as DIVERGE.
  RevGAT has no tuned file and no GNNEncoder equivalent; it is a grouped reversible GAT
  (PyG GroupAddRev, as in Li et al. 2021) with the TAPE paper's settings (REVGAT_HPARAMS).
- Views are combined by averaging their logits (the CSV also reports the average of
  their softmax probabilities).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cache.cache_embedding import CacheEmbedding
from common import compute_acc_and_f1, load_graph_dataset_for_tape, set_seed
from config import setup_finetuning_cfg
from dataset.data_utils import get_cached_dir
from gnns.gnn_mtrainer import GNNTrainer, set_mgnn_cfg, resolve_path_prefix, append_gnn_result_csv

DEFAULT_SEEDS = [42, 43, 44, 45]
EMBEDDING_KINDS = ('text', 'augmented')
STAGE_GNN_MODELS = ('SAGE', 'RevGAT', 'GCN', 'GIN')
AUGMENTED_TEXT_DIR = os.path.join('artifacts', 'gpt_4o')

# TAPE paper (Table 10) RevGAT settings; epochs/patience raised for the lower learning rate
REVGAT_HPARAMS = {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.75, 'lr': 0.002, 'weight_decay': 0.0,
                  'epochs': 500, 'early_stop': 50, 'num_groups': 2, 'heads': 1}


# ----------------------------------------------------------------------------- embeddings
def load_augmented_texts(dataset_name, raw_texts, responses_dir=AUGMENTED_TEXT_DIR):
    """
    GPT-4o response per node, aligned by its 'idx' field (computer/arxiv have a few nodes
    without a response, so line order is not node order). Nodes without a response keep
    their raw text. Returns (texts, n_missing).
    """
    path = os.path.join(responses_dir, f'{dataset_name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No augmented texts for '{dataset_name}': {path}")
    id2answer = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                id2answer[int(item['idx'])] = item['answer']
    texts = [id2answer.get(i, raw_texts[i]) for i in range(len(raw_texts))]
    n_missing = sum(1 for i in range(len(raw_texts)) if i not in id2answer)
    return texts, n_missing


class BaseModelEmbeddingCache(CacheEmbedding):
    """CacheEmbedding with the base model; kind='augmented' embeds the idx-aligned GPT-4o texts."""

    def __init__(self, cfg, kind, pooling='mean'):
        self.kind = kind
        self.n_missing_augmented = 0
        super().__init__(cfg, adapter_dir=None, pooling=pooling,
                         used_llm_responses=kind == 'augmented', used_based_model=True)

    def _load_llm_responses_fallback(self):
        texts, self.n_missing_augmented = load_augmented_texts(self.dataset_name, self.dataset.raw_texts)
        if self.n_missing_augmented:
            print(f"[augmented] {self.n_missing_augmented} node(s) without a GPT-4o response keep their raw text")
        return texts


def base_embedding_cache_path(model_name, dataset_name, kind, pooling='mean'):
    """Same file names as CacheEmbedding.save_cache with used_based_model=True."""
    if kind == 'text':
        fname = f"{model_name}_{dataset_name}_seqcls_pool-{pooling}_based_model.pt"
    elif kind == 'augmented':
        fname = f"{model_name}_{dataset_name}_seqcls_{pooling}_for_llm_responses_based_model.pt"
    else:
        raise ValueError(f"kind must be one of {EMBEDDING_KINDS}, got '{kind}'")
    return os.path.join(get_cached_dir(), fname)


def get_base_embeddings(dataset_name, llm_name, kind, batch_size=None, rebuild=False, peft_type='lora'):
    """Load (or build and cache) the base-model embeddings of one text source. Returns a CPU float tensor."""
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    path = base_embedding_cache_path(cfg.llm.model_name, dataset_name, kind)
    if os.path.exists(path) and not rebuild:
        print(f"[embeddings] {kind}: loading {path}")
        emb = torch.load(path, map_location='cpu', weights_only=False)['embeddings'].float()
        if emb.shape[0] != cfg.dataset.num_samples:
            raise ValueError(f"{path} has {emb.shape[0]} rows, dataset has {cfg.dataset.num_samples} nodes; "
                             f"rebuild it with --rebuild_embeddings")
        return emb

    print(f"[embeddings] {kind}: building with base {llm_name} -> {path}")
    if batch_size is not None:
        cfg.llm.caching_batch_size = batch_size
    cacher = BaseModelEmbeddingCache(cfg, kind)
    cache = cacher.build_cache()
    cache['meta'].update({'text_source': kind, 'adapter': None,
                          'n_missing_augmented': cacher.n_missing_augmented})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)
    print(f"[OK] Saved base-model {kind} embeddings to: {path}")
    emb = cache['embeddings'].float()
    del cacher, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


# ----------------------------------------------------------------------------- RevGAT
class _GATBlock(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        from torch_geometric.nn import GATConv
        self.conv = GATConv(channels, channels // heads, heads=heads)
        self.dropout = dropout

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = F.relu(self.norm(x))
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)


class RevGAT(nn.Module):
    """
    Grouped reversible GAT (Li et al. 2021, "Training Graph Neural Networks with 1000 Layers"):
    input projection -> num_layers GroupAddRev(GAT) blocks -> norm/relu/dropout -> linear.
    One dropout mask is shared by all blocks, as reversibility requires.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, dropout=0.5, num_groups=2, heads=1):
        super().__init__()
        from torch_geometric.nn import GroupAddRev
        if hidden_dim % num_groups or (hidden_dim // num_groups) % heads:
            raise ValueError("hidden_dim must be divisible by num_groups * heads")
        self.dropout = dropout
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GroupAddRev(_GATBlock(hidden_dim // num_groups, heads, dropout), num_groups=num_groups)
            for _ in range(n_layers)
        ])
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout).requires_grad_(False) / (1 - self.dropout)
        for layer in self.layers:
            x = layer(x, edge_index, mask)
        x = F.relu(self.norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)


# ----------------------------------------------------------------------------- GNN training
def load_split(dataset_name, seed, supervised=True):
    """Graph + seed-dependent split, loaded exactly as GNNTrainer would (shared by all GNNs of a seed)."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    data, _, _ = load_graph_dataset_for_tape(dataset_name, device, re_split=supervised,
                                             path_prefix=resolve_path_prefix(), seed=seed)
    return data


def gnn_cfg(dataset_name, gnn_name, seed, llm_name, supervised=True):
    """GNN config: tuned JSON for SAGE/GCN/GIN/GAT, REVGAT_HPARAMS for RevGAT."""
    # RevGAT only takes seed/device/split/batch_norm from the (SAGE) file, no GAT-tuned file exists
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name,
                       gnn_model_name='SAGE' if gnn_name == 'RevGAT' else gnn_name)
    if gnn_name == 'RevGAT':
        cfg.gnn_model_name = 'RevGAT'
        for k in ('hidden_dim', 'num_layers', 'dropout', 'lr', 'weight_decay', 'epochs', 'early_stop'):
            setattr(cfg, k, REVGAT_HPARAMS[k])
        cfg.hp_source = 'REVGAT_HPARAMS (TAPE paper Table 10)'
    if torch.cuda.is_available():
        cfg.device = max(cfg.device, 1)  # GNNTrainer only uses the GPU for device > 0
    return cfg


def train_gnn(dataset_name, embedding, gnn_name, seed, llm_name, data, supervised=True, label=''):
    """Train one GNN on `embedding`. Returns (results, best-epoch logits on CPU, cfg)."""
    cfg = gnn_cfg(dataset_name, gnn_name, seed, llm_name, supervised)
    if gnn_name == 'RevGAT':
        cfg.gnn_model_name = 'GAT'  # GNNTrainer builds a GNNEncoder; swapped for RevGAT below
        trainer = GNNTrainer(cfg, embedding, data=data, verbose=False)
        cfg.gnn_model_name = 'RevGAT'
        set_seed(seed)
        trainer.model = RevGAT(embedding.shape[1], cfg.hidden_dim, trainer.num_classes, n_layers=cfg.num_layers,
                               dropout=cfg.dropout, num_groups=REVGAT_HPARAMS['num_groups'],
                               heads=REVGAT_HPARAMS['heads']).to(trainer.device)
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        trainer.gnn_model_name = 'RevGAT'
    else:
        trainer = GNNTrainer(cfg, embedding, data=data, verbose=False)
    results, best_logits, _ = trainer.train()
    print(f"  {(gnn_name + ' ' + label).strip():<12} test_acc={results['test_acc']:.2f} test_f1={results['test_f1']:.2f} "
          f"val_acc={results['val_acc']:.2f} ({results['epochs_run']} epochs, {results['train_time_sec']:.1f}s) "
          f"hp={os.path.basename(str(cfg.hp_source))}")
    logits = best_logits.detach().float().cpu()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, logits, cfg


def evaluate_mean(logits_list, data, use_probs=False):
    """Average the views' logits (or softmax probabilities) and score val/test like GNNTrainer."""
    stacked = torch.stack([F.softmax(l, dim=-1) if use_probs else l for l in logits_list])
    pred = stacked.mean(dim=0).argmax(dim=-1).numpy()
    y = data.y.squeeze().cpu().numpy()
    out = {}
    for split, mask in (('val', data.val_mask), ('test', data.test_mask)):
        m = mask.cpu().numpy().astype(bool)
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(pred[m], y[m])
        out[split] = {'acc': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
    return out


def view_columns(prefix, results):
    return {f'{prefix}_test_acc': results['test_acc'], f'{prefix}_test_f1': results['test_f1'],
            f'{prefix}_test_weighted_f1': results['test_weight_f1'], f'{prefix}_val_acc': results['val_acc'],
            f'{prefix}_val_f1': results['val_f1'], f'{prefix}_epochs_run': results['epochs_run'],
            f'{prefix}_train_time_sec': results['train_time_sec']}


def mean_columns(prefix, scores):
    return {f'{prefix}_test_acc': scores['test']['acc'], f'{prefix}_test_f1': scores['test']['macro_f1'],
            f'{prefix}_test_weighted_f1': scores['test']['weighted_f1'], f'{prefix}_val_acc': scores['val']['acc'],
            f'{prefix}_val_f1': scores['val']['macro_f1']}


# ----------------------------------------------------------------------------- CSV
def append_row(row, csv_path):
    append_gnn_result_csv(row, csv_path)
    print(f"[OK] result row appended to: {csv_path}")


def write_summary(csv_path, summary_path, group_cols):
    """
    mean/std over seeds of every *_acc / *_f1 column, per group_cols. When a seed was
    run more than once, only its latest row counts.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=group_cols + ['seed'], keep='last')
    metric_cols = [c for c in df.columns if c.endswith(('_acc', '_f1'))]
    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row['n_seeds'] = len(g)
        row['seeds'] = ' '.join(str(s) for s in sorted(g['seed']))
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors='coerce').dropna()
            row[f'{c}_mean'] = round(vals.mean(), 2) if len(vals) else np.nan
            row[f'{c}_std'] = round(vals.std(ddof=0), 2) if len(vals) else np.nan
        rows.append(row)
    os.makedirs(os.path.dirname(summary_path) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[OK] summary (mean/std over seeds) written to: {summary_path}")
