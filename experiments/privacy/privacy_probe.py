"""Keyword-inference probe on saved DIVERGE node embeddings (Reviewer 1, Comment 8).

Question: can node text leak through the saved embeddings?
Attack: an adversary who obtains the embedding file trains a linear probe that predicts,
for each of the K most frequent content words, whether the word occurs in the node text.
Leakage is the mean per-word test ROC-AUC (0.5 = no leakage).

Controls / defense:
  - "label only": the same probe fed only the one-hot class label, i.e. what is inferable
    from the task itself, so AUC above this is leakage beyond the label.
  - Gaussian noise: e' = e + sigma * std_dim * N(0, 1) added to the stored embeddings; we report
    probe AUC and the GraphSAGE test accuracy obtained when the GNN is trained on e'.

Usage (from repo root):
  python experiments/privacy/privacy_probe.py                                   # all datasets in RUNS
  python experiments/privacy/privacy_probe.py --llm smollm2_1.7B --datasets cora,citeseer --seeds 42,43
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.dataloader import load_graph_dataset  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Seeds whose five per-seed LoRA caches exist; the adapter of seed s was fine-tuned on split s,
# so the probe and the GNN use the same split as the adapter (same embeddings as the main results).
RUNS = {"cora": [42, 43, 44, 45], "citeseer": [42, 43, 44, 45], "pubmed": [43],
        "wikics": [42], "reddit": [42], "computer": [42]}
INITS = ["gaussian", "orthogonal", "pissa", "eva", "loftq"]
NOISE = [0.5, 1.0, 2.0]
TOP_K = 100
GNN_SEEDS = [0, 1, 2]
OUT = os.path.join(ROOT, "results", "privacy")
DEFAULT_LLM = "llama_3.2_1B"


def out_file(llm):
    # The llama run keeps the original file name so existing results are resumed.
    return "privacy_probe_seeds.json" if llm == DEFAULT_LLM else f"privacy_probe_seeds_{llm}.json"


def emb_path(llm, ds, init, seed):
    name = (f"{llm}_{ds}_seqcls_pool-mean_based_model.pt" if init == "base"
            else f"{llm}_{ds}_seqcls_lora_init-{init}_pool-mean_seed{seed}_semi_supervised.pt")
    return os.path.join(ROOT, "artifacts", "cache", name)


def load_emb(llm, ds, init, n, seed):
    d = torch.load(emb_path(llm, ds, init, seed), weights_only=False, map_location="cpu")
    assert torch.equal(d["indices"].cpu(), torch.arange(n))
    return d["embeddings"].float()


def keyword_targets(texts, k):
    docs = [set(w for w in re.findall(r"[a-z]+", t.lower()) if len(w) >= 3 and w not in ENGLISH_STOP_WORDS)
            for t in texts]
    df = Counter(w for d in docs for w in d)
    vocab = [w for w, _ in df.most_common(k)]
    Y = torch.tensor([[w in d for w in vocab] for d in docs], dtype=torch.float32)
    return vocab, Y


def probe_auc(X, Y, tr, va, te, epochs=300):
    """Multi-output logistic-regression probe; epoch and weight decay chosen on validation AUC."""
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd

    def mean_auc(P, idx):
        y, p = Y[idx].cpu().numpy(), P[idx].cpu().numpy()
        return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])
                              if 0 < y[:, j].sum() < len(y)]))

    best = (-1, None)
    for wd in [1e-4, 1e-3, 1e-2]:
        torch.manual_seed(0)
        lin = nn.Linear(X.shape[1], Y.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=wd)
        for ep in range(epochs):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(lin(X[tr]), Y[tr]).backward()
            opt.step()
            if (ep + 1) % 25 == 0:
                with torch.no_grad():
                    P = lin(X)
                v = mean_auc(P, va)
                if v > best[0]:
                    best = (v, mean_auc(P, te))
    return best[1]


class SAGE(nn.Module):
    def __init__(self, d_in, d_h, d_out, layers, dropout, bn):
        super().__init__()
        dims = [d_in] + [d_h] * (layers - 1) + [d_out]
        self.convs = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(d_h) for _ in range(layers - 1)) if bn else None
        self.dropout = dropout

    def forward(self, x, ei):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, ei)
            if self.bns is not None:
                x = self.bns[i](x)
            x = F.dropout(F.relu(x), self.dropout, self.training)
        return self.convs[-1](x, ei)


def gnn_acc(X, data, hp):
    accs = []
    for s in GNN_SEEDS:
        torch.manual_seed(s)
        m = SAGE(X.shape[1], hp["hidden_dim"], int(data.y.max()) + 1, hp["num_layers"],
                 hp["dropout"], hp["batch_norm"]).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        x = X.to(DEVICE)
        best_va, best_te, wait = -1, 0, 0
        for _ in range(hp["epochs"]):
            m.train()
            opt.zero_grad()
            F.cross_entropy(m(x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
            opt.step()
            m.eval()
            with torch.no_grad():
                pred = m(x, data.edge_index).argmax(1)
            va = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            te = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            if va > best_va:
                best_va, best_te, wait = va, te, 0
            else:
                wait += 1
                if wait >= hp["early_stop"]:
                    break
        accs.append(best_te * 100)
    return float(np.mean(accs))


def run(llm, ds, seed):
    data = load_graph_dataset(ds, DEVICE, re_split=True, path_prefix=ROOT, seed=seed)
    n = data.num_nodes
    hp_dir = os.path.join(ROOT, "gnn_hyperparameters")
    cands = [os.path.join(hp_dir, f"{ds}_{llm}_SAGE_seed{s}.json") for s in (seed, 42)]
    hp_file = next((p for p in cands if os.path.exists(p)), None)
    if hp_file is None:
        raise FileNotFoundError(f"no SAGE hyperparameters for {ds}/{llm}: tried {cands}")
    with open(hp_file) as f:
        hp = json.load(f)
    vocab, Y = keyword_targets(data.raw_texts, TOP_K)
    tr, va, te = (m.nonzero().squeeze(1).cpu() for m in (data.train_mask, data.val_mask, data.test_mask))
    print(f"[{llm} {ds} seed{seed}] n={n} train/val/test={len(tr)}/{len(va)}/{len(te)} "
          f"keyword prevalence={Y.mean():.3f} top words={vocab[:10]}", flush=True)

    res = {"vocab": vocab, "rows": {}}
    onehot = F.one_hot(data.y.cpu(), int(data.y.max()) + 1).float()
    res["rows"]["label_only"] = {"auc": probe_auc(onehot, Y, tr, va, te), "acc": None}
    print(f"  label_only auc={res['rows']['label_only']['auc']:.4f}", flush=True)

    gen = torch.Generator().manual_seed(123)
    for init in ["base"] + INITS:
        X = load_emb(llm, ds, init, n, seed)
        row = {"auc": probe_auc(X, Y, tr, va, te), "acc": gnn_acc(X, data, hp)}
        res["rows"][init] = row
        print(f"  {init:10s} auc={row['auc']:.4f} acc={row['acc']:.2f}", flush=True)
        if init == "base":
            continue
        std = X.std(0, keepdim=True)
        for s in NOISE:
            Xn = X + s * std * torch.randn(X.shape, generator=gen)
            r = {"auc": probe_auc(Xn, Y, tr, va, te), "acc": gnn_acc(Xn, data, hp)}
            res["rows"][f"{init}_noise{s}"] = r
            print(f"  {init:10s} sigma={s} auc={r['auc']:.4f} acc={r['acc']:.2f}", flush=True)

    for s in [0.0] + NOISE:
        keys = INITS if s == 0 else [f"{i}_noise{s}" for i in INITS]
        res["rows"][f"lora_mean_noise{s}"] = {
            k: float(np.mean([res["rows"][x][k] for x in keys])) for k in ("auc", "acc")}
    return res


def summarize(per_seed):
    """Mean/std over seeds of auc and acc for every row."""
    out = {}
    for key in per_seed[0]["rows"]:
        out[key] = {}
        for k in ("auc", "acc"):
            vals = [r["rows"][key][k] for r in per_seed if r["rows"][key][k] is not None]
            out[key][k] = [float(np.mean(vals)), float(np.std(vals))] if vals else None
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default=DEFAULT_LLM, help="cache/hyperparameter name prefix, e.g. llama_3.2_1B, "
                    "smollm2_1.7B, gemma2_2B (default: %(default)s)")
    ap.add_argument("--datasets", default=",".join(RUNS), help="comma-separated (default: %(default)s)")
    ap.add_argument("--seeds", default=None, help="comma-separated split/adapter seeds used for every dataset "
                    "(default: per-dataset seeds in RUNS)")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        runs = {ds: seeds for ds in datasets}
    else:
        missing = [ds for ds in datasets if ds not in RUNS]
        if missing:
            ap.error(f"no default seeds for {missing}; pass --seeds")
        runs = {ds: RUNS[ds] for ds in datasets}
    return args.llm, runs


if __name__ == "__main__":
    llm, runs = parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, out_file(llm))
    allres = {}
    if os.path.exists(out_path):  # resume: a dataset is skipped when its requested seeds are all done
        with open(out_path) as f:
            allres = json.load(f)
    for ds, seeds in runs.items():
        if allres.get(ds, {}).get("seeds") == seeds:
            print(f"[{ds}] seeds {seeds} already done, skipping", flush=True)
            continue
        per_seed = []
        for seed in seeds:
            per_seed.append(run(llm, ds, seed))
            allres[ds] = {"seeds": seeds[:len(per_seed)], "per_seed": per_seed, "summary": summarize(per_seed)}
            with open(out_path, "w") as f:
                json.dump(allres, f, indent=2)
    print(json.dumps({ds: {k: v for k, v in allres[ds]["summary"].items()
                           if k.startswith(("label", "base", "lora_mean"))}
                      for ds in runs if ds in allres}, indent=2))
