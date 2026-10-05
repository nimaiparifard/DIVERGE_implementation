"""
On-disk layout of the per-approach GNN outputs shared by the two stages:

  <logits_root>/<dataset>/<llm>_<peft>_<gnn>_<split>_seed<seed>/
      <approach>_logits.pt   {'logits', 'results', 'meta'}
      <approach>_gnn.pt      {'state_dict', 'gnn_cfg', 'input_dim', 'num_classes'} (--save_weights)

'results' holds the scalar metrics/cost of gnn_train_and_report (what DIVERGE's report and
CSV need), 'meta' the provenance plus a fingerprint (GNN hyperparameters + embedding cache
file) used to detect stale logits after re-tuning or re-caching.
"""
import os

import torch

DEFAULT_LOGITS_ROOT = os.path.join('artifacts', 'diverge_logits')

# Scalar entries of gnn_train_and_report's results kept with the logits
RESULT_KEYS = ('test_acc', 'test_f1', 'test_weight_f1', 'val_acc', 'val_f1', 'val_weight_f1',
               'train_time_sec', 'peak_vram_mb', 'peak_vram_delta_mb', 'epochs_run', 'time_per_epoch_ms')
# GNN hyperparameters (set_mgnn_cfg) that change the trained model
GNN_CFG_KEYS = ('seed', 'gnn_model_name', 'hidden_dim', 'num_layers', 'dropout', 'lr', 'weight_decay',
                'epochs', 'early_stop', 'batch_norm', 're_split', 'hp_source')


def split_tag(supervised):
    return "supervised" if supervised else "semi_supervised"


def run_dir(dataset_name, llm_name, peft_type, gnn_tag, supervised, seed, logits_root=DEFAULT_LOGITS_ROOT):
    return os.path.join(logits_root, dataset_name,
                        f"{llm_name}_{peft_type}_{gnn_tag}_{split_tag(supervised)}_seed{seed}")


def logits_path(directory, approach):
    return os.path.join(directory, f"{approach}_logits.pt")


def weights_path(directory, approach):
    return os.path.join(directory, f"{approach}_gnn.pt")


def _py(value):
    """numpy/torch scalars -> plain Python (torch.load(weights_only=True) only accepts primitives)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return float(value)


def gnn_cfg_dict(gnn_cfg):
    return {k: _py(gnn_cfg[k]) for k in GNN_CFG_KEYS}


def fingerprint(gnn_cfg, embedding_path):
    """What the saved logits depend on besides (dataset, llm, peft, gnn, split, seed)."""
    return {
        "gnn_cfg": gnn_cfg_dict(gnn_cfg),
        "embedding_file": os.path.basename(embedding_path),
        "embedding_mtime": os.path.getmtime(embedding_path) if os.path.exists(embedding_path) else None,
    }


def is_current(path, expected_fingerprint):
    """True if `path` exists and was produced with the same fingerprint."""
    if not os.path.exists(path):
        return False
    try:
        meta = torch.load(path, map_location='cpu')['meta']
    except Exception as e:
        print(f"[logits] unreadable {path} ({e}); will regenerate")
        return False
    return meta.get("fingerprint") == expected_fingerprint


def save_logits(path, logits, results, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "logits": logits.detach().float().cpu(),
        "results": {k: _py(results.get(k)) for k in RESULT_KEYS},
        "meta": meta,
    }, path)


def save_weights(path, model, gnn_cfg, input_dim, num_classes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "gnn_cfg": gnn_cfg_dict(gnn_cfg),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
    }, path)


def load_run(directory, approaches, device='cpu'):
    """
    Load the saved logits of every approach of one run.

    Returns:
        results: {approach: scalar metrics dict}
        logits:  {approach: [num_nodes, num_classes] tensor on `device`}
        metas:   {approach: meta dict}
    """
    missing = [a for a in approaches if not os.path.exists(logits_path(directory, a))]
    if missing:
        raise FileNotFoundError(
            f"No saved logits for {missing} in {directory}\n"
            f"Run python -m diverge_for_larger_tags.train_gnn_export_logits first (same dataset/llm/gnn/split/seed).")

    results, logits, metas = {}, {}, {}
    for a in approaches:
        bundle = torch.load(logits_path(directory, a), map_location='cpu')
        results[a], logits[a], metas[a] = bundle["results"], bundle["logits"].to(device), bundle["meta"]

    shapes = {a: tuple(t.shape) for a, t in logits.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Saved logits have mismatching shapes: {shapes}")
    cfgs = {a: m["fingerprint"]["gnn_cfg"] for a, m in metas.items()}
    if len({repr(sorted(c.items())) for c in cfgs.values()}) != 1:
        print(f"[logits] WARNING: approaches were trained with different GNN hyperparameters: {cfgs}")
    return results, logits, metas
