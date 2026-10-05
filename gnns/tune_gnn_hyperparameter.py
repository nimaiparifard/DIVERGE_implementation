"""
Tune GNN hyperparameters on a cached LLM embedding, per (dataset, LLM, seed, GNN encoder).

Search strategy (successive halving, no extra dependencies):
  1. candidates  : full grid, or `--n_trials` random configs sampled from the grid
  2. rung 0      : every candidate trained with a short budget (`--rung0_epochs`), one seed,
                   and pruned once it falls clearly behind the best trial so far
  3. final rung  : the `--top_k` best candidates re-trained with the full budget and averaged
                   over `--n_eval_seeds` GNN init seeds; the best mean val acc wins
The graph is loaded once, features live on the device for the whole search, per-epoch
metrics are computed on-device, and only the winning config is plotted. Every trial is
logged to results/gnn_tuning/<run>/trials.jsonl, so an interrupted run resumes where it stopped.

The winner is saved to gnn_hyperparameters/<dataset>_<llm>_<gnn>[_semi_supervised]_seed<seed>.json,
which gnns.gnn_mtrainer.set_mgnn_cfg picks up automatically. Check with
gnns/verify_gnn_hyperparameters.py that GNN training actually uses it.
"""
from gnns.gnn_mtrainer import (set_mgnn_cfg, GNNTrainer, get_datasets_path, append_gnn_result_csv,
                               resolve_path_prefix, gnn_hyperparameter_filename, get_gnn_hyperparameter_dir,
                               SUPPORTED_GNN_MODELS, DEFAULT_GNN_MODEL)
import itertools
import json
import random
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TUNING_RESULTS_DIR = REPO_ROOT / "results" / "gnn_tuning"
INIT_WEIGHT_APPROACHES = ['pissa', 'gaussian', 'eva', 'loftq', 'orthogonal']


def _resolve_cfg_device(device):
    """Map CLI device string/int to cfg.device (>0 => CUDA, else CPU)."""
    if device == -1 or device == 'cpu':
        return 0
    if isinstance(device, str) and device.startswith('cuda'):
        return 1
    if isinstance(device, int) and device > 0:
        return device
    return 1 if torch.cuda.is_available() else 0


def get_embedding_cache_path(dataset_name, llm_name='llama_3.2_1B', peft_type='lora',
                              init_weight_approach='pissa', supervised=True, seed=None):
    """
    Return path to the LLM embedding cache to tune GNN hyperparameters on.

    Mirrors the naming used by cache/cache_embedding_with_diffrent_init_weights.py and
    dataset.data_utils.get_init_dataset_for_gnn: the seed-tagged semi-supervised cache
    (also used by the supervised split) is preferred, then the legacy (pre-seed)
    semi-supervised filename, then the plain un-suffixed cache.
    """
    cache_dir = Path(get_datasets_path()).parent / 'artifacts' / 'cache'
    base_name = f'{llm_name}_{dataset_name}_seqcls_{peft_type}_init-{init_weight_approach}_pool-mean'
    candidates = []
    if seed is not None:
        candidates.append(cache_dir / f'{base_name}_seed{seed}_semi_supervised.pt')
    candidates.append(cache_dir / f'{base_name}_semi_supervised.pt')
    candidates.append(cache_dir / f'{base_name}.pt')
    return next((p for p in candidates if p.exists()), candidates[0])


def get_candidate_grid_search_hyperparamters(search_space_path=None):
    """Search space as {name: [values]}; a JSON file with the same shape overrides it."""
    if search_space_path:
        with open(search_space_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    candidate_hyperparameters = {
        "learning_rate": [0.01, 0.005, 0.001, 0.0001],
        "hidden_dim": [128, 256, 512, 1024],
        "num_layers": [2, 3],
        "dropout": [0.0, 0.1, 0.01, 0.5],
        "weight_decay": [0.0, 0.01],
        "batch_norm": [0, 1],

    }
    return candidate_hyperparameters


def sample_candidates(search_space, search='random', n_trials=60, seed=0):
    """All grid combinations, or `n_trials` of them sampled without replacement (deterministic per seed)."""
    keys = list(search_space.keys())
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*search_space.values())]
    if search == 'grid' or n_trials >= len(grid):
        return grid
    return random.Random(seed).sample(grid, n_trials)


def _config_key(hp):
    return json.dumps(hp, sort_keys=True)


def apply_hyperparameters(cfg, hp):
    cfg.lr = float(hp["learning_rate"])
    cfg.hidden_dim = int(hp["hidden_dim"])
    cfg.num_layers = int(hp["num_layers"])
    cfg.dropout = float(hp["dropout"])
    cfg.weight_decay = float(hp["weight_decay"])
    cfg.batch_norm = bool(int(hp["batch_norm"]))
    return cfg


class TrialLog:
    """Append-only JSONL log of finished trials, keyed by (config, seed, budget), for resuming."""

    def __init__(self, path, resume=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        if resume and self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.records[self._key(rec['hp'], rec['seed'], rec['epochs'])] = rec
            print(f"Resuming: {len(self.records)} finished trials loaded from {self.path}")
        elif not resume and self.path.exists():
            self.path.unlink()

    @staticmethod
    def _key(hp, seed, epochs):
        return f"{_config_key(hp)}|{seed}|{epochs}"

    def get(self, hp, seed, epochs):
        return self.records.get(self._key(hp, seed, epochs))

    def add(self, rec):
        self.records[self._key(rec['hp'], rec['seed'], rec['epochs'])] = rec
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + "\n")


def run_trial(base_cfg, hp, features, data, seed, epochs, early_stop, prune_below=None, grace_epochs=0,
              track_history=False):
    """Train one GNN config; returns metrics (+ history if track_history)."""
    cfg = apply_hyperparameters(base_cfg.clone(), hp)
    cfg.seed = int(seed)
    cfg.epochs = int(epochs)
    cfg.early_stop = int(early_stop)

    running_best = [0.0]
    pruned = [False]

    def prune_callback(epoch, val_acc):
        running_best[0] = max(running_best[0], val_acc)
        if prune_below is not None and epoch >= grace_epochs and running_best[0] < prune_below:
            pruned[0] = True
            return True
        return False

    t0 = time.perf_counter()
    trainer = GNNTrainer(cfg, features, data=data, track_history=track_history, verbose=False)
    results, _, history = trainer.train(epoch_callback=prune_callback)
    rec = {
        "hp": hp, "seed": int(seed), "epochs": int(epochs),
        "val_acc": float(results["val_acc"]), "test_acc": float(results["test_acc"]),
        "val_f1": float(results["val_f1"]), "test_f1": float(results["test_f1"]),
        "epochs_run": len(history["loss"]), "pruned": pruned[0],
        "time_sec": round(time.perf_counter() - t0, 3),
    }
    return (rec, history) if track_history else rec


def tune_gnn_hyperparamters(dataset_name, features, device, supervised=True, seed=None, llm_name=None,
                            gnn_model_name=None, search='random', n_trials=60, rung0_epochs=50, top_k=8,
                            n_eval_seeds=3, prune_ratio=0.9, search_space=None, epochs=None, early_stop=None,
                            log_path=None, resume=True, data=None):
    """
    Successive-halving hyperparameter search for one (dataset, llm, seed, gnn) combination.

    Args:
        features: node features (num_nodes, feature_dim), e.g. a cached LLM embedding.
        seed: data-split / base GNN seed (also the LLM adapter seed the embedding came from).
        gnn_model_name: GNN encoder to tune (see SUPPORTED_GNN_MODELS).
        search: 'grid' (all combinations) or 'random' (`n_trials` sampled combinations).
        rung0_epochs: short screening budget; 0 screens every candidate with the full budget.
        top_k: candidates promoted to the full-budget, multi-seed final rung.
        n_eval_seeds: GNN init seeds averaged in the final rung (seed, seed+1000, ...).
        prune_ratio: stop a screening run whose best val acc stays below prune_ratio * best
            trial so far (after its early-stop grace period); 0 disables pruning.
        search_space: {name: [values]} (defaults to get_candidate_grid_search_hyperparamters()).
        epochs, early_stop: full budget (defaults to the values in the resolved cfg).
        log_path: JSONL trial log used for resuming.
        data: pre-loaded graph (loaded here if None).

    Returns:
        best_hyperparameters (dict), best_results (dict with mean val/test acc & f1 and per-seed values)
    """
    gnn_model_name = gnn_model_name or DEFAULT_GNN_MODEL
    cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name,
                       gnn_model_name=gnn_model_name)
    cfg.device = _resolve_cfg_device(device)
    cfg.re_split = bool(supervised)
    full_epochs = int(epochs or cfg.epochs)
    full_early_stop = int(early_stop or cfg.early_stop)
    base_seed = cfg.seed

    torch_device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
    if data is None:
        from common import load_graph_dataset_for_tape
        data, _, _ = load_graph_dataset_for_tape(dataset_name, torch_device, re_split=cfg.re_split,
                                                 path_prefix=resolve_path_prefix(), seed=cfg.split_seed)
    data = data.to(torch_device)
    features = features.to(torch_device)

    space = search_space or get_candidate_grid_search_hyperparamters()
    candidates = sample_candidates(space, search=search, n_trials=n_trials, seed=base_seed)
    log = TrialLog(log_path or (TUNING_RESULTS_DIR / "trials.jsonl"), resume=resume)

    if rung0_epochs and rung0_epochs < full_epochs:
        r0_epochs = int(rung0_epochs)
        r0_early_stop = min(full_early_stop, max(5, r0_epochs // 5))
    else:
        r0_epochs, r0_early_stop = full_epochs, full_early_stop
    eval_seeds = [base_seed + 1000 * i for i in range(max(1, n_eval_seeds))]

    print(f"GNN={gnn_model_name} | search={search} | candidates={len(candidates)} | "
          f"rung0: {r0_epochs} epochs (early_stop {r0_early_stop}) | final: top {top_k} x seeds {eval_seeds}, "
          f"{full_epochs} epochs (early_stop {full_early_stop})")

    # ---- Rung 0: screen every candidate with a short budget -------------------------------
    t_start = time.perf_counter()
    rung0 = []
    # Model selection uses validation accuracy only; test metrics are logged for reporting, never for selection.
    best_so_far = max((r['val_acc'] for r in log.records.values() if r['epochs'] == r0_epochs), default=0.0)
    for idx, hp in enumerate(candidates):
        rec = log.get(hp, base_seed, r0_epochs)
        if rec is None:
            prune_below = prune_ratio * best_so_far if prune_ratio and best_so_far > 0 else None
            try:
                rec = run_trial(cfg, hp, features, data, base_seed, r0_epochs, r0_early_stop,
                                prune_below=prune_below, grace_epochs=r0_early_stop)
            except Exception as e:
                print(f"[{idx + 1}/{len(candidates)}] FAILED {hp}: {e}")
                continue
            rec["rung"] = 0
            log.add(rec)
        rung0.append(rec)
        if rec['val_acc'] > best_so_far:
            best_so_far = rec['val_acc']
            print(f"[{idx + 1}/{len(candidates)}] new best val_acc={best_so_far:.4f} {hp}")
        elif (idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{len(candidates)}] screened (best val_acc so far {best_so_far:.4f})")

    if not rung0:
        raise RuntimeError("Every hyperparameter candidate failed; see the errors above.")

    # ---- Final rung: top-k with the full budget, averaged over seeds ------------------------
    promoted = sorted(rung0, key=lambda r: r['val_acc'], reverse=True)[:max(1, top_k)]
    print(f"\nRung 0 done in {time.perf_counter() - t_start:.1f}s; promoting {len(promoted)} configs.")
    final = []
    for rank, r0 in enumerate(promoted):
        hp = r0['hp']
        per_seed = []
        for s in eval_seeds:
            rec = log.get(hp, s, full_epochs)
            if rec is None or rec.get('pruned'):  # a pruned screening run (rung0 == full budget) is incomplete
                rec = run_trial(cfg, hp, features, data, s, full_epochs, full_early_stop)
                rec["rung"] = 1
                log.add(rec)
            per_seed.append(rec)
        summary = {
            "hp": hp,
            "val_acc": float(np.mean([r['val_acc'] for r in per_seed])),
            "val_acc_std": float(np.std([r['val_acc'] for r in per_seed])),
            "test_acc": float(np.mean([r['test_acc'] for r in per_seed])),
            "test_acc_std": float(np.std([r['test_acc'] for r in per_seed])),
            "val_f1": float(np.mean([r['val_f1'] for r in per_seed])),
            "test_f1": float(np.mean([r['test_f1'] for r in per_seed])),
            "val_acc_per_seed": [r['val_acc'] for r in per_seed],
            "test_acc_per_seed": [r['test_acc'] for r in per_seed],
        }
        final.append(summary)
        print(f"  [{rank + 1}/{len(promoted)}] val {summary['val_acc']:.4f}+/-{summary['val_acc_std']:.4f} "
              f"test {summary['test_acc']:.4f}+/-{summary['test_acc_std']:.4f} {hp}")

    best = max(final, key=lambda r: (r['val_acc'], -r['val_acc_std']))
    best_hyperparameters = dict(best['hp'])
    best_results = {k: v for k, v in best.items() if k != 'hp'}
    best_results.update({
        "eval_seeds": eval_seeds, "n_candidates": len(candidates), "n_promoted": len(promoted),
        "epochs": full_epochs, "early_stop": full_early_stop, "gnn_model_name": gnn_model_name,
        "search": search, "total_time_sec": round(time.perf_counter() - t_start, 1),
        "device": cfg.device, "seed": base_seed,
    })

    print(f"\n{'='*80}")
    print(f"Best hyperparameters ({gnn_model_name}): {best_hyperparameters}")
    print(f"Best Val Acc: {best['val_acc']:.4f} +/- {best['val_acc_std']:.4f} (over seeds {eval_seeds})")
    print(f"Best Test Acc: {best['test_acc']:.4f} +/- {best['test_acc_std']:.4f}")
    print(f"Best Val F1: {best['val_f1']:.4f}")
    print(f"Best Test F1: {best['test_f1']:.4f}")
    print(f"Tuning time: {best_results['total_time_sec']}s")
    return best_hyperparameters, best_results


def save_hyperparameters(dataset_name, best_hyperparameters, re_split=1, llm_name=None, seed=None,
                         gnn_model_name=None, best_results=None, tuning_info=None):
    """
    Save the best hyperparameters to gnn_hyperparameters/<name>.json, named with
    gnns.gnn_mtrainer.gnn_hyperparameter_filename so set_mgnn_cfg resolves it first for
    this (dataset, llm, gnn, seed). Returns the saved path.
    """
    gnn_model_name = gnn_model_name or DEFAULT_GNN_MODEL
    best_results = best_results or {}
    save_dir = get_gnn_hyperparameter_dir()
    save_dir.mkdir(exist_ok=True)

    hyperparams_to_save = {
        "seed": best_results.get("seed", seed if seed is not None else 42),
        "device": best_results.get("device", 1),
        "dataset": dataset_name,
        "gnn_model_name": gnn_model_name,
        "llm_name": llm_name if llm_name is not None else "llama_3.2_1B",
        "epochs": best_results.get("epochs", 200),
        "early_stop": best_results.get("early_stop", 20),
        "re_split": int(bool(re_split)),
        "lr": best_hyperparameters["learning_rate"],
        "hidden_dim": best_hyperparameters["hidden_dim"],
        "num_layers": best_hyperparameters["num_layers"],
        "dropout": best_hyperparameters["dropout"],
        "weight_decay": best_hyperparameters["weight_decay"],
        "batch_norm": int(best_hyperparameters["batch_norm"]),
        # Provenance, read by gnns/verify_gnn_hyperparameters.py (ignored by set_mgnn_cfg)
        "tuning": {
            **(tuning_info or {}),
            **{k: best_results[k] for k in ("val_acc", "val_acc_std", "test_acc", "test_acc_std", "val_f1",
                                             "test_f1", "val_acc_per_seed", "test_acc_per_seed", "eval_seeds",
                                             "n_candidates", "n_promoted", "search", "total_time_sec")
               if k in best_results},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    fname = gnn_hyperparameter_filename(dataset_name, supervised=bool(re_split), llm_name=llm_name, seed=seed,
                                        gnn_model_name=gnn_model_name)
    save_path = save_dir / fname
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams_to_save, f, indent=4)

    print(f"\nBest hyperparameters saved to: {save_path}")
    return save_path


def visualize_learning_curve(dataset_name, hyperparameters, history, save_path):
    """Plot loss / accuracy / F1 curves of one run (history from GNNTrainer with track_history=True)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    for ax, metric, title in ((axes[1], 'acc', 'Accuracy'), (axes[2], 'f1', 'F1 Score')):
        ax.plot(epochs, history[f'train_{metric}'], 'g-', linewidth=2, label='Train', alpha=0.7)
        ax.plot(epochs, history[f'val_{metric}'], 'b-', linewidth=2, label='Val')
        ax.plot(epochs, history[f'test_{metric}'], 'r-', linewidth=2, label='Test')
        ax.set_title(title, fontsize=14, fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    hyperparams_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())
    fig.suptitle(f'Learning Curves - {dataset_name}\n{hyperparams_str}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Learning curve saved to: {save_path}")
    return history


if __name__ == "__main__":
    from common import load_graph_dataset_for_tape
    import argparse

    parser = argparse.ArgumentParser(description='Tune GNN hyperparameters per dataset / LLM / seed / GNN encoder')
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        help='Dataset name (e.g., cora, citeseer, pubmed, wikics)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0). If not specified, will auto-detect')
    parser.add_argument('--re_split', type=int, default=0,
                        help='Re-split value (0 for semi-supervised, >0 for supervised)')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B',
                        help='LLM whose embedding cache to tune on (must match a cached LoRA adapter)')
    parser.add_argument('--peft_type', type=str, default='lora',
                        help='PEFT type used when the adapter/embedding cache was generated')
    parser.add_argument('--init_weight_approach', type=str, default='pissa', choices=INIT_WEIGHT_APPROACHES,
                        help='LoRA init approach whose embedding cache to tune on')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed the LoRA adapter/embedding cache was trained with; also the data-split and '
                             'base GNN seed. The tuned file is saved per seed.')
    parser.add_argument('--gnn_model_name', type=str, default=DEFAULT_GNN_MODEL, choices=SUPPORTED_GNN_MODELS,
                        help='GNN encoder to tune')
    parser.add_argument('--search', type=str, default='random', choices=['random', 'grid'],
                        help="'random' samples --n_trials configs from the grid; 'grid' screens all of them")
    parser.add_argument('--n_trials', type=int, default=60, help='Candidates to sample for --search random')
    parser.add_argument('--rung0_epochs', type=int, default=50,
                        help='Short screening budget per candidate (0 = screen with the full budget)')
    parser.add_argument('--top_k', type=int, default=8, help='Candidates promoted to the full multi-seed rung')
    parser.add_argument('--n_eval_seeds', type=int, default=3, help='GNN init seeds averaged in the final rung')
    parser.add_argument('--prune_ratio', type=float, default=0.9,
                        help='Stop a screening run below prune_ratio * best val acc so far (0 disables)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='Optional JSON file {"learning_rate": [...], "hidden_dim": [...], ...}')
    parser.add_argument('--epochs', type=int, default=None, help='Full training budget (default: from cfg)')
    parser.add_argument('--early_stop', type=int, default=None, help='Full early-stop patience (default: from cfg)')
    parser.add_argument('--no_resume', action='store_true', help='Ignore/overwrite the previous trial log')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip tuning if the target hyperparameter file already exists')
    parser.add_argument('--no_plot', action='store_true', help='Do not plot the learning curve of the best config')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    llm_name = args.llm_name
    seed = args.seed
    gnn_model_name = args.gnn_model_name
    device = args.device if args.device is not None else ('cuda:0' if torch.cuda.is_available() else -1)
    re_split = args.re_split
    supervised = re_split > 0

    target = get_gnn_hyperparameter_dir() / gnn_hyperparameter_filename(
        dataset_name, supervised, llm_name=llm_name, seed=seed, gnn_model_name=gnn_model_name)
    if args.skip_existing and target.exists():
        print(f"[skip] {target} already exists")
        raise SystemExit(0)

    print(f"=== Tuning GNN hyperparameters | dataset={dataset_name} llm={llm_name} gnn={gnn_model_name} "
          f"init={args.init_weight_approach} supervised={supervised} seed={seed} device={device}")

    cache_path = get_embedding_cache_path(dataset_name, llm_name=llm_name, peft_type=args.peft_type,
                                          init_weight_approach=args.init_weight_approach, supervised=supervised,
                                          seed=seed)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}\n"
            f"Run cache/cache_embedding_with_diffrent_init_weights.py for llm '{llm_name}', "
            f"init approach '{args.init_weight_approach}'"
            + (f", seed {seed}" if seed is not None else "") + " first."
        )
    if not supervised and seed is not None and f"_seed{seed}_" not in cache_path.name:
        print(f"WARNING: no seed-{seed} cache found, falling back to the un-seeded cache {cache_path.name}")
    features = torch.load(cache_path, weights_only=False)['embeddings']

    # Load the graph once for the whole search
    base_cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=seed, llm_name=llm_name,
                            gnn_model_name=gnn_model_name)
    torch_device = torch.device("cuda:0" if _resolve_cfg_device(device) > 0 else "cpu")
    data, num_classes, _ = load_graph_dataset_for_tape(dataset_name, torch_device, re_split=supervised,
                                                       path_prefix=resolve_path_prefix(), seed=base_cfg.split_seed)
    print(f"Dataset loaded - Nodes: {features.shape[0]}, Features: {features.shape[1]}, Classes: {num_classes}")

    run_tag = (f"{dataset_name}_{llm_name}_{gnn_model_name}_{'supervised' if supervised else 'semi_supervised'}"
               f"_seed{base_cfg.seed}_init-{args.init_weight_approach}")
    run_dir = TUNING_RESULTS_DIR / run_tag
    search_space = get_candidate_grid_search_hyperparamters(args.search_space)

    best_hyperparameters, best_results = tune_gnn_hyperparamters(
        dataset_name, features, device, supervised=supervised, seed=seed, llm_name=llm_name,
        gnn_model_name=gnn_model_name, search=args.search, n_trials=args.n_trials,
        rung0_epochs=args.rung0_epochs, top_k=args.top_k, n_eval_seeds=args.n_eval_seeds,
        prune_ratio=args.prune_ratio, search_space=search_space, epochs=args.epochs,
        early_stop=args.early_stop, log_path=run_dir / "trials.jsonl", resume=not args.no_resume, data=data)

    tuning_info = {
        "init_weight_approach": args.init_weight_approach,
        "peft_type": args.peft_type,
        "embedding_cache": str(cache_path.relative_to(REPO_ROOT)) if cache_path.is_relative_to(REPO_ROOT)
        else str(cache_path),
        "trial_log": str((run_dir / "trials.jsonl").relative_to(REPO_ROOT)),
        "search_space": search_space,
    }
    save_path = save_hyperparameters(dataset_name, best_hyperparameters, re_split=re_split, llm_name=llm_name,
                                     seed=seed, gnn_model_name=gnn_model_name, best_results=best_results,
                                     tuning_info=tuning_info)

    if not args.no_plot:
        base_cfg.device = _resolve_cfg_device(device)
        rec, history = run_trial(base_cfg, best_hyperparameters, features, data, best_results["seed"],
                                 best_results["epochs"], best_results["early_stop"], track_history=True)
        visualize_learning_curve(dataset_name, best_hyperparameters, history, run_dir / "best_learning_curve.png")

    append_gnn_result_csv({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name, "llm_name": llm_name, "gnn_model_name": gnn_model_name,
        "seed": best_results["seed"], "supervised": supervised, "init_weight_approach": args.init_weight_approach,
        **{k: best_hyperparameters[k] for k in best_hyperparameters},
        "val_acc": best_results["val_acc"], "val_acc_std": best_results["val_acc_std"],
        "test_acc": best_results["test_acc"], "test_acc_std": best_results["test_acc_std"],
        "n_candidates": best_results["n_candidates"], "total_time_sec": best_results["total_time_sec"],
        "hp_file": save_path.name,
    }, str(TUNING_RESULTS_DIR / "tuning_summary.csv"))

    print(f"\n{'='*80}")
    print("Hyperparameter tuning completed!")
    print(f"Best hyperparameters: {save_path}")
    print(f"Trial log / learning curve: {run_dir}")
    print(f"Verify with: python -m gnns.verify_gnn_hyperparameters --dataset_names {dataset_name} "
          f"--llm_names {llm_name} --gnn_model_names {gnn_model_name} --re_split {re_split}"
          + (f" --seeds {seed}" if seed is not None else "") + " --retrain")
