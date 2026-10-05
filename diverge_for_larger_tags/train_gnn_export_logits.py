"""
Stage 1 of DIVERGE for large TAGs: train one GNN per LoRA-init approach and save its logits.

experiments/diverge.py loads all 5 embedding caches onto the GPU before training, which
does not fit on a 12 GB card for large TAGs (computer, arxiv, ...). Here, approaches are
processed one at a time:

  load that approach's embeddings (CPU) -> train the GNN (same tuned hyperparameters,
  seed and per-GNN CSV row as experiments/diverge.py) -> save the best-epoch full-graph
  logits (+ optionally the GNN weights) -> free CPU/GPU memory -> next approach.

Approaches whose saved logits are still current (same GNN hyperparameters and embedding
cache file) are skipped, so the DIVERGE and M-ablation stages can share one export.
Then run diverge_for_larger_tags/diverge_from_logits.py and/or ablation_from_logits.py.
"""
import gc
import os
from datetime import datetime

import torch

from config import setup_finetuning_cfg
from dataset.data_utils import load_init_embedding, resolve_init_cache_path
from gnns.gnn_mtrainer import gnn_train_and_report, set_mgnn_cfg, SUPPORTED_GNN_MODELS, DEFAULT_GNN_MODEL
from experiments.diverge import APPROACHES, TITLES
from report.reporter import ReportResults
from train_llm.efficiency import ResourceMonitor
from diverge_for_larger_tags import logits_store as store


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(dataset_name, llm_name='llama_3.2_1B', peft_type='lora', seed=None, supervised=True,
         approaches=None, gnn_model_name=None, save_weights=False, overwrite=False,
         logits_root=store.DEFAULT_LOGITS_ROOT, reporter_index=0):
    """
    Train/export the GNN of every requested approach for one (dataset, llm, gnn, split, seed).

    Args: as experiments/diverge.main, plus
        save_weights: also save each GNN's best-epoch state_dict next to its logits.
        overwrite: retrain even if current logits exist.
        logits_root: root directory of the saved logits (see logits_store).

    Returns:
        directory holding the saved logits.
    """
    approaches = approaches or APPROACHES
    gnn_tag = gnn_model_name or DEFAULT_GNN_MODEL
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    if seed is not None:
        cfg.dataset.seed = seed
    resolved_seed = seed if seed is not None else cfg.dataset.seed
    gnn_cfg = set_mgnn_cfg(dataset_name, supervised=supervised, seed=resolved_seed, llm_name=llm_name,
                           gnn_model_name=gnn_model_name)
    out_dir = store.run_dir(dataset_name, llm_name, peft_type, gnn_tag, supervised, resolved_seed, logits_root)

    print("=" * 80)
    print(f"DIVERGE (large TAG) | stage 1: train GNNs + export logits | dataset={dataset_name} llm={llm_name} "
          f"peft={peft_type} seed={resolved_seed} supervised={supervised} gnn={gnn_tag} approaches={approaches}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    split = store.split_tag(supervised)
    reporter = ReportResults(cfg, save_dir=f"results/train_info/{dataset_name}_diverge_{llm_name}_{gnn_tag}_{split}_seed{resolved_seed}",
                              index_run=reporter_index)

    failed = []
    for name in approaches:
        path = store.logits_path(out_dir, name)
        emb_path = resolve_init_cache_path(cfg, name, seed=resolved_seed)
        fp = store.fingerprint(gnn_cfg, emb_path)
        weights_ok = not save_weights or os.path.exists(store.weights_path(out_dir, name))
        if not overwrite and weights_ok and store.is_current(path, fp):
            print(f"\n[{TITLES[name]}] up-to-date logits found, skipping: {path}")
            continue

        print(f"\n>>> [{TITLES[name]}] loading embeddings + training {gnn_tag}")
        emb = model = logits = None
        try:
            # Cost of this approach's whole stage (load + train), summed into DIVERGE's total run cost
            with ResourceMonitor() as mon:
                emb, emb_path = load_init_embedding(cfg, name, seed=resolved_seed, device='cpu')
                results, model, logits = gnn_train_and_report(
                    dataset_name, emb, title=TITLES[name], does_print_training_process=False, reporter=reporter,
                    supervised=supervised, seed=resolved_seed, llm_name=llm_name, peft_type=peft_type,
                    gnn_model_name=gnn_model_name, return_best_logits=True)

            meta = {
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_name": dataset_name, "llm_name": llm_name, "peft_type": peft_type,
                "seed": int(resolved_seed), "supervised": bool(supervised), "gnn_model_name": gnn_tag,
                "approach": name, "embedding_shape": [int(d) for d in emb.shape],
                "stage_cost": mon.metrics(),
                "fingerprint": fp,
            }
            store.save_logits(path, logits, results, meta)
            print(f"[{TITLES[name]}] logits {tuple(logits.shape)} saved to: {path}")
            if save_weights:
                wpath = store.weights_path(out_dir, name)
                store.save_weights(wpath, model, gnn_cfg, emb.shape[1], logits.shape[1])
                print(f"[{TITLES[name]}] GNN weights saved to: {wpath}")
        except Exception as e:
            print(f"[{TITLES[name]}] FAILED: {e}")
            reporter.report(f"Large-TAG export | {TITLES[name]} FAILED", str(e))
            failed.append(name)
        finally:
            del emb, model, logits
            free_memory()

    if failed:
        raise RuntimeError(f"GNN training/export failed for: {failed}")
    print(f"\n[OK] Logits for {approaches} are in: {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1 of DIVERGE for large TAGs: train one GNN per LoRA-init approach (one embedding "
                    "cache in memory at a time) and save its logits for diverge_from_logits.py / ablation_from_logits.py")
    parser.add_argument('--dataset_name', type=str, default='computer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed of the trained LoRA adapters/embedding caches (also the GNN seed)')
    parser.add_argument('--split', type=int, default=1,
                         help='1 = supervised embeddings/GNN split, 0 = semi-supervised (seed-tagged) split')
    parser.add_argument('--approaches', type=str, default=','.join(APPROACHES),
                         help=f"Comma-separated subset of {APPROACHES} to train")
    parser.add_argument('--gnn_model_name', type=str, default=None, choices=SUPPORTED_GNN_MODELS,
                         help='GNN encoder (default: from the tuned hyperparameter JSON)')
    parser.add_argument('--save_weights', action='store_true', help="Also save each GNN's best-epoch weights")
    parser.add_argument('--overwrite', action='store_true', help='Retrain even if up-to-date logits exist')
    parser.add_argument('--logits_root', type=str, default=store.DEFAULT_LOGITS_ROOT)
    parser.add_argument('--reporter_index', type=int, default=0)
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        llm_name=args.llm_name,
        peft_type=args.peft_type,
        seed=args.seed,
        supervised=args.split == 1,
        approaches=[a.strip() for a in args.approaches.split(',') if a.strip()],
        gnn_model_name=args.gnn_model_name,
        save_weights=args.save_weights,
        overwrite=args.overwrite,
        logits_root=args.logits_root,
        reporter_index=args.reporter_index,
    )
