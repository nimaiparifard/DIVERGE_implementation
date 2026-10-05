"""
Apply a seed snapshot made by scripts/export_seeds.py on another machine, so this server runs with the
same seeds and the same train/val/test splits.

    python scripts/import_seeds.py                                  # snapshot already at <repo>/seed_snapshot
    python scripts/import_seeds.py --snapshot /workspace/seed_snapshot   # copy it in from elsewhere first

What it does:
  1. copies the snapshot into <repo>/seed_snapshot (common/dataloader.py loads splits from there)
  2. checks every split file against the hashes in manifest.json and the local dataset's node count
  3. sets "seed" in configs/dataset/<name>.json back to the exporting machine's value (skip: --no_config)
  4. reports whether this server would have generated the same split on its own (informational only;
     the saved split is used either way)
Then run the usual scripts; sh_scripts/* pick up SEEDS from seed_snapshot/seeds.env.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.dataloader import compute_split  # noqa: E402
from scripts.export_seeds import SNAPSHOT_DIR, masks_to_indices, split_hash  # noqa: E402


def copy_snapshot(src):
    src, dst = os.path.abspath(src), os.path.abspath(SNAPSHOT_DIR)
    if src == dst:
        return
    if not os.path.exists(os.path.join(src, "manifest.json")):
        sys.exit(f"[ERROR] {src} has no manifest.json - is it a folder made by scripts/export_seeds.py?")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[OK] copied snapshot {src} -> {dst}")


def restore_config_seed(dataset_name, seed):
    path = os.path.join(REPO_ROOT, "configs", "dataset", f"{dataset_name}.json")
    if seed is None or not os.path.exists(path):
        return
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("seed") == seed:
        return
    print(f"  config seed {cfg.get('seed')} -> {seed}  ({os.path.relpath(path, REPO_ROOT)})")
    cfg["seed"] = seed
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)


def local_num_nodes(dataset_name):
    path = os.path.join(REPO_ROOT, "datasets", f"{dataset_name}.pt")
    if not os.path.exists(path):
        return None
    return int(torch.load(path, weights_only=False, map_location="cpu").num_nodes)


def main():
    parser = argparse.ArgumentParser(description="Apply a seed snapshot from another machine")
    parser.add_argument("--snapshot", default=SNAPSHOT_DIR, help="Folder made by export_seeds.py")
    parser.add_argument("--no_config", action="store_true", help="Do not touch configs/dataset/*.json")
    args = parser.parse_args()

    copy_snapshot(args.snapshot)
    with open(os.path.join(SNAPSHOT_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    print(f"Snapshot from host '{manifest.get('host')}' ({manifest.get('created')}), commit {manifest.get('git_commit')}")
    print(f"  exported with {manifest['versions']}")
    print(f"  this server:  {{'numpy': '{np.__version__}', 'torch': '{torch.__version__}'}}")
    print(f"Seeds: {manifest['seeds']}\n")

    errors, differ = 0, 0
    for dataset_name, entry in manifest["datasets"].items():
        print(f"[{dataset_name}]")
        if not args.no_config:
            restore_config_seed(dataset_name, entry.get("config_seed"))
        num_nodes = local_num_nodes(dataset_name)
        if num_nodes is None:
            print("  datasets/%s.pt not on this server yet - splits are kept, check again after downloading" % dataset_name)
        elif num_nodes != entry["num_nodes"]:
            print(f"  [ERROR] local dataset has {num_nodes} nodes, snapshot has {entry['num_nodes']} - different dataset file")
            errors += 1
            continue

        for name, info in entry["files"].items():
            path = os.path.join(SNAPSHOT_DIR, "splits", name)
            if not os.path.exists(path):
                print(f"  [ERROR] missing {name}")
                errors += 1
                continue
            saved = np.load(path)
            idx = [saved[k].astype(np.int64) for k in ("train_idx", "val_idx", "test_idx")]
            if split_hash(entry["num_nodes"], *idx) != info["sha256"]:
                print(f"  [ERROR] {name} is corrupted (hash mismatch)")
                errors += 1
                continue
            status = "saved"
            if num_nodes is not None:
                local = masks_to_indices(compute_split(dataset_name, num_nodes, info["split"], "cpu", seed=info["seed"]))
                same = split_hash(num_nodes, *local) == info["sha256"]
                differ += not same
                status = "identical to local RNG" if same else "local RNG differs -> saved split will be used"
            print(f"  [OK] split={info['split']} seed={info['seed']}  {status}")

    with open(os.path.join(SNAPSHOT_DIR, "seeds.env"), "w", newline="\n") as f:
        f.write(f'SEEDS="{" ".join(str(s) for s in manifest["seeds"])}"\n')

    print()
    if errors:
        sys.exit(f"[FAILED] {errors} problem(s) above - fix them before running experiments")
    print(f"[DONE] seeds {manifest['seeds']} applied; splits in {os.path.relpath(SNAPSHOT_DIR, REPO_ROOT)}/splits "
          f"are used automatically by common/dataloader.py")
    if differ:
        print(f"       ({differ} split(s) would have come out differently here without the snapshot)")


if __name__ == "__main__":
    main()
