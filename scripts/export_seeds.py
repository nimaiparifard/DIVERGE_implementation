"""
Snapshot the seeds of this machine's runs into seed_snapshot/ so another server can reproduce them exactly.

For every dataset x seed x split mode it saves the actual train/val/test node indices that seed produces
(seed_snapshot/splits/<dataset>_split<0|1>_seed<seed>.npz), plus seed_snapshot/manifest.json with the seed
list, each dataset's base seed from configs/dataset/<name>.json, file hashes and the library versions.
common/dataloader.py loads these saved splits whenever they exist, so copying seed_snapshot/ to the server
(then running scripts/import_seeds.py) gives it the same splits as here.

    python scripts/export_seeds.py                                   # all datasets, seeds 42 43 44 45
    python scripts/export_seeds.py --datasets cora pubmed --seeds 42 43 44 45
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.dataloader import compute_split, is_seeded_split  # noqa: E402

DEFAULT_SEEDS = [42, 43, 44, 45]
SNAPSHOT_DIR = os.path.join(REPO_ROOT, "seed_snapshot")


def split_file_name(dataset_name, re_split, seed):
    return f"{dataset_name}_split{int(bool(re_split))}_seed{int(seed)}.npz"


def split_hash(num_nodes, train_idx, val_idx, test_idx):
    h = hashlib.sha256()
    h.update(str(int(num_nodes)).encode())
    for idx in (train_idx, val_idx, test_idx):
        h.update(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
    return h.hexdigest()


def masks_to_indices(masks):
    return [m.cpu().nonzero(as_tuple=True)[0].numpy().astype(np.int64) for m in masks]


def load_num_nodes(dataset_name):
    graph = torch.load(os.path.join(REPO_ROOT, "datasets", f"{dataset_name}.pt"), weights_only=False, map_location="cpu")
    return int(graph.num_nodes)


def config_seed(dataset_name):
    path = os.path.join(REPO_ROOT, "configs", "dataset", f"{dataset_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("seed")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def default_datasets():
    names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(REPO_ROOT, "datasets", "*.pt"))]
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description="Export per-dataset seed splits to seed_snapshot/")
    parser.add_argument("--datasets", nargs="+", default=None, help="Default: every datasets/<name>.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1], choices=[0, 1],
                        help="--split values the runs use (0 = semi-supervised, 1 = 60/20/20 re-split)")
    args = parser.parse_args()

    datasets = args.datasets or default_datasets()
    splits_dir = os.path.join(SNAPSHOT_DIR, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_commit": git_commit(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "seeds": args.seeds,
        "splits": args.splits,
        "datasets": {},
    }

    for dataset_name in datasets:
        num_nodes = load_num_nodes(dataset_name)
        base_seed = config_seed(dataset_name)
        # The config seed is what a run uses when --seed is not passed, so pin it too
        seeds = list(dict.fromkeys(args.seeds + ([base_seed] if base_seed is not None else [])))
        entry = {"config_seed": base_seed, "num_nodes": num_nodes, "files": {}}
        for re_split in args.splits:
            if not is_seeded_split(dataset_name, re_split):
                continue  # dataset's own fixed split, independent of the seed
            for seed in seeds:
                train_idx, val_idx, test_idx = masks_to_indices(
                    compute_split(dataset_name, num_nodes, re_split, "cpu", seed=seed))
                name = split_file_name(dataset_name, re_split, seed)
                np.savez_compressed(os.path.join(splits_dir, name), num_nodes=num_nodes,
                                    train_idx=train_idx.astype(np.int32), val_idx=val_idx.astype(np.int32),
                                    test_idx=test_idx.astype(np.int32))
                entry["files"][name] = {
                    "split": re_split, "seed": seed,
                    "sizes": [len(train_idx), len(val_idx), len(test_idx)],
                    "sha256": split_hash(num_nodes, train_idx, val_idx, test_idx),
                }
        manifest["datasets"][dataset_name] = entry
        print(f"[OK] {dataset_name}: {num_nodes} nodes, config seed {base_seed}, {len(entry['files'])} split file(s)")

    with open(os.path.join(SNAPSHOT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(SNAPSHOT_DIR, "seeds.env"), "w", newline="\n") as f:
        f.write(f'SEEDS="{" ".join(str(s) for s in args.seeds)}"\n')
    print(f"\nSnapshot written to {SNAPSHOT_DIR}")
    print("Copy that folder into the repo on the other server and run: python scripts/import_seeds.py")


if __name__ == "__main__":
    main()
