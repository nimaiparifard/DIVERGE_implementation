# sh_scripts — running DIVERGE on Colab / RunPod (Linux)

| script | what it does |
|---|---|
| `bootstrap_cloud.sh` | setup_env + download_models + download_datasets (+ restore) in one go |
| `setup_env.sh` | torch 2.7.1+cu128 (same as the Windows box), matching PyG wheels, `requirements.txt`, sanity check |
| `download_models.sh` | `local_models/<name>` from `hf_repo_id` in `configs/llm_configs/<name>.json` (needs `HF_TOKEN` for Llama/Gemma) |
| `download_datasets.sh` | `datasets/<name>.pt` from HF `xxwu/LLMNodeBed` (+ `TAPE=1`, `EXTRAS_TAR=`) |
| `run_all_datasets_lora_init.sh` | = `scripts/run_all_datasets_lora_init.bat` |
| `run_cache_embedding_all_datasets.sh` | = `scripts/run_cache_embedding_all_datasets.bat` |
| `run_diverge.sh` | = `scripts/run_diverge.bat` |
| `run_lora_init_parallel.sh` | parallel training: several runs on one GPU (VRAM-gated queue, OOM retry) |
| `run_cache_embedding_parallel.sh` | parallel embedding caching (same queue, `BATCH_SIZE=` override) |
| `RUNPOD_GUIDE.txt` | step-by-step RunPod / RTX 4090 guide: which script, in which order, tuning `MAX_JOBS` |
| `export_outputs.sh` | pack adapters + caches + results (filterable by dataset/llm/init/seed) into one tar.gz |
| `restore_outputs.sh` | merge a tar.gz / sync folder into this checkout (Linux **or Git Bash on Windows**) |
| `list_runs.sh` | table of which (dataset, llm, init, seed) are trained / cached / DIVERGE-done |
| `pack_code.sh`, `pack_local_data.sh` | run locally: pack unpushed code / the non-HF dataset folders |

All run scripts are configured with env vars (`DATASETS`, `LLM_NAMES`, `INIT_APPROACHES`, `SEEDS`, `SPLIT`, …)
instead of editing the file, **skip combos that are already done** (`SKIP_EXISTING=1`, default), and with
`SYNC_DIR=...` copy every finished run to Drive / the network volume immediately.

## Colab

```python
# cell 1: persistent storage + token
from google.colab import drive, userdata; drive.mount('/content/drive')
import os; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```
```bash
# cell 2: code (either git clone after pushing, or the tarball from pack_code.sh)
!git clone -b master https://github.com/nimaiparifard/DIVERGE.git /content/DIVERGE   # work is on master
# !mkdir -p /content/DIVERGE && tar -xzf /content/drive/MyDrive/diverge_code.tar.gz -C /content/DIVERGE
%cd /content/DIVERGE
!RESTORE_FROM=/content/drive/MyDrive/DIVERGE_sync bash sh_scripts/bootstrap_cloud.sh   # RESTORE_FROM optional
```
```bash
# cell 3: train - every finished adapter lands in Drive right away
!SYNC_DIR=/content/drive/MyDrive/DIVERGE_sync DATASETS=cora LLM_NAMES=gemma2_2B SEEDS=42 \
  bash sh_scripts/run_all_datasets_lora_init.sh
```

## RunPod
Work under `/workspace` (the persistent volume): clone there, `export HF_TOKEN=...`,
`bash sh_scripts/bootstrap_cloud.sh`, run with `SYNC_DIR=/workspace/DIVERGE_sync`.
Use `tmux` so the sweep survives closing the web terminal. After a pod restart: `SKIP_SETUP=1 bash sh_scripts/bootstrap_cloud.sh`.

## Continue a seed on another machine (Colab -> your PC, or the other way)

The seed is not state that has to be carried over: it is the `--seed` argument, and it is written in
the adapter folder name (`..._seed42`) and in `training_results.csv`. Given the same seed + `--split` +
dataset `.pt`, `common/dataloader.py::re_split_data` gives **the same train/val/test nodes on every OS**
(numpy's seeded RNG is platform independent). So you only move the files:

```bash
# on Colab / RunPod
DATASETS=cora SEEDS=42 bash sh_scripts/export_outputs.sh      # -> exports/diverge_outputs_<host>_<time>.tar.gz
# download it (or use SYNC_DIR on Drive and download that folder)

# on Windows, in Git Bash from the repo root
bash sh_scripts/restore_outputs.sh ~/Downloads/diverge_outputs_<host>_<time>.tar.gz
bash sh_scripts/list_runs.sh                                    # check what is there now
```
Then continue with the **same seed and split**, e.g. in `scripts\run_cache_embedding_all_datasets.bat` set
`DATASETS=cora`, `SEEDS=42` (or the `.sh` versions). Already finished combos are skipped by the `.sh` scripts.

Rules that keep it reproducible:
- same `--seed`, same `--split`, same `configs/` (dataset/training_args/peft) on both machines;
- same base model weights (`local_models/<name>` from the same HF repo);
- an adapter trained on one machine can be **used** anywhere (cache/DIVERGE load the saved weights),
  but **re-training** the same seed on a different GPU/library version gives close, not bit-identical,
  numbers (bf16/4-bit kernels differ). Don't retrain a seed you already have - restore it instead.
