#!/usr/bin/env bash
# =============================================================================
# Show which (dataset, llm, init, seed) combos are done at each pipeline stage in THIS checkout:
#   adapter = trained LoRA adapter   cache = embedding cache   diverge = OK row in diverge history
#
#   bash sh_scripts/list_runs.sh
#   DATASETS="cora" bash sh_scripts/list_runs.sh
# =============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASETS="${DATASETS:-}" POOLING="${POOLING:-mean}" "$PYTHON" - <<'PY'
import csv, glob, os, re
want = set(os.environ["DATASETS"].split())
pool = os.environ["POOLING"]
pat = re.compile(r"^(?P<llm>.+?)_(?P<ds>[^_]+)_seqcls_(?P<peft>[^_]+)_semi_supervised_init_type_(?P<init>.+)_seed(?P<seed>\d+)$")

diverge_ok = set()
h = "results/run_status/diverge_run_history.csv"
if os.path.isfile(h):
    for r in csv.DictReader(open(h, encoding="utf-8")):
        if r.get("status") == "OK":
            diverge_ok.add((r["dataset"], r["llm_name"], r["seed"]))

rows = []
for p in sorted(glob.glob("artifacts/*_seed*")):
    m = pat.match(os.path.basename(p))
    if not m or (want and m["ds"] not in want):
        continue
    trained = os.path.isfile(os.path.join(p, "adapter_model.safetensors"))
    split = acc = "-"
    f = os.path.join(p, "training_results.csv")
    if os.path.isfile(f):
        r = list(csv.DictReader(open(f, encoding="utf-8")))[-1]
        split, acc = r.get("split", "-"), r.get("test_accuracy", "-")
        try: acc = f"{float(acc):.4f}"
        except ValueError: pass
    cache = os.path.isfile(f"artifacts/cache/{m['llm']}_{m['ds']}_seqcls_lora_init-{m['init']}_pool-{pool}_seed{m['seed']}_semi_supervised.pt")
    div = (m["ds"], m["llm"], m["seed"]) in diverge_ok
    rows.append((m["ds"], m["llm"], m["init"], m["seed"], split, acc, "yes" if trained else "NO", "yes" if cache else "-", "yes" if div else "-"))

hdr = ("dataset", "llm", "init", "seed", "split", "test_acc", "adapter", "cache", "diverge")
w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)] if rows else [len(x) for x in hdr]
line = lambda r: "  ".join(str(v).ljust(n) for v, n in zip(r, w))
print(line(hdr)); print("  ".join("-" * n for n in w))
for r in rows: print(line(r))
print(f"\n{len(rows)} adapter dirs")
PY
