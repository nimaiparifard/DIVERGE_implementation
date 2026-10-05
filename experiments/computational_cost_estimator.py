"""
Computational cost of DIVERGE (reviewer R1-C4, R2-C4, R4-C6), PiSSA only, for the three
sub-2B backbones (llama_3.2_1B, smollm2_1.7B, gemma2_2B).

Four stages per LoRA run: (1) LLM fine-tuning, (2) embedding caching, (3) GNN training,
plus (4) the ensemble head, trained once on the logits of the M=5 GNNs.

What is MEASURED and what is ESTIMATED (estimated cells get a dagger in the LaTeX rows):
  - Fine-tuning wall-clock / peak VRAM / adapter size / latency: measured, read from
    results/efficiency/lora_finetune_efficiency.csv (Cora, CiteSeer, PubMed, seeds 42-45).
    WikiCS and ArXiv are not measured: per LLM we fit (least squares)
        T_ft = t0 + c * (train_tokens + eval_tokens / 3)   (eval = forward only, ~1/3 of a train step)
    where t0 is the fixed per-run overhead (model load, final test pass),
    on the three measured datasets, where token counts use the real node texts tokenized
    with the model's own tokenizer (clipped at max_length) and the epochs / eval_steps of
    configs/training_args/<dataset>.json.
  - Caching: N_nodes * mean_tokens * (per-token latency measured at batch 8 during fine-tuning).
  - GNN / ensemble: measured on Cora (results/diverge/diverge_no_edge_editing_results.csv).
    Other datasets: Cora's per-epoch overhead + a FLOP term at --eff-tflops. Run with
    --benchmark (GPU must be free) to replace this with timings on the real graphs
    using random features of the LLM's hidden size.
  - Parallel: k = jobs that fit in the GPU (measured peak VRAM + CUDA context), aggregate
    throughput g(k) = 1 + eta*(k-1), eta from the micro-batch probe in
    outputs_eff/parallel_plan.json (throughput(bs=2)/throughput(bs=1) - 1). This is a
    projection; confirm with one real sh_scripts/run_lora_init_parallel.sh run.

Usage:
    python -m experiments.computational_cost_estimator
    python -m experiments.computational_cost_estimator --benchmark        # GPU must be idle
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

LLMS = ["llama_3.2_1B", "smollm2_1.7B", "gemma2_2B"]
LLM_LABEL = {"llama_3.2_1B": "LLaMA-3.2-1B", "smollm2_1.7B": "SmolLM2-1.7B", "gemma2_2B": "Gemma-2-2B"}
DATASETS = ["cora", "citeseer", "pubmed", "wikics", "arxiv"]
DS_LABEL = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "wikics": "WikiCS", "arxiv": "ArXiv"}
N_INITS = 5
SPLIT = (0.6, 0.2, 0.2)
# GNN settings of Table tab:hyperparams (layers, hidden)
GNN_CFG = {"cora": (3, 256), "citeseer": (2, 256), "pubmed": (3, 256), "arxiv": (2, 128), "wikics": (3, 128)}
ENS_GRID = 36          # ensemble configurations in the grid search
ENS_EPOCHS = 200
CUDA_CONTEXT_GB = 0.6  # per extra process
SAFETY = 0.90          # fraction of VRAM planned against

FT_CSV = "results/efficiency/lora_finetune_efficiency.csv"
DIVERGE_CSV = "results/diverge/diverge_no_edge_editing_results.csv"
PLAN_JSON = "outputs_eff/parallel_plan.json"
OUT_DIR = "results/computational_cost"


# ---------------------------------------------------------------- dataset statistics
def dataset_stats(llm, ds, cache):
    key = f"{llm}/{ds}"
    if key in cache:
        return cache[key]
    from transformers import AutoTokenizer
    data = torch.load(os.path.join("datasets", f"{ds}.pt"), weights_only=False, map_location="cpu")
    max_len = json.load(open(f"configs/dataset/{ds}.json")).get("max_length", 512)
    tok = AutoTokenizer.from_pretrained(os.path.join("local_models", llm), local_files_only=True)
    lens = []
    texts = list(data.raw_texts)
    for i in range(0, len(texts), 4096):
        enc = tok(texts[i:i + 4096], truncation=True, max_length=max_len, add_special_tokens=True)
        lens.extend(len(x) for x in enc["input_ids"])
    ei = data.edge_index
    cache[key] = {"nodes": int(len(texts)), "edges": int(ei.size(1)),
                  "classes": int(data.y.max().item()) + 1, "mean_tokens": float(np.mean(lens))}
    print(f"  stats {key}: {cache[key]}")
    return cache[key]


def token_budget(ds, st):
    t = json.load(open(f"configs/training_args/{ds}.json"))
    n_train, n_val = SPLIT[0] * st["nodes"], SPLIT[1] * st["nodes"]
    steps = t["num_train_epochs"] * n_train / (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"])
    n_evals = math.floor(steps / t["eval_steps"]) + 1   # periodic evals + final test pass
    return (t["num_train_epochs"] * n_train * st["mean_tokens"], n_evals * n_val * st["mean_tokens"])


# ---------------------------------------------------------------- GNN / ensemble
def analytic_gnn_ens(ds, st, hidden_in, cora_ref, eff_tflops):
    """Cora's measured per-epoch time is treated as fixed overhead; a FLOP term is added."""
    layers, h = GNN_CFG[ds]
    dims = [hidden_in] + [h] * (layers - 1) + [st["classes"]]
    # SAGEConv: two dense maps per layer + neighbour aggregation; x3 for forward+backward
    per_epoch = sum(3 * (4 * st["nodes"] * a * b + 2 * st["edges"] * a) for a, b in zip(dims[:-1], dims[1:]))
    gnn_sec = cora_ref["gnn_epochs"] * (cora_ref["gnn_sec_per_epoch"] + per_epoch / (eff_tflops * 1e12))
    feat_gb = st["nodes"] * hidden_in * 4 / 1e9
    act_gb = 3 * st["nodes"] * h * layers * 4 / 1e9
    gnn_gb = max(cora_ref["gnn_vram_gb"], feat_gb + act_gb + 0.15)
    ens_in, ens_h = N_INITS * st["classes"], 256
    ens_epoch = 3 * 2 * SPLIT[0] * st["nodes"] * (ens_in * ens_h + ens_h * st["classes"])
    ens_sec = ENS_GRID * ENS_EPOCHS * (cora_ref["ens_sec_per_epoch"] + ens_epoch / (eff_tflops * 1e12))
    return gnn_sec, gnn_gb, ens_sec


def benchmark_gnn_ens(ds, st, hidden_in, gnn_epochs):
    from torch_geometric.nn import SAGEConv
    dev = "cuda"
    data = torch.load(os.path.join("datasets", f"{ds}.pt"), weights_only=False)
    ei = data.edge_index.to(dev)
    x = torch.randn(st["nodes"], hidden_in, device=dev)
    y = torch.randint(0, st["classes"], (st["nodes"],), device=dev)
    layers, h = GNN_CFG[ds]
    dims = [hidden_in] + [h] * (layers - 1) + [st["classes"]]
    convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:])).to(dev)
    opt = torch.optim.Adam(convs.parameters(), lr=1e-3)

    def fwd():
        z = x
        for i, c in enumerate(convs):
            z = c(z, ei)
            if i < len(convs) - 1:
                z = torch.relu(z)
        return z

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(gnn_epochs):
        opt.zero_grad(); torch.nn.functional.cross_entropy(fwd(), y).backward(); opt.step()
        with torch.no_grad():
            fwd()                                   # per-epoch validation pass
    torch.cuda.synchronize()
    gnn_sec, gnn_gb = time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1e9
    del x, convs, opt; torch.cuda.empty_cache()

    n_tr = int(SPLIT[0] * st["nodes"])
    logits = torch.randn(n_tr, N_INITS * st["classes"], device=dev)
    yt = y[:n_tr]
    mlp = torch.nn.Sequential(torch.nn.Linear(logits.size(1), 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, st["classes"])).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(ENS_EPOCHS):
        opt.zero_grad(); torch.nn.functional.cross_entropy(mlp(logits), yt).backward(); opt.step()
    torch.cuda.synchronize()
    ens_sec = ENS_GRID * (time.perf_counter() - t0)
    return gnn_sec, gnn_gb, ens_sec


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="store_true", help="time GNN + ensemble on the GPU (must be idle)")
    ap.add_argument("--eff-tflops", type=float, default=5.0, help="effective FP32 TFLOP/s for the analytic GNN/ensemble term")
    ap.add_argument("--vram-gb", type=float, nargs="+", default=[12.0, 24.0], help="GPU memory budgets for the parallel table")
    ap.add_argument("--eta", type=float, default=None, help="override marginal throughput gain per extra concurrent job")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    stats_path = os.path.join(OUT_DIR, "dataset_token_stats.json")
    stats_cache = json.load(open(stats_path)) if os.path.isfile(stats_path) else {}

    ft = pd.read_csv(FT_CSV)
    ft = ft[(ft.lora_init == "pissa") & ft.model_name.isin(LLMS) & ft.adapter_dir.str.contains("_seed")]

    dv = pd.read_csv(DIVERGE_CSV)
    dv = dv[(dv.dataset_name == "cora") & (dv.llm_name == "llama_3.2_1B")]
    ens = dv[dv.ensemble_weight_approach == "learnable_classes"]
    runs = dv.drop_duplicates("run_id")
    cora_ref = {
        "gnn_epochs": runs.pissa_gnn_epochs_run.mean(),
        "gnn_sec_per_epoch": (runs.pissa_gnn_train_time_sec / runs.pissa_gnn_epochs_run).mean(),
        "gnn_vram_gb": runs.pissa_gnn_peak_vram_mb.mean() / 1024,
        "ens_sec_per_epoch": ens.ensemble_tuning_time_sec.mean() / (ENS_GRID * ENS_EPOCHS),
    }
    print("Cora reference (measured):", {k: round(v, 4) for k, v in cora_ref.items()})

    eta = args.eta
    if eta is None:
        probes = {p["micro_bs"]: p["tokens_per_sec"] for p in json.load(open(PLAN_JSON))["probes"] if not p["spilling"]}
        eta = probes[2] / probes[1] - 1
    print(f"eta (marginal throughput gain per extra concurrent job) = {eta:.3f}")

    rows = []
    for llm in LLMS:
        f = ft[ft.model_name == llm]
        hidden = json.load(open(os.path.join("local_models", llm, "config.json")))["hidden_size"]
        st = {ds: dataset_stats(llm, ds, stats_cache) for ds in DATASETS}
        json.dump(stats_cache, open(stats_path, "w"), indent=1)

        measured = {ds: f[f.dataset_name == ds] for ds in DATASETS if (f.dataset_name == ds).any()}
        X = np.array([token_budget(ds, st[ds]) for ds in measured])
        yv = np.array([g.wall_clock_train_sec.mean() for g in measured.values()])
        w = np.array([1.0, 1.0 / 3])
        xs = X @ w
        (t0, c), *_ = np.linalg.lstsq(np.c_[np.ones_like(xs), xs], yv, rcond=None)
        if t0 < 0:
            t0, c = 0.0, float(xs @ yv / (xs @ xs))
        fit_err = np.abs(t0 + c * xs - yv) / yv
        print(f"{llm}: T_ft = {t0:.0f}s + {c*1e3:.4f} ms per train-token-equivalent "
              f"(fit error on {', '.join(measured)}: {', '.join(f'{e:.1%}' for e in fit_err)})")
        tok_lat = np.mean([g.inference_ms_per_sample.mean() / st[ds]["mean_tokens"] for ds, g in measured.items()])

        for ds in DATASETS:
            s = st[ds]
            if ds in measured:
                g = measured[ds]
                ft_sec, ft_est = g.wall_clock_train_sec.mean(), False
                lat_ms = g.inference_ms_per_sample.mean()
            else:
                ft_sec, ft_est = t0 + c * float(np.dot(token_budget(ds, s), w)), True
                lat_ms = tok_lat * s["mean_tokens"]
            if args.benchmark:
                gnn_sec, gnn_gb, ens_sec = benchmark_gnn_ens(ds, s, hidden, int(round(cora_ref["gnn_epochs"])))
                gnn_est = False
            else:
                gnn_sec, gnn_gb, ens_sec = analytic_gnn_ens(ds, s, hidden, cora_ref, args.eff_tflops)
                gnn_est = ds != "cora"
            rows.append({
                "llm": llm, "dataset": ds, "nodes": s["nodes"], "mean_tokens": round(s["mean_tokens"], 1),
                "ft_min": ft_sec / 60, "ft_estimated": ft_est,
                "ft_peak_vram_gb": f.peak_vram_gb.max(), "adapter_mb": f.adapter_weights_mb.mean(),
                "trainable_params_m": f.trainable_params.mean() / 1e6, "total_params_b": f.total_params.mean() / 1e9,
                "latency_ms_per_node": lat_ms, "cache_min": s["nodes"] * lat_ms / 1000 / 60, "cache_estimated": ft_est,
                "gnn_sec": gnn_sec, "gnn_peak_vram_gb": gnn_gb, "gnn_estimated": gnn_est,
                "ens_sec": ens_sec, "ens_estimated": gnn_est,
            })

    df = pd.DataFrame(rows)
    df["one_run_gpu_h"] = (df.ft_min + df.cache_min + df.gnn_sec / 60) / 60
    df["diverge_seq_gpu_h"] = N_INITS * df.one_run_gpu_h + df.ens_sec / 3600
    df.to_csv(os.path.join(OUT_DIR, "cost_breakdown_pissa.csv"), index=False)

    # -------- sequential vs parallel, summed over the five datasets
    par = []
    for llm in LLMS:
        d = df[df.llm == llm]
        per_job_gb = d.ft_peak_vram_gb.iloc[0] + CUDA_CONTEXT_GB
        stage = N_INITS * (d.ft_min + d.cache_min + d.gnn_sec / 60).sum() / 60
        ens_h = d.ens_sec.sum() / 3600
        row = {"llm": llm, "per_job_vram_gb": per_job_gb, "single_run_h": stage / N_INITS + ens_h,
               "sequential_h": stage + ens_h}
        for v in args.vram_gb:
            k = int(min(N_INITS, max(1, (v * SAFETY) // per_job_gb)))
            gk = 1 + eta * (k - 1)
            # pool keeps k jobs running; the last partial wave runs at its own (smaller) concurrency
            full, rest = divmod(N_INITS, k)
            waves = full * k / gk + (rest / (1 + eta * (rest - 1)) if rest else 0)
            wall = stage / N_INITS * waves + ens_h
            row.update({f"k_{v:g}gb": k, f"parallel_h_{v:g}gb": wall, f"speedup_{v:g}gb": row["sequential_h"] / wall})
        par.append(row)
    pdf = pd.DataFrame(par)
    pdf.to_csv(os.path.join(OUT_DIR, "sequential_vs_parallel.csv"), index=False)

    write_latex(df, pdf, args.vram_gb)
    pd.set_option("display.width", 250)
    print(df[["llm", "dataset", "ft_min", "ft_estimated", "cache_min", "gnn_sec", "ens_sec", "one_run_gpu_h", "diverge_seq_gpu_h"]].round(2).to_string())
    print(pdf.round(2).to_string())
    print(f"[OK] wrote {OUT_DIR}/")


def write_latex(df, pdf, budgets):
    dag = lambda v, est, fmt: (fmt.format(v) + (r"$^\dagger$" if est else ""))
    lines = ["% Generated by experiments/computational_cost_estimator.py"]
    for llm in LLMS:
        d = df[df.llm == llm]
        r0 = d.iloc[0]
        for i, (_, r) in enumerate(d.iterrows()):
            head = (rf"\multirow{{5}}{{*}}{{\shortstack[l]{{{LLM_LABEL[llm]}\\ {r0.ft_peak_vram_gb:.1f} GB / "
                    rf"{r0.adapter_mb:.1f} MB}}}}") if i == 0 else ""
            lines.append(" & ".join([
                head, DS_LABEL[r.dataset],
                dag(r.ft_min, r.ft_estimated, "{:.1f}"),
                dag(r.cache_min, r.cache_estimated, "{:.1f}"),
                dag(r.gnn_sec, r.gnn_estimated, "{:.1f}"),
                dag(r.ens_sec, r.ens_estimated, "{:.0f}"),
                f"{r.one_run_gpu_h:.2f}", f"{r.diverge_seq_gpu_h:.2f}"]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = "% ---- parallel table rows"
    for _, r in pdf.iterrows():
        cells = [LLM_LABEL[r.llm], f"{r.per_job_vram_gb:.1f}", f"{r.single_run_h:.1f}", f"{r.sequential_h:.1f}"]
        for v in budgets:
            cells += [f"{int(r[f'k_{v:g}gb'])}", f"{r[f'parallel_h_{v:g}gb']:.1f}", rf"{r[f'speedup_{v:g}gb']:.2f}$\times$"]
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(OUT_DIR, "cost_table_rows.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
