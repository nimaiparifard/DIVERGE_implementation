"""
API cost simulator for augmentation-based LLM-GNN methods (Reviewer 1, Comment 5).

Estimates what TAPE, UltraTAG and E-LLaGNN would pay to run their LLM augmentation
step through the GPT-4o API on each benchmark, following the same token-based formula
TAPE uses in its Appendix B:

    cost = sum over calls of (input_tokens * input_price + output_tokens * output_price)

Input tokens are counted exactly with the GPT-4o tokenizer (tiktoken o200k_base) on the
real node texts and each method's prompt. Output tokens are measured from generations
already stored in this repo:
  - TAPE: real GPT-4o explanations in datasets/gpt_4o/{dataset}.json (PubMed has no
    GPT-4o file, so the pooled GPT-4o mean over the other datasets is used).
  - UltraTAG: summary + keywords + soft label from ULTRATAG/cache (Cora), re-serialized
    as the JSON object the prompt asks for.
  - E-LLaGNN: the prompt asks for a detailed explanation of the paper, so TAPE's
    measured GPT-4o explanation length is used as a proxy.

All estimates are lower bounds: one call per augmented item, no retries, no failed or
re-run queries, and E-LLaGNN's per-epoch re-augmentation is counted only once.

Sanity checks printed at the end:
  1. TAPE on ogbn-arxiv with GPT-3.5-turbo pricing and TAPE's own token averages
     reproduces the ~128 USD reported in the TAPE paper.
  2. Our input-token count on PubMed is compared with the prompt_tokens that the real
     GPT-3.5 responses in datasets/PubMed/*.json report.

Usage (from repo root):
    python -m experiments.api_simulator_calculator
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import tiktoken
import torch

from common.prompt import EXPLANATION_PROMPTS
from ULTRATAG.text_augmentation import build_augmentation_prompt, propagate_texts
from ULTRATAG.structure_augmentation import build_edge_confidence_prompt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["cora", "citeseer", "pubmed", "wikics", "arxiv"]
DISPLAY = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "wikics": "WikiCS", "arxiv": "ArXiv"}

# USD per 1M tokens, OpenAI pricing page (checked 2026-09-28)
PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-batch": {"input": 1.25, "output": 5.00},
    "gpt-3.5-turbo-tape": {"input": 1.50, "output": 2.00},  # price used in TAPE Appendix B
}

# Tokens added by the chat format per request (role tags / message framing)
CHAT_OVERHEAD = 7

# UltraTAG (paper Sec. 5.4, and ULTRATAG/pipeline.py): top-10% of training nodes by PageRank
ULTRATAG_IMPORTANT_RATIO = 0.1
ULTRATAG_EDGE_OUTPUT = '{"confidence": 0.85}'

# E-LLaGNN (Jaiswal et al., 2024): best budgets reported in their Tables 5 and 6
ELLAGNN_FRACTION = 0.25          # 25% of nodes, best setting on Cora and PubMed
ELLAGNN_ARXIV_NODES = 10_000     # best setting on ogbn-arxiv
ELLAGNN_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. The assistant "
    "gives helpful, detailed, and polite answers to the user's questions.\n"
    "Provide a detailed explanation for the given title and abstract of a paper in less than "
    "1000 words identifying the key technical concepts and simplifying them for ease of "
    "understanding.\n[ Start of Title ]\n{title}\n[ End of Title ].\n"
    "[ Start of Abstract ]\n{abstract}\n[ End of Abstract ]."
)


def load_dataset(name):
    data = torch.load(os.path.join(REPO_ROOT, "datasets", f"{name}.pt"), weights_only=False)
    return data


def split_title_abstract(text):
    """Best-effort title/abstract split; the prompt length is the same either way."""
    for sep in ["\nAbstract:", " : ", ". "]:
        if sep in text:
            title, abstract = text.split(sep, 1)
            return title.replace("Title:", "").strip(), abstract.strip()
    return "", text.strip()


def count(enc, texts):
    return np.array([len(enc.encode(t, disallowed_special=())) + CHAT_OVERHEAD for t in texts])


def measured_tape_outputs(enc):
    """Mean GPT-4o output tokens per dataset from the stored TAPE explanations."""
    means = {}
    for name in DATASETS:
        path = os.path.join(REPO_ROOT, "datasets", "gpt_4o", f"{name}.json")
        if not os.path.exists(path):
            continue
        lengths = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                lengths.append(len(enc.encode(json.loads(line)["answer"], disallowed_special=())))
        means[name] = (float(np.mean(lengths)), len(lengths))
    pooled = float(np.mean([m for m, _ in means.values()]))
    return means, pooled


def measured_ultratag_output(enc):
    """Mean output tokens of UltraTAG's JSON augmentation (summary, keywords, soft label)."""
    path = os.path.join(REPO_ROOT, "ULTRATAG", "cache", "cora_ratio0_augseed42_augmented_graph.pt")
    cache = torch.load(path, weights_only=False)
    lengths = []
    for text, label in zip(cache["augmented_texts"], cache["soft_labels"]):
        summary = re.search(r"Summary: (.*?)(?: Keywords: | Category: |$)", text)
        keywords = re.search(r"Keywords: (.*?)(?: Category: |$)", text)
        obj = {
            "summary": summary.group(1) if summary else "",
            "keywords": keywords.group(1).split(", ") if keywords else [],
            "soft_label": label,
        }
        lengths.append(len(enc.encode(json.dumps(obj), disallowed_special=())))
    return float(np.mean(lengths))


def pagerank(edge_index, num_nodes, damping=0.85, iters=100, tol=1e-10):
    """Power-iteration PageRank (same definition as networkx.pagerank, fast on ogbn-arxiv)."""
    import scipy.sparse as sp

    src, dst = edge_index
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float64)
    weights = 1.0 / out_deg[src]
    transition = sp.csr_matrix((weights, (dst, src)), shape=(num_nodes, num_nodes))
    dangling = out_deg == 0
    rank = np.full(num_nodes, 1.0 / num_nodes)
    for _ in range(iters):
        new = damping * (transition @ rank + rank[dangling].sum() / num_nodes) + (1 - damping) / num_nodes
        if np.abs(new - rank).sum() < tol:
            return new
        rank = new
    return rank


def price(input_tokens, output_tokens, model):
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1e6


def simulate(name, enc, tape_out, tape_pooled, ultratag_out):
    data = load_dataset(name)
    texts = [t or "" for t in data.raw_texts]
    n = len(texts)
    edge_index = data.edge_index.cpu().numpy()
    labels = [str(l) for l in data.label_name]
    rows = []

    # ---------------- TAPE: one explanation query per node ----------------
    prompt = EXPLANATION_PROMPTS.get(name) or EXPLANATION_PROMPTS[f"{name}.json"]
    tin = count(enc, [f"{t}\n\n{prompt}" for t in texts]).sum()
    out_mean = tape_out.get(name, (tape_pooled, 0))[0]
    rows.append(dict(method="TAPE", dataset=name, calls=n, input_tokens=int(tin),
                     output_tokens=int(round(out_mean * n))))

    # ---------------- UltraTAG: text augmentation + edge reconfiguration ----------------
    und = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    und = np.unique(und[:, und[0] != und[1]], axis=1)
    propagated = propagate_texts(texts, torch.from_numpy(und), n)
    aug_prompts = [build_augmentation_prompt(t, name, labels) for t in propagated]
    uin = count(enc, aug_prompts).sum()
    uout = ultratag_out * n

    num_train = int(n * 0.6)
    k = max(1, int(round(ULTRATAG_IMPORTANT_RATIO * num_train)))
    important = np.zeros(n, dtype=bool)
    important[np.argsort(-pagerank(und, n))[:k]] = True
    pairs = und[:, important[und[0]] & important[und[1]] & (und[0] < und[1])]
    edge_prompts = [build_edge_confidence_prompt(texts[s], texts[d], name) for s, d in pairs.T]
    ein = count(enc, edge_prompts).sum() if edge_prompts else 0
    eout = len(edge_prompts) * len(enc.encode(ULTRATAG_EDGE_OUTPUT))
    rows.append(dict(method="UltraTAG", dataset=name, calls=n + len(edge_prompts),
                     input_tokens=int(uin + ein), output_tokens=int(round(uout + eout)),
                     edge_calls=len(edge_prompts)))

    # ---------------- E-LLaGNN: budgeted on-demand enhancement ----------------
    budget = ELLAGNN_ARXIV_NODES if name == "arxiv" else int(round(ELLAGNN_FRACTION * n))
    # text-length heuristic from the paper: augment the nodes with the shortest texts
    selected = np.argsort([len(t.split()) for t in texts], kind="stable")[:budget]
    e_prompts = []
    for i in selected:
        title, abstract = split_title_abstract(texts[i])
        e_prompts.append(ELLAGNN_PROMPT.format(title=title, abstract=abstract))
    lin = count(enc, e_prompts).sum()
    rows.append(dict(method="E-LLaGNN", dataset=name, calls=budget, input_tokens=int(lin),
                     output_tokens=int(round(out_mean * budget))))

    # ---------------- DIVERGE: no API calls ----------------
    rows.append(dict(method="DIVERGE", dataset=name, calls=0, input_tokens=0, output_tokens=0))

    for r in rows:
        r["cost_gpt4o"] = price(r["input_tokens"], r["output_tokens"], "gpt-4o")
        r["cost_gpt4o_batch"] = price(r["input_tokens"], r["output_tokens"], "gpt-4o-batch")
    return rows


def sanity_checks(enc_35):
    print("\nSanity check 1: TAPE Appendix B (ogbn-arxiv, GPT-3.5-turbo, 285 in / 164 out tokens)")
    cost = price(285 * 169_343, 164 * 169_343, "gpt-3.5-turbo-tape")
    print(f"  reproduced cost = {cost:.2f} USD (TAPE paper reports ~128 USD)")

    print("Sanity check 2: our PubMed input-token count vs. real GPT-3.5 prompt_tokens")
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "datasets", "PubMed", "*.json")))
    real = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            real.append(json.load(f)["usage"]["prompt_tokens"])
    data = load_dataset("pubmed")
    ours = count(enc_35, [f"{t}\n\n{EXPLANATION_PROMPTS['pubmed']}" for t in data.raw_texts])
    print(f"  real mean prompt_tokens = {np.mean(real):.1f} over {len(real)} responses; "
          f"our estimate = {ours.mean():.1f} (cl100k tokenizer, same as GPT-3.5)")
    return cost, float(np.mean(real)), float(ours.mean())


def latex_table(rows):
    methods = ["TAPE", "UltraTAG", "E-LLaGNN", "DIVERGE"]
    lines = []
    for m in methods:
        cells, total = [], 0.0
        for d in DATASETS:
            r = next(x for x in rows if x["method"] == m and x["dataset"] == d)
            total += r["cost_gpt4o"]
            cells.append("0" if r["calls"] == 0 else f"{r['cost_gpt4o']:.2f}")
        name = "\\textbf{DIVERGE (ours)}" if m == "DIVERGE" else m
        tot = "\\textbf{0}" if m == "DIVERGE" else f"{total:.2f}"
        if m == "DIVERGE":
            cells = ["\\textbf{0}"] * len(cells)
        lines.append(f"{name} & " + " & ".join(cells) + f" & {tot} \\\\")
    calls = []
    for m in methods[:3]:
        cs = [next(x for x in rows if x["method"] == m and x["dataset"] == d)["calls"] for d in DATASETS]
        calls.append(f"% {m} API calls: " + ", ".join(f"{DISPLAY[d]}={c:,}" for d, c in zip(DATASETS, cs)))
    return "\n".join(calls + lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "api_cost"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("o200k_base")      # GPT-4o tokenizer
    enc_35 = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 tokenizer, for sanity check 2

    tape_out, tape_pooled = measured_tape_outputs(enc)
    ultratag_out = measured_ultratag_output(enc)
    print("Measured mean output tokens (GPT-4o tokenizer):")
    for d, (m, c) in tape_out.items():
        print(f"  TAPE {d:9s}: {m:7.1f} over {c:,} stored GPT-4o explanations")
    print(f"  TAPE pooled   : {tape_pooled:7.1f} (used for PubMed)")
    print(f"  UltraTAG JSON : {ultratag_out:7.1f} (Cora cache)")

    rows = []
    for name in DATASETS:
        print(f"Simulating {name} ...")
        rows.extend(simulate(name, enc, tape_out, tape_pooled, ultratag_out))

    print(f"\n{'Method':10s} {'Dataset':9s} {'Calls':>9s} {'In Mtok':>9s} {'Out Mtok':>9s} {'GPT-4o $':>10s} {'Batch $':>9s}")
    for r in rows:
        print(f"{r['method']:10s} {r['dataset']:9s} {r['calls']:9,d} {r['input_tokens']/1e6:9.2f} "
              f"{r['output_tokens']/1e6:9.2f} {r['cost_gpt4o']:10.2f} {r['cost_gpt4o_batch']:9.2f}")

    check = sanity_checks(enc_35)
    table = latex_table(rows)
    print("\nLaTeX rows:\n" + table)

    summary = {
        "prices_usd_per_1M_tokens": PRICES,
        "measured_output_tokens": {"tape_gpt4o": {k: v[0] for k, v in tape_out.items()},
                                   "tape_gpt4o_pooled": tape_pooled, "ultratag_json": ultratag_out},
        "sanity_checks": {"tape_arxiv_gpt35_reproduced_usd": check[0],
                          "pubmed_real_prompt_tokens": check[1], "pubmed_estimated_prompt_tokens": check[2]},
        "rows": rows,
    }
    with open(os.path.join(args.out_dir, "api_cost.json"), "w") as f:
        json.dump(summary, f, indent=2, default=int)
    with open(os.path.join(args.out_dir, "api_cost_table_rows.tex"), "w") as f:
        f.write(table + "\n")
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
