# STAGE — Simplified Text-Attributed Graph Embeddings

Implementation of the core method from:

> Zolnai-Lucas, Boylan, Hokamp, Ghaffari. *"STAGE: Simplified Text-Attributed Graph
> Embeddings Using Pre-trained LLMs."* arXiv:2407.12860, 2024. (`stage_paper.pdf`)

STAGE's headline claim: you don't need to finetune an LLM or run multi-stage
prompting pipelines (TAPE, GLEM, SimTeG) to get competitive node-classification
accuracy on text-attributed graphs. Instead:

1. Encode each node's raw text (title/abstract) with a **frozen, zero-shot** LLM,
   mean-pooling its final hidden layer into one embedding vector per node.
2. Train an **ensemble of GNN architectures** (GCN, GraphSAGE, RevGAT, MLP) on top
   of those embeddings + the graph's adjacency structure.
3. **Mean-pool the softmax predictions** across the ensemble for the final answer
   (paper Eq. 2).

This folder is a self-contained implementation of that pipeline. It only reads
from the rest of the DIVERGE repo (dataset loaders, `GNNEncoder`, metrics, local
model checkpoints) — nothing outside `STAGE/` is modified.

## Scope of this implementation

Reproducing the paper exactly requires 7B-parameter MTEB-leaderboard embedding
models and 5 benchmark datasets. Given this project's existing local checkpoints
and hardware, this implementation targets the paper's **main result (Table 1)**
with these deliberate substitutions — documented here rather than left implicit:

| Paper | This implementation | Why |
|---|---|---|
| SFR-Embedding-Mistral / gte-Qwen1.5-7B / LLM2Vec-Llama-3-8B as embedding model | Any of this repo's local frozen LMs (`llama_3.2_1B`, `qwen2.5_1.5B`, `deberta_v3_large`, etc. — see `configs/llm_configs/`) | Those checkpoints are already downloaded in `local_models/` and sized for this project's GPU; the *method* (frozen LLM, mean-pooled last hidden layer, zero-shot) is unchanged, just a smaller backbone. Swap `--llm_name` freely. |
| RevGAT (memory-efficient reversible backprop, Li et al. 2022a) | `GNNEncoder(gnn_type="GAT", residual_conn=1, batch_norm=1)` — "GAT + residual + norm" | Reversible backprop only matters for very deep nets on huge graphs; forward computation for the shallow (2-3 layer) networks used on Cora/PubMed is the same. See `STAGE/models.py` docstring. |
| 5 datasets (Cora, PubMed, ogbn-arxiv, ogbn-products, tape-arxiv23) | Cora, PubMed | These two are directly supported by this repo's dataset loader with raw text already available; ogbn-products and tape-arxiv23 are not currently loaded into this repo. `arxiv` *is* available (`common.load_graph_dataset_for_tape("arxiv", ...)`) and can be added by extending `DATASET_HYPERPARAMS` in `run_stage.py`, but was left out of this first pass by request. |
| Instruction-biased embeddings ablation (Table 2), diffusion-pattern GNNs (Simple-GCN/SIGN), PEFT ablation (Table 3) | Not implemented | Out of scope for this pass (core Table-1 result only). `STAGE.embedding.get_or_build_stage_embeddings` already accepts an `instruction=` prefix if you want to extend into the Table-2 ablation later. |

Everything else follows the paper directly: zero-shot frozen-LLM embeddings, no
prompting/data augmentation, ensembling by mean-pooling predicted probabilities
(Eq. 2), and a 60/20/20 train/val/test split re-sampled per seed for Cora/PubMed
(paper Section 4), reported as mean ± std over 4 seeds (paper Table 1).

## Files

- `embedding.py` — `StageTextEmbedder`: loads a frozen local LM, mean-pools its
  last hidden layer per text. `get_or_build_stage_embeddings(...)` caches results
  to `STAGE/cache/` (git-ignored) so repeated runs/seeds don't re-embed.
- `models.py` — GNN ensemble members. GCN/SAGE/RevGAT reuse `common.gnn.GNNEncoder`;
  `MLPNodeClassifier` is new (needed since it ignores `edge_index` entirely).
- `trainer.py` — `StageTrainer`: trains one ensemble member with early stopping on
  validation accuracy (mirrors `gnns/gnn_mtrainer.py::GNNTrainer`, generalized to
  accept any `nn.Module`).
- `ensemble.py` — mean-pools per-class probabilities across ensemble members
  (paper Eq. 2) and scores the result.
- `run_stage.py` — CLI entry point tying it all together and reporting a
  Table-1-style summary.

## Usage

From the repo root:

```bash
python -m STAGE.run_stage --dataset cora   --llm_name llama_3.2_1B --seeds 0 1 2 3
python -m STAGE.run_stage --dataset pubmed --llm_name llama_3.2_1B --seeds 0 1 2 3
```

Other local embedding backbones can be swapped in directly (any name with a
config under `configs/llm_configs/` and weights under `local_models/`):

```bash
python -m STAGE.run_stage --dataset cora --llm_name qwen2.5_1.5B --seeds 0 1 2 3
python -m STAGE.run_stage --dataset cora --llm_name deberta_v3_large --seeds 0 1 2 3
```

Results are printed to stdout and saved as JSON to
`STAGE/results/{dataset}_{llm_name}_{instruction_tag}.json` (per-seed metrics +
mean/std summary for every ensemble member and the ensemble itself).

## Results so far

`llama_3.2_1B` frozen embeddings, 4 seeds (0-3), 60/20/20 split:

| Dataset | MLP | GCN | SAGE | RevGAT | **Ensemble** | Paper Ensemble (SFR-Mistral-7B) |
|---|---|---|---|---|---|---|
| Cora | 79.61 ± 2.49 | 87.78 ± 1.16 | 87.45 ± 1.01 | 85.61 ± 1.18 | **88.28 ± 1.42** | 88.24 ± 1.55 |
| PubMed | 91.39 ± 0.31 | 87.72 ± 0.30 | 91.12 ± 0.30 | 87.23 ± 1.05 | **91.70 ± 0.50** | 92.65 ± 0.68 |

Despite using a much smaller (1B vs 7B), general-purpose (non-MTEB-tuned) frozen
LLM for embeddings, the ensemble result on both datasets lands close to the paper's
numbers (Cora: -0.04pp; PubMed: -0.95pp) — consistent with the paper's own finding
(Table 4) that the choice of embedding backbone matters less than the ensembling
step itself. On PubMed, RevGAT and GCN noticeably underperform MLP and SAGE; the
paper sees the same qualitative pattern on PubMed (Table 1: MLP is PubMed's best
*individual* model there too), so this isn't an artifact of the smaller embedding
model.
