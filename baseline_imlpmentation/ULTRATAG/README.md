# UltraTAG-S — Robust LLM-enhanced Text-Attributed Graph Learning

Implementation of the sparse-scenario instantiation from:

> Zhang, Li, Li, Li, Zhou, Wang. *"Toward General and Robust LLM-enhanced
> Text-attributed Graph Learning."* ICMR '26, 2026. (`ultratag.pdf`)

UltraTAG-S targets real-world TAG sparsity (missing node texts, missing edges) via
three modules: **(1) LLM-based Robustness Enhancement** (text propagation + LLM text
augmentation + LLM/PageRank-guided structure augmentation), **(2) LM-based Resilient
Representation Learning** (LoRA-finetune an LM classifier on the augmented texts),
and **(3) a Graph-Enhanced Robust Classifier** (a jointly-trained dual-GNN where a
learned similarity matrix refines the adjacency, except where an LLM already made a
structural judgment). This folder implements all three, reading from the rest of
DIVERGE (dataset loaders, `GNNEncoder`, metrics, `STAGE/`'s frozen-LM embedder) but
modifying nothing outside `ULTRATAG/`.

## Scope and deviations from the paper

This is a substantially heavier pipeline than `STAGE/` (real LLM *generation*, not
just embedding), so several substitutions were necessary for local reproducibility.
Each is documented here and, where it affects a specific equation, in the relevant
module's docstring.

| Paper | This implementation | Why |
|---|---|---|
| Meta-Llama-3-8B-Instruct via API for all LLM steps | `Qwen/Qwen2.5-1.5B-Instruct`, downloaded once and run fully offline (`ULTRATAG/generation.py`) | This repo has no instruct-tuned model, no local Ollama server, and the API keys already checked into `textual_enhancer/` and `tags_data_augmentation/` are **leaked secrets that must not be reused** (flagged during implementation — do not copy that pattern or commit those files). A small free instruct model keeps every judgment LLM-made (not a heuristic) while needing no API key. |
| Fixed cosine-similarity threshold tau_1 = 0.8 for virtual edges (Eq. 17) | Mean-centered embeddings + a **percentile-based** threshold (top 1% of same-soft-label pairs), `ULTRATAG/structure_augmentation.py::build_virtual_edges` | Measured directly on Cora: this repo's frozen local LLM's raw mean-pooled embeddings are highly anisotropic (off-diagonal cosine similarity averages **0.84** between *any* two nodes' texts). A fixed 0.8 cutoff calibrated for the paper's production embedding model would connect almost every same-label pair here. Centering + a percentile threshold is scale-invariant to whichever embedding model is plugged in. |
| Dense `[N, N]` similarity matrix blended into GNN_2's adjacency (Eq. 26) | Top-k (`k=10`) sparsified similarity edges via `torch_geometric.utils.coalesce`, `ULTRATAG/dual_gnn.py` | A dense `N x N` matrix is ~1.5GB for PubMed's ~20k nodes recomputed every epoch; top-k sparsification is the standard approach graph-structure-learning methods (e.g. IDGL) use for the same reason, and preserves the mechanism exactly: learned similarity augments the adjacency for non-important-node pairs, LLM judgments are untouched for important-node pairs. |
| 5 datasets, seeds 42-46, sparsity ratios 0/20/50/80% | Cora + PubMed, seeds 42/43/44, ratios **0% and 80%** only | Scope decisions made explicitly with the user given this pipeline's LLM-generation cost (see "Runtime" below) — ideal case vs. worst-case sparsity was chosen to establish the robustness story without the full 4-point curve. |
| Augmentation re-run per seed (implied by "averaged over five runs") | Augmentation (sparsity mask, text propagation/generation, virtual edges, PageRank selection, edge reconfiguration) computed **once** per (dataset, ratio) using a fixed `--augmentation_seed`; only LM-finetuning init and dual-GNN init/training vary across `--seeds` | Every augmentation run makes O(N) to O(N + \|E_important\|) local LLM generation calls — re-running that per training seed would multiply an already significant cost for no scientific benefit, since the interesting stochasticity (model init, LoRA finetuning) is downstream of it anyway. Mirrors `STAGE/`'s embedding-caching strategy. |
| PubMed's LLM-generated texts | Generated fresh with the local instruct model above | This repo's cached `datasets/enhanced_texts/pubmed/` is empty (only Cora was previously enhanced, and with a different, more capable external LLM lacking a soft-label field) — see exploration notes. Cora is also regenerated fresh here rather than reusing that cache, so both datasets go through an identical, comparable augmentation pipeline. |

Everything else follows the paper directly: text propagation via neighbor-text
concatenation (Eq. 10), LLM-generated summary + keywords + soft-label concatenated
into the final augmented text (Eq. 11-14), a PageRank-selected top-10%-of-training-
nodes "important" node set (Eq. 18-19) whose edges the LLM edge-reconfigurator
re-verifies (Eq. 20-21) and whose adjacency entries are frozen during dual-GNN
training (Eq. 26), and joint dual-GNN optimization by a single cross-entropy loss
(Eq. 27-28). LM-finetuning and dual-GNN hyperparameters (lr, epochs, dropout,
important-node ratio, edge-confidence threshold tau_2=0.5) are taken directly from
the paper's Section 5.4.

## Files

- `generation.py` — `InstructLLM`: downloads/loads the local instruct model once
  (cached under `ULTRATAG/models_cache/`, git-ignored) and does batched chat-template
  generation; `parse_json_object` robustly extracts structured output.
- `sparsity.py` — `simulate_sparsity`: randomly blanks node texts and drops edges at
  a given ratio (the paper's sparse-scenario simulation, Section 4).
- `text_augmentation.py` — `propagate_texts` (Eq. 10) and `augment_texts` (Eq. 11-14:
  LLM summary/keywords/soft-label generation + concatenation into T*).
- `structure_augmentation.py` — `add_virtual_edges` (Eq. 15-17), `pagerank_important_nodes`
  (Eq. 18-19, via `networkx`), `reconfigure_edges` (Eq. 20-21, LLM edge-confidence scoring).
- `lm_finetune.py` — `finetune_lm_and_extract_embeddings`: LoRA-finetunes an LM
  classifier (reusing `train_llm.peft_model`/`train_llm.training_config` as-is) on
  the augmented texts, then extracts pooled node embeddings H (Eq. 22-24).
- `dual_gnn.py` — `DualGNNTrainer`: the joint dual-GNN (Eq. 25-28); `WeightedGCN` is
  new (needed for dense edge-weight support that `common.gnn.GNNEncoder` lacks),
  GNN_1 reuses `common.gnn.GNNEncoder`.
- `pipeline.py` — orchestrates modules 1-3 end to end, with the augmentation-caching
  strategy described above (`build_augmented_graph` / `run_downstream`).
- `run_ultratag.py` / `run_ultratag.bat` — CLI entry point and Windows convenience script.

## Usage

From the repo root:

```bash
python -m ULTRATAG.run_ultratag --dataset cora   --ratios 0.0 0.8 --seeds 42 43 44
python -m ULTRATAG.run_ultratag --dataset pubmed --ratios 0.0 0.8 --seeds 42 43 44
```

Or on Windows: `ULTRATAG\run_ultratag.bat` (no args runs both datasets; pass args to
forward them to a single custom run). Results are printed to stdout and saved as
JSON to `ULTRATAG/results/{dataset}_{llm_name}.json`.

## Runtime

The dominant cost is LLM text-generation during augmentation (~0.17s/node batched at
16 on the local instruct model), run once per (dataset, ratio): roughly 8-12 minutes
for Cora (2,708 nodes) and 55-70 minutes for PubMed (19,717 nodes) per ratio. LM
finetuning and dual-GNN training are comparatively cheap and repeat per seed. See
`results/run_logs/` for per-run wall-clock timing once you've run a sweep.

**GPU-memory pitfall (fixed, kept here for anyone extending this code):** the
generator (`InstructLLM`) and embedder (`StageTextEmbedder`) are only needed while
building the augmented graph, not during the downstream LM-finetuning/dual-GNN
training that follows. An earlier version of `run_ultratag.py` kept both loaded for
the whole script; on a 12GB GPU that pushed VRAM to ~96% once LM-finetuning's own
model loaded on top, and empirically caused a >2000x per-step slowdown (~330s/step
measured vs. ~0.15s/step for the identical model/batch in isolation) -- almost
certainly CUDA falling back to memory-pressure paging rather than an OOM error, which
made it easy to miss. `run_ultratag.py` now loads the generator/embedder only on an
augmentation-cache miss and explicitly frees them (`del` + `torch.cuda.empty_cache()`)
before the per-seed downstream loop. If you see a similar cliff in per-step time
without an explicit OOM, check `nvidia-smi` for near-limit VRAM before assuming the
model or data is the problem.

## Results

`llama_3.2_1B` LM/embedder, `Qwen2.5-1.5B-Instruct` generator, 3 seeds (42-44),
augmentation seed 42, 60/20/20 split:

| Dataset | Ratio | Test Acc | Test Macro F1 |
|---|---|---|---|
| Cora | 0% (ideal) | 84.13 ± 0.60 | 81.19 ± 1.04 |
| Cora | 80% (severe sparsity) | 32.96 ± 2.32 | 13.21 ± 4.75 |

The 0% result is broadly consistent with STAGE's Cora numbers in this repo (mid-80s),
confirming the pipeline works correctly end to end. The 80% result shows real, but
much smaller, robustness than the paper's own UltraTAG-S (90.96% -> 57.57% on Cora,
a ~33pp drop, vs. our ~51pp drop) — our 80%-sparsity number is roughly in the range
the *paper's own non-robust baselines* land at under the same sparsity (their MLP:
30.41%, GCN: 41.96%), suggesting our substituted components (a 1.5B non-instruct-
tuned-for-this-task generator instead of Llama-3-8B-Instruct, recalibrated similarity
thresholds, top-k adjacency sparsification, a single fixed augmentation seed) recover
noticeably less structure/text signal at extreme sparsity than the paper's full-scale
setup, even though the mechanism (propagation, virtual edges, PageRank selection,
LLM edge reconfiguration) is implemented faithfully. This gap is worth being upfront
about in any write-up rather than presenting the 80% number as matching the paper's
robustness claim.

PubMed results pending (running in the background — see `results/*.json` once complete).
