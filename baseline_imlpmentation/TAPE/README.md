# TAPE — LLM-to-LM Interpreter for Text-Attributed Graphs

Implementation of:

> He, Bresson, Laurent, Perold, LeCun, Hooi. *"Harnessing Explanations: LLM-to-LM
> Interpreter for Enhanced Text-Attributed Graph Representation Learning."*
> ICLR 2024. (`TAPE.pdf`)

TAPE's core idea: prompt an LLM for a **ranked prediction list + a textual
explanation** per node, then fine-tune a small LM to turn that explanation (and the
original text) into node features, and finally train a GNN on the combination.
Concretely (Eq. 4-7): fine-tune LM_orig on the raw text and LM_expl on the LLM's
explanation, build a prediction feature h_pred from the ranked list, then
independently train a GNN on each of {h_orig, h_expl, h_pred} and mean-pool their
softmax outputs into the final h_TAPE prediction.

## Why this implementation is lighter than STAGE/ULTRATAG

This repo already contains full, pre-existing LLM outputs for exactly this purpose:
`datasets/gpt_4o/cora.json` (2,708/2,708 nodes) and `datasets/PubMed/{i}.json`
(19,717/19,717 nodes) are node-classification responses from an earlier pass through
this project, in the same "prediction + explanation" shape TAPE's prompt (Table 5)
asks for. **This implementation makes no new LLM API/generation calls at all** — it
parses and reuses that existing data. It also reuses this repo's existing LoRA/PEFT
finetuning class (`train_llm.train.Train`) unchanged for LM_orig/LM_expl (its
`used_llm_responses` flag already selects between raw text and the cached LLM
response text — exactly TAPE's two text sources), `STAGE.models`/`STAGE.trainer`/
`STAGE.ensemble` for the GNN-training-and-mean-pooling step (Eq. 6-7 is the same
mechanism as STAGE's ensemble, just applied across feature *sources* instead of
architectures), and `ULTRATAG.lm_finetune.extract_pooled_embeddings` for pooling a
fine-tuned LM's hidden states into node embeddings.

## Deviations from the paper

| Paper | This implementation | Why |
|---|---|---|
| GPT-3.5-turbo / Llama2-13b-chat generates predictions+explanations fresh per experiment | Reuses this repo's pre-existing cached responses (`datasets/gpt_4o/cora.json`, `datasets/PubMed/*.json`) | Already present, full node coverage, no API key or local instruct model needed — see "Why this implementation is lighter" above. |
| PubMed's cached response format doesn't literally follow the paper's "Answer: X, Y" prompt structure (it predates this paper's exact prompt wording — an earlier gpt-3.5-turbo-0301 pipeline) | `TAPE/prediction_parser.py` falls back to ranking candidate labels by first-mention order in the free text, using a small alias table (PubMed's cached text says "Type 1 diabetes", while the dataset's real `label_name` is "Diabetes Mellitus Type 1") | Verified empirically against the actual cached files (see module docstring) before relying on it — Cora's cache follows the clean "Answer: ..." format directly. |
| Top-k ranked predictions, k fixed by the dataset's prompt (Table 5) | k = num_classes for both Cora and PubMed | Cora/PubMed's own prompt wording (Table 5) doesn't fix a k ("comma-separated list... ordered from most to least related"), unlike ogbn-arxiv/products' explicit "top 5" — so every class is a valid ranking slot. |
| h_pred's one-hot-to-dense projection (Eq. 5) | Fixed, seed-deterministic random linear projection | The paper describes this as a preprocessing step with no explanation of a dedicated training loss for it (unlike h_orig/h_expl, which are explicitly cross-entropy fine-tuned per Eq. 5) — a fixed random projection is simple, reproducible, and doesn't require inventing an unstated training procedure. |
| 5 datasets (Cora, PubMed, ogbn-arxiv, ogbn-products, tape-arxiv23) | Cora + PubMed | Matches STAGE/ULTRATAG's established scope in this repo for comparability; also the only two datasets with full cached LLM responses available here. |
| RevGAT (Li et al., 2022a) | `STAGE.models.build_revgat`: GNNEncoder(GAT + residual + batch norm) | Same approximation already used and documented in `STAGE/README.md` — reused directly for consistency rather than re-justified here. |

Everything else follows the paper directly: LM_orig/LM_expl are LoRA-finetuned with
cross-entropy on their respective text sources (Eq. 4-5), h_pred is built from the
parsed ranked predictions (Eq. 5), all three GNNs per architecture are trained
independently and mean-pooled (Eq. 6-7), and GNN hyperparameters (hidden_dim=256,
3 layers, GCN/SAGE lr=0.01 dropout=0.5, RevGAT lr=0.002 dropout=0.75, 1000 max
epochs / 50-epoch early stopping) are taken directly from the paper's Table 10.

## Files

- `prediction_parser.py` — `parse_llm_response`: extracts a ranked label list from
  the cached LLM response text (used for h_pred only; the *full* raw text is what
  `used_llm_responses=True` already feeds to LM_expl, unchanged).
- `pred_features.py` — `build_h_pred`: one-hot + fixed random projection (Eq. 5).
- `finetune.py` — `finetune_and_extract`: LoRA-finetunes LM_orig or LM_expl via
  `train_llm.train.Train` and extracts pooled node embeddings.
- `run_tape.py` / `run_tape.bat` — orchestrates the full pipeline and reports results
  in the paper's Table 1/3 layout (per architecture: h_orig, h_expl, h_pred, h_TAPE).

## Usage

From the repo root:

```bash
python -m TAPE.run_tape --dataset cora   --seeds 42 43 44 45
python -m TAPE.run_tape --dataset pubmed --seeds 42 43 44 45
```

Or on Windows: `TAPE\run_tape.bat`. Results are printed to stdout and saved as JSON
to `TAPE/results/{dataset}_{llm_name}.json`.

## Results

_Populated after the first full run — see `TAPE/results/*.json`._
