# LLM-BP — Belief Propagation over LLM Task-Adaptive Embeddings

Implementation of the method from:

> Wang, Liu, Wei, Li. *"Generalization Principles for Inference over Text-Attributed
> Graphs with Large Language Models."* ICML 2025, arXiv:2502.11836. (`LLM_BP.pdf`)

The paper targets **zero-/few-shot** node classification and trains nothing:

1. **Principle I — task-adaptive embeddings.** A frozen decoder LLM (LLM2Vec) encodes each
   node with a class-conditional prompt (`Given the {task}, classify it into one of the
   following k classes: {labels} / Query: {text} / Response:`); the hidden state at the
   response position is the node embedding.
2. **Class embeddings** `q_k`: from LLM-labelled samples (zero-shot) or the mean embedding of
   the labelled nodes of class k (few-shot).
3. **Principle II — belief propagation.** The graph is modelled as an MRF. Node potentials are
   `softmax(cos(h_i, q_k)/tau)` (Eq. 5), and the edge potential is `r` for same-class pairs
   and `1-r` otherwise (Eq. 6). The homophily level `r` is estimated by an LLM from sampled
   edges. Inference runs loopy BP (Eq. 4) or its one-layer linear approximation (Eq. 7).

## Supervised adaptation

The method works for supervised node classification unchanged: the paper's few-shot branch
(Algorithm 1, lines 6-7) already builds class embeddings from labelled nodes, so a supervised
split is the few-shot setting with every training node as a shot. Labels also make two parts
of the pipeline simpler or stronger:

| Paper (zero/few-shot) | This implementation (supervised) |
|---|---|
| `q_k` from GPT-4o pseudo-labels or k shots | `q_k` = mean embedding of all training nodes of class k |
| `r` estimated by GPT-4o-mini from 50-100 sampled edges | `r` measured exactly on training edges (both endpoints labelled); `--homophily` overrides it |
| Scalar edge potential only | Also a full `c x c` class-compatibility matrix from training edges (`BP (matrix)`), the "more complex psi" the paper mentions after Eq. 6 |
| Labelled nodes only shape `q_k` | Training nodes are also observed evidence: their potential is clamped to their (smoothed) label during BP (`--no_clamp_train` turns this off) |
| Fixed `tau=0.025` (BP) / `0.01` (appr.), `L=5` | `tau` and `L` chosen on the validation set (`--no_tune` restores the paper's values) |
| LLM2Vec-Mistral-7B encoder | Any local checkpoint in `local_models/` (default `llama_3.2_1B`), last-token pooling |

No weights are updated anywhere. Once the embeddings are cached, a seed with the full
tau/L grid search takes a few seconds.

## Files

- `embedding.py` — class-conditional (`task_adaptive`) or `vanilla` prompts, last-token or
  mean pooling with a frozen local LLM; cached in `LLM_BP/cache/` (git-ignored).
- `bp.py` — class embeddings, node/edge potentials, loopy BP (log space, one message per
  directed edge) and the Eq. 7 approximation.
- `run_llm_bp.py` — CLI: embeddings → per-seed split → validation-tuned `Raw`, `BP`,
  `BP (appr.)`, `BP (matrix)` → summary + JSON in `LLM_BP/results/`.
- `run_llm_bp.bat` — Windows wrapper (default sweep or forwards arguments).

## Usage

From the repo root:

```bash
python -m LLM_BP.run_llm_bp --dataset cora --llm_name llama_3.2_1B --seeds 0 1 2 3
python -m LLM_BP.run_llm_bp --dataset citeseer --encoding vanilla          # no class info in the prompt
python -m LLM_BP.run_llm_bp --dataset cora --no_tune --no_clamp_train      # closest to the paper's few-shot run
python -m LLM_BP.run_llm_bp --dataset history --split default              # dataset's own 10/10/60 split
```

`--split repo` (default) uses the repo's 60/20/20 split per seed (`load_graph_dataset(re_split=True)`,
honours `seed_snapshot/splits`), so numbers are comparable with DIVERGE, STAGE and SKETCH.
The `Raw` row is nearest-class-embedding classification without the graph (the paper's "Raw").

## Results so far

Cora, `llama_3.2_1B`, 60/20/20 split, seeds 0-3, tau/L tuned on validation:

| Encoding / pooling | Raw | BP | BP (appr.) | BP (matrix) |
|---|---|---|---|---|
| task-adaptive / last | 69.37 ± 2.39 | 86.07 ± 2.03 | 84.69 ± 1.78 | 85.75 ± 1.90 |
| task-adaptive / mean | 68.59 ± 0.44 | 85.93 ± 1.95 | 83.94 ± 1.50 | 85.43 ± 1.90 |
| vanilla / last | 71.82 ± 1.68 | 86.39 ± 1.55 | 84.64 ± 1.37 | 86.53 ± 2.25 |
| vanilla / mean | 69.14 ± 0.94 | 85.88 ± 2.33 | 84.14 ± 1.54 | 85.75 ± 1.79 |

BP adds about 17 points over the graph-free baseline. With a 1B base model the
class-conditional prompt gives no gain, which matches the paper's observation (Sec. 4.1,
Fig. 4) that class information helps only encoders that are LLM decoders trained as embedders
(LLM2Vec). A stronger embedding backbone is the main lever for this method. For reference,
the STAGE ensemble (trained GNNs on the same frozen `llama_3.2_1B` features) reaches
88.28 ± 1.42 on Cora.
