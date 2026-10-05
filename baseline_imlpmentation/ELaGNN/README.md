# E-LLaGNN — Efficient LLMs augmented GNNs

Implementation of:

> Jaiswal, Choudhary, Adkathimar, Alagappan, Hiranandani, Ding, Wang, Huang, Subbian.
> *"All Against Some: Efficient Integration of Large Language Models for Message Passing
> in Graph Neural Networks."* arXiv:2407.14996, 2024. (`Elagnn.pdf`)

E-LLaGNN treats the LLM as an **on-demand service** during GNN training. Only a small,
heuristically chosen fraction of nodes is ever sent to the LLM. The GNN sees a random
mix of original and LLM-enhanced features, and inference can run **without the LLM**.

The four components of the paper (§2.2) and where they live:

| Paper component | Code |
|---|---|
| ① Query node enhancement `t_i → t_i⁺`, encoded with Sentence-BERT into `x_i⁺` | `generation.py`, `encoding.py` |
| ② Neighbourhood sampling and enrichment: a uniform GraphSAGE fan-out, re-ranked by cosine similarity to the (enhanced) query, top `K` kept (K = 5) | `model.py::sample_topk_neighbourhood` |
| ③ On-demand neighbour enhancement with a prompt catalog: each eligible node is shown enhanced with probability `p`, using a random prompt each epoch | `prompts.py`, `trainer.py::mix_features` |
| ④ Aggregation with a plain MEAN GraphSAGE (no norms or skips; 2/4/8 layers) | `model.py::PlainSAGE` |
| §2.4 Active node selection: PageRank, clustering distance, text length, degree | `selection.py` |
| §2.5 LLM-free inference pipeline (Figure 2) and LLM-incorporated inference (Table 8) | `trainer.py::evaluate` |
| §3.7 Gradient flow (Eq. 2), logged per epoch | `model.py::PlainSAGE.gradient_flow` |

The code only reads from the rest of the repo: `common` loaders and metrics, and
`ULTRATAG.generation.InstructLLM` with its already-downloaded Qwen model. Nothing
outside `ELaGNN/` is modified.

## Supervised setting and label safety

- `--split repo` (the default) uses the same 60/20/20 seeded split as the rest of DIVERGE.
  `--split planetoid` uses the dataset's own fixed masks (the public split for Cora,
  PubMed and CiteSeer).
- Node selection, LLM prompts and neighbourhood ranking never use node labels. The
  "argue category" prompt lists the dataset's class *names*, as in the paper's Table 2.
- The loss uses only train labels. Validation is used for early stopping. The test
  split is scored once, with the best-validation checkpoint.
- Model selection and the reported `test_acc` use **LLM-free** inference: original
  features, and neighbours ranked with the original query. The LLM-incorporated test
  accuracy is reported alongside it, as in Table 8.

## Differences from the paper

| Paper | Here | Why |
|---|---|---|
| Vicuna-7B / LLaMA-2-7B-Chat, temperature 0.7 | Qwen2.5-1.5B-Instruct (already cached by ULTRATAG), temperature 0.7, top-p 0.95; `--llm_repo` to change | Runs offline on a 12 GB GPU without new downloads. |
| An LLM call per selected node, per epoch, at runtime | Each (node, prompt) pair is generated once and cached in `ELaGNN/cache/`; every epoch then draws a random cached variant | Same knowledge and same diversity across epochs, but each generation is paid for only once. A larger budget reuses earlier generations. |
| The prompt catalog includes a pairwise prompt ("compare paper P1 with P2") | Three per-node prompts: explain, argue-category, summarize-entities | A pairwise prompt needs one generation per (query, neighbour) pair and can't be cached per node, which breaks the budget. It can be added in `prompts.py`. |
| Mini-batch GraphSAGE sampler | Full-batch training; the restricted top-K neighbourhood of every node is re-sampled each epoch | Same restricted neighbourhood, and faster for these graph sizes. Evaluation ranks all neighbours (no random fan-out), so it is deterministic. |
| Hyperparameters from GCNII / Duan et al. | lr 0.01, wd 5e-4, hidden 128, dropout 0.5, 500 epochs, patience 100 | All can be changed on the command line. |

## Usage

From the repo root:

```bash
# main setting: 25% of nodes enhanced, PageRank selection, 5 seeds
python -m ELaGNN.run_elagnn --dataset cora --budget 0.25 --heuristic pagerank

# no-LLM reference (0% column of Table 5), and deeper backbones
python -m ELaGNN.run_elagnn --dataset cora --budget 0
python -m ELaGNN.run_elagnn --dataset cora --budget 0.25 --n_layers 8

# Table 7: node selection heuristics
python -m ELaGNN.run_elagnn --dataset cora --budget 0.25 --heuristic text_length

# absolute budgets for large graphs (Table 6), and generation only
python -m ELaGNN.run_elagnn --dataset arxiv --budget 5000
python -m ELaGNN.run_elagnn --dataset pubmed --budget 0.25 --generate_only
```

`ELaGNN\run_elagnn.bat` with no arguments runs the Table 5 sweep on Cora
(budgets 0–50% × 2/4/8 layers).

**LLM cost.** Generation cost is `budget × number of prompts`. For example, Cora at 25%
is 677 nodes × 3 prompts, about 2k generations. Generation checkpoints to disk as it goes
and can be resumed.

**Outputs.** Results are written to
`ELaGNN/results/<dataset>_<split>_L<layers>_<heuristic>_b<budget>_K<K>.json`. Each file
holds per-seed val/test accuracy, macro-F1, weighted-F1, LLM-incorporated accuracy, best
epoch, and the per-epoch gradient flow.

## Verified so far (Cora, 60/20/20 split)

- Budget 0 (no LLM), 2 layers, seeds 42 and 43: test accuracy 88.10 ± 1.20.
- 8 layers, seed 42: 88.38.
- Budget of 20 nodes (text-length selection, 60 generations): the pipeline runs end to
  end, and the generated enhancements look sensible.

A full run at a 25% budget has not been done yet.
