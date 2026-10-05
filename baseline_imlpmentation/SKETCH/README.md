# SKETCH — Semantic Knowledge and Structure Enrichment

Implementation of:

> Zhou, Wang, Chen, Du, Zheng, Xu, Huang. *"Taming Language Models for Text-attributed
> Graph Learning with Decoupled Aggregation."* ACL 2025, pp. 3463–3474. (`SKETCH.pdf`)

SKETCH is a **supervised node-classification** method with no GNN. Node aggregation
happens in text: each node's text is extended with the texts of related nodes, and a
long-context LM is fine-tuned on the result with the node labels.

1. **Semantic retrieval** (§3.1): sentence-transformer embeddings of every node text
   and a FAISS cosine top-k search over the whole graph.
2. **Structural retrieval** (§3.2, Algorithm 1): the nodes within K hops are ranked by
   their multi-hop Jaccard similarity to the anchor (1-1, 1-2, 2-1 and 2-2 hop
   intersections). The Jaccard is estimated with MinHash signatures: signature length
   `l` for the 1-hop/1-hop pair and `m < l` for the other pairs.
3. **Aggregation** (§3.3): the two rankings are combined as `G = R_sem + w · R_struct`.
   Candidate texts are added after the anchor text in order of G until the token
   budget `L` is used.
4. **LM training**: LoRA fine-tuning of a sequence classifier on the enriched texts,
   with early stopping on validation accuracy. The test split is evaluated once, with
   the best-validation checkpoint.

The code only reads from the rest of the repo (`common` loaders and metrics,
`local_models/`, `configs/llm_configs/`). Nothing outside `SKETCH/` is modified.

## Supervised setting and label safety

- The default split is `--split repo`: the same 60/20/20 seeded split as the rest of
  DIVERGE, through `load_graph_dataset(re_split=True)`, so it uses `seed_snapshot/splits`
  when present. `--split paper` uses the paper's 80/10/10 split (Appendix A.1).
- Retrieval uses only node texts and edges, never labels. Test-node texts can appear as
  retrieved context, which is the standard transductive setting (a GNN sees them too).
  Their labels are never shown to the model.
- Only train-node labels are used in the loss. Validation labels are used only for
  early stopping and checkpoint selection.

## Differences from the paper

| Paper | This implementation | Why |
|---|---|---|
| Nomic-BERT (137M, 8k context) or Llama-3-8B as the LM | Any local model in `configs/llm_configs/`. Default: `llama_3.2_1B` (128k context) with LoRA | These checkpoints are already in `local_models/` and fit a 12 GB GPU. Encoder models such as `deberta_v3_large` are capped at 512 tokens automatically. |
| `L` up to 8k tokens | `--max_length 2048` by default | Memory on a 12 GB GPU. Raise it on a bigger card. |
| Rank-fusion formula `G = R_sem + w·R_struct` (rank normalisation not specified) | Each ranking is turned into a linear rank score in (0, 1]. A node missing from a list gets 0 from that list | The paper does not say how ranks are normalised. With this choice, a node found by both retrievers is ranked higher. |
| Datasets: ACM, Wikipedia, Amazon | Any dataset in `datasets/*.pt` that has `raw_texts` (cora, pubmed, citeseer, arxiv, wikics, …) | The paper's datasets are not in this repo. |
| Adam, lr 1e-3 | AdamW, lr 2e-4, warm-up then linear decay | A lower lr is the usual stable choice for LoRA on a decoder LM. Change it with `--lr`. |

Two MinHash details are unspecified in the paper and chosen here. `N^k(v)` is the set of
nodes reachable within k hops. Ties in the collision count are broken in favour of the
closer hop.

## Files

- `semantic.py`: sentence embeddings (cached) and FAISS exact cosine top-k, with a torch fallback.
- `structural.py`: universal-hash MinHash signatures for every hop (computed with scatter-min
  message passing), chunked sparse K-hop candidate generation, and multi-hop collision scoring.
- `aggregate.py`: rank fusion, token-budgeted context building, and the Table 2
  retrieval-strategy ablations.
- `classifier.py`: LoRA sequence classifier and its training/evaluation loop.
- `run_sketch.py`: command-line entry point.
- `run_sketch.bat`: Windows launcher, same pattern as `STAGE/run_stage.bat`.

## Usage

From the repo root:

```bash
python -m SKETCH.run_sketch --dataset cora --seeds 42 43 44 45
python -m SKETCH.run_sketch --dataset pubmed --split paper --seeds 42
python -m SKETCH.run_sketch --dataset cora --llm_name deberta_v3_large          # 512-token encoder
python -m SKETCH.run_sketch --dataset cora --retrieval_only                     # only build/inspect contexts
```

Table 2 ablations (`--strategy`): `original` (anchor text only), `shuffled` (random
nodes), `semantic`, `onehop` (random 1-hop), `multihop` (random K-hop), `combined`
(random semantic + K-hop), and `sketch` (the full method, the default).

Main hyperparameters: `--max_length` (L), `--structural_weight` (w), `--num_hops` (k),
`--num_hashes_l`/`--num_hashes_m` (l, m), `--top_k`, `--neighbor_tokens`, `--anchor_tokens`.

Results are written to `SKETCH/results/<dataset>_<llm>_<strategy>_<split>_L<L>_w<w>.json`.
Each file holds per-seed val/test accuracy, macro-F1, weighted-F1, the validation curve,
training time, and retrieval time.

## Retrieval sanity check (Cora)

Share of retrieved nodes that have the same label as the anchor:

| Retriever | top-5 |
|---|---|
| random nodes | 18.2% |
| semantic | 78.4% |
| structural (MinHash) | 80.2% |
| **SKETCH fusion** | **80.7%** |

The MinHash estimate of the 1-hop Jaccard (R = 512) has a correlation of 0.995 with the
exact value. Structural retrieval on Cora takes 0.1 s.
