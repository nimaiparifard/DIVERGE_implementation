"""
Aggregated learning of retrieved content (SKETCH paper Section 3.3, Figure 3).

The semantic ranking R_sem and the structural ranking R_struct are fused with a
weighted rank score G = R_sem + w * R_struct, the candidates are sorted by G, and
their texts are concatenated after the anchor text until the LM's token budget L is
used up. Each list's rank is turned into a score in (0, 1] (1 = best, decaying
linearly with rank position) so that a node absent from one list simply gets 0
from it and a node found by both retrievers is boosted.

`build_contexts` also implements the retrieval-strategy variants of the paper's
Table 2 so they can be run as ablations:
    original   anchor text only
    shuffled   random nodes from the whole graph
    semantic   semantic ranking only
    onehop     random 1-hop neighbours
    multihop   random K-hop neighbours
    combined   random mix of the semantic and K-hop candidate pools
    sketch     weighted rank fusion of semantic + structural rankings (the method)
"""
import numpy as np

STRATEGIES = ["sketch", "original", "shuffled", "semantic", "onehop", "multihop", "combined"]

ANCHOR_HEADER = "Target node: "
NEIGHBOR_HEADER = "\nRelated node: "


def _rank_scores(indices_row):
    valid = indices_row[indices_row >= 0]
    n = len(valid)
    return {int(c): (n - p) / n for p, c in enumerate(valid)}


def fuse_rankings(sem_row, struct_row, structural_weight):
    """Candidate ids sorted by G = score_sem + w * score_struct (best first)."""
    sem, struct = _rank_scores(sem_row), _rank_scores(struct_row)
    candidates = set(sem) | set(struct)
    return sorted(candidates, key=lambda c: -(sem.get(c, 0.0) + structural_weight * struct.get(c, 0.0)))


def truncate_texts(raw_texts, tokenizer, max_tokens):
    """Each node text cut to at most `max_tokens` tokens; also returns its token length."""
    texts, lengths = [], []
    batch = 512
    for start in range(0, len(raw_texts), batch):
        chunk = [t if t else "empty" for t in raw_texts[start:start + batch]]
        enc = tokenizer(chunk, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
        for ids in enc:
            texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
            lengths.append(len(ids))
    return texts, lengths


def order_candidates(strategy, num_nodes, sem_idx, struct_idx, khop_lists, onehop_lists,
                     structural_weight, rng):
    """Per-node ordered list of retrieved node ids for the chosen strategy."""
    orders = []
    for v in range(num_nodes):
        if strategy == "original":
            order = []
        elif strategy == "sketch":
            order = fuse_rankings(sem_idx[v], struct_idx[v], structural_weight)
        elif strategy == "semantic":
            order = [int(c) for c in sem_idx[v] if c >= 0]
        elif strategy == "shuffled":
            pool = rng.choice(num_nodes, size=min(64, num_nodes), replace=False)
            order = [int(c) for c in pool if c != v]
        elif strategy == "onehop":
            order = [int(c) for c in rng.permutation(onehop_lists[v])]
        elif strategy == "multihop":
            order = [int(c) for c in rng.permutation(khop_lists[v])]
        elif strategy == "combined":
            pool = set(int(c) for c in sem_idx[v] if c >= 0) | set(int(c) for c in khop_lists[v])
            order = [int(c) for c in rng.permutation(sorted(pool))]
        else:
            raise ValueError(f"Unknown strategy '{strategy}', choose from {STRATEGIES}")
        orders.append(order)
    return orders


def build_contexts(anchor_texts, anchor_lengths, neighbor_texts, neighbor_lengths, orders, max_tokens, reserve=8):
    """
    Anchor text followed by as many retrieved node texts (in `orders` order) as fit
    into `max_tokens`. Header strings and special tokens are covered by `reserve`
    plus a small per-neighbour allowance, and the tokenizer truncates at max_length
    anyway, so a slight overshoot is harmless.
    Returns (contexts list[str], num_neighbours_used list[int]).
    """
    header_cost = 5
    contexts, used = [], []
    for v, order in enumerate(orders):
        budget = max_tokens - reserve - anchor_lengths[v] - header_cost
        parts = [ANCHOR_HEADER + anchor_texts[v]]
        n_used = 0
        for c in order:
            cost = neighbor_lengths[c] + header_cost
            if cost > budget:
                break
            parts.append(NEIGHBOR_HEADER + neighbor_texts[c])
            budget -= cost
            n_used += 1
        contexts.append("".join(parts))
        used.append(n_used)
    return contexts, used
