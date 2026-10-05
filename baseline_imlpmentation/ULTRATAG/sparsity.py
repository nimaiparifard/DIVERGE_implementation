"""
Simulate real-world TAG sparsity (paper Section 4): randomly delete a fraction of
node texts and a fraction of edges, independently, at a given ratio.
"""
import random

import torch


def simulate_sparsity(edge_index, raw_texts, num_nodes, ratio, seed):
    """
    ratio: fraction in [0, 1) of node texts to blank out and of edges to drop.
    Returns (sparse_texts, sparse_edge_index, deleted_text_mask).
    """
    rng = random.Random(seed)

    sparse_texts = list(raw_texts)
    deleted_text_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if ratio > 0:
        num_to_delete = int(round(ratio * num_nodes))
        deleted_ids = rng.sample(range(num_nodes), num_to_delete)
        for idx in deleted_ids:
            sparse_texts[idx] = ""
            deleted_text_mask[idx] = True

    num_edges = edge_index.shape[1]
    if ratio > 0 and num_edges > 0:
        num_to_keep = int(round((1 - ratio) * num_edges))
        keep_idx = rng.sample(range(num_edges), num_to_keep)
        keep_idx = torch.tensor(sorted(keep_idx), dtype=torch.long)
        sparse_edge_index = edge_index[:, keep_idx]
    else:
        sparse_edge_index = edge_index

    return sparse_texts, sparse_edge_index, deleted_text_mask
