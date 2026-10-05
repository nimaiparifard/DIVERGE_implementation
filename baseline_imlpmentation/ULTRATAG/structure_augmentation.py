"""
Structure Augmentation (paper Section 4.1c, Eq. 15-21): Virtual Edge Generator,
PageRank-based Node Selector, and LLM-based Edge Reconfigurator.
"""
from collections import defaultdict

import torch
import torch.nn.functional as F

from ULTRATAG.generation import parse_json_object
from ULTRATAG.text_augmentation import DATASET_DOMAINS


def build_virtual_edges(embeddings, soft_labels, top_percentile=1.0):
    """Eq. 15-17: connect nodes sharing the same LLM soft label whose embeddings
    are highly cosine-similar.

    Deviation from the paper: Eq. 17 uses a fixed absolute threshold (tau_1=0.8),
    calibrated for the paper's production embedding model. Our substituted frozen
    causal LM (a small local checkpoint, chosen for offline reproducibility -- see
    ULTRATAG/README.md) produces mean-pooled hidden states with a well-documented
    anisotropy problem: measured on Cora, off-diagonal cosine similarity between
    *any* two nodes' raw embeddings averages 0.84 (min 0.49), making an absolute
    0.8 cutoff nearly meaningless (it would connect almost every pair). We instead
    (a) mean-center the embeddings, a standard cheap anisotropy correction, and
    (b) threshold same-soft-label pairs at the `top_percentile`% most similar
    globally, which is robust to whatever the embedding model's raw similarity
    scale happens to be and preserves the mechanism's intent: connect only the
    most semantically similar same-label pairs.
    """
    groups = defaultdict(list)
    for idx, label in enumerate(soft_labels):
        if label:
            groups[label].append(idx)

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=1)

    per_group = {}
    all_sims = []
    for label, idxs in groups.items():
        if len(idxs) < 2:
            continue
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        sim = normalized[idx_tensor] @ normalized[idx_tensor].t()
        iu = torch.triu_indices(len(idxs), len(idxs), offset=1)
        pair_sims = sim[iu[0], iu[1]]
        per_group[label] = (idxs, iu, pair_sims)
        all_sims.append(pair_sims)

    if not all_sims:
        return []

    all_sims_cat = torch.cat(all_sims)
    threshold = torch.quantile(all_sims_cat, 1 - top_percentile / 100.0).item()

    new_edges = []
    for idxs, iu, pair_sims in per_group.values():
        keep = pair_sims > threshold
        rows, cols = iu[0][keep], iu[1][keep]
        for r, c in zip(rows.tolist(), cols.tolist()):
            new_edges.append((idxs[r], idxs[c]))
            new_edges.append((idxs[c], idxs[r]))
    return new_edges


def add_virtual_edges(edge_index, embeddings, soft_labels, top_percentile=1.0):
    new_edges = build_virtual_edges(embeddings, soft_labels, top_percentile)
    if not new_edges:
        return edge_index
    new_edge_tensor = torch.tensor(new_edges, dtype=torch.long, device=edge_index.device).t()
    combined = torch.cat([edge_index, new_edge_tensor], dim=1)
    return torch.unique(combined, dim=1)


def pagerank_important_nodes(edge_index, num_nodes, num_reference_nodes, top_ratio=0.1):
    """Eq. 18-19: PageRank importance score, V_c = top-k nodes (k = top_ratio * num_reference_nodes)."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.t().tolist())
    scores = nx.pagerank(graph)

    k = max(1, int(round(top_ratio * num_reference_nodes)))
    top_nodes = sorted(scores, key=scores.get, reverse=True)[:k]
    return set(top_nodes)


def build_edge_confidence_prompt(text_i, text_j, dataset_name):
    domain = DATASET_DOMAINS.get(dataset_name, "a text-attributed graph")
    return (
        f"You are verifying a connection between two nodes in a graph about {domain}.\n"
        f'Node A: """{(text_i or "")[:400]}"""\n'
        f'Node B: """{(text_j or "")[:400]}"""\n\n'
        f"Should these two nodes be connected by an edge (e.g. because they are topically "
        f"related, cite each other, or belong to the same category)?\n"
        f'Respond with ONLY a JSON object: {{"confidence": <a number between 0 and 1>}}'
    )


def reconfigure_edges(edge_index, important_nodes, texts, generator, dataset_name, threshold=0.5):
    """Eq. 20-21: for each existing edge between two important nodes, ask the LLM for a
    confidence score and drop the edge if confidence <= threshold. Edges with at least
    one endpoint outside V_c are left untouched."""
    important_nodes = set(important_nodes)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()

    candidate_pairs = []
    seen = set()
    for s, d in zip(src, dst):
        if s in important_nodes and d in important_nodes and s != d:
            key = tuple(sorted((s, d)))
            if key not in seen:
                seen.add(key)
                candidate_pairs.append(key)

    if not candidate_pairs:
        return edge_index

    prompts = [build_edge_confidence_prompt(texts[s], texts[d], dataset_name) for s, d in candidate_pairs]
    raw_outputs = generator.generate(prompts, desc="LLM edge reconfiguration")

    confidences = {}
    for (s, d), raw in zip(candidate_pairs, raw_outputs):
        parsed = parse_json_object(raw, default={"confidence": 0.5})
        try:
            conf = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        confidences[(s, d)] = conf

    kept_columns = []
    for col, (s, d) in enumerate(zip(src, dst)):
        if s in important_nodes and d in important_nodes and s != d:
            key = tuple(sorted((s, d)))
            if confidences.get(key, 1.0) > threshold:
                kept_columns.append(col)
        else:
            kept_columns.append(col)

    kept_columns = torch.tensor(kept_columns, dtype=torch.long, device=edge_index.device)
    return edge_index[:, kept_columns]
