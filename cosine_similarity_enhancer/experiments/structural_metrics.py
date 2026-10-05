"""
Structural graph metrics for homophily analysis before/after cosine-similarity refinement.

Metrics (per proposed experiment #3):
  - edge homophily ratio (same-label edges / all edges)
  - average cosine similarity on retained vs. removed edges
  - percentage of edges removed at the chosen threshold
  - ESNR (cosine-based signal-to-noise ratio, GPS-inspired)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_dataset(dataset):
  if isinstance(dataset, tuple):
    return dataset[0]
  return dataset


def _resolve_labels(dataset) -> torch.Tensor:
  dataset = _resolve_dataset(dataset)
  labels = dataset.y
  if labels.dim() > 1:
    labels = labels.squeeze(-1)
  return labels.long()


def _resolve_edge_index(dataset) -> torch.Tensor:
  dataset = _resolve_dataset(dataset)
  return dataset.edge_index


def _get_evaluated_edge_mask(
  edge_index: torch.Tensor,
  node_subset: Optional[set[int]] = None,
) -> torch.Tensor:
  if node_subset is None:
    return torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

  node_subset_tensor = torch.tensor(
    sorted(node_subset),
    dtype=edge_index.dtype,
    device=edge_index.device,
  )
  src_in_subset = torch.isin(edge_index[0], node_subset_tensor)
  dst_in_subset = torch.isin(edge_index[1], node_subset_tensor)
  return src_in_subset | dst_in_subset


def compute_edge_cosine_similarities(
  edge_index: torch.Tensor,
  embeddings: torch.Tensor,
) -> torch.Tensor:
  """Vectorized cosine similarity for every directed edge."""
  if embeddings.device != edge_index.device:
    embeddings = embeddings.to(edge_index.device)

  emb_norm = F.normalize(embeddings.float(), dim=1)
  src_emb = emb_norm[edge_index[0]]
  dst_emb = emb_norm[edge_index[1]]
  return (src_emb * dst_emb).sum(dim=1)


def compute_edge_homophily_ratio(
  edge_index: torch.Tensor,
  labels: torch.Tensor,
  edge_mask: Optional[torch.Tensor] = None,
) -> float:
  """Fraction of edges whose endpoints share the same class label."""
  device = edge_index.device
  labels = labels.to(device)
  if edge_mask is None:
    edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
  else:
    edge_mask = edge_mask.to(device)

  src = edge_index[0, edge_mask]
  dst = edge_index[1, edge_mask]
  if src.numel() == 0:
    return float('nan')

  same_label = labels[src] == labels[dst]
  return same_label.float().mean().item()


def compute_homophily_esnr(homophily_ratio: float, num_classes: int) -> float:
  """
  Label-homophily ESNR normalized above the random baseline 1/C.

  ESNR_h = (h - 1/C) / (1 - 1/C)
  """
  if num_classes <= 1:
    return float('nan')

  random_baseline = 1.0 / num_classes
  denominator = 1.0 - random_baseline
  if denominator <= 0:
    return float('nan')
  return (homophily_ratio - random_baseline) / denominator


def compute_cosine_esnr(
  edge_index: torch.Tensor,
  embeddings: torch.Tensor,
  labels: torch.Tensor,
  edge_mask: Optional[torch.Tensor] = None,
) -> float:
  """
  Cosine-based edge signal-to-noise ratio (GPS-inspired).

  ESNR_cos = (mu_same - mu_diff) / (sigma_same + sigma_diff + eps)
  """
  device = edge_index.device
  labels = labels.to(device)
  if embeddings.device != device:
    embeddings = embeddings.to(device)

  if edge_mask is None:
    edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
  else:
    edge_mask = edge_mask.to(device)

  src = edge_index[0, edge_mask]
  dst = edge_index[1, edge_mask]
  if src.numel() == 0:
    return float('nan')

  sims = compute_edge_cosine_similarities(edge_index[:, edge_mask], embeddings)
  same_label = labels[src] == labels[dst]

  same_sims = sims[same_label]
  diff_sims = sims[~same_label]
  if same_sims.numel() == 0 or diff_sims.numel() == 0:
    return float('nan')

  signal = same_sims.mean() - diff_sims.mean()
  noise = same_sims.std(unbiased=False) + diff_sims.std(unbiased=False) + 1e-8
  return (signal / noise).item()


def compute_graph_structural_metrics(
  dataset,
  embeddings: torch.Tensor,
  similarity_threshold: Optional[float] = None,
  node_subset: Optional[set[int] | list[int]] = None,
) -> dict:
  """
  Compute structural metrics for the current graph (or after virtual pruning).

  When ``similarity_threshold`` is provided, edges below the threshold among the
  evaluated subset are treated as removed for similarity statistics. The returned
  ``edge_homophily_ratio`` always reflects the graph's current ``edge_index``.
  """
  dataset = _resolve_dataset(dataset)
  edge_index = _resolve_edge_index(dataset)
  labels = _resolve_labels(dataset)
  device = edge_index.device
  if embeddings.device != device:
    embeddings = embeddings.to(device)

  subset = set(node_subset) if node_subset is not None else None
  edge_mask = _get_evaluated_edge_mask(edge_index, subset)
  evaluated_edges = edge_index[:, edge_mask]

  homophily = compute_edge_homophily_ratio(edge_index, labels, edge_mask=edge_mask)
  num_classes = int(labels.max().item()) + 1
  homophily_esnr = compute_homophily_esnr(homophily, num_classes)
  cosine_esnr = compute_cosine_esnr(edge_index, embeddings, labels, edge_mask=edge_mask)

  if evaluated_edges.shape[1] == 0:
    return {
      'num_edges': 0,
      'num_evaluated_edges': 0,
      'edge_homophily_ratio': homophily,
      'homophily_esnr': homophily_esnr,
      'cosine_esnr': cosine_esnr,
      'avg_cosine_sim_all': float('nan'),
      'avg_cosine_sim_same_label': float('nan'),
      'avg_cosine_sim_diff_label': float('nan'),
      'avg_cosine_sim_retained': float('nan'),
      'avg_cosine_sim_removed': float('nan'),
      'pct_edges_removed': 0.0,
      'num_edges_removed': 0,
      'num_edges_retained': 0,
      'similarity_threshold': similarity_threshold,
    }

  sims = compute_edge_cosine_similarities(evaluated_edges, embeddings)
  src = evaluated_edges[0]
  dst = evaluated_edges[1]
  labels_on_device = labels.to(device)
  same_label = labels_on_device[src] == labels_on_device[dst]

  metrics = {
    'num_edges': int(edge_index.shape[1]),
    'num_evaluated_edges': int(evaluated_edges.shape[1]),
    'edge_homophily_ratio': homophily,
    'homophily_esnr': homophily_esnr,
    'cosine_esnr': cosine_esnr,
    'avg_cosine_sim_all': sims.mean().item(),
    'avg_cosine_sim_same_label': (
      sims[same_label].mean().item() if same_label.any() else float('nan')
    ),
    'avg_cosine_sim_diff_label': (
      sims[~same_label].mean().item() if (~same_label).any() else float('nan')
    ),
    'similarity_threshold': similarity_threshold,
  }

  if similarity_threshold is None:
    metrics.update({
      'avg_cosine_sim_retained': metrics['avg_cosine_sim_all'],
      'avg_cosine_sim_removed': float('nan'),
      'pct_edges_removed': 0.0,
      'num_edges_removed': 0,
      'num_edges_retained': int(evaluated_edges.shape[1]),
    })
    return metrics

  removed_mask = sims < similarity_threshold
  retained_mask = ~removed_mask
  num_removed = int(removed_mask.sum().item())
  num_retained = int(retained_mask.sum().item())

  metrics.update({
    'avg_cosine_sim_retained': (
      sims[retained_mask].mean().item() if num_retained > 0 else float('nan')
    ),
    'avg_cosine_sim_removed': (
      sims[removed_mask].mean().item() if num_removed > 0 else float('nan')
    ),
    'pct_edges_removed': (
      100.0 * num_removed / evaluated_edges.shape[1]
      if evaluated_edges.shape[1] > 0
      else 0.0
    ),
    'num_edges_removed': num_removed,
    'num_edges_retained': num_retained,
  })
  return metrics


def compare_before_after_metrics(
  original_dataset,
  refined_dataset,
  embeddings: torch.Tensor,
  similarity_threshold: float,
  node_subset: Optional[set[int] | list[int]] = None,
) -> dict:
  """Return before/after metric dictionaries and their deltas."""
  before = compute_graph_structural_metrics(
    original_dataset,
    embeddings,
    similarity_threshold=None,
    node_subset=node_subset,
  )
  after = compute_graph_structural_metrics(
    refined_dataset,
    embeddings,
    similarity_threshold=similarity_threshold,
    node_subset=node_subset,
  )

  removal_stats = compute_graph_structural_metrics(
    original_dataset,
    embeddings,
    similarity_threshold=similarity_threshold,
    node_subset=node_subset,
  )

  return {
    'before': before,
    'after': after,
    'removal_stats': removal_stats,
    'delta_homophily': after['edge_homophily_ratio'] - before['edge_homophily_ratio'],
    'delta_homophily_esnr': after['homophily_esnr'] - before['homophily_esnr'],
    'delta_cosine_esnr': after['cosine_esnr'] - before['cosine_esnr'],
    'pct_edges_removed': removal_stats['pct_edges_removed'],
    'similarity_threshold': similarity_threshold,
  }
