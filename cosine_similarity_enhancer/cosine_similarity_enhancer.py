import torch
import torch.nn.functional as F
import numpy as np

from gnns.gnn_mtrainer import gnn_train_and_report_with_modified_dataset


class SemanticSimilarityEnhancer:
  """
  Graph structural enhancement by removing edges whose endpoint node embeddings
  have cosine similarity below a threshold.
  """

  DEFAULT_THRESHOLD_RANGE = np.arange(0.1, 0.95, 0.02)

  @staticmethod
  def embeddings_to_dict(embeddings: torch.Tensor) -> dict:
    """Convert a node embedding matrix into a node-id -> embedding mapping."""
    node_embeddings = {}
    print(f"Loading precomputed embeddings for {embeddings.shape[0]} nodes...")
    for node_id in range(embeddings.shape[0]):
      node_embeddings[node_id] = embeddings[node_id]
    return node_embeddings

  @staticmethod
  def compute_edge_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two node embedding vectors."""
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

  @staticmethod
  def refine_graph_by_cosine_similarity(
    dataset,
    embeddings: torch.Tensor,
    similarity_threshold: float,
    node_subset=None,
  ):
    """
    Remove edges with cosine similarity below the given threshold.

    Args:
      dataset: Original graph dataset with edge_index.
      embeddings: Node embeddings tensor of shape [num_nodes, dim].
      similarity_threshold: Edges with similarity < threshold are removed.
      node_subset: Optional set/list of node ids. When provided, only edges
        incident to at least one node in this subset are considered for removal.
        When None, all edges are checked (as in Proposed Method II).

    Returns:
      modified_dataset: Dataset clone with refined edge_index.
      stats: Edge removal statistics.
    """
    if isinstance(dataset, tuple):
      dataset = dataset[0]

    edge_index = dataset.edge_index
    node_subset = set(node_subset) if node_subset is not None else None

    edges_to_check = []
    for edge_idx in range(edge_index.shape[1]):
      node1 = edge_index[0, edge_idx].item()
      node2 = edge_index[1, edge_idx].item()
      if node_subset is None or node1 in node_subset or node2 in node_subset:
        edges_to_check.append((edge_idx, node1, node2))

    print(f"  Edges to evaluate: {len(edges_to_check)}")

    edges_to_remove = set()
    edge_similarities = []

    for edge_idx, node1, node2 in edges_to_check:
      emb1 = embeddings[node1]
      emb2 = embeddings[node2]
      similarity = SemanticSimilarityEnhancer.compute_edge_cosine_similarity(emb1, emb2)
      edge_similarities.append(similarity)
      if similarity < similarity_threshold:
        edges_to_remove.add(edge_idx)

    edges_to_keep = []
    for edge_idx in range(edge_index.shape[1]):
      if edge_idx not in edges_to_remove:
        edges_to_keep.append(
          [edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()]
        )

    if edges_to_keep:
      modified_edge_index = torch.tensor(edges_to_keep, dtype=torch.long).t()
    else:
      modified_edge_index = torch.empty((2, 0), dtype=torch.long)

    modified_dataset = dataset.clone()
    modified_dataset.edge_index = modified_edge_index

    removed_sims = [
      edge_similarities[i]
      for i, (edge_idx, _, _) in enumerate(edges_to_check)
      if edge_idx in edges_to_remove
    ]
    kept_sims = [
      edge_similarities[i]
      for i, (edge_idx, _, _) in enumerate(edges_to_check)
      if edge_idx not in edges_to_remove
    ]

    stats = {
      'threshold': similarity_threshold,
      'total_checked': len(edges_to_check),
      'removed': len(edges_to_remove),
      'kept': len(edges_to_check) - len(edges_to_remove),
      'removed_sims': removed_sims,
      'kept_sims': kept_sims,
      'original_edges': edge_index.shape[1],
      'modified_edges': modified_edge_index.shape[1],
    }

    print(
      f"  Edges removed: {stats['removed']} "
      f"(similarity < {similarity_threshold})"
    )
    print(
      f"  Original edges: {stats['original_edges']} "
      f"-> Modified edges: {stats['modified_edges']}"
    )

    return modified_dataset, stats

  def tune_similarity_threshold(
    self,
    dataset,
    embeddings: torch.Tensor,
    dataset_name: str,
    threshold_range=None,
    node_subset=None,
    supervised: bool = True,
    reporter=None,
    title: str = "",
  ):
    """
    Search over cosine similarity thresholds and pick the best one by
    validation accuracy on a GNN trained on the refined graph.
    """
    if threshold_range is None:
      threshold_range = self.DEFAULT_THRESHOLD_RANGE

    best_val_acc = -1.0
    best_result = {
      'threshold': None,
      'modified_dataset': None,
      'stats': None,
      'results': None,
      'model': None,
    }

    print(
      f"Searching for best similarity threshold "
      f"(testing {len(threshold_range)} thresholds)..."
    )

    for threshold in threshold_range:
      print(f"\n--- Threshold: {threshold:.2f} ---")
      modified_dataset, stats = self.refine_graph_by_cosine_similarity(
        dataset=dataset,
        embeddings=embeddings,
        similarity_threshold=float(threshold),
        node_subset=node_subset,
      )
      results, model = gnn_train_and_report_with_modified_dataset(
        modified_dataset=modified_dataset,
        embedding=embeddings,
        dataset_name=dataset_name,
        supervised=supervised,
      )

      if results['val_acc'] > best_val_acc:
        best_val_acc = results['val_acc']
        best_result = {
          'threshold': float(threshold),
          'modified_dataset': modified_dataset,
          'stats': stats,
          'results': results,
          'model': model,
        }

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best threshold: {best_result['threshold']}")

    if reporter is not None and best_result['results'] is not None:
      report_title = (
        f"Best Similarity Threshold Search - {title}"
        if title
        else "Best Similarity Threshold Search"
      )
      best_stats = best_result['stats']
      best_results = best_result['results']
      edge_reduction = (
        (best_stats['original_edges'] - best_stats['modified_edges'])
        / best_stats['original_edges']
        * 100
      )
      report_text = f"""
Best Similarity Threshold Search Results:
  • Best Threshold: {best_result['threshold']:.4f}
  • Best Test Accuracy: {best_results['test_acc']:.4f} ({best_results['test_acc']:.2%})
  • Best Validation Accuracy: {best_results['val_acc']:.4f} ({best_results['val_acc']:.2%})
  • Best Test F1 (Macro): {best_results['test_f1']:.4f}
  • Best Test F1 (Weighted): {best_results['test_weight_f1']:.4f}
  • Best Validation F1 (Macro): {best_results['val_f1']:.4f}
  • Best Validation F1 (Weighted): {best_results['val_weight_f1']:.4f}

Edge Removal Statistics:
  • Total Edges Checked: {best_stats['total_checked']}
  • Edges Removed: {best_stats['removed']}
  • Edges Kept: {best_stats['kept']}
  • Original Edges: {best_stats['original_edges']}
  • Modified Edges: {best_stats['modified_edges']}
  • Edge Reduction: {edge_reduction:.2f}%
"""
      reporter.report(report_title, report_text)

    return best_result
