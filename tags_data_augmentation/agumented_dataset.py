import torch
import json
import os
import re
from dataset.dataset_loader import load_dataset, get_dataset_path

_ABSTRACT_PREFIX_CHARS = 200


def _norm_text(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def _split_title_abstract(text):
    """(title, abstract) of a dataset raw text ('Title: ...\\nAbstract: ...' or free text)."""
    text = str(text)
    m = re.search(r'title\s*:\s*(.*?)(?:\n|abstract\s*:|$)', text, flags=re.IGNORECASE | re.DOTALL)
    title = m.group(1) if m else text.split('\n', 1)[0]
    m = re.search(r'abstract\s*:\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
    abstract = m.group(1) if m else (text.split('\n', 1)[1] if '\n' in text else '')
    return title, abstract


def drop_items_overlapping_dataset(dataset_name, augmented_data):
    """
    Remove augmented items that are copies of nodes already in the dataset.

    Augmented items are fetched from arXiv / PubMed / OpenAlex / ... by label name, so some of them can be
    the very papers that are val/test nodes. Added as labelled train nodes they would leak those test
    labels (and texts) into LLM fine-tuning and GNN training. An item is dropped when its normalised
    title, or the first characters of its normalised abstract, match any dataset node.
    """
    path = os.path.join(get_dataset_path(), f"{dataset_name}.pt")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found; cannot check augmented items for overlap with the dataset")
        return augmented_data
    raw_texts = torch.load(path, weights_only=False).raw_texts
    titles, abstracts = set(), set()
    for t in raw_texts:
        title, abstract = _split_title_abstract(t)
        if len(_norm_text(title)) >= 20:
            titles.add(_norm_text(title))
        a = _norm_text(abstract)[:_ABSTRACT_PREFIX_CHARS]
        if len(a) >= 50:
            abstracts.add(a)
    kept = [item for item in augmented_data
            if _norm_text(item.get('title', '')) not in titles
            and _norm_text(item.get('abstract', ''))[:_ABSTRACT_PREFIX_CHARS] not in abstracts]
    n_dropped = len(augmented_data) - len(kept)
    if n_dropped:
        print(f"[leakage guard] dropped {n_dropped}/{len(augmented_data)} augmented items that duplicate "
              f"existing {dataset_name} nodes (possible val/test nodes)")
    return kept


def load_augmented_texts(dataset_name, drop_dataset_overlap=True):
    """
    Load augmented texts from JSON file.

    Args:
        dataset_name: Name of the dataset
        drop_dataset_overlap: drop items that duplicate existing dataset nodes (see
            drop_items_overlapping_dataset). Every consumer (embedding cache, LLM training, graph
            construction) goes through this function, so they all see the same filtered list.

    Returns:
        List of augmented text data
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'augmented_data', f'{dataset_name}.json')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Augmented data file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        augmented_data = json.load(f)

    print(f"[OK] Loaded {len(augmented_data)} augmented texts from {json_path}")
    if drop_dataset_overlap:
        augmented_data = drop_items_overlapping_dataset(dataset_name, augmented_data)
    return augmented_data


def predict_edges_for_new_nodes_cosine_similarity(new_embeddings, existing_embeddings,
                                                  dataset, threshold=0.5, top_k=10):
    """
    Predict edges between new nodes and existing nodes using cosine similarity.

    Since new nodes are not in the graph yet, we can't use the GNN encoder.
    Instead, we use cosine similarity between embeddings to find the most similar nodes.

    Args:
        new_embeddings: Embeddings for new nodes [num_new_nodes, embedding_dim]
        existing_embeddings: Embeddings for existing nodes [num_existing_nodes, embedding_dim]
        dataset: Original dataset (for reference, not used in similarity-based approach)
        threshold: Similarity threshold for creating edges (0-1 for cosine similarity)
        top_k: Maximum number of edges per new node

    Returns:
        new_edges: List of (new_node_id, existing_node_id) pairs
    """
    device = new_embeddings.device
    existing_embeddings = existing_embeddings.to(device)

    num_new_nodes = new_embeddings.shape[0]
    num_existing_nodes = existing_embeddings.shape[0]

    print(f"\nPredicting edges using cosine similarity for {num_new_nodes} new nodes...")
    print(f"Similarity threshold: {threshold}, Top-K: {top_k}")

    new_edges = []

    # Normalize embeddings for cosine similarity
    new_embeddings_norm = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
    existing_embeddings_norm = torch.nn.functional.normalize(existing_embeddings, p=2, dim=1)

    with torch.no_grad():
        for new_node_idx in range(num_new_nodes):
            if (new_node_idx + 1) % 10 == 0:
                print(f"  Processed {new_node_idx + 1}/{num_new_nodes} new nodes")

            # Get embedding for this new node
            new_node_emb = new_embeddings_norm[new_node_idx:new_node_idx + 1]  # [1, emb_dim]

            # Compute cosine similarity with all existing nodes
            similarities = torch.mm(new_node_emb, existing_embeddings_norm.t()).squeeze()  # [num_existing_nodes]

            # Select edges based on threshold and top_k
            valid_edges = (similarities > threshold).nonzero(as_tuple=True)[0]

            if len(valid_edges) > 0:
                # If more than top_k, select the top_k with highest similarities
                if len(valid_edges) > top_k:
                    top_sims, top_indices = torch.topk(similarities[valid_edges], k=top_k)
                    selected_edges = valid_edges[top_indices]
                else:
                    selected_edges = valid_edges

                # Add edges (new_node_id in augmented graph, existing_node_id)
                for existing_node_idx in selected_edges:
                    new_edges.append((new_node_idx + num_existing_nodes, existing_node_idx.item()))
            else:
                # If no edges pass threshold, connect to top_k most similar nodes anyway
                top_sims, top_indices = torch.topk(similarities, k=min(top_k, num_existing_nodes))
                for existing_node_idx in top_indices:
                    new_edges.append((new_node_idx + num_existing_nodes, existing_node_idx.item()))

    print(f"[OK] Created {len(new_edges)} edges for new nodes")
    avg_edges_per_node = len(new_edges) / num_new_nodes if num_new_nodes > 0 else 0
    print(f"     Average edges per new node: {avg_edges_per_node:.2f}")
    return new_edges


def predict_edges_for_new_nodes_k_nearest_neighbors(new_embeddings, existing_embeddings,
                                                    dataset, threshold=0.5, top_k=10):
    """
    Predict edges between new nodes and existing nodes using k-nearest neighbors algorithm.

    Uses Euclidean distance (L2) to find the k nearest neighbors for each new node.
    Unlike cosine similarity, embeddings are NOT normalized.

    Args:
        new_embeddings: Embeddings for new nodes [num_new_nodes, embedding_dim]
        existing_embeddings: Embeddings for existing nodes [num_existing_nodes, embedding_dim]
        dataset: Original dataset (for reference, not used in k-NN approach)
        threshold: Maximum distance threshold for creating edges (lower distance = closer)
        top_k: Number of nearest neighbors (k) to connect to
    """
    device = new_embeddings.device
    existing_embeddings = existing_embeddings.to(device)

    num_new_nodes = new_embeddings.shape[0]
    num_existing_nodes = existing_embeddings.shape[0]

    print(f"\nPredicting edges using k-nearest neighbors for {num_new_nodes} new nodes...")
    print(f"Distance threshold: {threshold}, K (neighbors): {top_k}")
    print(f"Using Euclidean distance (embeddings NOT normalized)")

    new_edges = []

    with torch.no_grad():
        for new_node_idx in range(num_new_nodes):
            if (new_node_idx + 1) % 10 == 0:
                print(f"  Processed {new_node_idx + 1}/{num_new_nodes} new nodes")

            # Get embedding for this new node
            new_node_emb = new_embeddings[new_node_idx:new_node_idx + 1]  # [1, emb_dim]

            # Compute Euclidean distance (L2) with all existing nodes
            # Distance = sqrt(sum((new_emb - existing_emb)^2))
            # We compute squared distance first, then take sqrt
            diff = new_node_emb - existing_embeddings  # [num_existing_nodes, emb_dim]
            distances_squared = torch.sum(diff ** 2, dim=1)  # [num_existing_nodes]
            distances = torch.sqrt(distances_squared)  # [num_existing_nodes]

            # Find k nearest neighbors (smallest distances)
            k = min(top_k, num_existing_nodes)
            top_distances, top_indices = torch.topk(distances, k=k, largest=False)  # smallest k distances

            # Filter by threshold (keep only neighbors within threshold distance)
            valid_neighbors = (top_distances <= threshold).nonzero(as_tuple=True)[0]

            if len(valid_neighbors) > 0:
                # Use valid neighbors that pass threshold
                selected_indices = top_indices[valid_neighbors]
                for existing_node_idx in selected_indices:
                    new_edges.append((new_node_idx + num_existing_nodes, existing_node_idx.item()))
            else:
                # If no neighbors pass threshold, use top_k nearest neighbors anyway
                for existing_node_idx in top_indices:
                    new_edges.append((new_node_idx + num_existing_nodes, existing_node_idx.item()))

    print(f"[OK] Created {len(new_edges)} edges for new nodes using k-nearest neighbors")
    avg_edges_per_node = len(new_edges) / num_new_nodes if num_new_nodes > 0 else 0
    print(f"     Average edges per new node: {avg_edges_per_node:.2f}")
    return new_edges


def predict_edges_for_new_nodes_edge_predictor(new_embeddings, existing_embeddings,
                                               edge_predictor_model, dataset,
                                               threshold=0.5, top_k=10):
    """
    Predict edges between new nodes and existing nodes using trained edge predictor.

    Uses the original edge_index (without new nodes) to encode all nodes through the GNN encoder.
    New nodes will be encoded as isolated nodes (no edges to them in the original graph).
    Then uses the trained decoder on GNN-encoded embeddings.

    Args:
        new_embeddings: Raw embeddings for new nodes [num_new_nodes, embedding_dim]
        existing_embeddings: Raw embeddings for existing nodes [num_existing_nodes, embedding_dim]
        edge_predictor_model: Trained EdgePredictorModel (from edge_predictor.py)
        dataset: Original dataset (needed for edge_index)
        threshold: Score threshold for creating edges (after sigmoid, 0-1)
        top_k: Maximum number of edges per new node

    Returns:
        new_edges: List of (new_node_id, existing_node_id) pairs
    """
    device = new_embeddings.device
    existing_embeddings = existing_embeddings.to(device)
    edge_predictor_model = edge_predictor_model.to(device)
    edge_predictor_model.eval()

    num_new_nodes = new_embeddings.shape[0]
    num_existing_nodes = existing_embeddings.shape[0]

    print(f"\nPredicting edges using trained edge predictor for {num_new_nodes} new nodes...")
    print(f"Using original edge_index (new nodes will be isolated in GNN encoding)")
    print(f"Score threshold: {threshold}, Top-K: {top_k}")

    # Combine all embeddings: [existing_nodes, new_nodes]
    all_embeddings = torch.cat([existing_embeddings, new_embeddings], dim=0)  # [num_existing + num_new, emb_dim]

    # Get original edge_index (only has edges between existing nodes)
    original_edge_index = dataset.edge_index.to(device)  # [2, num_edges]

    # Get encoder and decoder from the edge predictor model
    encoder = edge_predictor_model.encoder
    decoder = edge_predictor_model.decoder

    new_edges = []

    with torch.no_grad():
        # Encode all nodes using GNN encoder with original edge_index
        # New nodes will be isolated (no edges to them), but still get encoded
        gnn_encoded_embeddings = encoder(all_embeddings, original_edge_index)  # [num_existing + num_new, hidden_dim]

        # Split encoded embeddings
        existing_gnn_emb = gnn_encoded_embeddings[:num_existing_nodes]  # [num_existing, hidden_dim]
        new_gnn_emb = gnn_encoded_embeddings[num_existing_nodes:]  # [num_new, hidden_dim]

        # For each new node, compute edge scores with all existing nodes
        for new_node_idx in range(num_new_nodes):
            if (new_node_idx + 1) % 10 == 0:
                print(f"  Processed {new_node_idx + 1}/{num_new_nodes} new nodes")

            # Get GNN-encoded embedding for this new node
            new_node_gnn_emb = new_gnn_emb[new_node_idx:new_node_idx + 1]  # [1, hidden_dim]

            # Expand to match all existing nodes
            new_node_gnn_emb_expanded = new_node_gnn_emb.expand(num_existing_nodes, -1)  # [num_existing, hidden_dim]

            # Compute edge scores using decoder with GNN-encoded embeddings
            edge_scores = decoder(new_node_gnn_emb_expanded, existing_gnn_emb)  # [num_existing, 1]
            edge_scores = edge_scores.squeeze()  # [num_existing]

            # Apply sigmoid to get probabilities
            edge_probs = torch.sigmoid(edge_scores)

            # Select edges based on threshold and top_k
            valid_edges = (edge_probs > threshold).nonzero(as_tuple=True)[0]

            if len(valid_edges) > 0:
                # If more than top_k, select the top_k with highest scores
                if len(valid_edges) > top_k:
                    top_scores, top_indices = torch.topk(edge_probs[valid_edges], k=top_k)
                    selected_edges = valid_edges[top_indices]
                else:
                    selected_edges = valid_edges

                # Add edges (new_node_id in augmented graph, existing_node_id)
                for existing_node_idx in selected_edges:
                    new_edges.append((new_node_idx + num_existing_nodes, existing_node_idx.item()))
            else:
                # If no edges pass threshold, connect to top_k most similar nodes anyway
                top_scores, top_indices = torch.topk(edge_probs, k=min(top_k, num_existing_nodes))
                for existing_node_idx in top_indices:
                    new_edges.append((new_node_idx + num_existing_nodes, existing_node_idx.item()))

    print(f"[OK] Created {len(new_edges)} edges for new nodes using edge predictor")
    avg_edges_per_node = len(new_edges) / num_new_nodes if num_new_nodes > 0 else 0
    print(f"     Average edges per new node: {avg_edges_per_node:.2f}")
    return new_edges


def hybrid_edge_prediction_approach(new_embeddings, existing_embeddings, dataset,
                                    used_cosine_similarity, used_k_nearest_neighbors, used_edge_predictor,
                                    k_cosine_similarity, k_knearest_neighbors, k_edge_predictor,
                                    threshold_cosine_similarity, threshold_k_nearest_neighbors,
                                    threshold_edge_predictor,
                                    top_k_cosine_similarity, top_k_knearest_neighbors, top_k_edge_predictor,
                                    edge_predictor_model=None):
    """
    Hybrid edge prediction approach that runs multiple methods and intersects their results.

    Runs each enabled approach (cosine similarity, k-nearest neighbors, edge predictor)
    and returns edges that are found by ALL enabled approaches (intersection).

    Args:
        new_embeddings: Embeddings for new nodes [num_new_nodes, embedding_dim]
        existing_embeddings: Embeddings for existing nodes [num_existing_nodes, embedding_dim]
        dataset: Original dataset
        used_cosine_similarity: Whether to use cosine similarity approach
        used_k_nearest_neighbors: Whether to use k-nearest neighbors approach
        used_edge_predictor: Whether to use edge predictor approach
        k_cosine_similarity: K parameter for cosine similarity (unused, kept for consistency)
        k_knearest_neighbors: K parameter for k-nearest neighbors (unused, kept for consistency)
        k_edge_predictor: K parameter for edge predictor (unused, kept for consistency)
        threshold_cosine_similarity: Threshold for cosine similarity
        threshold_k_nearest_neighbors: Threshold for k-nearest neighbors (max distance)
        threshold_edge_predictor: Threshold for edge predictor
        top_k_cosine_similarity: Top-K for cosine similarity
        top_k_knearest_neighbors: Top-K for k-nearest neighbors
        top_k_edge_predictor: Top-K for edge predictor
        edge_predictor_model: Trained edge predictor model (required if used_edge_predictor=True)

    Returns:
        new_edges: List of (new_node_id, existing_node_id) pairs (intersection of all enabled approaches)
    """
    print(f"\n{'=' * 80}")
    print("Hybrid Edge Prediction Approach")
    print(f"{'=' * 80}")
    print(f"Enabled methods:")
    print(f"  - Cosine Similarity: {used_cosine_similarity}")
    print(f"  - K-Nearest Neighbors: {used_k_nearest_neighbors}")
    print(f"  - Edge Predictor: {used_edge_predictor}")
    print(f"{'=' * 80}\n")

    all_edge_sets = []

    # Run cosine similarity approach if enabled
    if used_cosine_similarity:
        print("Running cosine similarity approach...")
        edges_cosine = predict_edges_for_new_nodes_cosine_similarity(
            new_embeddings=new_embeddings,
            existing_embeddings=existing_embeddings,
            dataset=dataset,
            threshold=threshold_cosine_similarity,
            top_k=top_k_cosine_similarity
        )
        edges_cosine_set = set(edges_cosine)
        all_edge_sets.append(edges_cosine_set)
        print(f"  Found {len(edges_cosine_set)} edges\n")

    # Run k-nearest neighbors approach if enabled
    if used_k_nearest_neighbors:
        print("Running k-nearest neighbors approach...")
        edges_knn = predict_edges_for_new_nodes_k_nearest_neighbors(
            new_embeddings=new_embeddings,
            existing_embeddings=existing_embeddings,
            dataset=dataset,
            threshold=threshold_k_nearest_neighbors,
            top_k=top_k_knearest_neighbors
        )
        edges_knn_set = set(edges_knn)
        all_edge_sets.append(edges_knn_set)
        print(f"  Found {len(edges_knn_set)} edges\n")

    # Run edge predictor approach if enabled
    if used_edge_predictor:
        if edge_predictor_model is None:
            raise ValueError("edge_predictor_model is required when used_edge_predictor=True")
        print("Running edge predictor approach...")
        edges_predictor = predict_edges_for_new_nodes_edge_predictor(
            new_embeddings=new_embeddings,
            existing_embeddings=existing_embeddings,
            edge_predictor_model=edge_predictor_model,
            dataset=dataset,
            threshold=threshold_edge_predictor,
            top_k=top_k_edge_predictor
        )
        edges_predictor_set = set(edges_predictor)
        all_edge_sets.append(edges_predictor_set)
        print(f"  Found {len(edges_predictor_set)} edges\n")

    # Intersect all edge sets (edges that appear in ALL enabled approaches)
    if len(all_edge_sets) == 0:
        print("[WARNING] No approaches enabled! Returning empty edge list.")
        return []

    # Start with the first set and intersect with all others
    intersection_edges = all_edge_sets[0]
    for edge_set in all_edge_sets[1:]:
        intersection_edges = intersection_edges.intersection(edge_set)

    # Convert back to list
    final_edges = list(intersection_edges)

    print(f"{'=' * 80}")
    print(f"Hybrid Edge Prediction Results:")
    print(f"  Total edges from intersection: {len(final_edges)}")
    for i, edge_set in enumerate(all_edge_sets):
        method_name = ["Cosine Similarity", "K-Nearest Neighbors", "Edge Predictor"][i]
        print(f"  {method_name}: {len(edge_set)} edges")
    print(f"{'=' * 80}\n")

    return final_edges


def create_augmented_dataset(cfg, augmented_cache, threshold=0.5, top_k=10, init_weight_approach="pissa",
                             used_cosine_similarity=True, used_k_nearest_neighbors=True, used_edge_predictor=False,
                             k_cosine_similarity=5, k_knearest_neighbors=5, k_edge_predictor=5,
                             threshold_cosine_similarity=0.5, threshold_k_nearest_neighbors=0.5,
                             threshold_edge_predictor=0.5,
                             top_k_cosine_similarity=10, top_k_knearest_neighbors=10, top_k_edge_predictor=10,
                             edge_predictor_model=None, supervised=False, base_embeddings_from_augmented_cache=None):
    """
    Create augmented dataset by adding new nodes and edges to the original dataset.

    Uses hybrid edge prediction approach to connect new nodes to the existing graph.
    Can use cosine similarity, k-nearest neighbors, and/or edge predictor methods.

    Args:
        cfg: Configuration object
        augmented_cache: Dictionary with augmented embeddings and labels
        threshold: Legacy parameter (kept for backward compatibility, use threshold_cosine_similarity instead)
        top_k: Legacy parameter (kept for backward compatibility, use top_k_cosine_similarity instead)
        init_weight_approach: Initialization weight approach ('pissa', 'orthogonal', 'guassian', 'loftq', 'eva')
        used_cosine_similarity: Whether to use cosine similarity approach
        used_k_nearest_neighbors: Whether to use k-nearest neighbors approach
        used_edge_predictor: Whether to use edge predictor approach
        k_cosine_similarity: K parameter for cosine similarity (unused, kept for consistency)
        k_knearest_neighbors: K parameter for k-nearest neighbors (unused, kept for consistency)
        k_edge_predictor: K parameter for edge predictor (unused, kept for consistency)
        threshold_cosine_similarity: Threshold for cosine similarity (0-1)
        threshold_k_nearest_neighbors: Threshold for k-nearest neighbors (max distance)
        threshold_edge_predictor: Threshold for edge predictor (0-1, after sigmoid)
        top_k_cosine_similarity: Top-K for cosine similarity
        top_k_knearest_neighbors: Top-K for k-nearest neighbors
        top_k_edge_predictor: Top-K for edge predictor
        edge_predictor_model: Trained edge predictor model (required if used_edge_predictor=True)
        supervised: Whether to use supervised data for get_init_dataset_for_gnn
        base_embeddings_from_augmented_cache: Optional [N, H] tensor of base node embeddings from
            augmented-adapter cache; when provided, used instead of get_embedding_from_data(...)

    Returns:
        augmented_dataset: Modified dataset with new nodes
        augmented_embeddings: Combined embeddings (original + augmented)
    """
    print("\n" + "=" * 80)
    print("Creating Augmented Dataset")
    print("=" * 80)

    # Load original dataset
    dataset = load_dataset(cfg)
    original_num_nodes = dataset.y.shape[0]

    # Get device from dataset (dataset might be on CUDA)
    device = 'cuda'

    # Load augmented data
    augmented_texts = load_augmented_texts(cfg.dataset.name)

    # Extract augmented embeddings and labels
    new_embeddings = augmented_cache['embeddings']  # [num_new_nodes, emb_dim]
    new_labels = augmented_cache['labels']  # [num_new_nodes]
    num_new_nodes = new_embeddings.shape[0]
    if num_new_nodes != len(augmented_texts):
        raise ValueError(
            f"Augmented cache has {num_new_nodes} nodes but {len(augmented_texts)} augmented texts remain after "
            f"removing items that duplicate dataset nodes; regenerate the augmented embedding cache.")

    # Move new tensors to the same device as dataset
    new_embeddings = new_embeddings.to(device)
    new_labels = new_labels.to(device)

    print(f"\nOriginal dataset: {original_num_nodes} nodes")
    print(f"New nodes to add: {num_new_nodes} nodes")
    print(f"Device: {device}")

    # Create augmented dataset
    augmented_dataset = dataset.clone()
    # original nodes come first; load_graph_dataset splits only these and keeps the new nodes train-only
    augmented_dataset.num_base_nodes = original_num_nodes

    # Update y (labels) - ensure both are on same device
    augmented_dataset.y = torch.cat([dataset.y, new_labels], dim=0)

    # Update raw_texts
    new_raw_texts = [f"Title: {item['title']}\nAbstract: {item['abstract']}"
                     for item in augmented_texts]
    augmented_dataset.raw_texts = dataset.raw_texts + new_raw_texts

    # Predict edges for new nodes using hybrid edge prediction approach
    print("\nConnecting new nodes to existing graph using hybrid edge prediction approach...")

    # Load the embeddings used for the original dataset (or use base embeddings from augmented-adapter cache)
    if base_embeddings_from_augmented_cache is not None:
        existing_embeddings = base_embeddings_from_augmented_cache
    else:
        from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
        data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg, supervised=supervised)
        existing_embeddings = None
        if init_weight_approach == "pissa":
            existing_embeddings = get_embedding_from_data(data_pissa)
        elif init_weight_approach == "orthogonal":
            existing_embeddings = get_embedding_from_data(data_orthogonal)
        elif init_weight_approach == "guassian":
            existing_embeddings = get_embedding_from_data(data_guassian)
        elif init_weight_approach == "loftq":
            existing_embeddings = get_embedding_from_data(data_loftq)
        elif init_weight_approach == "eva":
            existing_embeddings = get_embedding_from_data(data_eva)

    new_edges = hybrid_edge_prediction_approach(
        new_embeddings=new_embeddings,
        existing_embeddings=existing_embeddings,
        dataset=dataset,
        used_cosine_similarity=used_cosine_similarity,
        used_k_nearest_neighbors=used_k_nearest_neighbors,
        used_edge_predictor=used_edge_predictor,
        k_cosine_similarity=k_cosine_similarity,
        k_knearest_neighbors=k_knearest_neighbors,
        k_edge_predictor=k_edge_predictor,
        threshold_cosine_similarity=threshold_cosine_similarity,
        threshold_k_nearest_neighbors=threshold_k_nearest_neighbors,
        threshold_edge_predictor=threshold_edge_predictor,
        top_k_cosine_similarity=top_k_cosine_similarity,
        top_k_knearest_neighbors=top_k_knearest_neighbors,
        top_k_edge_predictor=top_k_edge_predictor,
        edge_predictor_model=edge_predictor_model,
    )

    # Update edge_index
    if len(new_edges) > 0:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).t()  # [2, num_new_edges]
        # Add reverse edges for undirected graph
        reverse_edges = torch.stack([new_edge_tensor[1], new_edge_tensor[0]], dim=0)
        all_new_edges = torch.cat([new_edge_tensor, reverse_edges], dim=1)
        augmented_dataset.edge_index = torch.cat([dataset.edge_index, all_new_edges], dim=1)
    else:
        print("[WARNING] No edges created for new nodes!")

    # Update masks - new nodes should be in train mask
    new_train_mask = torch.zeros(num_new_nodes, dtype=torch.bool, device=device)
    new_train_mask[:] = True  # All augmented nodes participate in training
    augmented_dataset.train_mask = torch.cat([dataset.train_mask, new_train_mask], dim=0)

    new_val_mask = torch.zeros(num_new_nodes, dtype=torch.bool, device=device)
    augmented_dataset.val_mask = torch.cat([dataset.val_mask, new_val_mask], dim=0)

    new_test_mask = torch.zeros(num_new_nodes, dtype=torch.bool, device=device)
    augmented_dataset.test_mask = torch.cat([dataset.test_mask, new_test_mask], dim=0)

    # Update few-shot masks
    for mask_name in ['one_shot_train', 'three_shot_train', 'five_shot_train',
                      'one_shot_val', 'three_shot_val', 'five_shot_val',
                      'one_shot_test', 'three_shot_test', 'five_shot_test']:
        if hasattr(dataset, mask_name):
            new_mask = torch.zeros(num_new_nodes, dtype=torch.bool, device=device)
            setattr(augmented_dataset, mask_name,
                    torch.cat([getattr(dataset, mask_name), new_mask], dim=0))

    existing_embeddings = existing_embeddings.to(device)
    augmented_embeddings = torch.cat([existing_embeddings, new_embeddings], dim=0)

    print(f"\n[OK] Augmented Dataset Created:")
    print(f"  Total nodes: {augmented_dataset.y.shape[0]}")
    print(f"  Total edges: {augmented_dataset.edge_index.shape[1]}")
    print(f"  Train nodes: {augmented_dataset.train_mask.sum().item()}")
    print(f"  Val nodes: {augmented_dataset.val_mask.sum().item()}")
    print(f"  Test nodes: {augmented_dataset.test_mask.sum().item()}")
    print(f"  Embeddings shape: {augmented_embeddings.shape}")
    print("=" * 80)

    return augmented_dataset, augmented_embeddings