"""
Active node selection heuristics (E-LLaGNN paper Section 2.4).

Each heuristic returns all node ids ordered by priority for LLM enhancement; the
first `budget` of them are enhanced. None of them uses labels.

    pagerank    PageRank on the undirected graph, ascending (least central first)
    clustering  k-means (k = #classes) on the original text features; density of
                belonging 1 / (1 + d) with d the normalised distance to the node's
                centroid; lowest density first
    text_length word count after stop-word removal, shortest text first
    degree      node degree, ascending
    random      uniform random order (reference)
"""
import numpy as np

HEURISTICS = ["pagerank", "clustering", "text_length", "degree", "random"]


def _stop_words():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)


def pagerank_order(edge_index, num_nodes):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    pr = nx.pagerank(g)
    scores = np.array([pr[i] for i in range(num_nodes)])
    return np.argsort(scores, kind="stable")


def clustering_order(features, num_clusters, seed=0):
    from sklearn.cluster import KMeans

    x = features.float().cpu().numpy()
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit(x)
    dist = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
    # normalise distances within each cluster so clusters of different spread are comparable
    norm = np.zeros_like(dist)
    for c in range(num_clusters):
        m = km.labels_ == c
        if m.any():
            norm[m] = dist[m] / (dist[m].max() + 1e-12)
    density = 1.0 / (1.0 + norm)
    return np.argsort(density, kind="stable")


def text_length_order(raw_texts):
    stop = _stop_words()
    counts = np.array([sum(1 for w in (t or "").lower().split() if w.strip(".,;:!?()[]\"'") not in stop)
                       for t in raw_texts])
    return np.argsort(counts, kind="stable")


def degree_order(edge_index, num_nodes):
    deg = np.bincount(edge_index[1].cpu().numpy(), minlength=num_nodes)
    return np.argsort(deg, kind="stable")


def select_nodes(heuristic, budget, data, features, num_classes, seed=0):
    """Returns a sorted int64 array of the `budget` node ids chosen for LLM enhancement."""
    n = data.num_nodes
    budget = int(min(max(budget, 0), n))
    if budget == 0:
        return np.array([], dtype=np.int64)
    if heuristic == "pagerank":
        order = pagerank_order(data.edge_index, n)
    elif heuristic == "clustering":
        order = clustering_order(features, num_classes, seed=seed)
    elif heuristic == "text_length":
        order = text_length_order(data.raw_texts)
    elif heuristic == "degree":
        order = degree_order(data.edge_index, n)
    elif heuristic == "random":
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"Unknown heuristic '{heuristic}', choose from {HEURISTICS}")
    return np.sort(order[:budget]).astype(np.int64)
