"""
Semantic related retrieval (SKETCH paper Section 3.1, Figure 1).

Every node text is embedded with a sentence-transformer, and for each anchor node
the most cosine-similar node texts in the *whole* graph are retrieved with FAISS
(exact inner-product search on L2-normalised vectors = cosine similarity). Falls
back to a chunked torch matmul when faiss is not installed -- the result is the
same exact top-k, only slower on very large graphs.
"""
import os

import numpy as np
import torch


DEFAULT_SENTENCE_ENCODER = "sentence-transformers/all-mpnet-base-v2"


def get_cache_dir():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _encoder_tag(encoder_name):
    return encoder_name.replace("/", "__")


def get_or_build_sentence_embeddings(dataset_name, raw_texts, encoder_name=DEFAULT_SENTENCE_ENCODER,
                                     device=None, batch_size=64, force_rebuild=False):
    """[N, d] L2-normalised float32 sentence embeddings, cached under SKETCH/cache/."""
    cache_path = os.path.join(get_cache_dir(), f"{dataset_name}_{_encoder_tag(encoder_name)}_sem_emb.pt")
    if os.path.exists(cache_path) and not force_rebuild:
        return torch.load(cache_path, weights_only=False)

    from sentence_transformers import SentenceTransformer

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer(encoder_name, device=device)
    texts = [t if t else "empty" for t in raw_texts]
    embeddings = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_tensor=True, normalize_embeddings=True,
    ).float().cpu()
    torch.save(embeddings, cache_path)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def semantic_topk(embeddings, top_k, chunk_size=4096):
    """
    For every node, the `top_k` most cosine-similar *other* nodes.
    Returns (indices [N, top_k] int64, scores [N, top_k] float32), best first.
    """
    emb = embeddings.float().cpu().numpy().astype(np.float32)
    num_nodes = emb.shape[0]
    k = min(top_k + 1, num_nodes)  # +1 because every node retrieves itself first

    try:
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        scores, indices = index.search(emb, k)
    except ImportError:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        emb_t = torch.from_numpy(emb).to(device)
        all_scores, all_indices = [], []
        for start in range(0, num_nodes, chunk_size):
            sims = emb_t[start:start + chunk_size] @ emb_t.T
            s, i = sims.topk(k, dim=1)
            all_scores.append(s.cpu())
            all_indices.append(i.cpu())
        scores = torch.cat(all_scores).numpy()
        indices = torch.cat(all_indices).numpy()

    # Drop the anchor itself (normally at position 0, but duplicate texts can tie)
    out_idx = np.empty((num_nodes, k - 1), dtype=np.int64)
    out_score = np.empty((num_nodes, k - 1), dtype=np.float32)
    for v in range(num_nodes):
        keep = indices[v] != v
        idx_v, score_v = indices[v][keep][:k - 1], scores[v][keep][:k - 1]
        out_idx[v, :len(idx_v)] = idx_v
        out_score[v, :len(idx_v)] = score_v
        if len(idx_v) < k - 1:  # anchor was not among the results -> one extra slot
            out_idx[v, len(idx_v):] = -1
            out_score[v, len(idx_v):] = -np.inf
    return out_idx, out_score
