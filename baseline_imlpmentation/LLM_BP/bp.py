"""
Principle II of LLM-BP (Wang et al., ICML 2025): belief propagation over the graph MRF.

    P(Y | X) ∝ prod_i phi_i(y_i) prod_(i,j) psi_ij(y_i, y_j)

- Node potential (Eq. 5): p_i^(0)(y) = softmax_k(cos(h_i, q_k) / tau).
- Edge potential (Eq. 6): psi(y_i, y_j) = r if y_i == y_j else 1 - r, where r is the
  graph's homophily level. The paper asks an LLM to estimate r from sampled edges
  because it has no labels; with a supervised split we measure r on training edges.
- Loopy BP (Eq. 4) for L layers, or its single-layer linear approximation (Eq. 7).

Supervised extensions (not in the paper, which is zero/few-shot):
- "matrix" edge potential: a full class-compatibility matrix H[a, b] estimated from
  training edges -- the "more complex psi" the paper mentions after Eq. 6.
- Training nodes are observed evidence: their potential is clamped to (smoothed)
  one-hot labels and kept clamped during propagation.
"""
import torch
import torch.nn.functional as F


def class_embeddings(emb, y, train_mask, num_classes):
    """q_k = mean embedding of the labeled nodes of class k (paper's few-shot rule, Appendix B.2)."""
    q = torch.zeros(num_classes, emb.size(1), device=emb.device)
    counts = torch.zeros(num_classes, device=emb.device)
    q.index_add_(0, y[train_mask], emb[train_mask])
    counts.index_add_(0, y[train_mask], torch.ones_like(y[train_mask], dtype=torch.float))
    return q / counts.clamp(min=1).unsqueeze(1)


def log_node_prior(emb, q, tau):
    """Eq. 5: log softmax of cosine similarities to the class embeddings, temperature tau."""
    cos = F.normalize(emb, dim=-1) @ F.normalize(q, dim=-1).T
    return F.log_softmax(cos / tau, dim=-1)


def clamp_labeled(log_prior, y, mask, smoothing=1e-3):
    """Replace the potential of observed nodes by a smoothed one-hot of their label."""
    num_classes = log_prior.size(1)
    target = torch.full((int(mask.sum()), num_classes), smoothing / (num_classes - 1), device=log_prior.device)
    target.scatter_(1, y[mask].unsqueeze(1), 1.0 - smoothing)
    log_prior = log_prior.clone()
    log_prior[mask] = target.log()
    return log_prior


def prepare_edges(edge_index, num_nodes):
    """Drop self loops and return (src, dst, rev) where rev[e] is the index of the reverse edge of e."""
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    keys = src * num_nodes + dst
    keys, perm = keys.sort()
    src, dst = src[perm], dst[perm]
    rev = torch.searchsorted(keys, dst * num_nodes + src)
    rev = rev.clamp(max=keys.numel() - 1)
    if not torch.equal(keys[rev], dst * num_nodes + src):
        raise ValueError("Edge list is not symmetric; LLM-BP expects an undirected graph.")
    return src, dst, rev


def train_edge_statistics(src, dst, y, train_mask, num_classes):
    """Class co-occurrence counts over edges whose endpoints are both training nodes."""
    both = train_mask[src] & train_mask[dst]
    counts = torch.zeros(num_classes, num_classes, device=y.device)
    counts.index_put_((y[src[both]], y[dst[both]]), torch.ones(int(both.sum()), device=y.device), accumulate=True)
    return counts


def homophily_from_counts(counts):
    total = counts.sum()
    if total == 0:
        raise ValueError("No edge has both endpoints in the training set; cannot estimate homophily r.")
    return float(counts.diag().sum() / total)


def scalar_edge_potential(r, num_classes, eps=1e-4):
    """Eq. 6: psi = r on the diagonal and 1 - r elsewhere."""
    r = min(max(r, eps), 1 - eps)
    psi = torch.full((num_classes, num_classes), 1.0 - r)
    psi.fill_diagonal_(r)
    return psi


def matrix_edge_potential(counts, alpha=1.0):
    """Row-normalised class-compatibility matrix with Laplace smoothing alpha."""
    counts = counts + alpha
    return counts / counts.sum(dim=1, keepdim=True)


@torch.no_grad()
def loopy_bp(log_prior, src, dst, rev, psi, num_layers, clamp_mask=None):
    """
    Eq. 4 in log space. Message on edge e = (i -> j):
        log m_{i->j}(y_j) = LSE_{y_i}[log psi(y_i, y_j) + log p_i(y_i) - log m_{j->i}(y_i)]
    Belief: log p_i = log p_i^(0) + sum_{j in N(i)} log m_{j->i}.
    """
    num_nodes, num_classes = log_prior.shape
    psi = psi.to(log_prior.device, log_prior.dtype)
    log_msg = torch.full((src.numel(), num_classes), -torch.log(torch.tensor(float(num_classes))).item(), device=log_prior.device)
    log_belief = log_prior

    for _ in range(num_layers):
        pre = log_belief[src] - log_msg[rev]
        # LSE over y_i of pre[:, y_i] + log psi[y_i, y_j] == log(exp(pre - max) @ psi) + max
        shift = pre.max(dim=1, keepdim=True).values
        new_msg = torch.log((pre - shift).exp() @ psi + 1e-30) + shift
        log_msg = new_msg - torch.logsumexp(new_msg, dim=1, keepdim=True)

        agg = torch.zeros_like(log_prior).index_add_(0, dst, log_msg)
        log_belief = F.log_softmax(log_prior + agg, dim=-1)
        if clamp_mask is not None:
            log_belief = torch.where(clamp_mask.unsqueeze(1), log_prior, log_belief)

    return log_belief


@torch.no_grad()
def bp_approx(log_prior, src, dst, r):
    """Eq. 7: log p_i^(1) = log p_i^(0) + sgn(log(r / (1 - r))) * sum_{j in N(i)} log p_j^(0)."""
    sign = 1.0 if r > 0.5 else (-1.0 if r < 0.5 else 0.0)
    agg = torch.zeros_like(log_prior).index_add_(0, dst, log_prior[src])
    return F.log_softmax(log_prior + sign * agg, dim=-1)
