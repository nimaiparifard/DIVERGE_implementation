"""
Structural related retrieval (SKETCH paper Section 3.2, Algorithm 1).

Neighbours of an anchor within K hops are ranked by the (multi-hop) Jaccard
similarity of their neighbourhoods, estimated with MinHash instead of explicit set
intersections:

    Pr[MinHash(N^i(v)) == MinHash(N^j(s))] = J(N^i(v), N^j(s))

so the number of colliding MinHash values over R independent hash functions is an
unbiased (scaled) estimate of the Jaccard similarity. As in the paper we compare
every hop combination (1-1, 1-2, 2-1, 2-2, ... for K=2) and give the crucial
one-hop/one-hop intersection `l` hash functions while the other combinations get a
shorter signature of length `m < l`, which down-weights them.

Implementation notes
- N^k(v) is the set of nodes reachable from v within k hops (for k=1: the direct
  neighbours). Its MinHash is computed exactly for all nodes at once by k rounds of
  scatter-min message passing: sig_k(v) = min(sig_{k-1}(v), min_{u in N(v)} sig_{k-1}(u)).
- h_r(u) = (a_r * u + b_r) mod p with p = 2^31 - 1 is a universal hash family
  (Carter & Wegman, 1977), as referenced in the paper.
- Candidates for anchor v are all nodes within K hops of v. Pairs are scored in
  chunks of anchors with sparse matrix products, so the full N x N matrix is never
  materialised.
"""
import numpy as np
import scipy.sparse as sp
import torch

_PRIME = (1 << 31) - 1


def build_adjacency(edge_index, num_nodes):
    """Binary, symmetric CSR adjacency without self-loops."""
    ei = edge_index.cpu().numpy()
    adj = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(num_nodes, num_nodes))
    adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def minhash_signatures(edge_index, num_nodes, num_hops, num_hashes, seed=0):
    """
    Returns a list of `num_hops` int64 tensors [N, num_hashes]; entry k-1 holds the
    MinHash signature of every node's within-k-hop neighbourhood. Nodes with no
    neighbours get the sentinel value `_PRIME` (never collides with a real hash
    because real hashes are < _PRIME and we mask sentinel collisions out).
    """
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(1, _PRIME, (num_hashes,), generator=gen, dtype=torch.int64)
    b = torch.randint(0, _PRIME, (num_hashes,), generator=gen, dtype=torch.int64)
    node_ids = torch.arange(num_nodes, dtype=torch.int64)
    base = (node_ids[:, None] * a[None, :] + b[None, :]) % _PRIME  # h_r(u) for every node u

    ei = edge_index.cpu().long()
    src = torch.cat([ei[0], ei[1]])
    dst = torch.cat([ei[1], ei[0]])
    non_loop = src != dst
    src, dst = src[non_loop], dst[non_loop]

    sentinel = torch.full((num_nodes, num_hashes), _PRIME, dtype=torch.int64)
    signatures = []
    prev = None
    for hop in range(num_hops):
        # hop 1: min over direct neighbours' own hashes; hop k: also min over their (k-1)-hop signatures
        messages = base[src] if hop == 0 else torch.minimum(base[src], prev[src])
        cur = sentinel.clone().scatter_reduce(0, dst[:, None].expand(-1, num_hashes), messages, reduce="amin")
        if prev is not None:
            cur = torch.minimum(cur, prev)
        signatures.append(cur)
        prev = cur
    return signatures


def _khop_candidates(adj, rows, num_hops):
    """Sparse [len(rows), N] matrix whose nonzeros are the nodes within num_hops of each row; value = hop distance."""
    frontier = adj[rows]
    reach = frontier.copy()
    reach.data[:] = 1.0
    for hop in range(2, num_hops + 1):
        frontier = (frontier @ adj)
        frontier.data[:] = 1.0
        new = frontier - frontier.multiply(reach > 0)
        new.eliminate_zeros()
        new.data[:] = float(hop)
        reach = reach + new
    return reach.tocsr()


def structural_topk(edge_index, num_nodes, top_k, num_hops=2, num_hashes_l=64, num_hashes_m=16,
                    seed=0, chunk_size=512):
    """
    Rank the within-`num_hops` neighbours of every node by MinHash-estimated
    multi-hop Jaccard similarity (paper Algorithm 1) and keep the best `top_k`.

    Returns (indices [N, top_k] int64 padded with -1, scores [N, top_k] float32, hop [N, top_k] int8).
    Ties in the collision count are broken towards closer hops (the paper's Table 2
    shows one-hop text is the most valuable).
    """
    assert num_hashes_m <= num_hashes_l
    sigs = [s.numpy() for s in minhash_signatures(edge_index, num_nodes, num_hops, num_hashes_l, seed=seed)]
    adj = build_adjacency(edge_index, num_nodes)

    out_idx = np.full((num_nodes, top_k), -1, dtype=np.int64)
    out_score = np.full((num_nodes, top_k), -np.inf, dtype=np.float32)
    out_hop = np.zeros((num_nodes, top_k), dtype=np.int8)

    for start in range(0, num_nodes, chunk_size):
        rows = np.arange(start, min(start + chunk_size, num_nodes))
        reach = _khop_candidates(adj, rows, num_hops).tocoo()
        keep = reach.row + start != reach.col  # drop the anchor itself (reachable via 2-hop cycles)
        pair_anchor = reach.row[keep] + start
        pair_cand = reach.col[keep]
        pair_hop = reach.data[keep].astype(np.int8)
        if len(pair_anchor) == 0:
            continue

        score = np.zeros(len(pair_anchor), dtype=np.float32)
        for i in range(num_hops):
            for j in range(num_hops):
                r = num_hashes_l if (i == 0 and j == 0) else num_hashes_m
                sa = sigs[i][pair_anchor, :r]
                sc = sigs[j][pair_cand, :r]
                score += ((sa == sc) & (sa != _PRIME)).sum(axis=1)
        # tie-break towards closer hops without ever overriding a real collision difference
        score = score + 0.5 / pair_hop.astype(np.float32)

        # top_k per anchor: sort by (anchor asc, score desc) and take the first top_k of each group
        order = np.lexsort((-score, pair_anchor))
        pair_anchor, pair_cand, pair_hop, score = pair_anchor[order], pair_cand[order], pair_hop[order], score[order]
        group_start = np.searchsorted(pair_anchor, rows)
        group_end = np.searchsorted(pair_anchor, rows, side="right")
        for v, gs, ge in zip(rows, group_start, group_end):
            n = min(ge - gs, top_k)
            if n == 0:
                continue
            out_idx[v, :n] = pair_cand[gs:gs + n]
            out_score[v, :n] = score[gs:gs + n]
            out_hop[v, :n] = pair_hop[gs:gs + n]

    return out_idx, out_score, out_hop


def khop_neighbours(edge_index, num_nodes, num_hops, chunk_size=2048):
    """List of int64 arrays: all nodes within num_hops of each node (for the random-selection ablations)."""
    adj = build_adjacency(edge_index, num_nodes)
    result = []
    for start in range(0, num_nodes, chunk_size):
        rows = np.arange(start, min(start + chunk_size, num_nodes))
        reach = _khop_candidates(adj, rows, num_hops)
        for local, v in enumerate(rows):
            nbrs = reach.indices[reach.indptr[local]:reach.indptr[local + 1]]
            result.append(nbrs[nbrs != v].astype(np.int64))
    return result
