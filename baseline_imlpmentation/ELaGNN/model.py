"""
GNN backbone and high-quality neighbourhood sampler of E-LLaGNN.

The paper deliberately uses the simplest backbone: GraphSAGE sampling + MEAN
aggregation, no normalisation, no skip connections (Section 1 footnote, Section 3.1),
with 2, 4 or 8 layers. Messages flow only along the restricted neighbourhood
N_K(v): a uniform GraphSAGE-style sample of `fanout` neighbours, re-ranked by
cosine similarity to the (enhanced) query representation, of which the top K are
kept (Section 2.2, "Neighborhood Sampling and Enrichment"; K = 5 in the paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class PlainSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.convs = nn.ModuleList(SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(n_layers))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

    def gradient_flow(self):
        """Paper Eq. (2): mean over layers of the L2 norm of dL/dW_l."""
        norms = []
        for conv in self.convs:
            grads = [p.grad.flatten() for p in conv.parameters() if p.grad is not None]
            if grads:
                norms.append(torch.cat(grads).norm(2).item())
        return sum(norms) / max(1, len(norms)), norms


def rank_within_group(group, score):
    """Position of every element inside its group when the group is sorted by `score` ascending."""
    order = torch.argsort(score, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    sorted_group = group[order]
    first = torch.searchsorted(sorted_group, sorted_group, right=False)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=group.device) - first
    return rank


def sample_topk_neighbourhood(edge_index, query_emb, neigh_emb, k, fanout=None, generator=None):
    """
    edge_index: [2, E] messages src -> dst (both directions of every undirected edge).
    For each dst node: optionally keep a uniform sample of `fanout` incoming edges, then keep the
    `k` whose source is most cosine-similar to the dst's query embedding.
    """
    src, dst = edge_index
    if fanout is not None:
        rnd = torch.rand(src.shape[0], generator=generator, device="cpu").to(src.device)
        keep = rank_within_group(dst, rnd) < fanout
        src, dst = src[keep], dst[keep]
    sim = (F.normalize(query_emb, dim=-1)[dst] * F.normalize(neigh_emb, dim=-1)[src]).sum(-1)
    keep = rank_within_group(dst, -sim) < k
    return torch.stack([src[keep], dst[keep]])
