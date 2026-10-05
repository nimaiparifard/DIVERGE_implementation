"""
Graph-Enhanced Robust Classifier (paper Section 4.3, Eq. 25-28): a dual-GNN.

GNN_1 encodes the LLM-augmented graph (H, A*) into node embeddings; their cosine
similarity is blended into the adjacency for GNN_2, EXCEPT for edges between two
"important" (PageRank-selected) nodes, whose LLM-judged connectivity is preserved
unchanged. Both GNNs are optimized jointly with a single cross-entropy loss.

Scoping note: Eq. 26 defines a dense [N, N] similarity matrix. For PubMed-scale
graphs (~20k nodes) a dense N^2 matrix is a large (but not infeasible) memory cost;
we instead keep only each node's top-k most similar neighbors (`compute_topk_similarity_edges`),
a standard sparsification used by graph-structure-learning methods (e.g. IDGL) for
the same reason. This preserves the paper's mechanism (learned similarity refines
the adjacency for non-important-node pairs, LLM judgments are frozen for
important-node pairs) while keeping GNN_2 a sparse message-passing operation at any
graph scale, via `torch_geometric.nn.GCNConv`'s `edge_weight` support.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce

from common import GNNEncoder, compute_acc_and_f1


class WeightedGCN(nn.Module):
    """GNN_2: a plain GCN that accepts a dense edge_weight vector (GNNEncoder does not)."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight)


def compute_topk_similarity_edges(embeddings, important_mask, k=10, chunk_size=2048):
    """Vectorized top-k cosine-similarity neighbors per node, excluding self and
    excluding pairs where both endpoints are 'important' nodes (Eq. 26's else-branch)."""
    num_nodes = embeddings.shape[0]
    device = embeddings.device
    normalized = F.normalize(embeddings, p=2, dim=1)

    src_chunks, dst_chunks, sim_chunks = [], [], []
    k = min(k, num_nodes - 1)
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        block = normalized[start:end]
        sim_block = block @ normalized.t()
        rows_local = torch.arange(end - start, device=device)
        sim_block[rows_local, torch.arange(start, end, device=device)] = -2.0
        topk_vals, topk_idx = torch.topk(sim_block, k=k, dim=1)
        row_ids = torch.arange(start, end, device=device).unsqueeze(1).expand_as(topk_idx)
        src_chunks.append(row_ids.reshape(-1))
        dst_chunks.append(topk_idx.reshape(-1))
        sim_chunks.append(topk_vals.reshape(-1))

    src = torch.cat(src_chunks)
    dst = torch.cat(dst_chunks)
    sim = torch.cat(sim_chunks)

    both_important = important_mask[src] & important_mask[dst]
    keep = ~both_important
    return src[keep], dst[keep], sim[keep]


def build_blended_adjacency(edge_index, embeddings, important_mask, num_nodes, k=10):
    """Eq. 26: Ã*_ij = A*_ij + S_ij unless both i,j are important nodes (then unchanged)."""
    device = embeddings.device
    base_weight = torch.ones(edge_index.shape[1], device=device)

    sim_src, sim_dst, sim_val = compute_topk_similarity_edges(embeddings, important_mask, k=k)

    self_loops = torch.arange(num_nodes, device=device)
    self_weight = torch.ones(num_nodes, device=device)

    all_src = torch.cat([edge_index[0], sim_src, self_loops])
    all_dst = torch.cat([edge_index[1], sim_dst, self_loops])
    all_weight = torch.cat([base_weight, sim_val, self_weight])

    combined_edge_index = torch.stack([all_src, all_dst], dim=0)
    return coalesce(combined_edge_index, all_weight, num_nodes, reduce="sum")


class DualGNNTrainer:
    def __init__(
        self,
        num_node_features,
        num_classes,
        augmented_edge_index,
        important_mask,
        hidden_dim=64,
        dropout=0.5,
        lr=1e-2,
        weight_decay=5e-4,
        epochs=100,
        topk=10,
        device="cpu",
    ):
        self.augmented_edge_index = augmented_edge_index.to(device)
        self.important_mask = important_mask.to(device)
        self.epochs = epochs
        self.topk = topk
        self.device = device

        self.gnn1 = GNNEncoder(
            num_node_features, hidden_dim, hidden_dim, n_layers=2, gnn_type="GCN", dropout=dropout,
        ).to(device)
        self.gnn2 = WeightedGCN(num_node_features, hidden_dim, num_classes, n_layers=2, dropout=dropout).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.gnn1.parameters()) + list(self.gnn2.parameters()), lr=lr, weight_decay=weight_decay,
        )

    def _forward(self, features):
        h1 = self.gnn1(features, self.augmented_edge_index)
        blended_edge_index, blended_edge_weight = build_blended_adjacency(
            self.augmented_edge_index, h1, self.important_mask, features.shape[0], k=self.topk,
        )
        return self.gnn2(features, blended_edge_index, blended_edge_weight)

    def train(self, features, data, patience=20):
        features = features.to(self.device)
        data = data.to(self.device)

        best_val_acc, best_state, best_test_metrics, counter = -1.0, None, None, 0

        for _ in range(1, self.epochs + 1):
            self.gnn1.train()
            self.gnn2.train()
            self.optimizer.zero_grad()
            logits = self._forward(features)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            self.gnn1.eval()
            self.gnn2.eval()
            with torch.no_grad():
                logits = self._forward(features)
                pred = logits.argmax(dim=-1)
                split_metrics = {}
                for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
                    acc, macro_f1, weighted_f1 = compute_acc_and_f1(
                        pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()
                    )
                    split_metrics[name] = {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

            if split_metrics["val"]["acc"] > best_val_acc:
                best_val_acc = split_metrics["val"]["acc"]
                best_test_metrics = split_metrics["test"]
                best_state = (copy.deepcopy(self.gnn1.state_dict()), copy.deepcopy(self.gnn2.state_dict()))
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break

        if best_state is not None:
            self.gnn1.load_state_dict(best_state[0])
            self.gnn2.load_state_dict(best_state[1])

        return {
            "val_acc": best_val_acc,
            "test_acc": best_test_metrics["acc"],
            "test_macro_f1": best_test_metrics["macro_f1"],
            "test_weighted_f1": best_test_metrics["weighted_f1"],
        }
