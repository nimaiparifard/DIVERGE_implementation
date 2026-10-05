"""
GNN ensemble architectures for STAGE.

The paper ensembles GCN, GraphSAGE, RevGAT and a plain MLP. GCN and SAGE are reused
directly from this repo's common.gnn.GNNEncoder. RevGAT is approximated here with
GNNEncoder's GAT layer plus residual connections and batch norm -- i.e. the "GAT +
residual + norm" recipe from Li et al. (2022a), which is what makes RevGAT trainable
at depth. We do NOT reimplement RevGAT's memory-efficient reversible backward pass,
since that only matters for very deep networks on graphs far larger than Cora/PubMed;
functionally this gives the same forward computation RevGAT would produce for the
shallow (2-3 layer) networks used here. MLP has no GNNEncoder equivalent (it must
ignore edge_index entirely), so it is implemented directly below.
"""
import torch.nn as nn
import torch.nn.functional as F

from common import GNNEncoder


class MLPNodeClassifier(nn.Module):
    """Plain MLP baseline: node features only, no message passing."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5, batch_norm=True):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.bns = (
            nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)])
            if batch_norm
            else None
        )
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.bns is not None:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


def build_gcn(input_dim, output_dim, hidden_dim, n_layers, dropout, batch_norm=True):
    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        gnn_type="GCN",
        dropout=dropout,
        batch_norm=int(batch_norm),
    )


def build_sage(input_dim, output_dim, hidden_dim, n_layers, dropout, batch_norm=True):
    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        gnn_type="SAGE",
        dropout=dropout,
        batch_norm=int(batch_norm),
    )


def build_revgat(input_dim, output_dim, hidden_dim, n_layers, dropout, batch_norm=True):
    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        gnn_type="GAT",
        dropout=dropout,
        batch_norm=int(batch_norm),
        residual_conn=1,
    )


def build_mlp(input_dim, output_dim, hidden_dim, n_layers, dropout, batch_norm=True):
    return MLPNodeClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout,
        batch_norm=batch_norm,
    )


MODEL_BUILDERS = {
    "MLP": build_mlp,
    "GCN": build_gcn,
    "SAGE": build_sage,
    "RevGAT": build_revgat,
}
