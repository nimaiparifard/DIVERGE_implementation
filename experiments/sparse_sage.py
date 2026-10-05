"""Memory-light full-batch SAGE for large graphs (ArXiv): the mean aggregation is computed as a sparse
matrix product with the row-normalised adjacency instead of gathering one feature vector per edge
(1.17M edges x 2048 dims does not fit on a 12 GB GPU). Numerically the same model as GNNTrainer's."""
import torch
from torch_geometric.utils import to_torch_csr_tensor, sort_edge_index

from common import load_graph_dataset_for_tape
from gnns.gnn_mtrainer import GNNTrainer, resolve_path_prefix


def mean_adj_t(edge_index, num_nodes):
    """CSR adj_t with A[i, j] = 1/deg_in(i) for every edge j -> i (duplicates kept, as in scatter-mean)."""
    t = sort_edge_index(edge_index.flip(0), num_nodes=num_nodes)
    deg = torch.bincount(t[0], minlength=num_nodes).clamp(min=1).float()
    return to_torch_csr_tensor(t, edge_attr=1.0 / deg[t[0]], size=(num_nodes, num_nodes))


def load_sparse_graph(gnn_cfg):
    device = torch.device('cuda:0' if gnn_cfg.device > 0 or torch.cuda.is_available() else 'cpu')
    data, _, _ = load_graph_dataset_for_tape(gnn_cfg.dataset, device, re_split=gnn_cfg.re_split,
                                             path_prefix=resolve_path_prefix(), seed=gnn_cfg.split_seed)
    data.edge_index = mean_adj_t(data.edge_index, data.num_nodes)
    return data


def train_sparse_gnn_logits(gnn_cfg, emb, data):
    """Same contract as ensemble_common.train_gnn_logits; `data` comes from load_sparse_graph."""
    assert gnn_cfg.gnn_model_name == 'SAGE', 'normalised-sum trick is only valid for mean aggregation'
    trainer = GNNTrainer(gnn_cfg, emb, data=data.clone(), verbose=False)
    for conv in trainer.model.conv_layers:
        conv.aggr = 'sum'  # adj_t already holds 1/deg, so sum == mean
    results, _, _ = trainer.train()
    with torch.no_grad():
        trainer.model.eval()
        logits = trainer.model(trainer.features, trainer.graph).float().cpu()
    d = trainer.data
    meta = {'y': d.y.detach().cpu().long(), 'train_mask': d.train_mask.detach().cpu().bool(),
            'val_mask': d.val_mask.detach().cpu().bool(), 'test_mask': d.test_mask.detach().cpu().bool()}
    del trainer
    torch.cuda.empty_cache()
    return logits, results, meta
