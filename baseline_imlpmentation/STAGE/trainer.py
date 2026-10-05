"""
Training loop for a single STAGE GNN-ensemble member.

Generalizes gnns/gnn_mtrainer.py::GNNTrainer to accept an arbitrary pre-built
nn.Module (so MLP / GCN / SAGE / RevGAT can all be trained identically), without
modifying that file. Early stopping and model selection are both driven by
validation accuracy, matching the rest of this repo's convention.
"""
import copy

import torch
import torch.nn.functional as F

from common import compute_acc_and_f1


class StageTrainer:
    def __init__(self, model, features, data, lr=0.01, weight_decay=5e-4, epochs=200, patience=20, device="cpu"):
        self.model = model.to(device)
        self.features = features.to(device)
        self.data = data.to(device)
        self.epochs = epochs
        self.patience = patience
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.features, self.data.edge_index)
        loss = F.cross_entropy(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self.model(self.features, self.data.edge_index)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        split_metrics = {}
        for split_name, mask in [("train", self.data.train_mask), ("val", self.data.val_mask), ("test", self.data.test_mask)]:
            acc, macro_f1, weighted_f1 = compute_acc_and_f1(
                pred[mask].cpu().numpy(), self.data.y[mask].cpu().numpy()
            )
            split_metrics[split_name] = {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

        return split_metrics, logits, probs

    def train(self):
        best_val_acc = -1.0
        best_state = None
        best_test_metrics = None
        best_logits = None
        best_probs = None
        counter = 0

        for _ in range(1, self.epochs + 1):
            self._train_epoch()
            split_metrics, logits, probs = self._evaluate()

            if split_metrics["val"]["acc"] > best_val_acc:
                best_val_acc = split_metrics["val"]["acc"]
                best_test_metrics = split_metrics["test"]
                best_state = copy.deepcopy(self.model.state_dict())
                best_logits = logits.detach()
                best_probs = probs.detach()
                counter = 0
            else:
                counter += 1

            if counter >= self.patience:
                break

        self.model.load_state_dict(best_state)

        results = {
            "val_acc": best_val_acc,
            "test_acc": best_test_metrics["acc"],
            "test_macro_f1": best_test_metrics["macro_f1"],
            "test_weighted_f1": best_test_metrics["weighted_f1"],
        }
        return results, best_logits, best_probs
