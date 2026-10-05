"""
Ensemble step for STAGE: mean-pool per-class probabilities across GNN architectures
(paper Eq. 2: p_bar = (1/K) * sum_k p_k), then argmax for the final prediction.
"""
import torch

from common import compute_acc_and_f1


def ensemble_mean_probs(probs_list):
    return torch.stack(list(probs_list), dim=0).mean(dim=0)


def evaluate_ensemble(probs_list, y, mask):
    mean_probs = ensemble_mean_probs(probs_list)
    pred = mean_probs.argmax(dim=-1)
    acc, macro_f1, weighted_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), y[mask].cpu().numpy())
    return {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
