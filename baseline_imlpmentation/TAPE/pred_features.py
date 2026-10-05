"""
Ranked-prediction features h_pred (paper Eq. 5): one-hot encode each node's ranked
LLM predictions and project to a fixed-size dense vector.

The paper fixes k (top-k predictions requested) per dataset's prompt; Cora/PubMed's
prompts (Table 5) don't request a fixed top-k, so we use k = num_classes (every
class can appear somewhere in the ranking). The paper describes this as a
preprocessing step with no dedicated training loss of its own (unlike h_orig/h_expl,
which are cross-entropy finetuned) -- so we use a fixed, seed-deterministic random
linear projection rather than introducing an extra training loop for a component
the paper doesn't otherwise specify a loss for.
"""
import torch


def build_h_pred(predictions_per_node, label_names, d_pred=128, seed=42):
    num_classes = len(label_names)
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    k = num_classes

    num_nodes = len(predictions_per_node)
    one_hot = torch.zeros(num_nodes, k, num_classes)
    for i, predictions in enumerate(predictions_per_node):
        for slot, label in enumerate(predictions[:k]):
            idx = label_to_idx.get(label)
            if idx is not None:
                one_hot[i, slot, idx] = 1.0

    flat = one_hot.view(num_nodes, k * num_classes)

    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(k * num_classes, d_pred, generator=generator) / (k * num_classes) ** 0.5

    return flat @ projection
