"""
E-LLaGNN training loop (paper Sections 2.2 and 2.5).

Every epoch:
  1. each node chosen by active node selection is shown in its LLM-enhanced form
     x^+ with probability p (else its original x), with the prompt drawn at random
     from the ones available for that node -> a mix of original and LLM-tailored
     features that differs across epochs;
  2. the restricted neighbourhood N_K(v) is re-sampled: a uniform fan-out sample
     re-ranked by cosine similarity to the (possibly enhanced) query;
  3. one full-batch step of the plain mean-SAGE backbone on the train labels.

Model selection and evaluation use the LLM-free pipeline (Figure 2): original
features only, top-K neighbours ranked with the original query embedding. The
LLM-incorporated variant (Table 8) is also reported for the final checkpoint.
"""
import copy

import torch
import torch.nn.functional as F

from common.metrics import compute_acc_and_f1
from ELaGNN.model import PlainSAGE, sample_topk_neighbourhood


def mix_features(x_orig, sel_ids, aug_emb, avail, p, generator):
    """Original features with a random subset (prob. p) of the eligible nodes replaced by a random x^+."""
    x = x_orig.clone()
    if len(sel_ids) == 0:
        return x
    draw = torch.rand(len(sel_ids), generator=generator) < p
    scores = torch.rand(avail.shape, generator=generator).masked_fill(~avail, -1.0)
    choice = scores.argmax(dim=1)
    enhance = draw & avail.any(dim=1)
    if enhance.any():
        rows = enhance.nonzero().squeeze(-1)
        x[sel_ids[rows].to(x.device)] = aug_emb[rows, choice[rows]].to(x.device)
    return x


def llm_incorporated_features(x_orig, sel_ids, aug_emb, avail):
    """Inference-time features when the LLM is available: mean of a node's enhanced variants."""
    x = x_orig.clone()
    has = avail.any(dim=1)
    if len(sel_ids) == 0 or not has.any():
        return x
    mean = (aug_emb * avail.unsqueeze(-1)).sum(1) / avail.sum(1, keepdim=True).clamp(min=1)
    x[sel_ids[has].to(x.device)] = F.normalize(mean[has], dim=-1).to(x.device)
    return x


@torch.no_grad()
def evaluate(model, x, edge_index, data, top_k):
    model.eval()
    ei = sample_topk_neighbourhood(edge_index, x, x, top_k, fanout=None)
    pred = model(x, ei).argmax(-1).cpu().numpy()
    y = data.y.cpu().numpy()
    out = {}
    for split, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        m = mask.cpu().numpy()
        acc, macro_f1, weighted_f1 = compute_acc_and_f1(pred[m], y[m])
        out[f"{split}_acc"], out[f"{split}_macro_f1"], out[f"{split}_weighted_f1"] = acc, macro_f1, weighted_f1
    return out


def train_elagnn(data, x_orig, sel_ids, aug_emb, avail, num_classes, hp, seed, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x_orig = x_orig.to(device)
    edge_index = data.edge_index.to(device)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    model = PlainSAGE(x_orig.shape[1], hp["hidden_dim"], num_classes, hp["n_layers"], hp["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    best_val, best_state, best_epoch, bad = -1.0, None, 0, 0
    grad_flow = []
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        x = mix_features(x_orig, sel_ids, aug_emb, avail, hp["p"], gen)
        ei = sample_topk_neighbourhood(edge_index, x, x, hp["top_k"], fanout=hp["fanout"], generator=gen)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, ei)[train_mask], y[train_mask])
        loss.backward()
        grad_flow.append(round(model.gradient_flow()[0], 6))
        optimizer.step()

        metrics = evaluate(model, x_orig, edge_index, data, hp["top_k"])
        if metrics["val_acc"] > best_val:
            best_val, best_state, best_epoch, bad = metrics["val_acc"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
        if epoch % 50 == 0:
            print(f"    epoch {epoch:4d} loss={loss.item():.4f} val_acc={metrics['val_acc']:.2f} (best {best_val:.2f} @ {best_epoch})")

    model.load_state_dict(best_state)
    results = evaluate(model, x_orig, edge_index, data, hp["top_k"])
    llm_results = evaluate(model, llm_incorporated_features(x_orig, sel_ids, aug_emb, avail), edge_index, data, hp["top_k"])
    results["test_acc_llm_incorporated"] = llm_results["test_acc"]
    results["val_acc_llm_incorporated"] = llm_results["val_acc"]
    results["best_epoch"] = best_epoch
    results["gradient_flow"] = grad_flow
    return results
