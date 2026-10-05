"""
Text Propagation + Text Augmentation (paper Section 3.2 / 4.1, Eq. 10-14).

Text Propagation reconstructs/enriches every node's text by concatenating it with
its neighbors' texts (message-passing over raw text, before any embedding).
Text Augmentation then prompts an LLM to produce a summary, keywords, and a soft
label per node from the propagated text; the final augmented text is the
concatenation of all of these (paper's Eq. 14 AGG = concatenation).
"""
import os

import torch

from ULTRATAG.generation import parse_json_object

DATASET_DOMAINS = {
    "cora": "computer science research papers, categorized by machine-learning sub-field",
    "pubmed": "biomedical research papers about diabetes, categorized by diabetes type/treatment",
}


def propagate_texts(texts, edge_index, num_nodes, max_neighbors=5, max_chars_per_text=300):
    """Eq. 10: T'_i = t_i (+) {t_j | j in N_i}, truncated/capped for tractable prompt length."""
    neighbors = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        neighbors[s].append(d)

    propagated = []
    for i in range(num_nodes):
        own_text = (texts[i] or "").strip()[:max_chars_per_text]
        neighbor_ids = neighbors[i][:max_neighbors]
        neighbor_texts = [(texts[j] or "").strip()[:max_chars_per_text] for j in neighbor_ids if texts[j]]
        if own_text:
            combined = own_text
            if neighbor_texts:
                combined += " [Related: " + " | ".join(neighbor_texts) + "]"
        elif neighbor_texts:
            combined = "[Reconstructed from neighbors: " + " | ".join(neighbor_texts) + "]"
        else:
            combined = "[No text available]"
        propagated.append(combined)

    return propagated


def build_augmentation_prompt(propagated_text, dataset_name, label_names):
    domain = DATASET_DOMAINS.get(dataset_name, "a text-attributed graph")
    labels_str = ", ".join(label_names)
    return (
        f"You are analyzing a node from a graph dataset about {domain}.\n"
        f"Text:\n\"\"\"{propagated_text}\"\"\"\n\n"
        f"Respond with ONLY a JSON object with exactly these fields:\n"
        f'- "summary": a concise one-sentence summary of the text\n'
        f'- "keywords": a list of up to 5 important keywords from the text\n'
        f'- "soft_label": your best-guess category for this node, chosen EXACTLY '
        f"and verbatim from this list: [{labels_str}]\n"
    )


def generate_with_checkpoint(generator, prompts, checkpoint_path=None, checkpoint_every=25, desc="Generating"):
    """
    Like InstructLLM.generate(), but periodically persists partial progress to
    `checkpoint_path` and resumes from it if present. This is a per-node, O(N) LLM
    generation loop (the most expensive and longest-running step in the whole
    pipeline -- ~1.5h for PubMed's ~19.7k nodes); without checkpointing, a crash
    near the end (observed: an OOM at 96% through a PubMed run, see
    ULTRATAG/generation.py's `generate()` docstring/comment) loses all of it.
    """
    from tqdm import tqdm

    raw_outputs = [None] * len(prompts)
    start_batch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if checkpoint.get("num_prompts") == len(prompts):
            raw_outputs = checkpoint["raw_outputs"]
            start_batch = checkpoint["next_index"]
            print(f"Resuming '{desc}' from checkpoint: {start_batch}/{len(prompts)} already done.")

    batch_size = generator.batch_size
    batch_starts = list(range(start_batch, len(prompts), batch_size))
    for i, batch_start in enumerate(tqdm(batch_starts, desc=desc)):
        batch = prompts[batch_start:batch_start + batch_size]
        outputs = generator._generate_batch(batch)
        for offset, output in enumerate(outputs):
            raw_outputs[batch_start + offset] = output

        if torch.cuda.is_available() and (i + 1) % 25 == 0:
            torch.cuda.empty_cache()

        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            torch.save(
                {"raw_outputs": raw_outputs, "next_index": batch_start + len(batch), "num_prompts": len(prompts)},
                checkpoint_path,
            )

    if checkpoint_path:
        torch.save({"raw_outputs": raw_outputs, "next_index": len(prompts), "num_prompts": len(prompts)}, checkpoint_path)

    return raw_outputs


def augment_texts(propagated_texts, dataset_name, label_names, generator, checkpoint_path=None):
    """
    Eq. 11-14: LLM-generated summary/keywords/soft-label per node, aggregated
    (concatenated) with the propagated text into the final augmented text T*.

    Returns (augmented_texts: list[str], soft_labels: list[str]).
    """
    prompts = [build_augmentation_prompt(t, dataset_name, label_names) for t in propagated_texts]
    raw_outputs = generate_with_checkpoint(
        generator, prompts, checkpoint_path=checkpoint_path, desc=f"Augmenting texts ({dataset_name})",
    )

    augmented_texts, soft_labels = [], []
    for propagated_text, raw in zip(propagated_texts, raw_outputs):
        parsed = parse_json_object(raw, default={"summary": "", "keywords": [], "soft_label": ""})
        summary = str(parsed.get("summary", "") or "")
        keywords = parsed.get("keywords", []) or []
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        keywords_str = ", ".join(str(k) for k in keywords)
        soft_label = str(parsed.get("soft_label", "") or "")
        if soft_label not in label_names:
            soft_label = ""  # unparseable / hallucinated label -> excluded from virtual-edge grouping

        parts = [propagated_text]
        if summary:
            parts.append(f"Summary: {summary}")
        if keywords_str:
            parts.append(f"Keywords: {keywords_str}")
        if soft_label:
            parts.append(f"Category: {soft_label}")

        augmented_texts.append(" ".join(parts))
        soft_labels.append(soft_label)

    return augmented_texts, soft_labels
