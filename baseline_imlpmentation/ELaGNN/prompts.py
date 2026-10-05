"""
Prompt catalog for on-demand node enhancement (E-LLaGNN paper Section 2.2, Table 2).

The paper's catalog asks the LLM to (i) explain the key technical concepts of a node
text, (ii) rewrite it and argue which of the dataset's categories it fits, and (iii)
summarise / relate it to other nodes. Every eligible node can be enhanced with any
prompt, and a prompt is drawn at random each time the node is enhanced, so the GNN
sees diverse LLM knowledge for the same node across epochs.

The categories in prompt (ii) are the dataset's class *names* (the label set), never
the node's own label.
"""

NODE_TYPES = {
    "cora": "paper", "citeseer": "paper", "pubmed": "paper", "arxiv": "paper",
    "wikics": "Wikipedia article",
    "computer": "product", "photo": "product", "history": "book",
    "reddit": "user", "instagram": "user",
}

SYSTEM_MESSAGE = ("A chat between a curious user and an artificial intelligence assistant. The assistant gives "
                  "helpful, detailed, and polite answers to the user's questions.")

PROMPT_CATALOG = {
    "explain": (
        "Provide a detailed explanation for the given {node_type} in less than {max_words} words, identifying "
        "the key technical concepts and simplifying them for ease of understanding.\n\n"
        "[ Start of Text ]\n{text}\n[ End of Text ]"
    ),
    "argue_category": (
        "Rewrite the given {node_type} and provide arguments to support which of the following categories it is "
        "well suited for: {categories}. Answer in less than {max_words} words.\n\n"
        "[ Start of Text ]\n{text}\n[ End of Text ]"
    ),
    "summarize_entities": (
        "Summarize the given {node_type} in less than {max_words} words. Then list its key entities and concepts "
        "and the relationships between them.\n\n"
        "[ Start of Text ]\n{text}\n[ End of Text ]"
    ),
}


def build_prompt(prompt_id, dataset_name, text, label_names, max_words=150, max_text_words=350):
    words = (text or "empty").split()
    if len(words) > max_text_words:
        words = words[:max_text_words]
    template = PROMPT_CATALOG[prompt_id]
    body = template.format(
        node_type=NODE_TYPES.get(dataset_name, "node"),
        max_words=max_words,
        categories=", ".join(str(c) for c in label_names),
        text=" ".join(words),
    )
    return f"{SYSTEM_MESSAGE}\n\n{body}"
