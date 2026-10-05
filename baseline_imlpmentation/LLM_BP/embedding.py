"""
Principle I of LLM-BP (Wang et al., ICML 2025): task-adaptive node embeddings.

The paper encodes every node with a decoder LLM turned encoder (LLM2Vec, Mistral-7B)
using a class-conditional prompt (Eq. 1 / Appendix B.2):

    <Instruct> Given the {task description}, classify it into one of the following k classes:
    {class labels}
    <query> {raw node text} <response>

and takes the hidden state of the token right after the inputs as the node embedding.
For a causal decoder that is the last-token hidden state, which is what we pool here.

We reuse this repo's local checkpoints (configs/llm_configs/*, local_models/*) instead
of LLM2Vec-Mistral-7B; the prompting and pooling follow the paper.
"""
import os

import torch

from config import load_llm_config_json
from train_llm.peft_model import get_model_path


# {task description} per dataset, following paper Table 13 for the datasets it covers
# (Cora, CiteSeer, PubMed, WikiCS, History) and written in the same style for the rest.
TASK_DESCRIPTIONS = {
    "cora": "opening text of machine learning papers",
    "citeseer": "description or opening text of scientific publications",
    "pubmed": "title and abstract of scientific publications",
    "wikics": "entry and content of wikipedia",
    "history": "description or title of the book",
    "arxiv": "title and abstract of computer science papers on arXiv",
    "computer": "user review of an item in the computers category of an e-commerce site",
    "photo": "user review of an item in the photo category of an e-commerce site",
    "instagram": "profile description of an Instagram user",
    "reddit": "recent posts of a Reddit user",
}

ENCODINGS = ["task_adaptive", "vanilla"]


def build_prompt(text, task_description, class_names, encoding):
    if encoding == "task_adaptive":
        return (
            f"Instruct: Given the {task_description}, classify it into one of the following "
            f"{len(class_names)} classes:\n{', '.join(class_names)}\n"
            f"Query: {text}\nResponse:"
        )
    # Vanilla LLM2Vec prompt without class information (paper Eq. 8, Table 12)
    return f"Instruct: Encode the {task_description}:\nQuery: {text}\nResponse:"


class TaskAdaptiveEncoder:
    """Frozen local LLM producing one embedding per prompt (last-token or mean pooling)."""

    def __init__(self, llm_name="llama_3.2_1B", device=None, max_text_tokens=480, batch_size=16, pooling="last"):
        from transformers import AutoModel, AutoTokenizer

        self.llm_name = llm_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_text_tokens = max_text_tokens
        self.batch_size = batch_size
        self.pooling = pooling

        llm_cfg = load_llm_config_json(llm_name)
        model_path = get_model_path(llm_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Local model not found at '{model_path}'. LLM-BP reuses this repo's "
                f"local_models/ checkpoints -- place weights there or pick a different llm_name."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=llm_cfg.tokenizer.local_files_only,
            trust_remote_code=llm_cfg.tokenizer.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        # Last-token pooling below indexes the last non-pad position, which assumes right padding
        self.tokenizer.padding_side = "right"

        dtype = torch.float32
        if str(self.device).startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=llm_cfg.llm.local_files_only,
            trust_remote_code=llm_cfg.llm.trust_remote_code,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def truncate_texts(self, texts):
        """Truncate only the node text so the instruction and the trailing 'Response:' survive."""
        ids = self.tokenizer(
            [t if t else "empty" for t in texts],
            add_special_tokens=False, truncation=True, max_length=self.max_text_tokens,
        )["input_ids"]
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def encode(self, prompts, show_progress=True):
        from tqdm import tqdm

        # Batch prompts of similar length together, then restore the original order
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        embeddings = [None] * len(prompts)
        iterator = range(0, len(order), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Encoding with {self.llm_name} ({self.pooling} pooling)")

        for start in iterator:
            idx = order[start:start + self.batch_size]
            encoded = self.tokenizer(
                [prompts[i] for i in idx], return_tensors="pt", padding=True,
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"]
            if self.pooling == "last":
                last = mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
            else:
                m = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            for i, vec in zip(idx, pooled.float().cpu()):
                embeddings[i] = vec

        return torch.stack(embeddings, dim=0)


def get_cache_path(dataset_name, llm_name, encoding, pooling, cache_dir=None):
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{dataset_name}_{llm_name}_{encoding}_{pooling}_emb.pt")


def get_or_build_embeddings(
    dataset_name,
    raw_texts,
    class_names,
    llm_name="llama_3.2_1B",
    encoding="task_adaptive",
    pooling="last",
    device=None,
    max_text_tokens=480,
    batch_size=16,
    cache_dir=None,
    force_rebuild=False,
):
    """Load cached LLM-BP node embeddings for (dataset, llm, encoding, pooling) or build+cache them."""
    cache_path = get_cache_path(dataset_name, llm_name, encoding, pooling, cache_dir)
    if os.path.exists(cache_path) and not force_rebuild:
        return torch.load(cache_path, weights_only=False)["embeddings"]

    task_description = TASK_DESCRIPTIONS.get(dataset_name, "text of the node")
    encoder = TaskAdaptiveEncoder(
        llm_name=llm_name, device=device, max_text_tokens=max_text_tokens,
        batch_size=batch_size, pooling=pooling,
    )
    texts = encoder.truncate_texts(list(raw_texts))
    prompts = [build_prompt(t, task_description, class_names, encoding) for t in texts]
    print(f"Example prompt:\n{prompts[0]}\n")
    embeddings = encoder.encode(prompts)
    torch.save(
        {
            "embeddings": embeddings,
            "llm_name": llm_name,
            "dataset": dataset_name,
            "encoding": encoding,
            "pooling": pooling,
            "task_description": task_description,
            "class_names": list(class_names),
        },
        cache_path,
    )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings
