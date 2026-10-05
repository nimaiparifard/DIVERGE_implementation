"""
Sentence-BERT encoding of original (x_i) and LLM-enhanced (x_i^+) node texts
(E-LLaGNN paper Sections 2.2 and 3.1). Embeddings are L2-normalised and cached.
"""
import hashlib
import os

import torch

DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class SentenceEncoder:
    def __init__(self, encoder_name=DEFAULT_ENCODER, device=None, batch_size=64):
        from sentence_transformers import SentenceTransformer

        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(encoder_name, device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    def encode(self, texts, desc=None):
        if desc:
            print(f"Encoding {len(texts)} texts ({desc}) with {self.encoder_name}")
        texts = [t if t else "empty" for t in texts]
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                                 convert_to_tensor=True, normalize_embeddings=True).float().cpu()


def _tag(encoder_name):
    return encoder_name.replace("/", "__")


def get_original_embeddings(dataset_name, raw_texts, encoder):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{dataset_name}_{_tag(encoder.encoder_name)}_original.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    emb = encoder.encode(raw_texts, desc="original node texts")
    torch.save(emb, path)
    return emb


def get_augmented_embeddings(dataset_name, augmentations, node_ids, prompt_ids, encoder, llm_tag):
    """
    Returns aug_emb [len(node_ids), len(prompt_ids), d] and avail [len(node_ids), len(prompt_ids)] bool.
    Embeddings are cached per text hash, so growing the budget only encodes new texts.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{dataset_name}_{_tag(encoder.encoder_name)}_{llm_tag}_augmented.pt")
    cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}

    def key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    wanted = {}
    for v in node_ids:
        for pid in prompt_ids:
            text = augmentations.get(int(v), {}).get(pid)
            if text:
                wanted.setdefault(key(text), text)
    missing = [k for k in wanted if k not in cache]
    if missing:
        emb = encoder.encode([wanted[k] for k in missing], desc="LLM-enhanced texts")
        for k, e in zip(missing, emb):
            cache[k] = e
        torch.save(cache, path)

    dim = next(iter(cache.values())).shape[0] if cache else 0
    aug = torch.zeros(len(node_ids), len(prompt_ids), dim)
    avail = torch.zeros(len(node_ids), len(prompt_ids), dtype=torch.bool)
    for i, v in enumerate(node_ids):
        for j, pid in enumerate(prompt_ids):
            text = augmentations.get(int(v), {}).get(pid)
            if text:
                aug[i, j] = cache[key(text)]
                avail[i, j] = True
    return aug, avail
