from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import os
import json
from common import load_graph_dataset_for_tape

class TagsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long type
        return item

    def __len__(self):
        return len(self.labels)

def get_model_path(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    local_model_dir_path = os.path.join(repo_root, "local_models")
    model_name = model_name
    model_path = os.path.join(local_model_dir_path, model_name)
    return model_path

def get_tokenizer(cfg):
    tokenizer_path = get_model_path(cfg.llm.model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              local_files_only=cfg.llm.local_files_only,
                                              trust_remote_code=cfg.llm.trust_remote_code,
                                              padding_side=cfg.tokenizer.padding_side,
                                              max_length=cfg.tokenizer.max_length)
    # Causal LMs often lack a pad token; encoder models usually already have one.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Keep padding side aligned with config (left for decoder seq-cls, right for encoders)
    tokenizer.padding_side = cfg.tokenizer.padding_side
    return tokenizer

def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    datasets_path = os.path.join(repo_root, "datasets")
    return datasets_path

def load_dataset(cfg):
    datasets_path = get_dataset_path()
    dataset = torch.load(os.path.join(datasets_path, f"{cfg.dataset.name}.pt"), weights_only=False)
    return dataset

def load_gpt_enhanced_explanations(cfg):
    datasets_path = get_dataset_path()
    gpt_responses_path = os.path.join(datasets_path, cfg.dataset.llm_responses_name)
    gpt_responses_json_path = os.path.join(gpt_responses_path, f"{cfg.dataset.name}.json")
    llm_responses = []
    if cfg.dataset.name == 'pubmed':
        gpt_response_path = os.path.join(datasets_path, 'PubMed')
        llm_responses = []
        for i in range(0, cfg.dataset.num_samples):
            json_sample_path = os.path.join(gpt_response_path, f"{i}.json")
            json_sample = json.load(open(json_sample_path))
            llm_responses.append(json_sample['choices'][0]['message']['content'])
        return llm_responses
    try:
        with open(gpt_responses_json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    gpt_response = json.loads(line)
                    llm_responses.append(gpt_response['answer'])
        return llm_responses
    except FileNotFoundError:
        return None


def encoding_texts_to_tokens(tokenizer, texts, max_length=512):
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors=None)
    return encodings

def load_summary_explanation(cfg):
    datasets_path = get_dataset_path()
    gpt_response_path = os.path.join(datasets_path, 'enhanced_texts', cfg.dataset.name)
    llm_responses = []
    for i in range(0, cfg.dataset.num_samples):
        json_sample_path = os.path.join(gpt_response_path, f"{i}.json")
        json_sample = json.load(open(json_sample_path))
        llm_responses.append(json_sample['summary'])
    return llm_responses

def load_paraphrased_explanation(cfg):
    datasets_path = get_dataset_path()
    gpt_response_path = os.path.join(datasets_path, 'enhanced_texts', cfg.dataset.name)
    llm_responses = []
    for i in range(0, cfg.dataset.num_samples):
        json_sample_path = os.path.join(gpt_response_path, f"{i}.json")
        json_sample = json.load(open(json_sample_path))
        llm_responses.append(json_sample['paraphrased_text'])
    return llm_responses


def load_keywords(cfg, labels):
    datasets_path = get_dataset_path()
    gpt_response_path = os.path.join(datasets_path, 'enhanced_texts', cfg.dataset.name)
    llm_responses_keywords = []
    llm_responses_keywords_labels = []
    for i in range(0, cfg.dataset.num_samples):
        json_sample_path = os.path.join(gpt_response_path, f"{i}.json")
        json_sample = json.load(open(json_sample_path))
        keywords_text = ""
        for keywords in json_sample['keywords']:
            keywords_text = keywords_text +  keywords + " "
        llm_responses_keywords.append(keywords_text)
        llm_responses_keywords_labels.append(labels[i])
    return llm_responses_keywords, llm_responses_keywords_labels


def prepare_data_for_keyword_finetuning(cfg):
    # Seeded like every other split: an unseeded re-split is a random split, whose LLM training nodes would
    # overlap the val/test nodes of the (seeded) split the GNN is evaluated on.
    dataset, _, _ = load_graph_dataset_for_tape(cfg.dataset.name, cfg.device, re_split=1,
                                                path_prefix=os.path.dirname(get_dataset_path()), seed=cfg.dataset.seed)
    labels = dataset.y
    texts, labels = load_keywords(cfg, labels)
    tokenizer = get_tokenizer(cfg)

    print("=== PREPARING DATA FOR KEYWORD FINETUNING ===")
    
    # split data with 60/20/20
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
    val_texts = [texts[i] for i in range(len(texts)) if val_mask[i]]
    test_texts = [texts[i] for i in range(len(texts)) if test_mask[i]]
    train_labels = dataset.y[train_mask].tolist()
    val_labels = dataset.y[val_mask].tolist()
    test_labels = dataset.y[test_mask].tolist()

    print(f"Train size: {len(train_labels)}, Val size: {len(val_labels)}, Test size: {len(test_labels)}")
    
    # Check label distribution
    from collections import Counter
    # print(f"Train label distribution: {Counter(train_labels)}")
    # print(f"Val label distribution: {Counter(val_labels)}")
    # print(f"Test label distribution: {Counter(test_labels)}")

    # tokenize keywords texts encoding_texts_to_tokens
    print("Tokenizing texts...")
    train_encodings = encoding_texts_to_tokens(tokenizer, train_texts, cfg.dataset.max_length)
    val_encodings = encoding_texts_to_tokens(tokenizer, val_texts, cfg.dataset.max_length)
    test_encodings = encoding_texts_to_tokens(tokenizer, test_texts, cfg.dataset.max_length)

    # create datasets for train test val and return it.
    train_dataset = TagsDataset(train_encodings, train_labels)
    val_dataset = TagsDataset(val_encodings, val_labels)
    test_dataset = TagsDataset(test_encodings, test_labels)
    
    return train_dataset, val_dataset, test_dataset

    



def prepare_data_for_finetuning(cfg, used_llm_responses=False, used_summary_texts=False, used_paraphrased_texts=False, seed=None, path_prefix='../../../', split=1):
    dataset, _, _ = load_graph_dataset_for_tape(cfg.dataset.name, cfg.device,re_split=split, path_prefix=path_prefix, seed=seed)
    tokenizer = get_tokenizer(cfg)

    print("=== PREPARING DATA ===")

    # Use existing splits if available, otherwise create random splits
    print("Using existing train/val/test splits")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    if used_llm_responses:
        texts = load_gpt_enhanced_explanations(cfg)
    elif used_summary_texts:
        texts = load_summary_explanation(cfg)
    elif used_paraphrased_texts:
        texts = load_paraphrased_explanation(cfg)
    else:
        texts = dataset.raw_texts

    train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
    val_texts = [texts[i] for i in range(len(texts)) if val_mask[i]]
    test_texts = [texts[i] for i in range(len(texts)) if test_mask[i]]
    train_labels = dataset.y[train_mask].tolist()
    val_labels = dataset.y[val_mask].tolist()
    test_labels = dataset.y[test_mask].tolist()

    print(f"Train size: {len(train_labels)}, Val size: {len(val_labels)}, Test size: {len(test_labels)}")

    # Check label distribution
    from collections import Counter
    print(f"Train label distribution: {Counter(train_labels)}")
    print(f"Val label distribution: {Counter(val_labels)}")
    print(f"Test label distribution: {Counter(test_labels)}")

    # Tokenize with more conservative settings
    print("Tokenizing texts...")

    train_encodings = encoding_texts_to_tokens(tokenizer, train_texts, cfg.dataset.max_length)
    val_encodings = encoding_texts_to_tokens(tokenizer, val_texts, cfg.dataset.max_length)
    test_encodings = encoding_texts_to_tokens(tokenizer, test_texts, cfg.dataset.max_length)

    print(
        f"Tokenized shapes - Train: {len(train_encodings['input_ids'])}, Val: {len(val_encodings['input_ids'])}, Test: {len(test_encodings['input_ids'])}")

    # Create dataset objects
    train_dataset = TagsDataset(train_encodings, train_labels)
    val_dataset = TagsDataset(val_encodings, val_labels)
    test_dataset = TagsDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset


def prepare_data_for_finetuning_with_augmented_nodes(cfg, augmented_dataset, original_num_nodes=None,
                                                      used_llm_responses=False, 
                                                      used_summary_texts=False, used_paraphrased_texts=False, 
                                                      seed=None, path_prefix='../../../', split=1):
    """
    Prepare data for finetuning with augmented nodes included.
    
    Args:
        cfg: Configuration object
        augmented_dataset: Dataset object with augmented nodes (from create_augmented_dataset)
        original_num_nodes: Number of nodes in the original dataset (before augmentation).
                           If None, will be calculated by loading the original dataset.
        used_llm_responses: Whether to use LLM responses
        used_summary_texts: Whether to use summary texts
        used_paraphrased_texts: Whether to use paraphrased texts
        seed: Random seed
        path_prefix: Path prefix for datasets
        split: Split number
    
    Returns:
        train_dataset, val_dataset, test_dataset: Datasets with augmented nodes included
    """
    tokenizer = get_tokenizer(cfg)

    print("=== PREPARING DATA WITH AUGMENTED NODES ===")

    # Use existing splits from augmented dataset
    print("Using existing train/val/test splits (including augmented nodes)")
    train_mask = augmented_dataset.train_mask
    val_mask = augmented_dataset.val_mask
    test_mask = augmented_dataset.test_mask

    # Determine original dataset size (before augmentation)
    if original_num_nodes is None:
        # Load the original dataset to determine its size
        from common import load_graph_dataset_for_tape
        original_dataset, _, _ = load_graph_dataset_for_tape(cfg.dataset.name, cfg.device, re_split=split, path_prefix=path_prefix, seed=seed)
        original_num_nodes = len(original_dataset.raw_texts)
    
    if used_llm_responses:
        # For original nodes, use LLM responses if available
        # For augmented nodes, use their original texts
        llm_responses = load_gpt_enhanced_explanations(cfg)
        if llm_responses is not None and len(llm_responses) == original_num_nodes:
            texts = llm_responses + augmented_dataset.raw_texts[original_num_nodes:]
        else:
            # Fallback to original texts if LLM responses not available or mismatch
            texts = augmented_dataset.raw_texts
    elif used_summary_texts:
        summary_texts = load_summary_explanation(cfg)
        if summary_texts is not None and len(summary_texts) == original_num_nodes:
            texts = summary_texts + augmented_dataset.raw_texts[original_num_nodes:]
        else:
            texts = augmented_dataset.raw_texts
    elif used_paraphrased_texts:
        paraphrased_texts = load_paraphrased_explanation(cfg)
        if paraphrased_texts is not None and len(paraphrased_texts) == original_num_nodes:
            texts = paraphrased_texts + augmented_dataset.raw_texts[original_num_nodes:]
        else:
            texts = augmented_dataset.raw_texts
    else:
        texts = augmented_dataset.raw_texts

    train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
    val_texts = [texts[i] for i in range(len(texts)) if val_mask[i]]
    test_texts = [texts[i] for i in range(len(texts)) if test_mask[i]]
    train_labels = augmented_dataset.y[train_mask].tolist()
    val_labels = augmented_dataset.y[val_mask].tolist()
    test_labels = augmented_dataset.y[test_mask].tolist()

    print(f"Train size: {len(train_labels)} (including augmented nodes), Val size: {len(val_labels)}, Test size: {len(test_labels)}")

    # Check label distribution
    from collections import Counter
    print(f"Train label distribution: {Counter(train_labels)}")
    print(f"Val label distribution: {Counter(val_labels)}")
    print(f"Test label distribution: {Counter(test_labels)}")

    # Tokenize with more conservative settings
    print("Tokenizing texts...")

    train_encodings = encoding_texts_to_tokens(tokenizer, train_texts, cfg.dataset.max_length)
    val_encodings = encoding_texts_to_tokens(tokenizer, val_texts, cfg.dataset.max_length)
    test_encodings = encoding_texts_to_tokens(tokenizer, test_texts, cfg.dataset.max_length)

    print(
        f"Tokenized shapes - Train: {len(train_encodings['input_ids'])}, Val: {len(val_encodings['input_ids'])}, Test: {len(test_encodings['input_ids'])}")

    # Create dataset objects
    train_dataset = TagsDataset(train_encodings, train_labels)
    val_dataset = TagsDataset(val_encodings, val_labels)
    test_dataset = TagsDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
    from config import setup_finetuning_cfg
    cfg = setup_finetuning_cfg(dataset_name="pubmed", llm_name="llama_3.2_1B", peft_type="lora")
    tokenizer = get_tokenizer(cfg)
    prepare_data_for_finetuning(cfg, used_llm_responses=True)

