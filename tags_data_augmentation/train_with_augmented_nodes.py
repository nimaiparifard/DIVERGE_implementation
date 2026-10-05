import torch
from dataset.dataset_loader import load_dataset
from common.dataloader import is_seeded_split, compute_split

from config import setup_finetuning_cfg
from report.reporter import ReportResults
from train_llm.train import Train
from adapter_dir import get_semi_supervised_augmented_adapter_dir
from tags_data_augmentation.agumented_dataset import load_augmented_texts

def create_dataset_for_augmented_llm_training(dataset, augmented_texts, augmented_labels):
    """
    Create a modified dataset for LLM training with augmented nodes.
    
    This function adds augmented texts and labels to the dataset without modifying edges.
    All augmented nodes are added to the training mask (not validation or test).
    
    Args:
        dataset: Original dataset object with fields:
            - raw_texts: list of strings
            - y: torch.Tensor of labels
            - train_mask, val_mask, test_mask: torch.Tensor boolean masks
            - edge_index: torch.Tensor (not modified)
        augmented_texts: List of dictionaries with 'title' and 'abstract' fields
        augmented_labels: torch.Tensor of labels for augmented nodes [num_augmented_nodes]
    
    Returns:
        modified_dataset: Dataset object with augmented nodes added
    """
    # Clone the dataset to avoid modifying the original
    modified_dataset = dataset.clone()
    modified_dataset.num_base_nodes = int(dataset.y.shape[0])
    
    # Get device from dataset
    device = dataset.y.device if torch.is_tensor(dataset.y) else 'cpu'
    
    # Ensure augmented_labels is a tensor on the correct device
    if not torch.is_tensor(augmented_labels):
        augmented_labels = torch.tensor(augmented_labels, dtype=torch.long, device=device)
    else:
        augmented_labels = augmented_labels.to(device)
    
    num_augmented_nodes = len(augmented_texts)
    
    # Format augmented texts: "Title: {title}\nAbstract: {abstract}"
    new_raw_texts = [
        f"Title: {item['title']}\nAbstract: {item['abstract']}"
        for item in augmented_texts
    ]
    
    # Add augmented texts to raw_texts
    modified_dataset.raw_texts = dataset.raw_texts + new_raw_texts
    
    # Add augmented labels to y
    modified_dataset.y = torch.cat([dataset.y, augmented_labels], dim=0)
    
    # Update masks: augmented nodes go to train mask only
    new_train_mask = torch.ones(num_augmented_nodes, dtype=torch.bool, device=device)
    modified_dataset.train_mask = torch.cat([dataset.train_mask, new_train_mask], dim=0)
    
    new_val_mask = torch.zeros(num_augmented_nodes, dtype=torch.bool, device=device)
    modified_dataset.val_mask = torch.cat([dataset.val_mask, new_val_mask], dim=0)
    
    new_test_mask = torch.zeros(num_augmented_nodes, dtype=torch.bool, device=device)
    modified_dataset.test_mask = torch.cat([dataset.test_mask, new_test_mask], dim=0)
    
    # Update few-shot masks if they exist
    for mask_name in ['one_shot_train', 'three_shot_train', 'five_shot_train',
                      'one_shot_val', 'three_shot_val', 'five_shot_val',
                      'one_shot_test', 'three_shot_test', 'five_shot_test']:
        if hasattr(dataset, mask_name):
            new_mask = torch.zeros(num_augmented_nodes, dtype=torch.bool, device=device)
            setattr(modified_dataset, mask_name,
                    torch.cat([getattr(dataset, mask_name), new_mask], dim=0))
    
    print(f"[OK] Created augmented dataset for LLM training:")
    print(f"  Original nodes: {dataset.y.shape[0]}")
    print(f"  Augmented nodes: {num_augmented_nodes}")
    print(f"  Total nodes: {modified_dataset.y.shape[0]}")
    print(f"  Train nodes: {modified_dataset.train_mask.sum().item()}")
    print(f"  Val nodes: {modified_dataset.val_mask.sum().item()}")
    print(f"  Test nodes: {modified_dataset.test_mask.sum().item()}")
    
    return modified_dataset





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune an LLM adapter jointly on base graph nodes + augmented nodes')
    parser.add_argument('--dataset_name', type=str, default='citeseer')
    parser.add_argument('--llm_name', type=str, default='llama_3.2_1B')
    parser.add_argument('--peft_type', type=str, default='lora')
    parser.add_argument('--init_weight_approaches', type=str, nargs='+',
                        default=['pissa', 'orthogonal', 'eva', 'loftq', 'gaussian'])
    args = parser.parse_args()

    dataset_name = args.dataset_name
    llm_name = args.llm_name
    for init_weight in args.init_weight_approaches:
        cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=args.peft_type)
        cfg.peft.init_lora_weights = init_weight
        augmented_semi_supervised_adapter_dir = get_semi_supervised_augmented_adapter_dir(cfg)

        # Load original dataset
        dataset = load_dataset(cfg)
        if is_seeded_split(dataset_name, 0):
            # The GNN uses the seeded split of these nodes (common.dataloader.compute_split); the stored
            # .pt masks differ, and fine-tuning on them would put LLM training nodes in the GNN test set.
            split_masks = compute_split(dataset_name, int(dataset.y.shape[0]), 0, 'cpu', seed=cfg.dataset.seed)
            dataset.train_mask, dataset.val_mask, dataset.test_mask = split_masks
        print(f"Original dataset: {dataset}")

        # Load augmented texts from JSON file (label_id is already stored per item;
        # no embedding cache is needed here since this step runs BEFORE fine-tuning)
        augmented_texts = load_augmented_texts(dataset_name)
        augmented_labels = torch.tensor(
            [int(item['label_id']) for item in augmented_texts], dtype=torch.long
        )

        # Create modified dataset with augmented nodes
        modified_dataset = create_dataset_for_augmented_llm_training(
            dataset=dataset,
            augmented_texts=augmented_texts,
            augmented_labels=augmented_labels
        )

        # Setup reporter and trainer (use_augmented_nodes=True so Train uses prepare_data_for_finetuning_with_augmented_nodes)
        reporter = ReportResults(cfg, save_dir='results/train_info/test_run_semi_supervised_augmented')
        train = Train(
            cfg,
            save_dir=augmented_semi_supervised_adapter_dir,
            reporter=reporter,
            used_llm_responses=True,
            split=0,
            use_augmented_nodes=True,
            augmented_dataset=modified_dataset,
        )
        train.process()