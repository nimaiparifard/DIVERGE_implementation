from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset
import torch
from collections import Counter

from dataset.data_utils import get_init_dataset_for_gnn, get_embedding_from_data
from gnns.gnn_mtrainer import gnn_train_and_report

from visualize.visualize_gnn_mistakes import get_gnn_embedding_mistakes


def get_neighborhood_texts_for_mistakes(dataset, mistakes):
    """
    Get neighborhood texts for mistake nodes.
    For each mistake node, create a description with its text and connected neighbors' texts.
    
    Args:
        dataset: PyG dataset with raw_texts and edge_index
        mistakes: Set or list of mistake node indices
        
    Returns:
        Dictionary mapping mistake_node_id to neighborhood description string
    """
    from torch_geometric.utils import to_scipy_sparse_matrix
    import numpy as np
    
    # Convert edge_index to adjacency list for easier neighbor lookup
    edge_index = dataset.edge_index
    num_nodes = dataset.num_nodes
    
    # Create adjacency dictionary
    adj_dict = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src not in adj_dict:
            adj_dict[src] = []
        adj_dict[src].append(dst)
    
    neighborhood_texts = {}
    
    for mistake_idx in mistakes:
        mistake_idx = int(mistake_idx)
        mistake_text = dataset.raw_texts[mistake_idx]
        
        # Get neighbors
        neighbors = adj_dict.get(mistake_idx, [])
        
        # Build neighborhood description
        neighbor_descriptions = []
        for neighbor_idx in neighbors[:10]:  # Limit to first 10 neighbors to avoid too long text
            neighbor_text = dataset.raw_texts[neighbor_idx]
            # Truncate neighbor text if too long
            if len(neighbor_text) > 200:
                neighbor_text = neighbor_text[:200] + "..."
            neighbor_descriptions.append(f"Node {neighbor_idx}: {neighbor_text}")
        
        # Create full description
        neighborhood_desc = f"Mistake Node {mistake_idx} with text: {mistake_text}\n\n"
        neighborhood_desc += f"This node is connected to {len(neighbors)} neighbors. "
        neighborhood_desc += f"Sample neighbors:\n" + "\n".join(neighbor_descriptions)
        
        neighborhood_texts[mistake_idx] = neighborhood_desc
    
    return neighborhood_texts


"""
    enhacne textual features for the intersected mistakes nodes
    like this:
        analyze the neighborhood texts of the nodes.
        give him label of mistakes node and generate why the label it is this.
        also generate 2 paraphrased version of real text.
    save it in json files. in dataset\enhances_texts\{dataset_name}_mistakes_enhanced_texts.json
    {
    mistake_node_id: {
        "analyze_neighborhood": str,
        "paraphrased_versions": [str, str]
        "summary": str,
    }
"""

## write pydantic based model
from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json
import os


class MistakeEnhancedText(BaseModel):
    """Pydantic model for enhanced text of mistake nodes."""
    analyze_neighborhood: str = Field(description="Analysis of the neighborhood texts and why this node might be misclassified")
    label_explanation: str = Field(description="Explanation of why this node has this particular label")
    paraphrased_versions: List[str] = Field(description="Two paraphrased versions of the original node text")
    summary: str = Field(description="A concise summary of the node text and its context")


## write MistakeTextEnhancer class to enhance the intersected mistakes nodes like what i did in textual_enhancer.py
class MistakeTextEnhancer:
    """A class to enhance textual data for mistake nodes using LLMs."""
    
    def __init__(self, dataset_name: str, dataset, mistakes, llm_type: str = "ollama", api_key: str = None):
        """
        Initialize the MistakeTextEnhancer.
        
        Args:
            dataset_name: Name of the dataset
            dataset: PyG dataset object
            mistakes: Set or list of mistake node indices
            llm_type: Type of LLM to use - "ollama" or "openai" (default: "ollama")
            api_key: API key for OpenAI (required if llm_type is "openai")
        """
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.mistakes = list(mistakes)
        self.llm_type = llm_type
        self.parsing_errors_idx = []
        
        # Get neighborhood texts
        self.neighborhood_texts = get_neighborhood_texts_for_mistakes(dataset, mistakes)
        
        # Initialize parser
        self.parser = PydanticOutputParser(pydantic_object=MistakeEnhancedText)
        
        # Create system prompt
        self.system_prompt = """You are an expert text analysis assistant for graph neural network misclassification analysis. 
        
Your task is to analyze nodes that were misclassified by the GNN and provide:
1. An analysis of the neighborhood context and why this node might be hard to classify
2. An explanation of the node's actual label
3. Two different paraphrased versions of the node's text (keep the semantic meaning)
4. A concise summary of the node and its context

{format_instructions}

Be precise and maintain the academic tone of the original text."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Analyze this misclassified node:

Node Text: {node_text}

True Label: {label_name}

Neighborhood Context:
{neighborhood_context}

Please provide a comprehensive analysis.""")
        ])
        
        # Initialize LLM based on type
        if llm_type == "ollama":
            self.llm = ChatOllama(
                model="gemma3:12b",
                base_url="http://localhost:11434",
                temperature=0.7
            )
            print("Using Ollama LLM (gemma3:12b)")
        elif llm_type == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI LLM")
            self.llm = ChatOpenAI(
                model='gpt-4o-mini',
                base_url="https://api.metisai.ir/openai/v1",
                api_key=api_key,
                temperature=0.7
            )
            print("Using OpenAI LLM (gpt-4o-mini)")
        else:
            raise ValueError(f"Invalid llm_type: {llm_type}. Choose 'ollama' or 'openai'")
        
        self.chain = self.prompt | self.llm
    
    def enhance(self):
        """Enhance all mistake nodes."""
        results = {}
        
        for idx in self.mistakes:
            idx = int(idx)
            print(f"Enhancing mistake node {idx}...")
            
            try:
                enhanced = self.enhance_mistake_node(idx)
                results[idx] = enhanced
                print(f"Successfully enhanced node {idx}")
            except Exception as e:
                print(f"Error enhancing node {idx}: {e}")
                self.parsing_errors_idx.append(idx)
                results[idx] = {"error": str(e)}
        
        # Save all results to a single JSON file
        self.save_all_results(results)
        
        return results
    
    def enhance_mistake_node(self, node_idx: int) -> dict:
        """
        Enhance a single mistake node.
        
        Args:
            node_idx: Index of the mistake node
            
        Returns:
            Dictionary with enhanced information
        """
        node_text = self.dataset.raw_texts[node_idx]
        label_idx = self.dataset.y[node_idx].item()
        label_name = self.dataset.label_name[label_idx] if hasattr(self.dataset, 'label_name') else f"Class {label_idx}"
        neighborhood_context = self.neighborhood_texts.get(node_idx, "No neighborhood information available")
        
        # Format and invoke LLM
        formatted_prompt = self.prompt.format_messages(
            format_instructions=self.parser.get_format_instructions(),
            node_text=node_text,
            label_name=label_name,
            neighborhood_context=neighborhood_context
        )
        
        response = self.llm.invoke(formatted_prompt)
        
        try:
            enhanced = self.parser.parse(response.content)
            return {
                "node_id": node_idx,
                "original_text": node_text,
                "true_label": label_name,
                "analyze_neighborhood": enhanced.analyze_neighborhood,
                "label_explanation": enhanced.label_explanation,
                "paraphrased_versions": enhanced.paraphrased_versions,
                "summary": enhanced.summary
            }
        except Exception as parse_error:
            print(f"Parsing error for node {node_idx}: {parse_error}")
            print(f"Raw response: {response.content}")
            self.parsing_errors_idx.append(node_idx)
            return {
                "node_id": node_idx,
                "original_text": node_text,
                "true_label": label_name,
                "raw_response": response.content,
                "error": str(parse_error)
            }
    
    def save_all_results(self, results: dict):
        """Save all enhanced results to a single JSON file."""
        output_dir = self.get_output_dir()
        output_file = os.path.join(output_dir, f"{self.dataset_name}_mistakes_enhanced_texts.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved all results to {output_file}")
    
    def get_output_dir(self):
        """Get the output directory for enhanced texts."""
        from dataset.dataset_loader import get_dataset_path
        datasets_path = get_dataset_path()
        enhanced_dir = os.path.join(datasets_path, "enhanced_texts")
        os.makedirs(enhanced_dir, exist_ok=True)
        return enhanced_dir

def intersect_mistakes(mistake_lists, min_occurrences=None):
    """
    Count how many models misclassified each node and optionally filter by threshold.

    Args:
        mistake_lists: List of lists/sets, where each contains mistake node indices for a model
        min_occurrences: Minimum number of models that must have misclassified a node.
                        If None, returns all nodes that appear in intersection (all models).

    Returns:
        mistake_counts: Dictionary mapping node_idx to number of models that misclassified it
        filtered_mistakes: Set of node indices that were misclassified by at least min_occurrences models
    """
    if not mistake_lists:
        return {}, set()

    # Count occurrences of each mistake across all models
    mistake_counts = {}
    for mistakes in mistake_lists:
        for mistake_idx in mistakes:
            mistake_counts[mistake_idx] = mistake_counts.get(mistake_idx, 0) + 1

    # If no threshold specified, use full intersection (all models must have made the mistake)
    if min_occurrences is None:
        min_occurrences = len(mistake_lists)

    # Filter mistakes by threshold
    filtered_mistakes = {idx for idx, count in mistake_counts.items() if count >= min_occurrences}

    return mistake_counts, filtered_mistakes

# Example usage:
if __name__ == "__main__":
    dataset_name = 'cora'
    llm_name = 'llama_3.2_1B'
    peft_type = 'lora'
    cfg = setup_finetuning_cfg(dataset_name, llm_name, peft_type)
    dataset = load_dataset(cfg)
    labels = dataset.y
    n = 2708
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_pissa, data_orthogonal, data_guassian, data_loftq, data_eva = get_init_dataset_for_gnn(cfg)
    emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian = get_embedding_from_data(
        data_pissa), get_embedding_from_data(data_orthogonal), get_embedding_from_data(
        data_loftq), get_embedding_from_data(data_eva), get_embedding_from_data(data_guassian)
    titles_list = ['PISSA', 'ORTHOGONAL', 'LOFTQ', 'EVA', 'GUASSIAN']

    emb_list = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
    ## train gnn and get mistakes
    results_pissa, model_pissa = gnn_train_and_report(dataset_name, emb_pissa, title="PISSA",
                                                    does_print_training_process=False, reporter=None)
    results_orthogonal, model_orthogonal = gnn_train_and_report(dataset_name, emb_orthogonal, title="ORTHOGONAL",
                                                                does_print_training_process=False, reporter=None)
    results_loftq, model_loftq = gnn_train_and_report(dataset_name, emb_loftq, title="LOFTQ",
                                                    does_print_training_process=False, reporter=None)
    results_eva, model_eva = gnn_train_and_report(dataset_name, emb_eva, title="EVA", does_print_training_process=False,
                                                reporter=None)
    results_guassian, model_guassian = gnn_train_and_report(dataset_name, emb_guassian, title="GUASSIAN",
                                                            does_print_training_process=False, reporter=None)

    model_before_edge_editing_list = [model_pissa, model_orthogonal, model_loftq, model_eva, model_guassian]
    custom_before_edge_editing_embeddings = [emb_pissa, emb_orthogonal, emb_loftq, emb_eva, emb_guassian]
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name=llm_name, peft_type=peft_type)
    dataset = load_dataset(cfg)

    gnn_train_pissa_mistakes = get_gnn_embedding_mistakes(dataset, model_pissa, emb_pissa, output_mask='train')
    gnn_train_orthogonal_mistakes = get_gnn_embedding_mistakes(dataset, model_orthogonal, emb_orthogonal, output_mask='train')
    gnn_train_loftq_mistakes = get_gnn_embedding_mistakes(dataset, model_loftq, emb_loftq, output_mask='train')
    gnn_train_eva_mistakes = get_gnn_embedding_mistakes(dataset, model_eva, emb_eva, output_mask='train')
    gnn_train_guassian_mistakes = get_gnn_embedding_mistakes(dataset, model_guassian, emb_guassian, output_mask='train')

    mistake_lists = [gnn_train_pissa_mistakes,
                    gnn_train_orthogonal_mistakes,
                    gnn_train_loftq_mistakes,
                    gnn_train_eva_mistakes,
                    gnn_train_guassian_mistakes]
    
    # Get mistake counts and filter by threshold
    # Option 1: Get mistakes that occurred in ALL models (strict intersection)
    mistake_counts, intersected_mistakes = intersect_mistakes(mistake_lists, min_occurrences=3)
    
    # Option 2: Get mistakes that occurred in at least 3 models (more flexible)
    # mistake_counts, intersected_mistakes = intersect_mistakes(mistake_lists, min_occurrences=3)
    
    # Report statistics
    print(f"\n=== Mistake Analysis ===")
    print(f"Total unique mistakes across all models: {len(mistake_counts)}")
    print(f"Mistakes that occurred in at least {5} models: {len(intersected_mistakes)}")
    
    # Show distribution of mistakes by occurrence count
    occurrence_distribution = Counter(mistake_counts.values())
    print(f"\nMistake occurrence distribution:")
    for num_models in sorted(occurrence_distribution.keys(), reverse=True):
        count = occurrence_distribution[num_models]
        print(f"  {count} nodes were misclassified by {num_models} model(s)")
    
    print(f"\nNodes selected for enhancement: {len(intersected_mistakes)}")
    print(f"Sample mistake node IDs: {sorted(list(intersected_mistakes))[:10]}...")

    # Initialize the enhancer with intersected mistakes
    # Option 1: Using Ollama (local LLM)
    # enhancer = MistakeTextEnhancer(
    #     dataset_name=dataset_name,
    #     dataset=dataset,
    #     mistakes=intersected_mistakes,
    #     llm_type="ollama"
    # )
    
    # Option 2: Using OpenAI (uncomment and add your API key)
    api_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY in your environment
    enhancer = MistakeTextEnhancer(
        dataset_name=dataset_name,
        dataset=dataset,
        mistakes=intersected_mistakes,
        llm_type="openai",
        api_key=api_key
    )
    
    # Run the enhancement process
    results = enhancer.enhance()
    print(f"\nEnhanced {len(results)} mistake nodes")
    print(f"Parsing errors: {len(enhancer.parsing_errors_idx)}")
    if enhancer.parsing_errors_idx:
        print(f"Nodes with parsing errors: {enhancer.parsing_errors_idx}")
    
