import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from LLMReasoner.TAPE.peft_tape_finetuning.config import setup_finetuning_cfg
from LLMReasoner.TAPE.peft_tape_finetuning.dataset_loader import load_dataset
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json


# Define output structure
class EnhancedText(BaseModel):
    summary: str = Field(description="A concise summary of the text")
    keywords: List[str] = Field(description="List of important keywords extracted from the text")
    paraphrased_text: str = Field(description="A paraphrased version of the original text")

class TextualEnhancer:
    """A class to enhance textual data using LLMs."""
    def __init__(self, dataset: str, llm_type: str = "ollama", api_key: str = None):
        """
        Initialize the TextualEnhancer.
        
        Args:
            dataset: Name of the dataset to process
            llm_type: Type of LLM to use - "ollama" or "openai" (default: "ollama")
            api_key: API key for OpenAI (required if llm_type is "openai")
        """
        self.dataset_name = dataset
        self.llm_type = llm_type
        self.cfg = setup_finetuning_cfg(dataset_name=self.dataset_name, llm_name="llama_3.2_1B", peft_type="lora")
        self.dataset = load_dataset(self.cfg)
        self.texts = self.dataset.raw_texts
        self.parsing_errors_idx = []
        
        # Initialize parser
        self.parser = PydanticOutputParser(pydantic_object=EnhancedText)

        # Create system prompt
        self.system_prompt = """You are an expert text enhancement assistant. Your task is to analyze academic paper abstracts and provide:
        1. A concise summary of the main points
        2. A list of important keywords
        3. A paraphrased version of the text

        {format_instructions}

        Be precise and maintain the academic tone of the original text."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Analyze the following text:\n\n{text}")
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
        for idx in range(len(self.texts)):
            enhanced_text_path = self.get_enhanced_dir()
            enhanced_text_path_idx = os.path.join(enhanced_text_path, f'{idx}.json')
            if not os.path.exists(enhanced_text_path_idx):
                result = self.enhance_text(self.texts[idx], idx)
                print(f"Got the result from llm for: {idx}")
                self.save_text(result, idx)
            else:
                print(f"Enhanced text already exists for: {idx}, skipping...")

    def enhance_text(self, text: str, idx) -> dict:
        formatted_prompt = self.prompt.format_messages(
            format_instructions=self.parser.get_format_instructions(),
            text=text
        )
        response = self.llm.invoke(formatted_prompt)
        try:
            enhanced = self.parser.parse(response.content)
            return {
                "summary": enhanced.summary,
                "keywords": enhanced.keywords,
                "paraphrased_text": enhanced.paraphrased_text
            }
        except Exception as parse_error:
            print(f"Parsing error: {parse_error}")
            print(f"Raw response: {response.content}")
            self.parsing_errors_idx.append(idx)
            return {"raw_response": response.content}

    def save_text(self, enhanced_text, idx):
        enhanced_text_path = self.get_enhanced_dir()
        enhanced_text_path_idx = os.path.join(enhanced_text_path, f'{idx}.json')
        # save json file
        with open(enhanced_text_path_idx, 'w', encoding='utf-8') as f:
            json.dump(enhanced_text, f, indent=2)

    def get_enhanced_dir(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
        enhanced_dir = os.path.join(repo_root, "datasets", "enhanced_texts")
        os.makedirs(enhanced_dir, exist_ok=True)
        enhanced_database_dir = os.path.join(enhanced_dir, self.dataset_name)
        os.makedirs(enhanced_database_dir, exist_ok=True)
        return enhanced_database_dir

    def enhance_mistake_idx(self, idx):
        # enhance text and the file is json file is exist but with wrong answere api to openai and rewrite the json file
        text = self.texts[idx]
        enhanced_text_path_idx = os.path.join(self.get_enhanced_dir(), f'{idx}.json')

        # If a file exists, try to read and validate it. If it's invalid or unreadable,
        # re-run the enhancement and overwrite the file.
        if os.path.exists(enhanced_text_path_idx):
            try:
                with open(enhanced_text_path_idx, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                # Basic validation: must contain the expected keys
                if all(k in existing for k in ("summary", "keywords", "paraphrased_text")) and isinstance(existing.get("keywords"), list):
                    print(f"Existing enhanced file for idx={idx} looks valid — skipping re-enhance.")
                    return existing
                else:
                    print(f"Existing enhanced file for idx={idx} is invalid or incomplete — re-enhancing.")
            except Exception as e:
                print(f"Could not read existing enhanced file for idx={idx}: {e}. Will re-enhance.")

        # Call enhancer and overwrite the file with a fresh enhancement
        print(f"Re-enhancing text at idx={idx}...")
        enhanced = self.enhance_text(text, idx)
        self.save_text(enhanced, idx)
        return enhanced

# Process sample text
if __name__ == "__main__":
    # Example 1: Using Ollama (default)
    # print("=" * 80)
    # print("EXAMPLE 1: Using Ollama LLM")
    # print("=" * 80)
    # enhancer_ollama = TextualEnhancer(dataset="cora", llm_type="ollama")
    #
    # # Process just the first text as a sample
    # sample_text = enhancer_ollama.texts[0]
    # print(f"\nOriginal text: {sample_text[:200]}...\n")
    #
    # result = enhancer_ollama.enhance_text(sample_text, idx=0)
    # print("\nEnhanced Text Result:")
    # print(json.dumps(result, indent=2))
    #
    # # Save the result
    # enhancer_ollama.save_text(result, idx=0)
    # print("\n" + "=" * 80)
    
    # Example 2: Using OpenAI (uncomment and add your API key to use)
    print("\nOpenAI LLM")
    print("=" * 80)
    api_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY in your environment
    enhancer_openai = TextualEnhancer(dataset="cora", llm_type="openai", api_key=api_key)
    # result_openai = enhancer_openai.enhance_text(enhancer_openai.texts[0], idx=0)
    # print("\nEnhanced Text Result (OpenAI):")
    # print(json.dumps(result_openai, indent=2))
    # enhancer_openai.save_text(result_openai, idx=0)
    
    # To process all texts in the dataset, uncomment:
    list_of_wrong_json_extracted = [739, 735, 725, 702, 699, 691, 668, 656, 653, 649, 648, 639, 632, 612, 597, 593, 590, 586, 560, 558,
 557, 548, 482, 481, 478, 476, 474, 470, 467, 465, 462, 459, 455, 449, 435, 402, 396, 395, 387, 379, 364
 ,363, 334, 328, 318, 310, 298, 262, 256, 234, 216, 209, 204, 195, 194, 190, 186, 158, 140, 137, 135,
 86, 81, 72, 65, 63,30, 20, 16, 10]
    for wrong_idx in range(2708):
        enhancer_openai.enhance_mistake_idx(wrong_idx)
    # enhancer_openai.enhance()
    