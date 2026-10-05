# search in internet and find papers with specified subject and bring tile and abstaract fo the papers
# based in link predictor and cosine similarity we add it into graphs
# run gnn to see the result
# you should search to find paper wiht label name i give you all finded paper must be unique
"""{
  {
    "doi": 0,
    "title": "",
    "abstract": "",
    "label_id": "",
    "label_name": ""
  }
}"""
# save the format what i say above


import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from config import setup_finetuning_cfg
from dataset.dataset_loader import load_dataset
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import requests
import time
import json


# Define output structure
class Paper(BaseModel):
    doi: str = Field(description="Unique identifier of the item (a DOI for papers; a synthetic id otherwise)")
    title: str = Field(description="Title / name / headline of the item")
    abstract: str = Field(description="Main body text of the item (abstract, description, or post content)")
    label_id: int = Field(description="Numeric id of the label/category")
    label_name: str = Field(description="Name of the label/category")
    source: str = Field(default="generated", description="Source of the item: 'generated' or a web search source name")


class PapersResponse(BaseModel):
    papers: List[Paper] = Field(description="List of unique items for the given label")


# ---------------------------------------------------------------------------
# Per-dataset domain profiles: DIVERGE's datasets are NOT all academic-paper
# citation networks (computer/photo/history are Amazon product co-purchase
# networks, instagram/reddit are social-post networks), so a single generic
# "you are an academic search assistant" prompt + paper-search API doesn't
# fit all of them. Each profile customizes:
#   - persona / field_guidance: how the LLM should frame title/abstract for
#     this domain (kept as the SAME `title`/`abstract`/`doi` schema so the
#     rest of the pipeline - caching, graph augmentation - doesn't change).
#   - web_sources: which free, no-auth-required public APIs are actually
#     relevant for this domain (empty list => no good public source exists,
#     so we fall back to pure LLM generation for that dataset).
# ---------------------------------------------------------------------------
DATASET_PROFILES: Dict[str, dict] = {
    "cora": {
        "domain": "academic",
        "item_noun": "machine learning research papers",
        "persona": (
            "You are an academic search assistant specialized in the Cora citation network domain: "
            "classic machine learning / AI subfields such as Case-Based Reasoning, Genetic Algorithms, "
            "Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, and Learning Theory."
        ),
        "field_guidance": "'title' is the paper title, 'abstract' is a realistic paper abstract (3-8 sentences).",
        "web_sources": ["arxiv", "openalex", "semantic_scholar", "crossref"],
    },
    "citeseer": {
        "domain": "academic",
        "item_noun": "computer science research papers",
        "persona": (
            "You are an academic search assistant specialized in the CiteSeer citation network domain, "
            "spanning Agents, Machine Learning, Information Retrieval, Databases, Human-Computer Interaction, "
            "and Artificial Intelligence."
        ),
        "field_guidance": "'title' is the paper title, 'abstract' is a realistic paper abstract (3-8 sentences).",
        "web_sources": ["arxiv", "openalex", "semantic_scholar", "crossref"],
    },
    "arxiv": {
        "domain": "academic",
        "item_noun": "arXiv computer science preprints",
        "persona": "You are an academic search assistant specialized in arXiv computer science preprints.",
        "field_guidance": "'title' is the preprint title, 'abstract' is the preprint abstract (3-8 sentences).",
        "web_sources": ["arxiv", "openalex", "semantic_scholar", "crossref"],
    },
    "pubmed": {
        "domain": "academic",
        "item_noun": "biomedical research papers",
        "persona": (
            "You are a biomedical research assistant specialized in Diabetes Mellitus research "
            "(Type 1, Type 2, animal models, cellular/molecular mechanisms, immunology, and pharmacology)."
        ),
        "field_guidance": "'title' is the paper title, 'abstract' is a realistic clinical/biomedical abstract (3-8 sentences).",
        "web_sources": ["pubmed", "openalex", "semantic_scholar", "crossref"],
    },
    "wikics": {
        "domain": "academic",
        "item_noun": "Wikipedia computer science articles",
        "persona": "You are simulating Wikipedia articles about computer science concepts and subfields.",
        "field_guidance": "'title' is the article title, 'abstract' is an encyclopedic summary paragraph (3-8 sentences).",
        "web_sources": ["openalex", "semantic_scholar", "crossref"],
    },
    "computer": {
        "domain": "product",
        "item_noun": "Amazon Electronics product listings",
        "persona": "You are simulating entries from an Amazon Electronics product catalog / co-purchase network.",
        "field_guidance": (
            "'title' is the product name, 'abstract' is a product description or customer-review-style "
            "paragraph (3-8 sentences); 'doi' should be a synthetic product id."
        ),
        "web_sources": [],
    },
    "photo": {
        "domain": "product",
        "item_noun": "Amazon Photo & Camera product listings",
        "persona": "You are simulating entries from an Amazon Photo & Camera product catalog / co-purchase network.",
        "field_guidance": (
            "'title' is the product name, 'abstract' is a product description or customer-review-style "
            "paragraph (3-8 sentences); 'doi' should be a synthetic product id."
        ),
        "web_sources": [],
    },
    "history": {
        "domain": "product",
        "item_noun": "Amazon History book listings",
        "persona": "You are simulating entries from an Amazon History-book product catalog / co-purchase network.",
        "field_guidance": (
            "'title' is the book title, 'abstract' is a book blurb or customer-review-style paragraph "
            "(3-8 sentences); 'doi' should be a synthetic product id."
        ),
        "web_sources": [],
    },
    "instagram": {
        "domain": "social",
        "item_noun": "Instagram posts",
        "persona": "You are simulating short Instagram post captions from accounts belonging to a given community/topic.",
        "field_guidance": (
            "'title' is a short subject line for the post, 'abstract' is the caption text "
            "(hashtags allowed if natural); 'doi' should be a synthetic post id."
        ),
        "web_sources": [],
    },
    "reddit": {
        "domain": "social",
        "item_noun": "Reddit posts",
        "persona": "You are simulating Reddit posts (title + body) from a community/topic-specific subreddit.",
        "field_guidance": "'title' is the Reddit post title, 'abstract' is the post body text; 'doi' should be a synthetic post id.",
        "web_sources": ["reddit"],
    },
}

_DEFAULT_PROFILE = {
    "domain": "academic",
    "item_noun": "research papers",
    "persona": "You are an academic search assistant.",
    "field_guidance": "'title' is the paper title, 'abstract' is a realistic paper abstract (3-8 sentences).",
    "web_sources": ["openalex", "semantic_scholar", "crossref"],
}


def get_dataset_profile(dataset_name: str) -> dict:
    return DATASET_PROFILES.get(dataset_name, _DEFAULT_PROFILE)


class DataAugmentor:
    """A class to enhance textual data using LLMs and real web search."""

    def __init__(
        self,
        dataset: str,
        llm_type: str = "ollama",
        api_key: str = None,
        web_search_ratio: float = 0.5,
    ):
        """
        Initialize the TextualEnhancer.

        Args:
            dataset: Name of the dataset to process
            llm_type: Type of LLM to use - "ollama" or "openai" (default: "ollama")
            api_key: API key for OpenAI (required if llm_type is "openai")
            web_search_ratio: Fraction of new papers to get from web search (0.0–1.0).
                Remainder comes from LLM. E.g. 0.3 = 30% web, 70% generated. Default 0.5.
        """
        if not 0 <= web_search_ratio <= 1:
            raise ValueError("web_search_ratio must be between 0 and 1")
        self.dataset_name = dataset
        self.llm_type = llm_type
        self.profile = get_dataset_profile(self.dataset_name)

        # Web-search APIs only make sense for domains with a matching free, no-auth
        # public source (see DATASET_PROFILES); for domains without one (e.g. Amazon
        # product datasets, Instagram) we fall back to pure LLM generation.
        if not self.profile["web_sources"] and web_search_ratio > 0:
            print(
                f"[INFO] No suitable free web-search API for dataset '{self.dataset_name}' "
                f"(domain='{self.profile['domain']}'); forcing web_search_ratio=0.0 (LLM-only generation)."
            )
            web_search_ratio = 0.0
        self.web_search_ratio = web_search_ratio

        self.cfg = setup_finetuning_cfg(dataset_name=self.dataset_name, llm_name="llama_3.2_1B", peft_type="lora")
        self.dataset = load_dataset(self.cfg)
        self.parsing_errors_idx = []

        # Load label keywords from labels/{dataset_name}.json (optional; keys are label_id as str)
        self.label_keywords: Dict[int, List[str]] = self._load_label_keywords()

        # Initialize parser
        self.parser = PydanticOutputParser(pydantic_object=PapersResponse)

        # Create system prompt (persona + field mapping are dataset-specific; see DATASET_PROFILES)
        self.system_prompt = (
            f"{self.profile['persona']}\n\n"
            f"Given a dataset name, a label id, and a label name (category), you must propose a list of "
            f"realistic, unique {self.profile['item_noun']} that match this category.\n\n"
            "For each item you MUST provide:\n"
            "- a plausible unique identifier string in the 'doi' field (fabricate a realistic-looking one if none is known, do NOT return empty),\n"
            f"- {self.profile['field_guidance']}\n"
            "- the numeric label_id that was provided to you,\n"
            "- the label_name exactly as provided.\n\n"
            "All returned items must be unique (no duplicate titles or identifiers).\n\n"
            "{format_instructions}"
        )

        human_template = (
            "Dataset name: {dataset_name}\n"
            "Label id: {label_id}\n"
            "Label name: {label_name}\n"
            "Related keywords (use these to match the topic): {keywords}\n"
            "Number of " + self.profile["item_noun"] + " to return: {num_papers}\n\n"
            "Using your knowledge, produce a list of unique, high-quality " + self.profile["item_noun"] +
            " that match this label/topic and keywords."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", human_template),
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

        # Full chain: prompt -> LLM -> pydantic parser
        self.chain = self.prompt | self.llm | self.parser

    # ---------------- Data management ----------------
    def _load_label_keywords(self) -> Dict[int, List[str]]:
        """
        Load label keywords from labels/{dataset_name}.json.
        File format: { "0": ["keyword1", ...], "1": [...], ... } (keys are label_id as string).
        Returns dict mapping label_id (int) to list of keyword strings. Empty dict if file missing.
        """
        labels_dir = os.path.join(os.path.dirname(__file__), "labels")
        path = os.path.join(labels_dir, f"{self.dataset_name}.json")
        if not os.path.exists(path):
            print(f"[INFO] No label keywords file at {path}; using dataset label names only.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Keys in JSON are strings (e.g. "0", "1"); convert to int
            result = {}
            for k, v in raw.items():
                try:
                    lid = int(k)
                    result[lid] = v if isinstance(v, list) else [str(v)]
                except (ValueError, TypeError):
                    continue
            print(f"[INFO] Loaded keywords for {len(result)} labels from {path}")
            return result
        except Exception as e:
            print(f"[WARN] Failed to load label keywords from {path}: {e}")
            return {}

    def load_existing_papers(self, json_path: str) -> List[dict]:
        """Load existing papers from JSON file if it exists."""
        if not os.path.exists(json_path):
            print(f"[INFO] No existing JSON file found at {json_path}")
            return []
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                papers = json.load(f)
            print(f"[INFO] Loaded {len(papers)} existing papers from {json_path}")
            return papers
        except Exception as e:
            print(f"[ERROR] Failed to load existing JSON: {e}")
            return []

    def count_papers_per_label(self, papers: List[dict]) -> dict:
        """Count how many papers exist for each label_id."""
        counts = {}
        for paper in papers:
            label_id = paper.get("label_id")
            if label_id is not None:
                counts[label_id] = counts.get(label_id, 0) + 1
        return counts

    def check_needs_more_data(self, label_id: int, existing_counts: dict, target_count: int) -> tuple[int, int]:
        """
        Check if a label needs more data.
        Split is determined by self.web_search_ratio (web) and 1 - web_search_ratio (LLM).

        Returns:
            tuple (web_papers_needed, llm_papers_needed)
        """
        current_count = existing_counts.get(label_id, 0)
        if current_count >= target_count:
            return (0, 0)

        remaining = target_count - current_count
        web_needed = int(remaining * self.web_search_ratio)
        llm_needed = remaining - web_needed
        return (web_needed, llm_needed)

    # ---------------- LLM-based generation ----------------
    def generate_papers_for_label(
        self,
        label_id: int,
        label_name: str,
        num_papers: int = 5,
        keywords: List[str] | None = None,
    ) -> List[dict]:
        """
        Generate a list of papers for a given label.

        If keywords are provided (e.g. from labels/{dataset_name}.json), they are included in the prompt.

        Returns a list of dicts with keys:
            - doi
            - title
            - abstract
            - label_id
            - label_name
        """
        keywords_str = ", ".join(keywords) if keywords else label_name.replace("_", " ")
        try:
            result: PapersResponse = self.chain.invoke(
                {
                    "dataset_name": self.dataset_name,
                    "label_id": label_id,
                    "label_name": label_name,
                    "keywords": keywords_str,
                    "num_papers": num_papers,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
        except Exception as e:
            print(f"[ERROR] Failed to generate/parse papers for label '{label_name}' (id={label_id}): {e}")
            self.parsing_errors_idx.append(label_id)
            return []

        papers: List[dict] = []
        seen_doi = set()
        seen_title = set()

        for p in result.papers:
            # Enforce uniqueness by DOI and title
            if p.doi in seen_doi or p.title.lower() in seen_title:
                continue
            seen_doi.add(p.doi)
            seen_title.add(p.title.lower())
            papers.append(
                {
                    "doi": p.doi,
                    "title": p.title,
                    "abstract": p.abstract,
                    "label_id": p.label_id,
                    "label_name": p.label_name,
                    "source": "generated",
                }
            )

        return papers

    # ---------------- Internet search (multi-source with batching) --------

    def _build_query(self, label_name: str, keywords: List[str] | None) -> str:
        if keywords:
            return " ".join(keywords[:6])
        return label_name.replace("_", " ")

    def _extract_papers_from_items(
        self,
        items: List[dict],
        label_id: int,
        label_name: str,
        source_tag: str,
    ) -> List[dict]:
        """Normalize a list of raw API result dicts into our standard format."""
        papers: List[dict] = []
        seen_doi: set = set()
        seen_title: set = set()

        for item in items:
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()

            doi = ""
            external_ids = item.get("externalIds") or {}
            if external_ids:
                doi = external_ids.get("DOI") or external_ids.get("Doi") or ""
            if not doi:
                doi = item.get("doi") or item.get("DOI") or ""
            if not doi:
                doi = f"10.fake/{self.dataset_name}.{label_id}.{len(papers)+1}"

            if not title:
                continue
            if doi in seen_doi or title.lower() in seen_title:
                continue

            seen_doi.add(doi)
            seen_title.add(title.lower())
            papers.append({
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "label_id": label_id,
                "label_name": label_name,
                "source": source_tag,
            })
        return papers

    # ---- OpenAlex (primary) ------------------------------------------------
    def _search_openalex(
        self,
        query: str,
        limit: int,
        label_id: int,
        label_name: str,
    ) -> List[dict]:
        """
        Search OpenAlex Works API.
        Free tier: $10/day budget ≈ 10,000 search calls. 100 req/s hard cap.
        Set OPENALEX_API_KEY env var for authenticated access (free key from openalex.org).
        Polite pool (no key): include mailto in params for higher priority.
        """
        url = "https://api.openalex.org/works"
        params: dict = {
            "search": query,
            "per_page": min(limit, 50),
            "select": "doi,title,abstract_inverted_index",
        }
        api_key = os.environ.get("OPENALEX_API_KEY")
        if api_key:
            params["api_key"] = api_key
        else:
            params["mailto"] = "polite@example.com"

        all_items: List[dict] = []
        page = 1
        remaining = limit

        while remaining > 0:
            params["per_page"] = min(remaining, 50)
            params["page"] = page
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    print(f"  [OpenAlex] 429 rate limit, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [OpenAlex] Error for '{label_name}': {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                title = r.get("title") or ""
                doi = (r.get("doi") or "").replace("https://doi.org/", "")
                abstract = self._rebuild_abstract(r.get("abstract_inverted_index"))
                all_items.append({
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                })

            remaining -= len(results)
            page += 1
            if len(results) < params["per_page"]:
                break
            time.sleep(0.3)

        return self._extract_papers_from_items(all_items, label_id, label_name, "openalex")

    @staticmethod
    def _rebuild_abstract(inverted_index: dict | None) -> str:
        """OpenAlex returns abstracts as inverted index {word: [positions]}. Rebuild to text."""
        if not inverted_index:
            return ""
        word_pos: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_pos.append((pos, word))
        word_pos.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_pos)

    # ---- Semantic Scholar (secondary) --------------------------------------
    def _search_semantic_scholar(
        self,
        query: str,
        limit: int,
        label_id: int,
        label_name: str,
    ) -> List[dict]:
        """
        Search Semantic Scholar with small batches and fast-fail retry.
        Max 2 retries with short backoff so we don't waste minutes waiting.
        """
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        all_items: List[dict] = []
        batch_size = 20
        offset = 0
        remaining = limit

        while remaining > 0:
            fetch = min(remaining, batch_size)
            params = {
                "query": query,
                "limit": fetch,
                "offset": offset,
                "fields": "title,abstract,externalIds",
            }

            success = False
            for attempt in range(1, 3):  # max 2 retries (fast-fail)
                try:
                    resp = requests.get(url, params=params, headers=headers, timeout=15)
                    if resp.status_code in (429, 500):
                        wait = min(int(resp.headers.get("Retry-After", 10)), 30)
                        print(f"  [S2] {resp.status_code} for '{label_name}', attempt {attempt}/2, wait {wait}s")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    success = True
                    break
                except Exception as e:
                    print(f"  [S2] Error for '{label_name}': {e}")
                    break

            if not success:
                break

            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                all_items.append(item)

            remaining -= len(items)
            offset += len(items)
            if len(items) < fetch:
                break
            time.sleep(3.5)

        return self._extract_papers_from_items(all_items, label_id, label_name, "semantic_scholar")

    # ---- CrossRef (tertiary fallback, free, no key) ------------------------
    def _search_crossref(
        self,
        query: str,
        limit: int,
        label_id: int,
        label_name: str,
    ) -> List[dict]:
        """
        Search CrossRef /works endpoint. Free, no API key, 50 req/s.
        Abstracts are not always available but titles and DOIs are reliable.
        """
        url = "https://api.crossref.org/works"
        all_items: List[dict] = []
        batch_size = 20
        offset = 0
        remaining = limit

        while remaining > 0:
            fetch = min(remaining, batch_size)
            params = {
                "query": query,
                "rows": fetch,
                "offset": offset,
                "select": "DOI,title,abstract",
            }
            try:
                resp = requests.get(url, params=params, timeout=20,
                                    headers={"User-Agent": "LLMNodeBed/1.0 (mailto:polite@example.com)"})
                if resp.status_code == 429:
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [CrossRef] Error for '{label_name}': {e}")
                break

            items = data.get("message", {}).get("items", [])
            if not items:
                break

            for item in items:
                title_list = item.get("title", [])
                title = title_list[0] if title_list else ""
                abstract = item.get("abstract") or ""
                doi = item.get("DOI") or ""
                all_items.append({"title": title, "abstract": abstract, "doi": doi})

            remaining -= len(items)
            offset += len(items)
            if len(items) < fetch:
                break
            time.sleep(1)

        return self._extract_papers_from_items(all_items, label_id, label_name, "crossref")

    # ---- arXiv (CS/ML datasets: cora, citeseer, arxiv) ----------------------
    def _search_arxiv(
        self,
        query: str,
        limit: int,
        label_id: int,
        label_name: str,
    ) -> List[dict]:
        """
        Search arXiv's public Atom API. Free, no key required.
        Good fit for cora/citeseer/arxiv since their label taxonomies mirror
        arXiv-era CS/ML subfields.
        """
        import xml.etree.ElementTree as ET

        url = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 50)}
        all_items: List[dict] = []

        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.content)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
                abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
                arxiv_id = (id_el.text or "").strip().rsplit("/", 1)[-1] if id_el is not None else ""
                doi = f"10.48550/arXiv.{arxiv_id}" if arxiv_id else ""
                if title:
                    all_items.append({"title": title, "abstract": abstract, "doi": doi})
        except Exception as e:
            print(f"  [arXiv] Error for '{label_name}': {e}")

        return self._extract_papers_from_items(all_items, label_id, label_name, "arxiv")

    # ---- PubMed (biomedical datasets: pubmed) -------------------------------
    def _search_pubmed(
        self,
        query: str,
        limit: int,
        label_id: int,
        label_name: str,
    ) -> List[dict]:
        """
        Search PubMed via NCBI E-utilities (esearch -> efetch). Free, no key
        required (set PUBMED_API_KEY to raise the rate limit from 3 to 10 req/s).
        """
        import xml.etree.ElementTree as ET

        api_key = os.getenv("PUBMED_API_KEY")
        esearch_params = {"db": "pubmed", "term": query, "retmax": min(limit, 50), "retmode": "json"}
        if api_key:
            esearch_params["api_key"] = api_key

        try:
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=esearch_params, timeout=20,
            )
            resp.raise_for_status()
            pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"  [PubMed] esearch error for '{label_name}': {e}")
            return []

        if not pmids:
            return []

        efetch_params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
        if api_key:
            efetch_params["api_key"] = api_key

        all_items: List[dict] = []
        try:
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=efetch_params, timeout=20,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for article in root.findall(".//PubmedArticle"):
                title_el = article.find(".//ArticleTitle")
                title = (title_el.text or "").strip() if title_el is not None else ""
                abstract = " ".join(
                    (a.text or "").strip() for a in article.findall(".//Abstract/AbstractText") if a.text
                ).strip()
                doi = ""
                for elocation in article.findall(".//ELocationID"):
                    if elocation.get("EIdType") == "doi":
                        doi = (elocation.text or "").strip()
                        break
                if title:
                    all_items.append({"title": title, "abstract": abstract, "doi": doi})
        except Exception as e:
            print(f"  [PubMed] efetch error for '{label_name}': {e}")

        time.sleep(0.1 if api_key else 0.4)  # NCBI rate limit: 10 req/s (with key) vs 3 req/s (unauth)
        return self._extract_papers_from_items(all_items, label_id, label_name, "pubmed")

    # ---- Reddit (social datasets: reddit) -----------------------------------
    def _search_reddit(
        self,
        query: str,
        limit: int,
        label_id: int,
        label_name: str,
    ) -> List[dict]:
        """
        Search Reddit's public read-only search endpoint. No auth required for
        low-volume unauthenticated GETs; only wired up for the 'reddit' dataset.
        """
        url = "https://www.reddit.com/search.json"
        params = {"q": query, "limit": min(limit, 100), "sort": "relevance"}
        headers = {"User-Agent": "diverge-data-augmentor/1.0"}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 429:
                print(f"  [Reddit] Rate limited for '{label_name}', skipping this round.")
                return []
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
        except Exception as e:
            print(f"  [Reddit] Error for '{label_name}': {e}")
            return []

        all_items: List[dict] = []
        for child in children:
            post = child.get("data", {})
            title = (post.get("title") or "").strip()
            body = (post.get("selftext") or "").strip()
            post_id = post.get("id") or ""
            if title:
                all_items.append({"title": title, "abstract": body, "doi": f"reddit:{post_id}"})

        return self._extract_papers_from_items(all_items, label_id, label_name, "reddit")

    # ---- Public entry point: try all sources for this dataset's domain -----
    def search_papers_on_web(
        self,
        label_id: int,
        label_name: str,
        num_papers: int = 5,
        semantic_scholar_api_key: str | None = None,
        keywords: List[str] | None = None,
    ) -> List[dict]:
        """
        Search for real items using the APIs relevant to this dataset's domain
        (see DATASET_PROFILES), with automatic fallback between them.
        Each source is tried until we have enough items or all fail.
        """
        query = self._build_query(label_name, keywords)
        collected: List[dict] = []
        seen_titles: set = set()

        source_registry = {
            "openalex": ("OpenAlex", self._search_openalex),
            "semantic_scholar": ("Semantic Scholar", self._search_semantic_scholar),
            "crossref": ("CrossRef", self._search_crossref),
            "arxiv": ("arXiv", self._search_arxiv),
            "pubmed": ("PubMed", self._search_pubmed),
            "reddit": ("Reddit", self._search_reddit),
        }
        source_keys = self.profile["web_sources"] or ["openalex", "semantic_scholar", "crossref"]
        sources = [source_registry[k] for k in source_keys if k in source_registry]

        for src_name, src_fn in sources:
            still_needed = num_papers - len(collected)
            if still_needed <= 0:
                break
            print(f"    [{src_name}] Searching for {still_needed} papers...")
            try:
                results = src_fn(query, still_needed, label_id, label_name)
            except Exception as e:
                print(f"    [{src_name}] Failed: {e}")
                results = []

            for p in results:
                if p["title"].lower() in seen_titles:
                    continue
                seen_titles.add(p["title"].lower())
                collected.append(p)
                if len(collected) >= num_papers:
                    break
            print(f"    [{src_name}] Got {len(results)} papers (total so far: {len(collected)}/{num_papers})")

        return collected



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate augmented (web-searched / LLM-generated) nodes for a dataset')
    parser.add_argument('--dataset_name', type=str, default='pubmed')
    parser.add_argument('--llm_type', type=str, default='openai', choices=['openai', 'ollama'])
    parser.add_argument('--api_key', type=str, default=None,
                        help="OpenAI-compatible API key. Falls back to the OPENAI_API_KEY env var if omitted.")
    parser.add_argument('--web_search_ratio', type=float, default=0.9,
                        help="Fraction of new items from web search (0.0-1.0); remainder is LLM-generated.")
    parser.add_argument('--target_papers_per_label', type=int, default=1500)
    parser.add_argument('--web_search_delay_seconds', type=float, default=2.0)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    cfg = setup_finetuning_cfg(dataset_name=dataset_name, llm_name="llama_3.2_1B", peft_type="lora")
    dataset = load_dataset(cfg)

    # Fraction of new papers from web search (0.0–1.0); rest from LLM. E.g. 0.3 = 30% web, 70% generated.
    WEB_SEARCH_RATIO = args.web_search_ratio

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if args.llm_type == "openai" and not api_key:
        raise ValueError(
            "No API key provided. Pass --api_key or set the OPENAI_API_KEY environment variable."
        )

    augmentor = DataAugmentor(
        dataset=dataset_name,
        llm_type=args.llm_type,
        api_key=api_key,
        web_search_ratio=WEB_SEARCH_RATIO,
    )

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "augmented_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.json")

    # Load existing papers if available
    all_papers: List[dict] = augmentor.load_existing_papers(output_path)

    # Count existing papers per label
    existing_counts = augmentor.count_papers_per_label(all_papers)
    print(f"[INFO] Existing paper counts per label: {existing_counts}")

    # Target papers per label (adjust as needed)
    TARGET_PAPERS_PER_LABEL = args.target_papers_per_label

    # Delay (seconds) between labels to avoid hammering APIs.
    WEB_SEARCH_DELAY_SECONDS = args.web_search_delay_seconds

    print(f"Split: {WEB_SEARCH_RATIO*100:.0f}% web search, {(1-WEB_SEARCH_RATIO)*100:.0f}% LLM-generated")
    print(f"Generating augmented papers for dataset '{dataset_name}' ...")
    print("=" * 80)

    # Track global uniqueness
    existing_doi = {p["doi"] for p in all_papers}
    existing_title = {p["title"].lower() for p in all_papers}

    for label_id, label_name in enumerate(dataset.label_name):
        print(f"\nLabel {label_id}: {label_name}")
        keywords = augmentor.label_keywords.get(label_id)
        if keywords:
            print(f"  -> Keywords: {keywords[:5]}{'...' if len(keywords) > 5 else ''}")
        
        # Check if we need more data for this label
        web_needed, llm_needed = augmentor.check_needs_more_data(
            label_id, existing_counts, TARGET_PAPERS_PER_LABEL
        )
        
        current_count = existing_counts.get(label_id, 0)
        print(f"  -> Current count: {current_count}/{TARGET_PAPERS_PER_LABEL}")
        
        if web_needed == 0 and llm_needed == 0:
            print(f"  -> Sufficient data already exists. Skipping API calls.")
            continue

        # 1) Real web search (OpenAlex -> Semantic Scholar -> CrossRef fallback)
        web_papers = []
        if web_needed > 0:
            print(f"  -> Fetching {web_needed} papers from web search (multi-source)...")
            web_papers = augmentor.search_papers_on_web(
                label_id=label_id,
                label_name=label_name,
                num_papers=web_needed,
                keywords=keywords,
            )
            print(f"  -> Web search returned: {len(web_papers)} papers")
            time.sleep(WEB_SEARCH_DELAY_SECONDS)

        # 2) LLM-generated papers - only if needed (keywords passed to prompt when available)
        llm_papers = []
        if llm_needed > 0:
            print(f"  -> Generating {llm_needed} papers using LLM...")
            llm_papers = augmentor.generate_papers_for_label(
                label_id=label_id,
                label_name=label_name,
                num_papers=llm_needed,
                keywords=keywords,
            )
            print(f"  -> LLM generation returned: {len(llm_papers)} papers")

        # Merge new papers (web first, then LLM), avoiding duplicates
        new_papers_added = 0
        for p in web_papers + llm_papers:
            if p["doi"] in existing_doi or p["title"].lower() in existing_title:
                continue
            existing_doi.add(p["doi"])
            existing_title.add(p["title"].lower())
            all_papers.append(p)
            new_papers_added += 1
        
        print(f"  -> Added {new_papers_added} new unique papers")

        # Save incrementally after each label to avoid data loss
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_papers, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Final count: {len(all_papers)} total papers")
    print(f"Saved to: {output_path}")
    
    # Print final statistics
    final_counts = augmentor.count_papers_per_label(all_papers)
    print("\nFinal papers per label:")
    for label_id, label_name in enumerate(dataset.label_name):
        count = final_counts.get(label_id, 0)
        print(f"  Label {label_id} ({label_name}): {count} papers")
