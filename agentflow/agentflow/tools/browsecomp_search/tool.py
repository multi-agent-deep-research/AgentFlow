"""
BrowseComp-Plus Search Tool for AgentFlow

This tool provides access to the BrowseComp-Plus benchmark corpus (100K curated documents)
for evaluating Deep Research agents. It supports both BM25 and FAISS-based retrieval.

Usage:
    from agentflow.tools.browsecomp_search import BrowseComp_Search_Tool

    # Initialize with BM25 searcher (default)
    tool = BrowseComp_Search_Tool(index_type="bm25", index_path="path/to/index")

    # Or with FAISS searcher
    tool = BrowseComp_Search_Tool(index_type="faiss", index_path="path/to/index")

    # Search for documents
    results = tool.execute(query="What is the capital of France?")
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from dotenv import load_dotenv

from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

load_dotenv()
logger = logging.getLogger(__name__)

TOOL_NAME = "Web_Search_Tool"

LIMITATIONS = """
1. This tool searches a fixed corpus of ~100K curated web documents (not the live web).
2. BM25 retrieval is keyword-based — it may miss relevant documents if query terms don't overlap with document text.
3. A single query often returns only partially relevant results; multiple diverse queries are usually needed.
4. Snippets are truncated — important details may be cut off. Look for key facts in the visible portion.
5. The corpus may not contain a document that directly states the answer; you may need to piece together information from multiple results.
"""

BEST_PRACTICES = """
1. Reformulate the query in multiple ways: use synonyms, related terms, and different phrasings to maximize recall.
2. Extract specific entities, names, dates, or numbers from the question and use them as search keywords.
3. If the first search returns poor results, try a completely different angle — search for related entities or context clues.
4. For questions about specific facts (e.g., "who founded X"), search for the entity name directly rather than the full question.
5. Combine evidence from multiple search results to build your answer — rarely will a single result contain the complete answer.
6. Pay attention to document IDs and scores — higher-scored documents are more likely relevant.
7. Avoid overly broad queries (e.g., "history of science") — be as specific as possible to the question's requirements.
8. If you need a specific number, date, or name, include surrounding context in your query to help locate the right document.
"""

SUMMARIZE_RESULTS_PROMPT = """You are a research assistant. Given a search query and retrieved document snippets, extract and summarize the most relevant information.

Query: {query}

Retrieved Documents:
{documents}

Instructions:
- Focus only on information directly relevant to answering the query
- Preserve specific facts: names, dates, numbers, locations
- Cite document IDs [DocID: X] when referencing information
- If no documents are relevant, say so clearly
- Be concise but preserve all important details

Summary:"""

# Try to import BrowseComp-Plus searcher
try:
    import sys
    # Find BrowseComp-Plus directory - look in multiple possible locations
    # 1. Same level as AgentFlow repo (sibling directory)
    # 2. Inside AgentFlow repo
    # 3. As installed package

    current_file = Path(__file__).resolve()

    # Try sibling directory (BrowseComp-Plus as sibling to AgentFlow)
    repo_root = current_file.parent.parent.parent.parent.parent  # .../agentflow/agentflow/tools/browsecomp_search -> .../AgentFlow/
    sibling_browsecomp = repo_root.parent / "BrowseComp-Plus"  # Go to parent of AgentFlow

    # Try inside AgentFlow
    internal_browsecomp = repo_root / "BrowseComp-Plus"

    if sibling_browsecomp.exists():
        sys.path.insert(0, str(sibling_browsecomp))
        from searcher.searchers.base import BaseSearcher
        from searcher.searchers import SearcherType
        BROWSECOMP_AVAILABLE = True
    elif internal_browsecomp.exists():
        sys.path.insert(0, str(internal_browsecomp))
        from searcher.searchers.base import BaseSearcher
        from searcher.searchers import SearcherType
        BROWSECOMP_AVAILABLE = True
    else:
        # Try importing directly (might be installed as a package)
        from searcher.searchers.base import BaseSearcher
        from searcher.searchers import SearcherType
        BROWSECOMP_AVAILABLE = True
except ImportError:
    BROWSECOMP_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "BrowseComp-Plus searcher not found. "
        "Install it or ensure BrowseComp-Plus is in the parent directory."
    )


class BrowseComp_Search_Tool(BaseTool):
    """
    Search tool for BrowseComp-Plus benchmark corpus.

    This tool provides access to a fixed corpus of ~100K curated documents
    for reproducible Deep Research agent evaluation.

    Attributes:
        index_type: Type of index ("bm25" or "faiss")
        index_path: Path to the pre-built index
        k: Number of results to return
        include_get_document: Whether to include get_document functionality

    Environment Variables:
        BROWSECOMP_INDEX_PATH: Default path to the index (overrides index_path parameter)
        BROWSECOMP_INDEX_TYPE: Default index type ("bm25" or "faiss")
    """

    require_llm_engine = True

    def __init__(
        self,
        model_string: str = "gpt-4o-mini",
        index_type: Optional[str] = None,
        index_path: Optional[str] = None,
        max_chars_per_result: Optional[int] = 512,
        k: int = 5,
        include_get_document: bool = True,
    ):
        """
        Initialize the BrowseComp-Plus search tool.

        Args:
            model_string: Model string for LLM summarization engine
            index_type: Type of index ("bm25" or "faiss")
            index_path: Path to the pre-built index directory
            max_chars_per_result: Maximum characters per result snippet (fallback only)
            k: Number of search results to return
            include_get_document: Whether to support get_document functionality
        """
        if not BROWSECOMP_AVAILABLE:
            raise ImportError(
                "BrowseComp-Plus is not available. Please install it or ensure "
                "the BrowseComp-Plus directory is in the parent directory."
            )

        # Use environment variable as fallback for index path
        default_index_path = os.getenv("BROWSECOMP_INDEX_PATH", "")
        index_path = index_path or default_index_path

        # Use environment variable as fallback for index type
        default_index_type = os.getenv("BROWSECOMP_INDEX_TYPE", "bm25")
        index_type = index_type or default_index_type

        if not index_path:
            raise ValueError(
                "index_path must be provided either as a parameter or via "
                "BROWSECOMP_INDEX_PATH environment variable."
            )

        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                f"A web search tool that retrieves relevant documents from a curated corpus. "
                f"Returns top-{k} relevant documents with docid, score, and snippet."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query",
                "k": "int - Number of results to return (default: 5)",
            },
            output_type="str - Formatted search results with docid, score, and snippet",
            demo_commands=[
                {
                    "command": 'tool.execute(query="What is the capital of France?")',
                    "description": "Search for information about the capital of France.",
                },
                {
                    "command": 'tool.execute(query="Who won the 2024 Olympics?", k=10)',
                    "description": "Search with 10 results instead of default 5.",
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
                "index_type": index_type,
            },
        )

        self.index_type = index_type.lower()
        self.index_path = index_path
        self.max_chars_per_result = max_chars_per_result
        self.k = k
        self.include_get_document = include_get_document
        self.model_string = model_string

        # Initialize searcher
        self.searcher = self._create_searcher()

        # Initialize LLM engine for summarization
        print(f"Initializing BrowseComp Search Tool with summarization model: {self.model_string}")
        self.llm_engine = create_llm_engine(
            model_string=self.model_string,
            is_multimodal=False,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        logger.info(f"Initialized BrowseComp-Plus tool with {index_type} searcher at {index_path}")

    def _create_searcher(self) -> "BaseSearcher":
        """Create the searcher instance based on index type."""
        from argparse import Namespace

        if self.index_type == "bm25":
            args = Namespace(index_path=self.index_path)
            from searcher.searchers.bm25_searcher import BM25Searcher
            return BM25Searcher(args)
        elif self.index_type == "faiss":
            model_name = os.getenv("BROWSECOMP_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
            normalize = os.getenv("BROWSECOMP_NORMALIZE", "true").lower() == "true"

            args = Namespace(
                index_path=self.index_path,
                model_name=model_name,
                normalize=normalize,
                pooling="eos",
                torch_dtype="float16",
                dataset_name="Tevatron/browsecomp-plus-corpus",
                task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                max_length=8192,
                gpu_id=None,
            )
            from searcher.searchers.faiss_searcher import FaissSearcher
            return FaissSearcher(args)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. Use 'bm25' or 'faiss'.")

    def _format_results_raw(self, results: List[Dict], truncate: bool = True) -> str:
        """Format raw search results, optionally truncating snippets."""
        lines = []
        for idx, result in enumerate(results, start=1):
            docid = result.get("docid", "")
            score = result.get("score", 0)
            snippet = result.get("snippet", result.get("text", ""))

            if truncate and self.max_chars_per_result and snippet and len(snippet) > self.max_chars_per_result:
                snippet = snippet[:self.max_chars_per_result] + "..."

            line = f"{idx}. [DocID: {docid}] (Score: {score:.2f})\n   {snippet}"
            lines.append(line)

        return "\n\n".join(lines)

    def execute(self, query: str, k: Optional[int] = None) -> str:
        """
        Execute search query against BrowseComp-Plus corpus.

        Retrieves k results and uses an LLM to summarize them into a focused
        summary preserving key facts (names, dates, numbers, DocIDs).

        Args:
            query: The search query
            k: Number of results to return (overrides default)

        Returns:
            LLM-summarized search results, or truncated raw results on failure
        """
        k = k or self.k

        try:
            results = self.searcher.search(query, k)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Search failed for query '{query[:100]}': {type(e).__name__}: {e}\n{tb}")
            return f"Search failed: {type(e).__name__}: {e}"

        if not results:
            return "No results found."

        # Format full-text results for LLM summarization
        # Cap per-snippet and total size to stay within model context limits
        max_chars_per_snippet = 50_000  # ~12.5K tokens per snippet
        max_total_chars = 200_000      # ~50K tokens total for all docs
        doc_texts = []
        total_chars = 0
        for idx, result in enumerate(results, start=1):
            docid = result.get("docid", "")
            score = result.get("score", 0)
            snippet = result.get("snippet", result.get("text", ""))
            if snippet and len(snippet) > max_chars_per_snippet:
                snippet = snippet[:max_chars_per_snippet] + "... [truncated]"
            doc_text = f"[DocID: {docid}] (Score: {score:.2f})\n{snippet}"
            if total_chars + len(doc_text) > max_total_chars:
                print(f"[BrowseComp] Truncating at {idx-1}/{len(results)} docs ({total_chars} chars) to stay within context limit")
                break
            doc_texts.append(doc_text)
            total_chars += len(doc_text)
        documents_block = "\n\n---\n\n".join(doc_texts)

        # Summarize with LLM
        try:
            prompt = SUMMARIZE_RESULTS_PROMPT.format(query=query, documents=documents_block)
            print(f"[BrowseComp] Summarizing {len(results)} results with LLM ({self.model_string})...")
            summary = self.llm_engine(prompt)
            if summary and summary.strip():
                print(f"[BrowseComp] Summarization successful ({len(summary.strip())} chars)")
                return summary.strip()
            else:
                print("[BrowseComp] Summarization returned empty, falling back to truncated results")
                logger.warning("LLM summarization returned empty result, falling back to raw results")
                return self._format_results_raw(results, truncate=True)
        except Exception as e:
            print(f"[BrowseComp] Summarization failed: {e}, falling back to truncated results")
            logger.warning(f"LLM summarization failed: {e}, falling back to raw results")
            return self._format_results_raw(results, truncate=True)

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.

        Args:
            docid: Document ID to retrieve

        Returns:
            Document dictionary or None if not found
        """
        if not self.include_get_document:
            raise NotImplementedError("get_document is not enabled for this tool instance.")

        return self.searcher.get_document(docid)

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata including searcher information."""
        metadata = super().get_metadata()
        metadata["index_type"] = self.index_type
        metadata["index_path"] = self.index_path
        metadata["default_k"] = self.k
        metadata["include_get_document"] = self.include_get_document
        return metadata


if __name__ == "__main__":
    # Test the tool
    import argparse

    parser = argparse.ArgumentParser(description="Test BrowseComp-Plus Search Tool")
    parser.add_argument("--index-path", type=str, required=True, help="Path to the index")
    parser.add_argument("--index-type", type=str, default="bm25", choices=["bm25", "faiss"])
    parser.add_argument("--query", type=str, default="What is the capital of France?")
    args = parser.parse_args()

    tool = BrowseComp_Search_Tool(
        index_type=args.index_type,
        index_path=args.index_path,
    )

    print("Tool Metadata:")
    print(json.dumps(tool.get_metadata(), indent=2))
    print("\n" + "=" * 50)
    print(f"Query: {args.query}")
    print("=" * 50)

    result = tool.execute(args.query)
    print(result)
