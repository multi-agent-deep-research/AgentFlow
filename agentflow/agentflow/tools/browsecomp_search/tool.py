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

load_dotenv()
logger = logging.getLogger(__name__)

TOOL_NAME = "BrowseComp_Search_Tool"

LIMITATIONS = """
1. This tool searches a fixed corpus of ~100K documents (not the live web).
2. The corpus is static and may not contain the most recent information.
3. Results are limited to the pre-indexed documents.
4. The tool requires a pre-built index (BM25 or FAISS) to be available.
"""

BEST_PRACTICES = """
1. Use specific, targeted queries for better results.
2. The tool works best for factual questions about topics covered in the corpus.
3. For multi-step reasoning, break down complex questions into simpler queries.
4. Use the get_document function to retrieve full document content when needed.
"""

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
        snippet_max_tokens: Maximum tokens for document snippets (None = full text)
        k: Number of results to return
        include_get_document: Whether to include get_document functionality

    Environment Variables:
        BROWSECOMP_INDEX_PATH: Default path to the index (overrides index_path parameter)
        BROWSECOMP_INDEX_TYPE: Default index type ("bm25" or "faiss")
    """

    def __init__(
        self,
        index_type: Optional[str] = None,
        index_path: Optional[str] = None,
        # From BrowseComp-Plus paper: "The retriever tool is set to retrieve the top k=5 search results,
        # where each result is truncated to the first 512 tokens of the corresponding document.
        # This truncation is due to budget constraints, which prevent us from providing full document content."
        # Reference: https://arxiv.org/abs/2501.12959
        snippet_max_tokens: Optional[int] = 512,
        k: int = 5,
        include_get_document: bool = True,
    ):
        """
        Initialize the BrowseComp-Plus search tool.

        Args:
            index_type: Type of index ("bm25" or "faiss")
            index_path: Path to the pre-built index directory
            snippet_max_tokens: Maximum tokens for snippets (None for full text)
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
                f"A search tool for the BrowseComp-Plus benchmark corpus "
                f"(~100K curated documents). Returns top-{k} relevant documents "
                f"with docid, score, and snippet using {index_type.upper()} retrieval."
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
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        # Initialize searcher
        self.searcher = self._create_searcher()
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

    def execute(self, query: str, k: Optional[int] = None) -> str:
        """
        Execute search query against BrowseComp-Plus corpus.

        Args:
            query: The search query
            k: Number of results to return (overrides default)

        Returns:
            Formatted search results as a string
        """
        k = k or self.k

        try:
            results = self.searcher.search(query, k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"

        if not results:
            return "No results found."

        # Format results
        lines = []
        for idx, result in enumerate(results, start=1):
            docid = result.get("docid", "")
            score = result.get("score", 0)
            snippet = result.get("snippet", result.get("text", ""))

            # Truncate snippet if needed
            if self.snippet_max_tokens and snippet:
                snippet = self._truncate_by_tokens(snippet, self.snippet_max_tokens)

            line = f"{idx}. [DocID: {docid}] (Score: {score:.2f})\n   {snippet}"
            lines.append(line)

        return "\n\n".join(lines)

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

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens using simple tokenization."""
        # Simple word-based truncation (approximately token-based)
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens]) + "..."

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
