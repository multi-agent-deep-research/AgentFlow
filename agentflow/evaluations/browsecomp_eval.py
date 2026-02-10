"""
BrowseComp-Plus Evaluation for AgentFlow

This script evaluates AgentFlow on the BrowseComp-Plus benchmark.

Usage:
    from agentflow.evaluations import run_browsecomp_evaluation

    # Run evaluation
    results = run_browsecomp_evaluation(
        index_path="path/to/browsecomp/index",
        output_dir="runs/agentflow_browsecomp",
        model_engine=["trainable", "deepseek-chat", "deepseek-chat", "deepseek-chat"],
    )

    # Format results for leaderboard submission
    format_browsecomp_results("runs/agentflow_browsecomp")

Or via command line:
    python -m agentflow.evaluations browsecomp --index-path path/to/index --output-dir runs/agentflow
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

# Add BrowseComp-Plus to path
browsecomp_path = Path(__file__).parent.parent.parent.parent / "BrowseComp-Plus"
if browsecomp_path.exists():
    sys.path.insert(0, str(browsecomp_path))

import numpy as np
from tqdm import tqdm

from agentflow.solver import construct_solver


# BrowseComp-Plus query template
QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


class BrowseCompEvaluator:
    """
    Evaluator for AgentFlow on BrowseComp-Plus benchmark.
    """

    def __init__(
        self,
        index_path: str,
        index_type: str = "bm25",
        output_dir: str = "runs/agentflow_browsecomp",
        model_engine: List[str] = None,
        max_steps: int = 3,
        max_tokens: int = 512,
        timeout: int = 300,
        temperature: float = 0.0,
    ):
        """
        Initialize the evaluator.

        Args:
            index_path: Path to the BrowseComp-Plus index
            index_type: Type of index ("bm25" or "faiss")
            output_dir: Directory to save results
            model_engine: Model engine configuration for AgentFlow
            max_steps: Maximum search steps
            max_tokens: Maximum tokens per response
            timeout: Timeout per query in seconds
            temperature: LLM temperature
        """
        self.index_path = index_path
        self.index_type = index_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = temperature

        # Default model engine (planner, verifier, executor, etc.)
        if model_engine is None:
            model_engine = ["trainable", "deepseek-chat", "deepseek-chat", "deepseek-chat"]

        self.model_engine = model_engine

        # Tool configuration for BrowseComp-Plus
        self.enabled_tools = ["Base_Generator_Tool"]
        self.tool_engine = ["Default"]

    def load_queries(self, queries_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load BrowseComp-Plus queries.

        Args:
            queries_path: Path to queries TSV file (if None, uses default)

        Returns:
            List of query dictionaries with query_id and query
        """
        if queries_path is None:
            # Try default location
            default_queries = Path(self.index_path).parent.parent / "topics-qrels" / "queries.tsv"
            if default_queries.exists():
                queries_path = str(default_queries)

        if queries_path is None or not Path(queries_path).exists():
            # Download from HuggingFace
            from datasets import load_dataset
            dataset = load_dataset("Tevatron/browsecomp-plus", split="test")
            queries = []
            for record in dataset:
                queries.append({
                    "query_id": record["query_id"],
                    "query": record["query"]  # This will be encrypted
                })
            return queries

        # Load from TSV
        queries = []
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries.append({
                        "query_id": parts[0],
                        "query": parts[1]
                    })
        return queries

    def run_query(self, query: str, solver) -> Dict[str, Any]:
        """
        Run a single query through AgentFlow.

        Args:
            query: The query text
            solver: The AgentFlow solver instance

        Returns:
            Result dictionary with answer, explanation, confidence, tool_calls, etc.
        """
        try:
            result = solver.solve(query)
            return self._parse_result(result, query)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "result": [{"type": "output_text", "output": ""}],
                "retrieved_docids": [],
                "tool_call_counts": {},
            }

    def _parse_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Parse AgentFlow result into BrowseComp-Plus format.

        Args:
            result: Raw AgentFlow result
            query: Original query

        Returns:
            Formatted result dictionary
        """
        # Extract output text
        output_text = ""
        if "direct_output" in result:
            output_text = result["direct_output"]
        elif "final_answer" in result:
            output_text = result["final_answer"]

        # Parse explanation, exact answer, confidence
        explanation = ""
        exact_answer = ""
        confidence = 100

        # Try to extract Explanation
        exp_match = re.search(r'Explanation:\s*(.*?)(?=Exact Answer:|Confidence:|$)', output_text, re.DOTALL | re.IGNORECASE)
        if exp_match:
            explanation = exp_match.group(1).strip()

        # Try to extract Exact Answer
        ans_match = re.search(r'Exact Answer:\s*(.*?)(?=Confidence:|$)', output_text, re.DOTALL | re.IGNORECASE)
        if ans_match:
            exact_answer = ans_match.group(1).strip()

        # Try to extract Confidence
        conf_match = re.search(r'Confidence:\s*(\d+)\s*%', output_text, re.IGNORECASE)
        if conf_match:
            confidence = int(conf_match.group(1))

        # Extract retrieved docids from explanation
        docids = re.findall(r'\[(\d+)\]', explanation)

        # Extract tool calls (if available in result)
        tool_call_counts = defaultdict(int)
        if "tool_calls" in result:
            for tool_call in result["tool_calls"]:
                tool_name = tool_call.get("tool", "unknown")
                tool_call_counts[tool_name] += 1

        return {
            "status": "completed" if output_text else "failed",
            "result": [{"type": "output_text", "output": output_text}],
            "explanation": explanation,
            "exact_answer": exact_answer,
            "confidence": confidence,
            "retrieved_docids": docids,
            "tool_call_counts": dict(tool_call_counts),
        }

    def evaluate(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        num_queries: Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation on BrowseComp-Plus queries.

        Args:
            queries: List of queries (if None, loads from default location)
            num_queries: Maximum number of queries to evaluate (for testing)
            skip_existing: Skip queries that already have results

        Returns:
            Summary dictionary with metrics
        """
        # Load queries
        if queries is None:
            queries = self.load_queries()

        if num_queries:
            queries = queries[:num_queries]

        # Initialize solver
        from agentflow import NamedResources, LLM

        resources = NamedResources()

        # Set up the trainable model endpoint
        llm: LLM = resources.get("main_llm")

        solver = construct_solver(
            llm_engine_name="vllm-" + llm.model,
            enabled_tools=self.enabled_tools,
            tool_engine=self.tool_engine,
            model_engine=self.model_engine,
            output_types="direct",
            max_steps=self.max_steps,
            max_time=self.timeout,
            max_tokens=self.max_tokens,
            base_url=llm.endpoint,
            verbose=False,
            temperature=self.temperature,
        )

        # Run evaluation
        results = []
        completed = 0
        failed = 0

        for query_obj in tqdm(queries, desc="Evaluating"):
            query_id = query_obj["query_id"]
            query_text = query_obj["query"]

            # Check if already exists
            output_file = self.output_dir / f"{query_id}.json"
            if skip_existing and output_file.exists():
                with open(output_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
                    if result.get("status") == "completed":
                        completed += 1
                    else:
                        failed += 1
                continue

            # Run query
            result = self.run_query(query_text, solver)
            result["query_id"] = query_id

            # Save result
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            results.append(result)
            if result.get("status") == "completed":
                completed += 1
            else:
                failed += 1

        # Compute summary
        summary = {
            "total_queries": len(queries),
            "completed": completed,
            "failed": failed,
            "completion_rate": completed / len(queries) if queries else 0,
            "output_dir": str(self.output_dir),
        }

        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


def run_browsecomp_evaluation(
    index_path: str,
    index_type: str = "bm25",
    output_dir: str = "runs/agentflow_browsecomp",
    model_engine: List[str] = None,
    num_queries: Optional[int] = None,
    max_steps: int = 3,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Run AgentFlow evaluation on BrowseComp-Plus.

    Args:
        index_path: Path to BrowseComp-Plus index
        index_type: Type of index ("bm25" or "faiss")
        output_dir: Directory to save results
        model_engine: Model engine configuration
        num_queries: Number of queries to evaluate (None = all)
        max_steps: Maximum search steps per query
        temperature: LLM temperature

    Returns:
        Summary dictionary
    """
    evaluator = BrowseCompEvaluator(
        index_path=index_path,
        index_type=index_type,
        output_dir=output_dir,
        model_engine=model_engine,
        max_steps=max_steps,
        temperature=temperature,
    )

    return evaluator.evaluate(num_queries=num_queries)


def format_browsecomp_results(
    run_dir: str,
    retriever_name: str = "AgentFlow-BM25",
    link: str = "https://github.com/multi-agent-deep-research/AgentFlow",
) -> Dict[str, Any]:
    """
    Format AgentFlow results for BrowseComp-Plus leaderboard submission.

    Args:
        run_dir: Directory containing run results
        retriever_name: Name of the retriever configuration
        link: Link to paper/project

    Returns:
        Formatted submission dictionary
    """
    run_path = Path(run_dir)

    # Load summary
    with open(run_path / "summary.json", 'r') as f:
        summary = json.load(f)

    # Count total tool calls
    tool_stats = defaultdict(float)
    total_queries = 0

    for result_file in run_path.glob("*.json"):
        if result_file.name == "summary.json":
            continue

        with open(result_file, 'r') as f:
            result = json.load(f)

        total_queries += 1
        for tool, count in result.get("tool_call_counts", {}).items():
            tool_stats[tool] += count

    # Average tool stats
    avg_tool_stats = {
        tool: count / total_queries for tool, count in tool_stats.items()
    }

    submission = {
        "LLM": "AgentFlow",  # Can be customized
        "Accuracy (%)": summary.get("completion_rate", 0) * 100,
        "Recall (%)": 0,  # To be filled by BrowseComp evaluation script
        "avg_tool_stats": dict(avg_tool_stats),
        "Calibration Error (%)": 0,  # To be filled by BrowseComp evaluation script
        "Retriever": retriever_name,
        "Link": link,
        "Evaluation Date": datetime.now().strftime("%Y-%m-%d"),
    }

    return submission


def main():
    """CLI entry point for BrowseComp-Plus evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate AgentFlow on BrowseComp-Plus benchmark"
    )
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to BrowseComp-Plus index directory"
    )
    parser.add_argument(
        "--index-type",
        default="bm25",
        choices=["bm25", "faiss"],
        help="Type of index"
    )
    parser.add_argument(
        "--output-dir",
        default="runs/agentflow_browsecomp",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to evaluate (for testing)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum search steps per query"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature"
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Only format existing results for submission"
    )
    parser.add_argument(
        "--retriever-name",
        default="AgentFlow-BM25",
        help="Retriever name for leaderboard"
    )
    parser.add_argument(
        "--link",
        default="https://github.com/multi-agent-deep-research/AgentFlow",
        help="Project link for leaderboard"
    )

    args = parser.parse_args()

    if args.format_only:
        submission = format_browsecomp_results(
            args.output_dir,
            retriever_name=args.retriever_name,
            link=args.link,
        )
        print(json.dumps(submission, indent=2))
    else:
        summary = run_browsecomp_evaluation(
            index_path=args.index_path,
            index_type=args.index_type,
            output_dir=args.output_dir,
            num_queries=args.num_queries,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )

        print("\n" + "=" * 50)
        print("Evaluation Summary:")
        print("=" * 50)
        print(json.dumps(summary, indent=2))

        # Format for submission
        submission = format_browsecomp_results(
            args.output_dir,
            retriever_name=args.retriever_name,
            link=args.link,
        )

        print("\n" + "=" * 50)
        print("Leaderboard Submission Format:")
        print("=" * 50)
        print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
