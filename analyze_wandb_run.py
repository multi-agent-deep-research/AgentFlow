#!/usr/bin/env python3
"""
Analyze BrowseComp evaluation runs from W&B.

Downloads result artifacts and computes statistics:
- Max/avg context length (chars and estimated tokens)
- Average tool calls per query
- Tool usage distribution
- Accuracy, completion rate
- Step count distribution

Usage:
    # From W&B URL
    python analyze_wandb_run.py https://wandb-radfan.ru/multiagent-deepresearch-improvement/AgentFlow-Pro-Eval-BrowseComp-Plus/runs/h4dh79yg/overview

    # From run ID
    python analyze_wandb_run.py h4dh79yg

    # From local directory
    python analyze_wandb_run.py runs/agentflow_planner7b_qwen35/20260318_144941

    # Compare multiple runs
    python analyze_wandb_run.py h4dh79yg abc123de runs/local_dir
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

WANDB_ENTITY = "multiagent-deepresearch-improvement"
WANDB_PROJECT = "AgentFlow-Pro-Eval-BrowseComp-Plus"


def parse_wandb_url(url_or_id: str) -> str:
    """Extract run ID from W&B URL or return as-is if already an ID."""
    # URL format: https://wandb-radfan.ru/entity/project/runs/RUN_ID/overview
    match = re.search(r'/runs/([a-zA-Z0-9]+)', url_or_id)
    if match:
        return match.group(1)
    # Already a run ID or local path
    return url_or_id


def download_wandb_artifacts(run_id: str, output_dir: str) -> str:
    """Download result artifacts from W&B run. Returns path to results dir."""
    os.environ.setdefault("WANDB_BASE_URL", "https://wandb-radfan.ru")
    import wandb

    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")

    print(f"Run: {run.name}")
    print(f"Config: {json.dumps(dict(run.config), indent=2)}")

    # Download artifacts
    result_dir = Path(output_dir) / run_id
    if result_dir.exists() and any(result_dir.glob("*.json")):
        print(f"Results already cached at {result_dir}")
        return str(result_dir)

    result_dir.mkdir(parents=True, exist_ok=True)

    for artifact in run.logged_artifacts():
        if artifact.type == "evaluation_results":
            print(f"Downloading artifact: {artifact.name}...")
            artifact.download(root=str(result_dir))
            break
    else:
        print("No evaluation_results artifact found")
        sys.exit(1)

    # Save run config
    with open(result_dir / "run_config.json", "w") as f:
        json.dump(dict(run.config), f, indent=2)

    print(f"Downloaded to {result_dir}")
    return str(result_dir)


def load_results(result_dir: str) -> tuple:
    """Load all per-query result JSONs, summary, and judge results from a directory.

    Returns:
        (results_list, summary_dict, judge_dict)
    """
    results = []
    summary = {}
    judge = {}
    result_path = Path(result_dir)

    # Handle nested timestamp dirs
    json_files = list(result_path.glob("*.json"))
    if not any(f.name.isdigit() or f.stem.isdigit() for f in json_files):
        # Try subdirectories (timestamp dirs)
        for subdir in sorted(result_path.iterdir()):
            if subdir.is_dir():
                json_files = list(subdir.glob("*.json"))
                if json_files:
                    result_path = subdir
                    break

    # Load summary and judge results
    summary_path = result_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    judge_path = result_path / "judge_results.json"
    if judge_path.exists():
        with open(judge_path) as f:
            judge = json.load(f)

    for f in sorted(result_path.glob("*.json")):
        if f.name in ("summary.json", "judge_results.json", "run_config.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "query_id" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return results, summary, judge


def estimate_context_length(result: dict) -> dict:
    """Estimate context length for a single query result."""
    memory = result.get("memory", {})
    query_text = result.get("query_text", "")

    # Track cumulative context growth per step
    step_contexts = []
    cumulative_chars = len(query_text)

    for step_name in sorted(memory.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0):
        step = memory[step_name]
        sub_goal = str(step.get("sub_goal", ""))
        command = str(step.get("command", ""))
        result_data = step.get("result", "")
        if isinstance(result_data, list):
            result_str = " ".join(str(r) for r in result_data)
        else:
            result_str = str(result_data)

        step_chars = len(sub_goal) + len(command) + len(result_str)
        cumulative_chars += step_chars
        step_contexts.append(cumulative_chars)

    # Final output context
    final_output = result.get("final_output", "") or ""
    direct_output = result.get("direct_output", "") or ""
    total_chars = cumulative_chars + len(final_output) + len(direct_output)

    return {
        "query_chars": len(query_text),
        "max_step_context_chars": max(step_contexts) if step_contexts else len(query_text),
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,  # rough estimate
        "max_step_context_tokens": (max(step_contexts) if step_contexts else len(query_text)) // 4,
        "num_steps": len(step_contexts),
    }


def analyze_results(results: list, label: str = "", summary: dict = None, judge: dict = None) -> dict:
    """Compute statistics from result list."""
    summary = summary or {}
    judge = judge or {}

    if not results:
        print("No results to analyze")
        return {}

    # Basic counts
    total = len(results)
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = total - completed

    # Tool call counts
    all_tool_calls = []
    search_calls = []
    generate_calls = []
    tool_distribution = defaultdict(int)

    for r in results:
        tc = r.get("tool_call_counts", {})
        total_calls = sum(tc.values())
        all_tool_calls.append(total_calls)
        search_calls.append(tc.get("search", 0))
        generate_calls.append(tc.get("generate", 0))
        for tool, count in tc.items():
            tool_distribution[tool] += count

    # Context lengths
    context_stats = [estimate_context_length(r) for r in results]
    max_context_chars = [c["max_step_context_chars"] for c in context_stats]
    total_chars = [c["total_chars"] for c in context_stats]
    max_context_tokens = [c["max_step_context_tokens"] for c in context_stats]
    num_steps = [c["num_steps"] for c in context_stats]

    # Step counts
    step_counts = [r.get("step_count", 0) for r in results]

    # Execution times
    exec_times = [r.get("execution_time", 0) for r in results if r.get("execution_time", 0) > 0]

    # Retrieved docids
    docid_counts = [len(r.get("retrieved_docids", [])) for r in results]

    # Judge results (from summary if available)
    judge_correct = sum(1 for r in results
                        if r.get("status") == "completed"
                        and "Exact Answer:" in (r.get("final_output", "") or ""))

    stats = {
        "label": label,
        "total_queries": total,
        "completed": completed,
        "failed": failed,
        "completion_rate": completed / total if total > 0 else 0,

        # Tool calls
        "avg_tool_calls": np.mean(all_tool_calls) if all_tool_calls else 0,
        "median_tool_calls": np.median(all_tool_calls) if all_tool_calls else 0,
        "max_tool_calls": max(all_tool_calls) if all_tool_calls else 0,
        "avg_search_calls": np.mean(search_calls) if search_calls else 0,
        "avg_generate_calls": np.mean(generate_calls) if generate_calls else 0,
        "tool_distribution": dict(tool_distribution),

        # Context length (chars)
        "avg_max_context_chars": np.mean(max_context_chars) if max_context_chars else 0,
        "median_max_context_chars": np.median(max_context_chars) if max_context_chars else 0,
        "max_context_chars": max(max_context_chars) if max_context_chars else 0,
        "avg_total_chars": np.mean(total_chars) if total_chars else 0,

        # Context length (estimated tokens)
        "avg_max_context_tokens": np.mean(max_context_tokens) if max_context_tokens else 0,
        "median_max_context_tokens": np.median(max_context_tokens) if max_context_tokens else 0,
        "max_context_tokens": max(max_context_tokens) if max_context_tokens else 0,

        # Steps
        "avg_steps": np.mean(step_counts) if step_counts else 0,
        "median_steps": np.median(step_counts) if step_counts else 0,
        "max_steps": max(step_counts) if step_counts else 0,

        # Execution time
        "avg_exec_time_s": np.mean(exec_times) if exec_times else 0,
        "median_exec_time_s": np.median(exec_times) if exec_times else 0,
        "total_exec_time_min": sum(exec_times) / 60 if exec_times else 0,

        # Retrieved docs
        "avg_retrieved_docids": np.mean(docid_counts) if docid_counts else 0,
        "total_unique_docids": len(set(d for r in results for d in r.get("retrieved_docids", []))),

        # Judge results
        "judge_model": summary.get("judge_model") or judge.get("judge_model", "N/A"),
        "judge_correct": summary.get("judge_correct") or judge.get("correct", 0),
        "judge_total": summary.get("judge_total") or judge.get("total_judged", 0),
        "judge_accuracy": summary.get("judge_accuracy") or judge.get("accuracy", 0),
        "judge_parse_errors": judge.get("parse_errors", 0),
    }

    return stats


def print_stats(stats: dict):
    """Pretty-print statistics."""
    if not stats:
        return

    label = stats.get("label", "")
    header = f"=== Run Statistics: {label} ===" if label else "=== Run Statistics ==="
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")
    if stats.get("wandb_url"):
        print(f"  W&B: {stats['wandb_url']}")

    print(f"\n--- Queries ---")
    print(f"  Total:          {stats['total_queries']}")
    print(f"  Completed:      {stats['completed']}")
    print(f"  Failed:         {stats['failed']}")
    print(f"  Completion:     {stats['completion_rate']:.1%}")

    print(f"\n--- Tool Calls ---")
    print(f"  Avg per query:  {stats['avg_tool_calls']:.1f}")
    print(f"  Median:         {stats['median_tool_calls']:.0f}")
    print(f"  Max:            {stats['max_tool_calls']}")
    print(f"  Avg search:     {stats['avg_search_calls']:.1f}")
    print(f"  Avg generate:   {stats['avg_generate_calls']:.1f}")
    print(f"  Distribution:   {stats['tool_distribution']}")

    print(f"\n--- Context Length ---")
    print(f"  Avg (chars):    {stats['avg_max_context_chars']:,.0f}")
    print(f"  Median (chars): {stats['median_max_context_chars']:,.0f}")
    print(f"  Max (chars):    {stats['max_context_chars']:,.0f}")
    print(f"  Avg total:      {stats['avg_total_chars']:,.0f}")
    print(f"  Avg (~tokens):  {stats['avg_max_context_tokens']:,.0f}")
    print(f"  Median (~tok):  {stats['median_max_context_tokens']:,.0f}")
    print(f"  Max (~tokens):  {stats['max_context_tokens']:,.0f}")

    print(f"\n--- Steps ---")
    print(f"  Avg:            {stats['avg_steps']:.1f}")
    print(f"  Median:         {stats['median_steps']:.0f}")
    print(f"  Max:            {stats['max_steps']}")

    print(f"\n--- Execution Time ---")
    print(f"  Avg per query:  {stats['avg_exec_time_s']:.1f}s")
    print(f"  Median:         {stats['median_exec_time_s']:.1f}s")
    print(f"  Total:          {stats['total_exec_time_min']:.1f} min")

    print(f"\n--- Retrieved Documents ---")
    print(f"  Avg per query:  {stats['avg_retrieved_docids']:.1f}")
    print(f"  Total unique:   {stats['total_unique_docids']}")

    print(f"\n--- Judge Results ---")
    print(f"  Model:          {stats['judge_model']}")
    print(f"  Correct:        {stats['judge_correct']}/{stats['judge_total']}")
    print(f"  Accuracy:       {stats['judge_accuracy']:.1%}")
    print(f"  Parse errors:   {stats['judge_parse_errors']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BrowseComp evaluation runs from W&B or local directories"
    )
    parser.add_argument(
        "runs", nargs="+",
        help="W&B run URLs, run IDs, or local result directories"
    )
    parser.add_argument(
        "--cache-dir", default="wandb_cache",
        help="Directory to cache downloaded artifacts (default: wandb_cache)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output stats as JSON"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save output to a .txt file (in addition to printing)"
    )
    args = parser.parse_args()

    # Tee output to file if --output is specified
    _original_print = print
    output_lines = []

    def tee_print(*pargs, **kwargs):
        _original_print(*pargs, **kwargs)
        import io
        buf = io.StringIO()
        _original_print(*pargs, file=buf, **kwargs)
        output_lines.append(buf.getvalue())

    if args.output:
        import builtins
        builtins.print = tee_print

    all_stats = []

    for run_input in args.runs:
        # Check if local directory
        if os.path.isdir(run_input):
            label = Path(run_input).name
            print(f"\nLoading local results from: {run_input}")
            results, summary, judge = load_results(run_input)
        else:
            # W&B run URL or ID
            run_id = parse_wandb_url(run_input)
            label = run_id
            print(f"\nProcessing W&B run: {run_id}")
            result_dir = download_wandb_artifacts(run_id, args.cache_dir)
            results, summary, judge = load_results(result_dir)

        print(f"Loaded {len(results)} query results")
        stats = analyze_results(results, label=label, summary=summary, judge=judge)
        if not os.path.isdir(run_input):
            run_id = parse_wandb_url(run_input)
            stats["wandb_url"] = f"https://wandb-radfan.ru/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}/overview"
        all_stats.append(stats)

        if args.json:
            # Convert numpy types to Python types for JSON
            json_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in stats.items()}
            print(json.dumps(json_stats, indent=2))
        else:
            print_stats(stats)

    # Comparison table if multiple runs
    if len(all_stats) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        header = f"{'Metric':<30}"
        for s in all_stats:
            header += f" {s['label']:>15}"
        print(header)
        print("-" * (30 + 16 * len(all_stats)))

        metrics = [
            ("Queries", "total_queries", "d"),
            ("Completed", "completed", "d"),
            ("Completion %", "completion_rate", ".1%"),
            ("Avg tool calls", "avg_tool_calls", ".1f"),
            ("Avg search calls", "avg_search_calls", ".1f"),
            ("Avg context (chars)", "avg_max_context_chars", ",.0f"),
            ("Max context (chars)", "max_context_chars", ",.0f"),
            ("Avg context (~tokens)", "avg_max_context_tokens", ",.0f"),
            ("Max context (~tokens)", "max_context_tokens", ",.0f"),
            ("Avg steps", "avg_steps", ".1f"),
            ("Avg time (s)", "avg_exec_time_s", ".1f"),
            ("Avg retrieved docs", "avg_retrieved_docids", ".1f"),
            ("Judge correct", "judge_correct", "d"),
            ("Judge total", "judge_total", "d"),
            ("Judge accuracy", "judge_accuracy", ".1%"),
            ("Judge parse errors", "judge_parse_errors", "d"),
        ]

        for name, key, fmt in metrics:
            row = f"{name:<30}"
            for s in all_stats:
                val = s.get(key, 0)
                row += f" {val:>15{fmt}}"
            print(row)

    # Save to file
    if args.output and output_lines:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(output_lines)
        import builtins
        builtins.print = _original_print
        print(f"\nSaved output to {output_path}")


if __name__ == "__main__":
    main()
