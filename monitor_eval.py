#!/usr/bin/env python3
"""
Monitor parallel BrowseComp evaluation and log aggregated stats to W&B.

Runs alongside parallel workers, periodically scanning their output directories
and logging combined metrics to a single W&B run.

Usage (standalone):
    python monitor_eval.py --run-dir runs/setup_b_parallel/20260326_120000 --interval 60

Automatically started by run_browsecomp_parallel.sh.
"""

import argparse
import json
import os
import re
import sys
import time
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


def scan_workers(run_dir: str) -> dict:
    """Scan all worker directories and aggregate statistics."""
    run_path = Path(run_dir)
    worker_dirs = sorted(run_path.glob("worker_*"))

    if not worker_dirs:
        return {}

    total_completed = 0
    total_failed = 0
    total_correct = 0
    total_judged = 0
    total_parse_errors = 0
    total_search_calls = 0
    total_generate_calls = 0
    total_steps = 0
    total_exec_time = 0.0
    total_queries_with_time = 0
    worker_stats = {}

    for worker_dir in worker_dirs:
        worker_name = worker_dir.name

        # Count result files
        result_files = [f for f in worker_dir.glob("*.json")
                        if f.name not in ("summary.json", "judge_results.json", "run_config.json")]

        completed = 0
        failed = 0
        search_calls = 0
        generate_calls = 0
        steps = 0
        exec_time = 0.0

        for f in result_files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if "query_id" not in data:
                    continue
                if data.get("status") == "completed":
                    completed += 1
                else:
                    failed += 1

                tc = data.get("tool_call_counts", {})
                search_calls += tc.get("search", 0)
                generate_calls += tc.get("generate", 0)
                steps += data.get("step_count", 0)

                et = data.get("execution_time", 0)
                if et > 0:
                    exec_time += et
                    total_queries_with_time += 1
            except (json.JSONDecodeError, KeyError):
                continue

        # Parse judge accuracy from eval.log
        eval_log = worker_dir / "eval.log"
        worker_correct = 0
        worker_judged = 0
        worker_parse_errors = 0

        if eval_log.exists():
            try:
                with open(eval_log) as f:
                    log_text = f.read()

                # Get latest running accuracy line
                acc_matches = re.findall(
                    r'Running judge accuracy: (\d+)/(\d+)',
                    log_text
                )
                if acc_matches:
                    worker_correct = int(acc_matches[-1][0])
                    worker_judged = int(acc_matches[-1][1])

                # Count parse errors
                worker_parse_errors = log_text.count("PARSE_ERROR")
            except Exception:
                pass

        total_completed += completed
        total_failed += failed
        total_correct += worker_correct
        total_judged += worker_judged
        total_parse_errors += worker_parse_errors
        total_search_calls += search_calls
        total_generate_calls += generate_calls
        total_steps += steps
        total_exec_time += exec_time

        worker_stats[worker_name] = {
            "completed": completed,
            "failed": failed,
            "correct": worker_correct,
            "judged": worker_judged,
        }

    total_processed = total_completed + total_failed
    total_tool_calls = total_search_calls + total_generate_calls

    stats = {
        # Aggregate
        "total_processed": total_processed,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "completion_rate": total_completed / total_processed if total_processed > 0 else 0,

        # Judge
        "judge_correct": total_correct,
        "judge_total": total_judged,
        "judge_accuracy": total_correct / total_processed if total_processed > 0 else 0,
        "judge_parse_errors": total_parse_errors,

        # Tool calls
        "total_search_calls": total_search_calls,
        "total_generate_calls": total_generate_calls,
        "avg_tool_calls": total_tool_calls / total_processed if total_processed > 0 else 0,
        "avg_search_calls": total_search_calls / total_processed if total_processed > 0 else 0,

        # Steps
        "avg_steps": total_steps / total_processed if total_processed > 0 else 0,

        # Time
        "avg_exec_time_s": total_exec_time / total_queries_with_time if total_queries_with_time > 0 else 0,
        "total_exec_time_min": total_exec_time / 60,

        # Per worker
        "num_workers": len(worker_dirs),
    }

    # Add per-worker stats
    for wname, wstats in worker_stats.items():
        for k, v in wstats.items():
            stats[f"{wname}/{k}"] = v

    return stats


def main():
    parser = argparse.ArgumentParser(description="Monitor parallel BrowseComp evaluation")
    parser.add_argument("--run-dir", required=True, help="Path to the shared run directory")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds (default: 60)")
    parser.add_argument("--total-queries", type=int, default=830, help="Total queries expected")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (default: auto-generated)")
    parser.add_argument("--config", default=None, help="JSON string of run config to log to W&B")
    args = parser.parse_args()

    run_dir = args.run_dir

    # Initialize W&B
    os.environ.setdefault("WANDB_BASE_URL", "https://wandb-radfan.ru")
    import wandb

    wandb_config = {"run_dir": run_dir, "total_queries": args.total_queries, "monitor": True}
    if args.config:
        try:
            wandb_config.update(json.loads(args.config))
        except json.JSONDecodeError:
            pass

    wandb_name = args.wandb_name or f"monitor_{Path(run_dir).name}"
    wandb.init(
        entity="multiagent-deepresearch-improvement",
        project="AgentFlow-Pro-Eval-BrowseComp-Plus",
        config=wandb_config,
        name=wandb_name,
    )

    print(f"Monitor started for: {run_dir}")
    print(f"Polling every {args.interval}s")
    print(f"W&B run: {wandb.run.get_url()}")
    print()

    try:
        while True:
            stats = scan_workers(run_dir)

            if stats:
                total = stats["total_processed"]
                correct = stats["judge_correct"]
                accuracy = stats["judge_accuracy"]
                progress = total / args.total_queries * 100

                # Print summary
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] {total}/{args.total_queries} ({progress:.0f}%) | "
                    f"accuracy: {correct}/{total} ({accuracy:.1%}) | "
                    f"parse_errors: {stats['judge_parse_errors']} | "
                    f"avg_steps: {stats['avg_steps']:.1f} | "
                    f"avg_time: {stats['avg_exec_time_s']:.0f}s"
                )

                # Log to W&B
                wandb.log(stats)

                # Check if done
                if total >= args.total_queries:
                    print(f"\nAll {args.total_queries} queries processed!")
                    # Log final stats
                    for k, v in stats.items():
                        if "/" not in k:  # skip per-worker stats for final
                            wandb.summary[f"final/{k}"] = v
                    break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    finally:
        wandb.finish()
        print("W&B run finished")


if __name__ == "__main__":
    main()
