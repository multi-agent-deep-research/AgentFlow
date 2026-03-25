#!/usr/bin/env python3
"""
Monitor parallel BrowseComp evaluation and log aggregated statistics to W&B.

Watches worker directories for new result files and logs combined metrics
to a single W&B run in real time.

Usage:
    # After starting parallel eval:
    python monitor_eval.py runs/setup_b_parallel/20260326_120000

    # With custom poll interval (default: 30s):
    python monitor_eval.py runs/setup_b_parallel/20260326_120000 --interval 15

    # Without W&B (just print to terminal):
    python monitor_eval.py runs/setup_b_parallel/20260326_120000 --no-wandb
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


def find_run_dir(path: str) -> Path:
    """Find the run directory (containing worker_* subdirs)."""
    p = Path(path)
    # Direct path to timestamp dir with worker_* subdirs
    if any(p.glob("worker_*")):
        return p
    # Path to output_dir — find latest timestamp
    latest_file = p / "latest"
    if latest_file.exists():
        timestamp = latest_file.read_text().strip()
        candidate = p / timestamp
        if candidate.exists():
            return candidate
    # Try latest subdirectory
    subdirs = sorted([d for d in p.iterdir() if d.is_dir()], reverse=True)
    for d in subdirs:
        if any(d.glob("worker_*")):
            return d
    return p


def collect_worker_stats(run_dir: Path) -> dict:
    """Collect statistics from all worker directories."""
    worker_dirs = sorted(run_dir.glob("worker_*"))
    if not worker_dirs:
        return {}

    total_completed = 0
    total_failed = 0
    total_correct = 0
    total_judged = 0
    total_parse_errors = 0
    tool_calls = defaultdict(int)
    all_exec_times = []
    all_step_counts = []
    worker_stats = {}

    for worker_dir in worker_dirs:
        worker_name = worker_dir.name
        completed = 0
        failed = 0
        correct = 0
        judged = 0

        # Count result files
        for f in worker_dir.glob("*.json"):
            if f.name in ("summary.json", "judge_results.json", "run_config.json"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if "query_id" not in data:
                    continue
                if data.get("status") == "completed":
                    completed += 1
                else:
                    failed += 1

                # Tool calls
                tc = data.get("tool_call_counts", {})
                for tool, count in tc.items():
                    tool_calls[tool] += count

                # Execution time
                et = data.get("execution_time", 0)
                if et > 0:
                    all_exec_times.append(et)

                # Step count
                sc = data.get("step_count", 0)
                if sc > 0:
                    all_step_counts.append(sc)

            except (json.JSONDecodeError, KeyError):
                continue

        # Parse judge accuracy from eval.log
        eval_log = worker_dir / "eval.log"
        if eval_log.exists():
            try:
                with open(eval_log) as f:
                    content = f.read()
                # Get last "Running judge accuracy" line
                matches = re.findall(
                    r'Running judge accuracy: (\d+)/(\d+)',
                    content
                )
                if matches:
                    correct = int(matches[-1][0])
                    judged = int(matches[-1][1])

                # Count parse errors
                pe = len(re.findall(r'PARSE_ERROR', content))
                total_parse_errors += pe
            except Exception:
                pass

        total_completed += completed
        total_failed += failed
        total_correct += correct
        total_judged += judged

        worker_stats[worker_name] = {
            "completed": completed,
            "failed": failed,
            "correct": correct,
            "judged": judged,
        }

    total_processed = total_completed + total_failed
    import numpy as np

    stats = {
        # Aggregate
        "total_processed": total_processed,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "completion_rate": total_completed / total_processed if total_processed > 0 else 0,
        "judge_correct": total_correct,
        "judge_total": total_judged,
        "judge_accuracy": total_correct / total_processed if total_processed > 0 else 0,
        "judge_parse_errors": total_parse_errors,

        # Tool calls
        "total_search_calls": tool_calls.get("search", 0),
        "total_generate_calls": tool_calls.get("generate", 0),
        "avg_tool_calls": (sum(tool_calls.values()) / total_processed) if total_processed > 0 else 0,

        # Execution time
        "avg_exec_time_s": float(np.mean(all_exec_times)) if all_exec_times else 0,
        "median_exec_time_s": float(np.median(all_exec_times)) if all_exec_times else 0,

        # Steps
        "avg_steps": float(np.mean(all_step_counts)) if all_step_counts else 0,

        # Per-worker
        "num_workers": len(worker_dirs),
        "worker_stats": worker_stats,
    }

    return stats


def print_stats(stats: dict, prev_processed: int = 0):
    """Print a compact summary to terminal."""
    total = stats["total_processed"]
    correct = stats["judge_correct"]
    accuracy = stats["judge_accuracy"]
    rate = total - prev_processed

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Processed: {total}/830 | "
          f"Correct: {correct}/{total} ({accuracy:.1%}) | "
          f"Parse errors: {stats['judge_parse_errors']} | "
          f"Avg steps: {stats['avg_steps']:.1f} | "
          f"Avg time: {stats['avg_exec_time_s']:.0f}s | "
          f"+{rate} new")

    # Per-worker summary
    for name, ws in sorted(stats["worker_stats"].items()):
        total_w = ws["completed"] + ws["failed"]
        print(f"  {name}: {total_w} done, {ws['correct']}/{ws['judged']} correct")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor parallel BrowseComp eval and log to W&B"
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run directory (e.g. runs/setup_b_parallel/20260326_120000)"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Poll interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (terminal only)"
    )
    parser.add_argument(
        "--wandb-name", default=None,
        help="Custom W&B run name (default: monitor_<timestamp>)"
    )
    args = parser.parse_args()

    run_dir = find_run_dir(args.run_dir)
    print(f"Monitoring: {run_dir}")

    # Wait for worker dirs to appear
    while not any(run_dir.glob("worker_*")):
        print(f"Waiting for worker directories in {run_dir}...")
        time.sleep(5)

    worker_count = len(list(run_dir.glob("worker_*")))
    print(f"Found {worker_count} workers")

    # Try to read config from first worker's eval.log
    config = {}
    for worker_dir in sorted(run_dir.glob("worker_*")):
        eval_log = worker_dir / "eval.log"
        if eval_log.exists():
            with open(eval_log) as f:
                content = f.read()
            cmd_match = re.search(r'Command: (.*)', content)
            if cmd_match:
                config["command"] = cmd_match.group(1)
            # Extract model info from log
            model_match = re.search(r'Model \(all components\): (\S+)', content)
            if model_match:
                config["model"] = model_match.group(1)
            judge_match = re.search(r'Judge: (\S+)', content)
            if judge_match:
                config["judge_model"] = judge_match.group(1)
            steps_match = re.search(r'Max steps: (\d+)', content)
            if steps_match:
                config["max_steps"] = int(steps_match.group(1))
            break

    config["num_workers"] = worker_count
    config["run_dir"] = str(run_dir)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Initialize W&B
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_BASE_URL", "https://wandb-radfan.ru")

            timestamp = run_dir.name
            wandb_name = args.wandb_name or f"monitor_{timestamp}_{worker_count}w"

            wandb_run = wandb.init(
                entity="multiagent-deepresearch-improvement",
                project="AgentFlow-Pro-Eval-BrowseComp-Plus",
                config=config,
                name=wandb_name,
                tags=["monitor", f"{worker_count}workers"],
            )
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed: {e}, continuing without W&B")
            wandb_run = None

    # Monitor loop
    prev_processed = 0
    try:
        while True:
            stats = collect_worker_stats(run_dir)
            if not stats:
                print("No stats yet, waiting...")
                time.sleep(args.interval)
                continue

            print_stats(stats, prev_processed)

            # Log to W&B with total_processed as x-axis
            if wandb_run is not None:
                import wandb
                step = stats["total_processed"]
                if step > 0 and step != prev_processed:
                    wandb.log({
                        "total_processed": step,
                        "total_completed": stats["total_completed"],
                        "total_failed": stats["total_failed"],
                        "completion_rate": stats["completion_rate"],
                        "judge_correct": stats["judge_correct"],
                        "judge_accuracy": stats["judge_accuracy"],
                        "judge_parse_errors": stats["judge_parse_errors"],
                        "avg_tool_calls": stats["avg_tool_calls"],
                        "total_search_calls": stats["total_search_calls"],
                        "total_generate_calls": stats["total_generate_calls"],
                        "avg_exec_time_s": stats["avg_exec_time_s"],
                        "median_exec_time_s": stats["median_exec_time_s"],
                        "avg_steps": stats["avg_steps"],
                    }, step=step)

            prev_processed = stats["total_processed"]

            # Check if all done (830 queries)
            if stats["total_processed"] >= 830:
                print("\n=== All 830 queries processed! ===")
                print(f"Final accuracy: {stats['judge_correct']}/{stats['total_processed']} ({stats['judge_accuracy']:.1%})")
                print(f"Parse errors: {stats['judge_parse_errors']}")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        if wandb_run is not None:
            # Log final summary
            import wandb
            wandb.log({
                "final/total_processed": stats.get("total_processed", 0),
                "final/judge_correct": stats.get("judge_correct", 0),
                "final/judge_accuracy": stats.get("judge_accuracy", 0),
                "final/judge_parse_errors": stats.get("judge_parse_errors", 0),
                "final/avg_steps": stats.get("avg_steps", 0),
                "final/avg_exec_time_s": stats.get("avg_exec_time_s", 0),
            })
            wandb.finish()
            print("W&B run finished.")


if __name__ == "__main__":
    main()
