# agentflow/evaluations/__init__.py
"""
Evaluation scripts for AgentFlow benchmarks.

This package contains scripts for evaluating AgentFlow on various benchmarks:
- BrowseComp-Plus: Deep research benchmark with ~100K documents
"""

from .browsecomp_eval import run_browsecomp_evaluation, format_browsecomp_results

__all__ = ['run_browsecomp_evaluation', 'format_browsecomp_results']


def main():
    """CLI entry point for evaluations."""
    from .browsecomp_eval import main as browsecomp_main
    browsecomp_main()


if __name__ == "__main__":
    main()
