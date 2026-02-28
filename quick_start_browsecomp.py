#!/usr/bin/env python3
"""
AgentFlow Quick Start with BrowseComp-Plus Search Tool

Demonstrates AgentFlow using BrowseComp-Plus for deep research queries.

Usage:
    python quick_start_browsecomp.py
"""

from agentflow.agentflow.solver import construct_solver

# Setup JAVA_HOME for BrowseComp-Plus BM25 search (if using conda Java)
import os
if "JAVA_HOME" not in os.environ:
    default_java = os.path.expanduser("~/miniconda3")
    if os.path.exists(default_java):
        os.environ["JAVA_HOME"] = default_java

# Set the LLM engine name
llm_engine_name = "deepseek-chat"

print("=" * 70)
print("AgentFlow Quick Start with BrowseComp-Plus")
print("=" * 70)
print(f"\nUsing LLM engine: {llm_engine_name}")

# Construct solver with BrowseComp_Search_Tool enabled
solver = construct_solver(
    llm_engine_name=llm_engine_name,
    model_engine=[llm_engine_name, llm_engine_name, llm_engine_name, llm_engine_name],
    enabled_tools=["Base_Generator_Tool", "BrowseComp_Search_Tool", "Google_Search_Tool"],
    tool_engine=[llm_engine_name, "Default", llm_engine_name],
    max_steps=3
)

# Demo query - a BrowseComp-Plus style deep research question
query = "What is the capital of France?"
print(f"\nQuery: {query}")
print("=" * 70)

# Solve
output = solver.solve(query)
print(output["direct_output"])

print("\n" + "=" * 70)
print("✓ BrowseComp-Plus search integrated with AgentFlow!")
print("=" * 70)
