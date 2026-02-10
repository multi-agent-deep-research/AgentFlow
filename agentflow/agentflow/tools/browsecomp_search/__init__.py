# agentflow/agentflow/tools/browsecomp_search/__init__.py
"""
BrowseComp-Plus Search Tool for AgentFlow.

This tool integrates the BrowseComp-Plus benchmark corpus (100K curated documents)
into AgentFlow, enabling:
1. Deep research evaluation on a fixed, reproducible corpus
2. Fair comparison with other Deep Research agents
3. Leaderboard submission capability
"""

from .tool import BrowseComp_Search_Tool

__all__ = ['BrowseComp_Search_Tool']
