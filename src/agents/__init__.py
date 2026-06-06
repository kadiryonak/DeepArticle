"""Agents module for the Multi-Agent Academic Paper Analysis System."""

from .search_agent import search_agent_node, search_cs_sources, merge_papers
from .metadata_agent import metadata_agent_node, enrich_paper_metadata
from .analysis_agent import analysis_agent_node
from .summarizer_agent import summarizer_agent_node
from .prioritizer_agent import prioritizer_agent_node, optimize_reading_order
from .orchestrator import orchestrator_node, should_continue, final_output_node

__all__ = [
    # Search
    "search_agent_node",
    "search_cs_sources",
    "merge_papers",
    # Metadata
    "metadata_agent_node",
    "enrich_paper_metadata",
    # Analysis
    "analysis_agent_node",
    # Summarizer
    "summarizer_agent_node",
    # Prioritizer
    "prioritizer_agent_node",
    "optimize_reading_order",
    # Orchestrator
    "orchestrator_node",
    "should_continue",
    "final_output_node"
]
