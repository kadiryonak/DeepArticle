"""
LangGraph workflow definition for the Multi-Agent Academic Paper Analysis System.
Enhanced with Query Analyzer for intelligent search expansion.
"""

from typing import Annotated, List
import operator

from langgraph.graph import StateGraph, START, END

# Import state
import sys
sys.path.insert(0, '..')

from state.system_state import SystemState

# Import agent nodes
from agents.orchestrator import orchestrator_node, should_continue, final_output_node
from agents.query_analyzer import query_analyzer_node
from agents.search_agent import search_agent_node
from agents.metadata_agent import metadata_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.summarizer_agent import summarizer_agent_node
from agents.prioritizer_agent import prioritizer_agent_node
from agents.recommender_agent import recommender_node
from agents.resources_agent import resources_node


def create_workflow():

    # Create the state graph
    workflow = StateGraph(SystemState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("query_analyzer", query_analyzer_node)  # NEW: Query expansion
    workflow.add_node("search", search_agent_node)
    workflow.add_node("metadata", metadata_agent_node)
    workflow.add_node("analysis", analysis_agent_node)
    workflow.add_node("summarizer", summarizer_agent_node)
    workflow.add_node("prioritizer", prioritizer_agent_node)
    workflow.add_node("recommender", recommender_node)  # NEW: study plan & groups
    workflow.add_node("resources", resources_node)      # NEW: GitHub/Medium/YouTube
    workflow.add_node("final", final_output_node)
    
    # Define edges
    # Start -> Orchestrator
    workflow.add_edge(START, "orchestrator")
    
    # Orchestrator -> Query Analyzer (NEW)
    workflow.add_edge("orchestrator", "query_analyzer")
    
    # Query Analyzer -> Search
    workflow.add_edge("query_analyzer", "search")
    
    # Search -> Metadata
    workflow.add_edge("search", "metadata")
    
    # Metadata -> Analysis
    workflow.add_edge("metadata", "analysis")
    
    # Analysis -> Summarizer
    workflow.add_edge("analysis", "summarizer")
    
    # Summarizer -> Prioritizer
    workflow.add_edge("summarizer", "prioritizer")
    
    # Prioritizer -> Recommender -> Resources -> Final
    workflow.add_edge("prioritizer", "recommender")
    workflow.add_edge("recommender", "resources")
    workflow.add_edge("resources", "final")
    
    # Final -> End
    workflow.add_edge("final", END)
    
    # Compile the workflow
    compiled = workflow.compile()
    
    return compiled


def run_analysis(query: str) -> dict:
    # Create initial state
    initial_state = {
        "query": query,
        "search_queries": [],  # Will be populated by query_analyzer
        "topic_analysis": None,
        "papers": [],
        "search_completed": False,
        "metadata_enriched": False,
        "analysis_completed": False,
        "summarization_completed": False,
        "prioritization_completed": False,
        "recommendation_completed": False,
        "resources_completed": False,
        "ranked_papers": [],
        "reading_order": [],
        "groups": {},
        "reading_path": [],
        "start_here": None,
        "resources": {},
        "messages": [],
        "errors": []
    }
    
    # Create and run workflow
    workflow = create_workflow()
    result = workflow.invoke(initial_state)
    
    return result
