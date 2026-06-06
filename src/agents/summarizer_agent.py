"""
Summarizer Agent - Generates concise summaries of papers using LLM.
Supports multiple LLM providers via the LLM factory.
"""

from typing import Dict, Any, List

import sys
sys.path.insert(0, '..')

from config import LLM_MODEL, LLM_PROVIDER
from utils.llm_factory import create_llm


def generate_summary(llm, paper: Dict[str, Any]) -> str:
    """
    Generate a concise summary for a paper using the LLM.
    
    Args:
        llm: The LLM instance
        paper: Paper dictionary with abstract
        
    Returns:
        Generated summary string
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    
    if not abstract:
        return "No abstract available for summarization."
    
    prompt = f"""Analyze this academic paper and provide a concise 2-3 sentence summary highlighting the key findings and significance.

Title: {title}

Abstract: {abstract}

Summary:"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Summary generation failed: {str(e)}"


def summarizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for the Summarizer Agent.
    
    Args:
        state: The current system state
        
    Returns:
        Updated state with paper summaries
    """
    ranked_papers = state.get("ranked_papers", [])
    
    if not ranked_papers:
        return {
            "ranked_papers": [],
            "summarization_completed": True,
            "errors": state.get("errors", []) + ["No papers to summarize"]
        }
    
    print("\n" + "=" * 50)
    print("📝 SUMMARIZER AGENT")
    print("=" * 50)
    
    # Create LLM using factory
    llm = create_llm()
    
    if not llm:
        print("⚠ LLM not available. Using abstracts as summaries.")
        # Fall back to using truncated abstracts
        for paper in ranked_papers:
            abstract = paper.get("abstract", "")
            if len(abstract) > 300:
                paper["summary"] = abstract[:297] + "..."
            else:
                paper["summary"] = abstract or "No summary available."
        
        return {
            "ranked_papers": ranked_papers,
            "summarization_completed": True,
            "messages": state.get("messages", []) + [
                {"role": "system", "content": "Summarization completed using abstracts (LLM unavailable)."}
            ]
        }
    
    provider_info = f"{LLM_PROVIDER}/{LLM_MODEL}" if LLM_PROVIDER else LLM_MODEL
    print(f"Generating summaries using {provider_info}...\n")
    
    # Only summarize top 10 papers to save API calls
    papers_to_summarize = ranked_papers[:10]
    
    for i, paper in enumerate(papers_to_summarize):
        title = paper.get("title", "")[:40]
        print(f"  [{i+1}/{len(papers_to_summarize)}] Summarizing: {title}...")
        
        summary = generate_summary(llm, paper)
        paper["summary"] = summary
        
        # Truncate for display
        display_summary = summary[:80] + "..." if len(summary) > 80 else summary
        print(f"           ✓ {display_summary}")
    
    # For remaining papers, use abstracts
    for paper in ranked_papers[10:]:
        abstract = paper.get("abstract", "")
        if len(abstract) > 300:
            paper["summary"] = abstract[:297] + "..."
        else:
            paper["summary"] = abstract or "No summary available."
    
    print(f"\n✓ Summarization complete")
    
    return {
        "ranked_papers": ranked_papers,
        "summarization_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Generated AI summaries for top {len(papers_to_summarize)} papers."}
        ]
    }
