

from typing import Dict, Any


def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:

    query = state.get("query", "")
    

    print(" MULTI-AGENT ACADEMIC PAPER ANALYSIS SYSTEM")

    print(f"\n Query: {query}")
    print("\nStarting analysis pipeline...")
    
    return {
        "messages": [
            {"role": "system", "content": f"Started analysis for query: {query}"}
        ]
    }


def should_continue(state: Dict[str, Any]) -> str:

    if not state.get("search_completed"):
        return "search"
    
    if not state.get("metadata_enriched"):
        return "metadata"
    
    if not state.get("analysis_completed"):
        return "analysis"
    
    if not state.get("summarization_completed"):
        return "summarizer"
    
    if not state.get("prioritization_completed"):
        return "prioritizer"
    
    return "end"


def final_output_node(state: Dict[str, Any]) -> Dict[str, Any]:

    reading_order = state.get("reading_order", [])
    
    print("✅ ANALYSIS COMPLETE")
    
    print(f"\n Results Summary:")
    print(f"   - Papers found: {len(state.get('papers', []))}")
    print(f"   - Papers ranked: {len(state.get('ranked_papers', []))}")
    print(f"   - Reading list: {len(reading_order)} papers")
    
    if reading_order:
        print(f"\n Top Recommendation:")
        top = reading_order[0]
        print(f"   \"{top.get('title', 'Unknown')}\"")
        print(f"   Score: {top.get('total_score', 0):.2f}")
    
    errors = state.get("errors", [])
    if errors:
        print(f"\n Warnings/Errors: {len(errors)}")
        for error in errors[:3]:
            print(f"   - {error}")
    
    return {
        "messages": state.get("messages", []) + [
            {"role": "system", "content": "Analysis complete!"}
        ]
    }
