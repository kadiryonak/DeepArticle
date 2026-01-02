"""
Prioritizer Agent - Optimizes reading order based on multiple factors.
"""

from typing import Dict, Any, List


def optimize_reading_order(papers: List[Dict[str, Any]], max_papers: int = 20) -> List[Dict[str, Any]]:

    if not papers:
        return []
    
    # Start with top-scored papers but ensure source diversity
    selected = []
    source_counts = {}
    max_per_source = max(3, max_papers // 4)  
    
    for paper in papers:
        if len(selected) >= max_papers:
            break
        
        source = paper.get("source", "unknown")
        
        # Check source limit
        if source_counts.get(source, 0) >= max_per_source:
            continue
        
        # Add paper
        source_counts[source] = source_counts.get(source, 0) + 1
        paper["reading_priority"] = len(selected) + 1
        selected.append(paper)
    
    # If we didn't reach max_papers, fill with remaining high-score papers
    if len(selected) < max_papers:
        for paper in papers:
            if paper not in selected and len(selected) < max_papers:
                paper["reading_priority"] = len(selected) + 1
                selected.append(paper)
    
    return selected


def prioritizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ranked_papers = state.get("ranked_papers", [])
    
    if not ranked_papers:
        return {
            "reading_order": [],
            "prioritization_completed": True,
            "errors": state.get("errors", []) + ["No papers to prioritize"]
        }
    
    print("\n" + "=" * 50)
    print("📋 PRIORITIZER AGENT")
    print("=" * 50)
    print("Optimizing reading order...\n")
    
    # Optimize reading order
    reading_order = optimize_reading_order(ranked_papers, max_papers=20)
    
    # Print reading order
    print("Recommended Reading Order:")
    print("-" * 40)
    
    for i, paper in enumerate(reading_order[:10], 1):
        title = paper.get("title", "No Title")
        if len(title) > 45:
            title = title[:42] + "..."
        source = paper.get("source", "?")
        score = paper.get("total_score", 0)
        
        print(f"  {i}. [{source[:2].upper()}] {title}")
        print(f"      Score: {score:.2f}")
    
    if len(reading_order) > 10:
        print(f"\n  ... and {len(reading_order) - 10} more papers")
    
    # Calculate source distribution
    sources = {}
    for paper in reading_order:
        source = paper.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\nSource Distribution:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  - {source}: {count} papers")
    
    print(f"\n✓ Reading order optimized - {len(reading_order)} papers")
    
    return {
        "reading_order": reading_order,
        "prioritization_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Reading order optimized with {len(reading_order)} papers across {len(sources)} sources."}
        ]
    }
