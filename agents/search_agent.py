"""
Search Agent - Searches multiple academic sources for Computer Science papers.
Enhanced with multi-query search and known systems discovery.
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage

# Import tools
import sys
sys.path.insert(0, '..')

from tools.arxiv_tools import search_arxiv
from tools.semantic_scholar_tools import search_semantic_scholar


def search_arxiv_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search ArXiv with a single query."""
    try:
        results = search_arxiv.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except:
        pass
    return []


def search_semantic_scholar_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search Semantic Scholar with a single query."""
    try:
        results = search_semantic_scholar.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            # Filter for CS
            cs_fields = {"Computer Science", "Engineering", "Mathematics"}
            filtered = []
            for paper in results:
                fields = set(paper.get("fields_of_study", []) or [])
                if not fields or fields.intersection(cs_fields):
                    filtered.append(paper)
            return filtered
    except:
        pass
    return []


def search_cs_sources(queries: List[str], max_results_per_query: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search Computer Science sources with multiple queries.
    
    Args:
        queries: List of search queries
        max_results_per_query: Maximum results per query
        
    Returns:
        Dictionary mapping source names to lists of papers
    """
    all_arxiv = []
    all_ss = []
    
    print(f"\n🔍 Executing {len(queries)} search queries...\n")
    
    for i, query in enumerate(queries):
        short_query = query[:50] + "..." if len(query) > 50 else query
        print(f"   [{i+1}/{len(queries)}] \"{short_query}\"")
        
        # ArXiv search
        arxiv_results = search_arxiv_with_query(query, max_results_per_query)
        if arxiv_results:
            all_arxiv.extend(arxiv_results)
            print(f"         ArXiv: +{len(arxiv_results)} papers")
        
        # Semantic Scholar (limit to avoid rate limiting)
        if i < 5:  # Only first 5 queries for Semantic Scholar
            ss_results = search_semantic_scholar_with_query(query, max_results_per_query)
            if ss_results:
                all_ss.extend(ss_results)
                print(f"         Semantic Scholar: +{len(ss_results)} papers")
    
    return {
        "arxiv": all_arxiv,
        "semantic_scholar": all_ss
    }


def merge_papers(search_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge papers from all sources, removing duplicates.
    Prioritizes papers with higher citation counts.
    """
    seen_titles = {}  # title_key -> paper
    
    for source, papers in search_results.items():
        for paper in papers:
            title = paper.get("title", "").lower().strip()
            title_key = "".join(c for c in title if c.isalnum())[:60]
            
            if not title_key:
                continue
            
            citations = paper.get("citation_count", 0) or 0
            
            if title_key not in seen_titles:
                seen_titles[title_key] = paper
            else:
                existing_citations = seen_titles[title_key].get("citation_count", 0) or 0
                if citations > existing_citations:
                    seen_titles[title_key] = paper
    
    # Sort by citation count
    merged = sorted(seen_titles.values(), key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
    
    return merged


def search_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for the Search Agent.
    Uses multiple queries for comprehensive search.
    """
    query = state.get("query", "")
    search_queries = state.get("search_queries", [query])  # Use expanded queries if available
    
    if not query and not search_queries:
        return {
            "papers": [],
            "search_completed": True,
            "errors": state.get("errors", []) + ["No query provided"]
        }
    
    print("\n" + "=" * 60)
    print("📚 SEARCH AGENT - Multi-Query CS Search")
    print("=" * 60)
    print(f"Original Topic: {query}")
    print(f"Search Queries: {len(search_queries)}")
    
    # Search with all queries
    search_results = search_cs_sources(search_queries, max_results_per_query=15)
    
    # Merge and deduplicate
    papers = merge_papers(search_results)
    
    # Statistics
    high_citations = sum(1 for p in papers if (p.get("citation_count", 0) or 0) >= 50)
    total_citations = sum(p.get("citation_count", 0) or 0 for p in papers)
    
    print(f"\n{'=' * 60}")
    print(f"📊 Search Results Summary:")
    print(f"   - Total unique papers: {len(papers)}")
    print(f"   - Papers with 50+ citations: {high_citations}")
    print(f"   - Total citations: {total_citations:,}")
    
    if papers:
        print(f"\n📑 Top 5 Papers by Citations:")
        for i, p in enumerate(papers[:5], 1):
            title = p.get("title", "")[:55]
            cites = p.get("citation_count", 0)
            print(f"   {i}. [{cites} cites] {title}...")
    
    return {
        "papers": papers,
        "search_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Multi-query search found {len(papers)} unique papers with {total_citations:,} citations."}
        ]
    }
