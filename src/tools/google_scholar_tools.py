"""
Google Scholar search tools for academic paper retrieval.
Uses the scholarly library for free access to Google Scholar.
"""

from typing import List, Dict, Any
from langchain_core.tools import tool

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False


@tool
def search_google_scholar(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Google Scholar for academic papers matching the query.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata
    """
    if not SCHOLARLY_AVAILABLE:
        return [{"error": "scholarly package not installed. Run: pip install scholarly"}]
    
    papers = []
    
    try:
        search_query = scholarly.search_pubs(query)
        
        count = 0
        for result in search_query:
            if count >= max_results:
                break
            
            # Extract basic info
            bib = result.get("bib", {})
            
            # Get authors
            authors = bib.get("author", [])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(" and ")]
            
            # Get publication year
            year = bib.get("pub_year", "")
            
            paper = {
                "paper_id": result.get("author_pub_id", "") or f"gs_{count}",
                "title": bib.get("title", ""),
                "authors": authors,
                "abstract": bib.get("abstract", ""),
                "published_date": str(year) if year else None,
                "source": "google_scholar",
                "url": result.get("pub_url", "") or result.get("eprint_url", ""),
                "doi": None,
                "journal_name": bib.get("venue", "") or bib.get("journal", ""),
                "venue": bib.get("venue", ""),
                "citation_count": result.get("num_citations", 0) or 0,
                "impact_factor": None,
                "q_quartile": None,
                "relevance_score": 0.0,
                "total_score": 0.0,
                "summary": None,
                "reading_priority": 0
            }
            papers.append(paper)
            count += 1
            
    except Exception as e:
        return [{"error": f"Google Scholar search failed: {str(e)}"}]
    
    return papers


def get_google_scholar_tools():
    """Return list of Google Scholar tools."""
    return [search_google_scholar]
