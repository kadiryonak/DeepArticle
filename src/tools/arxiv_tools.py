"""
ArXiv search tools for academic paper retrieval.
Enhanced with citation data from Semantic Scholar.
"""

from typing import List, Dict, Any
import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.logging_config import get_logger
from utils.cache import disk_cache

logger = get_logger(__name__)

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False


@disk_cache(namespace="s2_citations")
def get_citations_from_semantic_scholar(arxiv_id: str) -> int:
    """Get citation count from Semantic Scholar using ArXiv ID."""
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        params = {"fields": "citationCount"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("citationCount", 0) or 0
    except Exception as e:
        logger.debug("Citation lookup failed for arXiv:%s: %s", arxiv_id, e)
    return 0


@tool
def search_arxiv(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search ArXiv for academic papers matching the query.
    Fetches citation data from Semantic Scholar for ranking.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, sorted by citations
    """
    if not ARXIV_AVAILABLE:
        return [{"error": "arxiv package not installed. Run: pip install arxiv"}]
    
    papers = []
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            
            # Get citation count from Semantic Scholar
            citation_count = get_citations_from_semantic_scholar(arxiv_id)
            
            paper = {
                "paper_id": arxiv_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "published_date": result.published.strftime("%Y-%m-%d") if result.published else None,
                "source": "arxiv",
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "doi": result.doi,
                "journal_name": result.journal_ref,
                "venue": result.journal_ref,
                "citation_count": citation_count,
                "impact_factor": None,
                "q_quartile": None,
                "relevance_score": 0.0,
                "total_score": 0.0,
                "summary": None,
                "reading_priority": 0,
                "categories": result.categories if hasattr(result, 'categories') else []
            }
            papers.append(paper)
        
        # Sort by citation count (highest first)
        papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
            
    except Exception as e:
        return [{"error": f"ArXiv search failed: {str(e)}"}]
    
    return papers


def get_arxiv_tools():
    """Return list of ArXiv tools."""
    return [search_arxiv]
