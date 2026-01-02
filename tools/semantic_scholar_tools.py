"""
Semantic Scholar search tools for academic paper retrieval.
Enhanced with citation sorting and detailed metadata.
"""

from typing import List, Dict, Any
import requests
from langchain_core.tools import tool


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"


@tool
def search_semantic_scholar(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar for academic papers matching the query.
    Returns papers sorted by citation count with detailed metadata.
    """
    papers = []
    
    try:
        # Request more fields for better analysis
        fields = "paperId,title,authors,abstract,year,venue,citationCount,influentialCitationCount,externalIds,url,openAccessPdf,publicationVenue,fieldsOfStudy"
        params = {
            "query": query,
            "limit": max_results,
            "fields": fields
        }
        
        response = requests.get(SEMANTIC_SCHOLAR_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        for result in data.get("data", []):
            authors = result.get("authors", [])
            author_names = [a.get("name", "") for a in authors if a]
            
            external_ids = result.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            
            # Get publication venue info
            pub_venue = result.get("publicationVenue", {}) or {}
            venue_name = pub_venue.get("name", "") or result.get("venue", "")
            venue_type = pub_venue.get("type", "")  # journal, conference, etc.
            
            # Get open access PDF URL
            open_access = result.get("openAccessPdf", {}) or {}
            pdf_url = open_access.get("url", "")
            
            # Fields of study for topic analysis
            fields_of_study = result.get("fieldsOfStudy", []) or []
            
            paper = {
                "paper_id": result.get("paperId", ""),
                "title": result.get("title", ""),
                "authors": author_names,
                "abstract": result.get("abstract", "") or "",
                "published_date": str(result.get("year", "")) if result.get("year") else None,
                "source": "semantic_scholar",
                "url": f"https://www.semanticscholar.org/paper/{result.get('paperId', '')}",
                "pdf_url": pdf_url,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "journal_name": venue_name,
                "venue": venue_name,
                "venue_type": venue_type,
                "citation_count": result.get("citationCount", 0) or 0,
                "influential_citations": result.get("influentialCitationCount", 0) or 0,
                "impact_factor": None,
                "q_quartile": None,
                "relevance_score": 0.0,
                "total_score": 0.0,
                "summary": None,
                "reading_priority": 0,
                "fields_of_study": fields_of_study
            }
            papers.append(paper)
        
        # Sort by citation count (highest first)
        papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
            
    except requests.exceptions.RequestException as e:
        return [{"error": f"Semantic Scholar search failed: {str(e)}"}]
    except Exception as e:
        return [{"error": f"Semantic Scholar search failed: {str(e)}"}]
    
    return papers


@tool
def get_paper_citations(paper_id: str) -> Dict[str, Any]:
    """
    Get citation details for a specific paper from Semantic Scholar.
    """
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {"fields": "citationCount,influentialCitationCount,citations.title"}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return {
            "paper_id": paper_id,
            "citation_count": data.get("citationCount", 0),
            "influential_citation_count": data.get("influentialCitationCount", 0),
            "recent_citations": [c.get("title", "") for c in data.get("citations", [])[:5]]
        }
        
    except Exception as e:
        return {"error": f"Failed to get citations: {str(e)}"}


def get_semantic_scholar_tools():
    return [search_semantic_scholar, get_paper_citations]
