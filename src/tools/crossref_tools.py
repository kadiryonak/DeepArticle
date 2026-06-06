"""
CrossRef API tools: DOI metadata lookup + full-text query search.

CrossRef indexes ~150M scholarly works with DOIs, venue names and citation
counts (``is-referenced-by-count``). No API key required.
"""

import re
from typing import Dict, Any, List
import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

CROSSREF_API = "https://api.crossref.org/works"
_HEADERS = {"User-Agent": "DeepArticle/1.0 (mailto:deeparticle@example.com)"}


def _strip_jats(text: str) -> str:
    """CrossRef abstracts are JATS XML; strip tags to plain text."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _crossref_date(item: Dict[str, Any]) -> str:
    parts = (
        (item.get("published") or item.get("published-print") or item.get("published-online") or {})
        .get("date-parts", [[]])
    )
    if parts and parts[0]:
        p = parts[0]
        y = p[0]
        m = p[1] if len(p) > 1 else 1
        d = p[2] if len(p) > 2 else 1
        return f"{y}-{str(m).zfill(2)}-{str(d).zfill(2)}"
    return None


def _parse_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in item.get("author", []) or []
        if a.get("family") or a.get("given")
    ]
    doi = (item.get("DOI") or "").lower()
    return {
        "paper_id": f"crossref:{doi}" if doi else "",
        "title": (item.get("title") or [""])[0] or "",
        "authors": authors,
        "abstract": _strip_jats(item.get("abstract", "")),
        "published_date": _crossref_date(item),
        "source": "crossref",
        "url": item.get("URL", "") or (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": "",
        "doi": doi,
        "journal_name": (item.get("container-title") or [""])[0] or "",
        "venue": (item.get("container-title") or [""])[0] or "",
        "venue_type": item.get("type", "") or "",
        "publication_type": item.get("type", "") or "",
        "is_thesis": item.get("type") == "dissertation",
        "language": item.get("language", "") or "",
        "citation_count": item.get("is-referenced-by-count", 0) or 0,
        "influential_citations": 0,
        "impact_factor": None,
        "q_quartile": None,
        "relevance_score": 0.0,
        "total_score": 0.0,
        "summary": None,
        "reading_priority": 0,
        "fields_of_study": [],
    }


@disk_cache(namespace="crossref_search")
def _search_crossref_cached(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    try:
        response = requests.get(
            CROSSREF_API,
            params={"query": query, "rows": min(max_results, 50)},
            headers=_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", []) or []
        papers = [_parse_crossref_item(it) for it in items]
        papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
        return papers
    except requests.exceptions.RequestException as e:
        logger.warning("CrossRef search failed for query %r: %s", query, e)
        return [{"error": f"CrossRef search failed: {str(e)}"}]
    except Exception as e:
        logger.warning("CrossRef parse failed for query %r: %s", query, e)
        return [{"error": f"CrossRef search failed: {str(e)}"}]


@tool
def search_crossref(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search CrossRef for scholarly works matching the query.
    Returns papers with DOIs, venues and citation counts. No API key required.
    """
    return _search_crossref_cached(query, max_results)


@tool
def get_paper_metadata_by_doi(doi: str) -> Dict[str, Any]:
    """
    Get paper metadata from CrossRef using DOI.
    
    Args:
        doi: The DOI of the paper
        
    Returns:
        Dictionary with paper metadata
    """
    if not doi:
        return {"error": "DOI is required"}
    
    try:
        url = f"{CROSSREF_API}/{doi}"
        headers = {
            "User-Agent": "AcademicPaperAnalyzer/1.0 (mailto:example@example.com)"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        message = data.get("message", {})
        
        # Extract authors
        authors = []
        for author in message.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)
        
        # Extract publication date
        date_parts = message.get("published-print", {}).get("date-parts", [[]])
        if not date_parts or not date_parts[0]:
            date_parts = message.get("published-online", {}).get("date-parts", [[]])
        
        date_str = None
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            if len(parts) >= 1:
                year = parts[0]
                month = parts[1] if len(parts) >= 2 else 1
                day = parts[2] if len(parts) >= 3 else 1
                date_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
        
        # Get title
        titles = message.get("title", [])
        title = titles[0] if titles else ""
        
        # Get container (journal) title
        container_titles = message.get("container-title", [])
        journal_name = container_titles[0] if container_titles else ""
        
        # Get ISSN for potential journal lookup
        issns = message.get("ISSN", [])
        
        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "published_date": date_str,
            "journal_name": journal_name,
            "issn": issns[0] if issns else None,
            "publisher": message.get("publisher", ""),
            "type": message.get("type", ""),
            "references_count": message.get("references-count", 0),
            "is_referenced_by_count": message.get("is-referenced-by-count", 0),
            "url": message.get("URL", ""),
            "found": True
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"doi": doi, "found": False, "error": "DOI not found"}
        return {"error": f"CrossRef lookup failed: {str(e)}"}
    except Exception as e:
        return {"error": f"CrossRef lookup failed: {str(e)}"}


def get_crossref_tools():
    """Return list of CrossRef tools."""
    return [search_crossref, get_paper_metadata_by_doi]
