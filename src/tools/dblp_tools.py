"""
DBLP search — the computer-science bibliography.

DBLP indexes CS conference/journal publications with clean venue metadata.
No API key required. https://dblp.org/faq/How+to+use+the+dblp+search+API.html
"""

from typing import List, Dict, Any

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

DBLP_API = "https://dblp.org/search/publ/api"
_HEADERS = {"User-Agent": "DeepArticle/1.0"}


def _dblp_authors(info: Dict[str, Any]) -> List[str]:
    authors = (info.get("authors") or {}).get("author", [])
    if isinstance(authors, dict):
        authors = [authors]
    names = []
    for a in authors:
        if isinstance(a, dict):
            name = a.get("text", "")
        else:
            name = str(a)
        if name:
            names.append(name)
    return names


def _parse_dblp(hit: Dict[str, Any]) -> Dict[str, Any]:
    info = hit.get("info", {}) or {}
    title = info.get("title", "")
    if isinstance(title, dict):
        title = title.get("text", "")
    doi = (info.get("doi") or "").lower()
    year = info.get("year")
    return {
        "paper_id": f"dblp:{hit.get('@id', info.get('key', ''))}",
        "title": (title or "").rstrip("."),
        "authors": _dblp_authors(info),
        "abstract": "",  # DBLP does not provide abstracts
        "published_date": str(year) if year else None,
        "source": "dblp",
        "url": info.get("ee", "") or info.get("url", ""),
        "pdf_url": "",
        "doi": doi,
        "journal_name": info.get("venue", "") or "",
        "venue": info.get("venue", "") or "",
        "venue_type": info.get("type", "") or "",
        "publication_type": info.get("type", "") or "",
        "is_thesis": "thesis" in (info.get("type", "") or "").lower(),
        "language": "",
        "citation_count": 0,
        "influential_citations": 0,
        "impact_factor": None,
        "q_quartile": None,
        "relevance_score": 0.0,
        "total_score": 0.0,
        "summary": None,
        "reading_priority": 0,
        "fields_of_study": [],
    }


@disk_cache(namespace="dblp_search")
def _search_dblp_cached(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    try:
        response = requests.get(
            DBLP_API,
            params={"q": query, "format": "json", "h": min(max_results, 50)},
            headers=_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        hits = response.json().get("result", {}).get("hits", {}).get("hit", []) or []
        if isinstance(hits, dict):
            hits = [hits]
        return [_parse_dblp(h) for h in hits]
    except requests.exceptions.RequestException as e:
        logger.warning("DBLP search failed for query %r: %s", query, e)
        return [{"error": f"DBLP search failed: {str(e)}"}]
    except Exception as e:
        logger.warning("DBLP parse failed for query %r: %s", query, e)
        return [{"error": f"DBLP search failed: {str(e)}"}]


@tool
def search_dblp(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search DBLP (computer-science bibliography) for publications.
    Clean venue metadata; no abstracts/citations. No API key required.
    """
    return _search_dblp_cached(query, max_results)


def get_dblp_tools():
    """Return list of DBLP tools."""
    return [search_dblp]
