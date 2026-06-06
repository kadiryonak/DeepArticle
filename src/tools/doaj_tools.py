"""
DOAJ (Directory of Open Access Journals) search.

DOAJ indexes peer-reviewed open-access journal articles — with abstracts and
full-text links. No API key required. https://doaj.org/api/
"""

from typing import List, Dict, Any
from urllib.parse import quote

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

DOAJ_API = "https://doaj.org/api/search/articles"
_HEADERS = {"User-Agent": "DeepArticle/1.0"}


def _parse_doaj(result: Dict[str, Any]) -> Dict[str, Any]:
    bj = result.get("bibjson", {}) or {}
    authors = [a.get("name", "") for a in bj.get("author", []) or [] if a.get("name")]
    journal = bj.get("journal", {}) or {}

    url, pdf_url, doi = "", "", ""
    for link in bj.get("link", []) or []:
        if link.get("url") and not url:
            url = link["url"]
        if link.get("type") == "fulltext" and link.get("url"):
            pdf_url = link["url"]
    for ident in bj.get("identifier", []) or []:
        if ident.get("type") == "doi":
            doi = (ident.get("id") or "").lower()

    year = bj.get("year")
    return {
        "paper_id": f"doaj:{result.get('id', '')}",
        "title": bj.get("title", "") or "",
        "authors": authors,
        "abstract": bj.get("abstract", "") or "",
        "published_date": str(year) if year else None,
        "source": "doaj",
        "url": url or pdf_url,
        "pdf_url": pdf_url,
        "doi": doi,
        "journal_name": journal.get("title", "") or "",
        "venue": journal.get("title", "") or "",
        "venue_type": "journal",
        "publication_type": "article",
        "is_thesis": False,
        "language": (journal.get("language") or [""])[0] if journal.get("language") else "",
        "citation_count": 0,
        "influential_citations": 0,
        "impact_factor": None,
        "q_quartile": None,
        "relevance_score": 0.0,
        "total_score": 0.0,
        "summary": None,
        "reading_priority": 0,
        "fields_of_study": bj.get("keywords", []) or [],
    }


@disk_cache(namespace="doaj_search")
def _search_doaj_cached(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    try:
        url = f"{DOAJ_API}/{quote(query)}"
        response = requests.get(
            url, params={"pageSize": min(max_results, 50)}, headers=_HEADERS, timeout=30
        )
        response.raise_for_status()
        results = response.json().get("results", []) or []
        return [_parse_doaj(r) for r in results]
    except requests.exceptions.RequestException as e:
        logger.warning("DOAJ search failed for query %r: %s", query, e)
        return [{"error": f"DOAJ search failed: {str(e)}"}]
    except Exception as e:
        logger.warning("DOAJ parse failed for query %r: %s", query, e)
        return [{"error": f"DOAJ search failed: {str(e)}"}]


@tool
def search_doaj(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search DOAJ for open-access journal articles (with abstracts and full text).
    No API key required.
    """
    return _search_doaj_cached(query, max_results)


def get_doaj_tools():
    """Return list of DOAJ tools."""
    return [search_doaj]
