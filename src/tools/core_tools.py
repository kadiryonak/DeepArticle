"""
CORE search tools (https://core.ac.uk).

CORE aggregates ~300M open-access research outputs harvested from repositories
worldwide, with **full-text PDFs** and an explicit ``documentType`` (so we can
target ``thesis`` for PhD/Master's dissertations). Coverage is multilingual,
including many Turkish university repositories.

An API key is optional — anonymous access works but is rate-limited. Set
``CORE_API_KEY`` (https://core.ac.uk/services/api) for higher limits.
"""

from typing import List, Dict, Any, Optional

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from config import CORE_API_KEY
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

CORE_API = "https://api.core.ac.uk/v3/search/works"


def _parse_core_work(work: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single CORE work record into our paper schema."""
    authors = [
        a.get("name", "") for a in (work.get("authors", []) or []) if a.get("name")
    ]

    language = work.get("language") or {}
    lang_code = language.get("code", "") if isinstance(language, dict) else (language or "")

    doc_type = (work.get("documentType") or "").lower()
    is_thesis = doc_type == "thesis"

    doi = work.get("doi") or ""
    if doi:
        doi = doi.replace("https://doi.org/", "")

    pdf_url = work.get("downloadUrl", "") or ""

    return {
        "paper_id": str(work.get("id", "")),
        "title": work.get("title", "") or "",
        "authors": authors,
        "abstract": work.get("abstract", "") or "",
        "published_date": work.get("publishedDate") or (
            str(work.get("yearPublished")) if work.get("yearPublished") else None
        ),
        "source": "core",
        "url": pdf_url or (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": pdf_url,
        "doi": doi,
        "journal_name": work.get("publisher", "") or "",
        "venue": work.get("publisher", "") or "",
        "venue_type": doc_type,
        "publication_type": doc_type,
        "is_thesis": is_thesis,
        "language": lang_code,
        "citation_count": work.get("citationCount", 0) or 0,
        "influential_citations": 0,
        "impact_factor": None,
        "q_quartile": None,
        "relevance_score": 0.0,
        "total_score": 0.0,
        "summary": None,
        "reading_priority": 0,
        "fields_of_study": work.get("fieldOfStudy", []) if isinstance(work.get("fieldOfStudy"), list) else [],
    }


@disk_cache(namespace="core_search")
def _query_core(
    query: str,
    max_results: int = 20,
    thesis_only: bool = False,
) -> List[Dict[str, Any]]:
    """Core CORE query. When ``thesis_only`` is set, restricts to theses."""
    papers: List[Dict[str, Any]] = []

    q = query
    if thesis_only:
        q = f"({query}) AND documentType:thesis"

    headers = {"User-Agent": "DeepArticle/1.0"}
    if CORE_API_KEY:
        headers["Authorization"] = f"Bearer {CORE_API_KEY}"

    try:
        response = requests.get(
            CORE_API,
            params={"q": q, "limit": min(max_results, 50)},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        for work in data.get("results", []) or []:
            papers.append(_parse_core_work(work))

        papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)

    except requests.exceptions.RequestException as e:
        logger.warning("CORE search failed for query %r: %s", query, e)
        return [{"error": f"CORE search failed: {str(e)}"}]
    except Exception as e:
        logger.warning("CORE parse failed for query %r: %s", query, e)
        return [{"error": f"CORE search failed: {str(e)}"}]

    return papers


@tool
def search_core(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search CORE for open-access papers (with full-text PDFs) matching the query.
    Multilingual coverage. API key optional (rate-limited without one).
    """
    return _query_core(query, max_results)


@tool
def search_core_theses(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search CORE for open-access theses/dissertations matching the query.
    Restricts results to documentType:thesis. Multilingual (incl. Turkish).
    """
    return _query_core(query, max_results, thesis_only=True)


def get_core_tools():
    """Return list of CORE tools."""
    return [search_core, search_core_theses]
