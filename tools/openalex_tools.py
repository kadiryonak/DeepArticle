"""
OpenAlex search tools for academic paper retrieval.

OpenAlex (https://openalex.org) is a free, open catalog of ~250M scholarly
works with no API key required. It complements ArXiv and Semantic Scholar with
broad, cross-disciplinary coverage and citation counts.
"""

from typing import List, Dict, Any

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

OPENALEX_API = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild plain text."""
    if not inverted_index:
        return ""
    positions: List[tuple] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


@tool
def search_openalex(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for academic papers matching the query.
    Returns papers with metadata sorted by citation count. No API key required.
    """
    return _search_openalex_cached(query, max_results)


@disk_cache(namespace="openalex_search")
def _search_openalex_cached(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Cached core of the OpenAlex paper search (see ``search_openalex``)."""
    papers: List[Dict[str, Any]] = []

    try:
        params = {
            "search": query,
            "per-page": min(max_results, 50),
            "sort": "cited_by_count:desc",
            # A contact mailto gets us into OpenAlex's faster "polite pool".
            "mailto": "deeparticle@example.com",
        }
        response = requests.get(OPENALEX_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for work in data.get("results", []):
            authorships = work.get("authorships", []) or []
            author_names = [
                a.get("author", {}).get("display_name", "")
                for a in authorships
                if a.get("author")
            ]

            primary_location = work.get("primary_location") or {}
            source = primary_location.get("source") or {}
            venue_name = source.get("display_name", "") or ""
            venue_type = source.get("type", "") or ""

            ids = work.get("ids", {}) or {}
            doi = work.get("doi") or ids.get("doi")
            if doi:
                doi = doi.replace("https://doi.org/", "")

            abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or {})

            pdf_url = ""
            best_oa = work.get("best_oa_location") or {}
            if best_oa:
                pdf_url = best_oa.get("pdf_url", "") or ""

            paper = {
                "paper_id": work.get("id", ""),
                "title": work.get("title", "") or work.get("display_name", "") or "",
                "authors": author_names,
                "abstract": abstract,
                "published_date": work.get("publication_date") or (
                    str(work.get("publication_year")) if work.get("publication_year") else None
                ),
                "source": "openalex",
                "url": ids.get("doi") or work.get("id", ""),
                "pdf_url": pdf_url,
                "doi": doi,
                "journal_name": venue_name,
                "venue": venue_name,
                "venue_type": venue_type,
                "citation_count": work.get("cited_by_count", 0) or 0,
                "influential_citations": 0,
                "impact_factor": None,
                "q_quartile": None,
                "relevance_score": 0.0,
                "total_score": 0.0,
                "summary": None,
                "reading_priority": 0,
                "fields_of_study": [
                    c.get("display_name", "")
                    for c in (work.get("concepts", []) or [])[:5]
                ],
            }
            papers.append(paper)

        papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)

    except requests.exceptions.RequestException as e:
        logger.warning("OpenAlex search failed for query %r: %s", query, e)
        return [{"error": f"OpenAlex search failed: {str(e)}"}]
    except Exception as e:
        logger.warning("OpenAlex parse failed for query %r: %s", query, e)
        return [{"error": f"OpenAlex search failed: {str(e)}"}]

    return papers


def get_openalex_tools():
    """Return list of OpenAlex tools."""
    return [search_openalex]
