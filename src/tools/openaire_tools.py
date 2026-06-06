"""
OpenAIRE search — European open-access research aggregator.

OpenAIRE harvests publications (incl. theses) from repositories across Europe
and beyond. Its JSON is deeply nested and varies per record, so parsing is
defensive: malformed records are skipped rather than failing the search.
No API key required. https://graph.openaire.eu/develop/api.html
"""

import html
from typing import List, Dict, Any

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

OPENAIRE_API = "https://api.openaire.eu/search/publications"
_HEADERS = {"User-Agent": "DeepArticle/1.0 (mailto:deeparticle@example.com)"}


def _txt(v: Any) -> str:
    """Extract text from OpenAIRE's {'$': value} / list / scalar shapes."""
    if isinstance(v, dict):
        return html.unescape(str(v.get("$", "") or ""))
    if isinstance(v, list):
        return _txt(v[0]) if v else ""
    return html.unescape(str(v)) if v is not None else ""


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _parse_openaire(result: Dict[str, Any]) -> Dict[str, Any]:
    md = result.get("metadata", {}).get("oaf:entity", {}).get("oaf:result", {})

    title = _txt(md.get("title"))
    authors = [_txt(c) for c in _as_list(md.get("creator")) if _txt(c)]

    doi = ""
    for pid in _as_list(md.get("pid")):
        if isinstance(pid, dict) and pid.get("@classid") == "doi":
            doi = (_txt(pid)).lower()
            break

    rtype = ""
    rt = md.get("resulttype")
    if isinstance(rt, dict):
        rtype = rt.get("@classname", "") or ""
    is_thesis = "thesis" in rtype.lower() or "doctoral" in rtype.lower()

    date = _txt(md.get("dateofacceptance"))
    url = f"https://doi.org/{doi}" if doi else _txt(md.get("originalId"))

    return {
        "paper_id": f"openaire:{doi or _txt(md.get('originalId'))}",
        "title": title,
        "authors": authors,
        "abstract": _txt(md.get("description")),
        "published_date": date or None,
        "source": "openaire",
        "url": url,
        "pdf_url": "",
        "doi": doi,
        "journal_name": _txt(md.get("journal")),
        "venue": _txt(md.get("journal")),
        "venue_type": rtype,
        "publication_type": rtype,
        "is_thesis": is_thesis,
        "language": (md.get("language") or {}).get("@classid", "") if isinstance(md.get("language"), dict) else "",
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


@disk_cache(namespace="openaire_search")
def _search_openaire_cached(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    try:
        response = requests.get(
            OPENAIRE_API,
            params={"keywords": query, "format": "json", "size": min(max_results, 50)},
            headers=_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("response", {}).get("results", {})
        items = _as_list(results.get("result")) if results else []

        papers = []
        for item in items:
            try:
                paper = _parse_openaire(item)
                if paper["title"]:
                    papers.append(paper)
            except Exception as e:  # skip malformed records, keep the rest
                logger.debug("OpenAIRE record skipped: %s", e)
        return papers
    except requests.exceptions.RequestException as e:
        logger.warning("OpenAIRE search failed for query %r: %s", query, e)
        return [{"error": f"OpenAIRE search failed: {str(e)}"}]
    except Exception as e:
        logger.warning("OpenAIRE parse failed for query %r: %s", query, e)
        return [{"error": f"OpenAIRE search failed: {str(e)}"}]


@tool
def search_openaire(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search OpenAIRE for open-access publications and theses across Europe.
    No API key required.
    """
    return _search_openaire_cached(query, max_results)


def get_openaire_tools():
    """Return list of OpenAIRE tools."""
    return [search_openaire]
