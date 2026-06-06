"""
YÖK Ulusal Tez Merkezi (https://tez.yok.gov.tr) — Turkish National Thesis Center.

This is the authoritative database of Turkish PhD and Master's theses. It has
**no public API**: the search is a session-based JSP form (``tarama.jsp`` →
``SearchTez``) that the center actively restricts against automated access, so
this source is **best-effort** — it gracefully returns an empty list when the
site blocks the request, and never breaks the pipeline.

NOTE: For reliable Turkish thesis coverage, prefer the ``openalex_thesis`` and
``core`` sources, which index a large share of the same YÖK theses (including
Turkish-language ones) through stable APIs.
"""

import re
from typing import List, Dict, Any

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

BASE = "https://tez.yok.gov.tr/UlusalTezMerkezi"
SEARCH_FORM = f"{BASE}/tarama.jsp"
SEARCH_ACTION = f"{BASE}/SearchTez"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": SEARCH_FORM,
}

# Map YÖK thesis-type labels to PhD / Master's.
_THESIS_TYPES = {
    "Doktora": "PhD",
    "Yüksek Lisans": "Master",
    "Tıpta Uzmanlık": "Specialty",
    "Sanatta Yeterlik": "Proficiency in Art",
}


@disk_cache(namespace="yoktez_search")
def _query_yoktez(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Best-effort YÖK Tez search. Returns [] if the site blocks the request."""
    papers: List[Dict[str, Any]] = []

    try:
        session = requests.Session()
        session.headers.update(_HEADERS)
        # Establish a session and obtain the search form (sets JSESSIONID).
        session.get(SEARCH_FORM, timeout=30)

        payload = {
            "keyword": query,
            "ops_field": "2",   # search within thesis title/abstract
            "nevi": "0",
            "tip": "0",
            "Tur": "0",
            "Durum": "0",
            "Dil": "0",
            "izin": "0",
            "yil1": "0",
            "yil2": "0",
        }
        resp = session.post(SEARCH_ACTION, data=payload, timeout=40)
        # The site serves Turkish content as ISO-8859-9.
        resp.encoding = "ISO-8859-9"
        html = resp.text

        if "Hata" in html[:600] or "tezDetay" not in html:
            logger.info("YÖK Tez returned no parseable results (likely access-restricted).")
            return papers

        papers = _parse_results(html, max_results)

    except requests.exceptions.RequestException as e:
        logger.warning("YÖK Tez search failed for query %r: %s", query, e)
        return papers
    except Exception as e:
        logger.warning("YÖK Tez parse failed for query %r: %s", query, e)
        return papers

    return papers


def _parse_results(html: str, max_results: int) -> List[Dict[str, Any]]:
    """Parse the YÖK result table into our paper schema (best-effort)."""
    papers: List[Dict[str, Any]] = []

    # Each result row links to a detail page: tezDetay.jsp?id=...&no=...
    rows = re.split(r'tezDetay\.jsp\?', html)[1:]
    for chunk in rows[:max_results]:
        id_match = re.search(r'id=([^"&]+)', chunk)
        thesis_id = id_match.group(1) if id_match else ""
        # Strip tags to read the row's visible cells.
        text_cells = [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r">([^<]{2,})<", chunk[:1500])
        ]
        text_cells = [c for c in text_cells if c and not c.startswith("&")]
        if not text_cells:
            continue

        title = text_cells[0] if text_cells else ""
        thesis_type = next(
            (_THESIS_TYPES[k] for c in text_cells for k in _THESIS_TYPES if k in c),
            "Thesis",
        )
        year = next((c for c in text_cells if re.fullmatch(r"(19|20)\d{2}", c)), None)

        papers.append({
            "paper_id": f"yoktez:{thesis_id}",
            "title": title,
            "authors": [text_cells[1]] if len(text_cells) > 1 else [],
            "abstract": "",
            "published_date": year,
            "source": "yoktez",
            "url": f"{BASE}/tezDetay.jsp?id={thesis_id}" if thesis_id else SEARCH_FORM,
            "pdf_url": "",
            "doi": "",
            "journal_name": "YÖK Ulusal Tez Merkezi",
            "venue": "YÖK Ulusal Tez Merkezi",
            "venue_type": "thesis",
            "publication_type": "thesis",
            "is_thesis": True,
            "thesis_type": thesis_type,
            "language": "tr",
            "citation_count": 0,
            "influential_citations": 0,
            "impact_factor": None,
            "q_quartile": None,
            "relevance_score": 0.0,
            "total_score": 0.0,
            "summary": None,
            "reading_priority": 0,
            "fields_of_study": [],
        })

    return papers


@tool
def search_yoktez(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search YÖK Ulusal Tez Merkezi for Turkish PhD/Master's theses (best-effort).
    Returns an empty list if the site restricts the automated request.
    """
    return _query_yoktez(query, max_results)


def get_yoktez_tools():
    """Return list of YÖK Tez tools."""
    return [search_yoktez]
