"""
Search Agent - Searches multiple academic sources for Computer Science papers.
Enhanced with multi-query search and known systems discovery.
"""

from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import tools
import sys
sys.path.insert(0, '..')

from config import SOURCES, SEARCH_MAX_WORKERS, SEMANTIC_SCHOLAR_QUERY_LIMIT
from tools.arxiv_tools import search_arxiv
from tools.semantic_scholar_tools import search_semantic_scholar
from tools.openalex_tools import search_openalex, search_openalex_theses
from tools.core_tools import search_core
from tools.crossref_tools import search_crossref
from tools.doaj_tools import search_doaj
from tools.dblp_tools import search_dblp
from tools.openaire_tools import search_openaire
from utils.logging_config import get_logger

logger = get_logger(__name__)


def search_arxiv_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search ArXiv with a single query."""
    try:
        results = search_arxiv.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except Exception as e:
        logger.warning("ArXiv search failed for query %r: %s", query, e)
    return []


def search_semantic_scholar_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search Semantic Scholar with a single query."""
    try:
        results = search_semantic_scholar.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            # Filter for CS
            cs_fields = {"Computer Science", "Engineering", "Mathematics"}
            filtered = []
            for paper in results:
                fields = set(paper.get("fields_of_study", []) or [])
                if not fields or fields.intersection(cs_fields):
                    filtered.append(paper)
            return filtered
    except Exception as e:
        logger.warning("Semantic Scholar search failed for query %r: %s", query, e)
    return []


def search_openalex_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search OpenAlex with a single query."""
    try:
        results = search_openalex.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except Exception as e:
        logger.warning("OpenAlex search failed for query %r: %s", query, e)
    return []


def search_pubmed_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search PubMed with a single query (optional biomedical source)."""
    try:
        from tools.pubmed_tools import search_pubmed
        results = search_pubmed.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except Exception as e:
        logger.warning("PubMed search failed for query %r: %s", query, e)
    return []


def search_openalex_theses_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search OpenAlex dissertations (PhD/Master's theses) with a single query."""
    try:
        results = search_openalex_theses.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except Exception as e:
        logger.warning("OpenAlex thesis search failed for query %r: %s", query, e)
    return []


def search_core_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search CORE (open-access full text + theses) with a single query."""
    try:
        results = search_core.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except Exception as e:
        logger.warning("CORE search failed for query %r: %s", query, e)
    return []


def search_yoktez_with_query(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search YÖK Ulusal Tez Merkezi (Turkish theses, best-effort) with a single query."""
    try:
        from tools.yoktez_tools import search_yoktez
        results = search_yoktez.invoke({"query": query, "max_results": max_results})
        if results and not any("error" in r for r in results):
            return results
    except Exception as e:
        logger.warning("YÖK Tez search failed for query %r: %s", query, e)
    return []


def _simple_source(search_tool, name: str):
    """Build a (query -> papers) wrapper for a source that needs no extra filtering."""
    def _run(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
        try:
            results = search_tool.invoke({"query": query, "max_results": max_results})
            if results and not any("error" in r for r in results):
                return results
        except Exception as e:
            logger.warning("%s search failed for query %r: %s", name, query, e)
        return []
    return _run


search_crossref_with_query = _simple_source(search_crossref, "CrossRef")
search_doaj_with_query = _simple_source(search_doaj, "DOAJ")
search_dblp_with_query = _simple_source(search_dblp, "DBLP")
search_openaire_with_query = _simple_source(search_openaire, "OpenAIRE")


# Maps a source name to (search function, applies-to-query-index predicate).
# Add a source to the SOURCES env var (comma-separated) to enable it.
_SOURCE_FUNCS = {
    "arxiv": (search_arxiv_with_query, lambda i: True),
    "semantic_scholar": (
        search_semantic_scholar_with_query,
        lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT,
    ),
    "openalex": (search_openalex_with_query, lambda i: True),
    "openalex_thesis": (search_openalex_theses_with_query, lambda i: True),
    "core": (search_core_with_query, lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT),
    "crossref": (search_crossref_with_query, lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT),
    "doaj": (search_doaj_with_query, lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT),
    "dblp": (search_dblp_with_query, lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT),
    "openaire": (search_openaire_with_query, lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT),
    "pubmed": (search_pubmed_with_query, lambda i: i < SEMANTIC_SCHOLAR_QUERY_LIMIT),
    # YÖK Tez is fragile/rate-limited — only the first couple of queries.
    "yoktez": (search_yoktez_with_query, lambda i: i < 2),
}


def search_cs_sources(queries: List[str], max_results_per_query: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search the configured academic sources with multiple queries, concurrently.

    Queries × sources are dispatched to a thread pool so the (network-bound)
    searches run in parallel instead of sequentially.

    Args:
        queries: List of search queries
        max_results_per_query: Maximum results per query

    Returns:
        Dictionary mapping source names to lists of papers
    """
    results_by_source: Dict[str, List[Dict[str, Any]]] = {src: [] for src in SOURCES}

    # Build the list of (source, query) tasks honoring per-source query limits.
    tasks = []
    for i, query in enumerate(queries):
        for source in SOURCES:
            entry = _SOURCE_FUNCS.get(source)
            if not entry:
                logger.warning("Unknown source %r in SOURCES; skipping", source)
                continue
            func, applies = entry
            if applies(i):
                tasks.append((source, func, query))

    print(
        f"\n🔍 Executing {len(queries)} queries across {len(SOURCES)} sources "
        f"({len(tasks)} parallel tasks)...\n"
    )

    with ThreadPoolExecutor(max_workers=SEARCH_MAX_WORKERS) as executor:
        future_to_meta = {
            executor.submit(func, query, max_results_per_query): (source, query)
            for source, func, query in tasks
        }
        for future in as_completed(future_to_meta):
            source, query = future_to_meta[future]
            try:
                papers = future.result()
            except Exception as e:
                logger.warning("Search task failed (%s, %r): %s", source, query, e)
                papers = []
            if papers:
                results_by_source.setdefault(source, []).extend(papers)
                short_query = query[:45] + "..." if len(query) > 45 else query
                print(f"   [{source}] +{len(papers)} papers  «{short_query}»")

    return results_by_source


def _norm_doi(paper: Dict[str, Any]) -> str:
    """Normalized DOI (lowercased, URL prefix stripped), or '' if none."""
    doi = (paper.get("doi") or "").strip().lower()
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "")


def _norm_title(paper: Dict[str, Any]) -> str:
    """Normalized title: lowercased, only alphanumerics (for fuzzy matching)."""
    return "".join(c for c in (paper.get("title") or "").lower() if c.isalnum())


# Titles shorter than this (normalized length) aren't trusted for title-based
# dedup — too generic (e.g. "Survey", "Overview") and could collide unrelated work.
_MIN_TITLE_LEN = 8


def _add_provenance(paper: Dict[str, Any], source: str) -> None:
    """Record which database a paper came from and its link there (in place)."""
    link = {
        "source": source,
        "url": paper.get("url", "") or "",
        "pdf_url": paper.get("pdf_url", "") or "",
    }
    found_in = paper.setdefault("found_in", [])
    if source not in found_in:
        found_in.append(source)
    links = paper.setdefault("source_links", [])
    if not any(existing["source"] == source for existing in links):
        links.append(link)


def _merge_duplicate(into: Dict[str, Any], other: Dict[str, Any]) -> None:
    """
    Fold ``other`` (a duplicate of ``into``) into ``into`` (in place).

    Keeps the richest data: the higher citation count, a non-empty abstract,
    a usable PDF link, the union of fields/provenance — so the same work found
    in several databases collapses into ONE entry that lists all of them.
    """
    if (other.get("citation_count", 0) or 0) > (into.get("citation_count", 0) or 0):
        into["citation_count"] = other.get("citation_count", 0) or 0
    if not into.get("abstract") and other.get("abstract"):
        into["abstract"] = other["abstract"]
    if not into.get("pdf_url") and other.get("pdf_url"):
        into["pdf_url"] = other["pdf_url"]
    if not into.get("doi") and other.get("doi"):
        into["doi"] = other["doi"]
    if not into.get("is_thesis") and other.get("is_thesis"):
        into["is_thesis"] = True
    # Union fields of study.
    existing_fields = into.get("fields_of_study") or []
    for f in other.get("fields_of_study") or []:
        if f and f not in existing_fields:
            existing_fields.append(f)
    into["fields_of_study"] = existing_fields
    # Carry over each source the duplicate was found in.
    for link in other.get("source_links", []):
        _add_provenance(into, link["source"])
        for existing in into["source_links"]:
            if existing["source"] == link["source"]:
                existing["url"] = existing["url"] or link["url"]
                existing["pdf_url"] = existing["pdf_url"] or link["pdf_url"]


def merge_papers(search_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge papers from all sources, removing duplicates.

    A work is matched by DOI **or** normalized title, so the same paper collapses
    into one entry even when sources disagree on identifiers (e.g. one has a DOI
    and another doesn't, or a preprint and the published version carry different
    DOIs). The surviving entry's ``found_in`` lists every database it appeared in
    and ``source_links`` holds each database's link — nothing is shown twice, but
    every link is preserved.
    """
    entries: List[Dict[str, Any]] = []
    by_doi: Dict[str, Dict[str, Any]] = {}
    by_title: Dict[str, Dict[str, Any]] = {}

    for source, papers in search_results.items():
        for paper in papers:
            doi = _norm_doi(paper)
            title = _norm_title(paper)
            title_usable = len(title) >= _MIN_TITLE_LEN

            _add_provenance(paper, source)

            if not doi and not title_usable:
                # No reliable identifier — keep the paper, but never merge it
                # (a generic/short title could collide with unrelated work).
                entries.append(paper)
                continue

            # Find an existing entry this paper duplicates (DOI first, then title).
            entry = None
            if doi and doi in by_doi:
                entry = by_doi[doi]
            elif title_usable and title in by_title:
                entry = by_title[title]

            if entry is None:
                entries.append(paper)
                entry = paper
            else:
                _merge_duplicate(entry, paper)

            # Register both keys so later records match by either identifier.
            if doi:
                by_doi[doi] = entry
            if title_usable:
                by_title[title] = entry

    return sorted(entries, key=lambda p: p.get("citation_count", 0) or 0, reverse=True)


def search_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for the Search Agent.
    Uses multiple queries for comprehensive search.
    """
    query = state.get("query", "")
    search_queries = state.get("search_queries", [query])  # Use expanded queries if available
    
    if not query and not search_queries:
        return {
            "papers": [],
            "search_completed": True,
            "errors": state.get("errors", []) + ["No query provided"]
        }
    
    print("\n" + "=" * 60)
    print("📚 SEARCH AGENT - Multi-Query CS Search")
    print("=" * 60)
    print(f"Original Topic: {query}")
    print(f"Search Queries: {len(search_queries)}")
    
    # Search with all queries
    search_results = search_cs_sources(search_queries, max_results_per_query=15)
    
    # Merge and deduplicate
    papers = merge_papers(search_results)
    
    # Statistics
    high_citations = sum(1 for p in papers if (p.get("citation_count", 0) or 0) >= 50)
    total_citations = sum(p.get("citation_count", 0) or 0 for p in papers)
    
    print(f"\n{'=' * 60}")
    print(f"📊 Search Results Summary:")
    print(f"   - Total unique papers: {len(papers)}")
    print(f"   - Papers with 50+ citations: {high_citations}")
    print(f"   - Total citations: {total_citations:,}")
    
    if papers:
        print(f"\n📑 Top 5 Papers by Citations:")
        for i, p in enumerate(papers[:5], 1):
            title = p.get("title", "")[:55]
            cites = p.get("citation_count", 0)
            print(f"   {i}. [{cites} cites] {title}...")
    
    return {
        "papers": papers,
        "search_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Multi-query search found {len(papers)} unique papers with {total_citations:,} citations."}
        ]
    }
