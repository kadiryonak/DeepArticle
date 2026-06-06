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


def _dedup_key(paper: Dict[str, Any]) -> str:
    """
    Build a deduplication key for a paper.

    Prefers the DOI (the strongest cross-source identifier); falls back to a
    normalized title so the same work from different sources — or the
    preprint/published versions of it — collapse into one entry.
    """
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        # Normalize common DOI URL prefixes.
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        return f"doi:{doi}"

    title = (paper.get("title") or "").lower().strip()
    return "title:" + "".join(c for c in title if c.isalnum())[:80]


def merge_papers(search_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge papers from all sources, removing duplicates.

    Deduplicates by DOI (when present) then by normalized title, keeping the
    record with the higher citation count.
    """
    seen = {}  # dedup_key -> paper

    for _source, papers in search_results.items():
        for paper in papers:
            key = _dedup_key(paper)
            if key in ("doi:", "title:"):
                continue

            citations = paper.get("citation_count", 0) or 0

            if key not in seen:
                seen[key] = paper
            else:
                existing_citations = seen[key].get("citation_count", 0) or 0
                if citations > existing_citations:
                    seen[key] = paper

    # Sort by citation count
    merged = sorted(seen.values(), key=lambda p: p.get("citation_count", 0) or 0, reverse=True)

    return merged


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
