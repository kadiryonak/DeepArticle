"""
LLM reranker — language-agnostic relevance scoring.

Keyword relevance is biased toward the query's language (a Turkish query only
matches Turkish papers, so equally-relevant English papers are penalized). This
reranker asks an LLM to score each candidate's relevance to the topic on 0-100
**regardless of language**, then recomputes the blended score and drops clearly
off-topic papers. Run at temperature 0 for deterministic, reproducible ranking.

The judge is the configured LLM (set ``LLM_PROVIDER=anthropic`` to use Haiku).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from config import SCORING_WEIGHTS, RERANK_TOP_K, RELEVANCE_MIN
from utils.logging_config import get_logger

logger = get_logger(__name__)

_ABSTRACT_CHARS = 240


class _Rel(BaseModel):
    id: int = Field(description="the paper's number as shown")
    relevance: int = Field(description="0-100 relevance to the topic, language-agnostic")


class _Rels(BaseModel):
    items: List[_Rel] = Field(default_factory=list)


def _candidate_pool(papers: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Pick up to ``k`` candidates language-neutrally: the union of the most-cited
    and the current top-scored papers, so relevant English papers (low keyword
    score but high citations) and relevant Turkish papers both make the cut.
    """
    by_cite = sorted(papers, key=lambda p: p.get("citation_count", 0) or 0, reverse=True)[:k]
    by_score = sorted(papers, key=lambda p: p.get("total_score", 0) or 0, reverse=True)[:k]
    seen, pool = set(), []
    for p in by_cite + by_score:
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            pool.append(p)
        if len(pool) >= k:
            break
    return pool


def _recompute_total(paper: Dict[str, Any]) -> float:
    """Recompute the blended score using the (updated) relevance + stored quality."""
    q = paper.get("quality_metrics", {}) or {}
    score = (
        q.get("citation_score", 0) * SCORING_WEIGHTS["citation_count"]
        + paper.get("relevance_score", 0) * SCORING_WEIGHTS["relevance"]
        + q.get("venue_score", 0) * SCORING_WEIGHTS["venue_quality"]
        + q.get("recency_score", 0) * SCORING_WEIGHTS["recency"]
        + q.get("influential_score", 0) * SCORING_WEIGHTS["influential_citations"]
    )
    return round(score, 2)


def llm_rerank(topic: str, papers: List[Dict[str, Any]], llm,
               top_k: int = None, min_relevance: int = None) -> List[Dict[str, Any]]:
    """
    Rerank ``papers`` by LLM-judged, language-agnostic relevance to ``topic``.

    Returns the full list re-sorted: reranked candidates (off-topic ones below
    ``min_relevance`` removed) first, then the untouched tail. On any failure the
    original list is returned unchanged.
    """
    top_k = RERANK_TOP_K if top_k is None else top_k
    min_relevance = RELEVANCE_MIN if min_relevance is None else min_relevance
    if not papers or llm is None:
        return papers

    pool = _candidate_pool(papers, top_k)
    catalogue = "\n".join(
        f"[{i+1}] {(p.get('title') or '')[:140]} — {(p.get('abstract') or '')[:_ABSTRACT_CHARS]}"
        for i, p in enumerate(pool)
    )
    prompt = f"""Score how relevant each paper is to this research topic, 0-100.

TOPIC: "{topic}"

Judge ONLY topical relevance — NOT language, citations, venue or recency. A paper
in ANY language that is directly about the topic scores high (80-100). A paper
about a different, broader or merely adjacent subject scores low (0-30). Be
strict: papers that drift off-topic must score low.

PAPERS:
{catalogue}

Return a relevance score for every paper number above."""

    try:
        result = llm.with_structured_output(_Rels).invoke(prompt)
        scores = {item.id: max(0, min(100, item.relevance)) for item in result.items}
    except Exception as e:
        logger.warning("LLM rerank failed (%s); keeping original ranking", e)
        return papers

    pool_set = {id(p) for p in pool}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for i, p in enumerate(pool):
        if (i + 1) in scores:
            p["relevance_score"] = float(scores[i + 1])
            p["reranked"] = True
            p["total_score"] = _recompute_total(p)
        if p.get("relevance_score", 0) < min_relevance:
            dropped += 1
            continue
        kept.append(p)

    tail = [p for p in papers if id(p) not in pool_set]

    # Safety: never drop everything. If the threshold filtered the whole pool,
    # keep the most relevant candidates anyway so results are never empty.
    if not kept and pool:
        kept = sorted(pool, key=lambda p: p.get("relevance_score", 0), reverse=True)[:10]
        dropped = len(pool) - len(kept)

    kept.sort(key=lambda p: p.get("total_score", 0), reverse=True)
    tail.sort(key=lambda p: p.get("total_score", 0), reverse=True)

    logger.info("Rerank: scored %d candidates, dropped %d off-topic (< %d).",
                len(pool), dropped, min_relevance)
    print(f"   🎯 Reranked {len(pool)} candidates with LLM; dropped {dropped} off-topic.")

    ranked = kept + tail
    for i, p in enumerate(ranked):
        p["reading_priority"] = i + 1
    return ranked
