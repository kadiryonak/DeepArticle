"""
Recommender Agent - turns a ranked paper list into an AI-assisted study plan.

Produces three things the UI surfaces for a researcher:

* ``groups``       - the same papers sliced into useful buckets (newest,
                     most-cited, top-venue, theses, open-access) by ``paper_id``.
* ``reading_path`` - an ordered, staged plan (Foundational -> Core -> Advanced)
                     with a one-line reason per paper, so a newcomer knows what
                     to read first and what comes next.
* ``start_here``   - the single best paper to begin with.

The reading path is LLM-assisted; if no LLM is configured (or it errors), a
deterministic heuristic produces a sensible path instead.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from utils.llm_factory import create_llm
from utils.logging_config import get_logger

logger = get_logger(__name__)


class _PathItem(BaseModel):
    """One step of the reading path (structured-output schema for the LLM)."""
    paper_id: str = Field(description="exact id of one of the listed papers")
    stage: str = Field(description="Foundational, Core, or Advanced / Recent")
    reason: str = Field(description="one-sentence reason to read it at this stage")


class _ReadingPath(BaseModel):
    path: List[_PathItem] = Field(default_factory=list)

_GROUP_SIZE = 5
_Q_RANK = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1, None: 0, "": 0}


def _year(paper: Dict[str, Any]) -> int:
    """Best-effort publication year (0 if unknown)."""
    raw = str(paper.get("published_date") or "")
    m = re.search(r"(19|20)\d{2}", raw)
    return int(m.group(0)) if m else 0


def _ensure_ids(papers: List[Dict[str, Any]]) -> None:
    """Guarantee every paper has a non-empty paper_id (for group references)."""
    for i, p in enumerate(papers):
        if not p.get("paper_id"):
            p["paper_id"] = f"p{i}"


def build_groups(papers: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Slice papers into useful buckets. Returns paper_id lists (top N each)."""
    def ids(items):
        return [p["paper_id"] for p in items[:_GROUP_SIZE]]

    cite = lambda p: p.get("citation_count", 0) or 0  # noqa: E731

    newest = sorted(papers, key=lambda p: (_year(p), cite(p)), reverse=True)
    most_cited = sorted(papers, key=cite, reverse=True)
    top_venues = sorted(
        [p for p in papers if _Q_RANK.get(p.get("q_quartile"), 0) > 0 or p.get("venue")],
        key=lambda p: (_Q_RANK.get(p.get("q_quartile"), 0), cite(p)),
        reverse=True,
    )
    theses = sorted([p for p in papers if p.get("is_thesis")], key=cite, reverse=True)
    open_access = sorted([p for p in papers if p.get("pdf_url")], key=cite, reverse=True)

    return {
        "newest": ids(newest),
        "most_cited": ids(most_cited),
        "top_venues": ids(top_venues),
        "theses": ids(theses),
        "open_access": ids(open_access),
    }


def _heuristic_reading_path(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback study plan with no LLM: foundational (older & highly cited) first,
    then core (high score), then advanced (most recent).
    """
    cite = lambda p: p.get("citation_count", 0) or 0  # noqa: E731
    pool = papers[:12]
    foundational = sorted(pool, key=cite, reverse=True)[:2]
    recent = sorted(pool, key=_year, reverse=True)[:2]
    chosen_ids = {p["paper_id"] for p in foundational + recent}
    core = [p for p in pool if p["paper_id"] not in chosen_ids][:2]

    path = []
    for stage, group in (("Foundational", foundational), ("Core", core), ("Advanced / Recent", recent)):
        for p in group:
            path.append({
                "stage": stage,
                "paper_id": p["paper_id"],
                "title": p.get("title", ""),
                "reason": {
                    "Foundational": "Highly cited — establishes the groundwork.",
                    "Core": "Strong, representative work on the topic.",
                    "Advanced / Recent": "Recent advance to see the current state of the art.",
                }[stage],
            })
    return path


def _llm_reading_path(topic: str, papers: List[Dict[str, Any]], llm) -> Optional[List[Dict[str, Any]]]:
    """Ask the LLM to order the top papers into a staged reading path."""
    top = papers[:12]
    catalogue = "\n".join(
        f"{i+1}. id={p['paper_id']} | {p.get('title','')[:90]} "
        f"({_year(p) or '?'}, {p.get('citation_count',0)} cites)"
        for i, p in enumerate(top)
    )
    prompt = f"""You are advising a researcher new to this topic: "{topic}".

Here are candidate papers (with ids):
{catalogue}

Design a reading path: choose 4-6 of these papers and order them so a newcomer
builds understanding step by step. Assign each a stage — "Foundational", "Core",
or "Advanced / Recent" — and give a one-sentence reason. Use the exact ids shown."""

    try:
        structured = llm.with_structured_output(_ReadingPath)
        result = structured.invoke(prompt)
        by_id = {p["paper_id"]: p for p in top}
        path = []
        for item in result.path:
            if item.paper_id in by_id:
                path.append({
                    "stage": item.stage or "Core",
                    "paper_id": item.paper_id,
                    "title": by_id[item.paper_id].get("title", ""),
                    "reason": (item.reason or "").strip(),
                })
        return path or None
    except Exception as e:
        logger.warning("LLM reading path failed: %s", e)
        return None


def recommender_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: build groups + an AI reading path from the ranked papers."""
    papers = state.get("reading_order") or state.get("ranked_papers") or []
    if not papers:
        return {"groups": {}, "reading_path": [], "start_here": None,
                "recommendation_completed": True}

    print("\n" + "=" * 50)
    print("🧭 RECOMMENDER AGENT - study plan & groups")
    print("=" * 50)

    _ensure_ids(papers)
    groups = build_groups(papers)

    llm = create_llm()
    reading_path = (_llm_reading_path(state.get("query", ""), papers, llm) if llm else None)
    if not reading_path:
        reading_path = _heuristic_reading_path(papers)

    start_here = None
    if reading_path:
        first = reading_path[0]
        start_here = {
            "paper_id": first["paper_id"],
            "title": first["title"],
            "reason": first.get("reason", ""),
        }

    print(f"   Groups: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print(f"   Reading path: {len(reading_path)} papers")

    return {
        "groups": groups,
        "reading_path": reading_path,
        "start_here": start_here,
        "recommendation_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Built study plan ({len(reading_path)} steps) and {len(groups)} groups."}
        ],
    }
