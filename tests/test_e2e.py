"""
End-to-end pipeline test.

Runs the *entire* LangGraph workflow (query analysis → multi-source search →
metadata → scoring → summaries → reading order) against the live APIs and a real
LLM. Because it needs network access and an API key, it is **opt-in**: it only
runs when ``RUN_E2E=1`` is set and an LLM provider key is configured. The default
``pytest`` run stays fast and offline.

Run it with:
    RUN_E2E=1 pytest tests/test_e2e.py -v -m e2e        # bash
    $env:RUN_E2E="1"; pytest tests/test_e2e.py -v -m e2e  # PowerShell
"""

import os

import pytest

from config import get_active_provider

RUN_E2E = os.getenv("RUN_E2E", "").lower() in ("1", "true", "yes")
HAS_LLM = get_active_provider() is not None

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not RUN_E2E, reason="set RUN_E2E=1 to run the live end-to-end test"),
    pytest.mark.skipif(not HAS_LLM, reason="no LLM provider API key configured"),
]


@pytest.fixture(scope="module")
def result():
    """Run the full pipeline once and share the result across assertions."""
    from graph.workflow import run_analysis
    return run_analysis("retrieval augmented generation")


def test_pipeline_returns_ranked_papers(result):
    papers = result.get("reading_order") or result.get("ranked_papers") or []
    assert papers, "pipeline returned no papers"
    # Every paper carries the core fields the UI relies on.
    for p in papers[:10]:
        assert p.get("title")
        assert "found_in" in p and p["found_in"], "missing provenance (found_in)"
        assert "source_links" in p


def test_pipeline_generates_bilingual_queries(result):
    queries = result.get("search_queries") or []
    assert len(queries) >= 5, "query analyzer did not expand the topic"


def test_no_duplicate_papers(result):
    """The same work must not appear twice (dedup across databases)."""
    papers = result.get("reading_order") or result.get("ranked_papers") or []

    def norm(title):
        return "".join(c for c in (title or "").lower() if c.isalnum())

    titles = [norm(p.get("title")) for p in papers if p.get("title")]
    assert len(titles) == len(set(titles)), "duplicate papers found in results"
