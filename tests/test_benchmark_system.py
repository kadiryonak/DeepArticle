"""
Offline/system-level smoke tests for the benchmark harness.

These tests mock external dependencies (LLMs, search, scoring) so the
benchmark can be exercised in CI without real API keys or network access.
"""

import sys
sys.path.insert(0, "..")

from evals import benchmark as bm
from evals.benchmark_questions import get_questions


class DummyLLM:
    pass


class DummyJudge:
    def get_model_name(self):
        return "dummy-judge"


def test_run_one_shallow(monkeypatch):
    q = get_questions(limit=1)[0]

    import agents.query_analyzer as qa

    monkeypatch.setattr(qa, "analyze_topic_with_llm", lambda topic, llm: {
        "queries": ["machine learning healthcare"],
        "queries_tr": ["makine öğrenimi sağlık"],
    })
    monkeypatch.setattr(qa, "generate_search_queries", lambda topic, analysis: ["q1", "q2"])

    # Avoid depending on deepeval / LLM judge; return a deterministic score
    monkeypatch.setattr(bm, "_score_query_relevance", lambda topic, queries, judge: 0.8)

    rec = bm.run_one(q, DummyLLM(), DummyJudge(), deep=False)

    assert rec["metrics"]["query_count"] == 2
    assert rec["metrics"]["query_relevance"] == 0.8
    assert rec["metrics"]["bilingual_coverage"] is True


def test_run_one_deep(monkeypatch):
    q = get_questions(limit=1)[0]

    import agents.query_analyzer as qa
    import agents.search_agent as sa
    import agents.summarizer_agent as suma

    monkeypatch.setattr(qa, "analyze_topic_with_llm", lambda topic, llm: {"queries": ["q1"], "queries_tr": []})
    monkeypatch.setattr(qa, "generate_search_queries", lambda topic, analysis: ["q1"])

    monkeypatch.setattr(bm, "_score_query_relevance", lambda topic, queries, judge: 0.5)

    # Mock search to return a single paper with title+abstract
    monkeypatch.setattr(sa, "search_cs_sources", lambda queries, max_results_per_query=15: [[{"title": "T", "abstract": "A"}]])
    monkeypatch.setattr(sa, "merge_papers", lambda results: [{"title": "T", "abstract": "A"}])

    # Mock summarizer and evaluation scoring
    monkeypatch.setattr(suma, "generate_summary", lambda llm, paper: "This is a summary.")
    monkeypatch.setattr(bm, "_score_summary", lambda topic, summary, abstract, judge: {"summary_faithfulness": 0.9, "summary_relevancy": 0.8})

    rec = bm.run_one(q, DummyLLM(), DummyJudge(), deep=True)

    assert rec["metrics"]["retrieval_count"] == 1
    assert rec["metrics"]["dedup_integrity"] is True
    assert rec["metrics"]["summary_faithfulness"] == 0.9
