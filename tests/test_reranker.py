"""
Offline tests for the language-agnostic LLM reranker (fake LLM, no network).
"""

import sys
sys.path.insert(0, '..')

from agents.reranker import llm_rerank, _candidate_pool, _recompute_total, _Rels, _Rel


class _FakeStructured:
    def __init__(self, mapping):
        self.mapping = mapping

    def invoke(self, _prompt):
        return _Rels(items=[_Rel(id=i, relevance=r) for i, r in self.mapping.items()])


class FakeLLM:
    """with_structured_output(...).invoke(...) returns the fixed relevance map."""
    def __init__(self, mapping):
        self.mapping = mapping

    def with_structured_output(self, _schema):
        return _FakeStructured(self.mapping)


def _paper(title, cites, rel=0.0):
    return {
        "title": title, "abstract": "abstract of " + title, "citation_count": cites,
        "relevance_score": rel, "total_score": rel,
        "quality_metrics": {"citation_score": 50, "venue_score": 0,
                            "recency_score": 0, "influential_score": 0},
    }


class TestRerank:
    def test_drops_off_topic_and_rescores(self):
        papers = [_paper("On-topic paper", 100), _paper("Off-topic paper", 50)]
        # [1] = most-cited (on-topic), [2] = off-topic
        out = llm_rerank("the topic", papers, FakeLLM({1: 90, 2: 10}),
                         top_k=10, min_relevance=40)
        titles = [p["title"] for p in out]
        assert "On-topic paper" in titles
        assert "Off-topic paper" not in titles      # dropped (relevance 10 < 40)
        kept = out[0]
        assert kept["relevance_score"] == 90.0
        assert kept.get("reranked") is True

    def test_language_agnostic_keeps_relevant_regardless_of_keyword(self):
        # English paper with 0 keyword relevance but high LLM relevance is kept.
        papers = [_paper("English relevant paper", 200, rel=0.0)]
        out = llm_rerank("konu", papers, FakeLLM({1: 95}), top_k=10, min_relevance=40)
        assert out and out[0]["relevance_score"] == 95.0

    def test_never_returns_empty(self):
        papers = [_paper("A", 100), _paper("B", 50)]
        # Everything scored below threshold -> safety keeps the most relevant.
        out = llm_rerank("topic", papers, FakeLLM({1: 5, 2: 3}),
                         top_k=10, min_relevance=40)
        assert len(out) >= 1

    def test_no_llm_returns_unchanged(self):
        papers = [_paper("A", 1)]
        assert llm_rerank("t", papers, None) is papers

    def test_recompute_total_uses_relevance(self):
        p = _paper("x", 0, rel=80.0)
        p["quality_metrics"]["citation_score"] = 100
        total = _recompute_total(p)
        assert total > 0

    def test_candidate_pool_unions_cite_and_score(self):
        papers = [_paper(f"p{i}", cites=i, rel=100 - i) for i in range(10)]
        pool = _candidate_pool(papers, 3)
        assert len(pool) <= 3
