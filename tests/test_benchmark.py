"""
Offline tests for the benchmark harness pure logic (no network / no LLM).
"""

import sys
sys.path.insert(0, '..')

from evals import benchmark as bm
from evals.benchmark_questions import BENCHMARK_QUESTIONS, get_questions


class TestQuestions:
    def test_has_100_questions(self):
        # At least 100 benchmark topics (dataset may grow over time).
        assert len(BENCHMARK_QUESTIONS) >= 100

    def test_ids_unique(self):
        ids = [q["id"] for q in BENCHMARK_QUESTIONS]
        assert len(ids) == len(set(ids))

    def test_bilingual_balanced(self):
        en = sum(q["lang"] == "en" for q in BENCHMARK_QUESTIONS)
        tr = sum(q["lang"] == "tr" for q in BENCHMARK_QUESTIONS)
        assert {q["lang"] for q in BENCHMARK_QUESTIONS} == {"en", "tr"}
        assert en == tr  # kept 50/50 balanced

    def test_filter_and_limit(self):
        tr_total = sum(q["lang"] == "tr" for q in BENCHMARK_QUESTIONS)
        assert len(get_questions(lang="tr")) == tr_total
        assert len(get_questions(limit=5)) == 5
        assert all(q["lang"] == "en" for q in get_questions(lang="en", limit=3))


class TestPassed:
    def test_threshold_metric(self):
        assert bm._passed("query_relevance", 0.7) is True
        assert bm._passed("query_relevance", 0.5) is False

    def test_bool_metric(self):
        assert bm._passed("bilingual_coverage", True) is True
        assert bm._passed("dedup_integrity", False) is False

    def test_none_value(self):
        assert bm._passed("query_relevance", None) is None


class TestAggregate:
    def _records(self):
        return [
            {"id": "a", "error": None, "latency_s": 2.0, "metrics": {
                "query_relevance": 0.9, "bilingual_coverage": True, "query_count": 30}},
            {"id": "b", "error": None, "latency_s": 4.0, "metrics": {
                "query_relevance": 0.4, "bilingual_coverage": True, "query_count": 8}},
            {"id": "c", "error": "Boom", "latency_s": 1.0, "metrics": {}},
        ]

    def test_counts_and_means(self):
        s = bm.aggregate(self._records())
        assert s["n"] == 3
        assert s["errors"] == 1
        qr = s["metrics"]["query_relevance"]
        assert qr["n"] == 2
        assert abs(qr["mean"] - 0.65) < 1e-6
        assert qr["pass_rate"] == 0.5  # one of two >= 0.60

    def test_pass_rate_bool(self):
        s = bm.aggregate(self._records())
        assert s["metrics"]["bilingual_coverage"]["pass_rate"] == 1.0

    def test_latency_percentiles(self):
        s = bm.aggregate(self._records())
        assert "latency" in s and s["latency"]["max"] == 4.0


class TestVerdictAndReport:
    def test_verdict_thresholds(self):
        assert "Production-ready" in bm._verdict({"metrics": {"m": {"pass_rate": 0.95}}})
        assert "Near" in bm._verdict({"metrics": {"m": {"pass_rate": 0.8}}})
        assert "Not ready" in bm._verdict({"metrics": {"m": {"pass_rate": 0.5}}})

    def test_markdown_renders(self):
        s = bm.aggregate([
            {"id": "a", "error": None, "latency_s": 2.0,
             "metrics": {"query_relevance": 0.9, "bilingual_coverage": True, "query_count": 30}},
        ])
        md = bm.to_markdown(s, {"date": "now", "judge": "test", "lang": "all", "mode": "shallow"})
        assert "# DeepArticle Benchmark Report" in md
        assert "query_relevance" in md
