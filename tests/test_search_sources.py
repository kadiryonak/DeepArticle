"""
Offline tests for multi-source / parallel search behavior.
"""

import sys
sys.path.insert(0, '..')

from tools.openalex_tools import _reconstruct_abstract
import agents.search_agent as sa


class TestOpenAlexAbstract:
    def test_reconstruct_abstract_orders_words(self):
        inverted = {"Large": [0], "language": [1], "models": [2]}
        assert _reconstruct_abstract(inverted) == "Large language models"

    def test_reconstruct_handles_repeated_words(self):
        inverted = {"the": [0, 3], "quick": [1], "fox": [2], "jumps": [4]}
        assert _reconstruct_abstract(inverted) == "the quick fox the jumps"

    def test_reconstruct_empty(self):
        assert _reconstruct_abstract({}) == ""


class TestParallelSearch:
    def test_dispatches_to_configured_sources(self, monkeypatch):
        def fake_a(query, max_results):
            return [{"title": f"A:{query}", "source": "src_a", "citation_count": 1}]

        def fake_b(query, max_results):
            return [{"title": f"B:{query}", "source": "src_b", "citation_count": 2}]

        monkeypatch.setattr(sa, "SOURCES", ["src_a", "src_b"])
        monkeypatch.setattr(
            sa,
            "_SOURCE_FUNCS",
            {
                "src_a": (fake_a, lambda i: True),
                "src_b": (fake_b, lambda i: True),
            },
        )

        results = sa.search_cs_sources(["q1", "q2"], max_results_per_query=5)

        # Both sources present, each got both queries.
        assert set(results.keys()) == {"src_a", "src_b"}
        assert len(results["src_a"]) == 2
        assert len(results["src_b"]) == 2

    def test_respects_per_source_query_limit(self, monkeypatch):
        def fake(query, max_results):
            return [{"title": query, "source": "limited", "citation_count": 0}]

        monkeypatch.setattr(sa, "SOURCES", ["limited"])
        monkeypatch.setattr(
            sa,
            "_SOURCE_FUNCS",
            {"limited": (fake, lambda i: i < 1)},  # only first query
        )

        results = sa.search_cs_sources(["q1", "q2", "q3"], max_results_per_query=5)
        assert len(results["limited"]) == 1
