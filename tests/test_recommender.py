"""
Offline tests for the recommender agent (grouping + heuristic reading path).
No LLM / no network.
"""

import sys
sys.path.insert(0, '..')

from agents.recommender_agent import (
    build_groups, _heuristic_reading_path, _ensure_ids, _year, recommender_node,
)


def _papers():
    return [
        {"paper_id": "a", "title": "Foundational Work", "citation_count": 1000,
         "published_date": "2015-01-01", "q_quartile": "Q1", "venue": "ICML",
         "pdf_url": "http://x/a.pdf", "is_thesis": False},
        {"paper_id": "b", "title": "Recent Advance", "citation_count": 5,
         "published_date": "2024-06-01", "venue": "", "is_thesis": False},
        {"paper_id": "c", "title": "A Turkish Thesis", "citation_count": 20,
         "published_date": "2021", "is_thesis": True, "pdf_url": ""},
        {"paper_id": "d", "title": "Mid Paper", "citation_count": 300,
         "published_date": "2019", "q_quartile": "Q2", "venue": "NeurIPS"},
    ]


class TestYear:
    def test_parses_year(self):
        assert _year({"published_date": "2021-05-01"}) == 2021
        assert _year({"published_date": "2019"}) == 2019
        assert _year({"published_date": ""}) == 0


class TestEnsureIds:
    def test_fills_missing_ids(self):
        papers = [{"title": "x"}, {"paper_id": "keep", "title": "y"}]
        _ensure_ids(papers)
        assert papers[0]["paper_id"] == "p0"
        assert papers[1]["paper_id"] == "keep"


class TestBuildGroups:
    def test_buckets(self):
        g = build_groups(_papers())
        assert g["most_cited"][0] == "a"      # 1000 cites
        assert g["newest"][0] == "b"          # 2024
        assert g["theses"] == ["c"]           # only the thesis
        assert "a" in g["open_access"]        # has pdf_url
        assert g["top_venues"][0] == "a"      # Q1

    def test_group_size_capped(self):
        many = [{"paper_id": str(i), "title": f"t{i}", "citation_count": i,
                 "published_date": "2020"} for i in range(20)]
        g = build_groups(many)
        assert len(g["most_cited"]) == 5


class TestHeuristicPath:
    def test_stages_present(self):
        path = _heuristic_reading_path(_papers())
        assert path
        stages = {s["stage"] for s in path}
        assert "Foundational" in stages
        assert all("reason" in s and s["title"] for s in path)


class TestNode:
    def test_node_outputs_plan(self, monkeypatch):
        # Force the offline (heuristic) path — no LLM / no network.
        import agents.recommender_agent as ra
        monkeypatch.setattr(ra, "create_llm", lambda *a, **k: None)
        out = recommender_node({"query": "x", "reading_order": _papers()})
        assert out["recommendation_completed"] is True
        assert out["groups"]["most_cited"]
        assert out["reading_path"]
        assert out["start_here"]["paper_id"]

    def test_node_handles_empty(self):
        out = recommender_node({"query": "x", "reading_order": []})
        assert out["groups"] == {}
        assert out["reading_path"] == []
