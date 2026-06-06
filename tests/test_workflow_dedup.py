"""
Regression test for the paper-doubling bug.

The `papers` state channel previously used operator.add, so when the metadata
agent returned the (full) enriched list it was concatenated onto the search
list, silently doubling every paper. This test runs the real workflow with all
node functions stubbed and asserts the count is preserved end to end.
"""

import sys
sys.path.insert(0, '..')

import graph.workflow as wf


def test_papers_are_not_doubled(monkeypatch):
    papers = [{"title": "A", "paper_id": "1"}, {"title": "B", "paper_id": "2"}]

    monkeypatch.setattr(wf, "orchestrator_node", lambda s: {})
    monkeypatch.setattr(wf, "query_analyzer_node", lambda s: {"search_queries": ["q"]})
    monkeypatch.setattr(wf, "search_agent_node",
                        lambda s: {"papers": list(papers), "search_completed": True})
    # Metadata returns the COMPLETE enriched list (same length).
    monkeypatch.setattr(wf, "metadata_agent_node",
                        lambda s: {"papers": [dict(p, enriched=True) for p in s["papers"]],
                                   "metadata_enriched": True})
    monkeypatch.setattr(wf, "analysis_agent_node",
                        lambda s: {"ranked_papers": s["papers"], "analysis_completed": True})
    monkeypatch.setattr(wf, "summarizer_agent_node", lambda s: {"summarization_completed": True})
    monkeypatch.setattr(wf, "prioritizer_agent_node",
                        lambda s: {"reading_order": s["ranked_papers"][:20],
                                   "prioritization_completed": True})
    monkeypatch.setattr(wf, "recommender_node",
                        lambda s: {"groups": {}, "reading_path": [], "start_here": None,
                                   "recommendation_completed": True})
    monkeypatch.setattr(wf, "resources_node",
                        lambda s: {"resources": {}, "resources_completed": True})
    monkeypatch.setattr(wf, "final_output_node", lambda s: {})

    result = wf.run_analysis("some topic")

    # The bug doubled this to 4. It must stay 2.
    assert len(result["papers"]) == 2
    assert len(result["ranked_papers"]) == 2
    assert all(p.get("enriched") for p in result["papers"])
