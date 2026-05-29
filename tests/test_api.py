"""
Offline tests for the FastAPI backend (no network / no LLM).
"""

import json

import sys
sys.path.insert(0, '..')

from fastapi.testclient import TestClient

import api.server as server
from api.server import app

client = TestClient(app)


def _parse_sse(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


class TestConfigEndpoint:
    def test_returns_provider_and_sources(self):
        r = client.get("/api/config")
        assert r.status_code == 200
        body = r.json()
        assert "provider" in body
        assert "sources" in body
        assert isinstance(body["sources"], list)


class TestSearchValidation:
    def test_empty_query_rejected(self):
        r = client.post("/api/search", json={"query": "   "})
        assert r.status_code == 400


class TestStreaming:
    def test_stream_emits_lifecycle_events(self, monkeypatch):
        class FakeWorkflow:
            def stream(self, state, stream_mode="updates"):
                yield {"orchestrator": {"messages": []}}
                yield {"query_analyzer": {"topic_analysis": {"concepts": ["x"]}}}
                yield {"prioritizer": {"reading_order": [{"title": "Paper", "total_score": 80}]}}
                yield {"final": {"messages": []}}

        monkeypatch.setattr(server, "create_workflow", lambda: FakeWorkflow())

        with client.stream("GET", "/api/search/stream?query=test") as resp:
            assert resp.status_code == 200
            text = "".join(resp.iter_text())

        events = _parse_sse(text)
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert "progress" in types
        assert "result" in types
        assert types[-1] == "done"

        result = next(e for e in events if e["type"] == "result")
        assert result["count"] == 1
        assert result["papers"][0]["title"] == "Paper"

    def test_stream_empty_query(self):
        with client.stream("GET", "/api/search/stream?query=") as resp:
            text = "".join(resp.iter_text())
        events = _parse_sse(text)
        assert any(e["type"] == "error" for e in events)
