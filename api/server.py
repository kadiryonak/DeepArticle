"""
FastAPI backend for DeepArticle.

Exposes the multi-agent paper analysis pipeline over HTTP and streams live
agent progress to the browser via Server-Sent Events (SSE), powered by
LangGraph's ``stream()``.

Run with:
    uvicorn api.server:app --reload
or:
    python -m api.server
"""

import json
import os
from typing import Dict, Any, Iterator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_llm_info, SOURCES
from graph.workflow import create_workflow
from utils.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="DeepArticle API",
    description="Multi-agent academic paper discovery & ranking",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Human-friendly labels for each pipeline node, used in progress events.
_NODE_LABELS = {
    "orchestrator": "Starting pipeline",
    "query_analyzer": "Analyzing topic & expanding queries",
    "search": "Searching academic sources",
    "metadata": "Enriching metadata",
    "analysis": "Scoring & ranking papers",
    "summarizer": "Summarizing top papers",
    "prioritizer": "Optimizing reading order",
    "final": "Finalizing",
}
_NODE_ORDER = list(_NODE_LABELS.keys())


class SearchRequest(BaseModel):
    query: str


def _initial_state(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "search_queries": [],
        "topic_analysis": None,
        "papers": [],
        "search_completed": False,
        "metadata_enriched": False,
        "analysis_completed": False,
        "summarization_completed": False,
        "prioritization_completed": False,
        "ranked_papers": [],
        "reading_order": [],
        "messages": [],
        "errors": [],
    }


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/config")
def get_config() -> JSONResponse:
    """Return the active LLM provider/model and configured sources."""
    info = get_llm_info()
    info["sources"] = SOURCES
    return JSONResponse(info)


@app.post("/api/search")
def search(req: SearchRequest) -> JSONResponse:
    """Run the full pipeline synchronously and return the ranked reading list."""
    query = (req.query or "").strip()
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    workflow = create_workflow()
    result = workflow.invoke(_initial_state(query))
    papers = result.get("reading_order") or result.get("ranked_papers") or []
    return JSONResponse(
        {
            "query": query,
            "topic_analysis": result.get("topic_analysis"),
            "papers": papers,
            "count": len(papers),
        }
    )


@app.get("/api/search/stream")
def search_stream(query: str) -> StreamingResponse:
    """
    Run the pipeline and stream agent progress via SSE.

    Events (JSON in the ``data:`` field):
      * ``{"type": "start", "query": ...}``
      * ``{"type": "progress", "node": ..., "label": ..., "step": n, "total": m}``
      * ``{"type": "result", "papers": [...], "topic_analysis": {...}}``
      * ``{"type": "error", "message": ...}``
      * ``{"type": "done"}``
    """
    query = (query or "").strip()

    def generate() -> Iterator[str]:
        if not query:
            yield _sse({"type": "error", "message": "Query is required"})
            yield _sse({"type": "done"})
            return

        yield _sse({"type": "start", "query": query})

        merged: Dict[str, Any] = {}
        total = len(_NODE_ORDER)
        try:
            workflow = create_workflow()
            for chunk in workflow.stream(_initial_state(query), stream_mode="updates"):
                for node, out in chunk.items():
                    if isinstance(out, dict):
                        merged.update(out)
                    step = _NODE_ORDER.index(node) + 1 if node in _NODE_ORDER else 0
                    yield _sse(
                        {
                            "type": "progress",
                            "node": node,
                            "label": _NODE_LABELS.get(node, node),
                            "step": step,
                            "total": total,
                        }
                    )

            papers = merged.get("reading_order") or merged.get("ranked_papers") or []
            yield _sse(
                {
                    "type": "result",
                    "papers": papers,
                    "count": len(papers),
                    "topic_analysis": merged.get("topic_analysis"),
                }
            )
        except Exception as e:  # pragma: no cover - surfaced to client
            logger.exception("Pipeline failed")
            yield _sse({"type": "error", "message": str(e)})

        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the static React single-page app at the root.
if os.path.isdir(_STATIC_DIR):
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
