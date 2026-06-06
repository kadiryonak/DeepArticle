"""
Resources Agent - gathers real, verifiable supplementary material for a topic:
popular GitHub projects, Medium articles and (optionally) YouTube videos.

Everything comes from live APIs/feeds — never LLM-generated — so the results are
always real and verifiable, or simply absent.
"""

from typing import Any, Dict

from config import ENABLE_RESOURCES
from tools.resources_tools import gather_resources
from utils.logging_config import get_logger

logger = get_logger(__name__)

_EMPTY = {"github": [], "articles": [], "videos": []}


def resources_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: attach supplementary resources for the query topic."""
    if not ENABLE_RESOURCES:
        return {"resources": dict(_EMPTY), "resources_completed": True}

    # GitHub/YouTube/Medium have far better coverage for English queries, so for
    # a Turkish topic prefer an English query produced by the query analyzer.
    topic = state.get("query", "")
    analysis = state.get("topic_analysis") or {}
    english_queries = analysis.get("queries") or []
    search_topic = (english_queries[0] if english_queries else topic).strip().strip('"').strip()

    print("\n" + "=" * 50)
    print("🌐 RESOURCES AGENT - GitHub / Medium / YouTube")
    print("=" * 50)

    try:
        resources = gather_resources(search_topic)
    except Exception as e:  # never break the pipeline for supplementary data
        logger.warning("Resource gathering failed: %s", e)
        resources = dict(_EMPTY)

    print(f"   GitHub repos: {len(resources['github'])} · "
          f"Articles: {len(resources['articles'])} · Videos: {len(resources['videos'])}")

    return {
        "resources": resources,
        "resources_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": (
                f"Gathered {len(resources['github'])} GitHub repos, "
                f"{len(resources['articles'])} articles, {len(resources['videos'])} videos."
            )}
        ],
    }
