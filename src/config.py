"""
Configuration for the Multi-Agent Academic Paper Analysis System.
Supports multiple LLM providers: Groq, OpenAI, Anthropic, Google.
"""

import os
import sys
from dotenv import load_dotenv

# The agents print emoji-rich progress to stdout. On consoles whose encoding
# isn't UTF-8 (e.g. Windows cp1252/cp1254), those prints raise
# UnicodeEncodeError and crash the pipeline. Reconfigure stdout/stderr to UTF-8
# with a safe fallback so progress output never breaks a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Not a reconfigurable stream (e.g. pytest capture); ignore.
        pass

# Load environment variables
load_dotenv()

# =============================================================================
# API Keys - Get from environment variables
# =============================================================================

# LLM Provider API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Search API Keys
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
# CORE (open-access aggregator) — optional; anonymous access works but is rate-limited.
CORE_API_KEY = os.getenv("CORE_API_KEY", "")

# Supplementary resources (GitHub repos, Medium articles, YouTube videos).
# GitHub works without a token (low rate limit); add GITHUB_TOKEN for more.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MIN_STARS = int(os.getenv("GITHUB_MIN_STARS", "100"))
# YouTube requires an API key; videos are simply omitted when it's not set.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
# Master switch for the supplementary-resources step.
ENABLE_RESOURCES = os.getenv("ENABLE_RESOURCES", "1").lower() not in ("0", "false", "no")

# =============================================================================
# LLM Provider Selection
# =============================================================================

# Available providers: "groq", "openai", "anthropic", "google"
# Auto-detect based on available API keys
def get_active_provider():
    """Determine which LLM provider to use based on available API keys."""
    if GROQ_API_KEY:
        return "groq"
    elif OPENAI_API_KEY:
        return "openai"
    elif ANTHROPIC_API_KEY:
        return "anthropic"
    elif GOOGLE_API_KEY:
        return "google"
    return None

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "") or get_active_provider()

# =============================================================================
# LLM Model Settings per Provider
# =============================================================================

LLM_MODELS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",  # Fast and capable
        "alternatives": ["mixtral-8x7b-32768", "llama-3.1-8b-instant"]
    },
    "openai": {
        "model": "gpt-4o-mini",  # Cost-effective
        "alternatives": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "anthropic": {
        "model": "claude-haiku-4-5",  # Fast & cost-effective
        "alternatives": ["claude-opus-4-8", "claude-sonnet-4-6"]
    },
    "google": {
        "model": "gemini-1.5-flash",  # Fast
        "alternatives": ["gemini-1.5-pro", "gemini-2.0-flash-exp"]
    }
}

# Get current model based on provider
LLM_MODEL = os.getenv("LLM_MODEL", "") or (
    LLM_MODELS.get(LLM_PROVIDER, {}).get("model", "llama-3.3-70b-versatile")
)

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# =============================================================================
# Search Settings
# =============================================================================

MAX_RESULTS_PER_SOURCE = int(os.getenv("MAX_RESULTS_PER_SOURCE", "15"))
# Max expanded search queries. Kept small for focused, on-topic, deterministic
# results (more queries = broader recall but more topic drift).
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "12"))

# Number of worker threads for concurrent searching across queries/sources
SEARCH_MAX_WORKERS = int(os.getenv("SEARCH_MAX_WORKERS", "8"))

# Active sources for paper search. Configurable via the SOURCES env var
# (comma-separated). Supported:
#   arxiv, semantic_scholar, openalex   — papers
#   openalex_thesis                     — PhD/Master's theses (any language, incl. Turkish)
#   core                                — open-access full text + theses (multilingual)
#   crossref                            — DOI metadata + citation counts (~150M works)
#   doaj                                — open-access journal articles
#   dblp                                — computer-science bibliography
#   openaire                            — European open-access aggregator (optional)
#   pubmed                              — biomedical (optional)
#   yoktez                              — YÖK Ulusal Tez Merkezi (Turkish theses, best-effort)
SOURCES = [
    s.strip()
    for s in os.getenv(
        "SOURCES",
        "arxiv,semantic_scholar,openalex,openalex_thesis,core,crossref,doaj,dblp",
    ).split(",")
    if s.strip()
]

# Limit Semantic Scholar to the first N queries to avoid aggressive rate limits.
SEMANTIC_SCHOLAR_QUERY_LIMIT = int(os.getenv("SEMANTIC_SCHOLAR_QUERY_LIMIT", "5"))

# =============================================================================
# Multilingual Search
# =============================================================================

# When enabled, the query analyzer generates search queries in both English and
# Turkish so the system finds relevant work in either language. Disable with
# BILINGUAL_SEARCH=0.
BILINGUAL_SEARCH = os.getenv("BILINGUAL_SEARCH", "1").lower() not in ("0", "false", "no")

# Target languages for query generation (comma-separated ISO codes).
SEARCH_LANGUAGES = [
    s.strip()
    for s in os.getenv("SEARCH_LANGUAGES", "en,tr").split(",")
    if s.strip()
]

# =============================================================================
# Scoring Weights for Paper Ranking
# =============================================================================

SCORING_WEIGHTS = {
    "citation_count": 0.25,
    "relevance": 0.25,
    "venue_quality": 0.20,
    "recency": 0.15,
    "influential_citations": 0.15
}

# Q Quartile Scores
Q_QUARTILE_SCORES = {
    "Q1": 100,
    "Q2": 75,
    "Q3": 50,
    "Q4": 25,
    None: 0
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_llm_info():
    """Get current LLM configuration info."""
    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "has_api_key": bool(
            (LLM_PROVIDER == "groq" and GROQ_API_KEY) or
            (LLM_PROVIDER == "openai" and OPENAI_API_KEY) or
            (LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY) or
            (LLM_PROVIDER == "google" and GOOGLE_API_KEY)
        )
    }


def print_config():
    """Print current configuration for debugging."""
    info = get_llm_info()
    print(f"LLM Provider: {info['provider']}")
    print(f"LLM Model: {info['model']}")
    print(f"API Key Available: {info['has_api_key']}")
    print(f"Max Results per Source: {MAX_RESULTS_PER_SOURCE}")
    print(f"Max Search Queries: {MAX_SEARCH_QUERIES}")
