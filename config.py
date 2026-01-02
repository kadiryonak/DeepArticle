"""
Configuration for the Multi-Agent Academic Paper Analysis System.
Supports multiple LLM providers: Groq, OpenAI, Anthropic, Google.
"""

import os
from dotenv import load_dotenv

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
        "model": "claude-3-5-sonnet-20241022",  # Balanced
        "alternatives": ["claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
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
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "20"))

# Active sources for paper search
SOURCES = ["arxiv", "semantic_scholar"]  # Removed pubmed for CS focus

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
