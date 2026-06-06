# 🎓 Multi-Agent Academic Paper Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An AI-powered multi-agent system for discovering, analyzing, and ranking academic papers**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [API Keys](#-api-keys)

</div>

---

## 🚀 Features

- **🔬 Intelligent Query Expansion**: LLM analyzes your topic and generates 20-30 targeted search queries
- **🌍 Bilingual (TR + EN)**: Generates queries in both English and Turkish to find relevant work in either language
- **📚 Multi-Source Search**: Searches ArXiv, Semantic Scholar, OpenAlex and **CORE** concurrently (PubMed optional)
- **🎓 Thesis Discovery**: Finds PhD/Master's theses via OpenAlex dissertations and CORE — including Turkish theses (plus best-effort YÖK Ulusal Tez Merkezi)
- **⚡ Parallel & Cached**: Queries run in a thread pool and API responses are cached on disk (≈75× faster repeat lookups, fewer rate-limit errors)
- **📊 Smart Ranking**: Papers scored by citations, relevance, venue quality, recency and influential citations
- **🤖 Multi-LLM Support**: Works with Groq, OpenAI, Anthropic, or Google AI
- **💡 AI Summaries**: Generates concise summaries for top papers
- **📄 PDF Full-Text**: Optionally extracts full text & IMRaD sections from open-access PDFs
- **🌐 Web UI**: FastAPI backend + React SPA with **live agent-progress streaming** (plus a Chainlit prototype)
- **🧪 Agent Evaluation**: LLM-as-judge quality checks via [DeepEval](evals/README.md)
- **📄 Export**: Results exportable as JSON / Markdown

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kadiryonak/agentic_systemm.git
cd agentic_systemm
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Standard install
pip install -r requirements.txt

# OR install as an editable package (recommended for development)
pip install -e ".[dev]"
```

### 4. Configure API Keys

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key (at least one required)
```

**Minimum requirement**: One LLM API key (Groq recommended - free tier available)

---

## 🔑 API Keys

| Provider | Get API Key | Free Tier |
|----------|-------------|-----------|
| **Groq** (Recommended) | [console.groq.com](https://console.groq.com/keys) | ✅ Yes |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | ❌ No |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | ❌ No |
| Google AI | [makersuite.google.com](https://makersuite.google.com/app/apikey) | ✅ Yes |

---

## 💻 Usage

### Command Line Interface

```bash
# Basic search
python main.py "unit test generation using large language models"

# With output file
python main.py "LLM code generation" --output results.json

# Interactive mode
python main.py --interactive
```

### Web UI (FastAPI + React) — recommended

A modern single-page React UI with a FastAPI backend that **streams live agent
progress** (Server-Sent Events) as the pipeline runs. The frontend is a single
static file (React via CDN) — no `npm`/build step required.

```bash
# Install the API extra
pip install -e ".[api]"        # or: pip install fastapi "uvicorn[standard]"

# Start the server
uvicorn api.server:app --reload
# or
python -m api.server
```

Then open http://localhost:8000 in your browser. Features: live pipeline
progress bar, ranked paper cards with score badges, filters (min citations,
source, sort), and JSON export.

**API endpoints:**
- `GET  /api/config` — active provider, model and sources
- `POST /api/search` — `{ "query": "..." }`, returns the ranked list
- `GET  /api/search/stream?query=...` — SSE stream of agent progress + results

### 🐳 Run with Docker (any machine, no Python setup)

The easiest way to run DeepArticle on another computer — only Docker required:

```bash
# 1. Configure your keys (at minimum one LLM provider)
cp .env.example .env        # then edit .env and add e.g. GROQ_API_KEY

# 2. Build & start (FastAPI + React UI)
docker compose up --build

# 3. Open the UI
#    http://localhost:8000
```

The API response cache is persisted in a named Docker volume (`deeparticle-cache`)
so repeat searches stay fast across restarts. To stop: `docker compose down`.

Without compose, plain Docker works too:

```bash
docker build -t deeparticle .
docker run --rm -p 8000:8000 --env-file .env deeparticle
```

### Web UI (Chainlit) — quick chat prototype

```bash
pip install chainlit
chainlit run app.py
```

Then open http://localhost:8000 in your browser.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                         │
│              (Coordinates the workflow)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 QUERY ANALYZER AGENT                         │
│    • LLM-powered topic analysis                              │
│    • Discovers relevant tools/systems                        │
│    • Generates 20+ targeted search queries                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    SEARCH AGENT                              │
│    • ArXiv API (CS categories: cs.SE, cs.AI, cs.LG)         │
│    • Semantic Scholar API                                    │
│    • Deduplication & citation-based sorting                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   METADATA AGENT                             │
│    • Enriches papers with additional metrics                 │
│    • SCImago journal rankings                                │
│    • CrossRef metadata                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   ANALYSIS AGENT                             │
│    • Calculates relevance scores                             │
│    • Quality metrics (citations, venue, recency)             │
│    • Total score computation                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  SUMMARIZER AGENT                            │
│    • LLM-generated summaries for top 10 papers               │
│    • Fallback to truncated abstracts                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRIORITIZER AGENT                           │
│    • Optimizes reading order                                 │
│    • Source diversity                                        │
│    • Final ranking                                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                              │
│    • Formatted reading list                                  │
│    • JSON export                                             │
│    • Statistics                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Scoring Algorithm

Papers are ranked using a weighted multi-factor algorithm:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Citations** | 25% | Total citation count (log-scaled) |
| **Relevance** | 25% | Keyword match in title/abstract |
| **Venue** | 20% | Conference/journal quality (ICSE, TSE, etc.) |
| **Recency** | 15% | Publication year (2024-2025 highest) |
| **Influence** | 15% | Influential citations from Semantic Scholar |

---

## 📁 Project Structure

All application code lives under `src/` (flat layout). Entry points (`main.py`,
`app.py`) stay at the repo root.

```
DeepArticle/
├── src/                         # ← all application code
│   ├── config.py                # Configuration
│   ├── cli.py                   # CLI implementation (run via ./main.py)
│   ├── agents/
│   │   ├── orchestrator.py      # Main coordinator
│   │   ├── query_analyzer.py    # LLM topic analysis (bilingual TR/EN)
│   │   ├── search_agent.py      # Multi-source parallel search
│   │   ├── metadata_agent.py    # Metadata enrichment
│   │   ├── analysis_agent.py    # Scoring & ranking
│   │   ├── summarizer_agent.py  # LLM summaries
│   │   └── prioritizer_agent.py # Reading order
│   ├── tools/
│   │   ├── arxiv_tools.py             # ArXiv API (cached citation lookups)
│   │   ├── semantic_scholar_tools.py  # Semantic Scholar API (cached)
│   │   ├── openalex_tools.py          # OpenAlex papers + theses (no key)
│   │   ├── core_tools.py              # CORE open-access full text + theses
│   │   ├── yoktez_tools.py            # YÖK Ulusal Tez Merkezi (best-effort)
│   │   ├── scimago_tools.py           # Journal rankings (Q quartile)
│   │   ├── crossref_tools.py          # DOI metadata
│   │   ├── pdf_tools.py               # PDF full-text & section extraction
│   │   ├── pubmed_tools.py            # PubMed API (optional source)
│   │   └── google_scholar_tools.py    # Google Scholar (optional, slow)
│   ├── graph/workflow.py        # LangGraph workflow
│   ├── state/system_state.py    # Shared graph state & PaperMetadata model
│   ├── utils/                   # llm_factory, scoring, cache, logging, formatters
│   ├── api/
│   │   ├── server.py            # FastAPI backend (REST + SSE streaming)
│   │   └── static/index.html    # React single-page UI (no build step)
│   └── evals/                   # DeepEval agent-quality evaluation suite
├── tests/                       # Unit & integration tests (offline)
├── conftest.py                  # Puts src/ on the import path for tests
├── main.py                      # CLI launcher (delegates to src/cli.py)
├── app.py                       # Chainlit Web UI (prototype)
├── Dockerfile / docker-compose.yml
├── .github/                     # CI workflow, issue/PR templates
├── pyproject.toml               # Packaging & tooling config (src layout)
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Dev/test dependencies
├── CONTRIBUTING.md / CODE_OF_CONDUCT.md / SECURITY.md / LICENSE
├── .env.example
└── .gitignore
```

> **Note:** `pip install -e .` puts `src/` on the import path, so existing
> commands (`uvicorn api.server:app`, `python main.py …`) work unchanged.
> `pubmed_tools.py` and `google_scholar_tools.py` are optional sources; the
> default `SOURCES` is `arxiv,semantic_scholar,openalex,openalex_thesis,core`.

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_core.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

**Test Results**: 28/28 tests passing ✅

---

## 🧪 Agent Evaluation (DeepEval)

Beyond unit tests, the project ships an **LLM-as-judge** evaluation suite built
on [DeepEval](https://docs.confident-ai.com/) to measure the *quality* of the
LLM-powered agents (not just that they run):

- **Query Analyzer** → `GEval` for query relevance & specificity
- **Summarizer** → `FaithfulnessMetric` (no hallucinations) + `AnswerRelevancyMetric`

The judge reuses the project's own multi-provider LLM factory, so one API key is
enough. Evals are **skipped automatically** when no key is set, keeping the
default `pytest` run offline.

```bash
pip install -e ".[eval]"
pytest src/evals/ -v -m eval
```

See [`src/evals/README.md`](src/evals/README.md) for details.

---

## 📈 Example Output

```
🎯 MULTI-AGENT ACADEMIC PAPER ANALYSIS SYSTEM
============================================================

📝 Query: unit test generation using large language models

🔬 QUERY ANALYZER AGENT - Deep Topic Analysis
✓ LLM discovered:
   📚 Core Concepts: LLM, unit testing, automated testing
   🔧 Discovered Systems: EvoSuite, Randoop, Pynguin, HITS
   🔍 Generated Queries: 14

📚 SEARCH AGENT - Multi-Query CS Search
   Found: 45 unique papers

📊 ANALYSIS AGENT - Top 5 Papers:
   ⭐ #1: TestPilot (TSE) - Score: 81.0 - Citations: 379
   ⭐ #2: HITS (ASE 2024) - Score: 72.5 - Citations: 60
   ⭐ #3: LLM Evaluation Study (ASE) - Score: 69.5 - Citations: 63
   ⭐ #4: AgoneTest (ICSTW) - Score: 61.5 - Citations: 18
   ⭐ #5: Domain Adaptation (ISSTA) - Score: 57.0 - Citations: 18

✓ Exported to results.json
```

---

## 🔧 Configuration

Edit `.env` file:

```bash
# LLM Provider (auto-detected if not set)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Search Settings
MAX_RESULTS_PER_SOURCE=15
MAX_SEARCH_QUERIES=20

# Sources (papers + theses)
SOURCES=arxiv,semantic_scholar,openalex,openalex_thesis,core

# Bilingual search (English + Turkish)
BILINGUAL_SEARCH=1
SEARCH_LANGUAGES=en,tr
```

### 🌍 Multilingual & 🎓 Thesis Search

DeepArticle searches in **both English and Turkish**. With `BILINGUAL_SEARCH=1`
(the default), the query analyzer produces queries in each language, so a single
topic surfaces relevant work regardless of the language it was published in.

**Theses (PhD / Master's)** are discovered through:

| Source | What it covers | Key needed |
|--------|----------------|-----------|
| `openalex_thesis` | Dissertations in any language — incl. **Turkish** theses indexed from YÖK | No |
| `core` | Open-access full-text theses & papers (multilingual) | Optional (`CORE_API_KEY`) |
| `yoktez` | YÖK Ulusal Tez Merkezi (Turkish theses) — **best-effort** | No |

> **Note on YÖK Tez:** the Ulusal Tez Merkezi has no public API and restricts
> automated access, so the `yoktez` source is best-effort and gracefully returns
> nothing when blocked. For reliable Turkish thesis coverage, `openalex_thesis`
> and `core` index a large share of the same theses through stable APIs. Enable
> the best-effort scraper by adding `yoktez` to `SOURCES`.

### ⚡ Caching

API responses (citations, search results) are cached on disk under `.cache/`,
so repeat searches are dramatically faster and far less likely to hit API rate
limits. A single ArXiv query that previously fired 45+ citation lookups now
serves them from disk on subsequent runs (~75× faster per lookup).

```bash
# Disable caching
DEEPARTICLE_NO_CACHE=1
# Change cache lifetime (seconds, default 7 days)
DEEPARTICLE_CACHE_TTL=604800
```

---

## 📄 License

MIT License - feel free to use this project for your research!

---

## 🤝 Contributing

Contributions are welcome! 🎉 Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, the pull-request process, and style guidelines. By
participating you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

Good first issues: adding a new paper source (OpenAlex, DBLP, CORE), improving
error handling/caching, or expanding test coverage.

---

<div align="center">

**Made with ❤️ using LangChain & LangGraph**

</div>
