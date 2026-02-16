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

- **🔬 Intelligent Query Expansion**: LLM analyzes your topic and generates 20+ targeted search queries
- **📚 Multi-Source Search**: Searches ArXiv and Semantic Scholar simultaneously
- **📊 Smart Ranking**: Papers scored by citations, relevance, venue quality, and recency
- **🤖 Multi-LLM Support**: Works with Groq, OpenAI, Anthropic, or Google AI
- **💡 AI Summaries**: Generates concise summaries for top papers
- **🌐 Web UI**: Beautiful Chainlit interface for interactive exploration
- **📄 Export**: Results exportable as JSON

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
pip install -r requirements.txt
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

### Web UI (Chainlit)

```bash
# Install chainlit
pip install chainlit

# Run the web interface
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

```
agentic_systemm/
├── agents/
│   ├── orchestrator.py      # Main coordinator
│   ├── query_analyzer.py    # LLM topic analysis
│   ├── search_agent.py      # Multi-source search
│   ├── metadata_agent.py    # Metadata enrichment
│   ├── analysis_agent.py    # Scoring & ranking
│   ├── summarizer_agent.py  # LLM summaries
│   └── prioritizer_agent.py # Reading order
├── tools/
│   ├── arxiv_tools.py       # ArXiv API
│   ├── semantic_scholar_tools.py
│   ├── scimago_tools.py     # Journal rankings
│   └── crossref_tools.py    # DOI metadata
├── graph/
│   └── workflow.py          # LangGraph workflow
├── utils/
│   ├── llm_factory.py       # Multi-provider LLM
│   ├── scoring.py           # Scoring algorithm
│   └── formatters.py        # Output formatting
├── tests/
│   ├── test_core.py         # Unit tests
│   ├── test_integration.py  # Integration tests
│   └── test_tools.py        # Tool tests
├── docs/
│   └── PAPER_SELECTION.md   # Scoring documentation
├── app.py                   # Chainlit Web UI
├── main.py                  # CLI entry point
├── config.py                # Configuration
├── requirements.txt
├── .env.example
└── .gitignore
```

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
```

---

## 📄 License

MIT License - feel free to use this project for your research!

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

**Made with ❤️ using LangChain & LangGraph**

</div>
