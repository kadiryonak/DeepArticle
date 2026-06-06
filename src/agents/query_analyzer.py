"""
Query Analyzer Agent - Uses an LLM to analyze a research topic and generate a
small set of focused, on-topic search queries.

Domain-agnostic and deterministic: it adapts to whatever the topic is (it is NOT
specialized for any single field) and is run at temperature 0 so the same topic
yields the same queries. The emphasis is on staying tightly on-topic rather than
maximizing recall, to avoid pulling in unrelated papers.
"""

from typing import Dict, Any, List

import sys
sys.path.insert(0, '..')

# Use the LLM factory for multi-provider support
from utils.llm_factory import create_llm
from config import BILINGUAL_SEARCH, MAX_SEARCH_QUERIES


def analyze_topic_with_llm(topic: str, llm) -> Dict[str, Any]:
    """
    Analyze the research topic and generate a few precise, on-topic queries.
    """
    bilingual_block = ""
    queries_format = """QUERIES:
- specific query 1
- specific query 2
...(5-6 total)"""

    if BILINGUAL_SEARCH:
        bilingual_block = """
## BILINGUAL OUTPUT
The audience reads both English and Turkish. Give the queries in BOTH languages:
put the English queries under QUERIES and their Turkish equivalents (phrased the
way Turkish academics actually write them) under QUERIES_TR. They must be
translations of the SAME on-topic queries — do not introduce new subtopics.
"""
        queries_format = """QUERIES:
- specific English query 1
- specific English query 2
...(5-6 total)
QUERIES_TR:
- aynı sorgunun Türkçesi 1
- aynı sorgunun Türkçesi 2
...(5-6 total)"""

    prompt = f"""You are an expert academic research librarian. Find papers that are
STRICTLY about one specific research topic.

RESEARCH TOPIC: "{topic}"

Rules:
- Stay tightly focused on THIS exact topic. Do NOT drift into adjacent, broader,
  or merely related fields. Every query must match papers specifically about
  "{topic}".
- Do NOT invent tools/systems you are unsure about. If none are clearly relevant,
  leave SYSTEMS empty.
- Prefer the precise terminology researchers use in paper titles for this topic.

Provide:

## 1. CORE CONCEPTS (3-5 key terms that are central to THIS topic)

## 2. KEY METHODS/SYSTEMS (named methods, models or datasets specific to THIS
topic — only if you are confident; otherwise leave empty)

## 3. TARGETED SEARCH QUERIES (exactly 5-6 precise, on-topic queries)
{bilingual_block}
Format your response EXACTLY like this:

CONCEPTS: concept1, concept2, concept3
SYSTEMS: system1, system2
{queries_format}

Keep every query on-topic and specific. No generic or off-topic queries."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        result = {
            "concepts": [],
            "systems": [],
            "categories": [],
            "queries": [],
            "queries_tr": [],
        }

        lines = content.split("\n")
        section = None  # None | "queries" | "queries_tr"

        for line in lines:
            line = line.strip()

            if line.startswith("CONCEPTS:"):
                result["concepts"] = [k.strip() for k in line.replace("CONCEPTS:", "").split(",") if k.strip()]
                section = None
            elif line.startswith("SYSTEMS:"):
                result["systems"] = [s.strip() for s in line.replace("SYSTEMS:", "").split(",") if s.strip()]
                section = None
            elif line.startswith("CATEGORIES:"):
                result["categories"] = [c.strip() for c in line.replace("CATEGORIES:", "").split(",") if c.strip()]
                section = None
            elif line.startswith("QUERIES_TR:"):
                section = "queries_tr"
            elif line.startswith("QUERIES:"):
                section = "queries"
            elif section and line.startswith("-"):
                query = line.lstrip("-").strip()
                if query and len(query) > 5:
                    result[section].append(query)

        return result
        
    except Exception as e:
        print(f"   ⚠ LLM analysis failed: {e}")
        return None


def generate_search_queries(topic: str, analysis: Dict[str, Any] = None) -> List[str]:
    """
    Build a small, focused query list from the LLM analysis.

    Deliberately conservative: just the original topic plus the LLM's on-topic
    queries (English then Turkish). We no longer append generic expansions
    ("<topic> survey", "<system> test generation", bare system names, concept
    combos) — those broadened recall but caused topic drift into unrelated work.
    """
    queries: List[str] = [topic]

    if analysis:
        queries.extend(analysis.get("queries", []))
        queries.extend(analysis.get("queries_tr", []))

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for q in queries:
        q_clean = q.lower().strip().strip('"')
        if q_clean not in seen and len(q) > 4:
            seen.add(q_clean)
            unique.append(q)

    return unique[:MAX_SEARCH_QUERIES]


def query_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for Query Analyzer.
    Performs deep topic analysis for comprehensive paper discovery.
    """
    topic = state.get("query", "")
    
    if not topic:
        return state
    
    print("\n" + "=" * 65)
    print("🔬 QUERY ANALYZER AGENT - Deep Topic Analysis")
    print("=" * 65)
    print(f"Topic: {topic}\n")
    
    # Temperature 0 → deterministic, reproducible query generation.
    llm = create_llm(temperature=0.0)
    analysis = None

    if llm:
        print("📝 Performing focused, deterministic analysis with LLM...")
        print("   (Extracting concepts and a few on-topic queries)\n")
        analysis = analyze_topic_with_llm(topic, llm)
        
        if analysis:
            print("✓ Analysis Complete!")
            print(f"\n   📚 Core Concepts: {', '.join(analysis.get('concepts', [])[:5])}")
            print(f"   🔧 Discovered Systems: {', '.join(analysis.get('systems', [])[:6])}")
            print(f"   📂 Categories: {', '.join(analysis.get('categories', [])[:4])}")
            print(f"   🔍 Generated Queries: {len(analysis.get('queries', []))}")
    else:
        print("   ⚠ LLM not available - using basic search")
    
    # Generate comprehensive queries
    queries = generate_search_queries(topic, analysis)
    
    print(f"\n{'─' * 65}")
    print(f"📋 Search Plan ({len(queries)} queries):")
    for i, q in enumerate(queries[:10], 1):
        q_display = q[:55] + "..." if len(q) > 55 else q
        print(f"   {i:2}. {q_display}")
    if len(queries) > 10:
        print(f"   ... and {len(queries) - 10} more queries")
    
    return {
        **state,
        "search_queries": queries,
        "topic_analysis": analysis,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Deep analysis complete. {len(queries)} queries, {len((analysis or {}).get('systems', []))} systems discovered."}
        ]
    }
