"""
Query Analyzer Agent - Uses LLM to analyze topic and generate targeted search queries.
Specialized for Software Engineering research with category-based discovery.
"""

from typing import Dict, Any, List

import sys
sys.path.insert(0, '..')

# Use the LLM factory for multi-provider support
from utils.llm_factory import create_llm


def analyze_topic_with_llm(topic: str, llm) -> Dict[str, Any]:
    """
    Use LLM to deeply analyze the research topic and generate categorized search queries.
    """
    prompt = f"""You are an expert researcher in Software Engineering, specifically in automated testing and AI/ML for code.

RESEARCH TOPIC: "{topic}"

I need to find ALL relevant academic papers on this topic. Analyze the topic deeply and provide:

## 1. CORE CONCEPTS (5-7 key technical terms)
List the fundamental concepts in this research area.

## 2. KNOWN TOOLS/SYSTEMS (list all you know)
List specific tools, frameworks, and systems that are famous in this research area.
Think about papers from ICSE, FSE, ASE, ISSTA, ESEC, TSE, TOSEM conferences/journals.
Examples might include specific named systems (like EvoSuite for test generation).

## 3. RESEARCH CATEGORIES
Categorize the research into sub-areas like:
- Technique-focused papers
- Empirical studies
- Tool papers
- Benchmark papers
- Survey papers

## 4. TARGETED SEARCH QUERIES (10-15 very specific queries)
Create search queries that would find:
- Papers about specific tools/systems
- Empirical evaluations
- Comparative studies
- State-of-the-art surveys
- Recent advances (2023-2025)

Format your response EXACTLY like this:

CONCEPTS: concept1, concept2, concept3, concept4, concept5
SYSTEMS: system1, system2, system3, system4, system5, system6
CATEGORIES: category1, category2, category3, category4
QUERIES:
- specific query 1
- specific query 2
- specific query 3
...etc

Be very specific! Generic queries like "AI testing" won't find the right papers.
Use terms that researchers actually use in paper titles."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        result = {
            "concepts": [],
            "systems": [],
            "categories": [],
            "queries": []
        }
        
        lines = content.split("\n")
        in_queries = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("CONCEPTS:"):
                result["concepts"] = [k.strip() for k in line.replace("CONCEPTS:", "").split(",") if k.strip()]
            elif line.startswith("SYSTEMS:"):
                result["systems"] = [s.strip() for s in line.replace("SYSTEMS:", "").split(",") if s.strip()]
            elif line.startswith("CATEGORIES:"):
                result["categories"] = [c.strip() for c in line.replace("CATEGORIES:", "").split(",") if c.strip()]
            elif line.startswith("QUERIES:"):
                in_queries = True
            elif in_queries and line.startswith("-"):
                query = line.lstrip("-").strip()
                if query and len(query) > 5:
                    result["queries"].append(query)
        
        return result
        
    except Exception as e:
        print(f"   ⚠ LLM analysis failed: {e}")
        return None


def generate_search_queries(topic: str, analysis: Dict[str, Any] = None) -> List[str]:
    """
    Generate comprehensive search queries from LLM analysis.
    """
    queries = []
    
    if analysis:
        # 1. Use LLM-generated queries (most important)
        queries.extend(analysis.get("queries", []))
        
        # 2. Search for each discovered system/tool
        for system in analysis.get("systems", [])[:8]:
            if len(system) > 2:
                queries.append(f"{system} test generation")
                queries.append(system)  # Also search just the name
        
        # 3. Core concept combinations
        concepts = analysis.get("concepts", [])
        if len(concepts) >= 2:
            queries.append(" ".join(concepts[:3]))
            queries.append(" ".join(concepts[1:4]))
    
    # 4. Always include original topic variations
    queries.insert(0, topic)
    queries.append(f"{topic} empirical study")
    queries.append(f"{topic} survey")
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for q in queries:
        q_clean = q.lower().strip()
        if q_clean not in seen and len(q) > 4:
            seen.add(q_clean)
            unique.append(q)
    
    return unique[:20]  # Up to 20 queries for comprehensive coverage


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
    
    llm = create_llm()
    analysis = None
    
    if llm:
        print("📝 Performing deep analysis with LLM...")
        print("   (Discovering concepts, tools, systems, and generating queries)\n")
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
