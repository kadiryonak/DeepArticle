"""
Chainlit Web UI for Multi-Agent Academic Paper Analysis System.
Provides an interactive chat interface for paper search and analysis.
"""

import chainlit as cl
from typing import Dict, Any, List
import json
import os

# Application code lives under src/ — put it on the import path.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from graph.workflow import run_analysis
from config import LLM_PROVIDER, LLM_MODEL, get_llm_info


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    llm_info = get_llm_info()
    
    await cl.Message(
        content=f"""# 🎓 Academic Paper Analysis System

Welcome! I can help you find and analyze academic papers on any research topic.

**Current Configuration:**
- LLM Provider: `{llm_info['provider'] or 'Not configured'}`
- Model: `{llm_info['model']}`
- API Key: {'✅ Available' if llm_info['has_api_key'] else '❌ Missing'}

**How to use:**
1. Enter a research topic (e.g., "LLM unit test generation")
2. I'll search ArXiv and Semantic Scholar
3. You'll get a ranked list of relevant papers

**Example queries:**
- "unit test generation using large language models"
- "automated software testing with AI"
- "code generation deep learning"

Type your research topic to begin! 🔍
"""
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages and run analysis."""
    query = message.content.strip()
    
    if not query:
        await cl.Message(content="Please enter a research topic to search.").send()
        return
    
    # Show thinking message
    thinking_msg = cl.Message(content="🔍 Analyzing your topic and searching academic databases...")
    await thinking_msg.send()
    
    try:
        # Run the analysis pipeline
        result = await cl.make_async(run_analysis)(query)
        
        # Get the results
        papers = result.get("reading_order", []) or result.get("ranked_papers", [])
        topic_analysis = result.get("topic_analysis", {})
        
        if not papers:
            await cl.Message(
                content="❌ No papers found for this query. Try a different search term."
            ).send()
            return
        
        # Format the response
        response = format_results(query, papers, topic_analysis)
        
        # Send the response
        await cl.Message(content=response).send()
        
        # Also create a downloadable JSON file
        if papers:
            json_content = json.dumps(papers[:20], indent=2, ensure_ascii=False)
            elements = [
                cl.Text(name="results.json", content=json_content, display="inline")
            ]
            await cl.Message(
                content="📥 **Download Results:**",
                elements=elements
            ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ An error occurred: {str(e)}\n\nPlease try again or check your API configuration."
        ).send()


def format_results(query: str, papers: List[Dict], topic_analysis: Dict = None) -> str:
    """Format the analysis results as a markdown message."""
    
    lines = [
        f"# 📚 Results for: \"{query}\"",
        ""
    ]
    
    # Add topic analysis if available
    if topic_analysis:
        systems = topic_analysis.get("systems", [])
        concepts = topic_analysis.get("concepts", [])
        
        if systems or concepts:
            lines.append("## 🔬 Topic Analysis")
            if concepts:
                lines.append(f"**Core Concepts:** {', '.join(concepts[:5])}")
            if systems:
                lines.append(f"**Related Systems:** {', '.join(systems[:5])}")
            lines.append("")
    
    # Summary stats
    total_citations = sum(p.get("citation_count", 0) for p in papers)
    high_quality = sum(1 for p in papers if p.get("total_score", 0) >= 50)
    
    lines.extend([
        "## 📊 Summary",
        f"- **Total Papers Found:** {len(papers)}",
        f"- **Total Citations:** {total_citations:,}",
        f"- **High-Quality Papers (Score ≥ 50):** {high_quality}",
        ""
    ])
    
    # Top papers
    lines.append("## 📄 Top Papers")
    lines.append("")
    
    for i, paper in enumerate(papers[:10], 1):
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:3]) if authors else "Unknown"
        if len(authors) > 3:
            author_str += f" (+{len(authors)-3} more)"
        
        year = paper.get("published_date", "")[:4] if paper.get("published_date") else ""
        venue = paper.get("venue", "") or paper.get("journal_name", "") or ""
        citations = paper.get("citation_count", 0)
        score = paper.get("total_score", 0)
        relevance = paper.get("relevance_score", 0)
        url = paper.get("url", "")
        summary = paper.get("summary", "")[:200] + "..." if paper.get("summary") else ""
        
        # Star for high-scoring papers
        star = "⭐ " if score >= 60 else ""
        
        lines.extend([
            f"### {star}#{i}: {title}",
            f"👤 **Authors:** {author_str}",
            f"📅 **Year:** {year} | 📰 **Venue:** {venue}" if venue else f"📅 **Year:** {year}",
            f"📊 **Score:** {score:.1f} | 📈 **Citations:** {citations} | 🎯 **Relevance:** {relevance}%",
        ])
        
        if summary:
            lines.append(f"💡 {summary}")
        
        if url:
            lines.append(f"🔗 [{url}]({url})")
        
        lines.append("")
    
    if len(papers) > 10:
        lines.append(f"*...and {len(papers) - 10} more papers*")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Run with: chainlit run app.py
    print("Run with: chainlit run app.py")
