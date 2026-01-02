"""
Output formatters for displaying paper results.
"""

from typing import List, Dict, Any
import json


def format_paper_summary(paper: Dict[str, Any]) -> str:
    """
    Format a single paper for display.
    """
    lines = []
    lines.append(f"📄 {paper.get('title', 'No Title')}")
    lines.append(f"   Authors: {', '.join(paper.get('authors', [])[:3])}")
    
    if paper.get("published_date"):
        lines.append(f"   Published: {paper['published_date']}")
    
    if paper.get("journal_name"):
        lines.append(f"   Journal: {paper['journal_name']}")
    
    metrics = []
    if paper.get("citation_count"):
        metrics.append(f"Citations: {paper['citation_count']}")
    if paper.get("q_quartile"):
        metrics.append(f"Quartile: {paper['q_quartile']}")
    if paper.get("impact_factor"):
        metrics.append(f"IF: {paper['impact_factor']:.2f}")
    
    if metrics:
        lines.append(f"   Metrics: {' | '.join(metrics)}")
    
    if paper.get("total_score"):
        lines.append(f"   📊 Score: {paper['total_score']:.2f}")
    
    if paper.get("url"):
        lines.append(f"   🔗 {paper['url']}")
    
    return "\n".join(lines)


def format_reading_list(papers: List[Dict[str, Any]]) -> str:
    """
    Format a list of papers as a detailed reading list.
    Enhanced with links, quality metrics, and relevance scores.
    """
    if not papers:
        return "No papers found."
    
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║           📚 RECOMMENDED READING ORDER                       ║",
        "╚══════════════════════════════════════════════════════════════╝",
        ""
    ]
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'No Title')
        
        # Quality indicator based on score
        score = paper.get("total_score", 0)
        if score >= 70:
            quality_icon = "🌟"  # Excellent
        elif score >= 50:
            quality_icon = "⭐"  # Good
        elif score >= 30:
            quality_icon = "📄"  # Average
        else:
            quality_icon = "📋"  # Basic
        
        lines.append(f"{'─' * 65}")
        lines.append(f"{quality_icon} #{i}: {title}")
        
        # Authors (first 3)
        authors = paper.get("authors", [])
        if authors:
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" (+{len(authors) - 3} more)"
            lines.append(f"   👤 {author_str}")
        
        # Venue/Journal
        venue = paper.get("venue", "") or paper.get("journal_name", "")
        pub_date = paper.get("published_date", "")
        if venue or pub_date:
            venue_str = venue if venue else "Unknown venue"
            if pub_date:
                venue_str += f" ({pub_date[:4]})" if len(pub_date) >= 4 else ""
            q = paper.get("q_quartile", "")
            if q:
                venue_str += f" [{q}]"
            lines.append(f"   📰 {venue_str}")
        
        # Metrics line
        metrics = []
        metrics.append(f"Score: {score:.1f}")
        
        citations = paper.get("citation_count", 0)
        if citations > 0:
            metrics.append(f"Citations: {citations:,}")
        
        relevance = paper.get("relevance_score", 0)
        if relevance > 0:
            metrics.append(f"Relevance: {relevance:.0f}%")
        
        influential = paper.get("influential_citations", 0)
        if influential > 0:
            metrics.append(f"Influential: {influential}")
        
        lines.append(f"   📊 {' | '.join(metrics)}")
        
        # Summary
        if paper.get("summary"):
            summary = paper["summary"]
            if len(summary) > 180:
                summary = summary[:177] + "..."
            lines.append(f"   💡 {summary}")
        
        # Links
        url = paper.get("url", "")
        pdf_url = paper.get("pdf_url", "")
        doi = paper.get("doi", "")
        
        link_parts = []
        if url:
            link_parts.append(f"🔗 {url}")
        if pdf_url and pdf_url != url:
            link_parts.append(f"📥 PDF: {pdf_url}")
        if doi:
            link_parts.append(f"DOI: {doi}")
        
        if link_parts:
            lines.append(f"   {link_parts[0]}")
            for link in link_parts[1:]:
                lines.append(f"   {link}")
        
        lines.append("")
    
    # Summary footer
    total_citations = sum(p.get("citation_count", 0) for p in papers)
    high_score = sum(1 for p in papers if p.get("total_score", 0) >= 50)
    
    lines.append("─" * 65)
    lines.append(f"📈 Statistics: {len(papers)} papers | {total_citations:,} total citations | {high_score} high-quality")
    lines.append("")
    
    return "\n".join(lines)


def format_search_results(results: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Format search results from multiple sources.
    """
    lines = ["🔍 SEARCH RESULTS", "=" * 50, ""]
    
    total = 0
    for source, papers in results.items():
        if papers and not any("error" in p for p in papers):
            lines.append(f"📁 {source.upper()}: {len(papers)} papers found")
            total += len(papers)
    
    lines.append(f"\n📊 Total: {total} papers")
    lines.append("")
    
    return "\n".join(lines)


def export_to_json(papers: List[Dict[str, Any]], filepath: str) -> bool:
    """
    Export papers to a JSON file.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def export_to_markdown(papers: List[Dict[str, Any]], filepath: str) -> bool:
    """
    Export papers to a Markdown file.
    """
    try:
        lines = ["# Academic Paper Analysis Results", "", "## Recommended Reading Order", ""]
        
        for i, paper in enumerate(papers, 1):
            lines.append(f"### {i}. {paper.get('title', 'No Title')}")
            lines.append("")
            
            if paper.get("authors"):
                lines.append(f"**Authors:** {', '.join(paper['authors'][:5])}")
            
            if paper.get("published_date"):
                lines.append(f"**Published:** {paper['published_date']}")
            
            if paper.get("journal_name"):
                lines.append(f"**Journal:** {paper['journal_name']}")
            
            lines.append("")
            
            # Metrics table
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            
            if paper.get("total_score"):
                lines.append(f"| Score | {paper['total_score']:.2f} |")
            if paper.get("citation_count"):
                lines.append(f"| Citations | {paper['citation_count']} |")
            if paper.get("q_quartile"):
                lines.append(f"| Q Quartile | {paper['q_quartile']} |")
            if paper.get("impact_factor"):
                lines.append(f"| Impact Factor | {paper['impact_factor']:.2f} |")
            
            lines.append("")
            
            if paper.get("summary"):
                lines.append("**Summary:**")
                lines.append(f"> {paper['summary']}")
                lines.append("")
            
            if paper.get("url"):
                lines.append(f"**Link:** [{paper['url']}]({paper['url']})")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return True
    except Exception:
        return False
