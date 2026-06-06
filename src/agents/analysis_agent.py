"""
Analysis Agent - Scores and ranks papers based on multiple criteria.
Enhanced with topic relevance analysis and deeper quality assessment.
"""

from typing import Dict, Any, List
import re

import sys
sys.path.insert(0, '..')

from config import SCORING_WEIGHTS
from utils.scoring import calculate_total_score, rank_papers


def calculate_topic_relevance(paper: Dict[str, Any], query: str) -> float:

    if not query:
        return 50.0
    
    # Normalize query keywords
    query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
    query_keywords.discard('and')
    query_keywords.discard('or')
    query_keywords.discard('the')
    query_keywords.discard('with')
    query_keywords.discard('for')
    
    if not query_keywords:
        return 50.0
    
    # Get text to match against
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    fields = " ".join(paper.get("fields_of_study", [])).lower()
    
    full_text = f"{title} {abstract} {fields}"
    
    # Count keyword matches
    matches = 0
    for keyword in query_keywords:
        if len(keyword) >= 3:  # Skip very short words
            count = full_text.count(keyword)
            if count > 0:
                matches += min(count, 5)  # Cap at 5 per keyword
    
    # Title matches are worth more
    title_matches = sum(1 for kw in query_keywords if kw in title and len(kw) >= 3)
    
    # Calculate score
    base_score = min(matches * 10, 70)  # Max 70 from frequency
    title_bonus = min(title_matches * 15, 30)  # Max 30 from title
    
    return min(base_score + title_bonus, 100)


def calculate_quality_score(paper: Dict[str, Any]) -> Dict[str, Any]:

    quality = {
        "citation_score": 0,
        "venue_score": 0,
        "recency_score": 0,
        "completeness_score": 0,
        "influential_score": 0
    }
    
    # Citation score (logarithmic scale for fairness)
    citations = paper.get("citation_count", 0)
    if citations > 1000:
        quality["citation_score"] = 100
    elif citations > 500:
        quality["citation_score"] = 90
    elif citations > 200:
        quality["citation_score"] = 80
    elif citations > 100:
        quality["citation_score"] = 70
    elif citations > 50:
        quality["citation_score"] = 60
    elif citations > 20:
        quality["citation_score"] = 50
    elif citations > 10:
        quality["citation_score"] = 40
    elif citations > 5:
        quality["citation_score"] = 30
    elif citations > 0:
        quality["citation_score"] = 20
    
    # Venue/Journal score
    venue = paper.get("venue", "") or paper.get("journal_name", "")
    venue_type = paper.get("venue_type", "")
    q_quartile = paper.get("q_quartile")
    
    if q_quartile == "Q1":
        quality["venue_score"] = 100
    elif q_quartile == "Q2":
        quality["venue_score"] = 75
    elif q_quartile == "Q3":
        quality["venue_score"] = 50
    elif q_quartile == "Q4":
        quality["venue_score"] = 25
    elif venue_type == "journal":
        quality["venue_score"] = 60
    elif venue_type == "conference":
        quality["venue_score"] = 50
    elif venue:
        quality["venue_score"] = 30
    
    # Recency score
    pub_date = paper.get("published_date", "")
    if pub_date:
        try:
            year = int(pub_date[:4])
            from datetime import datetime
            current_year = datetime.now().year
            age = current_year - year
            
            if age == 0:
                quality["recency_score"] = 100
            elif age == 1:
                quality["recency_score"] = 90
            elif age <= 2:
                quality["recency_score"] = 80
            elif age <= 3:
                quality["recency_score"] = 70
            elif age <= 5:
                quality["recency_score"] = 50
            elif age <= 10:
                quality["recency_score"] = 30
            else:
                quality["recency_score"] = 10
        except:
            quality["recency_score"] = 30
    
    # Completeness score (has abstract, authors, venue, etc.)
    completeness = 0
    if paper.get("abstract"):
        completeness += 30
    if paper.get("authors"):
        completeness += 20
    if paper.get("venue") or paper.get("journal_name"):
        completeness += 20
    if paper.get("doi") or paper.get("url"):
        completeness += 15
    if paper.get("pdf_url"):
        completeness += 15
    quality["completeness_score"] = completeness
    
    # Influential citations score
    influential = paper.get("influential_citations", 0)
    if influential > 50:
        quality["influential_score"] = 100
    elif influential > 20:
        quality["influential_score"] = 80
    elif influential > 10:
        quality["influential_score"] = 60
    elif influential > 5:
        quality["influential_score"] = 40
    elif influential > 0:
        quality["influential_score"] = 20
    
    return quality


def enhanced_rank_papers(papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:

    for paper in papers:

        relevance = calculate_topic_relevance(paper, query)
        paper["relevance_score"] = relevance
        
        # Calculate quality metrics
        quality = calculate_quality_score(paper)
        paper["quality_metrics"] = quality

        # Weights are read from config.SCORING_WEIGHTS (single source of truth)
        enhanced_score = (
            quality["citation_score"] * SCORING_WEIGHTS["citation_count"] +
            relevance * SCORING_WEIGHTS["relevance"] +
            quality["venue_score"] * SCORING_WEIGHTS["venue_quality"] +
            quality["recency_score"] * SCORING_WEIGHTS["recency"] +
            quality["influential_score"] * SCORING_WEIGHTS["influential_citations"]
        )

        paper["total_score"] = round(enhanced_score, 2)
    
    # Sort by total score
    ranked = sorted(papers, key=lambda p: p.get("total_score", 0), reverse=True)
    
    # Assign reading priority
    for i, paper in enumerate(ranked):
        paper["reading_priority"] = i + 1
    
    return ranked


def analysis_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:

    papers = state.get("papers", [])
    query = state.get("query", "")
    
    if not papers:
        return {
            "ranked_papers": [],
            "analysis_completed": True,
            "errors": state.get("errors", []) + ["No papers to analyze"]
        }
    

    print("ANALYSIS AGENT - Enhanced Quality Assessment")

    print(f"Analyzing {len(papers)} papers for query: '{query}'...\n")
    
    # Enhanced ranking with topic relevance
    ranked = enhanced_rank_papers(papers, query)
    
    # Print top 10 with detailed metrics
    print("Top 10 Papers by Quality Score:")

    
    for i, paper in enumerate(ranked[:10], 1):
        title = paper.get("title", "No Title")
        if len(title) > 55:
            title = title[:52] + "..."
        
        score = paper.get("total_score", 0)
        citations = paper.get("citation_count", 0)
        relevance = paper.get("relevance_score", 0)
        quartile = paper.get("q_quartile", "-")
        venue = paper.get("venue", "") or paper.get("journal_name", "") or ""
        if venue and len(venue) > 30:
            venue = venue[:27] + "..."
        
        url = paper.get("url", "")
        
        print(f"\n  {i}. {title}")
        print(f"      Score: {score:.1f} | Relevance: {relevance:.0f}% | Citations: {citations}")
        if venue:
            print(f"      Venue: {venue} | Q: {quartile}")
        if url:
            print(f"     🔗 {url}")
    
    # Summary statistics
    total_citations = sum(p.get("citation_count", 0) for p in ranked)
    avg_citations = total_citations / len(ranked) if ranked else 0
    high_quality = sum(1 for p in ranked if p.get("total_score", 0) >= 50)
    

    print(f" Summary Statistics:")
    print(f"   - Total papers: {len(ranked)}")
    print(f"   - Total citations: {total_citations:,}")
    print(f"   - Average citations: {avg_citations:.1f}")
    print(f"   - High quality papers (score ≥50): {high_quality}")
 
    
    return {
        "ranked_papers": ranked,
        "analysis_completed": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Analysis completed. Found {high_quality} high-quality papers. Top: {ranked[0].get('title', 'Unknown')[:50] if ranked else 'None'}"}
        ]
    }
