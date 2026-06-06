"""
Scoring algorithm for academic paper ranking.
"""

from typing import Dict, Any, List
from datetime import datetime
import sys
sys.path.append('..')
from config import SCORING_WEIGHTS, Q_QUARTILE_SCORES


def calculate_citation_score(citation_count: int, max_citations: int = 1000) -> float:
    """
    Calculate normalized citation score (0-100).
    Uses log scale to handle high citation papers.
    """
    if citation_count <= 0:
        return 0.0
    
    import math
    # Log scale normalization
    score = (math.log10(citation_count + 1) / math.log10(max_citations + 1)) * 100
    return min(score, 100.0)


def calculate_recency_score(published_date: str, max_years: int = 10) -> float:
    """
    Calculate recency score (0-100).
    More recent papers get higher scores.
    """
    if not published_date:
        return 50.0  # Default score for unknown dates
    
    try:
        # Parse year from various date formats
        year = None
        if len(published_date) >= 4:
            year = int(published_date[:4])
        
        if year:
            current_year = datetime.now().year
            age = current_year - year
            
            if age <= 0:
                return 100.0
            elif age >= max_years:
                return 10.0
            else:
                # Linear decay
                score = 100.0 - (age / max_years * 90.0)
                return max(score, 10.0)
    except (ValueError, TypeError):
        pass
    
    return 50.0


def calculate_quartile_score(q_quartile: str) -> float:
    """Calculate score based on journal Q quartile."""
    return Q_QUARTILE_SCORES.get(q_quartile, 0)


def calculate_impact_factor_score(impact_factor: float, max_if: float = 50.0) -> float:
    """
    Calculate normalized impact factor score (0-100).
    """
    if not impact_factor or impact_factor <= 0:
        return 0.0
    
    score = (impact_factor / max_if) * 100
    return min(score, 100.0)


def calculate_venue_score(paper: Dict[str, Any]) -> float:
    """
    Calculate a venue-quality score (0-100) from the journal Q quartile,
    falling back to the venue type (journal/conference) when no quartile
    is available.
    """
    quartile_score = calculate_quartile_score(paper.get("q_quartile"))
    if quartile_score > 0:
        return quartile_score

    venue_type = (paper.get("venue_type") or "").lower()
    if venue_type == "journal":
        return 60.0
    if venue_type == "conference":
        return 50.0
    if paper.get("venue") or paper.get("journal_name"):
        return 30.0
    return 0.0


def calculate_influential_score(influential_citations: int) -> float:
    """Calculate a score (0-100) from the number of influential citations."""
    if influential_citations > 50:
        return 100.0
    if influential_citations > 20:
        return 80.0
    if influential_citations > 10:
        return 60.0
    if influential_citations > 5:
        return 40.0
    if influential_citations > 0:
        return 20.0
    return 0.0


def calculate_total_score(paper: Dict[str, Any]) -> float:
    """
    Calculate the total weighted score for a paper.

    Uses the weights defined in ``config.SCORING_WEIGHTS`` as the single
    source of truth:
    ``citation_count``, ``relevance``, ``venue_quality``, ``recency`` and
    ``influential_citations``.

    Args:
        paper: Dictionary containing paper metadata

    Returns:
        Total weighted score (0-100)
    """
    weights = SCORING_WEIGHTS

    # Calculate individual component scores
    citation_score = calculate_citation_score(paper.get("citation_count", 0))
    venue_score = calculate_venue_score(paper)
    recency_score = calculate_recency_score(paper.get("published_date", ""))
    relevance_score = paper.get("relevance_score", 50.0)  # Default 50 if not set
    influential_score = calculate_influential_score(paper.get("influential_citations", 0))

    # Calculate weighted total
    total = (
        citation_score * weights["citation_count"] +
        relevance_score * weights["relevance"] +
        venue_score * weights["venue_quality"] +
        recency_score * weights["recency"] +
        influential_score * weights["influential_citations"]
    )

    return round(total, 2)


def rank_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank papers by their total scores.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        List of papers sorted by total score (descending)
    """
    # Calculate scores for all papers
    for paper in papers:
        paper["total_score"] = calculate_total_score(paper)
    
    # Sort by total score
    ranked = sorted(papers, key=lambda p: p.get("total_score", 0), reverse=True)
    
    # Assign reading priority
    for i, paper in enumerate(ranked):
        paper["reading_priority"] = i + 1
    
    return ranked


def get_top_papers(papers: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """
    Get the top N papers by score.
    
    Args:
        papers: List of paper dictionaries
        n: Number of top papers to return
        
    Returns:
        List of top N papers
    """
    ranked = rank_papers(papers)
    return ranked[:n]
