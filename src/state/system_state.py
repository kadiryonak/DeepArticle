"""
State definitions for the Multi-Agent Academic Paper Analysis System.
"""

from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PaperMetadata(BaseModel):
    """Represents a single academic paper with all its metadata."""
    
    # Basic Info
    paper_id: str = Field(default="")
    title: str = Field(default="")
    authors: List[str] = Field(default_factory=list)
    abstract: str = Field(default="")
    published_date: Optional[str] = None
    
    # Source Info
    source: str = Field(default="")  # arxiv, semantic_scholar, pubmed, google_scholar
    url: Optional[str] = None
    doi: Optional[str] = None
    
    # Journal/Venue Info
    journal_name: Optional[str] = None
    venue: Optional[str] = None
    
    # Metrics
    citation_count: int = Field(default=0)
    impact_factor: Optional[float] = None
    q_quartile: Optional[str] = None  # Q1, Q2, Q3, Q4
    
    # Computed Fields
    relevance_score: float = Field(default=0.0)
    total_score: float = Field(default=0.0)
    summary: Optional[str] = None
    reading_priority: int = Field(default=0)
    
    def to_dict(self) -> dict:
        return self.model_dump()


class SystemState(TypedDict):
    """
    The shared state that flows through all agents in the system.
    Uses TypedDict for LangGraph compatibility.
    """
    
    # User Query
    query: str
    
    # Query Analysis (from Query Analyzer Agent)
    search_queries: List[str]  # Expanded search queries
    topic_analysis: Optional[dict]  # LLM topic analysis result
    
    # Paper Collections. Each stage (search -> metadata) returns the COMPLETE
    # current list, so this channel uses replace semantics (last write wins).
    # (Previously used operator.add, which concatenated the enriched list onto
    # the search list and silently doubled every paper.)
    papers: List[dict]
    
    # Processing Status
    search_completed: bool
    metadata_enriched: bool
    analysis_completed: bool
    summarization_completed: bool
    prioritization_completed: bool
    recommendation_completed: bool
    resources_completed: bool

    # Final Results
    ranked_papers: List[dict]
    reading_order: List[dict]

    # Study plan (from Recommender Agent)
    groups: dict            # category -> list of paper_id
    reading_path: List[dict]  # ordered staged plan
    start_here: Optional[dict]

    # Supplementary resources (from Resources Agent): github / articles / videos
    resources: dict
    
    # Messages for agent communication
    messages: List[dict]
    
    # Error tracking
    errors: List[str]


def create_initial_state(query: str) -> SystemState:
    """Create the initial state for a new query."""
    return SystemState(
        query=query,
        papers=[],
        search_completed=False,
        metadata_enriched=False,
        analysis_completed=False,
        summarization_completed=False,
        prioritization_completed=False,
        recommendation_completed=False,
        resources_completed=False,
        ranked_papers=[],
        reading_order=[],
        groups={},
        reading_path=[],
        start_here=None,
        resources={},
        messages=[],
        errors=[]
    )
