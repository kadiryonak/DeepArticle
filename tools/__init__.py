"""Tools module for the Multi-Agent Academic Paper Analysis System."""

from .arxiv_tools import search_arxiv, get_arxiv_tools
from .semantic_scholar_tools import search_semantic_scholar, get_paper_citations, get_semantic_scholar_tools
from .pubmed_tools import search_pubmed, get_pubmed_tools
from .google_scholar_tools import search_google_scholar, get_google_scholar_tools
from .scimago_tools import get_journal_quartile, get_scimago_tools
from .crossref_tools import get_paper_metadata_by_doi, get_crossref_tools

__all__ = [
    # ArXiv
    "search_arxiv",
    "get_arxiv_tools",
    # Semantic Scholar
    "search_semantic_scholar",
    "get_paper_citations",
    "get_semantic_scholar_tools",
    # PubMed
    "search_pubmed",
    "get_pubmed_tools",
    # Google Scholar
    "search_google_scholar",
    "get_google_scholar_tools",
    # SCImago
    "get_journal_quartile",
    "get_scimago_tools",
    # CrossRef
    "get_paper_metadata_by_doi",
    "get_crossref_tools"
]
