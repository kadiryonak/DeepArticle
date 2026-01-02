"""
CrossRef API tools for DOI-based metadata retrieval.
"""

from typing import Dict, Any
import requests
from langchain_core.tools import tool


CROSSREF_API = "https://api.crossref.org/works"


@tool
def get_paper_metadata_by_doi(doi: str) -> Dict[str, Any]:
    """
    Get paper metadata from CrossRef using DOI.
    
    Args:
        doi: The DOI of the paper
        
    Returns:
        Dictionary with paper metadata
    """
    if not doi:
        return {"error": "DOI is required"}
    
    try:
        url = f"{CROSSREF_API}/{doi}"
        headers = {
            "User-Agent": "AcademicPaperAnalyzer/1.0 (mailto:example@example.com)"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        message = data.get("message", {})
        
        # Extract authors
        authors = []
        for author in message.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)
        
        # Extract publication date
        date_parts = message.get("published-print", {}).get("date-parts", [[]])
        if not date_parts or not date_parts[0]:
            date_parts = message.get("published-online", {}).get("date-parts", [[]])
        
        date_str = None
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            if len(parts) >= 1:
                year = parts[0]
                month = parts[1] if len(parts) >= 2 else 1
                day = parts[2] if len(parts) >= 3 else 1
                date_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
        
        # Get title
        titles = message.get("title", [])
        title = titles[0] if titles else ""
        
        # Get container (journal) title
        container_titles = message.get("container-title", [])
        journal_name = container_titles[0] if container_titles else ""
        
        # Get ISSN for potential journal lookup
        issns = message.get("ISSN", [])
        
        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "published_date": date_str,
            "journal_name": journal_name,
            "issn": issns[0] if issns else None,
            "publisher": message.get("publisher", ""),
            "type": message.get("type", ""),
            "references_count": message.get("references-count", 0),
            "is_referenced_by_count": message.get("is-referenced-by-count", 0),
            "url": message.get("URL", ""),
            "found": True
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"doi": doi, "found": False, "error": "DOI not found"}
        return {"error": f"CrossRef lookup failed: {str(e)}"}
    except Exception as e:
        return {"error": f"CrossRef lookup failed: {str(e)}"}


def get_crossref_tools():
    """Return list of CrossRef tools."""
    return [get_paper_metadata_by_doi]
