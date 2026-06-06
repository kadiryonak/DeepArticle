"""
SCImago tools for journal metrics (Q Quartile, SJR).
Uses web scraping as SCImago doesn't provide an official API.
"""

from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# Cache for journal metrics to avoid repeated requests
JOURNAL_CACHE: Dict[str, Dict[str, Any]] = {}


@tool
def get_journal_quartile(journal_name: str) -> Dict[str, Any]:
    """
    Get the Q quartile and SJR metrics for a journal from SCImago.
    
    Args:
        journal_name: Name of the journal to look up
        
    Returns:
        Dictionary with journal metrics including Q quartile and SJR
    """
    if not journal_name:
        return {"error": "Journal name is required"}
    
    # Check cache first
    cache_key = journal_name.lower().strip()
    if cache_key in JOURNAL_CACHE:
        return JOURNAL_CACHE[cache_key]
    
    try:
        # Search SCImago for the journal
        search_url = "https://www.scimagojr.com/journalsearch.php"
        params = {"q": journal_name}
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(search_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find journal links in search results
        journal_links = soup.select("a.jrnlname")
        
        if not journal_links:
            result = {
                "journal_name": journal_name,
                "q_quartile": None,
                "sjr": None,
                "h_index": None,
                "found": False
            }
            JOURNAL_CACHE[cache_key] = result
            return result
        
        # Get the first matching journal's page
        journal_url = "https://www.scimagojr.com/" + journal_links[0].get("href", "")
        journal_response = requests.get(journal_url, headers=headers, timeout=30)
        journal_response.raise_for_status()
        
        journal_soup = BeautifulSoup(journal_response.content, "html.parser")
        
        # Extract Q quartile
        q_quartile = None
        quartile_cell = journal_soup.select_one(".cell1")
        if quartile_cell:
            q_text = quartile_cell.get_text(strip=True)
            if "Q1" in q_text:
                q_quartile = "Q1"
            elif "Q2" in q_text:
                q_quartile = "Q2"
            elif "Q3" in q_text:
                q_quartile = "Q3"
            elif "Q4" in q_text:
                q_quartile = "Q4"
        
        # Extract SJR
        sjr = None
        sjr_cell = journal_soup.select_one(".sjrbig")
        if sjr_cell:
            try:
                sjr = float(sjr_cell.get_text(strip=True))
            except ValueError:
                pass
        
        # Extract H-Index
        h_index = None
        h_index_div = journal_soup.find("div", text=lambda t: t and "H Index" in t if t else False)
        if h_index_div:
            h_value = h_index_div.find_next_sibling("div")
            if h_value:
                try:
                    h_index = int(h_value.get_text(strip=True))
                except ValueError:
                    pass
        
        result = {
            "journal_name": journal_name,
            "q_quartile": q_quartile,
            "sjr": sjr,
            "h_index": h_index,
            "found": True
        }
        
        JOURNAL_CACHE[cache_key] = result
        return result
        
    except requests.exceptions.RequestException as e:
        return {"error": f"SCImago lookup failed: {str(e)}"}
    except Exception as e:
        return {"error": f"SCImago lookup failed: {str(e)}"}


def get_scimago_tools():
    """Return list of SCImago tools."""
    return [get_journal_quartile]
