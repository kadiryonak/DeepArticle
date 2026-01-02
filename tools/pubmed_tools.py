"""
PubMed search tools for biomedical paper retrieval.
"""

from typing import List, Dict, Any
import requests
from langchain_core.tools import tool

try:
    import xmltodict
    XMLTODICT_AVAILABLE = True
except ImportError:
    XMLTODICT_AVAILABLE = False


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@tool
def search_pubmed(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search PubMed for biomedical papers matching the query.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata
    """
    if not XMLTODICT_AVAILABLE:
        return [{"error": "xmltodict package not installed. Run: pip install xmltodict"}]
    
    papers = []
    
    try:
        # First, search for paper IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        search_response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=30)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return []
        
        # Fetch details for each paper
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        fetch_response = requests.get(PUBMED_FETCH_URL, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        
        data = xmltodict.parse(fetch_response.content)
        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        
        # Handle single article case
        if isinstance(articles, dict):
            articles = [articles]
        
        for article in articles:
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            
            # Extract PMID
            pmid = medline.get("PMID", {})
            if isinstance(pmid, dict):
                pmid = pmid.get("#text", "")
            
            # Extract title
            title = article_data.get("ArticleTitle", "")
            if isinstance(title, dict):
                title = title.get("#text", "")
            
            # Extract authors
            author_list = article_data.get("AuthorList", {}).get("Author", [])
            if isinstance(author_list, dict):
                author_list = [author_list]
            
            authors = []
            for author in author_list:
                if isinstance(author, dict):
                    last_name = author.get("LastName", "")
                    fore_name = author.get("ForeName", "")
                    if last_name or fore_name:
                        authors.append(f"{fore_name} {last_name}".strip())
            
            # Extract abstract
            abstract_text = article_data.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_text, list):
                abstract_text = " ".join([
                    a.get("#text", "") if isinstance(a, dict) else str(a)
                    for a in abstract_text
                ])
            elif isinstance(abstract_text, dict):
                abstract_text = abstract_text.get("#text", "")
            
            # Extract journal
            journal = article_data.get("Journal", {})
            journal_title = journal.get("Title", "")
            
            # Extract publication date
            pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", "")
            month = pub_date.get("Month", "")
            day = pub_date.get("Day", "")
            date_str = f"{year}-{month}-{day}" if year else None
            
            paper = {
                "paper_id": str(pmid),
                "title": title,
                "authors": authors,
                "abstract": abstract_text,
                "published_date": date_str,
                "source": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "doi": None,
                "journal_name": journal_title,
                "venue": None,
                "citation_count": 0,
                "impact_factor": None,
                "q_quartile": None,
                "relevance_score": 0.0,
                "total_score": 0.0,
                "summary": None,
                "reading_priority": 0
            }
            papers.append(paper)
            
    except requests.exceptions.RequestException as e:
        return [{"error": f"PubMed search failed: {str(e)}"}]
    except Exception as e:
        return [{"error": f"PubMed search failed: {str(e)}"}]
    
    return papers


def get_pubmed_tools():
    """Return list of PubMed tools."""
    return [search_pubmed]
