"""
Metadata Agent - Enriches papers with journal metrics and additional metadata.
"""

from typing import Dict, Any, List

import sys
sys.path.insert(0, '..')

from tools.scimago_tools import get_journal_quartile
from tools.crossref_tools import get_paper_metadata_by_doi


def enrich_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:

    enriched = paper.copy()
    
    # Try to get journal metrics if we have a journal name
    journal_name = paper.get("journal_name") or paper.get("venue")
    
    if journal_name:
        try:
            metrics = get_journal_quartile.invoke({"journal_name": journal_name})
            if metrics and metrics.get("found"):
                enriched["q_quartile"] = metrics.get("q_quartile")
                if metrics.get("sjr"):
                    # Use SJR as a proxy for impact factor if not available
                    if not enriched.get("impact_factor"):
                        enriched["impact_factor"] = metrics.get("sjr")
        except Exception:
            pass
    
    # Try to get additional metadata from CrossRef if we have a DOI
    doi = paper.get("doi")
    
    if doi:
        try:
            crossref_data = get_paper_metadata_by_doi.invoke({"doi": doi})
            if crossref_data and crossref_data.get("found"):
                # Update citation count if CrossRef has more
                crossref_citations = crossref_data.get("is_referenced_by_count", 0)
                if crossref_citations > enriched.get("citation_count", 0):
                    enriched["citation_count"] = crossref_citations
                
                # Fill in missing journal name
                if not enriched.get("journal_name") and crossref_data.get("journal_name"):
                    enriched["journal_name"] = crossref_data["journal_name"]
        except Exception:
            pass
    
    return enriched


def metadata_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    papers = state.get("papers", [])
    
    if not papers:
        return {
            "papers": [],
            "metadata_enriched": True,
            "errors": state.get("errors", []) + ["No papers to enrich"]
        }
    
    print("\n" + "=" * 50)
    print("📊 METADATA AGENT")
    print("=" * 50)
    print(f"Enriching {len(papers)} papers...\n")
    
    enriched_papers = []
    
    for i, paper in enumerate(papers):
        title = paper.get("title", "")[:50]
        print(f"  [{i+1}/{len(papers)}] Processing: {title}...")
        
        try:
            enriched = enrich_paper_metadata(paper)
            enriched_papers.append(enriched)
            
            # Show what was found
            additions = []
            if enriched.get("q_quartile"):
                additions.append(f"Q: {enriched['q_quartile']}")
            if enriched.get("impact_factor"):
                additions.append(f"IF: {enriched['impact_factor']:.2f}")
            if enriched.get("citation_count"):
                additions.append(f"Cites: {enriched['citation_count']}")
            
            if additions:
                print(f"           ✓ {', '.join(additions)}")
            else:
                print(f"           - No additional metrics found")
                
        except Exception as e:
            print(f"           ✗ Error: {e}")
            enriched_papers.append(paper)
    
    print(f"\n✓ Metadata enrichment complete")
    
    return {
        "papers": enriched_papers,
        "metadata_enriched": True,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Metadata enrichment completed for {len(enriched_papers)} papers."}
        ]
    }
