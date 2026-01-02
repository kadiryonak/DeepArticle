"""
Integration Tests for the Multi-Agent Academic Paper Analysis System.
Uses FakeListLLM for deterministic testing without API calls.
"""

import pytest
import sys
sys.path.insert(0, '..')

from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import AIMessage
from unittest.mock import patch, MagicMock


class FakeChatModel:
    """
    Fake Chat Model that mimics LangChain chat models.
    Returns predefined responses for testing.
    """
    
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
    
    def invoke(self, prompt):
        """Return the next response from the list."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
        else:
            response = self.responses[-1] if self.responses else ""
        
        # Return an object with .content attribute like real chat models
        return MagicMock(content=response)


class TestQueryAnalyzerIntegration:
    """Integration tests for the Query Analyzer Agent."""
    
    def test_query_analyzer_with_fake_llm(self):
        """Test full query analysis flow with fake LLM."""
        from agents.query_analyzer import analyze_topic_with_llm, generate_search_queries
        
        # Create fake LLM with expected response format
        fake_response = """CONCEPTS: LLM, unit testing, automated testing, code generation, software engineering
SYSTEMS: EvoSuite, Randoop, Pynguin, ChatUniTest, CodaMOSA, HITS
CATEGORIES: Technique-focused papers, Empirical studies, Tool papers, Benchmark papers
QUERIES:
- LLM-based unit test generation for Java
- ChatUniTest effectiveness evaluation
- CodaMOSA vs EvoSuite comparison
- Large language models for automated testing
- Neural code generation for test cases"""
        
        fake_llm = FakeChatModel([fake_response])
        
        # Run analysis
        analysis = analyze_topic_with_llm("unit test generation using LLM", fake_llm)
        
        # Verify concepts extracted
        assert "LLM" in analysis["concepts"]
        assert "unit testing" in analysis["concepts"]
        
        # Verify systems discovered
        assert "EvoSuite" in analysis["systems"]
        assert "CodaMOSA" in analysis["systems"]
        
        # Verify queries generated
        assert len(analysis["queries"]) >= 5
    
    def test_query_analyzer_handles_malformed_response(self):
        """Test that analyzer handles malformed LLM responses gracefully."""
        from agents.query_analyzer import analyze_topic_with_llm
        
        fake_response = "This is not in the expected format at all."
        fake_llm = FakeChatModel([fake_response])
        
        analysis = analyze_topic_with_llm("test query", fake_llm)
        
        # Should return empty lists instead of crashing
        assert analysis["concepts"] == []
        assert analysis["systems"] == []
        assert analysis["queries"] == []


class TestSummarizerIntegration:
    """Integration tests for the Summarizer Agent."""
    
    def test_summarizer_with_fake_llm(self):
        """Test summarization with fake LLM."""
        from agents.summarizer_agent import generate_summary
        
        fake_response = "This paper presents a novel approach to unit test generation using LLMs, achieving 80% code coverage."
        fake_llm = FakeChatModel([fake_response])
        
        paper = {
            "title": "LLM-based Unit Test Generation",
            "abstract": "We propose a new method for generating unit tests using large language models..."
        }
        
        summary = generate_summary(fake_llm, paper)
        
        assert "novel approach" in summary
        assert "80%" in summary
    
    def test_summarizer_handles_empty_abstract(self):
        """Test summarizer handles papers without abstracts."""
        from agents.summarizer_agent import generate_summary
        
        fake_llm = FakeChatModel(["Some summary"])
        
        paper = {
            "title": "Test Paper",
            "abstract": ""
        }
        
        summary = generate_summary(fake_llm, paper)
        
        assert "No abstract available" in summary


class TestSearchAgentIntegration:
    """Integration tests for the Search Agent."""
    
    def test_merge_papers_from_multiple_sources(self):
        """Test merging papers from different sources."""
        from agents.search_agent import merge_papers
        
        search_results = {
            "arxiv": [
                {
                    "title": "ChatUniTest: Unit Test Generation Using Large Language Model",
                    "citation_count": 45,
                    "source": "arxiv"
                },
                {
                    "title": "Another Paper",
                    "citation_count": 10,
                    "source": "arxiv"
                }
            ],
            "semantic_scholar": [
                {
                    "title": "ChatUniTest: Unit Test Generation Using Large Language Model",
                    "citation_count": 50,  # Higher citation from S2
                    "source": "semantic_scholar"
                },
                {
                    "title": "CodaMOSA: Combining Test Generation with Search",
                    "citation_count": 30,
                    "source": "semantic_scholar"
                }
            ]
        }
        
        merged = merge_papers(search_results)
        
        # Should have 3 unique papers
        assert len(merged) == 3
        
        # ChatUniTest should keep higher citation count
        chatunitest = next(p for p in merged if "ChatUniTest" in p["title"])
        assert chatunitest["citation_count"] == 50


class TestWorkflowIntegration:
    """End-to-end workflow tests."""
    
    @patch('agents.query_analyzer.create_llm')
    def test_query_analyzer_node(self, mock_create_llm):
        """Test query analyzer node with mocked LLM."""
        from agents.query_analyzer import query_analyzer_node
        
        # Mock LLM response
        fake_response = """CONCEPTS: LLM, testing
SYSTEMS: EvoSuite
CATEGORIES: tool papers
QUERIES:
- LLM testing"""
        
        mock_create_llm.return_value = FakeChatModel([fake_response])
        
        state = {
            "query": "unit test generation",
            "search_queries": [],
            "topic_analysis": None,
            "messages": []
        }
        
        result = query_analyzer_node(state)
        
        # Should have generated queries
        assert len(result["search_queries"]) > 0
        
        # Should have topic analysis
        assert result["topic_analysis"] is not None


class TestPaperSelectionCriteria:
    """
    Tests documenting why papers are selected and how scoring works.
    
    PAPER SELECTION RATIONALE:
    
    1. Citation Count (25% weight)
       - Papers with more citations are considered more influential
       - 100+ citations: High score
       - 50-100 citations: Medium score
       - <50 citations: Low score
    
    2. Relevance Score (25% weight)
       - Based on keyword matching in title/abstract
       - Higher scores for papers containing search terms
       - Domain field matching (CS, SE preferred)
    
    3. Venue Quality (20% weight)
       - Top venues: ICSE, FSE, ASE, ISSTA (high score)
       - Good journals: TSE, TOSEM (high score)
       - Unknown venues get lower scores
    
    4. Recency (15% weight)
       - 2024-2025: Highest score
       - 2022-2023: Medium score
       - Older papers: Lower score
    
    5. Influential Citations (15% weight)
       - Papers cited by other influential papers
       - From Semantic Scholar's influence metric
    """
    
    def test_high_citation_paper_ranks_higher(self):
        """High citation papers should rank higher."""
        from agents.analysis_agent import calculate_quality_score
        
        high_cited = {"citation_count": 200, "published_date": "2023"}
        low_cited = {"citation_count": 5, "published_date": "2023"}
        
        high_metrics = calculate_quality_score(high_cited)
        low_metrics = calculate_quality_score(low_cited)
        
        assert high_metrics["citation_score"] > low_metrics["citation_score"]
    
    def test_recent_paper_scores_higher(self):
        """Recent papers should get higher recency score."""
        from agents.analysis_agent import calculate_quality_score
        
        recent = {"published_date": "2024", "citation_count": 10}
        old = {"published_date": "2018", "citation_count": 10}
        
        recent_metrics = calculate_quality_score(recent)
        old_metrics = calculate_quality_score(old)
        
        assert recent_metrics["recency_score"] > old_metrics["recency_score"]
    
    def test_relevant_paper_scores_higher(self):
        """Papers matching query terms should score higher."""
        from agents.analysis_agent import calculate_topic_relevance
        
        relevant = {
            "title": "LLM-based Unit Test Generation",
            "abstract": "Using large language models for automated unit testing"
        }
        
        irrelevant = {
            "title": "Database Performance Optimization",
            "abstract": "Improving query execution in distributed systems"
        }
        
        query = "LLM unit test generation"
        
        rel_score = calculate_topic_relevance(relevant, query)
        irr_score = calculate_topic_relevance(irrelevant, query)
        
        assert rel_score > irr_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
