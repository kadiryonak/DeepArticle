"""
Unit Tests for LLM Factory
Uses FakeListLLM for testing without actual API calls.
"""

import pytest
import sys
sys.path.insert(0, '..')

from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import AIMessage


class TestLLMFactory:
    """Tests for the LLM factory module."""
    
    def test_create_fake_llm(self):
        """Test creating a FakeListLLM for testing purposes."""
        responses = [
            "CONCEPTS: unit testing, LLM, code generation\nSYSTEMS: EvoSuite, Randoop\nCATEGORIES: tool papers\nQUERIES:\n- LLM test generation"
        ]
        fake_llm = FakeListLLM(responses=responses)
        
        result = fake_llm.invoke("test prompt")
        assert "CONCEPTS" in result
        assert "EvoSuite" in result
    
    def test_fake_llm_multiple_responses(self):
        """Test FakeListLLM with multiple sequential responses."""
        responses = [
            "First response: CONCEPTS: testing",
            "Second response: SYSTEMS: Pynguin",
            "Third response: Summary of the paper"
        ]
        fake_llm = FakeListLLM(responses=responses)
        
        # LLM returns responses in order
        assert "First" in fake_llm.invoke("prompt1")
        assert "Second" in fake_llm.invoke("prompt2")
        assert "Third" in fake_llm.invoke("prompt3")


class TestQueryAnalyzer:
    """Tests for query analysis functionality."""
    
    def test_generate_search_queries_with_analysis(self):
        """Test query generation from LLM analysis."""
        from agents.query_analyzer import generate_search_queries
        
        analysis = {
            "concepts": ["LLM", "unit testing", "code generation"],
            "systems": ["EvoSuite", "ChatUniTest", "CodaMOSA"],
            "categories": ["tool papers", "empirical studies"],
            "queries": [
                "LLM-based unit test generation",
                "ChatUniTest evaluation",
                "CodaMOSA test coverage"
            ]
        }
        
        queries = generate_search_queries("unit test generation", analysis)
        
        # Original topic should be first
        assert queries[0] == "unit test generation"
        
        # Should include LLM-generated queries
        assert any("LLM-based" in q for q in queries)
        
        # Should include system-specific queries
        assert any("EvoSuite" in q for q in queries)
        
        # Should have <= 20 queries
        assert len(queries) <= 20
    
    def test_generate_search_queries_without_analysis(self):
        """Test query generation when LLM is unavailable."""
        from agents.query_analyzer import generate_search_queries
        
        queries = generate_search_queries("unit test generation", None)
        
        # Should still include original topic
        assert queries[0] == "unit test generation"
        
        # Should include empirical study and survey variations
        assert any("empirical" in q.lower() for q in queries)
        assert any("survey" in q.lower() for q in queries)


class TestAnalysisAgent:
    """Tests for paper analysis and scoring."""
    
    def test_calculate_topic_relevance(self):
        """Test topic relevance scoring."""
        from agents.analysis_agent import calculate_topic_relevance
        
        paper = {
            "title": "LLM-based Unit Test Generation for Java",
            "abstract": "We present a novel approach using large language models for automated unit test generation.",
            "fields_of_study": ["Computer Science", "Software Engineering"]
        }
        
        query = "LLM unit test generation"
        score = calculate_topic_relevance(paper, query)
        
        # High relevance expected
        assert score >= 80
    
    def test_calculate_topic_relevance_low(self):
        """Test low relevance scoring for unrelated paper."""
        from agents.analysis_agent import calculate_topic_relevance
        
        paper = {
            "title": "Climate Change in Arctic Regions",
            "abstract": "A study on temperature changes in polar ice caps.",
            "fields_of_study": ["Environmental Science"]
        }
        
        query = "LLM unit test generation"
        score = calculate_topic_relevance(paper, query)
        
        # Low relevance expected
        assert score < 50
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        from agents.analysis_agent import calculate_quality_score
        
        paper = {
            "citation_count": 100,
            "venue": "ICSE 2024",
            "venue_type": "conference",
            "q_quartile": "Q1",
            "published_date": "2024",
            "abstract": "Full abstract here",
            "authors": ["Author A", "Author B"],
            "doi": "10.1234/test",
            "pdf_url": "https://example.com/paper.pdf",
            "influential_citations": 10
        }
        
        metrics = calculate_quality_score(paper)
        
        # Citation score should be high for 100 citations
        assert metrics["citation_score"] >= 60
        
        # Venue score should be present
        assert metrics["venue_score"] >= 0
        
        # Completeness should be high
        assert metrics["completeness_score"] >= 80


class TestFormatters:
    """Tests for output formatting functions."""
    
    def test_format_reading_list(self):
        """Test reading list formatting."""
        from utils.formatters import format_reading_list
        
        papers = [
            {
                "title": "Test Paper 1",
                "authors": ["Author A", "Author B"],
                "published_date": "2024",
                "venue": "ICSE",
                "citation_count": 50,
                "relevance_score": 95,
                "total_score": 75.5,
                "summary": "This is a test summary.",
                "url": "https://example.com/paper1",
                "pdf_url": "https://example.com/paper1.pdf",
                "doi": "10.1234/test1",
                "q_quartile": "Q1"
            }
        ]
        
        output = format_reading_list(papers)
        
        # Should contain paper title
        assert "Test Paper 1" in output
        
        # Should contain author
        assert "Author A" in output
        
        # Should contain metrics
        assert "75.5" in output or "75" in output


class TestMergePapers:
    """Tests for paper deduplication."""
    
    def test_merge_removes_duplicates(self):
        """Test that merge removes duplicate papers."""
        from agents.search_agent import merge_papers
        
        search_results = {
            "arxiv": [
                {"title": "Test Paper", "citation_count": 10},
                {"title": "Another Paper", "citation_count": 5}
            ],
            "semantic_scholar": [
                {"title": "Test Paper", "citation_count": 15},  # Same but more citations
                {"title": "Different Paper", "citation_count": 20}
            ]
        }
        
        merged = merge_papers(search_results)
        
        # Should have 3 unique papers
        assert len(merged) == 3
        
        # "Test Paper" should have the higher citation count
        test_paper = next(p for p in merged if "Test" in p["title"])
        assert test_paper["citation_count"] == 15
    
    def test_merge_sorts_by_citations(self):
        """Test that merged papers are sorted by citation count."""
        from agents.search_agent import merge_papers
        
        search_results = {
            "source1": [
                {"title": "Low Cited", "citation_count": 5},
                {"title": "High Cited", "citation_count": 100},
                {"title": "Medium Cited", "citation_count": 50}
            ]
        }
        
        merged = merge_papers(search_results)
        
        # First paper should have highest citations
        assert merged[0]["citation_count"] == 100
        assert merged[1]["citation_count"] == 50
        assert merged[2]["citation_count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
