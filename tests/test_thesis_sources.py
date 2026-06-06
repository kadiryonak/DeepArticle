"""
Offline tests for thesis / multilingual sources (OpenAlex theses, CORE, YÖK Tez)
and bilingual query parsing. No network access.
"""

import sys
sys.path.insert(0, '..')

from tools.openalex_tools import _parse_work
from tools.core_tools import _parse_core_work
from tools.yoktez_tools import _parse_results
from agents.query_analyzer import analyze_topic_with_llm


class TestOpenAlexThesisParsing:
    def test_dissertation_marked_as_thesis(self):
        work = {
            "id": "https://openalex.org/W1",
            "title": "Makine Öğrenmesi ile Nüfus Tahmini",
            "type": "dissertation",
            "language": "tr",
            "publication_year": 2019,
            "cited_by_count": 3,
        }
        paper = _parse_work(work)
        assert paper["is_thesis"] is True
        assert paper["source"] == "openalex_thesis"
        assert paper["language"] == "tr"
        assert paper["publication_type"] == "dissertation"

    def test_article_not_thesis(self):
        paper = _parse_work({"id": "x", "title": "A paper", "type": "article"})
        assert paper["is_thesis"] is False
        assert paper["source"] == "openalex"


class TestCoreParsing:
    def test_thesis_document_type(self):
        work = {
            "id": 42,
            "title": "Deep Learning Thesis",
            "documentType": "thesis",
            "authors": [{"name": "Jane Doe"}],
            "yearPublished": 2021,
            "language": {"code": "en"},
            "downloadUrl": "http://x/pdf",
        }
        paper = _parse_core_work(work)
        assert paper["is_thesis"] is True
        assert paper["source"] == "core"
        assert paper["pdf_url"] == "http://x/pdf"
        assert paper["authors"] == ["Jane Doe"]
        assert paper["language"] == "en"

    def test_handles_missing_fields(self):
        paper = _parse_core_work({"id": 1, "title": "T"})
        assert paper["is_thesis"] is False
        assert paper["authors"] == []


class TestYokTezParsing:
    def test_parse_results_extracts_thesis(self):
        html = (
            '<a href="tezDetay.jsp?id=ABC123&no=1">'
            '<td>Makine Öğrenmesi Uygulamaları</td>'
            '<td>Ahmet Yılmaz</td>'
            '<td>Doktora</td>'
            '<td>2020</td></a>'
        )
        papers = _parse_results(html, max_results=10)
        assert len(papers) == 1
        p = papers[0]
        assert p["is_thesis"] is True
        assert p["source"] == "yoktez"
        assert p["thesis_type"] == "PhD"
        assert "ABC123" in p["url"]
        assert p["language"] == "tr"

    def test_no_results(self):
        assert _parse_results("<html>no theses here</html>", 10) == []


class TestBilingualQueryParsing:
    class _FakeLLM:
        """Returns a fixed analysis response with EN + TR queries."""
        def __init__(self, content):
            self._content = content

        def invoke(self, _prompt):
            class R:
                pass
            r = R()
            r.content = self._content
            return r

    def test_parses_english_and_turkish_queries(self):
        content = (
            "CONCEPTS: rag, retrieval\n"
            "SYSTEMS: DPR, FiD\n"
            "CATEGORIES: technique, survey\n"
            "QUERIES:\n"
            "- retrieval augmented generation survey\n"
            "- dense passage retrieval evaluation\n"
            "QUERIES_TR:\n"
            "- geri çağırma destekli üretim derlemesi\n"
            "- yoğun pasaj geri çağırma değerlendirmesi\n"
        )
        result = analyze_topic_with_llm("rag", self._FakeLLM(content))
        assert len(result["queries"]) == 2
        assert len(result["queries_tr"]) == 2
        assert "geri çağırma destekli üretim derlemesi" in result["queries_tr"]
        # English and Turkish must not bleed into each other.
        assert "geri çağırma destekli üretim derlemesi" not in result["queries"]
