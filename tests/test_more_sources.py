"""
Offline parse tests for the additional sources: CrossRef, DOAJ, DBLP, OpenAIRE.
No network access.
"""

import sys
sys.path.insert(0, '..')

from tools.crossref_tools import _parse_crossref_item, _strip_jats
from tools.doaj_tools import _parse_doaj
from tools.dblp_tools import _parse_dblp
from tools.openaire_tools import _parse_openaire


class TestCrossRef:
    def test_parse(self):
        item = {
            "title": ["A Paper"],
            "author": [{"given": "Jane", "family": "Doe"}],
            "DOI": "10.1/X",
            "container-title": ["NeurIPS"],
            "is-referenced-by-count": 42,
            "type": "journal-article",
            "abstract": "<jats:p>Hello world</jats:p>",
            "published": {"date-parts": [[2020, 5, 1]]},
            "URL": "http://x",
        }
        p = _parse_crossref_item(item)
        assert p["title"] == "A Paper"
        assert p["authors"] == ["Jane Doe"]
        assert p["doi"] == "10.1/x"
        assert p["citation_count"] == 42
        assert p["abstract"] == "Hello world"
        assert p["source"] == "crossref"
        assert p["published_date"] == "2020-05-01"

    def test_strip_jats(self):
        assert _strip_jats("<jats:p>Hi <b>there</b></jats:p>") == "Hi there"


class TestDOAJ:
    def test_parse(self):
        result = {"id": "abc", "bibjson": {
            "title": "T", "author": [{"name": "A. Author"}], "abstract": "Ab",
            "year": "2021", "journal": {"title": "J Open", "language": ["en"]},
            "link": [{"type": "fulltext", "url": "http://pdf"}],
            "identifier": [{"type": "doi", "id": "10.2/Y"}], "keywords": ["k"],
        }}
        p = _parse_doaj(result)
        assert p["title"] == "T"
        assert p["authors"] == ["A. Author"]
        assert p["pdf_url"] == "http://pdf"
        assert p["doi"] == "10.2/y"
        assert p["journal_name"] == "J Open"
        assert p["source"] == "doaj"


class TestDBLP:
    def test_parse_multiple_authors(self):
        hit = {"@id": "1", "info": {
            "title": "Deep Title.", "authors": {"author": [{"text": "X"}, {"text": "Y"}]},
            "year": "2019", "venue": "ICML", "doi": "10.3/Z", "ee": "http://ee",
            "type": "Conference and Workshop Papers",
        }}
        p = _parse_dblp(hit)
        assert p["title"] == "Deep Title"  # trailing period stripped
        assert p["authors"] == ["X", "Y"]
        assert p["doi"] == "10.3/z"
        assert p["venue"] == "ICML"
        assert p["url"] == "http://ee"
        assert p["source"] == "dblp"

    def test_parse_single_author(self):
        hit = {"info": {"title": "Solo", "authors": {"author": {"text": "Z"}}, "year": "2020"}}
        p = _parse_dblp(hit)
        assert p["authors"] == ["Z"]


class TestOpenAIRE:
    def test_parse_unescapes_and_extracts(self):
        result = {"metadata": {"oaf:entity": {"oaf:result": {
            "title": {"$": "OT &amp; more"},
            "creator": [{"$": "Auth One"}, {"$": "Auth Two"}],
            "pid": [{"@classid": "doi", "$": "10.4/Q"}],
            "dateofacceptance": {"$": "2018-01-01"},
            "resulttype": {"@classname": "publication"},
            "description": {"$": "Desc"},
        }}}}
        p = _parse_openaire(result)
        assert p["title"] == "OT & more"
        assert p["authors"] == ["Auth One", "Auth Two"]
        assert p["doi"] == "10.4/q"
        assert p["published_date"] == "2018-01-01"
        assert p["source"] == "openaire"

    def test_thesis_flagged(self):
        result = {"metadata": {"oaf:entity": {"oaf:result": {
            "title": {"$": "A Thesis"},
            "resulttype": {"@classname": "Doctoral thesis"},
        }}}}
        p = _parse_openaire(result)
        assert p["is_thesis"] is True


class TestWiring:
    def test_new_sources_registered(self):
        import agents.search_agent as sa
        for src in ("crossref", "doaj", "dblp", "openaire", "pubmed"):
            assert src in sa._SOURCE_FUNCS
