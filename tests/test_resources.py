"""
Offline tests for supplementary resources (GitHub / Medium / YouTube) and the
resources agent node. No network access.
"""

import sys
sys.path.insert(0, '..')

from tools.resources_tools import _slugify, _parse_rss
import tools.resources_tools as rt
import agents.resources_agent as ra


class TestSlugify:
    def test_basic(self):
        assert _slugify("Retrieval Augmented Generation") == "retrieval-augmented-generation"
        assert _slugify("makine öğrenmesi!!") == "makine-renmesi"  # non-ascii dropped
        assert _slugify("  C++  & AI  ") == "c-ai"


class TestParseRss:
    def test_parses_items(self):
        xml = """
        <rss><channel>
          <item><title><![CDATA[First Post]]></title><link>https://m.com/a?x=1</link>
            <dc:creator>Jane</dc:creator><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
          <item><title>Second</title><link>https://m.com/b</link></item>
        </channel></rss>
        """
        items = _parse_rss(xml, 5)
        assert len(items) == 2
        assert items[0]["title"] == "First Post"
        assert items[0]["url"] == "https://m.com/a"  # query stripped
        assert items[0]["author"] == "Jane"
        assert items[1]["title"] == "Second"

    def test_skips_items_without_link(self):
        xml = "<item><title>No link</title></item>"
        assert _parse_rss(xml, 5) == []


class TestGitHubParsingOffline:
    def test_no_network_when_topic_empty(self):
        # gather_resources short-circuits on empty topic (no calls made).
        out = rt.gather_resources("")
        assert out == {"github": [], "articles": [], "videos": []}


class TestYouTubeNoKey:
    def test_returns_empty_without_key(self, monkeypatch):
        monkeypatch.setattr(rt, "YOUTUBE_API_KEY", "")
        assert rt.search_youtube("anything") == []


class TestResourcesNode:
    def test_node_attaches_resources(self, monkeypatch):
        monkeypatch.setattr(ra, "ENABLE_RESOURCES", True)
        monkeypatch.setattr(ra, "gather_resources",
                            lambda topic: {"github": [{"name": "x/y", "stars": 5, "url": "u",
                                                       "description": "", "language": "Py"}],
                                           "articles": [], "videos": []})
        out = ra.resources_node({"query": "topic"})
        assert out["resources_completed"] is True
        assert out["resources"]["github"][0]["name"] == "x/y"

    def test_node_disabled(self, monkeypatch):
        monkeypatch.setattr(ra, "ENABLE_RESOURCES", False)
        out = ra.resources_node({"query": "topic"})
        assert out["resources"] == {"github": [], "articles": [], "videos": []}

    def test_node_survives_errors(self, monkeypatch):
        monkeypatch.setattr(ra, "ENABLE_RESOURCES", True)
        def boom(_):
            raise RuntimeError("network down")
        monkeypatch.setattr(ra, "gather_resources", boom)
        out = ra.resources_node({"query": "topic"})
        assert out["resources"] == {"github": [], "articles": [], "videos": []}
