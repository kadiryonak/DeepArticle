"""
Offline tests for PDF text utilities (no network).
"""

import sys
sys.path.insert(0, '..')

from tools.pdf_tools import clean_pdf_text, split_sections


class TestCleanPdfText:
    def test_collapses_whitespace(self):
        assert clean_pdf_text("a   b\t\tc") == "a b c"

    def test_collapses_blank_lines(self):
        assert clean_pdf_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_null_bytes(self):
        assert "\x00" not in clean_pdf_text("a\x00b")


class TestSplitSections:
    def test_splits_imrad_sections(self):
        text = (
            "Abstract\nThis paper studies X.\n"
            "1. Introduction\nBackground here.\n"
            "Method\nWe do Y.\n"
            "Results\nWe found Z.\n"
            "Conclusion\nWe conclude."
        )
        sections = split_sections(text)
        assert "abstract" in sections
        assert "method" in sections
        assert "results" in sections
        assert "We do Y" in sections["method"]

    def test_no_sections_returns_empty(self):
        assert split_sections("just some random text with no headers") == {}
