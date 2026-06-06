"""
PDF full-text extraction tools.

Many papers expose an open-access PDF (``pdf_url``). This module downloads and
extracts the text so agents can go beyond the abstract — e.g. to summarize the
methodology or results. Uses PyMuPDF (``fitz``) when available.
"""

import re
from typing import Dict, Any, List

import requests
from langchain_core.tools import tool

import sys
sys.path.insert(0, '..')
from utils.cache import disk_cache
from utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Common IMRaD-style section headers to split a paper into parts.
_SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\b(?:1\.?\s*)?introduction\b",
    "method": r"\b(?:method(?:s|ology)?|approach)\b",
    "results": r"\b(?:results|evaluation|experiments?)\b",
    "conclusion": r"\b(?:conclusion|conclusions|discussion)\b",
}


def clean_pdf_text(text: str) -> str:
    """Collapse excessive whitespace and strip control characters from PDF text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> Dict[str, str]:
    """
    Heuristically split paper text into IMRaD sections.

    Returns a dict that may include any of: abstract, introduction, method,
    results, conclusion. Best-effort — papers vary widely in formatting.
    """
    lowered = text.lower()
    hits: List[tuple] = []
    for name, pattern in _SECTION_PATTERNS.items():
        m = re.search(pattern, lowered)
        if m:
            hits.append((m.start(), name))
    hits.sort()

    sections: Dict[str, str] = {}
    for idx, (start, name) in enumerate(hits):
        end = hits[idx + 1][0] if idx + 1 < len(hits) else len(text)
        sections[name] = text[start:end].strip()
    return sections


@disk_cache(namespace="pdf_text")
def _fetch_pdf_text(pdf_url: str, max_pages: int = 30) -> str:
    """Download a PDF and return its cleaned text (cached)."""
    headers = {"User-Agent": "DeepArticle/0.1 (research tool)"}
    response = requests.get(pdf_url, headers=headers, timeout=60)
    response.raise_for_status()

    parts: List[str] = []
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        for page in doc[:max_pages]:
            parts.append(page.get_text())
    return clean_pdf_text("\n".join(parts))


@tool
def extract_pdf_text(pdf_url: str, max_pages: int = 30) -> Dict[str, Any]:
    """
    Download a paper's PDF and extract its full text and IMRaD sections.

    Args:
        pdf_url: Direct URL to the PDF file.
        max_pages: Maximum number of pages to read (default 30).

    Returns:
        Dict with ``text``, ``sections``, ``char_count`` and ``found`` —
        or an ``error`` message on failure.
    """
    if not PYMUPDF_AVAILABLE:
        return {"error": "PyMuPDF not installed. Run: pip install PyMuPDF", "found": False}
    if not pdf_url:
        return {"error": "pdf_url is required", "found": False}

    try:
        text = _fetch_pdf_text(pdf_url, max_pages)
        if not text:
            return {"pdf_url": pdf_url, "found": False, "text": "", "sections": {}, "char_count": 0}
        return {
            "pdf_url": pdf_url,
            "found": True,
            "text": text,
            "sections": split_sections(text),
            "char_count": len(text),
        }
    except requests.exceptions.RequestException as e:
        logger.warning("PDF download failed for %s: %s", pdf_url, e)
        return {"error": f"PDF download failed: {str(e)}", "found": False}
    except Exception as e:
        logger.warning("PDF parse failed for %s: %s", pdf_url, e)
        return {"error": f"PDF parse failed: {str(e)}", "found": False}


def get_pdf_tools():
    """Return list of PDF tools."""
    return [extract_pdf_text]
