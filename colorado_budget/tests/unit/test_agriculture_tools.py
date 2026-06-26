"""
Unit tests for agriculture.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.agriculture import (
    _agrfig_urls,
    _extract_doc_links,
    _head_check,
    find_agriculture_appropriations,
    find_agriculture_budget,
    find_agriculture_programs,
)

# ---------------------------------------------------------------------------
# _agrfig_urls
# ---------------------------------------------------------------------------

def test_agrfig_urls_covers_cases_and_hosts():
    urls = _agrfig_urls("2026-27")
    joined = " ".join(urls)
    assert "FY2026-27_agrfig.pdf" in joined        # observed real file
    assert "fy2026-27_agrfig.pdf" in joined
    assert "content.leg.colorado.gov" in joined
    assert "leg.colorado.gov" in joined


def test_agrfig_urls_bad_input():
    assert _agrfig_urls("nope") == []


# ---------------------------------------------------------------------------
# _extract_doc_links
# ---------------------------------------------------------------------------

_AG_HTML = """
<html><body>
<a href="/sites/ag/files/fy2025-26_cda_budget.pdf">FY2025-26 CDA Budget</a>
<a href="/sites/ag/files/cda_at_a_glance_2025.pdf">CDA at a Glance 2025</a>
<a href="/category/grants">Not a document</a>
</body></html>
"""


def test_extract_doc_links_finds_pdfs():
    links = _extract_doc_links(_AG_HTML)
    urls = [l["url"] for l in links]
    assert any("cda_budget" in u for u in urls)
    assert len(links) == 2


def test_extract_doc_links_prepends_base():
    links = _extract_doc_links(_AG_HTML)
    assert all(l["url"].startswith("http") for l in links)
    assert any("ag.colorado.gov" in l["url"] for l in links)


def test_extract_doc_links_empty():
    assert _extract_doc_links("<html><body>none</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.agriculture.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_0_on_error():
    with patch("servers.agriculture.httpx.head", side_effect=Exception("timeout")):
        assert _head_check("https://example.com/doc.pdf") == 0


# ---------------------------------------------------------------------------
# helpers for tool tests
# ---------------------------------------------------------------------------

def _mock_head_200(url, **kwargs):
    r = MagicMock()
    r.status_code = 200
    return r


def _empty_exa():
    mock = MagicMock()
    mock.search.return_value.results = []
    return mock


# ---------------------------------------------------------------------------
# find_agriculture_budget
# ---------------------------------------------------------------------------

def test_find_agriculture_budget_returns_docs():
    with patch("servers.agriculture._fetch_page", return_value=_AG_HTML), \
         patch("servers.agriculture._get_exa", return_value=_empty_exa()), \
         patch("servers.agriculture.httpx.head", side_effect=_mock_head_200):
        result = find_agriculture_budget("2025-26")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any("2025-26" in u for u in urls)


def test_find_agriculture_budget_no_results():
    with patch("servers.agriculture._fetch_page", return_value="<html></html>"), \
         patch("servers.agriculture._get_exa", return_value=_empty_exa()):
        result = find_agriculture_budget("2099-00")
    data = json.loads(result)
    assert data["found"] is False
    assert "state_budget_page" in data


# ---------------------------------------------------------------------------
# find_agriculture_appropriations
# ---------------------------------------------------------------------------

def test_find_agriculture_appropriations_constructs_url():
    with patch("servers.agriculture.httpx.head", side_effect=_mock_head_200):
        result = find_agriculture_appropriations("2026-27")
    data = json.loads(result)
    assert data["documents"]
    assert "agrfig" in data["documents"][0]["url"]
    assert "2026-27" in data["documents"][0]["url"]


def test_find_agriculture_appropriations_no_results():
    def _head_404(url, **kwargs):
        r = MagicMock()
        r.status_code = 404
        return r

    with patch("servers.agriculture.httpx.head", side_effect=_head_404), \
         patch("servers.agriculture._get_exa", return_value=_empty_exa()):
        result = find_agriculture_appropriations("2099-00")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# find_agriculture_programs
# ---------------------------------------------------------------------------

def test_find_agriculture_programs_includes_curated_pages():
    with patch("servers.agriculture._get_exa", return_value=_empty_exa()):
        result = find_agriculture_programs()
    data = json.loads(result)
    urls = [r["url"] for r in data["resources"]]
    assert any("performance-plan" in u for u in urls)
    assert any("/category/grants" in u for u in urls)
