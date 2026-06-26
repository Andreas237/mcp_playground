"""
Unit tests for cdot.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.cdot import (
    _extract_doc_links,
    _fy_full,
    _fy_short,
    _head_check,
    _trafig_urls,
    find_cdot_appropriations,
    find_cdot_budget,
    find_stip,
)

# ---------------------------------------------------------------------------
# fiscal-year helpers
# ---------------------------------------------------------------------------

def test_fy_full():
    assert _fy_full("2026-27") == "2026-2027"
    assert _fy_full("2025-26") == "2025-2026"
    assert _fy_full("nope") is None


def test_fy_short():
    assert _fy_short("2026-27") == "26-27"
    assert _fy_short("2025-26") == "25-26"
    assert _fy_short("nope") is None


def test_trafig_urls_covers_both_year_forms():
    urls = _trafig_urls("2026-27")
    joined = " ".join(urls)
    assert "fy26-27_trafig.pdf" in joined        # observed short form
    assert "fy2026-27_trahrg.pdf" in joined       # observed full form (hearing)
    assert "content.leg.colorado.gov" in joined
    assert "leg.colorado.gov" in joined


def test_trafig_urls_bad_input():
    assert _trafig_urls("nope") == []


# ---------------------------------------------------------------------------
# _extract_doc_links
# ---------------------------------------------------------------------------

_CDOT_HTML = """
<html><body>
<a href="/business/budget/documents/fy-2026-27-final-budget.pdf">FY 2026-27 Final Budget</a>
<a href="/programs/planning/assets/stip-exec-summary-fy27.pdf">STIP Executive Summary</a>
<a href="/business/budget">Not a document</a>
</body></html>
"""


def test_extract_doc_links_finds_pdfs():
    links = _extract_doc_links(_CDOT_HTML)
    urls = [l["url"] for l in links]
    assert any("final-budget" in u for u in urls)
    assert any("stip-exec" in u for u in urls)
    assert len(links) == 2


def test_extract_doc_links_prepends_base():
    links = _extract_doc_links(_CDOT_HTML)
    assert all(l["url"].startswith("http") for l in links)
    assert any("codot.gov" in l["url"] for l in links)


def test_extract_doc_links_empty():
    assert _extract_doc_links("<html><body>none</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.cdot.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_0_on_error():
    with patch("servers.cdot.httpx.head", side_effect=Exception("timeout")):
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
# find_cdot_budget
# ---------------------------------------------------------------------------

def test_find_cdot_budget_returns_docs():
    with patch("servers.cdot._fetch_page", return_value=_CDOT_HTML), \
         patch("servers.cdot._get_exa", return_value=_empty_exa()), \
         patch("servers.cdot.httpx.head", side_effect=_mock_head_200):
        result = find_cdot_budget("2026-27")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any("2026-27" in u for u in urls)


def test_find_cdot_budget_no_results():
    with patch("servers.cdot._fetch_page", return_value="<html></html>"), \
         patch("servers.cdot._get_exa", return_value=_empty_exa()):
        result = find_cdot_budget("2099-00")
    data = json.loads(result)
    assert data["found"] is False
    assert "budget_index" in data


# ---------------------------------------------------------------------------
# find_cdot_appropriations
# ---------------------------------------------------------------------------

def test_find_cdot_appropriations_constructs_url():
    with patch("servers.cdot.httpx.head", side_effect=_mock_head_200):
        result = find_cdot_appropriations("2026-27")
    data = json.loads(result)
    assert data["documents"]
    assert any(s in data["documents"][0]["url"] for s in ("trafig", "trahrg", "trabrf"))


def test_find_cdot_appropriations_no_results():
    def _head_404(url, **kwargs):
        r = MagicMock()
        r.status_code = 404
        return r

    with patch("servers.cdot.httpx.head", side_effect=_head_404), \
         patch("servers.cdot._get_exa", return_value=_empty_exa()):
        result = find_cdot_appropriations("2099-00")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# find_stip
# ---------------------------------------------------------------------------

def test_find_stip_returns_docs():
    with patch("servers.cdot._fetch_page", return_value=_CDOT_HTML), \
         patch("servers.cdot._get_exa", return_value=_empty_exa()), \
         patch("servers.cdot.httpx.head", side_effect=_mock_head_200):
        result = find_stip()
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any("stip" in u.lower() for u in urls)


def test_find_stip_no_results():
    with patch("servers.cdot._fetch_page", return_value="<html></html>"), \
         patch("servers.cdot._get_exa", return_value=_empty_exa()):
        result = find_stip("2099")
    data = json.loads(result)
    assert data["found"] is False
    assert "stip_page" in data
