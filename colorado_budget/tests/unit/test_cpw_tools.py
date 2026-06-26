"""
Unit tests for cpw.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.cpw import (
    _extract_doc_links,
    _head_check,
    _natfig_urls,
    find_cpw_appropriations,
    find_cpw_financial_reports,
    get_sources_and_uses,
)

# ---------------------------------------------------------------------------
# _natfig_urls
# ---------------------------------------------------------------------------

def test_natfig_urls_covers_stems_and_hosts():
    urls = _natfig_urls("2025-26")
    joined = " ".join(urls)
    assert "fy2025-26_natfig_0.pdf" in joined        # observed real file
    assert "natbrf1" in joined                        # briefing variant
    assert "content.leg.colorado.gov" in joined
    assert "leg.colorado.gov" in joined


def test_natfig_urls_bad_input():
    assert _natfig_urls("nope") == []


# ---------------------------------------------------------------------------
# _extract_doc_links
# ---------------------------------------------------------------------------

_CPW_HTML = """
<html><body>
<a href="/sites/default/files/dam/abc/item.9_financial_report_fy24_q4.pdf">Financial Report FY24 Q4</a>
<a href="/sites/default/files/dam/xyz/item.10_financial_update_may_2025.pdf">Financial Update May 2025</a>
<a href="/committees/meetings">Not a document</a>
</body></html>
"""


def test_extract_doc_links_finds_pdfs():
    links = _extract_doc_links(_CPW_HTML)
    urls = [l["url"] for l in links]
    assert any("fy24_q4" in u for u in urls)
    assert any("may_2025" in u for u in urls)
    assert len(links) == 2


def test_extract_doc_links_prepends_base():
    links = _extract_doc_links(_CPW_HTML)
    assert all(l["url"].startswith("http") for l in links)
    assert any("cpw.state.co.us" in l["url"] for l in links)


def test_extract_doc_links_empty():
    assert _extract_doc_links("<html><body>none</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.cpw.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_0_on_error():
    with patch("servers.cpw.httpx.head", side_effect=Exception("timeout")):
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
# find_cpw_financial_reports
# ---------------------------------------------------------------------------

def test_find_cpw_financial_reports_returns_docs():
    with patch("servers.cpw._fetch_page", return_value=_CPW_HTML), \
         patch("servers.cpw._get_exa", return_value=_empty_exa()), \
         patch("servers.cpw.httpx.head", side_effect=_mock_head_200):
        result = find_cpw_financial_reports("2025")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any("may_2025" in u for u in urls)


def test_find_cpw_financial_reports_no_results():
    with patch("servers.cpw._fetch_page", return_value="<html></html>"), \
         patch("servers.cpw._get_exa", return_value=_empty_exa()):
        result = find_cpw_financial_reports("2099")
    data = json.loads(result)
    assert data["found"] is False
    assert "plans_reports_page" in data


# ---------------------------------------------------------------------------
# get_sources_and_uses
# ---------------------------------------------------------------------------

def test_get_sources_and_uses_includes_fact_sheet_when_reachable():
    with patch("servers.cpw.httpx.head", side_effect=_mock_head_200), \
         patch("servers.cpw._get_exa", return_value=_empty_exa()):
        result = get_sources_and_uses()
    data = json.loads(result)
    urls = [r["url"] for r in data["resources"]]
    assert any("Sources_and_Uses_of_Funds_Fact_Sheet.pdf" in u for u in urls)
    assert any("funding-colorado-parks-and-wildlife" in u for u in urls)


def test_get_sources_and_uses_omits_factsheet_when_unreachable():
    def _head_404(url, **kwargs):
        r = MagicMock()
        r.status_code = 404
        return r

    with patch("servers.cpw.httpx.head", side_effect=_head_404), \
         patch("servers.cpw._get_exa", return_value=_empty_exa()):
        result = get_sources_and_uses()
    data = json.loads(result)
    urls = [r["url"] for r in data["resources"]]
    assert not any("Sources_and_Uses_of_Funds_Fact_Sheet.pdf" in u for u in urls)
    # funding overview page is always present
    assert any("funding-colorado-parks-and-wildlife" in u for u in urls)


# ---------------------------------------------------------------------------
# find_cpw_appropriations
# ---------------------------------------------------------------------------

def test_find_cpw_appropriations_constructs_url():
    with patch("servers.cpw.httpx.head", side_effect=_mock_head_200):
        result = find_cpw_appropriations("2025-26")
    data = json.loads(result)
    assert data["documents"]
    assert "2025-26" in data["documents"][0]["url"]
    assert any(s in data["documents"][0]["url"] for s in ("natfig", "natbrf"))


def test_find_cpw_appropriations_no_results_falls_back():
    def _head_404(url, **kwargs):
        r = MagicMock()
        r.status_code = 404
        return r

    with patch("servers.cpw.httpx.head", side_effect=_head_404), \
         patch("servers.cpw._get_exa", return_value=_empty_exa()):
        result = find_cpw_appropriations("2099-00")
    data = json.loads(result)
    assert data["found"] is False
