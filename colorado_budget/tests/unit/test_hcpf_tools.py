"""
Unit tests for hcpf.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.hcpf import (
    _extract_doc_links,
    _head_check,
    _hcpfig_urls,
    find_caseload_reports,
    find_hcpf_appropriations,
    find_hcpf_budget_request,
)

# ---------------------------------------------------------------------------
# _hcpfig_urls
# ---------------------------------------------------------------------------

def test_hcpfig_urls_contains_both_conventions():
    urls = _hcpfig_urls("2025-26")
    joined = " ".join(urls)
    assert "FY2025-26_hcpfig1.pdf" in joined
    assert "fy2025-26_hcpfig1.pdf" in joined


def test_hcpfig_urls_bad_input():
    assert _hcpfig_urls("not-a-year") == []


# ---------------------------------------------------------------------------
# _extract_doc_links — PDF and Excel
# ---------------------------------------------------------------------------

_HCPF_HTML = """
<html><body>
<a href="/sites/hcpf/files/fy2025-26_budget_request.pdf">FY 2025-26 Budget Request</a>
<a href="/sites/hcpf/files/2025_july_caseload.xlsx">July 2025 Caseload</a>
<a href="/budget/contacts">Not a document</a>
</body></html>
"""


def test_extract_doc_links_finds_pdf_and_excel():
    links = _extract_doc_links(_HCPF_HTML)
    urls = [l["url"] for l in links]
    assert any(u.endswith(".pdf") for u in urls)
    assert any(u.endswith(".xlsx") for u in urls)
    assert len(links) == 2


def test_extract_doc_links_prepends_base():
    links = _extract_doc_links(_HCPF_HTML)
    assert all(l["url"].startswith("http") for l in links)
    assert any("hcpf.colorado.gov" in l["url"] for l in links)


def test_extract_doc_links_empty():
    assert _extract_doc_links("<html><body>nothing</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.hcpf.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_0_on_error():
    with patch("servers.hcpf.httpx.head", side_effect=Exception("timeout")):
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
# find_hcpf_budget_request
# ---------------------------------------------------------------------------

def test_find_hcpf_budget_request_returns_docs():
    with patch("servers.hcpf._fetch_page", return_value=_HCPF_HTML), \
         patch("servers.hcpf._get_exa", return_value=_empty_exa()), \
         patch("servers.hcpf.httpx.head", side_effect=_mock_head_200):
        result = find_hcpf_budget_request("2025-26")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any("2025-26" in u for u in urls)


def test_find_hcpf_budget_request_no_results():
    with patch("servers.hcpf._fetch_page", return_value="<html></html>"), \
         patch("servers.hcpf._get_exa", return_value=_empty_exa()):
        result = find_hcpf_budget_request("2099-00")
    data = json.loads(result)
    assert data["found"] is False
    assert "budget_index" in data


# ---------------------------------------------------------------------------
# find_caseload_reports
# ---------------------------------------------------------------------------

def test_find_caseload_reports_returns_excel():
    with patch("servers.hcpf._fetch_page", return_value=_HCPF_HTML), \
         patch("servers.hcpf._get_exa", return_value=_empty_exa()), \
         patch("servers.hcpf.httpx.head", side_effect=_mock_head_200):
        result = find_caseload_reports("2025")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any(u.endswith(".xlsx") and "2025" in u for u in urls)


def test_find_caseload_reports_no_results():
    with patch("servers.hcpf._fetch_page", return_value="<html></html>"), \
         patch("servers.hcpf._get_exa", return_value=_empty_exa()):
        result = find_caseload_reports("2099")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# find_hcpf_appropriations
# ---------------------------------------------------------------------------

def test_find_hcpf_appropriations_constructs_url():
    # First candidate hcpfig URL HEAD-checks 200 → returned without Exa
    with patch("servers.hcpf.httpx.head", side_effect=_mock_head_200):
        result = find_hcpf_appropriations("2025-26")
    data = json.loads(result)
    assert data["documents"]
    assert "hcpfig1" in data["documents"][0]["url"]
    assert "2025-26" in data["documents"][0]["url"]


def test_find_hcpf_appropriations_no_results_falls_back():
    def _head_404(url, **kwargs):
        r = MagicMock()
        r.status_code = 404
        return r

    with patch("servers.hcpf.httpx.head", side_effect=_head_404), \
         patch("servers.hcpf._get_exa", return_value=_empty_exa()):
        result = find_hcpf_appropriations("2099-00")
    data = json.loads(result)
    assert data["found"] is False
