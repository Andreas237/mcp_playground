"""
Unit tests for ospb.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.ospb import (
    _extract_pdf_links,
    _fy_variants,
    _head_check,
    find_budget_amendments,
    find_governor_budget,
    find_revenue_forecast,
)

# ---------------------------------------------------------------------------
# _fy_variants
# ---------------------------------------------------------------------------

def test_fy_variants_contains_full():
    variants = _fy_variants("2026-27")
    assert "2026-27" in variants


def test_fy_variants_contains_end_year():
    variants = _fy_variants("2026-27")
    assert "2027" in variants


def test_fy_variants_contains_short():
    variants = _fy_variants("2026-27")
    assert "26-27" in variants


def test_fy_variants_old_year():
    variants = _fy_variants("2024-25")
    assert "2025" in variants
    assert "2024-25" in variants


# ---------------------------------------------------------------------------
# _extract_pdf_links
# ---------------------------------------------------------------------------

_PDF_PAGE_HTML = """
<html><body>
<a href="/sites/default/files/FY2026-27GovernorsBudget.pdf">FY2026-27 Governor's Budget</a>
<a href="/sites/default/files/FY2025-26GovernorsBudget.pdf">FY2025-26 Governor's Budget</a>
<a href="https://external.gov/other.pdf">External Doc</a>
<a href="/not-a-pdf.html">Not a PDF</a>
</body></html>
"""

def test_extract_pdf_links_finds_all_pdfs():
    links = _extract_pdf_links(_PDF_PAGE_HTML)
    urls = [l["url"] for l in links]
    assert any("FY2026-27" in u for u in urls)
    assert any("FY2025-26" in u for u in urls)
    assert any("external.gov" in u for u in urls)


def test_extract_pdf_links_skips_non_pdf():
    links = _extract_pdf_links(_PDF_PAGE_HTML)
    urls = [l["url"] for l in links]
    assert not any("not-a-pdf.html" in u for u in urls)


def test_extract_pdf_links_prepends_base():
    links = _extract_pdf_links(_PDF_PAGE_HTML)
    relative_links = [l for l in links if "ospb.colorado.gov" in l["url"]]
    assert len(relative_links) >= 2


def test_extract_pdf_links_captures_label():
    links = _extract_pdf_links(_PDF_PAGE_HTML)
    labels = [l["label"] for l in links]
    assert any("Governor" in lb for lb in labels)


def test_extract_pdf_links_empty_html():
    assert _extract_pdf_links("<html><body>no links</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.ospb.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_404():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("servers.ospb.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/missing.pdf") == 404


def test_head_check_returns_0_on_error():
    with patch("servers.ospb.httpx.head", side_effect=Exception("timeout")):
        assert _head_check("https://example.com/doc.pdf") == 0


# ---------------------------------------------------------------------------
# find_governor_budget
# ---------------------------------------------------------------------------

_BUDGET_PAGE_HTML = """
<html><body>
<h1>Governor's Budget</h1>
<a href="/sites/default/files/FY2026-27GovernorsBudget.pdf">FY2026-27 Governor's Budget (Full)</a>
<a href="/sites/default/files/FY2025-26GovernorsBudget.pdf">FY2025-26 Governor's Budget (Full)</a>
</body></html>
"""

def _mock_fetch_page(url):
    return _BUDGET_PAGE_HTML


def _mock_head_200(url, **kwargs):
    r = MagicMock()
    r.status_code = 200
    return r


def test_find_governor_budget_returns_json():
    with patch("servers.ospb._fetch_page", side_effect=_mock_fetch_page), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_governor_budget("2026-27")
    data = json.loads(result)
    assert "documents" in data
    assert data["fiscal_year"] == "2026-27"


def test_find_governor_budget_finds_matching_year():
    with patch("servers.ospb._fetch_page", side_effect=_mock_fetch_page), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_governor_budget("2026-27")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any("2026-27" in u or "2027" in u for u in urls)


def test_find_governor_budget_excludes_wrong_year():
    with patch("servers.ospb._fetch_page", side_effect=_mock_fetch_page), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_governor_budget("2026-27")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert not any("2025-26" in u for u in urls)


def test_find_governor_budget_no_results_falls_back():
    """When page scrape finds nothing, returns helpful message (Exa fallback also mocked empty)."""
    def _mock_no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.ospb._fetch_page", return_value="<html></html>"), \
         patch("servers.ospb._get_exa") as mock_exa:
        mock_exa.return_value.search = _mock_no_exa
        result = find_governor_budget("2099-00")
    data = json.loads(result)
    assert data["found"] is False
    assert "ospb_budget_page" in data


# ---------------------------------------------------------------------------
# find_revenue_forecast
# ---------------------------------------------------------------------------

_FORECAST_PAGE_HTML = """
<html><body>
<a href="/sites/default/files/September2025RevenueForecast.pdf">September 2025 Revenue Forecast</a>
<a href="/sites/default/files/June2025RevenueForecast.pdf">June 2025 Revenue Forecast</a>
<a href="/sites/default/files/March2025RevenueForecast.pdf">March 2025 Revenue Forecast</a>
</body></html>
"""

def _mock_forecast_page(url):
    return _FORECAST_PAGE_HTML


def test_find_revenue_forecast_returns_json():
    with patch("servers.ospb._fetch_page", side_effect=_mock_forecast_page), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_revenue_forecast("2025")
    data = json.loads(result)
    assert "documents" in data


def test_find_revenue_forecast_filters_year():
    with patch("servers.ospb._fetch_page", side_effect=_mock_forecast_page), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_revenue_forecast("2025")
    data = json.loads(result)
    assert len(data["documents"]) == 3  # all three are 2025


def test_find_revenue_forecast_filters_quarter():
    with patch("servers.ospb._fetch_page", side_effect=_mock_forecast_page), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_revenue_forecast("2025", "september")
    data = json.loads(result)
    assert len(data["documents"]) == 1
    assert "September" in data["documents"][0]["url"]


def test_find_revenue_forecast_no_results():
    def _mock_no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.ospb._fetch_page", return_value=None), \
         patch("servers.ospb._get_exa") as mock_exa:
        mock_exa.return_value.search = _mock_no_exa
        result = find_revenue_forecast("2099")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# find_budget_amendments
# ---------------------------------------------------------------------------

_AMEND_PAGE_HTML = """
<html><body>
<a href="/sites/default/files/FY2025-26SupplementalRequest.pdf">FY2025-26 Supplemental Budget Request</a>
</body></html>
"""

def test_find_budget_amendments_returns_json():
    with patch("servers.ospb._fetch_page", return_value=_AMEND_PAGE_HTML), \
         patch("servers.ospb.httpx.head", side_effect=_mock_head_200):
        result = find_budget_amendments("2025-26")
    data = json.loads(result)
    assert "documents" in data
    assert data["fiscal_year"] == "2025-26"


def test_find_budget_amendments_no_results():
    def _no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.ospb._fetch_page", return_value="<html></html>"), \
         patch("servers.ospb._get_exa") as mock_exa:
        mock_exa.return_value.search = _no_exa
        result = find_budget_amendments("2099-00")
    data = json.loads(result)
    assert data["found"] is False
