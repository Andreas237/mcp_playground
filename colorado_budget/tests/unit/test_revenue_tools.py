"""
Unit tests for revenue.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.revenue import (
    _extract_pdf_links,
    _head_check,
    find_legislative_forecast,
    find_tabor_resources,
    find_tax_expenditure_report,
)

# ---------------------------------------------------------------------------
# _extract_pdf_links
# ---------------------------------------------------------------------------

_FORECAST_PAGE_HTML = """
<html><body>
<a href="/sites/default/files/Dec2025Forecast.pdf">Economic & Revenue Forecast December 2025</a>
<a href="/sites/default/files/sept2025forecast.pdf">September 2025 Forecast</a>
<a href="/sites/default/files/june-2026-forecast-accessible.pdf">June 2026 Forecast</a>
<a href="/publications/forecast-december-2025">Not a PDF</a>
</body></html>
"""


def test_extract_pdf_links_finds_pdfs():
    links = _extract_pdf_links(_FORECAST_PAGE_HTML)
    urls = [l["url"] for l in links]
    assert any("Dec2025Forecast" in u for u in urls)
    assert any("sept2025forecast" in u for u in urls)


def test_extract_pdf_links_skips_non_pdf():
    links = _extract_pdf_links(_FORECAST_PAGE_HTML)
    urls = [l["url"] for l in links]
    assert not any("forecast-december-2025" in u and ".pdf" not in u for u in urls)


def test_extract_pdf_links_prepends_base():
    links = _extract_pdf_links(_FORECAST_PAGE_HTML)
    assert all(l["url"].startswith("http") for l in links)
    assert any("content.leg.colorado.gov" in l["url"] for l in links)


def test_extract_pdf_links_empty():
    assert _extract_pdf_links("<html><body>nothing</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.revenue.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_0_on_error():
    with patch("servers.revenue.httpx.head", side_effect=Exception("timeout")):
        assert _head_check("https://example.com/doc.pdf") == 0


# ---------------------------------------------------------------------------
# find_legislative_forecast
# ---------------------------------------------------------------------------

def _mock_head_200(url, **kwargs):
    r = MagicMock()
    r.status_code = 200
    return r


def test_find_legislative_forecast_returns_json():
    with patch("servers.revenue._fetch_page", return_value=_FORECAST_PAGE_HTML), \
         patch("servers.revenue.httpx.head", side_effect=_mock_head_200):
        result = find_legislative_forecast("2025")
    data = json.loads(result)
    assert "documents" in data
    assert data["year"] == "2025"


def test_find_legislative_forecast_filters_year():
    with patch("servers.revenue._fetch_page", return_value=_FORECAST_PAGE_HTML), \
         patch("servers.revenue.httpx.head", side_effect=_mock_head_200):
        result = find_legislative_forecast("2025")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert all("2025" in u for u in urls)
    assert not any("2026" in u for u in urls)


def test_find_legislative_forecast_filters_quarter():
    with patch("servers.revenue._fetch_page", return_value=_FORECAST_PAGE_HTML), \
         patch("servers.revenue.httpx.head", side_effect=_mock_head_200):
        result = find_legislative_forecast("2025", "december")
    data = json.loads(result)
    assert len(data["documents"]) == 1
    assert "Dec2025" in data["documents"][0]["url"]


def test_find_legislative_forecast_no_results_falls_back():
    def _no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.revenue._fetch_page", return_value="<html></html>"), \
         patch("servers.revenue._get_exa") as mock_exa:
        mock_exa.return_value.search = _no_exa
        result = find_legislative_forecast("2099")
    data = json.loads(result)
    assert data["found"] is False
    assert "forecast_page" in data


# ---------------------------------------------------------------------------
# find_tax_expenditure_report
# ---------------------------------------------------------------------------

_TAX_EXP_HTML = """
<html><body>
<a href="/sites/default/files/2025-te_tax_compilation_report-accessible.pdf">Tax Expenditures Compilation Report 2025</a>
<a href="/sites/default/files/2019-te_tax_expenditures_compilation_report.pdf">2019 Compilation Report</a>
</body></html>
"""


def test_find_tax_expenditure_report_returns_json():
    with patch("servers.revenue._fetch_page", return_value=_TAX_EXP_HTML), \
         patch("servers.revenue.httpx.head", side_effect=_mock_head_200):
        result = find_tax_expenditure_report("2025")
    data = json.loads(result)
    assert "documents" in data
    assert len(data["documents"]) == 1
    assert "2025" in data["documents"][0]["url"]


def test_find_tax_expenditure_report_no_results():
    def _no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.revenue._fetch_page", return_value="<html></html>"), \
         patch("servers.revenue._get_exa") as mock_exa:
        mock_exa.return_value.search = _no_exa
        result = find_tax_expenditure_report("2099")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# find_tabor_resources
# ---------------------------------------------------------------------------

def test_find_tabor_resources_returns_curated_refs_without_topic():
    """With no topic, returns the curated reference pages and makes no Exa call."""
    result = find_tabor_resources()
    data = json.loads(result)
    assert len(data["resources"]) >= 3
    urls = [r["url"] for r in data["resources"]]
    assert any("tabor-revenue-limit" in u for u in urls)
    assert any("tax.colorado.gov/tabor-refund" in u for u in urls)


def test_find_tabor_resources_adds_search_results_with_topic():
    mock_result = MagicMock()
    mock_result.title = "TABOR Surplus Projection"
    mock_result.url = "https://leg.colorado.gov/tabor-surplus"
    mock_result.highlights = ["projected surplus of $1.5 billion"]

    def _search(*args, **kwargs):
        r = MagicMock()
        r.results = [mock_result]
        return r

    with patch("servers.revenue._get_exa") as mock_exa:
        mock_exa.return_value.search = _search
        result = find_tabor_resources("surplus projection")
    data = json.loads(result)
    urls = [r["url"] for r in data["resources"]]
    # curated refs + the one search hit
    assert "https://leg.colorado.gov/tabor-surplus" in urls
    assert len(data["resources"]) >= 4
