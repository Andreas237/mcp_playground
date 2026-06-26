"""
Unit tests for school_finance.py helper functions and MCP tools.
All network calls are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.school_finance import (
    _extract_doc_links,
    _head_check,
    find_finance_formula_resources,
    find_per_pupil_funding,
    find_school_finance_act,
)

# ---------------------------------------------------------------------------
# _extract_doc_links — handles PDF and Excel
# ---------------------------------------------------------------------------

_FINANCE_HTML = """
<html><body>
<a href="/sites/default/files/docs/cdefinance/fy2025-26_total_program.pdf">FY2025-26 Total Program Funding</a>
<a href="/sites/default/files/docs/cdefinance/fy2024-25_per_pupil.xlsx">FY2024-25 Per Pupil by District</a>
<a href="/sites/default/files/docs/cdefinance/HB24-1448_MLO_Cap.xlsx">HB24-1448 MLO Cap Calculation</a>
<a href="/cdefinance/contacts">Not a document</a>
</body></html>
"""


def test_extract_doc_links_finds_pdf_and_excel():
    links = _extract_doc_links(_FINANCE_HTML)
    urls = [l["url"] for l in links]
    assert any(u.endswith(".pdf") for u in urls)
    assert any(u.endswith(".xlsx") for u in urls)
    assert len(links) == 3


def test_extract_doc_links_skips_non_documents():
    links = _extract_doc_links(_FINANCE_HTML)
    assert not any("contacts" in l["url"] for l in links)


def test_extract_doc_links_prepends_base():
    links = _extract_doc_links(_FINANCE_HTML)
    assert all(l["url"].startswith("http") for l in links)
    assert any("cde.state.co.us" in l["url"] for l in links)


def test_extract_doc_links_empty():
    assert _extract_doc_links("<html><body>nothing</body></html>") == []


# ---------------------------------------------------------------------------
# _head_check
# ---------------------------------------------------------------------------

def test_head_check_returns_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("servers.school_finance.httpx.head", return_value=mock_resp):
        assert _head_check("https://example.com/doc.pdf") == 200


def test_head_check_returns_0_on_error():
    with patch("servers.school_finance.httpx.head", side_effect=Exception("timeout")):
        assert _head_check("https://example.com/doc.pdf") == 0


# ---------------------------------------------------------------------------
# find_school_finance_act
# ---------------------------------------------------------------------------

def _mock_head_200(url, **kwargs):
    r = MagicMock()
    r.status_code = 200
    return r


def _empty_exa():
    """A mock Exa whose search returns no results (isolates tests from network)."""
    mock = MagicMock()
    mock.search.return_value.results = []
    return mock


def test_find_school_finance_act_returns_docs():
    with patch("servers.school_finance._fetch_page", return_value=_FINANCE_HTML), \
         patch("servers.school_finance._get_exa", return_value=_empty_exa()), \
         patch("servers.school_finance.httpx.head", side_effect=_mock_head_200):
        result = find_school_finance_act("2025-26")
    data = json.loads(result)
    assert "documents" in data
    urls = [d["url"] for d in data["documents"]]
    assert any("2025-26" in u for u in urls)


def test_find_school_finance_act_filters_year():
    with patch("servers.school_finance._fetch_page", return_value=_FINANCE_HTML), \
         patch("servers.school_finance._get_exa", return_value=_empty_exa()), \
         patch("servers.school_finance.httpx.head", side_effect=_mock_head_200):
        result = find_school_finance_act("2025-26")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert all("2024-25" not in u for u in urls)


def test_find_school_finance_act_falls_back_to_exa():
    def _no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.school_finance._fetch_page", return_value="<html></html>"), \
         patch("servers.school_finance._get_exa") as mock_exa:
        mock_exa.return_value.search = _no_exa
        result = find_school_finance_act("2099-00")
    data = json.loads(result)
    assert data["found"] is False
    assert "finance_page" in data


# ---------------------------------------------------------------------------
# find_per_pupil_funding
# ---------------------------------------------------------------------------

def test_find_per_pupil_funding_returns_excel():
    with patch("servers.school_finance._fetch_page", return_value=_FINANCE_HTML), \
         patch("servers.school_finance._get_exa", return_value=_empty_exa()), \
         patch("servers.school_finance.httpx.head", side_effect=_mock_head_200):
        result = find_per_pupil_funding("2024-25")
    data = json.loads(result)
    urls = [d["url"] for d in data["documents"]]
    assert any(u.endswith(".xlsx") and "2024-25" in u for u in urls)


def test_find_per_pupil_funding_no_results():
    def _no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.school_finance._fetch_page", return_value="<html></html>"), \
         patch("servers.school_finance._get_exa") as mock_exa:
        mock_exa.return_value.search = _no_exa
        result = find_per_pupil_funding("2099-00")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# find_finance_formula_resources
# ---------------------------------------------------------------------------

def test_find_finance_formula_resources_includes_curated_refs():
    def _no_exa(*args, **kwargs):
        r = MagicMock()
        r.results = []
        return r

    with patch("servers.school_finance._get_exa") as mock_exa:
        mock_exa.return_value.search = _no_exa
        result = find_finance_formula_resources()
    data = json.loads(result)
    urls = [r["url"] for r in data["resources"]]
    assert any("cdefinance" in u for u in urls)
    assert any("legislative-council-staff/school-finance" in u for u in urls)


def test_find_finance_formula_resources_adds_search_hits():
    hit = MagicMock()
    hit.title = "HB24-1448 New Formula Phase-In"
    hit.url = "https://leg.colorado.gov/bills/HB24-1448"
    hit.highlights = ["phases in a new formula through 2030"]

    def _search(*args, **kwargs):
        r = MagicMock()
        r.results = [hit]
        return r

    with patch("servers.school_finance._get_exa") as mock_exa:
        mock_exa.return_value.search = _search
        result = find_finance_formula_resources("phase-in")
    data = json.loads(result)
    urls = [r["url"] for r in data["resources"]]
    assert "https://leg.colorado.gov/bills/HB24-1448" in urls
