import json
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from servers.legislature import (
    _normalize_bill_id,
    _find_recent_fiscal_note_url,
    _jbc_url,
    get_bill_details,
    get_fiscal_note,
    find_appropriations_documents,
)


# -- _normalize_bill_id --

def test_normalize_standard():
    assert _normalize_bill_id("HB24-1234") == "hb24-1234"

def test_normalize_without_hyphen():
    assert _normalize_bill_id("HB241234") == "hb24-1234"

def test_normalize_senate():
    assert _normalize_bill_id("SB23-050") == "sb23-050"

def test_normalize_spaces():
    assert _normalize_bill_id("HB 24-1234") == "hb24-1234"

def test_normalize_already_lower():
    assert _normalize_bill_id("hb24-1234") == "hb24-1234"


# -- _jbc_url --

def test_jbc_url_new_fy_uppercase():
    url = _jbc_url("2026-27", "edu")
    assert "/FY2026-27_edufig1.pdf" in url

def test_jbc_url_old_fy_lowercase():
    url = _jbc_url("2025-26", "hhs")
    assert "/fy2025-26_hhsfig1.pdf" in url

def test_jbc_url_brf_type():
    url = _jbc_url("2024-25", "trs", "brf")
    assert "brf" in url


# -- _find_recent_fiscal_note_url --

def _fiscal_html():
    dq = chr(34)
    return (
        f"<div><button data-url={dq}/bill_files/999/download{dq}>"
        "<div>Recent Fiscal Note</div></button></div>"
    )


def _no_fis_html():
    return "<div><a href=/intro>Introduced Bill</a></div>"


def test_find_fiscal_note_from_button():
    url = _find_recent_fiscal_note_url(_fiscal_html())
    assert url == "https://leg.colorado.gov/bill_files/999/download"

def test_find_fiscal_note_none_when_absent():
    assert _find_recent_fiscal_note_url(_no_fis_html()) is None


# -- get_bill_details (mocked) --

def _make_bill_html():
    dq = chr(34)
    return (
        "<html><body><h1>CONCERNING APPROPRIATIONS</h1>"
        f"<button data-url={dq}/bill_files/500/download{dq}>"
        "<div>Recent Fiscal Note</div></button>"
        f"<button data-url={dq}/bill_files/501/download{dq}>"
        "<div>Introduced Bill</div></button></body></html>"
    )


def _mock_get_bill(url, **kwargs):
    r = mock.MagicMock()
    r.text = _make_bill_html()
    r.raise_for_status = mock.MagicMock()
    return r


def test_bill_details_returns_json():
    with mock.patch("servers.legislature.httpx.get", side_effect=_mock_get_bill):
        result = get_bill_details("HB24-1234")
    data = json.loads(result)
    assert data["bill_id"] == "HB24-1234"
    assert "url" in data
    assert data["fiscal_note_url"] is not None

def test_bill_details_http_error():
    import httpx as _h
    with mock.patch(
        "servers.legislature.httpx.get",
        side_effect=_h.HTTPError("404")
    ):
        result = get_bill_details("HB24-9999")
    assert "BILL_FETCH_FAILED" in result


# -- get_fiscal_note (mocked) --

def test_fiscal_note_no_link():
    def _mock_no_link(url, **kwargs):
        r = mock.MagicMock()
        r.text = "<html><body><h1>HB Test</h1></body></html>"
        r.raise_for_status = mock.MagicMock()
        return r
    with mock.patch("servers.legislature.httpx.get", side_effect=_mock_no_link):
        result = get_fiscal_note("HB24-0001")
    assert "No fiscal note found" in result

def test_fiscal_note_bill_fetch_error():
    import httpx as _h
    with mock.patch(
        "servers.legislature.httpx.get",
        side_effect=_h.HTTPError("conn err")
    ):
        result = get_fiscal_note("SB24-0001")
    assert "BILL_FETCH_FAILED" in result


# -- find_appropriations_documents --

def test_find_approps_known_dept():
    with mock.patch("servers.legislature.httpx.head") as mock_head:
        mock_head.return_value.status_code = 200
        result = find_appropriations_documents("2025-26", "education")
    data = json.loads(result)
    assert data["fiscal_year"] == "2025-26"
    assert "edu" in data["jbc_figure_setting_pdf"]

def test_find_approps_unknown_dept():
    result = find_appropriations_documents("2025-26", "nonexistent_dept_xyz")
    assert "Unknown department" in result

def test_find_approps_404_warning():
    with mock.patch("servers.legislature.httpx.head") as mock_head:
        mock_head.return_value.status_code = 404
        result = find_appropriations_documents("2025-26", "transportation")
    data = json.loads(result)
    assert "warning" in data
