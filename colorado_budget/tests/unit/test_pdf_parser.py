"""
Unit tests for tools/pdf_parser.py.
HTTP and pdfplumber are both mocked — no network or real PDFs required.
"""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tools.pdf_parser import fetch_and_parse_pdf


def _make_mock_page(text: str, tables=None):
    page = MagicMock()
    page.extract_text.return_value = text
    page.extract_tables.return_value = tables or []
    return page


def _mock_pdf_response(content=b"%PDF-1.4 fake", content_type="application/pdf"):
    mock = MagicMock()
    mock.status_code = 200
    mock.content = content
    mock.headers = {"content-type": content_type}
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# HTTP-level failures
# ---------------------------------------------------------------------------

def test_http_error_returns_fetch_failed():
    import httpx
    with patch("tools.pdf_parser.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            ))
        )
        result = fetch_and_parse_pdf(url="https://example.com/doc.pdf")
    assert result.startswith("PDF_FETCH_FAILED")


def test_non_pdf_content_type_returns_redirect_warning():
    with patch("tools.pdf_parser.httpx.get") as mock_get:
        mock_get.return_value = _mock_pdf_response(content_type="text/html", content=b"<html>portal</html>")
        result = fetch_and_parse_pdf(url="https://example.com/not-a-pdf")
    assert "portal redirect" in result or "content-type" in result


# ---------------------------------------------------------------------------
# Normal extraction
# ---------------------------------------------------------------------------

def test_extracts_text_from_pages():
    mock_resp = _mock_pdf_response()
    pages = [
        _make_mock_page("This is page one about the General Fund appropriation."),
        _make_mock_page("This is page two about Cash Funds."),
    ]

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = pages
            mock_open.return_value = mock_pdf

            result = fetch_and_parse_pdf(url="https://example.com/budget.pdf")

    assert "General Fund" in result
    assert "Cash Funds" in result
    assert "Page 1" in result
    assert "Page 2" in result


def test_keyword_filter_returns_only_matching_pages():
    mock_resp = _mock_pdf_response()
    pages = [
        _make_mock_page("This page has nothing relevant."),
        _make_mock_page("This page mentions the Colorado School for the Deaf and Blind CSDB funding."),
        _make_mock_page("Unrelated page about CDOT."),
    ]

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = pages
            mock_open.return_value = mock_pdf

            result = fetch_and_parse_pdf(url="https://example.com/budget.pdf", keyword_filter="CSDB")

    assert "Colorado School for the Deaf" in result
    assert "nothing relevant" not in result
    assert "CDOT" not in result


def test_keyword_filter_no_match_returns_no_text_message():
    mock_resp = _mock_pdf_response()
    pages = [_make_mock_page("Page with unrelated content.")]

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = pages
            mock_open.return_value = mock_pdf

            result = fetch_and_parse_pdf(url="https://example.com/budget.pdf", keyword_filter="CSDB")

    assert "No text extracted" in result


def test_page_range_limits_pages():
    mock_resp = _mock_pdf_response()
    # 10 pages, ask for pages 3-5
    pages = [_make_mock_page(f"Content of page {i+1}.") for i in range(10)]

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = pages
            mock_open.return_value = mock_pdf

            result = fetch_and_parse_pdf(url="https://example.com/budget.pdf", page_range="3-5")

    assert "page 3" in result.lower() or "Page 3" in result
    assert "Content of page 1" not in result
    assert "Content of page 10" not in result


def test_table_extraction_included_in_output():
    mock_resp = _mock_pdf_response()
    table = [["Agency", "FY 2023", "FY 2024"], ["CSDB", "$10M", "$11M"]]
    pages = [_make_mock_page("Budget summary.", tables=[table])]

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = pages
            mock_open.return_value = mock_pdf

            result = fetch_and_parse_pdf(url="https://example.com/budget.pdf")

    assert "Agency" in result
    assert "CSDB" in result
    assert "$10M" in result


def test_parse_error_returns_error_string():
    mock_resp = _mock_pdf_response()

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open", side_effect=Exception("corrupt PDF")):
            result = fetch_and_parse_pdf(url="https://example.com/corrupt.pdf")

    assert result.startswith("PDF_PARSE_ERROR")
    assert "corrupt PDF" in result


def test_large_result_is_truncated():
    mock_resp = _mock_pdf_response()
    big_text = "A" * 50000
    pages = [_make_mock_page(big_text)]

    with patch("tools.pdf_parser.httpx.get", return_value=mock_resp):
        with patch("tools.pdf_parser.pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
            mock_pdf.__exit__ = MagicMock(return_value=False)
            mock_pdf.pages = pages
            mock_open.return_value = mock_pdf

            result = fetch_and_parse_pdf(url="https://example.com/big.pdf")

    assert len(result) <= 41000  # 40000 chars + truncation notice
    assert "truncated" in result
