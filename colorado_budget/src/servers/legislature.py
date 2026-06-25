"""
Colorado Legislature MCP server.

Provides structured access to leg.colorado.gov for bill search,
bill details, fiscal notes (fund-type tables), and JBC/Long Bill document URLs.

Run standalone:
    python servers/legislature.py                    # streamable-http on port 8003
    python servers/legislature.py --port 8003
    python servers/legislature.py --transport stdio

Requires EXA_API_KEY for search_bills (same key as web_search server).
"""
import argparse
import io
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import httpx
import pdfplumber
from loguru import logger
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_api_keys

mcp = FastMCP("colorado-legislature")

TIMEOUT = 30
BASE = "https://leg.colorado.gov"

# ---------------------------------------------------------------------------
# JBC document URL catalogue — fiscal year → department → PDF URL
# Populated from known-working URLs discovered during agent smoke tests.
# Department codes: edu, hhs (health+human services), trs (transportation),
#   nat (natural resources), gg (general government), pub (public safety),
#   jud (judicial), leg (legislative), cor (corrections)
# ---------------------------------------------------------------------------
_JBC_BASE = "https://content.leg.colorado.gov/sites/default/files"

_JBC_DEPT_CODES = {
    "education": "edu",
    "health_human_services": "hhs",
    "transportation": "trs",
    "natural_resources": "nat",
    "general_government": "gg",
    "public_safety": "pub",
    "judicial": "jud",
    "legislative": "leg",
    "corrections": "cor",
}

# FY ≥ 2026-27 uses uppercase "FY"; older years use lowercase "fy"
def _jbc_url(fiscal_year: str, dept_code: str, doc_type: str = "fig") -> str:
    prefix = "FY" if fiscal_year >= "2026" else "fy"
    return f"{_JBC_BASE}/{prefix}{fiscal_year}_{dept_code}{doc_type}1.pdf"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _normalize_bill_id(bill_id: str) -> str:
    """'HB24-1234' → 'hb24-1234', 'SB 21 050' → 'sb21-050'"""
    clean = bill_id.lower().strip()
    clean = re.sub(r"[\s_]", "", clean)
    # Insert hyphen if missing: 'hb211271' → 'hb21-1271'
    if re.match(r"^[a-z]{2}\d{2}\d{3,4}$", clean):
        clean = clean[:4] + "-" + clean[4:]
    return clean


def _fetch_bill_page(bill_id: str) -> tuple[str, str]:
    """Returns (normalized_id, html_text). Raises httpx.HTTPError on failure."""
    norm = _normalize_bill_id(bill_id)
    url = f"{BASE}/bills/{norm}"
    r = httpx.get(url, headers={"User-Agent": "ColoradoBudgetResearchAgent/1.0"},
                  timeout=TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    return norm, r.text


def _extract_bill_file_links(html: str) -> list[dict]:
    """Find all /bill_files/{id}/download links with their labels."""
    results = []
    # Pattern: aria-label or nearby text + href /bill_files/N/download
    for m in re.finditer(
        r'(?:aria-label="([^"]*)"[^>]*|data-url=")(/bill_files/(\d+)/download)',
        html,
    ):
        label = m.group(1) or ""
        path = m.group(2)
        results.append({"label": label.strip(), "path": path, "url": BASE + path})
    # Also grab the "Recent Fiscal Note" button label pattern
    for m in re.finditer(
        r'data-url="(/bill_files/(\d+)/download)"[^>]*>.*?<div[^>]*>\s*([^<]+)',
        html,
        re.DOTALL,
    ):
        label = m.group(3).strip()
        path = m.group(1)
        if not any(r["path"] == path for r in results):
            results.append({"label": label, "path": path, "url": BASE + path})
    return results


def _find_recent_fiscal_note_url(html: str) -> Optional[str]:
    """Return the URL for the most recent fiscal note, or None."""
    # Primary: aria-label and data-url on the SAME element
    m = re.search(
        r'aria-label="([^"]*Recent Fiscal Note[^"]*)"[^>]*data-url="(/bill_files/(\d+)/download)"',
        html,
        re.IGNORECASE,
    )
    if m:
        return BASE + m.group(2)
    # Fallback: data-url whose download link is followed closely by the label text
    # (no other data-url= may appear between the two)
    m = re.search(
        r'data-url="(/bill_files/(\d+)/download)"(?:(?!data-url=).)*?Recent Fiscal Note',
        html,
        re.DOTALL,
    )
    if m:
        return BASE + m.group(1)
    # Fallback: aria-label containing 'fiscal'
    links = _extract_bill_file_links(html)
    fis_links = [l for l in links if "fiscal" in l["label"].lower()]
    return fis_links[0]["url"] if fis_links else None


def _parse_pdf_text(content: bytes, url: str, keyword_filter: str = "") -> str:
    """Extract text (+ tables) from PDF bytes. Returns formatted string."""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            total = len(pdf.pages)
            pages = pdf.pages[:30]
            sections = []
            for i, page in enumerate(pages):
                text = page.extract_text() or ""
                if keyword_filter and keyword_filter.lower() not in text.lower():
                    continue
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cleaned = [str(cell or "").strip() for cell in row]
                        if any(cleaned):
                            table_text += " | ".join(cleaned) + "\n"
                combined = text
                if table_text:
                    combined += f"\n\n[Tables]\n{table_text}"
                if combined.strip():
                    sections.append(f"--- Page {i+1} ---\n{combined.strip()}")
            if not sections:
                kw = f" containing '{keyword_filter}'" if keyword_filter else ""
                return f"No text extracted from PDF{kw}. May be scanned images."
            result = f"PDF: {url}\nTotal pages: {total}\nExtracted: {len(sections)} page(s)\n\n"
            result += "\n\n".join(sections)
            if len(result) > 40000:
                result = result[:40000] + "\n\n[truncated — use keyword_filter to target specific sections]"
            return result
    except Exception as e:
        return f"PDF_PARSE_ERROR: {e}"


def _get_exa():
    from exa_py import Exa
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY not set. Add it to colorado_budget/.env")
    return Exa(api_key=key)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_bills(topic: str, year: str = "", limit: int = 8) -> str:
    """Search for Colorado bills on leg.colorado.gov by topic.

    Uses Exa neural search restricted to the official Colorado General Assembly site.
    Returns bill IDs, titles, URLs, and relevant text excerpts.

    Use this to find which bills addressed a specific policy area, then call
    get_bill_details or get_fiscal_note to verify claims about those bills.

    Args:
        topic: Policy topic to search for (e.g. "affordable housing zoning reform",
               "CSDB Colorado School for the Deaf and Blind", "Medicaid mental health",
               "oil and gas regulation", "school finance formula")
        year: Optional 2-digit legislative year to filter (e.g. "21" for 2021,
              "24" for 2024). Leave empty to search all years.
        limit: Number of results to return (default 8, max 20)
    """
    exa = _get_exa()
    query = f"Colorado {topic} bill legislation site:leg.colorado.gov"
    if year:
        query += f" 20{year}"
    try:
        results = exa.search(
            f"Colorado {topic} bill",
            type="auto",
            num_results=min(limit, 20),
            include_domains=["leg.colorado.gov"],
            contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    output = []
    for r in results.results:
        bill_match = re.search(r"/bills/([a-z]{2}\d{2}-\d+)", r.url, re.IGNORECASE)
        entry: dict = {
            "title": r.title,
            "url": r.url,
            "bill_id": bill_match.group(1).upper() if bill_match else None,
        }
        if r.highlights:
            entry["highlights"] = r.highlights[:3]
        output.append(entry)

    if not output:
        return f"No bills found for '{topic}'. Try broader keywords or search_colorado_government."

    logger.info(f"bill search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def get_bill_details(bill_id: str) -> str:
    """Get metadata for a specific Colorado bill: title, sponsors, status, and document links.

    Returns the bill's title, primary sponsors, available document versions
    (introduced, engrossed, signed act), and the URL for the most recent fiscal note.

    Call this after search_bills to confirm the bill matches your research question,
    then call get_fiscal_note to extract the fund-type financial breakdown.

    Args:
        bill_id: Bill identifier in any common format:
                 'HB24-1234', 'SB23-050', 'hb24-1234', 'HB241234'
    """
    try:
        norm, html = _fetch_bill_page(bill_id)
    except httpx.HTTPError as e:
        return f"BILL_FETCH_FAILED: {e}. Check the bill ID format (e.g. 'HB24-1234')."

    bill_url = f"{BASE}/bills/{norm}"

    # Title
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Unknown"

    # Sponsors — filter out email addresses and generic labels
    sponsor_raw = re.findall(r"sponsor[^>]*>.*?<a[^>]+>([^<]+)</a>", html, re.IGNORECASE | re.DOTALL)
    sponsors = [s.strip() for s in sponsor_raw if "@" not in s and s.strip() not in ("PDF", "")]

    # Bill document links (all versions)
    file_links = _extract_bill_file_links(html)
    bill_docs = [l for l in file_links if "fiscal" not in l["label"].lower()]
    fis_url = _find_recent_fiscal_note_url(html)

    # Signed/enacted status
    signed = any("signed" in l["label"].lower() or "act" in l["label"].lower() for l in bill_docs)

    result = {
        "bill_id": norm.upper(),
        "url": bill_url,
        "title": title,
        "sponsors": sponsors[:6],
        "signed_into_law": signed,
        "fiscal_note_url": fis_url,
        "bill_versions_available": [l["label"] for l in bill_docs if l["label"]][:8],
    }

    logger.info(f"bill details: {norm.upper()} — {title[:60]}")
    return json.dumps(result, indent=2)


@mcp.tool()
def get_fiscal_note(bill_id: str, keyword_filter: str = "") -> str:
    """Download and extract the fiscal note PDF for a Colorado bill.

    Fiscal notes show the projected fiscal impact of a bill broken down by fund type:
    General Fund, Cash Funds, Federal Funds, Reappropriated Funds, and TOTAL.
    They also describe which state agencies are affected and the FTE impact.

    This is the primary tool for verifying political claims about a bill's cost
    and which budget fund it draws from.

    Args:
        bill_id: Bill identifier: 'HB24-1234', 'SB23-050', 'hb21-1271'
        keyword_filter: Optional keyword to narrow extracted pages (e.g. "General Fund",
                        "Department of Education", "FTE"). Useful for long fiscal notes.
    """
    try:
        _, html = _fetch_bill_page(bill_id)
    except httpx.HTTPError as e:
        return f"BILL_FETCH_FAILED: {e}. Verify the bill ID format."

    fis_url = _find_recent_fiscal_note_url(html)
    if not fis_url:
        return (
            f"No fiscal note found for {bill_id.upper()}. "
            "The bill may not have a fiscal note (e.g. it was introduced but not advanced), "
            "or the page structure has changed. Try get_bill_details to see available documents."
        )

    logger.info(f"fiscal note: {bill_id.upper()} → {fis_url}")

    try:
        r = httpx.get(
            fis_url,
            headers={"User-Agent": "ColoradoBudgetResearchAgent/1.0"},
            timeout=60,
            follow_redirects=True,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"FISCAL_NOTE_FETCH_FAILED: {e}"

    content_type = r.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not fis_url.lower().endswith(".pdf"):
        return f"Unexpected content type ({content_type}) from {fis_url}. Expected a PDF."

    return _parse_pdf_text(r.content, fis_url, keyword_filter)


@mcp.tool()
def find_appropriations_documents(fiscal_year: str, department: str = "education") -> str:
    """Return PDF URLs for JBC figure-setting and appropriations documents for a fiscal year.

    JBC (Joint Budget Committee) publishes figure-setting documents for each department
    each year. These contain line-item appropriations, year-over-year changes, fund-type
    breakdowns, and JBC staff analysis. Use these with fetch_and_parse_pdf to extract
    specific agency appropriations.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26', '2024-25', etc.
                     Covers FY 2018-19 through FY 2026-27.
        department: Department area — one of:
                    'education', 'health_human_services', 'transportation',
                    'natural_resources', 'general_government', 'public_safety',
                    'judicial', 'legislative', 'corrections'
                    Default: 'education'
    """
    dept_code = _JBC_DEPT_CODES.get(department.lower().replace(" ", "_"))
    if not dept_code:
        return (
            f"Unknown department '{department}'. Valid options: "
            + ", ".join(_JBC_DEPT_CODES.keys())
        )

    # Build document URLs
    fig_url = _jbc_url(fiscal_year, dept_code, "fig")
    brf_url = _jbc_url(fiscal_year, dept_code, "brf")

    # Appropriations history report (different URL pattern)
    # fy24-25apprept.pdf → extract 2-digit years
    yy_match = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if yy_match:
        y1 = yy_match.group(1)[2:]  # e.g. "24"
        y2 = yy_match.group(2)       # e.g. "25"
        appr_url = f"https://leg.colorado.gov/sites/default/files/fy{y1}-{y2}apprept.pdf"
    else:
        appr_url = None

    result = {
        "fiscal_year": fiscal_year,
        "department": department,
        "jbc_figure_setting_pdf": fig_url,
        "jbc_briefing_pdf": brf_url,
        "appropriations_history_report": appr_url,
        "usage_notes": (
            "Pass jbc_figure_setting_pdf to fetch_and_parse_pdf with a keyword_filter "
            "to extract a specific agency's appropriation. The figure-setting document "
            "contains line-item tables with General Fund, Cash Funds, and Federal Funds columns. "
            "The appropriations_history_report covers all departments but is very large."
        ),
    }

    # Quick head-check on the figure-setting URL to warn if it's a 404
    try:
        check = httpx.head(fig_url, timeout=8, follow_redirects=True,
                           headers={"User-Agent": "ColoradoBudgetResearchAgent/1.0"})
        result["jbc_figure_setting_status"] = check.status_code
        if check.status_code == 404:
            result["warning"] = (
                f"Figure-setting PDF returned 404. Try the briefing PDF, or search for "
                f"'JBC {fiscal_year} {department} figure setting' via search_colorado_government."
            )
    except Exception:
        pass

    logger.info(f"appropriations docs: FY{fiscal_year} {department}")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-legislature MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
