"""
Colorado HCPF (Health Care Policy & Financing / Medicaid) MCP server.

HCPF runs Colorado Medicaid and CHP+ and is the single largest General Fund
department. This server covers the STATE budget side of HCPF:
- The department's annual budget request (the "ask" to the JBC)
- Premiums, Expenditures & Caseload (PECR) reports — HCPF's signature monthly
  data on Medicaid enrollment and spending
- JBC figure-setting (the "approved" appropriation)

The FEDERAL Medicaid match is covered separately by the federal-funds server
(USAspending). HCPF's funding is famously split ~50/50 General Fund / Federal
Funds, so the two servers together tell the whole story.

Run standalone:
    python servers/hcpf.py               # streamable-http on port 8008
    python servers/hcpf.py --port 8008

Requires EXA_API_KEY for search (same key as other servers).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_api_keys

mcp = FastMCP("colorado-hcpf")

TIMEOUT = 20
HCPF_BASE = "https://hcpf.colorado.gov"
LEG_BASE = "https://content.leg.colorado.gov"
# hcpf.colorado.gov returns 403 to non-browser User-Agents, so present a
# browser UA (verified to pass) for this server's requests.
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# Known landing pages (Exa-verified) — scraped dynamically as starting points
# ---------------------------------------------------------------------------
_BUDGET_INDEX = f"{HCPF_BASE}/budget-requests"
_BUDGET_PAGE = f"{HCPF_BASE}/budget"
_PECR_PAGE = f"{HCPF_BASE}/premiums-expenditures-and-caseload-reports"

_FINANCE_DOMAINS = ["hcpf.colorado.gov", "leg.colorado.gov"]

# HCPF publishes data as PDF and Excel
_DOC_RE = re.compile(r'<a[^>]+href="([^"]*\.(?:pdf|xlsx?|xls)[^"]*)"[^>]*>\s*([^<]{1,120})',
                     re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_exa():
    from exa_py import Exa
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY not set")
    return Exa(api_key=key)


def _fetch_page(url: str) -> Optional[str]:
    try:
        r = httpx.get(url, headers=_UA, timeout=TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning(f"fetch {url}: {e}")
        return None


def _extract_doc_links(html: str, base: str = HCPF_BASE) -> list[dict]:
    """Extract PDF/Excel document links from an HTML page with their anchor text."""
    results = []
    seen: set[str] = set()
    for m in _DOC_RE.finditer(html):
        raw_url = m.group(1).strip()
        label = re.sub(r"\s+", " ", m.group(2)).strip()
        url = raw_url if raw_url.startswith("http") else base + raw_url
        if url not in seen:
            seen.add(url)
            results.append({"label": label, "url": url})
    return results


def _head_check(url: str) -> int:
    try:
        r = httpx.head(url, headers=_UA, timeout=8, follow_redirects=True)
        return r.status_code
    except Exception:
        return 0


def _exa_results(query: str, num_results: int = 8) -> list[dict]:
    """Exa search restricted to HCPF/leg domains. Returns {label,url,is_document}."""
    out: list[dict] = []
    try:
        exa = _get_exa()
        results = exa.search(
            query, type="auto", num_results=num_results, include_domains=_FINANCE_DOMAINS,
        )
        for r in results.results:
            is_doc = bool(re.search(r"\.(pdf|xlsx?|xls)(\?|$)", r.url, re.IGNORECASE))
            out.append({"label": r.title or "", "url": r.url, "is_document": is_doc})
    except Exception as e:
        logger.warning(f"Exa search failed: {e}")
    return out


def _collect(year: str, scrape_pages: list[str], exa_query: str) -> tuple[list[dict], list[dict]]:
    """Gather downloadable docs (HEAD-checked 200) and reference pages."""
    raw_docs: list[dict] = []
    for page in scrape_pages:
        html = _fetch_page(page)
        if not html:
            continue
        for link in _extract_doc_links(html):
            text = (link["label"] + " " + link["url"]).lower()
            if year and year not in text:
                continue
            raw_docs.append(link)

    pages: list[dict] = []
    for r in _exa_results(exa_query):
        if r["is_document"]:
            raw_docs.append({"label": r["label"], "url": r["url"]})
        else:
            pages.append({"label": r["label"], "url": r["url"]})

    documents: list[dict] = []
    seen: set[str] = set()
    for d in raw_docs:
        if d["url"] in seen:
            continue
        seen.add(d["url"])
        if _head_check(d["url"]) == 200:
            documents.append({**d, "http_status": 200})

    return documents[:10], pages[:6]


def _hcpfig_urls(fiscal_year: str) -> list[str]:
    """Candidate JBC HCPF figure-setting PDF URLs for a fiscal year.

    Naming: FY >= 2026-27 uses uppercase 'FY{year}_hcpfig1.pdf' (sometimes with a
    '_0' suffix); older years use lowercase 'fy{year}_hcpfig1.pdf'.
    """
    m = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if not m:
        return []
    base = f"{LEG_BASE}/sites/default/files"
    return [
        f"{base}/FY{fiscal_year}_hcpfig1.pdf",
        f"{base}/FY{fiscal_year}_hcpfig1_0.pdf",
        f"{base}/fy{fiscal_year}_hcpfig1.pdf",
        f"{base}/fy{fiscal_year}_hcpfig1_0.pdf",
    ]


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_hcpf(topic: str, fiscal_year: str = "") -> str:
    """Search HCPF (Colorado Medicaid) sources via Exa neural search.

    Searches hcpf.colorado.gov and leg.colorado.gov for documents on Medicaid
    budget, enrollment/caseload, expenditures, premiums, provider rates, and
    program changes. Pass a resulting URL to fetch_and_parse_pdf or fetch_webpage
    to extract numbers.

    Args:
        topic: What to search for — e.g., "Medicaid caseload forecast", "provider
               rate increase", "long bill Medicaid expenditures", "CHP+ enrollment",
               "behavioral health budget request".
        fiscal_year: Optional fiscal year to narrow: "2025-26", "2026-27".
    """
    exa = _get_exa()
    query = f"Colorado HCPF Medicaid {topic}"
    if fiscal_year:
        query += f" FY{fiscal_year}"
    try:
        results = exa.search(
            query, type="auto", num_results=10,
            include_domains=_FINANCE_DOMAINS, contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results.results:
        return (
            f"No HCPF results for '{topic}'. Try broader terms, or use "
            "find_hcpf_budget_request / find_caseload_reports / find_hcpf_appropriations."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)
    logger.info(f"hcpf search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_hcpf_budget_request(fiscal_year: str) -> str:
    """Find HCPF's annual budget request documents for a fiscal year.

    HCPF submits its budget request (the department's "ask") to the JBC each
    November, broken out by General Fund, Cash Funds, Reappropriated Funds, and
    Federal Funds. Medicaid caseload and per-capita cost assumptions drive the
    request. Compare with find_hcpf_appropriations to see request vs. approved.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26'.
    """
    pages = [
        f"{HCPF_BASE}/budget/fy-{fiscal_year}-budget-request",
        f"{HCPF_BASE}/budget/fy-{fiscal_year}-budget-requests",
        f"{HCPF_BASE}/budget-requests/fy-{fiscal_year}-requests",
        _BUDGET_INDEX,
    ]
    documents, ref_pages = _collect(
        fiscal_year, pages,
        f"Colorado HCPF Medicaid budget request FY {fiscal_year} General Fund",
    )
    if not documents and not ref_pages:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No HCPF budget request docs found for FY{fiscal_year}. "
                f"Browse {_BUDGET_INDEX} with fetch_webpage, or use search_hcpf "
                "with 'budget request'."
            ),
            "budget_index": _BUDGET_INDEX,
        }, indent=2)
    result = {
        "fiscal_year": fiscal_year,
        "budget_index": _BUDGET_INDEX,
        "documents": documents,
        "pages": ref_pages,
        "usage_note": (
            "Pass a PDF URL to fetch_and_parse_pdf with keyword_filter like 'General Fund' "
            "or 'caseload'. HTML request pages link the individual decision items; fetch "
            "them with fetch_webpage. Compare with find_hcpf_appropriations (JBC approved)."
        ),
    }
    logger.info(f"hcpf budget request FY{fiscal_year}: {len(documents)} docs, {len(ref_pages)} pages")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_caseload_reports(year: str = "", month: str = "") -> str:
    """Find HCPF Premiums, Expenditures & Caseload (PECR) reports.

    The PECR reports are HCPF's signature monthly data: Medicaid and CHP+
    enrollment (caseload) and expenditures by eligibility category. They are the
    authoritative source for "how many Coloradans are on Medicaid" and "what is
    it costing", and they drive the budget request and supplementals.

    Args:
        year: Calendar/fiscal year to filter, e.g. '2025', '2024'.
              Leave empty for the most recent reports.
        month: Optional month name to filter, e.g. 'July', 'January'.
    """
    documents, ref_pages = _collect(
        year, [_PECR_PAGE, f"{HCPF_BASE}/budget/FY-Premiums-Expenditures-Caseload-Reports"],
        f"Colorado HCPF premiums expenditures caseload report {month} {year}".strip(),
    )
    # Month filter applies to the discovered documents/pages
    if month:
        ml = month.lower()
        documents = [d for d in documents if ml in (d["label"] + " " + d["url"]).lower()]
        ref_pages = [p for p in ref_pages if ml in (p["label"] + " " + p["url"]).lower()] or ref_pages

    if not documents and not ref_pages:
        return json.dumps({
            "found": False,
            "message": (
                "No PECR reports located. "
                f"Browse {_PECR_PAGE} with fetch_webpage, or use search_hcpf "
                "with 'premiums expenditures caseload'."
            ),
            "pecr_page": _PECR_PAGE,
        }, indent=2)
    result = {
        "year": year or "recent",
        "month": month or "all",
        "pecr_page": _PECR_PAGE,
        "documents": documents,
        "pages": ref_pages,
        "usage_note": (
            "PECR data is often Excel (.xlsx), which fetch_and_parse_pdf cannot read; "
            "fetch the PECR landing page with fetch_webpage to see the monthly index. "
            "Use caseload for enrollment trends and expenditures for spending."
        ),
    }
    logger.info(f"hcpf caseload: {year or 'recent'} {month or 'all'} — {len(documents)} docs")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_hcpf_appropriations(fiscal_year: str) -> str:
    """Find the JBC HCPF figure-setting (appropriations) document for a year.

    The JBC figure-setting document is the legislature's detailed staff analysis
    and recommended appropriation for HCPF — the "approved" side to pair with the
    department's request (find_hcpf_budget_request). It shows the Medicaid Long
    Bill line items by fund type and the caseload/cost assumptions the JBC adopted.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26'.
    """
    found: list[dict] = []
    for url in _hcpfig_urls(fiscal_year):
        status = _head_check(url)
        if status == 200:
            found.append({"label": f"JBC HCPF Figure Setting FY{fiscal_year}", "url": url,
                          "http_status": 200})

    if not found:
        for r in _exa_results(f"Colorado JBC HCPF figure setting FY{fiscal_year} hcpfig"):
            if r["is_document"]:
                if _head_check(r["url"]) == 200:
                    found.append({"label": r["label"], "url": r["url"], "http_status": 200})

    if not found:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No JBC HCPF figure-setting PDF found for FY{fiscal_year}. "
                "Try search_hcpf with 'figure setting', or the legislature server's "
                "find_appropriations_documents."
            ),
        }, indent=2)

    result = {
        "fiscal_year": fiscal_year,
        "documents": found[:5],
        "usage_note": (
            "Pass the URL to fetch_and_parse_pdf with keyword_filter such as 'General Fund', "
            "'Medical Services Premiums', or 'caseload' to extract the approved figures. "
            "Compare with find_hcpf_budget_request to quantify what the JBC added or cut."
        ),
    }
    logger.info(f"hcpf appropriations FY{fiscal_year}: {len(found)} docs")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-hcpf MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
