"""
Colorado Revenue & TABOR MCP server.

Provides structured access to Colorado's revenue side of the budget:
- Legislative Council Staff (LCS) quarterly Economic & Revenue Forecast
- TABOR (Taxpayer's Bill of Rights) revenue limit, surplus, and refund mechanics
- Tax expenditure evaluations / compilation reports (Office of the State Auditor)

This is the revenue counterpart to the OSPB server. Colorado has TWO official
forecasts each quarter: the Governor's (OSPB) and the legislature's (LCS). The
JBC sets the budget off one of them, and the gap between the two is frequently
cited in budget debates. The OSPB server's find_revenue_forecast covers the
Governor's forecast; find_legislative_forecast here covers the legislature's.

TABOR is central to Colorado fiscal politics: it caps state revenue growth and
refunds the surplus to taxpayers, which constrains how much the General Fund
can actually spend even when collections are strong.

Run standalone:
    python servers/revenue.py               # streamable-http on port 8005
    python servers/revenue.py --port 8005

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

mcp = FastMCP("colorado-revenue")

TIMEOUT = 20
LEG_BASE = "https://content.leg.colorado.gov"
_UA = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}

# ---------------------------------------------------------------------------
# Known landing pages — scraped dynamically; these are starting points
# ---------------------------------------------------------------------------
_FORECAST_PAGE = f"{LEG_BASE}/EconomicForecasts"
_FORECASTING_AGENCY_PAGE = f"{LEG_BASE}/agencies/legislative-council-staff/forecasting"
_TAX_EXPENDITURE_PAGE = "https://leg.colorado.gov/agencies/office-state-auditor/tax-expenditure-evaluations"
_TABOR_LIMIT_PAGE = f"{LEG_BASE}/publications/tabor-revenue-limit"

# Curated TABOR reference pages (mechanics + refund status). Returned by
# find_tabor_resources alongside live Exa results.
_TABOR_REFERENCES = [
    {"label": "The TABOR Revenue Limit (Legislative Council Staff)", "url": _TABOR_LIMIT_PAGE},
    {"label": "TABOR Refund (Department of Revenue)", "url": "https://tax.colorado.gov/tabor-refund"},
    {"label": "TABOR Frequently Asked Questions (Department of Revenue)", "url": "https://tax.colorado.gov/tabor-faqs"},
]

# Forecasts are published quarterly: March, June, September, December.
_FORECAST_MONTHS = {
    "march":     ["March", "march", "Mar", "mar"],
    "june":      ["June",  "june",  "Jun", "jun"],
    "september": ["September", "september", "Sept", "sept", "Sep", "sep"],
    "december":  ["December",  "december",  "Dec", "dec"],
}

# Exa domains relevant to revenue/TABOR/tax questions
_REVENUE_DOMAINS = ["leg.colorado.gov", "tax.colorado.gov", "cdor.colorado.gov"]


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
    """Fetch a page, return HTML text or None on error."""
    try:
        r = httpx.get(url, headers=_UA, timeout=TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning(f"fetch {url}: {e}")
        return None


def _extract_pdf_links(html: str, base: str = LEG_BASE) -> list[dict]:
    """Extract all PDF links from an HTML page with their anchor text."""
    results = []
    seen: set[str] = set()

    for m in re.finditer(
        r'<a[^>]+href="([^"]*\.pdf[^"]*)"[^>]*>\s*([^<]{1,120})',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        raw_url = m.group(1).strip()
        label = re.sub(r"\s+", " ", m.group(2)).strip()
        url = raw_url if raw_url.startswith("http") else base + raw_url
        if url not in seen:
            seen.add(url)
            results.append({"label": label, "url": url})

    for m in re.finditer(
        r'(?:data-href|data-url)="([^"]*\.pdf[^"]*)"',
        html,
        re.IGNORECASE,
    ):
        url = m.group(1).strip()
        if not url.startswith("http"):
            url = base + url
        if url not in seen:
            seen.add(url)
            results.append({"label": "", "url": url})

    return results


def _head_check(url: str) -> int:
    """Return HTTP status for url, or 0 on connection error."""
    try:
        r = httpx.head(url, headers=_UA, timeout=8, follow_redirects=True)
        return r.status_code
    except Exception:
        return 0


def _exa_pdf_search(query: str, num_results: int = 8) -> list[dict]:
    """Run an Exa search restricted to revenue domains; return PDF links."""
    found: list[dict] = []
    try:
        exa = _get_exa()
        results = exa.search(
            query,
            type="auto",
            num_results=num_results,
            include_domains=_REVENUE_DOMAINS,
        )
        for r in results.results:
            if ".pdf" in r.url.lower():
                found.append({"label": r.title or "", "url": r.url})
    except Exception as e:
        logger.warning(f"Exa fallback failed: {e}")
    return found


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_revenue(topic: str, year: str = "") -> str:
    """Search Colorado revenue, TABOR, and tax sources via Exa neural search.

    Searches across leg.colorado.gov (Legislative Council forecasts, Office of
    the State Auditor tax evaluations), tax.colorado.gov, and cdor.colorado.gov
    (Department of Revenue). Use this to locate documents on revenue projections,
    TABOR surplus/refunds, tax collections, and tax expenditures, then pass a
    resulting URL to fetch_and_parse_pdf or fetch_webpage to extract numbers.

    Args:
        topic: What to search for — e.g., "General Fund revenue forecast",
               "TABOR surplus refund mechanism", "sales tax collections",
               "tax expenditure income tax credit", "severance tax revenue"
        year: Optional calendar/fiscal year to narrow results: "2025", "2026"
    """
    exa = _get_exa()
    query = f"Colorado {topic}"
    if year:
        query += f" {year}"
    try:
        results = exa.search(
            query,
            type="auto",
            num_results=10,
            include_domains=_REVENUE_DOMAINS,
            contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results.results:
        return (
            f"No results for '{topic}'. Try broader terms, or use "
            "find_legislative_forecast / find_tax_expenditure_report / "
            "find_tabor_resources for direct document discovery."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)

    logger.info(f"revenue search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_legislative_forecast(year: str = "", quarter: str = "") -> str:
    """Find the Legislative Council Staff (LCS) Economic & Revenue Forecast PDFs.

    Colorado publishes TWO official forecasts each quarter (March, June,
    September, December): the Governor's (OSPB) and the legislature's (this one,
    from Legislative Council Staff). The LCS forecast projects General Fund
    revenue, the TABOR revenue limit and projected surplus/refund, and cash fund
    revenue. The JBC relies on these projections when setting appropriations; a
    downward revision often forces mid-year budget cuts.

    To compare the two forecasts (a frequently cited gap), use this alongside
    find_revenue_forecast on the OSPB server.

    Args:
        year: Calendar year of the forecast: '2026', '2025', '2024'.
              Leave empty to return all recent forecasts found.
        quarter: 'march', 'june', 'september', or 'december'.
                 Leave empty to return all quarters for the year.
    """
    found: list[dict] = []
    html = _fetch_page(_FORECAST_PAGE) or _fetch_page(_FORECASTING_AGENCY_PAGE)

    if html:
        for link in _extract_pdf_links(html):
            text = (link["label"] + " " + link["url"]).lower()
            if "forecast" not in text and "forecast" not in link["url"].lower():
                # Keep only forecast-looking PDFs
                if not re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s*20\d\d", text):
                    continue
            if year and year not in text:
                continue
            if quarter:
                q_variants = [v.lower() for v in _FORECAST_MONTHS.get(quarter.lower(), [quarter])]
                if not any(v in text for v in q_variants):
                    continue
            found.append(link)

    if not found:
        query = "Colorado Legislative Council economic revenue forecast"
        if quarter:
            query += f" {quarter}"
        if year:
            query += f" {year}"
        found = _exa_pdf_search(query)

    if not found:
        return json.dumps({
            "found": False,
            "message": (
                "No Legislative Council forecast PDFs located. "
                f"Browse {_FORECAST_PAGE} with fetch_webpage, or use search_revenue "
                "with 'economic and revenue forecast'."
            ),
            "forecast_page": _FORECAST_PAGE,
        }, indent=2)

    checked: list[dict] = []
    for link in found[:10]:
        status = _head_check(link["url"])
        checked.append({**link, "http_status": status, "available": status == 200})

    result = {
        "year": year or "all",
        "quarter": quarter or "all",
        "forecast_page": _FORECAST_PAGE,
        "documents": checked,
        "usage_note": (
            "Pass a document URL to fetch_and_parse_pdf with keyword_filter such as "
            "'General Fund', 'TABOR', or 'surplus' to extract specific projections. "
            "Compare with the OSPB forecast (find_revenue_forecast) to see the gap "
            "between the legislature's and the Governor's revenue expectations."
        ),
    }
    logger.info(f"LCS forecast: {year or 'all'} {quarter or 'all'} — {len(checked)} docs")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_tax_expenditure_report(year: str = "") -> str:
    """Find Colorado tax expenditure evaluations / compilation reports.

    Tax expenditures are credits, deductions, and exemptions that reduce state
    revenue — effectively spending delivered through the tax code. The Office of
    the State Auditor publishes evaluations of individual tax expenditures and an
    annual Tax Expenditures Compilation Report estimating the total revenue
    forgone. These figures are politically significant: they are 'spending' that
    never appears in the Long Bill.

    Args:
        year: Calendar year of the report: '2025', '2024', '2023'.
              Leave empty to return the most recent reports found.
    """
    found: list[dict] = []
    html = _fetch_page(_TAX_EXPENDITURE_PAGE)
    if html:
        for link in _extract_pdf_links(html, base="https://leg.colorado.gov"):
            text = (link["label"] + " " + link["url"]).lower()
            if year and year not in text:
                continue
            found.append(link)

    if not found:
        query = "Colorado tax expenditure compilation evaluation report Office of State Auditor"
        if year:
            query += f" {year}"
        found = _exa_pdf_search(query)

    if not found:
        return json.dumps({
            "found": False,
            "message": (
                "No tax expenditure reports located. "
                f"Browse {_TAX_EXPENDITURE_PAGE} with fetch_webpage, or use "
                "search_revenue with 'tax expenditure compilation report'."
            ),
            "tax_expenditure_page": _TAX_EXPENDITURE_PAGE,
        }, indent=2)

    checked: list[dict] = []
    for link in found[:10]:
        status = _head_check(link["url"])
        checked.append({**link, "http_status": status, "available": status == 200})

    result = {
        "year": year or "recent",
        "tax_expenditure_page": _TAX_EXPENDITURE_PAGE,
        "documents": checked,
        "usage_note": (
            "The compilation report aggregates total revenue forgone; individual "
            "evaluations cover a single credit/deduction. Pass a URL to "
            "fetch_and_parse_pdf with a keyword_filter for the specific expenditure."
        ),
    }
    logger.info(f"tax expenditure: {year or 'recent'} — {len(checked)} docs")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_tabor_resources(topic: str = "") -> str:
    """Find authoritative TABOR (Taxpayer's Bill of Rights) reference material.

    TABOR caps the annual growth of state revenue (to population + inflation) and
    requires the surplus above that cap to be refunded to taxpayers. This limits
    how much the state can actually spend even in strong revenue years, and the
    refund mechanism (sales tax refund, income tax rate reduction, etc.) is a
    recurring political flashpoint.

    Returns curated explainer pages on the revenue limit and refund mechanics,
    plus live search results. For the dollar figures of the current projected
    surplus/refund, use find_legislative_forecast — every quarterly forecast
    includes an updated TABOR surplus projection.

    Args:
        topic: Optional focus — e.g., "refund mechanism", "revenue limit Referendum C",
               "surplus projection", "TABOR cap calculation". Leave empty for the
               core reference pages.
    """
    resources = list(_TABOR_REFERENCES)

    # Add live Exa results for the specific topic
    if topic:
        try:
            exa = _get_exa()
            results = exa.search(
                f"Colorado TABOR {topic}",
                type="auto",
                num_results=6,
                include_domains=_REVENUE_DOMAINS,
                contents={"highlights": True},
            )
            for r in results.results:
                entry: dict = {"label": r.title or "", "url": r.url}
                if r.highlights:
                    entry["highlights"] = r.highlights[:2]
                resources.append(entry)
        except Exception as e:
            logger.warning(f"TABOR search failed: {e}")

    result = {
        "topic": topic or "TABOR overview",
        "resources": resources,
        "usage_note": (
            "These pages explain how TABOR works. For current surplus/refund dollar "
            "amounts, call find_legislative_forecast — the latest quarterly forecast "
            "contains the up-to-date TABOR surplus projection."
        ),
    }
    logger.info(f"TABOR resources: '{topic or 'overview'}' — {len(resources)} entries")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-revenue MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
