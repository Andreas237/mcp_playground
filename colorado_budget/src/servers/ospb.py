"""
Colorado OSPB (Office of State Planning & Budgeting) MCP server.

Provides structured access to ospb.colorado.gov for:
- Governor's annual budget requests (submitted to legislature each November)
- Quarterly revenue forecasts
- Budget amendments and decision items

The Governor's budget request is the "ask" side of appropriations; the JBC
figure-setting documents (legislature server) are the "approved" side.
Comparing the two reveals politically significant add/cut decisions.

Run standalone:
    python servers/ospb.py               # streamable-http on port 8004
    python servers/ospb.py --port 8004

Requires EXA_API_KEY for search_ospb (same key as other servers).
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

mcp = FastMCP("colorado-ospb")

TIMEOUT = 20
OSPB_BASE = "https://ospb.colorado.gov"
_UA = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}

# ---------------------------------------------------------------------------
# Known page URLs — scraped dynamically; these are starting points
# ---------------------------------------------------------------------------
_BUDGET_PAGE = f"{OSPB_BASE}/governor-budget"
_FORECAST_PAGE = f"{OSPB_BASE}/economic-outlook-revenue-forecast"
_AMENDMENTS_PAGE = f"{OSPB_BASE}/budget-amendments"

# Fiscal year → known PDF filename fragments (best-effort; Exa fills gaps)
# Pattern: ospb.colorado.gov/sites/default/files/<filename>
_KNOWN_BUDGET_FILES: dict[str, list[str]] = {
    "2026-27": ["FY2027GovernorsBudget", "FY2026-27GovernorsBudget", "fy2026-27gb"],
    "2025-26": ["FY2026GovernorsBudget", "FY2025-26GovernorsBudget", "fy2025-26gb"],
    "2024-25": ["FY2025GovernorsBudget", "FY2024-25GovernorsBudget", "fy2024-25gb"],
    "2023-24": ["FY2024GovernorsBudget", "FY2023-24GovernorsBudget", "fy2023-24gb"],
}

# Quarter labels Colorado uses in their forecast filenames/titles
_QUARTERS = {
    "march":     ["March", "march", "Mar"],
    "june":      ["June",  "june",  "Jun"],
    "september": ["September", "september", "Sept", "sep"],
    "december":  ["December",  "december",  "Dec"],
}


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


def _extract_pdf_links(html: str, base: str = OSPB_BASE) -> list[dict]:
    """Extract all PDF links from an HTML page with their anchor text."""
    results = []
    seen: set[str] = set()

    # Match <a href="...pdf...">label</a>
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

    # Also catch data-href / data-url patterns
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


def _fy_variants(fiscal_year: str) -> list[str]:
    """Return common fiscal-year string variants used in filenames."""
    # "2026-27" → ["2026-27", "2027", "26-27", "FY2027", "FY2026-27"]
    m = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if not m:
        return [fiscal_year]
    y1, y2 = m.group(1), m.group(2)
    y1_short = y1[2:]
    end_year = str(int(y1) + 1)
    return [
        fiscal_year,            # "2026-27"
        f"{y1_short}-{y2}",     # "26-27"
        end_year,               # "2027"
        f"FY{end_year}",        # "FY2027"
        f"FY{fiscal_year}",     # "FY2026-27"
        f"fy{y1_short}-{y2}",   # "fy26-27"
    ]


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_ospb(topic: str, fiscal_year: str = "") -> str:
    """Search ospb.colorado.gov for budget documents and analysis.

    Uses Exa neural search restricted to the OSPB site.
    Returns pages/documents matching the topic with URLs and excerpts.

    The OSPB publishes:
    - Governor's annual budget request (November each year)
    - Quarterly revenue and economic forecasts
    - Decision items (agency-specific budget justifications)
    - Budget amendments and supplemental appropriations
    - Long-range financial planning documents

    Use this to find relevant OSPB documents, then pass the URL to
    fetch_and_parse_pdf to extract specific numbers.

    Args:
        topic: What to search for — e.g., "Governor budget request education",
               "revenue forecast General Fund", "Medicaid decision item",
               "supplemental appropriation", "OSPB long-range financial plan"
        fiscal_year: Optional fiscal year to narrow results: "2026-27", "2025-26"
    """
    exa = _get_exa()
    query = f"Colorado OSPB {topic}"
    if fiscal_year:
        query += f" FY{fiscal_year}"
    try:
        results = exa.search(
            query,
            type="auto",
            num_results=10,
            include_domains=["ospb.colorado.gov"],
            contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results.results:
        return (
            f"No OSPB results for '{topic}'. Try broader terms or "
            "find_governor_budget / find_revenue_forecast for direct PDF discovery."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)

    logger.info(f"ospb search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_governor_budget(fiscal_year: str) -> str:
    """Find PDF URLs for the Colorado Governor's annual budget request.

    The Governor's budget is submitted to the Joint Budget Committee each
    November. It contains department-by-department funding requests broken
    down by fund type (General Fund, Cash Funds, Federal Funds), plus
    "Decision Items" — specific new requests or cuts the Governor proposes.

    Comparing the Governor's request to the final JBC appropriation (via
    find_appropriations_documents on the legislature server) reveals the
    politically significant add/cut decisions made by the legislature.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26',
                     '2024-25', '2023-24'. Covers roughly FY2020-21 onward.
    """
    variants = _fy_variants(fiscal_year)

    # Try scraping the OSPB governor-budget page first
    found: list[dict] = []
    html = _fetch_page(_BUDGET_PAGE)
    if html:
        all_pdfs = _extract_pdf_links(html)
        # Filter to PDFs that match the fiscal year
        for link in all_pdfs:
            text = (link["label"] + " " + link["url"]).lower()
            if any(v.lower() in text for v in variants):
                found.append(link)

    # If page scrape found nothing, try Exa
    if not found:
        try:
            exa = _get_exa()
            results = exa.search(
                f"Colorado Governor budget request FY{fiscal_year} PDF filetype",
                type="auto",
                num_results=5,
                include_domains=["ospb.colorado.gov"],
            )
            for r in results.results:
                if ".pdf" in r.url.lower():
                    found.append({"label": r.title or "", "url": r.url})
        except Exception as e:
            logger.warning(f"Exa fallback failed: {e}")

    # HEAD-check all candidates and flag availability
    checked: list[dict] = []
    for link in found[:8]:
        status = _head_check(link["url"])
        checked.append({**link, "http_status": status, "available": status == 200})

    if not checked:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No Governor's budget PDFs found for FY{fiscal_year}. "
                "Try search_ospb with 'Governor budget request' to locate the document, "
                "or fetch ospb.colorado.gov/governor-budget directly with fetch_webpage."
            ),
            "ospb_budget_page": _BUDGET_PAGE,
        }, indent=2)

    result = {
        "fiscal_year": fiscal_year,
        "ospb_budget_page": _BUDGET_PAGE,
        "documents": checked,
        "usage_note": (
            "Pass a document URL to fetch_and_parse_pdf with keyword_filter to extract "
            "a specific department or fund type. Compare with JBC figure-setting PDFs "
            "(find_appropriations_documents) to see Governor request vs. legislature approved."
        ),
    }
    logger.info(f"governor budget: FY{fiscal_year} — {len(checked)} docs found")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_revenue_forecast(year: str = "", quarter: str = "") -> str:
    """Find Colorado quarterly revenue forecast PDFs from OSPB.

    OSPB publishes four revenue forecasts per year (March, June, September,
    December). These are the authoritative projections for General Fund,
    Cash Fund, and total state revenue used by the JBC when setting
    appropriations. Revenue shortfalls often trigger mid-year budget cuts.

    Args:
        year: Calendar year of the forecast: '2025', '2024', '2023'.
              Leave empty to return all recent forecasts found.
        quarter: Quarter of the forecast: 'march', 'june', 'september',
                 'december'. Leave empty to return all quarters for the year.
    """
    # Scrape the OSPB forecasts page
    found: list[dict] = []
    html = _fetch_page(_FORECAST_PAGE)

    # Try alternate page names if main one fails
    if not html:
        for alt in [
            f"{OSPB_BASE}/revenue-forecasts",
            f"{OSPB_BASE}/publications/revenue-forecasts",
            f"{OSPB_BASE}/economic-outlook",
        ]:
            html = _fetch_page(alt)
            if html:
                break

    if html:
        all_pdfs = _extract_pdf_links(html)
        for link in all_pdfs:
            text = (link["label"] + " " + link["url"]).lower()
            # Filter by year if provided
            if year and year not in text:
                continue
            # Filter by quarter if provided
            if quarter:
                q_variants = [v.lower() for v in _QUARTERS.get(quarter.lower(), [quarter])]
                if not any(v in text for v in q_variants):
                    continue
            found.append(link)

    # Exa fallback
    if not found:
        try:
            exa = _get_exa()
            query = "Colorado OSPB revenue forecast"
            if year:
                query += f" {year}"
            if quarter:
                query += f" {quarter}"
            results = exa.search(
                query,
                type="auto",
                num_results=8,
                include_domains=["ospb.colorado.gov"],
            )
            for r in results.results:
                if ".pdf" in r.url.lower():
                    found.append({"label": r.title or "", "url": r.url})
        except Exception as e:
            logger.warning(f"Exa forecast fallback: {e}")

    if not found:
        return json.dumps({
            "found": False,
            "message": (
                "No revenue forecast PDFs located. "
                "Try fetch_webpage on ospb.colorado.gov/economic-outlook-revenue-forecast "
                "to browse available forecasts, or use search_ospb with 'revenue forecast'."
            ),
            "ospb_forecast_page": _FORECAST_PAGE,
        }, indent=2)

    # HEAD-check up to 10
    checked: list[dict] = []
    for link in found[:10]:
        status = _head_check(link["url"])
        checked.append({**link, "http_status": status, "available": status == 200})

    result = {
        "year": year or "all",
        "quarter": quarter or "all",
        "ospb_forecast_page": _FORECAST_PAGE,
        "documents": checked,
        "usage_note": (
            "Pass a document URL to fetch_and_parse_pdf with keyword_filter "
            "such as 'General Fund' or 'total revenue' to extract specific projections."
        ),
    }
    logger.info(f"revenue forecast: {year or 'all'} {quarter or 'all'} — {len(checked)} docs")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_budget_amendments(fiscal_year: str) -> str:
    """Find supplemental budget request and amendment PDFs for a fiscal year.

    The Governor may submit supplemental budget requests mid-year to address
    shortfalls, emergencies, or unanticipated needs. The JBC then passes a
    supplemental appropriations bill. These are distinct from the annual
    Long Bill and often contain politically significant emergency spending.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2025-26', '2024-25'.
    """
    variants = _fy_variants(fiscal_year)
    found: list[dict] = []

    html = _fetch_page(_AMENDMENTS_PAGE)
    if not html:
        html = _fetch_page(f"{OSPB_BASE}/supplemental-budget")
    if html:
        all_pdfs = _extract_pdf_links(html)
        for link in all_pdfs:
            text = (link["label"] + " " + link["url"]).lower()
            if any(v.lower() in text for v in variants):
                found.append(link)

    if not found:
        try:
            exa = _get_exa()
            results = exa.search(
                f"Colorado supplemental budget amendment FY{fiscal_year}",
                type="auto",
                num_results=5,
                include_domains=["ospb.colorado.gov"],
            )
            for r in results.results:
                if ".pdf" in r.url.lower():
                    found.append({"label": r.title or "", "url": r.url})
        except Exception as e:
            logger.warning(f"Exa amendment fallback: {e}")

    if not found:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No amendment PDFs found for FY{fiscal_year}. "
                "Try search_ospb with 'supplemental appropriation' or 'budget amendment'."
            ),
        }, indent=2)

    checked = [
        {**link, "http_status": _head_check(link["url"])}
        for link in found[:6]
    ]
    logger.info(f"budget amendments: FY{fiscal_year} — {len(checked)} docs")
    return json.dumps({"fiscal_year": fiscal_year, "documents": checked}, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-ospb MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
