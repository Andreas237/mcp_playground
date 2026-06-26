"""
Colorado Department of Agriculture (CDA) MCP server.

CDA is one of Colorado's smaller departments. Its budget is funded largely by
Cash Funds (inspection, registration, and licensing fees from the industries it
regulates), with a modest General Fund share and some federal dollars, plus
periodic targeted General Fund investments (e.g. drought/agricultural resilience
packages). This makes it a clean, tractable case study in fund types.

This server covers:
- CDA's state budget page and budget proposals (the department's funding story)
- JBC Agriculture figure-setting (the appropriation by fund type)
- CDA programs / performance plan (what the money funds)

Run standalone:
    python servers/agriculture.py            # streamable-http on port 8010
    python servers/agriculture.py --port 8010

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

mcp = FastMCP("colorado-agriculture")

TIMEOUT = 20
AG_BASE = "https://ag.colorado.gov"
LEG_HOSTS = ["https://content.leg.colorado.gov", "https://leg.colorado.gov"]
# ag.colorado.gov returns 403 to non-browser User-Agents; present a browser UA.
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
_STATE_BUDGET_PAGE = f"{AG_BASE}/category/state-budget"
_PERFORMANCE_PLAN_PAGE = f"{AG_BASE}/home/about-us/cda-strategic-initiatives/cdas-performance-plan"
_GRANTS_PAGE = f"{AG_BASE}/category/grants"

_FINANCE_DOMAINS = ["ag.colorado.gov", "leg.colorado.gov"]

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


def _extract_doc_links(html: str, base: str = AG_BASE) -> list[dict]:
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


def _agrfig_urls(fiscal_year: str) -> list[str]:
    """Candidate JBC Agriculture figure-setting PDF URLs for a fiscal year.

    Observed naming: 'agrfig' (e.g. FY2026-27_agrfig.pdf, fy2020-21_agrfig.pdf),
    upper/lower 'FY', optional '_0', served from either leg host.
    """
    m = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if not m:
        return []
    urls = []
    for host in LEG_HOSTS:
        base = f"{host}/sites/default/files"
        for stem in ("agrfig", "agrfig1"):
            for fy in (f"FY{fiscal_year}", f"fy{fiscal_year}"):
                urls.append(f"{base}/{fy}_{stem}.pdf")
                urls.append(f"{base}/{fy}_{stem}_0.pdf")
    return urls


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_agriculture(topic: str, fiscal_year: str = "") -> str:
    """Search Colorado Department of Agriculture sources via Exa neural search.

    Searches ag.colorado.gov and leg.colorado.gov for documents on CDA's budget,
    programs, grants, and funding. Pass a resulting URL to fetch_and_parse_pdf or
    fetch_webpage to extract details.

    Args:
        topic: What to search for — e.g., "agricultural drought resilience funding",
               "brand inspection fees", "Colorado Proud", "Markets Division budget",
               "soil health program".
        fiscal_year: Optional fiscal year to narrow: "2025-26", "2026-27".
    """
    exa = _get_exa()
    query = f"Colorado Department of Agriculture {topic}"
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
            f"No CDA results for '{topic}'. Try broader terms, or use "
            "find_agriculture_budget / find_agriculture_appropriations / find_agriculture_programs."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)
    logger.info(f"agriculture search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_agriculture_budget(fiscal_year: str = "") -> str:
    """Find Colorado Department of Agriculture budget documents and proposals.

    CDA's state budget page collects its budget requests, the Governor's
    agriculture budget proposals, and Long Bill funding summaries. CDA is small
    and funded mostly by Cash Funds (industry fees) with a modest General Fund
    share; recent years added targeted General Fund investments.

    Args:
        fiscal_year: Fiscal year to filter, e.g. '2025-26', '2026-27', or a single
                     year like '2025'. Leave empty for the most recent items.
    """
    documents, pages = _collect(
        fiscal_year, [_STATE_BUDGET_PAGE],
        f"Colorado Department of Agriculture budget request {fiscal_year}".strip(),
    )
    if not documents and not pages:
        return json.dumps({
            "fiscal_year": fiscal_year or "recent",
            "found": False,
            "message": (
                f"No CDA budget docs found for '{fiscal_year}'. "
                f"Browse {_STATE_BUDGET_PAGE} with fetch_webpage, or use "
                "search_agriculture with 'budget request'."
            ),
            "state_budget_page": _STATE_BUDGET_PAGE,
        }, indent=2)
    result = {
        "fiscal_year": fiscal_year or "recent",
        "state_budget_page": _STATE_BUDGET_PAGE,
        "documents": documents,
        "pages": pages,
        "usage_note": (
            "CDA's budget story is largely on HTML pages (state-budget posts, press "
            "releases) — fetch those with fetch_webpage. For the appropriation by fund "
            "type, use find_agriculture_appropriations."
        ),
    }
    logger.info(f"agriculture budget {fiscal_year or 'recent'}: {len(documents)} docs, {len(pages)} pages")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_agriculture_appropriations(fiscal_year: str) -> str:
    """Find the JBC Agriculture figure-setting (appropriations) document for a year.

    The JBC 'agrfig' document is the legislature's staff analysis and recommended
    appropriation for the Department of Agriculture, broken down by fund type
    (General Fund, Cash Funds, Reappropriated, Federal). It is the authoritative
    "approved" figure for CDA's budget.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26'.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for url in _agrfig_urls(fiscal_year):
        if url in seen:
            continue
        seen.add(url)
        if _head_check(url) == 200:
            found.append({"label": f"JBC Agriculture Figure Setting FY{fiscal_year}",
                          "url": url, "http_status": 200})

    if not found:
        for r in _exa_results(f"Colorado JBC Agriculture figure setting FY{fiscal_year} agrfig"):
            if r["is_document"] and _head_check(r["url"]) == 200:
                found.append({"label": r["label"], "url": r["url"], "http_status": 200})

    if not found:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No JBC Agriculture figure-setting PDF found for FY{fiscal_year}. "
                "Try search_agriculture with 'figure setting', or the legislature server's "
                "find_appropriations_documents."
            ),
        }, indent=2)

    result = {
        "fiscal_year": fiscal_year,
        "documents": found[:5],
        "usage_note": (
            "Pass the URL to fetch_and_parse_pdf with keyword_filter such as 'General Fund', "
            "'Cash Funds', or a division name. CDA is small and cash-fund-heavy; the General "
            "Fund line is the politically salient part."
        ),
    }
    logger.info(f"agriculture appropriations FY{fiscal_year}: {len(found)} docs")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_agriculture_programs() -> str:
    """Find CDA program / performance information — what the department funds.

    Returns CDA's performance plan, "CDA at a Glance" snapshot, and grants pages,
    which describe the divisions and programs (Markets, Animal Health, Plant
    Industry, Conservation Services, brand inspection, Colorado Proud, soil health)
    that the budget pays for. Useful for connecting dollars to what they do.
    """
    resources = [
        {"label": "CDA Performance Plan", "url": _PERFORMANCE_PLAN_PAGE},
        {"label": "CDA Grants", "url": _GRANTS_PAGE},
    ]
    for r in _exa_results("Colorado Department of Agriculture programs performance plan at a glance"):
        if r["url"] not in {x["url"] for x in resources}:
            resources.append({"label": r["label"], "url": r["url"]})

    result = {
        "resources": resources[:8],
        "usage_note": (
            "Fetch these pages with fetch_webpage to see CDA's divisions and programs. "
            "Pair with find_agriculture_appropriations to map programs to fund types."
        ),
    }
    logger.info(f"agriculture programs: {len(resources)} resources")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-agriculture MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
