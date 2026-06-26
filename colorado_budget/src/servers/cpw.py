"""
Colorado Parks & Wildlife (CPW) MCP server.

CPW is a division of the Department of Natural Resources and is unusual among
state agencies: it is largely ENTERPRISE / CASH-FUNDED — hunting & fishing
licenses, state park passes, GOCO lottery dollars, and federal excise-tax
apportionments (Pittman-Robertson / Dingell-Johnson) — with very little General
Fund. That makes it a good contrast to GF-driven departments: a politician's
claim about "state funding for parks" usually means cash funds and fees, not the
General Fund.

This server covers CPW's own financial reporting plus its state appropriation:
- Quarterly financial reports / updates to the Parks & Wildlife Commission
- The Sources & Uses of Funds fact sheet (the funding-model explainer)
- JBC Natural Resources figure-setting / budget briefing (the appropriation)

Run standalone:
    python servers/cpw.py               # streamable-http on port 8009
    python servers/cpw.py --port 8009

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

mcp = FastMCP("colorado-parks-wildlife")

TIMEOUT = 20
CPW_BASE = "https://cpw.state.co.us"
LEG_HOSTS = ["https://content.leg.colorado.gov", "https://leg.colorado.gov"]
_UA = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}

# ---------------------------------------------------------------------------
# Known landing pages (Exa-verified) — scraped dynamically as starting points
# ---------------------------------------------------------------------------
_PLANS_REPORTS_PAGE = f"{CPW_BASE}/plans-and-reports"
_FUNDING_PAGE = f"{CPW_BASE}/funding-colorado-parks-and-wildlife"
_COMMISSION_PAGE = f"{CPW_BASE}/committees/colorado-parks-and-wildlife-commission"
# Stable URL for the funding-model fact sheet
_SOURCES_USES_PDF = f"{CPW_BASE}/Documents/About/Reports/Sources_and_Uses_of_Funds_Fact_Sheet.pdf"

_FINANCE_DOMAINS = ["cpw.state.co.us", "leg.colorado.gov"]

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


def _extract_doc_links(html: str, base: str = CPW_BASE) -> list[dict]:
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


def _natfig_urls(fiscal_year: str) -> list[str]:
    """Candidate JBC Natural Resources figure-setting / briefing PDF URLs.

    CPW's appropriation lives within DNR's Natural Resources budget. Naming varies:
    'natfig'/'natfig1' (figure-setting) and 'natbrf1' (briefing), upper/lower 'FY',
    optional '_0', served from either leg host.
    """
    m = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if not m:
        return []
    urls = []
    for host in LEG_HOSTS:
        base = f"{host}/sites/default/files"
        for stem in ("natfig1", "natfig", "natbrf1"):
            for fy in (f"FY{fiscal_year}", f"fy{fiscal_year}"):
                urls.append(f"{base}/{fy}_{stem}.pdf")
                urls.append(f"{base}/{fy}_{stem}_0.pdf")
    return urls


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_cpw(topic: str, fiscal_year: str = "") -> str:
    """Search Colorado Parks & Wildlife sources via Exa neural search.

    Searches cpw.state.co.us and leg.colorado.gov for documents on CPW finances,
    license/pass revenue, park funding, GOCO grants, and the Natural Resources
    budget. Pass a resulting URL to fetch_and_parse_pdf or fetch_webpage.

    Args:
        topic: What to search for — e.g., "hunting license revenue", "state park
               fee increase", "Keep Colorado Wild pass", "GOCO funding", "CPW
               financial sustainability".
        fiscal_year: Optional fiscal year to narrow: "2025-26", "2026-27".
    """
    exa = _get_exa()
    query = f"Colorado Parks Wildlife {topic}"
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
            f"No CPW results for '{topic}'. Try broader terms, or use "
            "find_cpw_financial_reports / get_sources_and_uses / find_cpw_appropriations."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)
    logger.info(f"cpw search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_cpw_financial_reports(year: str = "") -> str:
    """Find CPW quarterly financial reports / updates to the Commission.

    CPW staff present financial reports and updates to the Parks & Wildlife
    Commission through the year. These show cash-fund balances, license and park
    revenue versus projections, and expenditures — the authoritative picture of
    CPW's (mostly fee-funded) finances.

    Args:
        year: Calendar/fiscal year to filter, e.g. '2025', '2024', '2025-26'.
              Leave empty for the most recent reports.
    """
    documents, pages = _collect(
        year,
        [_PLANS_REPORTS_PAGE, _COMMISSION_PAGE, _FUNDING_PAGE],
        f"Colorado Parks Wildlife Commission financial report update {year}".strip(),
    )
    if not documents and not pages:
        return json.dumps({
            "found": False,
            "message": (
                f"No CPW financial reports located for '{year}'. "
                f"Browse {_PLANS_REPORTS_PAGE} with fetch_webpage, or use search_cpw "
                "with 'Commission financial report'."
            ),
            "plans_reports_page": _PLANS_REPORTS_PAGE,
        }, indent=2)
    result = {
        "year": year or "recent",
        "plans_reports_page": _PLANS_REPORTS_PAGE,
        "documents": documents,
        "pages": pages,
        "usage_note": (
            "Financial reports are PDFs — pass to fetch_and_parse_pdf with keyword_filter "
            "like 'Wildlife Cash Fund' or 'Parks'. For the funding model overview, use "
            "get_sources_and_uses."
        ),
    }
    logger.info(f"cpw financial reports: {year or 'recent'} — {len(documents)} docs, {len(pages)} pages")
    return json.dumps(result, indent=2)


@mcp.tool()
def get_sources_and_uses() -> str:
    """Get CPW's Sources & Uses of Funds fact sheet — its funding-model explainer.

    CPW is funded very differently from General-Fund agencies. This fact sheet
    breaks down where CPW's money comes from (hunting/fishing licenses, state park
    passes incl. Keep Colorado Wild, GOCO lottery, federal excise-tax
    apportionments, severance tax) and where it goes. Use it to explain why "state
    funding for parks/wildlife" is mostly cash funds and fees, not General Fund.
    """
    resources = []
    if _head_check(_SOURCES_USES_PDF) == 200:
        resources.append({
            "label": "CPW Sources & Uses of Funds Fact Sheet (PDF)",
            "url": _SOURCES_USES_PDF, "http_status": 200,
        })
    # Always include the funding overview page + live search for the latest version
    resources.append({"label": "Funding Colorado Parks and Wildlife (overview page)",
                      "url": _FUNDING_PAGE})
    for r in _exa_results("Colorado Parks Wildlife sources and uses of funds funding model"):
        if r["url"] not in {x["url"] for x in resources}:
            resources.append({"label": r["label"], "url": r["url"]})

    result = {
        "resources": resources[:8],
        "usage_note": (
            "Pass the fact-sheet PDF URL to fetch_and_parse_pdf, or fetch the funding "
            "overview page with fetch_webpage. Key point for analysis: CPW is "
            "enterprise/cash-funded — licenses, park passes, GOCO, and federal "
            "apportionments — with minimal General Fund."
        ),
    }
    logger.info(f"cpw sources & uses: {len(resources)} resources")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_cpw_appropriations(fiscal_year: str) -> str:
    """Find the JBC Natural Resources figure-setting / briefing for CPW's budget.

    CPW's state appropriation is set within the Department of Natural Resources
    budget. The JBC 'natfig' (figure-setting) and 'natbrf' (budget briefing)
    documents contain the Parks & Wildlife line items by fund type — overwhelmingly
    Cash Funds and Federal Funds, with the General Fund share called out.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26'.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for url in _natfig_urls(fiscal_year):
        if url in seen:
            continue
        seen.add(url)
        if _head_check(url) == 200:
            stem = "briefing" if "natbrf" in url else "figure setting"
            found.append({"label": f"JBC Natural Resources {stem} FY{fiscal_year}",
                          "url": url, "http_status": 200})

    if not found:
        for r in _exa_results(f"Colorado JBC Natural Resources figure setting FY{fiscal_year} natfig"):
            if r["is_document"] and _head_check(r["url"]) == 200:
                found.append({"label": r["label"], "url": r["url"], "http_status": 200})

    if not found:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No JBC Natural Resources figure-setting PDF found for FY{fiscal_year}. "
                "Try search_cpw with 'Natural Resources figure setting', or the legislature "
                "server's find_appropriations_documents."
            ),
        }, indent=2)

    result = {
        "fiscal_year": fiscal_year,
        "documents": found[:5],
        "usage_note": (
            "Pass the URL to fetch_and_parse_pdf with keyword_filter such as 'Parks and "
            "Wildlife', 'Wildlife Cash Fund', or 'General Fund' to see CPW's appropriation "
            "by fund type. CPW is dominated by Cash and Federal Funds."
        ),
    }
    logger.info(f"cpw appropriations FY{fiscal_year}: {len(found)} docs")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-parks-wildlife MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
