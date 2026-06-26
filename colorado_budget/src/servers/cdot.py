"""
Colorado Department of Transportation (CDOT) MCP server.

CDOT is large and funded very differently from General-Fund agencies: its money
comes from the Highway Users Tax Fund (HUTF — state gas tax & vehicle fees, a
Cash Fund), substantial Federal Funds (FHWA/FTA), and newer fees/transfers
(e.g. SB21-260 enterprises). Very little General Fund. This makes CDOT the
clearest example of a cash/federal-funded department.

This server covers CDOT's budget and capital program. It complements the
colorado-open-data server, which has CDOT *expense and payroll* datasets
(actuals): use those for "what was spent", and these tools for "what was
budgeted/appropriated/planned".

Tools:
- CDOT annual Budget Allocation Plan (the budget book)
- JBC Transportation figure-setting / hearing (the appropriation)
- Statewide Transportation Improvement Program (STIP — the 4-year capital plan)

Run standalone:
    python servers/cdot.py               # streamable-http on port 8011
    python servers/cdot.py --port 8011

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

mcp = FastMCP("colorado-cdot")

TIMEOUT = 20
CDOT_BASE = "https://www.codot.gov"
LEG_HOSTS = ["https://content.leg.colorado.gov", "https://leg.colorado.gov"]
_UA = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}

# ---------------------------------------------------------------------------
# Known landing pages (Exa-verified) — scraped dynamically as starting points
# ---------------------------------------------------------------------------
_BUDGET_INDEX = f"{CDOT_BASE}/business/budget/cdot-annual-budget-reports-and-information"
_BUDGET_PAGE = f"{CDOT_BASE}/business/budget"
_STIP_PAGE = f"{CDOT_BASE}/programs/planning/transportation-plans-and-studies/stip"
_REVENUE_EXP_PAGE = f"{CDOT_BASE}/performance/revenue-expenditures"

_FINANCE_DOMAINS = ["codot.gov", "leg.colorado.gov"]

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


def _extract_doc_links(html: str, base: str = CDOT_BASE) -> list[dict]:
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


def _fy_full(fiscal_year: str) -> Optional[str]:
    """'2026-27' -> '2026-2027' for CDOT budget page slugs."""
    m = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if not m:
        return None
    return f"{m.group(1)}-20{m.group(2)}"


def _fy_short(fiscal_year: str) -> Optional[str]:
    """'2026-27' -> '26-27' for JBC transportation file names."""
    m = re.match(r"(\d{4})-(\d{2})", fiscal_year)
    if not m:
        return None
    return f"{m.group(1)[2:]}-{m.group(2)}"


def _trafig_urls(fiscal_year: str) -> list[str]:
    """Candidate JBC Transportation figure-setting / hearing PDF URLs.

    The transportation JBC files use an inconsistent year form: 'fy26-27_trafig.pdf'
    (short) but 'fy2025-26_trahrg.pdf' (full). Try both year forms, both stems,
    both cases, both hosts.
    """
    if not re.match(r"\d{4}-\d{2}", fiscal_year):
        return []
    forms = [f for f in (fiscal_year, _fy_short(fiscal_year)) if f]
    urls = []
    for host in LEG_HOSTS:
        base = f"{host}/sites/default/files"
        for stem in ("trafig", "trahrg", "trabrf", "trafig1"):
            for fy in forms:
                for cased in (f"FY{fy}", f"fy{fy}"):
                    urls.append(f"{base}/{cased}_{stem}.pdf")
                    urls.append(f"{base}/{cased}_{stem}_0.pdf")
    return urls


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_cdot(topic: str, fiscal_year: str = "") -> str:
    """Search Colorado Department of Transportation sources via Exa neural search.

    Searches codot.gov and leg.colorado.gov for documents on CDOT's budget,
    revenue (HUTF, federal funds, SB21-260 fees), the STIP capital program, and
    specific projects. Pass a resulting URL to fetch_and_parse_pdf or fetch_webpage.

    For CDOT actual expenditures and payroll, use the colorado-open-data tools
    (datasets n5ku-eixc and rkmy-yymq) instead — this server is for budget/plan.

    Args:
        topic: What to search for — e.g., "HUTF revenue forecast", "bridge
               enterprise", "I-70 project funding", "transit funding", "SB21-260
               fees".
        fiscal_year: Optional fiscal year to narrow: "2025-26", "2026-27".
    """
    exa = _get_exa()
    query = f"Colorado CDOT transportation {topic}"
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
            f"No CDOT results for '{topic}'. Try broader terms, or use "
            "find_cdot_budget / find_cdot_appropriations / find_stip."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)
    logger.info(f"cdot search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_cdot_budget(fiscal_year: str) -> str:
    """Find CDOT's annual Budget Allocation Plan for a fiscal year.

    The Budget Allocation Plan is CDOT's budget book — total program by funding
    source (HUTF/state, Federal, SB21-260 enterprises, etc.) and by program area
    (asset management, maintenance, multimodal, safety). It is the authoritative
    "what CDOT plans to spend and from where" document.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26'.
    """
    full = _fy_full(fiscal_year)
    scrape_pages = [_BUDGET_INDEX, _REVENUE_EXP_PAGE]
    if full:
        scrape_pages.insert(0, f"{CDOT_BASE}/business/budget/cdot-budget/fy-{full}-final-budget-allocation-plan")

    documents, pages = _collect(
        fiscal_year, scrape_pages,
        f"CDOT FY{fiscal_year} final budget allocation plan funding",
    )
    if not documents and not pages:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No CDOT budget docs found for FY{fiscal_year}. "
                f"Browse {_BUDGET_INDEX} with fetch_webpage, or use search_cdot "
                "with 'budget allocation plan'."
            ),
            "budget_index": _BUDGET_INDEX,
        }, indent=2)
    result = {
        "fiscal_year": fiscal_year,
        "budget_index": _BUDGET_INDEX,
        "documents": documents,
        "pages": pages,
        "usage_note": (
            "The Budget Allocation Plan is often an HTML page — fetch it with fetch_webpage; "
            "PDFs go to fetch_and_parse_pdf (keyword_filter 'HUTF', 'Federal', 'Total Program'). "
            "For CDOT actual spending, use the colorado-open-data CDOT datasets."
        ),
    }
    logger.info(f"cdot budget FY{fiscal_year}: {len(documents)} docs, {len(pages)} pages")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_cdot_appropriations(fiscal_year: str) -> str:
    """Find the JBC Transportation figure-setting / hearing document for a year.

    The JBC 'trafig' (figure-setting), 'trahrg' (hearing), and 'trabrf' (briefing)
    documents are the legislature's analysis of CDOT's budget. They show how CDOT's
    funding breaks down by source — overwhelmingly Cash Funds (HUTF) and Federal
    Funds, with the small General Fund / transfer pieces called out.

    Args:
        fiscal_year: Fiscal year in 'YYYY-YY' format: '2026-27', '2025-26'.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for url in _trafig_urls(fiscal_year):
        if url in seen:
            continue
        seen.add(url)
        if _head_check(url) == 200:
            stem = ("hearing" if "trahrg" in url else
                    "briefing" if "trabrf" in url else "figure setting")
            found.append({"label": f"JBC Transportation {stem} FY{fiscal_year}",
                          "url": url, "http_status": 200})

    if not found:
        for r in _exa_results(f"Colorado JBC Transportation figure setting FY{fiscal_year} trafig CDOT"):
            if r["is_document"] and _head_check(r["url"]) == 200:
                found.append({"label": r["label"], "url": r["url"], "http_status": 200})

    if not found:
        return json.dumps({
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No JBC Transportation figure-setting PDF found for FY{fiscal_year}. "
                "Try search_cdot with 'figure setting', or the legislature server's "
                "find_appropriations_documents."
            ),
        }, indent=2)

    result = {
        "fiscal_year": fiscal_year,
        "documents": found[:5],
        "usage_note": (
            "Pass the URL to fetch_and_parse_pdf with keyword_filter such as 'HUTF', "
            "'Federal Funds', or 'General Fund' to see CDOT's appropriation by source. "
            "CDOT is dominated by Cash (HUTF) and Federal Funds."
        ),
    }
    logger.info(f"cdot appropriations FY{fiscal_year}: {len(found)} docs")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_stip(fiscal_year: str = "") -> str:
    """Find the Statewide Transportation Improvement Program (STIP).

    The STIP is CDOT's rolling four-year program of capital transportation
    projects and their funding — the concrete list of what gets built and how much
    it costs. Use it to connect budget dollars to specific projects.

    Args:
        fiscal_year: Optional starting fiscal year to narrow, e.g. '2027' (for the
                     FY2027–FY2030 STIP). Leave empty for the current STIP.
    """
    documents, pages = _collect(
        fiscal_year, [_STIP_PAGE],
        f"Colorado CDOT Statewide Transportation Improvement Program STIP {fiscal_year}".strip(),
    )
    if not documents and not pages:
        return json.dumps({
            "found": False,
            "message": (
                f"No STIP documents located. Browse {_STIP_PAGE} with fetch_webpage, "
                "or use search_cdot with 'STIP'."
            ),
            "stip_page": _STIP_PAGE,
        }, indent=2)
    result = {
        "fiscal_year": fiscal_year or "current",
        "stip_page": _STIP_PAGE,
        "documents": documents,
        "pages": pages,
        "usage_note": (
            "The STIP executive summary (PDF) has the program totals by funding source; "
            "the full STIP lists individual projects. Fetch the STIP page with fetch_webpage "
            "for the latest version."
        ),
    }
    logger.info(f"cdot STIP {fiscal_year or 'current'}: {len(documents)} docs, {len(pages)} pages")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-cdot MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
