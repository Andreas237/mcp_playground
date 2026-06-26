"""
Colorado School Finance MCP server.

Provides structured access to Colorado public school funding from the Colorado
Department of Education (CDE) and the legislature's school-finance resources.

K-12 school finance is the single largest area of state General Fund spending
and a perennial election-season topic. The key numbers:
- Total Program funding (the statewide K-12 funding pool)
- Per-pupil funding (statewide base and by district)
- The Public School Finance Act formula — currently being overhauled by
  HB24-1448, which phases in a new formula through the late 2020s.

This complements the legislature server (JBC Education figure-setting / Long
Bill) and the OSPB server (Governor's education request) with CDE's own
authoritative funding data and district-level detail.

Run standalone:
    python servers/school_finance.py           # streamable-http on port 8007
    python servers/school_finance.py --port 8007

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

mcp = FastMCP("colorado-school-finance")

TIMEOUT = 20
# CDE's school-finance site lives on ed.cde.state.co.us; www.cde.state.co.us
# 301-redirects there (and relative links resolved against www 404). Use ed.
CDE_BASE = "https://ed.cde.state.co.us"
_UA = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}

# ---------------------------------------------------------------------------
# Known landing pages (verified) — scraped dynamically as starting points
# ---------------------------------------------------------------------------
_FINANCE_PAGE = f"{CDE_BASE}/cdefinance"
_TRANSPARENCY_PAGE = f"{CDE_BASE}/cdefinance/sffpp/sffinancialtransparency"
_SCHOOL_DISTRICT_OPS_PAGE = f"{CDE_BASE}/cdefinance/schooldistrictoperations"
_LCS_SCHOOL_FINANCE = "https://leg.colorado.gov/agencies/legislative-council-staff/school-finance"

# Curated formula-reform references (HB24-1448 new Public School Finance formula)
_FORMULA_REFERENCES = [
    {"label": "CDE School Finance (Public School Finance Unit)", "url": _FINANCE_PAGE},
    {"label": "Legislative Council Staff — School Finance", "url": _LCS_SCHOOL_FINANCE},
]

# Exa domains relevant to school finance
_FINANCE_DOMAINS = ["cde.state.co.us", "leg.colorado.gov"]

# CDE publishes school finance data as PDF and Excel (.xlsx/.xls)
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


def _extract_doc_links(html: str, base: str = CDE_BASE) -> list[dict]:
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
    """Exa search restricted to school-finance domains.

    Returns all hits as {label, url, is_document}. CDE funding info lives mostly
    on HTML pages (fact sheets, per-year funding pages), not loose files, so we
    keep pages too — not only PDF/Excel links.
    """
    out: list[dict] = []
    try:
        exa = _get_exa()
        results = exa.search(
            query,
            type="auto",
            num_results=num_results,
            include_domains=_FINANCE_DOMAINS,
        )
        for r in results.results:
            is_doc = bool(re.search(r"\.(pdf|xlsx?|xls)(\?|$)", r.url, re.IGNORECASE))
            out.append({"label": r.title or "", "url": r.url, "is_document": is_doc})
    except Exception as e:
        logger.warning(f"Exa search failed: {e}")
    return out


def _collect(year: str, scrape_pages: list[str], exa_query: str) -> tuple[list[dict], list[dict]]:
    """Gather downloadable docs (HEAD-checked 200) and reference pages.

    Combines a light scrape of known CDE pages for direct PDF/Excel files with an
    Exa search, which reliably surfaces the right fact-sheet / funding pages even
    as CDE reorganizes its site. Returns (documents, pages); documents are
    deduped and filtered to those that actually resolve (HTTP 200).
    """
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


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_school_finance(topic: str, year: str = "") -> str:
    """Search CDE and legislative school-finance sources via Exa neural search.

    Searches cde.state.co.us (Colorado Department of Education) and
    leg.colorado.gov (Legislative Council school-finance materials) for documents
    on K-12 funding, per-pupil amounts, the Public School Finance Act, total
    program funding, mill levies, and the HB24-1448 formula overhaul. Pass a
    resulting URL to fetch_and_parse_pdf or fetch_webpage to extract numbers.

    Args:
        topic: What to search for — e.g., "total program funding", "per pupil
               funding by district", "HB24-1448 new formula phase-in",
               "at-risk funding", "mill levy override".
        year: Optional school/fiscal year: "2025", "2024", "2025-26".
    """
    exa = _get_exa()
    query = f"Colorado school finance {topic}"
    if year:
        query += f" {year}"
    try:
        results = exa.search(
            query,
            type="auto",
            num_results=10,
            include_domains=_FINANCE_DOMAINS,
            contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    if not results.results:
        return (
            f"No results for '{topic}'. Try broader terms, or use "
            "find_school_finance_act / find_per_pupil_funding / "
            "find_finance_formula_resources for direct document discovery."
        )

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights[:2]
        output.append(entry)

    logger.info(f"school finance search: {len(output)} results for '{topic}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def find_school_finance_act(year: str = "") -> str:
    """Find Colorado School Finance Act funding documents for a year.

    The School Finance Act sets statewide Total Program funding and the base
    per-pupil amount each year. CDE publishes the funding workbooks/reports that
    show Total Program, the statewide average per-pupil funding, and the state
    vs. local share. These are the authoritative K-12 funding numbers.

    Args:
        year: School/fiscal year, e.g. '2025-26', '2024-25', or a single year
              like '2025'. Leave empty to return the most recent documents found.
    """
    documents, pages = _collect(
        year,
        [_FINANCE_PAGE, _TRANSPARENCY_PAGE],
        f"Colorado School Finance Act {year} total program per pupil funding fact sheet".strip(),
    )

    if not documents and not pages:
        return json.dumps({
            "found": False,
            "message": (
                f"No School Finance Act documents located for '{year}'. "
                f"Browse {_FINANCE_PAGE} with fetch_webpage, or use search_school_finance "
                "with 'total program funding'."
            ),
            "finance_page": _FINANCE_PAGE,
        }, indent=2)

    result = {
        "year": year or "recent",
        "finance_page": _FINANCE_PAGE,
        "documents": documents,
        "pages": pages,
        "usage_note": (
            "Pass a PDF document URL to fetch_and_parse_pdf with keyword_filter such as "
            "'Total Program' or 'per pupil'. CDE fact sheets and per-year funding pages are "
            "HTML — fetch those with fetch_webpage. Excel (.xlsx) files hold per-district "
            "detail but fetch_and_parse_pdf only handles PDFs."
        ),
    }
    logger.info(f"school finance act: {year or 'recent'} — {len(documents)} docs, {len(pages)} pages")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_per_pupil_funding(year: str = "") -> str:
    """Find per-pupil / district-level school funding data files from CDE.

    CDE publishes per-pupil funding and Total Program by district (typically as
    Excel workbooks). Use this to compare funding across districts or find a
    specific district's per-pupil amount. The statewide base per-pupil amount is
    in the School Finance Act documents (find_school_finance_act).

    Args:
        year: School/fiscal year, e.g. '2025-26', '2024-25', '2025'.
              Leave empty to return the most recent files found.
    """
    documents, pages = _collect(
        year,
        [_TRANSPARENCY_PAGE, _FINANCE_PAGE, _SCHOOL_DISTRICT_OPS_PAGE],
        f"Colorado per pupil funding by district school finance {year}".strip(),
    )

    if not documents and not pages:
        return json.dumps({
            "found": False,
            "message": (
                f"No per-pupil funding resources located for '{year}'. "
                f"Browse {_TRANSPARENCY_PAGE} with fetch_webpage, or use "
                "search_school_finance with 'per pupil funding by district'."
            ),
            "transparency_page": _TRANSPARENCY_PAGE,
        }, indent=2)

    result = {
        "year": year or "recent",
        "transparency_page": _TRANSPARENCY_PAGE,
        "documents": documents,
        "pages": pages,
        "usage_note": (
            "Per-year 'School Finance Funding' pages (HTML) list district detail — fetch with "
            "fetch_webpage. District-level files are often Excel (.xlsx), which "
            "fetch_and_parse_pdf cannot read. For statewide totals and the base per-pupil "
            "amount, use find_school_finance_act."
        ),
    }
    logger.info(f"per pupil funding: {year or 'recent'} — {len(documents)} docs, {len(pages)} pages")
    return json.dumps(result, indent=2)


@mcp.tool()
def find_finance_formula_resources(topic: str = "") -> str:
    """Find resources on Colorado's Public School Finance formula, incl. HB24-1448.

    Colorado is overhauling its school finance formula via HB24-1448 (2024),
    phasing in a new formula through the late 2020s that changes how at-risk
    students, English-language learners, and district size factor into funding.
    Returns CDE and Legislative Council reference pages plus live search results
    on the formula and its phase-in.

    Args:
        topic: Optional focus — e.g., "HB24-1448 phase-in", "at-risk factor",
               "averaging vs single count", "new formula vs old formula".
               Leave empty for the core reference pages.
    """
    resources = list(_FORMULA_REFERENCES)

    query = "Colorado HB24-1448 school finance formula"
    if topic:
        query += f" {topic}"
    try:
        exa = _get_exa()
        results = exa.search(
            query,
            type="auto",
            num_results=6,
            include_domains=_FINANCE_DOMAINS,
            contents={"highlights": True},
        )
        for r in results.results:
            entry: dict = {"label": r.title or "", "url": r.url}
            if r.highlights:
                entry["highlights"] = r.highlights[:2]
            resources.append(entry)
    except Exception as e:
        logger.warning(f"formula search failed: {e}")

    result = {
        "topic": topic or "school finance formula (HB24-1448)",
        "resources": resources,
        "usage_note": (
            "These explain the formula. For the dollar amounts it produces, use "
            "find_school_finance_act (statewide totals) and find_per_pupil_funding "
            "(by district)."
        ),
    }
    logger.info(f"formula resources: '{topic or 'overview'}' — {len(resources)} entries")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8007)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-school-finance MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
