"""
Colorado Federal Funds MCP server.

Provides structured access to federal money flowing into Colorado via the
USAspending.gov v2 REST API (no API key required). This is the "Federal Funds"
side of the fund-type story: how much of Colorado's spending originates from
federal grants and awards rather than the General Fund or Cash Funds.

Federal funds are politically significant because they can disappear with
federal policy changes — a department heavily reliant on federal dollars (e.g.
Health Care Policy & Financing / Medicaid) faces very different risk than one
funded mostly by the General Fund.

Tools answer:
- How much total federal money came to Colorado in a fiscal year?
- Which federal agencies send the most money to Colorado?
- Which Colorado recipients (incl. state agencies) receive the most?
- What specific awards went to a named recipient (e.g. CDOT, HCPF)?

Run standalone:
    python servers/federal_funds.py            # streamable-http on port 8006
    python servers/federal_funds.py --port 8006

No API key required (USAspending.gov is public).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_api_keys

mcp = FastMCP("colorado-federal-funds")

TIMEOUT = 60
API_BASE = "https://api.usaspending.gov"
_UA = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}
CO_FIPS = "08"  # Colorado state FIPS code

# Friendly award-type groups → USAspending award_type_codes.
# Assistance (grants etc.) is what shows up as "Federal Funds" in agency budgets.
_AWARD_TYPES = {
    "grants":     ["02", "03", "04", "05"],              # block, formula, project, cooperative agreement
    "assistance": ["02", "03", "04", "05", "06", "10"],  # + direct payments / other financial assistance
    "contracts":  ["A", "B", "C", "D"],                  # procurement contracts
    "all":        ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fy_time_period(fiscal_year: str) -> list[dict]:
    """Federal fiscal year 'YYYY' → time_period filter (Oct 1 prior year → Sep 30)."""
    try:
        fy = int(fiscal_year)
    except (TypeError, ValueError):
        from datetime import date
        fy = date.today().year
    return [{"start_date": f"{fy - 1}-10-01", "end_date": f"{fy}-09-30"}]


def _award_type_codes(kind: str) -> list[str]:
    return _AWARD_TYPES.get((kind or "grants").lower(), _AWARD_TYPES["grants"])


def _post(endpoint: str, body: dict) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_BASE}{endpoint}", json=body, headers=_UA, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"POST {endpoint}: {e}")
        return None


def _get(endpoint: str, params: dict) -> Optional[dict]:
    try:
        r = httpx.get(f"{API_BASE}{endpoint}", params=params, headers=_UA, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"GET {endpoint}: {e}")
        return None


def _co_filters(fiscal_year: str, award_type: str) -> dict:
    """Standard filter block: recipients located in Colorado, FY, award types."""
    return {
        "recipient_locations": [{"country": "USA", "state": "CO"}],
        "award_type_codes": _award_type_codes(award_type),
        "time_period": _fy_time_period(fiscal_year),
    }


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def colorado_federal_summary(fiscal_year: str = "2024") -> str:
    """Top-line total of federal award money to Colorado for a fiscal year.

    Returns the total federal prime-award dollars and award count flowing to all
    recipients located in Colorado (state agencies, local governments,
    universities, businesses, nonprofits, individuals), from USAspending.gov.
    Use this for the headline "how much federal money came to Colorado" figure,
    then drill down with federal_funding_by_agency or top_federal_recipients.

    Note: this counts ALL Colorado recipients, not just state government. For
    state-agency-specific federal funds, use top_federal_recipients or
    search_federal_awards.

    Args:
        fiscal_year: Federal fiscal year, e.g. '2024', '2023'. Federal FY runs
                     Oct 1 – Sep 30. Use 'all' or 'latest' for cumulative/latest.
    """
    year = fiscal_year if fiscal_year in ("all", "latest") else fiscal_year
    data = _get(f"/api/v2/recipient/state/{CO_FIPS}/", {"year": year})
    if not data:
        return json.dumps({
            "found": False,
            "message": "USAspending state profile unavailable. Try again or use top_federal_recipients.",
        }, indent=2)

    result = {
        "state": data.get("name", "Colorado"),
        "fiscal_year": fiscal_year,
        "total_federal_prime_amount": data.get("total_prime_amount"),
        "total_federal_prime_awards": data.get("total_prime_awards"),
        "population": data.get("population"),
        "median_household_income": data.get("median_household_income"),
        "source": "USAspending.gov",
        "usage_note": (
            "This is ALL federal money to Colorado recipients. Use "
            "federal_funding_by_agency (which federal agencies send it) or "
            "top_federal_recipients (which CO entities receive it) to break it down."
        ),
    }
    logger.info(f"CO federal summary FY{fiscal_year}: ${data.get('total_prime_amount', 0):,.0f}")
    return json.dumps(result, indent=2)


@mcp.tool()
def federal_funding_by_agency(fiscal_year: str = "2024", award_type: str = "grants", limit: int = 10) -> str:
    """Which federal agencies send the most money to Colorado, ranked by dollars.

    Aggregates federal awards to Colorado recipients by the AWARDING federal
    agency (e.g. Department of Health and Human Services, Department of
    Transportation). Reveals where Colorado's federal dollars originate — useful
    for assessing exposure to specific federal funding streams.

    Args:
        fiscal_year: Federal fiscal year, e.g. '2024'.
        award_type: 'grants' (default — block/formula/project/cooperative),
                    'assistance' (grants + direct payments), 'contracts'
                    (procurement), or 'all'.
        limit: Number of agencies to return (default 10).
    """
    body = {
        "category": "awarding_agency",
        "filters": _co_filters(fiscal_year, award_type),
        "limit": max(1, min(limit, 50)),
    }
    data = _post("/api/v2/search/spending_by_category/awarding_agency/", body)
    if not data:
        return json.dumps({"found": False, "message": "USAspending query failed."}, indent=2)

    agencies = [
        {"agency": x.get("name"), "amount": x.get("amount")}
        for x in data.get("results", [])
    ]
    result = {
        "fiscal_year": fiscal_year,
        "award_type": award_type,
        "awarding_agencies": agencies,
        "source": "USAspending.gov",
    }
    logger.info(f"federal by agency FY{fiscal_year} ({award_type}): {len(agencies)} agencies")
    return json.dumps(result, indent=2)


@mcp.tool()
def top_federal_recipients(fiscal_year: str = "2024", award_type: str = "grants", limit: int = 10) -> str:
    """Which Colorado recipients receive the most federal money, ranked by dollars.

    Aggregates federal awards by RECIPIENT located in Colorado. The top
    recipients are typically state agencies — e.g. the Department of Health Care
    Policy & Financing (Medicaid) usually dominates, followed by Human Services,
    the Office of Information Technology, CDOT, and the Board of Education. This
    is the best single view of how federal dollars are distributed across
    Colorado state government.

    Args:
        fiscal_year: Federal fiscal year, e.g. '2024'.
        award_type: 'grants' (default), 'assistance', 'contracts', or 'all'.
        limit: Number of recipients to return (default 10).
    """
    body = {
        "category": "recipient",
        "filters": _co_filters(fiscal_year, award_type),
        "limit": max(1, min(limit, 50)),
    }
    data = _post("/api/v2/search/spending_by_category/recipient/", body)
    if not data:
        return json.dumps({"found": False, "message": "USAspending query failed."}, indent=2)

    recipients = [
        {"recipient": x.get("name"), "amount": x.get("amount")}
        for x in data.get("results", [])
    ]
    result = {
        "fiscal_year": fiscal_year,
        "award_type": award_type,
        "top_recipients": recipients,
        "source": "USAspending.gov",
        "usage_note": (
            "Top recipients are usually state agencies. For the awards behind a "
            "single recipient, call search_federal_awards with its name."
        ),
    }
    logger.info(f"top CO recipients FY{fiscal_year} ({award_type}): {len(recipients)}")
    return json.dumps(result, indent=2)


@mcp.tool()
def search_federal_awards(recipient: str, fiscal_year: str = "2024", award_type: str = "grants", limit: int = 10) -> str:
    """Search individual federal awards to a named Colorado recipient.

    Returns the largest federal awards matching a recipient name — e.g.
    'Colorado Department of Transportation', 'Health Care Policy', 'State of
    Colorado'. Each result shows the awarding federal agency and dollar amount,
    so you can see exactly which federal programs fund a specific state agency.

    Args:
        recipient: Recipient name or fragment to search, e.g.
                   'Colorado Department of Transportation', 'Health Care Policy
                   and Financing', 'Department of Human Services Colorado'.
        fiscal_year: Federal fiscal year, e.g. '2024'.
        award_type: 'grants' (default), 'assistance', 'contracts', or 'all'.
        limit: Number of awards to return (default 10).
    """
    filters = _co_filters(fiscal_year, award_type)
    filters["recipient_search_text"] = [recipient]
    body = {
        "filters": filters,
        "fields": ["Award ID", "Recipient Name", "Awarding Agency", "Award Amount", "Award Type"],
        "limit": max(1, min(limit, 50)),
        "sort": "Award Amount",
        "order": "desc",
    }
    data = _post("/api/v2/search/spending_by_award/", body)
    if not data:
        return json.dumps({"found": False, "message": "USAspending query failed."}, indent=2)

    awards = [
        {
            "recipient": x.get("Recipient Name"),
            "awarding_agency": x.get("Awarding Agency"),
            "amount": x.get("Award Amount"),
            "award_type": x.get("Award Type"),
            "award_id": x.get("Award ID"),
        }
        for x in data.get("results", [])
    ]
    if not awards:
        return json.dumps({
            "recipient": recipient,
            "fiscal_year": fiscal_year,
            "found": False,
            "message": (
                f"No {award_type} awards found for '{recipient}' in FY{fiscal_year}. "
                "Try a shorter name fragment, a different award_type, or another year."
            ),
        }, indent=2)

    result = {
        "recipient_query": recipient,
        "fiscal_year": fiscal_year,
        "award_type": award_type,
        "awards": awards,
        "source": "USAspending.gov",
    }
    logger.info(f"award search '{recipient}' FY{fiscal_year}: {len(awards)} awards")
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-federal-funds MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
