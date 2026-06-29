"""
Map every agent tool to the MCP server (or "inline") that provides it.

Trajectory evals assert at the *server* level ("did the agent route to the HCPF
server?") rather than pinning exact tool names, because the agent may legitimately
pick `find_hcpf_appropriations` vs `find_hcpf_budget_request` — either proves
correct routing. This module is the lookup that makes that possible.

The map is maintained by hand (it's the authoritative list of what each server
exposes); `test_tool_server_map.py` validates it against the servers' live
tools/list so it can't silently drift.
"""
from __future__ import annotations

# server name (FastMCP name) -> its tool names
SERVER_TOOLS: dict[str, list[str]] = {
    "colorado-open-data": [
        "list_datasets", "get_dataset_metadata", "query_dataset",
    ],
    "web-search": [
        "search_web", "search_colorado_government",
    ],
    "colorado-legislature": [
        "search_bills", "get_bill_details", "get_fiscal_note",
        "find_appropriations_documents",
    ],
    "colorado-ospb": [
        "search_ospb", "find_governor_budget", "find_revenue_forecast",
        "find_budget_amendments",
    ],
    "colorado-revenue": [
        "search_revenue", "find_legislative_forecast",
        "find_tax_expenditure_report", "find_tabor_resources",
    ],
    "colorado-federal-funds": [
        "colorado_federal_summary", "federal_funding_by_agency",
        "top_federal_recipients", "search_federal_awards",
    ],
    "colorado-school-finance": [
        "search_school_finance", "find_school_finance_act",
        "find_per_pupil_funding", "find_finance_formula_resources",
    ],
    "colorado-hcpf": [
        "search_hcpf", "find_hcpf_budget_request", "find_caseload_reports",
        "find_hcpf_appropriations",
    ],
    "colorado-parks-wildlife": [
        "search_cpw", "find_cpw_financial_reports", "get_sources_and_uses",
        "find_cpw_appropriations",
    ],
    "colorado-agriculture": [
        "search_agriculture", "find_agriculture_budget",
        "find_agriculture_appropriations", "find_agriculture_programs",
    ],
    "colorado-cdot": [
        "search_cdot", "find_cdot_budget", "find_cdot_appropriations",
        "find_stip",
    ],
    "data-dot-gov": [
        "get_number_data_publishing_organizations",
        "get_number_data_publishing_organization_url_slugs",
        "search_datasets", "get_keywords", "search_locations",
        "get_location_geometry", "get_harvest_record",
        "get_harvest_record_raw", "get_harvest_record_transformed",
    ],
    # Not an MCP server, but the agent's two inline tools — map them so they
    # don't count as "unknown" and don't get scored as a server.
    "inline": [
        "fetch_webpage", "fetch_and_parse_pdf",
    ],
}

# tool name -> server name
TOOL_SERVER: dict[str, str] = {
    tool: server for server, tools in SERVER_TOOLS.items() for tool in tools
}


def server_for(tool: str) -> str:
    """Server name that provides `tool`, or 'unknown' if unmapped."""
    return TOOL_SERVER.get(tool, "unknown")


def servers_touched(trajectory: list[str], include_inline: bool = False) -> set[str]:
    """Set of MCP servers represented in a tool-call trajectory.

    By default excludes the `inline` pseudo-server (fetch_webpage / parse_pdf),
    since those are support tools, not routing decisions.
    """
    out = {server_for(t) for t in trajectory}
    if not include_inline:
        out.discard("inline")
    out.discard("unknown")
    return out
