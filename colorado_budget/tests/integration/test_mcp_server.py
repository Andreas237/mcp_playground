"""
Integration tests — start the MCP server as a subprocess and verify it
responds correctly to MCP protocol requests via the Python MCP SDK.

Requires no LLM and no external APIs.

Run with:
    pytest tests/integration/test_mcp_server.py -v
"""
import asyncio
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# The open_data_server fixture starts the server on port 8001 (see conftest.py).


async def _list_tools(port: int = 8001) -> list[dict]:
    """Connect to the MCP server and return the list of tools."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(f"http://localhost:{port}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{"name": t.name, "description": t.description} for t in result.tools]


def test_tools_list_returns_three_tools(open_data_server):
    tools = asyncio.run(_list_tools())
    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}: {tools}"


def test_tools_have_correct_names(open_data_server):
    tools = asyncio.run(_list_tools())
    names = {t["name"] for t in tools}
    assert names == {"list_datasets", "get_dataset_metadata", "query_dataset"}


def test_tools_have_descriptions(open_data_server):
    tools = asyncio.run(_list_tools())
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


# ---------------------------------------------------------------------------
# Legislature MCP server (port 8003)
# ---------------------------------------------------------------------------

def test_legislature_tools_list_returns_four_tools(legislature_server):
    tools = asyncio.run(_list_tools(8003))
    assert len(tools) == 4, f"Expected 4 tools, got {len(tools)}: {tools}"


def test_legislature_tools_have_correct_names(legislature_server):
    tools = asyncio.run(_list_tools(8003))
    names = {t["name"] for t in tools}
    assert names == {"search_bills", "get_bill_details", "get_fiscal_note", "find_appropriations_documents"}


def test_legislature_tools_have_descriptions(legislature_server):
    tools = asyncio.run(_list_tools(8003))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


def test_legislature_tools_count(legislature_server):
    tools = asyncio.run(_list_tools(8003))
    assert len(tools) == 4, f"Expected 4 legislature tools, got {len(tools)}"


def test_legislature_tool_names(legislature_server):
    tools = asyncio.run(_list_tools(8003))
    names = {t["name"] for t in tools}
    expected = {"search_bills", "get_bill_details", "get_fiscal_note", "find_appropriations_documents"}
    assert names == expected, f"Tool names mismatch: {names}"


def test_legislature_tools_have_descriptions(legislature_server):
    tools = asyncio.run(_list_tools(8003))
    for t in tools:
        assert t["description"]


# ---------------------------------------------------------------------------
# OSPB MCP server (port 8004)
# ---------------------------------------------------------------------------

def test_ospb_tools_list_returns_four_tools(ospb_server):
    tools = asyncio.run(_list_tools(8004))
    assert len(tools) == 4, f"Expected 4 OSPB tools, got {len(tools)}: {tools}"


def test_ospb_tools_have_correct_names(ospb_server):
    tools = asyncio.run(_list_tools(8004))
    names = {t["name"] for t in tools}
    assert names == {"search_ospb", "find_governor_budget", "find_revenue_forecast", "find_budget_amendments"}


def test_ospb_tools_have_descriptions(ospb_server):
    tools = asyncio.run(_list_tools(8004))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


# ---------------------------------------------------------------------------
# Revenue & TABOR MCP server (port 8005)
# ---------------------------------------------------------------------------

def test_revenue_tools_list_returns_four_tools(revenue_server):
    tools = asyncio.run(_list_tools(8005))
    assert len(tools) == 4, f"Expected 4 revenue tools, got {len(tools)}: {tools}"


def test_revenue_tools_have_correct_names(revenue_server):
    tools = asyncio.run(_list_tools(8005))
    names = {t["name"] for t in tools}
    assert names == {
        "search_revenue",
        "find_legislative_forecast",
        "find_tax_expenditure_report",
        "find_tabor_resources",
    }


def test_revenue_tools_have_descriptions(revenue_server):
    tools = asyncio.run(_list_tools(8005))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


# ---------------------------------------------------------------------------
# Federal Funds MCP server (port 8006)
# ---------------------------------------------------------------------------

def test_federal_tools_list_returns_four_tools(federal_funds_server):
    tools = asyncio.run(_list_tools(8006))
    assert len(tools) == 4, f"Expected 4 federal tools, got {len(tools)}: {tools}"


def test_federal_tools_have_correct_names(federal_funds_server):
    tools = asyncio.run(_list_tools(8006))
    names = {t["name"] for t in tools}
    assert names == {
        "colorado_federal_summary",
        "federal_funding_by_agency",
        "top_federal_recipients",
        "search_federal_awards",
    }


def test_federal_tools_have_descriptions(federal_funds_server):
    tools = asyncio.run(_list_tools(8006))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


# ---------------------------------------------------------------------------
# School Finance MCP server (port 8007)
# ---------------------------------------------------------------------------

def test_school_finance_tools_list_returns_four_tools(school_finance_server):
    tools = asyncio.run(_list_tools(8007))
    assert len(tools) == 4, f"Expected 4 school-finance tools, got {len(tools)}: {tools}"


def test_school_finance_tools_have_correct_names(school_finance_server):
    tools = asyncio.run(_list_tools(8007))
    names = {t["name"] for t in tools}
    assert names == {
        "search_school_finance",
        "find_school_finance_act",
        "find_per_pupil_funding",
        "find_finance_formula_resources",
    }


def test_school_finance_tools_have_descriptions(school_finance_server):
    tools = asyncio.run(_list_tools(8007))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


# ---------------------------------------------------------------------------
# HCPF / Medicaid MCP server (port 8008)
# ---------------------------------------------------------------------------

def test_hcpf_tools_list_returns_four_tools(hcpf_server):
    tools = asyncio.run(_list_tools(8008))
    assert len(tools) == 4, f"Expected 4 HCPF tools, got {len(tools)}: {tools}"


def test_hcpf_tools_have_correct_names(hcpf_server):
    tools = asyncio.run(_list_tools(8008))
    names = {t["name"] for t in tools}
    assert names == {
        "search_hcpf",
        "find_hcpf_budget_request",
        "find_caseload_reports",
        "find_hcpf_appropriations",
    }


def test_hcpf_tools_have_descriptions(hcpf_server):
    tools = asyncio.run(_list_tools(8008))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"


# ---------------------------------------------------------------------------
# Parks & Wildlife MCP server (port 8009)
# ---------------------------------------------------------------------------

def test_cpw_tools_list_returns_four_tools(cpw_server):
    tools = asyncio.run(_list_tools(8009))
    assert len(tools) == 4, f"Expected 4 CPW tools, got {len(tools)}: {tools}"


def test_cpw_tools_have_correct_names(cpw_server):
    tools = asyncio.run(_list_tools(8009))
    names = {t["name"] for t in tools}
    assert names == {
        "search_cpw",
        "find_cpw_financial_reports",
        "get_sources_and_uses",
        "find_cpw_appropriations",
    }


def test_cpw_tools_have_descriptions(cpw_server):
    tools = asyncio.run(_list_tools(8009))
    for t in tools:
        assert t["description"], f"Tool {t['name']} has no description"
