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
