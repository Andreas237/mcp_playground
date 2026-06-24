"""
Web search MCP server — Exa neural search for Colorado policy context.

Run standalone:
    python servers/web_search.py                    # streamable-http on port 8002
    python servers/web_search.py --transport stdio  # stdio transport

Requires EXA_API_KEY in environment or colorado_budget/.env.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_api_keys

mcp = FastMCP("web-search")


def _get_exa():
    from exa_py import Exa
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY not set. Add it to colorado_budget/.env")
    return Exa(api_key=key)


@mcp.tool()
def search_web(query: str, num_results: int = 8) -> str:
    """Search the web for Colorado policy, budget, and legislation context.

    Use this when government data portals don't cover the question — e.g. press coverage
    of a specific bill, policy analysis, historical context, or comparisons to other states.
    Prefer Colorado-specific queries; include 'Colorado' in the query for best results.

    Args:
        query: Search query (e.g. "Colorado affordable housing permitting costs 2023",
               "Colorado School for Deaf Blind funding history")
        num_results: Number of results to return (default 8, max 20)
    """
    exa = _get_exa()
    try:
        results = exa.search(
            query,
            type="auto",
            num_results=min(num_results, 20),
            contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    output = []
    for r in results.results:
        entry: dict = {
            "title": r.title,
            "url": r.url,
            "published_date": getattr(r, "published_date", None),
        }
        if r.highlights:
            entry["highlights"] = r.highlights
        output.append(entry)

    if not output:
        return f"No results for: {query}"

    logger.info(f"search: {len(output)} results for '{query}'")
    return json.dumps(output, indent=2)


@mcp.tool()
def search_colorado_government(query: str, num_results: int = 8) -> str:
    """Search Colorado government and official sources specifically.

    Restricts results to .gov, leg.colorado.gov, ospb.colorado.gov, and authoritative
    Colorado policy sources. Use for finding official documents, agency reports, and
    legislative records that don't have a direct structured API.

    Args:
        query: Search query focused on Colorado government topics
        num_results: Number of results (default 8)
    """
    exa = _get_exa()
    try:
        results = exa.search(
            query,
            type="auto",
            num_results=min(num_results, 20),
            include_domains=[
                "colorado.gov",
                "leg.colorado.gov",
                "ospb.colorado.gov",
                "cde.state.co.us",
                "cdhs.colorado.gov",
                "hcpf.colorado.gov",
                "coloradofiscalinstitute.org",
            ],
            contents={"highlights": True},
        )
    except Exception as e:
        return f"Search error: {e}"

    output = []
    for r in results.results:
        entry: dict = {"title": r.title, "url": r.url}
        if r.highlights:
            entry["highlights"] = r.highlights
        output.append(entry)

    if not output:
        return f"No government-source results for: {query}. Try search_web for broader coverage."

    logger.info(f"gov search: {len(output)} results for '{query}'")
    return json.dumps(output, indent=2)


if __name__ == "__main__":
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http", choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting web-search MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
