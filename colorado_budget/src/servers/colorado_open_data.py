"""
Colorado Open Data MCP server — wraps the Socrata SODA API at data.colorado.gov.

Run standalone:
    python servers/colorado_open_data.py                    # streamable-http on port 8001
    python servers/colorado_open_data.py --port 8001        # explicit port
    python servers/colorado_open_data.py --transport stdio  # stdio (for subprocess use)
"""
import argparse
import json

import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("colorado-open-data")

BASE_URL = "https://data.colorado.gov"
TIMEOUT = 30


@mcp.tool()
def list_datasets(query: str) -> str:
    """Search the Colorado Open Data catalog (data.colorado.gov) for datasets matching a topic.

    Returns dataset names, 4x4 IDs (needed for query_dataset), and short descriptions.
    Use before querying to discover what structured data exists.

    Args:
        query: Topic to search for (e.g. "state expenditures", "medicaid spending",
               "school for the deaf blind", "housing permits", "CDOT payroll")
    """
    url = f"{BASE_URL}/api/catalog/v1"
    params = {"q": query, "limit": 10, "domains": "data.colorado.gov", "only": "datasets"}
    try:
        r = httpx.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error searching catalog: {e}"

    results = []
    for item in r.json().get("results", []):
        res = item.get("resource", {})
        results.append({
            "id": res.get("id"),
            "name": res.get("name"),
            "description": (res.get("description") or "")[:400],
            "updated_at": res.get("updatedAt"),
        })

    if not results:
        return f"No datasets found for '{query}'. Try broader keywords."

    logger.info(f"catalog: {len(results)} results for '{query}'")
    return json.dumps(results, indent=2)


@mcp.tool()
def get_dataset_metadata(dataset_id: str) -> str:
    """Get column names, types, and descriptions for a data.colorado.gov dataset.

    Always call this before query_dataset to learn the exact column names and values
    to use in WHERE clauses.

    Args:
        dataset_id: The 4x4 ID from list_datasets (e.g. 'fjyf-bdat')
    """
    url = f"{BASE_URL}/api/views/{dataset_id}.json"
    try:
        r = httpx.get(url, timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error fetching metadata for {dataset_id}: {e}"

    data = r.json()
    columns = [
        {
            "name": col.get("name"),
            "fieldName": col.get("fieldName"),
            "dataTypeName": col.get("dataTypeName"),
            "description": (col.get("description") or "")[:200],
        }
        for col in data.get("columns", [])
    ]
    return json.dumps({
        "id": dataset_id,
        "name": data.get("name"),
        "description": (data.get("description") or "")[:600],
        "rowsUpdatedAt": data.get("rowsUpdatedAt"),
        "columns": columns,
    }, indent=2)


@mcp.tool()
def query_dataset(
    dataset_id: str,
    where_clause: str = "",
    select_columns: str = "",
    order_by: str = "",
    limit: int = 100,
) -> str:
    """Query a data.colorado.gov dataset using SoQL (Socrata Query Language).

    Call get_dataset_metadata first to learn column names.
    String values in WHERE clauses must be single-quoted (e.g., agency_name='CDHS').

    Args:
        dataset_id: The 4x4 dataset ID (e.g. 'fjyf-bdat')
        where_clause: SoQL filter (e.g. "fiscal_year >= '2015' AND fiscal_year <= '2024'")
        select_columns: Comma-separated columns to return (empty = all)
        order_by: Sort column (e.g. "fiscal_year ASC" or "amount DESC")
        limit: Rows to return, max 1000 (default 100)
    """
    url = f"{BASE_URL}/resource/{dataset_id}.json"
    params: dict = {"$limit": min(limit, 1000)}
    if where_clause:
        params["$where"] = where_clause
    if select_columns:
        params["$select"] = select_columns
    if order_by:
        params["$order"] = order_by

    try:
        r = httpx.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error querying {dataset_id}: {e}"

    data = r.json()
    if not data:
        return (
            f"No rows returned from {dataset_id}. "
            "Verify column names with get_dataset_metadata and check string quoting in WHERE clause."
        )

    logger.info(f"query: {len(data)} rows from {dataset_id}")
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http", choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting colorado-open-data MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
