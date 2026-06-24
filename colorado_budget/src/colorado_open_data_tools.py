import json
import re
from html.parser import HTMLParser

import httpx
from loguru import logger
from strands import tool


class _TextExtractor(HTMLParser):
    """Strip HTML tags; collapse whitespace."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style", "noscript"}
        self._in_skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._in_skip += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._in_skip = max(0, self._in_skip - 1)

    def handle_data(self, data):
        if not self._in_skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = " ".join(self._parts)
        return re.sub(r"\s+", " ", text).strip()

BASE_URL = "https://data.colorado.gov"
TIMEOUT = 30


@tool
def list_datasets(query: str) -> str:
    """Search the Colorado Open Data catalog (data.colorado.gov) for datasets matching a topic.

    Use this to discover which datasets exist before querying them. Returns dataset names,
    4x4 IDs (needed for query_dataset), and short descriptions.

    Args:
        query: Topic to search for (e.g. "school for the deaf blind", "housing permits",
               "state expenditures", "medicaid spending")
    """
    url = f"{BASE_URL}/api/catalog/v1"
    params = {"q": query, "limit": 10, "domains": "data.colorado.gov"}
    try:
        response = httpx.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error searching catalog: {e}"

    data = response.json()
    results = []
    for item in data.get("results", []):
        resource = item.get("resource", {})
        results.append({
            "id": resource.get("id"),
            "name": resource.get("name"),
            "description": (resource.get("description") or "")[:400],
            "updated_at": resource.get("updatedAt"),
            "type": resource.get("type"),
        })

    if not results:
        return f"No datasets found for query: '{query}'. Try broader keywords."

    logger.info(f"Found {len(results)} datasets for query '{query}'")
    return json.dumps(results, indent=2)


@tool
def get_dataset_metadata(dataset_id: str) -> str:
    """Get column names, types, and descriptions for a Colorado Open Data dataset.

    Call this before query_dataset so you know which column names and values to use
    in your WHERE clauses.

    Args:
        dataset_id: The 4x4 dataset ID from list_datasets (e.g. 'abc1-def2')
    """
    url = f"{BASE_URL}/api/views/{dataset_id}.json"
    try:
        response = httpx.get(url, timeout=TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error fetching metadata for {dataset_id}: {e}"

    data = response.json()
    columns = [
        {
            "name": col.get("name"),
            "fieldName": col.get("fieldName"),
            "dataTypeName": col.get("dataTypeName"),
            "description": (col.get("description") or "")[:200],
        }
        for col in data.get("columns", [])
    ]

    metadata = {
        "id": dataset_id,
        "name": data.get("name"),
        "description": (data.get("description") or "")[:600],
        "rowsUpdatedAt": data.get("rowsUpdatedAt"),
        "columns": columns,
    }

    logger.info(f"Fetched metadata for dataset {dataset_id}: {len(columns)} columns")
    return json.dumps(metadata, indent=2)


@tool
def query_dataset(
    dataset_id: str,
    where_clause: str = "",
    select_columns: str = "",
    order_by: str = "",
    limit: int = 100,
) -> str:
    """Query a Colorado Open Data dataset using SoQL (Socrata Query Language).

    Always call get_dataset_metadata first to learn the exact column names.
    String values in WHERE clauses must be single-quoted (e.g., agency='CDHS').
    Numeric comparisons do not need quotes (e.g., fiscal_year=2023).

    Args:
        dataset_id: The 4x4 dataset ID (e.g. 'abc1-def2')
        where_clause: SoQL filter, e.g. "agency_name='Department of Human Services'"
                      or "fiscal_year >= '2015' AND fiscal_year <= '2024'"
        select_columns: Comma-separated column names to return (empty = all columns)
        order_by: Column to sort by, e.g. "fiscal_year ASC" or "amount DESC"
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
        response = httpx.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error querying dataset {dataset_id}: {e}"

    data = response.json()
    if not data:
        return (
            f"No rows returned from dataset {dataset_id}. "
            f"Check that column names and string values in the WHERE clause are correct. "
            f"Use get_dataset_metadata to verify column names."
        )

    logger.info(f"Query returned {len(data)} rows from dataset {dataset_id}")
    return json.dumps(data, indent=2)


@tool
def fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and return its readable text content (HTML tags stripped).

    Use this to retrieve information from Colorado government sites that don't have
    structured APIs, such as:
    - leg.colorado.gov — bill search, fiscal notes, Long Bill documents
    - ospb.colorado.gov — Governor's budget requests and forecasts
    - leg.colorado.gov/offices/joint-budget-committee — JBC documents and hearings

    Args:
        url: Full URL to fetch (https://...)
        max_chars: Maximum characters to return (default 8000, increase for longer docs)
    """
    try:
        headers = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}
        response = httpx.get(url, headers=headers, timeout=TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"

    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        parser = _TextExtractor()
        parser.feed(response.text)
        text = parser.get_text()
    else:
        text = response.text

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated — {len(text) - max_chars} chars remaining]"

    logger.info(f"Fetched {url}: {len(text)} chars")
    return text
