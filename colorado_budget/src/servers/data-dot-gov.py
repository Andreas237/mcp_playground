import argparse
import json
import os
import sys
from pathlib import Path
import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_api_keys

mcp = FastMCP("data-dot-gov", instructions="Provides tools for accessing the APIs on data.gov")

server_base_url='https://api.gsa.gov'
# Harvest-record endpoints are served by catalog.data.gov (not the GSA API) and
# need no API key.
catalog_base_url='https://catalog.data.gov'
HEADERS = {}

def _set_data_dot_gov() -> str:
    load_api_keys()
    key = os.environ.get("DATA_DOT_GOV_API_KEY")
    if key is None:
        raise RuntimeError("DATA_DOT_GOV_API_KEY not set. Add it to colorado_budget/.env")
    global PARAMS
    HEADERS['X-Api-Key'] = key

def _get_exa():
    from exa_py import Exa
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY not set")
    return Exa(api_key=key)

def _ensure_key():
    """Populate HEADERS with the API key if it isn't set yet.

    The server's __main__ calls _set_data_dot_gov() at startup, but calling this
    lazily lets the tools also work when imported directly (e.g. in tests).
    """
    if "X-Api-Key" not in HEADERS:
        _set_data_dot_gov()

def _harvest_record_url(record_id: str, suffix: str = "") -> str:
    """Build a catalog.data.gov harvest_record URL.

    Accepts either a bare UUID or the full harvest_record URL that appears in a
    search_datasets() result's `harvest_record` field, and appends an optional
    '/raw' or '/transformed' suffix.
    """
    if record_id.startswith("http"):
        rec = record_id.rstrip("/")
        for s in ("/raw", "/transformed"):
            if rec.endswith(s):
                rec = rec[: -len(s)]
        return rec + suffix
    return f"{catalog_base_url}/harvest_record/{record_id}{suffix}"

@mcp.tool()
def get_number_data_publishing_organizations():
    """Find the number of organizations that publish data on data.gov.
    Returns an integer as the number of publishing organizations, -1 if none found


    Retrieves the number of organizations.  Each response looks like
    {
  "organizations": [
    {
      "id": UUID,
      "name": Organization display name,
      "slug": URL-friendly identifier, usable as org_slug in search,
      "organization_type": Type of organization: [Federal Government, State Government, City Government, County Government, University, Tribal, or Non-Profit],
      "aliases": Alternative names or abbreviations for the organization,
      "dataset_count": Number of datasets published by this organization
    }
  ],
  "total": total number of publishing organizations
}
    """
    global PARAMS
    URL = f'{server_base_url}/technology/datagov/v4/organizations'
    r = httpx.get(URL, headers=HEADERS)
    logger.info(f'data.gov organzations data: {r.json()}')
    return r.json().get('total', 0)

@mcp.tool()
def get_number_data_publishing_organization_url_slugs():
    """Find the URL slugs of organizations that publish data on data.gov
    Returns tuple: (name, slug)
    """
    global PARAMS
    URL = f'{server_base_url}/technology/datagov/v4/organizations'
    r = httpx.get(URL, headers=HEADERS)
    org_name_tuples = list(map(lambda item: (item.get('name', 'NO NAME'), item.get('slug', 'NO SLUG')), r.json().get('organizations')))
    return org_name_tuples

@mcp.tool()
def search_datasets(query: str = "", sort: str = "relevance", per_page: int = 10,
                    org_slug: str = "", org_type: str = "", keyword: str = "",
                    after: str = ""):
    """Search datasets on data.gov by keyword, organization, type, and tag.

    Wraps GET /technology/datagov/v4/search. Returns a compact, LLM-friendly page
    of results plus an `after` cursor for pagination (the catalog has no total
    count for searches; page with the cursor).

    Args:
        query: Full-text search terms, e.g. "colorado water quality". Empty
               returns the most relevant/popular datasets.
        sort: One of "relevance", "popularity", "distance", "last_harvested_date".
        per_page: Number of results to return (default 10).
        org_slug: Restrict to one organization by slug (e.g. "census"). Get slugs
                  from get_number_data_publishing_organization_url_slugs().
        org_type: Restrict by organization type, e.g. "Federal Government",
                  "State Government", "University".
        keyword: Restrict to datasets tagged with this keyword (see get_keywords()).
                 Comma-separate to require multiple keywords.
        after: Pagination cursor from a previous call; pass it to get the next page.

    Returns dict:
    {
      "count_returned": int,
      "after": cursor string for the next page (null when no more),
      "results": [ {title, identifier, slug, organization, keyword,
                    last_harvested_date, harvest_record, description} ]
    }
    """
    _ensure_key()
    params: dict = {"sort": sort, "per_page": per_page}
    if query:
        params["q"] = query
    if org_slug:
        params["org_slug"] = org_slug
    if org_type:
        params["org_type"] = org_type
    if keyword:
        params["keyword"] = [k.strip() for k in keyword.split(",") if k.strip()]
    if after:
        params["after"] = after
    URL = f'{server_base_url}/technology/datagov/v4/search'
    r = httpx.get(URL, headers=HEADERS, params=params, timeout=30)
    data = r.json()
    results = []
    for item in data.get("results", []) or []:
        org = item.get("organization") or {}
        desc = item.get("description") or ""
        results.append({
            "title": item.get("title"),
            "identifier": item.get("identifier"),
            "slug": item.get("slug"),
            "organization": org.get("name") if isinstance(org, dict) else org,
            "keyword": item.get("keyword"),
            "last_harvested_date": item.get("last_harvested_date"),
            "harvest_record": item.get("harvest_record"),
            "description": (desc[:300] + "…") if len(desc) > 300 else desc,
        })
    logger.info(f"data.gov search '{query}': {len(results)} results")
    return {"count_returned": len(results), "after": data.get("after"), "results": results}

@mcp.tool()
def get_keywords(size: int = 50, min_count: int = 1):
    """Get commonly used dataset keywords (tags) on data.gov with usage counts.

    Wraps GET /technology/datagov/v4/keywords. Use this to discover valid
    `keyword` filters for search_datasets() and to gauge what topics the catalog
    covers. Keywords are returned most-used first.

    Args:
        size: Maximum number of keywords to return (1-1000, default 50).
        min_count: Only return keywords used by at least this many datasets.

    Returns the API response:
    {"keywords": [{"keyword": str, "count": int}], "size": int,
     "min_count": int, "total": int}
    """
    _ensure_key()
    URL = f'{server_base_url}/technology/datagov/v4/keywords'
    r = httpx.get(URL, headers=HEADERS, params={"size": size, "min_count": min_count}, timeout=30)
    return r.json()

@mcp.tool()
def search_locations(query: str, size: int = 10):
    """Search for place names on data.gov for spatial filtering.

    Wraps GET /technology/datagov/v4/locations/search. Returns matching locations
    with their ids; pass an id to get_location_geometry() to get its boundary.

    Args:
        query: Partial or full location name, e.g. "colorado", "denver county".
        size: Maximum number of matches to return (default 10).

    Returns: {"locations": [{"id": str, "display_name": str}], "size": int, "total": int}
    """
    _ensure_key()
    URL = f'{server_base_url}/technology/datagov/v4/locations/search'
    r = httpx.get(URL, headers=HEADERS, params={"q": query, "size": size}, timeout=30)
    return r.json()

@mcp.tool()
def get_location_geometry(location_id: str):
    """Get the geographic boundary (GeoJSON) for a data.gov location id.

    Wraps GET /technology/datagov/v4/location/{location_id}. Get the id from
    search_locations() (e.g. "6" is Colorado). The API returns the geometry as a
    GeoJSON string; this tool parses it into an object.

    Args:
        location_id: The location id from search_locations().

    Returns: {"id": str, "geometry": <GeoJSON geometry dict>}
    """
    _ensure_key()
    URL = f'{server_base_url}/technology/datagov/v4/location/{location_id}'
    r = httpx.get(URL, headers=HEADERS, timeout=30)
    data = r.json()
    geom = data.get("geometry")
    if isinstance(geom, str):
        try:
            data["geometry"] = json.loads(geom)
        except (json.JSONDecodeError, TypeError):
            pass
    return data

@mcp.tool()
def get_harvest_record(record_id: str):
    """Get metadata about how a dataset was harvested (ingested) into data.gov.

    Wraps GET https://catalog.data.gov/harvest_record/{id}. The id (or full URL)
    comes from a search_datasets() result's `harvest_record` field. Harvest
    endpoints are served by catalog.data.gov, not the GSA API, and need no key.

    Args:
        record_id: A harvest record UUID, or the full harvest_record URL from a
                   search result.

    Returns the harvest record JSON (action, status, date_created, date_finished,
    harvest_job_id, source_hash, ckan_id, ...), or a {found: false, ...} dict on
    a non-200 response.
    """
    url = _harvest_record_url(record_id)
    r = httpx.get(url, timeout=30, follow_redirects=True)
    if r.status_code != 200:
        return {"found": False, "http_status": r.status_code, "url": url}
    return r.json()

@mcp.tool()
def get_harvest_record_raw(record_id: str):
    """Get the original, unmodified source metadata for a harvest record.

    Wraps GET https://catalog.data.gov/harvest_record/{id}/raw — the dataset's
    metadata exactly as harvested from its source (usually DCAT JSON, sometimes
    XML or text). Returned as a string.

    Args:
        record_id: A harvest record UUID or the full harvest_record URL.
    """
    url = _harvest_record_url(record_id, "/raw")
    r = httpx.get(url, timeout=30, follow_redirects=True)
    if r.status_code != 200:
        return f"Not found (HTTP {r.status_code}) for {url}"
    return r.text

@mcp.tool()
def get_harvest_record_transformed(record_id: str):
    """Get the DCAT-US transformed metadata for a harvest record.

    Wraps GET https://catalog.data.gov/harvest_record/{id}/transformed — the
    source metadata after data.gov normalizes it to the DCAT-US schema.

    NOTE: this endpoint returns Not Found for harvest records that have no stored
    transformed payload (observed for ordinary dataset records); in that case use
    get_harvest_record_raw() for the as-harvested metadata.

    Args:
        record_id: A harvest record UUID or the full harvest_record URL.
    """
    url = _harvest_record_url(record_id, "/transformed")
    r = httpx.get(url, timeout=30, follow_redirects=True)
    if r.status_code != 200:
        return {"found": False, "http_status": r.status_code, "url": url,
                "message": "No transformed payload for this record; try get_harvest_record_raw."}
    return r.json()

if __name__ == '__main__':
    _set_data_dot_gov()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="streamable-http",
                        choices=["streamable-http", "sse", "stdio"])
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    mcp.settings.port = args.port
    logger.info(f"Starting data.gov MCP server ({args.transport}, port {args.port})")
    mcp.run(transport=args.transport)
