"""
Unit tests for the SODA API tools in servers/colorado_open_data.py.
All HTTP calls are mocked — no network required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.colorado_open_data import get_dataset_metadata, list_datasets, query_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(payload, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = payload
    mock.raise_for_status.return_value = None
    return mock


def _mock_http_error(status_code=500):
    import httpx
    mock = MagicMock()
    mock.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}", request=MagicMock(), response=MagicMock()
    )
    return mock


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

CATALOG_RESPONSE = {
    "results": [
        {
            "resource": {
                "id": "fjyf-bdat",
                "name": "Colorado Transparency Online Project Statewide",
                "description": "Statewide revenue and expenditures for Colorado.",
                "updatedAt": "2024-01-15",
            }
        },
        {
            "resource": {
                "id": "abc1-2345",
                "name": "State Employee Salaries",
                "description": "Annual salaries for state employees.",
                "updatedAt": "2024-03-01",
            }
        },
    ]
}


def test_list_datasets_returns_json_array():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(CATALOG_RESPONSE)
        result = list_datasets("spending")
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == "fjyf-bdat"
    assert "Statewide" in data[0]["name"]


def test_list_datasets_passes_query_param():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(CATALOG_RESPONSE)
        list_datasets("CSDB deaf blind")
    called_params = mock_get.call_args.kwargs.get("params", {})
    assert called_params.get("q") == "CSDB deaf blind"


def test_list_datasets_empty_returns_helpful_message():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response({"results": []})
        result = list_datasets("xyzzy nonexistent topic")
    assert "No datasets found" in result
    assert "broader keywords" in result


def test_list_datasets_truncates_long_descriptions():
    long_desc = "x" * 1000
    catalog = {"results": [{"resource": {"id": "aa11-bb22", "name": "Test", "description": long_desc, "updatedAt": None}}]}
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(catalog)
        result = list_datasets("test")
    data = json.loads(result)
    assert len(data[0]["description"]) <= 400


def test_list_datasets_http_error_returns_error_string():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_http_error()
        result = list_datasets("spending")
    assert result.startswith("Error searching catalog")


# ---------------------------------------------------------------------------
# get_dataset_metadata
# ---------------------------------------------------------------------------

METADATA_RESPONSE = {
    "name": "Colorado TOPS",
    "description": "Statewide transparency dataset.",
    "rowsUpdatedAt": 1700000000,
    "columns": [
        {"name": "Agency", "fieldName": "agency_name", "dataTypeName": "text", "description": "Agency name"},
        {"name": "Fiscal Year", "fieldName": "fiscal_year", "dataTypeName": "text", "description": ""},
        {"name": "Amount", "fieldName": "amount", "dataTypeName": "number", "description": "Dollar amount"},
    ],
}


def test_get_metadata_returns_columns():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(METADATA_RESPONSE)
        result = get_dataset_metadata("fjyf-bdat")
    data = json.loads(result)
    assert data["id"] == "fjyf-bdat"
    assert data["name"] == "Colorado TOPS"
    assert len(data["columns"]) == 3
    field_names = [c["fieldName"] for c in data["columns"]]
    assert "agency_name" in field_names
    assert "fiscal_year" in field_names


def test_get_metadata_truncates_column_descriptions():
    long_col_desc = "y" * 500
    meta = {**METADATA_RESPONSE, "columns": [
        {"name": "Col", "fieldName": "col", "dataTypeName": "text", "description": long_col_desc}
    ]}
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(meta)
        result = get_dataset_metadata("aa11-bb22")
    data = json.loads(result)
    assert len(data["columns"][0]["description"]) <= 200


def test_get_metadata_http_error():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_http_error()
        result = get_dataset_metadata("bad-id00")
    assert "Error fetching metadata" in result
    assert "bad-id00" in result


# ---------------------------------------------------------------------------
# query_dataset
# ---------------------------------------------------------------------------

QUERY_ROWS = [
    {"agency_name": "CDHS", "fiscal_year": "2023", "amount": "1500000"},
    {"agency_name": "CDHS", "fiscal_year": "2024", "amount": "1600000"},
]


def test_query_dataset_returns_rows():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(QUERY_ROWS)
        result = query_dataset("fjyf-bdat", where_clause="agency_name='CDHS'")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["agency_name"] == "CDHS"


def test_query_dataset_passes_soql_params():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(QUERY_ROWS)
        query_dataset(
            "fjyf-bdat",
            where_clause="fiscal_year='2023'",
            select_columns="agency_name,amount",
            order_by="amount DESC",
            limit=50,
        )
    params = mock_get.call_args.kwargs.get("params", {})
    assert params["$where"] == "fiscal_year='2023'"
    assert params["$select"] == "agency_name,amount"
    assert params["$order"] == "amount DESC"
    assert params["$limit"] == 50


def test_query_dataset_caps_limit_at_1000():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(QUERY_ROWS)
        query_dataset("fjyf-bdat", limit=9999)
    params = mock_get.call_args.kwargs.get("params", {})
    assert params["$limit"] == 1000


def test_query_dataset_empty_returns_helpful_message():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_response([])
        result = query_dataset("fjyf-bdat", where_clause="agency_name='NONEXISTENT'")
    assert "No rows returned" in result
    assert "get_dataset_metadata" in result


def test_query_dataset_http_error():
    with patch("servers.colorado_open_data.httpx.get") as mock_get:
        mock_get.return_value = _mock_http_error()
        result = query_dataset("fjyf-bdat")
    assert "Error querying" in result
