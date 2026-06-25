"""
Integration tests — real calls to data.colorado.gov (no LLM, no MCP server).
Requires network access. Marked with @pytest.mark.integration.

Notes on the real API behavior:
- Catalog search needs short, broad queries ("CDOT", "budget") — multi-word phrases often return nothing
- TOPS dataset (fjyf-bdat) metadata has 0 columns and its resource is 403 Forbidden
- CDOT datasets (n5ku-eixc, rkmy-yymq) are reliably accessible

Run with:
    pytest tests/integration/test_open_data_live.py -v
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.colorado_open_data import get_dataset_metadata, list_datasets, query_dataset

pytestmark = pytest.mark.integration


def test_list_datasets_budget_returns_results():
    result = list_datasets("budget")
    # API returns results for short queries; "state expenditures budget" often returns nothing
    if result.startswith("No datasets"):
        pytest.skip(f"Catalog returned no results for 'budget': {result}")
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert "id" in item
        assert "name" in item


def test_list_datasets_cdot_returns_results():
    result = list_datasets("CDOT")
    assert not result.startswith("Error"), f"API error: {result[:200]}"
    assert not result.startswith("No datasets"), f"No CDOT datasets found: {result}"
    data = json.loads(result)
    assert len(data) > 0
    names = [d["name"] for d in data]
    assert any("CDOT" in n for n in names), f"No CDOT in names: {names}"


def test_list_datasets_nonexistent_topic_returns_message():
    result = list_datasets("xyzzy_nonexistent_topic_12345")
    assert "No datasets found" in result or result.startswith("Error")


def test_get_cdot_expenses_metadata():
    """CDOT Expenses (n5ku-eixc) reliably has columns and accessible resource."""
    result = get_dataset_metadata("n5ku-eixc")
    assert not result.startswith("Error"), f"Metadata error: {result[:200]}"
    data = json.loads(result)
    assert data["id"] == "n5ku-eixc"
    assert "columns" in data
    assert len(data["columns"]) > 0
    field_names = [c["fieldName"] for c in data["columns"]]
    assert len(field_names) > 0


def test_query_cdot_expenses_returns_rows():
    """CDOT Expenses dataset is consistently queryable."""
    result = query_dataset("n5ku-eixc", limit=5)
    if result.startswith("Error") or result.startswith("No rows"):
        pytest.skip(f"CDOT expenses not queryable: {result[:200]}")
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) > 0


def test_query_dataset_bad_id_returns_error():
    result = query_dataset("xxxx-xxxx")
    assert "Error" in result or "No rows" in result


def test_get_metadata_bad_id_returns_error():
    result = get_dataset_metadata("xxxx-xxxx")
    assert "Error" in result
