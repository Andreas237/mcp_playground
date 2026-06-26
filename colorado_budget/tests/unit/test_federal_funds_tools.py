"""
Unit tests for federal_funds.py helper functions and MCP tools.
All network calls (USAspending.gov) are mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from servers.federal_funds import (
    _award_type_codes,
    _co_filters,
    _fy_time_period,
    colorado_federal_summary,
    federal_funding_by_agency,
    search_federal_awards,
    top_federal_recipients,
)

# ---------------------------------------------------------------------------
# _fy_time_period
# ---------------------------------------------------------------------------

def test_fy_time_period_maps_federal_year():
    tp = _fy_time_period("2024")
    assert tp == [{"start_date": "2023-10-01", "end_date": "2024-09-30"}]


def test_fy_time_period_bad_input_falls_back():
    tp = _fy_time_period("not-a-year")
    assert tp[0]["start_date"].endswith("-10-01")
    assert tp[0]["end_date"].endswith("-09-30")


# ---------------------------------------------------------------------------
# _award_type_codes
# ---------------------------------------------------------------------------

def test_award_type_codes_grants_default():
    assert _award_type_codes("grants") == ["02", "03", "04", "05"]


def test_award_type_codes_contracts():
    assert _award_type_codes("contracts") == ["A", "B", "C", "D"]


def test_award_type_codes_unknown_defaults_to_grants():
    assert _award_type_codes("nonsense") == ["02", "03", "04", "05"]


# ---------------------------------------------------------------------------
# _co_filters
# ---------------------------------------------------------------------------

def test_co_filters_shape():
    f = _co_filters("2024", "grants")
    assert f["recipient_locations"] == [{"country": "USA", "state": "CO"}]
    assert f["award_type_codes"] == ["02", "03", "04", "05"]
    assert f["time_period"][0]["start_date"] == "2023-10-01"


# ---------------------------------------------------------------------------
# colorado_federal_summary
# ---------------------------------------------------------------------------

def test_colorado_federal_summary_returns_total():
    fake = {"name": "Colorado", "total_prime_amount": 63513679552.77,
            "total_prime_awards": 12345, "population": 5900000,
            "median_household_income": 90000}
    with patch("servers.federal_funds._get", return_value=fake):
        result = colorado_federal_summary("2024")
    data = json.loads(result)
    assert data["state"] == "Colorado"
    assert data["total_federal_prime_amount"] == 63513679552.77
    assert data["fiscal_year"] == "2024"


def test_colorado_federal_summary_handles_failure():
    with patch("servers.federal_funds._get", return_value=None):
        result = colorado_federal_summary("2024")
    data = json.loads(result)
    assert data["found"] is False


# ---------------------------------------------------------------------------
# federal_funding_by_agency
# ---------------------------------------------------------------------------

def test_federal_funding_by_agency_ranks_agencies():
    fake = {"results": [
        {"name": "Department of Health and Human Services", "amount": 10854416758},
        {"name": "Department of Transportation", "amount": 1148384053},
    ]}
    with patch("servers.federal_funds._post", return_value=fake):
        result = federal_funding_by_agency("2024", "grants")
    data = json.loads(result)
    assert len(data["awarding_agencies"]) == 2
    assert data["awarding_agencies"][0]["agency"].startswith("Department of Health")


def test_federal_funding_by_agency_failure():
    with patch("servers.federal_funds._post", return_value=None):
        result = federal_funding_by_agency("2024")
    assert json.loads(result)["found"] is False


# ---------------------------------------------------------------------------
# top_federal_recipients
# ---------------------------------------------------------------------------

def test_top_federal_recipients_returns_list():
    fake = {"results": [
        {"name": "STATE OF COLORADO - DEPT OF HEALTH CARE POLICY & FINANCING", "amount": 8701740515},
        {"name": "COLORADO DEPARTMENT OF TRANSPORTATION", "amount": 815915683},
    ]}
    with patch("servers.federal_funds._post", return_value=fake):
        result = top_federal_recipients("2024", "grants", limit=2)
    data = json.loads(result)
    assert len(data["top_recipients"]) == 2
    assert "HEALTH CARE POLICY" in data["top_recipients"][0]["recipient"]


# ---------------------------------------------------------------------------
# search_federal_awards
# ---------------------------------------------------------------------------

def test_search_federal_awards_maps_fields():
    fake = {"results": [
        {"Recipient Name": "COLORADO DEPARTMENT OF TRANSPORTATION",
         "Awarding Agency": "Department of Transportation",
         "Award Amount": 79587165, "Award Type": "04", "Award ID": "ABC123"},
    ]}
    with patch("servers.federal_funds._post", return_value=fake):
        result = search_federal_awards("Colorado Department of Transportation", "2024")
    data = json.loads(result)
    assert len(data["awards"]) == 1
    assert data["awards"][0]["amount"] == 79587165
    assert data["awards"][0]["awarding_agency"] == "Department of Transportation"


def test_search_federal_awards_empty_results():
    with patch("servers.federal_funds._post", return_value={"results": []}):
        result = search_federal_awards("Nonexistent Agency", "2024")
    data = json.loads(result)
    assert data["found"] is False
