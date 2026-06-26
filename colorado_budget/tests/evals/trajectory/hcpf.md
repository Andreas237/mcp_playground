# Trajectory Test Plan — `colorado-hcpf` (HCPF / Medicaid)

Tools: `search_hcpf`, `find_hcpf_budget_request`, `find_caseload_reports`,
`find_hcpf_appropriations`. See [README.md](README.md) for the harness and scoring.

## Scope beyond unit tests

Unit tests already prove: hcpfig URL construction, doc/Excel link extraction, year
filtering, the browser-User-Agent path (hcpf.colorado.gov 403s plain UAs), and the
empty-result fallbacks. These trajectory cases prove the **agent** reaches for the
HCPF server (not the generic legislature server) for Medicaid questions, and chains
request→appropriation and HCPF→federal-funds correctly.

## Cases

| # | Question (input) | Expected trajectory | Matcher | Output assertions |
|---|------------------|--------------------|---------|-------------------|
| H1 | "How much did HCPF request for Medicaid in FY2025-26, and how did it compare to what the JBC approved?" | `find_hcpf_budget_request` → `find_hcpf_appropriations` → `fetch_and_parse_pdf` | in_order | mentions **General Fund**; gives a request figure *and* an approved figure; names the gap |
| H2 | "How many Coloradans are enrolled in Medicaid and how is enrollment trending?" | `find_caseload_reports` → (`fetch_webpage` \| `fetch_and_parse_pdf`) | in_order | cites a caseload/enrollment count; references PECR/monthly report |
| H3 | "What share of HCPF's budget comes from federal funds vs the General Fund?" | `find_hcpf_appropriations` (or `find_hcpf_budget_request`) + `top_federal_recipients` (or `search_federal_awards`) | any_order | states the ~50/50 GF/Federal split; cites both a state budget doc and USAspending |
| H4 | "Find HCPF documents about provider rate increases for FY2026-27." | `search_hcpf` | in_order | returns hcpf.colorado.gov links; mentions provider rates |
| H5 (negative) | "What is Colorado's TABOR surplus this year?" | must **not** call any `*_hcpf*` tool (expect `find_legislative_forecast`/`find_tabor_resources`) | any_order (assert HCPF tools absent) | answer about TABOR, not Medicaid |

H5 is a *routing* guard: it confirms the broad HCPF tool descriptions don't cause the
agent to grab an HCPF tool for an unrelated revenue question.

## Runnable cases

```python
from strands_evals import Case

CASES = [
    Case[str, str](
        name="H1-request-vs-approved",
        input=("How much did HCPF request for Medicaid in FY2025-26, and how did it "
               "compare to what the JBC approved?"),
        expected_trajectory=["find_hcpf_budget_request", "find_hcpf_appropriations",
                             "fetch_and_parse_pdf"],
        metadata={"matcher": "in_order",
                  "expect_facts": ["General Fund", "request", "approved"]},
    ),
    Case[str, str](
        name="H2-caseload",
        input="How many Coloradans are enrolled in Medicaid and how is enrollment trending?",
        expected_trajectory=["find_caseload_reports"],
        metadata={"matcher": "in_order", "expect_facts": ["caseload", "enroll"]},
    ),
    Case[str, str](
        name="H3-fund-split",
        input="What share of HCPF's budget comes from federal funds vs the General Fund?",
        expected_trajectory=["find_hcpf_appropriations", "top_federal_recipients"],
        metadata={"matcher": "any_order", "expect_facts": ["General Fund", "federal"]},
    ),
    Case[str, str](
        name="H4-search",
        input="Find HCPF documents about provider rate increases for FY2026-27.",
        expected_trajectory=["search_hcpf"],
        metadata={"matcher": "in_order", "expect_facts": ["provider rate"]},
    ),
    Case[str, str](
        name="H5-routing-guard-tabor",
        input="What is Colorado's TABOR surplus this year?",
        expected_trajectory=["find_legislative_forecast"],
        metadata={"matcher": "any_order", "forbid_tools": [
            "search_hcpf", "find_hcpf_budget_request", "find_caseload_reports",
            "find_hcpf_appropriations"]},
    ),
]
```

## Live-source sanity (no LLM)

Independent of the agent, the tools should keep returning reachable docs as CO sites
change. A cheap guard (run quarterly), distinct from mocked unit tests:

- `find_hcpf_appropriations("2025-26")` → ≥1 document, `http_status == 200`
  (currently `…/fy2025-26_hcpfig1.pdf`).
- `find_caseload_reports("2025")` → ≥1 page from `hcpf.colorado.gov` (browser UA must
  still bypass the 403).
- If hcpf.colorado.gov starts returning 403 again, the User-Agent in `hcpf.py` needs
  refreshing — that is the most likely breakage.
