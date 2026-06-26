# Trajectory Test Plan — `colorado-cdot` (CDOT)

Tools: `search_cdot`, `find_cdot_budget`, `find_cdot_appropriations`, `find_stip`.
See [README.md](README.md) for the harness and scoring.

## Scope beyond unit tests

Unit tests already prove: trafig/trahrg URL construction across both year forms
(`fy26-27` short and `fy2025-26` full), the `_fy_full`/`_fy_short` conversions,
doc-link extraction, year filtering, and empty-result fallbacks. These trajectory
cases prove the **agent** distinguishes CDOT *budget/plan* (this server) from CDOT
*actuals* (the open-data datasets), and pulls the appropriation by funding source.

## Cases

| # | Question (input) | Expected trajectory | Matcher | Output assertions |
|---|------------------|--------------------|---------|-------------------|
| D1 | "How is CDOT funded — how much comes from federal funds, the HUTF, and the General Fund?" | `find_cdot_appropriations` → `fetch_and_parse_pdf` | in_order | names **HUTF** (cash) and **Federal Funds** as the main sources; notes minimal General Fund |
| D2 | "What's in CDOT's budget for FY2025-26?" | `find_cdot_budget` → (`fetch_webpage` \| `fetch_and_parse_pdf`) | in_order | cites the Budget Allocation Plan; gives total program / funding-source figures |
| D3 | "What major projects are in Colorado's current STIP and how is the capital program funded?" | `find_stip` → (`fetch_webpage` \| `fetch_and_parse_pdf`) | in_order | references the STIP (FY2027–FY2030); names projects or program totals |
| D4 (routing) | "How has CDOT's actual spending and payroll changed since 2018?" | open-data tools (`list_datasets`/`query_dataset`), **not** `find_cdot_budget` | any_order (assert an open-data tool present) | uses CDOT expense/payroll datasets (n5ku-eixc / rkmy-yymq); distinguishes actuals from budget |
| D5 (negative) | "How is the Colorado Department of Agriculture funded?" | must **not** call any `*cdot*` tool (expect `find_agriculture_*`) | any_order (assert CDOT tools absent) | answer about CDA funding, not transportation |

D4 is the important one: "actuals vs. budget" is exactly the split between this server
and the open-data CDOT datasets, and the system prompt tells the agent to use open-data
for actuals. D5 guards against cross-department leakage.

## Runnable cases

```python
from strands_evals import Case

CASES = [
    Case[str, str](
        name="D1-fund-sources",
        input=("How is CDOT funded — how much comes from federal funds, the HUTF, and "
               "the General Fund?"),
        expected_trajectory=["find_cdot_appropriations", "fetch_and_parse_pdf"],
        metadata={"matcher": "in_order",
                  "expect_facts": ["HUTF", "Federal", "General Fund"]},
    ),
    Case[str, str](
        name="D2-budget",
        input="What's in CDOT's budget for FY2025-26?",
        expected_trajectory=["find_cdot_budget"],
        metadata={"matcher": "in_order",
                  "expect_facts": ["Budget Allocation Plan", "Total Program"]},
    ),
    Case[str, str](
        name="D3-stip",
        input=("What major projects are in Colorado's current STIP and how is the "
               "capital program funded?"),
        expected_trajectory=["find_stip"],
        metadata={"matcher": "in_order", "expect_facts": ["STIP"]},
    ),
    Case[str, str](
        name="D4-actuals-vs-budget-routing",
        input="How has CDOT's actual spending and payroll changed since 2018?",
        expected_trajectory=["query_dataset"],   # open-data, not find_cdot_budget
        metadata={"matcher": "any_order",
                  "expect_tools_present": ["list_datasets", "query_dataset"],
                  "forbid_tools": ["find_cdot_budget"]},
    ),
    Case[str, str](
        name="D5-routing-guard-agriculture",
        input="How is the Colorado Department of Agriculture funded?",
        expected_trajectory=["find_agriculture_appropriations"],
        metadata={"matcher": "any_order", "forbid_tools": [
            "search_cdot", "find_cdot_budget", "find_cdot_appropriations", "find_stip"]},
    ),
]
```

## Live-source sanity (no LLM)

- `find_cdot_appropriations("2025-26")` → ≥1 document, `http_status == 200`
  (currently `…/fy2025-26_trafig.pdf` and `…/fy2025-26_trahrg.pdf`; note the JBC
  transportation files mix short `fy26-27` and full `fy2025-26` year forms).
- `find_stip()` → ≥1 document or page from codot.gov (currently the FY2027–FY2030
  STIP executive-summary PDFs).
- `find_cdot_budget("2025-26")` → ≥1 page (the Budget Allocation Plan lives on an
  HTML page; CDOT's deep PDF paths churn, so prefer the page + fetch_webpage).
