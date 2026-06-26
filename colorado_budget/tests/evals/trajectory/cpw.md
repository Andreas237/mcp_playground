# Trajectory Test Plan — `colorado-parks-wildlife` (CPW)

Tools: `search_cpw`, `find_cpw_financial_reports`, `get_sources_and_uses`,
`find_cpw_appropriations`. See [README.md](README.md) for the harness and scoring.

## Scope beyond unit tests

Unit tests already prove: natfig/natbrf URL construction across hosts/cases, doc-link
extraction, year filtering, the Sources & Uses fact-sheet HEAD gate, and empty-result
fallbacks. These trajectory cases prove the **agent** recognizes CPW as
**enterprise/cash-funded** (not General Fund), reaches for `get_sources_and_uses` to
explain the funding model, and uses the Natural Resources appropriation rather than a
generic search.

## Cases

| # | Question (input) | Expected trajectory | Matcher | Output assertions |
|---|------------------|--------------------|---------|-------------------|
| W1 | "How is Colorado Parks & Wildlife funded — how much comes from the General Fund versus licenses and fees?" | `get_sources_and_uses` (± `find_cpw_appropriations`) | in_order | states CPW is enterprise/cash-funded; names licenses / park passes / GOCO; says **minimal General Fund** |
| W2 | "What do CPW's recent financial reports to the Commission show about its cash fund balances and revenue?" | `find_cpw_financial_reports` → `fetch_and_parse_pdf` | in_order | cites a Commission financial report; references cash-fund balance or license/park revenue |
| W3 | "How much was appropriated to Parks & Wildlife for FY2025-26, and from which fund types?" | `find_cpw_appropriations` → `fetch_and_parse_pdf` | in_order | cites the JBC Natural Resources (natfig) doc; breaks the appropriation down by fund type (cash/federal dominate) |
| W4 | "Find CPW documents about the Keep Colorado Wild pass." | `search_cpw` | in_order | returns cpw.state.co.us links; mentions Keep Colorado Wild / state park pass |
| W5 (negative) | "How much federal Medicaid money does Colorado receive?" | must **not** call any `*cpw*` tool (expect `top_federal_recipients`/`find_hcpf_*`) | any_order (assert CPW tools absent) | answer about Medicaid/federal funds, not parks |

W1 is the headline case — the fund-type contrast is CPW's whole reason to be a separate
server. W5 guards against the broad CPW descriptions pulling the agent off-topic.

## Runnable cases

```python
from strands_evals import Case

CASES = [
    Case[str, str](
        name="W1-funding-model",
        input=("How is Colorado Parks & Wildlife funded — how much comes from the "
               "General Fund versus licenses and fees?"),
        expected_trajectory=["get_sources_and_uses"],
        metadata={"matcher": "in_order",
                  "expect_facts": ["cash", "license", "General Fund"]},
    ),
    Case[str, str](
        name="W2-financial-reports",
        input=("What do CPW's recent financial reports to the Commission show about its "
               "cash fund balances and revenue?"),
        expected_trajectory=["find_cpw_financial_reports", "fetch_and_parse_pdf"],
        metadata={"matcher": "in_order", "expect_facts": ["cash fund", "revenue"]},
    ),
    Case[str, str](
        name="W3-appropriation",
        input=("How much was appropriated to Parks & Wildlife for FY2025-26, and from "
               "which fund types?"),
        expected_trajectory=["find_cpw_appropriations", "fetch_and_parse_pdf"],
        metadata={"matcher": "in_order",
                  "expect_facts": ["Parks", "Cash Funds", "Federal Funds"]},
    ),
    Case[str, str](
        name="W4-search",
        input="Find CPW documents about the Keep Colorado Wild pass.",
        expected_trajectory=["search_cpw"],
        metadata={"matcher": "in_order", "expect_facts": ["Keep Colorado Wild"]},
    ),
    Case[str, str](
        name="W5-routing-guard-medicaid",
        input="How much federal Medicaid money does Colorado receive?",
        expected_trajectory=["top_federal_recipients"],
        metadata={"matcher": "any_order", "forbid_tools": [
            "search_cpw", "find_cpw_financial_reports", "get_sources_and_uses",
            "find_cpw_appropriations"]},
    ),
]
```

## Live-source sanity (no LLM)

- `find_cpw_appropriations("2025-26")` → ≥1 document, `http_status == 200`
  (currently `…/fy2025-26_natfig.pdf` on content.leg.colorado.gov; note the stem is
  `natfig`, not `natfig1`).
- `find_cpw_financial_reports("2025")` → ≥1 PDF from `cpw.state.co.us/.../dam/…`
  (Commission financial reports live under randomized `dam/{hash}/` paths, so this
  must come from page-scrape/Exa, never a constructed URL).
- `get_sources_and_uses()` → ≥1 resource; the stable fact-sheet URL currently
  HEAD-404s but Exa still surfaces it — if it disappears from Exa too, refresh the
  funding-page scrape.
