# Trajectory Test Plan — `colorado-agriculture` (CDA)

Tools: `search_agriculture`, `find_agriculture_budget`,
`find_agriculture_appropriations`, `find_agriculture_programs`. See
[README.md](README.md) for the harness and scoring.

## Scope beyond unit tests

Unit tests already prove: agrfig URL construction across hosts/cases, doc-link
extraction, year filtering, the browser-User-Agent path (ag.colorado.gov 403s plain
UAs), curated program pages, and empty-result fallbacks. These trajectory cases prove
the **agent** routes Agriculture questions to this server (not CPW, the other small
resource department) and pulls the appropriation by fund type.

## Cases

| # | Question (input) | Expected trajectory | Matcher | Output assertions |
|---|------------------|--------------------|---------|-------------------|
| A1 | "How is the Colorado Department of Agriculture funded — how much from the General Fund versus cash funds?" | `find_agriculture_appropriations` → `fetch_and_parse_pdf` | in_order | small department; mostly **Cash Funds** (industry fees); names the General Fund share |
| A2 | "What did the FY2025-26 state budget include for the Department of Agriculture?" | `find_agriculture_budget` → (`fetch_webpage` \| `fetch_and_parse_pdf`) | in_order | cites the CDA budget briefing / state-budget materials; gives a funding figure |
| A3 | "What programs does the Colorado Department of Agriculture run?" | `find_agriculture_programs` → `fetch_webpage` | in_order | names divisions/programs (Markets, Animal Health, Conservation Services, brand inspection, Colorado Proud) |
| A4 | "Find documents about Colorado agricultural drought / resilience funding." | `search_agriculture` | in_order | returns ag.colorado.gov / leg.colorado.gov links on ag resilience funding |
| A5 (negative) | "How is Colorado Parks & Wildlife funded?" | must **not** call any `*agriculture*` tool (expect `get_sources_and_uses`/`find_cpw_*`) | any_order (assert agriculture tools absent) | answer about CPW funding, not agriculture |

A5 is the key routing guard: CDA and CPW are both small, fee-heavy resource
departments, so we explicitly confirm the agent doesn't cross them.

## Runnable cases

```python
from strands_evals import Case

CASES = [
    Case[str, str](
        name="A1-fund-split",
        input=("How is the Colorado Department of Agriculture funded — how much from "
               "the General Fund versus cash funds?"),
        expected_trajectory=["find_agriculture_appropriations", "fetch_and_parse_pdf"],
        metadata={"matcher": "in_order",
                  "expect_facts": ["Cash Funds", "General Fund"]},
    ),
    Case[str, str](
        name="A2-budget",
        input="What did the FY2025-26 state budget include for the Department of Agriculture?",
        expected_trajectory=["find_agriculture_budget"],
        metadata={"matcher": "in_order", "expect_facts": ["Agriculture", "budget"]},
    ),
    Case[str, str](
        name="A3-programs",
        input="What programs does the Colorado Department of Agriculture run?",
        expected_trajectory=["find_agriculture_programs", "fetch_webpage"],
        metadata={"matcher": "in_order", "expect_facts": ["Markets", "inspection"]},
    ),
    Case[str, str](
        name="A4-search",
        input="Find documents about Colorado agricultural drought / resilience funding.",
        expected_trajectory=["search_agriculture"],
        metadata={"matcher": "in_order", "expect_facts": ["agricultur"]},
    ),
    Case[str, str](
        name="A5-routing-guard-cpw",
        input="How is Colorado Parks & Wildlife funded?",
        expected_trajectory=["get_sources_and_uses"],
        metadata={"matcher": "any_order", "forbid_tools": [
            "search_agriculture", "find_agriculture_budget",
            "find_agriculture_appropriations", "find_agriculture_programs"]},
    ),
]
```

## Live-source sanity (no LLM)

- `find_agriculture_appropriations("2025-26")` → ≥1 document, `http_status == 200`
  (currently `…/fy2025-26_agrfig.pdf`; stem is `agrfig`, no "1").
- `find_agriculture_budget("2025-26")` → ≥1 document (e.g. `fy2025-26_agrbrf.pdf`
  budget briefing) or ≥1 page from ag.colorado.gov.
- ag.colorado.gov 403s plain User-Agents; the browser UA in `agriculture.py` must keep
  working. If page-scrape starts returning 403, refresh the UA.
