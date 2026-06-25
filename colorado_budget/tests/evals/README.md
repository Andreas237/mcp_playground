# Agent Evaluation Framework

## Overview

This eval suite tests the **quality** of the Colorado Budget Research Agent's responses — not just whether they run, but whether they're accurate, well-cited, and useful for fact-checking political claims.

It uses an **LLM-as-judge** pattern (Claude Haiku) to score responses on three dimensions, inspired by the approach from Vercel AI SDK's testing guidance: mock for unit tests, LLM judge for quality assessment.

---

## Architecture

```
evals/
├── dataset.py      ← gold-standard Q&A pairs with known correct facts
├── judge.py        ← Claude Haiku scores responses on 3 dimensions
├── run_evals.py    ← runner: agent → response → judge → results JSON
└── results/        ← timestamped JSON outputs (gitignored)
```

---

## Scoring Dimensions

| Dimension | What it measures | Weight |
|---|---|---|
| **factual_accuracy** | Are specific claims (numbers, dates, agencies) correct? | 1/3 |
| **fund_type_accuracy** | Are General Fund / Cash Fund / Federal Fund correctly distinguished? | 1/3 |
| **completeness** | Does the response fully address the question? | 1/3 |

Plus two **programmatic checks** (no LLM needed):
- `citation_count` — number of `[N]` inline citations in the response
- `has_sources_section` — does the response have a Sources footer?

---

## Running Evals

```bash
cd colorado_budget/tests

# All 5 eval cases (takes ~10 minutes, costs ~$0.50 in API)
python evals/run_evals.py

# Single case
python evals/run_evals.py --cases csdb_10yr

# Multiple specific cases
python evals/run_evals.py --cases csdb_10yr jbc_process housing_permitting

# Preview questions without running
python evals/run_evals.py --dry-run
```

---

## Eval Cases

| ID | Question summary | Tests |
|---|---|---|
| `csdb_10yr` | CSDB funding changes over 10 years | Factual data from JBC PDFs, GF vs capital |
| `general_fund_structure` | What are CO's fund types? | Fund type knowledge |
| `cdot_spending` | CDOT spending since 2018 | Structured data (SODA API), federal funds |
| `jbc_process` | How does the JBC budget process work? | Web search + gov docs |
| `housing_permitting` | Permitting impact on affordable housing | Web search fallback, policy context |

---

## Adding New Eval Cases

Add a new `EvalCase` to `dataset.py`:

```python
EvalCase(
    id="my_new_case",
    question="Your question here",
    expected_facts=["fact1", "fact2"],        # strings that should appear in a correct answer
    fund_types_expected=["General Fund"],      # fund types the response should mention
    notes="Ground truth for the judge: what the correct answer actually contains...",
)
```

---

## Interpreting Results

| Score | Meaning |
|---|---|
| 0.9–1.0 | Excellent — accurate, complete, well-cited |
| 0.7–0.9 | Good — minor gaps or imprecision |
| 0.5–0.7 | Needs improvement — missing key facts or fund type confusion |
| < 0.5 | Poor — significant inaccuracies or incomplete |

Target: **≥ 0.75 overall** across all 5 cases.

The `citation_count` metric should be **≥ 2** for research questions. Questions that require synthesis across multiple PDFs should show **≥ 3 citations**.

---

## What Vercel AI SDK Taught Us

Vercel AI SDK's testing docs focus on **mocking the LLM** for deterministic unit tests — useful for testing the *plumbing* (does the agent start? do tools get called?). For quality measurement, we need a separate eval layer with an LLM judge.

The key insight: **unit tests check correctness of code; evals check quality of outputs.** Both are necessary.
