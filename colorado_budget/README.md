# Colorado Budget Research Agent

A conversational research tool for investigating Colorado state government spending. Ask broad policy questions in plain English; the agent finds, queries, and synthesizes data from Colorado government sources.

Built as a [Strands](https://strandsagents.com) agent backed by Claude Sonnet.

---

## Quick Start

```bash
cd colorado_budget/src

# Default question (lists available datasets)
python agent.py

# Ask a specific question
python agent.py "What changes have been made to funding for the Colorado School for the Deaf and the Blind in the last 10 years?"

python agent.py "How has Colorado's Medicaid spending changed since 2018?"

python agent.py "What bills passed in 2023 that affected housing affordability funding?"
```

**API key:** The agent uses `ANTHROPIC_API_KEY`. If that's not set, it automatically falls back to `OPENWEBUI_ANTHROPIC_API_KEY`. No `.env` file is required if either variable is in your shell environment.

---

## What It Does

The agent receives a natural-language question and autonomously:
1. Searches the Colorado Open Data catalog for relevant structured datasets
2. Inspects column names, then queries datasets across multiple fiscal years
3. Falls back to fetching pages from Colorado legislative and budget websites when structured data is sparse
4. Synthesizes findings into a sourced, cited answer with dollar amounts and year-over-year trends

---

## Current Tools

| Tool | Source | What it provides |
|------|--------|-----------------|
| `list_datasets` | data.colorado.gov Socrata catalog API | Discover which structured datasets exist for a topic |
| `get_dataset_metadata` | data.colorado.gov Socrata views API | Column names and types before writing queries |
| `query_dataset` | data.colorado.gov Socrata SODA API | SoQL queries; filter by agency, year, fund type, etc. |
| `fetch_webpage` | Any URL | HTML-stripped text from leg.colorado.gov, ospb.colorado.gov, JBC pages |

---

## Key Data Sources

### data.colorado.gov (Socrata SODA API)
Structured, queryable data. Strongest coverage in CDOT; state-wide financial data is limited.

| Dataset | ID | Coverage |
|---------|----|----|
| TOPS — State Revenue & Expenditures | `fjyf-bdat` | All departments, multiple years (through ~2022) |
| CDOT Expenses | `n5ku-eixc` | Current and prior fiscal year |
| CDOT Payroll | `rkmy-yymq` | Current and prior fiscal year |
| Marijuana Tax Revenue | `3sm5-jtur` | Monthly, by county |

### Colorado General Assembly (leg.colorado.gov)
- **Long Bill** — the annual appropriations act (HBxx-1xxx each session), organized by department
- **Fiscal notes** — attached to every bill; show the dollar impact by fund type and agency
- **JBC Appropriations History Reports** — 10-year tables of every department's appropriation
- **Bill search** — search by topic, sponsor, session year, status

### OSPB (ospb.colorado.gov)
- Governor's budget requests (submitted each November)
- Long-range financial forecasts
- Revenue and expenditure breakdowns by fund type

---

## Known Limitations

**Colorado budget PDFs are not yet parseable.** The JBC Appropriations History Reports and Long Bill PDFs are the gold standard for exact line-item figures, but they're delivered through an external document portal (Box) that the `fetch_webpage` tool cannot reach. The agent can identify which documents you need and describe trends from what it can access, but will flag when it cannot retrieve exact dollar figures.

**data.colorado.gov coverage is uneven.** The portal has detailed CDOT data and a state-wide TOPS overview, but many agencies (CDHS, CDE, HCPF) don't publish queryable line-item data there.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the roadmap to address both gaps.

---

## Project Structure

```
colorado_budget/
├── README.md              ← you are here
├── ARCHITECTURE.md        ← design doc and roadmap (iterate before building)
└── src/
    ├── agent.py                    ← entry point; Strands agent wiring
    ├── colorado_open_data_tools.py ← Strands tools: SODA API + fetch_webpage
    └── utils.py                    ← API key loading with env var fallback
```
