# Colorado Budget Research — Architecture & Roadmap

This document captures the design intent and open questions for the expanded system. **Edit this file freely** — it's a thinking canvas, not a spec. Once an approach feels settled, it moves to implementation.

---

## What We're Building

A research system that lets anyone ask broad, natural-language questions about Colorado state spending and get answers grounded in primary government sources. The driving use case is election-season fact-checking: politicians make claims; this tool finds the receipts.

### Example questions the full system should answer

- "What changes have been made to funding for the deaf and blind in Colorado in the last 10 years?"
- "How have permitting regulations impacted the cost of building affordable housing in Colorado since 2020?"
- "Which bills passed in the 2024 session drew from the General Fund vs specific cash funds?"
- "How has CDOT spending on rural roads changed vs urban highways since 2018?"
- "What programs does the Department of Human Services run that are funded entirely by federal dollars?"
- "Which legislators have championed increased mental health funding, and what did those bills actually appropriate?"

---

## Current State (implemented)

A single Strands agent ([`agent.py`](src/agent.py)) orchestrates **7 MCP servers** plus 2 inline tools. On each run the agent spawns every server as a subprocess (streamable-HTTP, one per port), connects with an `MCPClient`, runs the query, and shuts the servers down. The LLM is **configurable** per run via [`config.toml`](config.toml) (see [Decision 2](#decision-2--model--configurable-via-configtoml-)).

**What works now:** dataset discovery and trends (open-data); bill search, fiscal notes, and JBC appropriations (legislature); Governor's budget request and forecasts (OSPB); Legislative Council revenue forecast, TABOR, and tax expenditures (revenue); federal funding by agency/recipient (federal-funds); K-12 Total Program and the HB24-1448 formula (school-finance); and Exa web search. Exact line-item PDFs are reachable via `fetch_and_parse_pdf`.

**Still sparse:** current-year *actual* expenditures (TOPS checkbook is a non-tabular 403 dead end; we rely on appropriations + forecasts instead), and per-district data delivered as Excel (`fetch_and_parse_pdf` is PDF-only).

---

## Implemented MCP Servers & Port Map

All servers live in [`src/servers/`](src/servers/), run on `streamable-http` transport, and are registered in `MCP_SERVERS` in [`agent.py`](src/agent.py). Each is independently runnable (`python servers/<name>.py --port <port>`) and reusable by Claude Code, not just this agent.

| Port | Server (FastMCP name) | File | Tools | Data source |
|------|----------------------|------|-------|-------------|
| 8001 | `colorado-open-data` | `colorado_open_data.py` | `list_datasets`, `get_dataset_metadata`, `query_dataset` | data.colorado.gov (Socrata SODA) — no key |
| 8002 | `web-search` | `web_search.py` | `search_web`, `search_colorado_government` | Exa neural search — `EXA_API_KEY` |
| 8003 | `colorado-legislature` | `legislature.py` | `search_bills`, `get_bill_details`, `get_fiscal_note`, `find_appropriations_documents` | leg.colorado.gov + Exa |
| 8004 | `colorado-ospb` | `ospb.py` | `search_ospb`, `find_governor_budget`, `find_revenue_forecast`, `find_budget_amendments` | ospb.colorado.gov + Exa |
| 8005 | `colorado-revenue` | `revenue.py` | `search_revenue`, `find_legislative_forecast`, `find_tax_expenditure_report`, `find_tabor_resources` | Legislative Council / OSA / DOR + Exa |
| 8006 | `colorado-federal-funds` | `federal_funds.py` | `colorado_federal_summary`, `federal_funding_by_agency`, `top_federal_recipients`, `search_federal_awards` | USAspending.gov v2 API — no key |
| 8007 | `colorado-school-finance` | `school_finance.py` | `search_school_finance`, `find_school_finance_act`, `find_per_pupil_funding`, `find_finance_formula_resources` | CDE (ed.cde.state.co.us) + leg + Exa |

**Inline tools** (in [`src/tools/`](src/tools/), not MCP — passed directly to the agent): `fetch_webpage(url)` and `fetch_and_parse_pdf(url, page_range, keyword_filter)`.

### Implementation patterns

- **Scraper servers** (legislature, ospb, revenue, school-finance): page-scrape known landing pages for document links → **Exa fallback** (host-agnostic, survives site reorganizations) → **HEAD-check** candidate URLs so only reachable docs are returned. Helpers `_fetch_page`, `_extract_*_links`/`_doc_links`, `_head_check`, `_get_exa` are duplicated per server by design (each owns its source). Discovery returns document URLs (and, for school-finance, HTML `pages`) which the agent then reads with `fetch_and_parse_pdf` / `fetch_webpage`.
- **API servers** (open-data, federal-funds): hit a clean REST/JSON API directly and return structured results — no scraping, most reliable.
- **Fiscal-year naming** is inconsistent across CO sources; servers carry helpers (`_fy_variants`, `_fy_time_period`, month/quarter maps) to match the many conventions ("2026-27", "FY2027", "fy26-27", federal Oct–Sep, etc.).
- **All servers require `EXA_API_KEY`** except open-data and federal-funds. Keys load from `colorado_budget/.env` via `utils.load_api_keys` and are **never** committed (see security note).

> **Security:** API keys live only in `colorado_budget/.env` (gitignored). Never put keys in `config.toml`, `ARCHITECTURE.md`, or any tracked file. Each model profile in `config.toml` references its key by env-var name (`api_key_env`), not value.

---

## Architecture Decisions

### Decision 1: Proper MCP servers ✓

Each data source becomes a standalone MCP server (a separate Python process). The agent connects to them at runtime. Claude Code can also register these servers and use them in any session — independently of this agent.

**Why not keep Strands inline tools:** they're locked inside the agent process. MCP servers are reusable, versionable, and can be tested in isolation.

**Why not one big MCP server:** separation of concerns. Each server knows one data source deeply; the agent knows how to reason. A Legislature MCP can be improved without touching the OSPB MCP.

The pattern is already in the repo at `llm_testing_ground/eval/claude_langsmith.py`:
```python
weather_server = create_sdk_mcp_server(name="weather", version="1.0.0", tools=[get_weather])
options = ClaudeAgentOptions(mcp_servers={"weather": weather_server}, ...)
```

The current `colorado_open_data_tools.py` Strands tools will be promoted to an MCP server first, since they already work.

---

### Decision 2: Model — configurable via `config.toml` ✓

The LLM is no longer hard-coded. [`config.toml`](config.toml) defines named **profiles** (provider, model ID, `api_key_env`, `max_tokens`) plus the shared `system_prompt`. Select per run with `python agent.py --profile <name>` or set `active_profile`. [`model_config.py`](src/model_config.py) reads the profile and builds the matching Strands provider.

| Profile | Provider | Model | Status |
|---------|----------|-------|--------|
| `claude` (default) | Anthropic | `claude-sonnet-4-6` | ✅ verified, full toolset |
| `devstral` | OpenAI-compatible → `api.mistral.ai/v1` | `devstral-small-latest` | ✅ verified |
| `nvidia` | OpenAI-compatible → NVIDIA NIM | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | ✅ verified |

**Primary remains `claude-sonnet-4-6`** — reasoning quality matters most for synthesizing sparse, inconsistent government data. Add any OpenAI-compatible endpoint (OpenRouter, local vLLM/Ollama) by copying a profile block.

**Tool-calling is version-sensitive:** the agent exposes 27 tools at once (25 across 7 servers + 2 inline). `nemotron-super-49b-**v1**` returned an empty completion on the full toolset; **v1.5** fixed it. Always verify a new model/version before trusting it. Devstral uses the `openai` provider (not Strands' `MistralModel`) because the project pins `mistralai>=2.2.0`, incompatible with that provider.

---

### Decision 3: Single agent with built-in reflection ✓

**Not multi-agent.** The research literature finding that "single agent + many tools" outperforms multi-agent systems is capturing something real: the model's native step-by-step reasoning at each ReAct iteration outperforms a rigid pre-planned sequence.

**The "tool-ordering assistant" pattern considered and rejected.** The proposal was: a lightweight assistant agent that determines the order of tool calls before the main agent executes them. The problem for this use case is that **the plan goes stale immediately**. Our agent's most valuable behavior is adaptive — when data.colorado.gov returns sparse results, pivot to leg.colorado.gov; when a PDF is behind a portal, find the index page instead. A pre-planned sequence can't adapt to what the data actually says.

**What we do instead — reflection in the system prompt.** The agent is instructed to:
1. State its research plan in plain text before the first tool call
2. After each tool result, explicitly note what it found and reassess what to call next
3. Stop and synthesize when it has enough to answer the question confidently

This gives the auditability benefit of a planning step at zero extra cost — one model, one API call per step.

**If the agent makes too many tool calls** (the deaf-and-blind test used 84 — that's a lot), the right fix is a `max_tool_calls` budget cap in the agent loop, not a second planning agent.

**Future consideration:** If questions become highly parallelizable (e.g., "compare 5 departments' spending"), a lightweight Haiku dispatcher that fans out to 5 parallel Sonnet calls could make sense. Hold for Phase 5.

---

## Proposed MCP Servers

> **Historical / design rationale.** All of these are now implemented (and the set has grown to 7 — see [Implemented MCP Servers & Port Map](#implemented-mcp-servers--port-map) for current ports and tool names). Tool names below reflect the original proposal and may differ from what shipped; the port map is authoritative. Kept here for the design reasoning, especially the **Fund Types** table, which still drives the agent's analysis.

### 1. `colorado-open-data` (Socrata SODA API)
**Already implemented as Strands tools — promote to MCP server.**

Tools:
- `list_datasets(query)` — catalog search
- `get_dataset_metadata(dataset_id)` — column names and types
- `query_dataset(dataset_id, where_clause, select_columns, order_by, limit)` — SoQL query

Data source: `data.colorado.gov` — free, no auth, public API.

**Limitation:** data.colorado.gov has uneven coverage. CDOT is deep; most other agencies are sparse. TOPS (`fjyf-bdat`) is the only cross-agency spending dataset and its SoQL columns need validation.

---

### 2. `colorado-legislature` (leg.colorado.gov)
**New — highest value for bill tracking and fund-type attribution.**

Tools:
- `search_bills(topic, session_year, status)` — find bills by subject and year
- `get_bill_detail(bill_id)` — sponsors, status, summary, fiscal note link
- `get_fiscal_note(bill_id)` — dollar impact by fund type and agency
- `get_long_bill_section(session_year, department)` — appropriations for a department from the annual Long Bill
- `get_appropriations_history(department, start_year, end_year)` — 10-year trend for a department

**Key concept: Fund Types.** Every appropriation in the Long Bill specifies:
| Fund Type | What it is | Political significance |
|-----------|-----------|----------------------|
| General Fund | State income + sales tax | Discretionary; requires legislative vote |
| Cash Funds | Fees collected for specific purposes (DMV, hunting, oil & gas) | Earmarked; often off-limits for other uses |
| Federal Funds | Federal grants, Medicaid matching, highway $ | Dependent on federal policy; can disappear |
| Reappropriated Funds | Money transferred between agencies | Often confusing in political claims |

A politician claiming to "fund" something from a Cash Fund is very different from appropriating General Fund dollars. **This distinction is one of the most important things the agent should track.**

**Implementation challenge:** leg.colorado.gov serves PDFs through an external portal. We likely need to:
1. Scrape the document listing pages to find PDF URLs
2. Download PDFs with httpx
3. Parse with `pdfplumber` to extract text and tables

---

### 3. `colorado-ospb` (ospb.colorado.gov)
**New — Governor's budget requests and long-range forecasts.**

Tools:
- `get_budget_request(fiscal_year)` — Governor's proposed budget for a given year
- `search_budget_narrative(fiscal_year, keywords)` — full-text search within a budget document
- `get_forecast(report_type)` — revenue/expenditure forecast

**Implementation:** These are PDFs. Same approach as Legislature MCP — fetch, parse with pdfplumber.

---

### 4. `web-search` (internet search)
**New — fills gaps that structured data can't cover.**

For questions about policy context, press coverage of specific bills, national comparisons, or data that predates Colorado's open data portal, we need general internet search.

Options:
- **Tavily** — good for factual/research queries; needs `TAVILY_API_KEY`
- **Exa** — good for finding specific documents and web pages; needs `EXA_API_KEY`
- **strands_tools.tavily / strands_tools.exa** — already in strands-agents-tools, need to test if they register correctly as Strands tools (the `http_request` tool did not)
- **DuckDuckGo scraping** — free, no API key, but fragile

**Recommendation:** Start with Tavily (straightforward API, free tier). If keys aren't available, fall back to targeted `fetch_webpage` calls to known good sources (CPR News, Colorado Sun, Denver Post budget coverage).

---

## Agentic Architecture — Pattern Ranking

Four patterns worth comparing for this use case:

---

### 1. ReAct — **use now** ✓

**Think → Act → Observe → repeat.** At each step the model reasons about what to do, calls one tool, reads the result, and decides what to do next.

This is what the current agent runs. It's adaptive — if data.colorado.gov comes back sparse, the model reads that result and pivots to leg.colorado.gov on the next step. That adaptivity is exactly what we need given how uneven Colorado's data sources are.

**Weakness for our case:** purely linear, one tool call at a time. The deaf-and-blind test used 84 sequential calls. No backtracking — if the agent goes down a dead-end PDF path it keeps trying rather than moving on.

---

### 2. Reflexion (ReAct + failure reflection) — **active ✓**

**ReAct + an explicit self-critique step after a failed tool call.** When a fetch fails or returns empty, the agent writes a short reflection ("the PDF is behind a portal; I should find the document listing page first") before trying again.

This directly targets our biggest known problem: the agent spending 30+ calls trying to access leg.colorado.gov PDFs that are behind Box. A reflection step would catch that pattern early and change strategy.

Implementation: add a failure detection wrapper around `fetch_webpage` that triggers a reflection message when it sees repeated failures on the same domain or consecutive empty results.

**Cost:** near zero — no extra LLM calls, just a system-prompt addition and a tool-call result flag.

---

### 3. LATS (Language Agent Tree Search) — **consider for Phase 5**

**Monte Carlo Tree Search applied to agent reasoning.** The agent explores multiple research paths simultaneously, scores each branch by how promising it looks (how much relevant data it returned), and backtracks from dead ends.

For a question like "compare 5 departments' budget history" or "find all bills from 2019–2024 affecting a topic," LATS would genuinely improve quality — it would explore the SODA API path, the legislature scraper path, and the OSPB PDF path in parallel and pick the richest branch.

**Why not now:** expensive (many parallel LLM calls), complex to implement, and we haven't yet hit a question where ReAct + Reflexion clearly fails. Build the data sources first; reach for LATS when questions get complex enough to need it.

---

### 4. ReWOO — **not a good fit here**

**Plan the entire tool-call sequence upfront, execute all calls (parallelized), then synthesize.** ReWOO's efficiency win (fewer LLM inference calls, parallelizable execution) is real on tasks where the research path is predictable.

It's a poor fit here because **our data availability is fundamentally uncertain**. We don't know upfront whether SODA has the data we need, whether a leg.colorado.gov PDF is accessible, or whether we'll need to fall back to OSPB. A pre-planned call sequence goes stale the moment the first result comes back empty. ReWOO's gains evaporate and its brittleness hurts.

---

### Summary ranking

| Pattern | Fit | When |
|---------|-----|------|
| ReAct | ★★★★★ | Now — baseline, already running |
| Reflexion | ★★★★☆ | Active — injected into system prompt |
| LATS | ★★★☆☆ | Phase 5 — for complex multi-path questions |
| ReWOO | ★★☆☆☆ | Not recommended — brittle against uncertain data availability |

---

## Roadmap

### Phase 2 — PDF parsing (unlocks exact dollar figures)
**Prerequisite for legislative and OSPB data.**

1. Add `pdfplumber` to dependencies
2. Add `download_and_parse_pdf(url)` tool — fetch a PDF by URL, extract text and tables
3. Test against JBC Appropriations History Report and a fiscal note
4. Use it inside the Legislature MCP to serve `get_fiscal_note` and `get_appropriations_history`

---

### Phase 3 — Legislature MCP (bill tracking + fund types)
1. Scrape leg.colorado.gov bill search for a given topic + year range
2. For each bill: extract bill number, title, sponsors, final status, fiscal note URL
3. Parse fiscal notes to extract: fund type breakdown, agency impact, dollar amounts
4. Add `get_long_bill_section` to pull a department's annual appropriation by fund type

**This is the unlock for "General Fund vs cash fund" questions.**

---

### Phase 4 — OSPB MCP + internet search
1. Scrape ospb.colorado.gov for budget request PDFs by year
2. Parse and index the narrative sections
3. Add `web_search` (Tavily or Exa) so the agent can find press coverage and context

---

### Phase 5 — Conversation mode
Today the agent answers one question and exits. For election-season use, we want:
- Follow-up questions ("what about 2021 specifically?")
- "Drill down" capability ("show me the fiscal note for that bill")
- Session memory so you don't repeat context

This means moving from single-shot `agent(question)` to a conversation loop with `SlidingWindowConversationManager` (already in the Strands dependency).

---

## Open Questions

These are unsettled — add your thoughts here.

**Q1: MCP server transport.** The Claude Agent SDK example uses in-process MCP servers (`create_sdk_mcp_server`). For standalone MCP servers usable by Claude Code, we'd use stdio or SSE transport. Which do we want first?
> *Let us use SSE/ streamable HTTP*

**Q2: PDF portal.** leg.colorado.gov PDFs go through an external portal (Box). Do we try to reverse-engineer the Box links, or is there a direct PDF URL pattern for the Long Bill and fiscal notes that we haven't found yet?
> *Let's try for a direct URL.  We will figure it out*

**Q3: Data freshness.** data.colorado.gov TOPS data is through ~2022. For current-year spending, what's the right source? The OSPB forecast shows projections; the Long Bill shows appropriations. Is there a source for actual expenditures year-to-date?
> *The most detailed official sources for Colorado’s budget and actual spending are the Joint Budget Committee (JBC) and the Governor's Office of State Planning and Budgeting (OSPB).For the most recent and upcoming budget data, you can check these specific resources:Official Department Appropriations: For in-depth breakdowns of operating, capital, and departmental budgets, use the Colorado General Assembly Appropriations Reports.Interactive Breakdown: To visually explore how state tax dollars are allocated across major departments (like Healthcare, Education, and Corrections), use the Colorado General Assembly Explore Budget Tool.State Revenue Forecasts & Spending Plans: For the most current revenue forecasts, executive budget proposals, and total operating fund spending, view the Colorado Office of State Planning and Budgeting.Audited Expenditures: For historical and exact actual spending audits, check the Colorado State Treasury Expenditures page.*

**Q4: Search API keys.** Do we have Tavily or Exa keys? If not, should we add them, or rely on targeted `fetch_webpage` calls instead?
> *Using Exa (trial key). Key stored in `colorado_budget/.env` as `EXA_API_KEY`. Using `exa-py`, `type="auto"`, `contents={"highlights": True}`. For harder synthesis queries, use `type="deep"` with `outputSchema`. Docs: https://docs.exa.ai/reference/search-api-guide-for-coding-agents*

**Q5: Output format.** The current agent outputs narrative text. For election-season use, would it be more useful to output structured citations (source, URL, dollar amount, fiscal year) that can be copy-pasted into a document or shared? Or is narrative fine?
> *narrative is fine, but I would like a list of sources at the end of the response.  all the better if it is structured wikipedia style-- "some random claim"[citation number] and at the end of the response [citation number] source document are shown.*

---

## File Layout (current)

```
colorado_budget/
├── README.md
├── ARCHITECTURE.md            ← this file
├── config.toml               ← model profiles + shared system prompt
├── .env                      ← API keys (gitignored — never committed)
└── src/
    ├── agent.py              ← orchestrator: spawns servers, wires MCPClients, runs query
    ├── model_config.py       ← reads config.toml profile → builds Strands model
    ├── utils.py              ← API key loading (.env → env vars)
    │
    ├── servers/              ← MCP servers (one file per data source; ports 8001–8007)
    │   ├── colorado_open_data.py   ← 8001  Socrata SODA
    │   ├── web_search.py           ← 8002  Exa
    │   ├── legislature.py          ← 8003  leg.colorado.gov bills, fiscal notes, JBC
    │   ├── ospb.py                 ← 8004  Governor's budget, forecasts
    │   ├── revenue.py              ← 8005  LCS forecast, TABOR, tax expenditures
    │   ├── federal_funds.py        ← 8006  USAspending.gov
    │   └── school_finance.py       ← 8007  CDE K-12 funding, HB24-1448
    │
    └── tools/                ← inline Strands tools (not MCP)
        ├── fetch_webpage.py  ← HTML fetcher/stripper
        └── pdf_parser.py     ← pdfplumber wrapper (fetch_and_parse_pdf)
```

Tests live in `tests/` (`unit/`, `integration/`, `evals/`, `manual/`); each server has unit tests (mocked network) and an integration test that boots the subprocess and checks its tool list. See [tests/README.md](tests/README.md).
