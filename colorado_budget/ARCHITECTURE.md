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

## Current State (Phase 1 — done)

A single Strands agent (`agent.py`) with four inline tools:
- `list_datasets`, `get_dataset_metadata`, `query_dataset` — Socrata SODA API wrapper
- `fetch_webpage` — HTML page fetcher for leg.colorado.gov and ospb.colorado.gov

**What works:** Discovery questions, broad trends, identifying the right documents to read.

**What doesn't:** Exact line-item figures (PDFs behind a portal), bill-level fund-type tracking, internet search.

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

### Decision 2: Model — Claude Sonnet ✓

**Primary model:** `claude-sonnet-4-6`. Reasoning quality matters more than cost for this use case — the agent needs to synthesize across sparse, inconsistent government data sources.

**Mistral Devstral** (free tier) is worth knowing about for future cost reduction, but it's optimized for code generation, not open-ended policy research synthesis. The Mistral API key (`MISTRAL_API_KEY`) is already in the environment if we want to experiment.

**Haiku** (`claude-haiku-4-5`) is a reasonable choice for a lightweight "planning" or "reflection" step if we add one (see Decision 3 below).

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

## File Layout (target state)

```
colorado_budget/
├── README.md
├── ARCHITECTURE.md            ← this file
└── src/
    ├── agent.py               ← orchestrator agent (Claude Sonnet)
    ├── utils.py               ← API key loading
    │
    ├── mcp/                   ← MCP servers (one file per data source)
    │   ├── colorado_open_data.py   ← Socrata SODA (promoted from tools)
    │   ├── legislature.py          ← leg.colorado.gov bills, fiscal notes, Long Bill
    │   ├── ospb.py                 ← ospb.colorado.gov budget requests
    │   └── web_search.py           ← Tavily/Exa or targeted scraping
    │
    └── tools/                 ← inline Strands tools (not promoted to MCP)
        ├── fetch_webpage.py   ← ad-hoc HTML fetcher
        └── pdf_parser.py      ← pdfplumber wrapper (Phase 2)
```
