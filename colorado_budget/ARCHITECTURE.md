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
> *I just signed up for a trial Exa key.  Here is the setup prompt """> **Canonical reference:** https://docs.exa.ai/reference/search-api-guide-for-coding-agents
>
> If anything below looks outdated or contradicts real API behavior, fetch that URL — it is the source of truth for search types, parameters, and response shape. Report staleness back to the user.

---

# Exa API Setup Guide

## Your Configuration

| Setting | Value |
|---------|-------|
| Coding Tool | Claude |
| Integration | Python |
| Use Case | Web search tool |
| Search Type | Auto - Balanced relevance and speed (default) |
| Content | Highlights |

**Project Description:** i want to create an agent to answer questions about colorado's budget and spending.  it's election season and politicians are making claims about where the state's money is going.  i want to verify it


---

## API Key Setup

### Environment Variable

```bash
export EXA_API_KEY="YOUR_API_KEY"
```

### .env File

```env
EXA_API_KEY=YOUR_API_KEY
```

### Usage in Code

```python
import os
from exa_py import Exa

exa = Exa(api_key=os.environ.get("EXA_API_KEY"))
```

---

## Quick Start (Python)

```bash
pip install exa-py==2.14.0
```

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_API_KEY")

results = exa.search(
    "recent product announcements from developer tools companies",
    type="auto",
    num_results=10,
    contents={"highlights": True}
)

for result in results.results:
    print(result.title, result.url)
```

---

## Pick Your Search Pattern

Start with one of these two patterns:

### 1. Raw retrieval for your own agent

Use this when your app should inspect `results` directly, pass `highlights` into your own LLM, or expose Exa as a tool inside an existing agent loop.

```json
{
  "query": "recent product announcements from developer tools companies",
  "type": "auto",
  "numResults": 10,
  "contents": {
    "highlights": true
  }
}
```

### 2. Synthesized search when you want grounded output

Use this when you want Exa to synthesize a grounded answer or structured payload for you. `systemPrompt` sets behavior and source preferences; `outputSchema` sets the shape of `output.content`.

```json
{
  "query": "recent product announcements from developer tools companies",
  "type": "deep",
  "systemPrompt": "Prefer official sources, collapse duplicate reporting, and keep the output grounded.",
  "outputSchema": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string",
        "description": "A grounded summary of the most important findings"
      }
    },
    "required": [
      "summary"
    ]
  },
  "contents": {
    "highlights": true
  }
}
```

### Deep search notes

- Use `deep` when you need harder comparisons, structured synthesis, or multi-step reasoning across many sources.
- Use `additionalQueries` only on `deep-lite`, `deep`, and `deep-reasoning` when you want to force a few explicit query angles instead of relying entirely on automatic query expansion.
- If you only need raw search results and excerpts, stay on `results` + `highlights` and skip `outputSchema`.

---

## Function Calling / Tool Use

Function calling (also known as tool use) allows your AI agent to dynamically decide when to search the web based on the conversation context. Instead of searching on every request, the LLM intelligently determines when real-time information would improve its response—making your agent more efficient and accurate.

**Why use function calling with Exa?**
- Your agent can ground responses in current, factual information
- Reduces hallucinations by fetching real sources when needed
- Enables multi-step reasoning where the agent searches, analyzes, and responds

📚 **Full documentation**: https://docs.exa.ai/reference/openai-tool-calling

### OpenAI Function Calling

```python
import json
from openai import OpenAI
from exa_py import Exa

openai = OpenAI()
exa = Exa(api_key="YOUR_EXA_API_KEY")

tools = [{
    "type": "function",
    "function": {
        "name": "exa_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    }
}]

def exa_search(query: str) -> str:
    results = exa.search(query, type="auto", num_results=10, contents={"highlights": True})
    return "\n".join([f"{r.title}: {r.url}" for r in results.results])

messages = [{"role": "user", "content": "What's the latest in AI safety?"}]
response = openai.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    search_results = exa_search(json.loads(tool_call.function.arguments)["query"])
    messages.append(response.choices[0].message)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": search_results})
    final = openai.chat.completions.create(model="gpt-4o", messages=messages)
    print(final.choices[0].message.content)
```

### Anthropic Tool Use

```python
import anthropic
from exa_py import Exa

client = anthropic.Anthropic()
exa = Exa(api_key="YOUR_EXA_API_KEY")

tools = [{
    "name": "exa_search",
    "description": "Search the web for current information.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"]
    }
}]

def exa_search(query: str) -> str:
    results = exa.search(query, type="auto", num_results=10, contents={"highlights": True})
    return "\n".join([f"{r.title}: {r.url}" for r in results.results])

messages = [{"role": "user", "content": "Latest quantum computing developments?"}]
response = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=4096, tools=tools, messages=messages)

if response.stop_reason == "tool_use":
    tool_use = next(b for b in response.content if b.type == "tool_use")
    tool_result = exa_search(tool_use.input["query"])
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}]})
    final = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=4096, tools=tools, messages=messages)
    print(final.content[0].text)
```

---

## Search Type Reference

| Type | Best For | Approx Latency | Depth |
|------|----------|----------------|-------|
| `auto` | Most queries — balanced relevance and speed | ~1 second | Smart | ← your selection
| `fast` | Latency-sensitive queries that still need good relevance | ~450 ms | Basic |
| `instant` | Chat, voice, autocomplete, quick lookups | ~250 ms | Basic |
| `deep-lite` | Cheaper synthesis when full deep search is overkill | 4 seconds | Deep |
| `deep` | Research, enrichment, thorough results | 4-15 seconds | Deep |
| `deep-reasoning` | Complex research, multi-step reasoning, hard synthesis tasks | 12-40 seconds | Deepest |

Latency numbers are ballpark — synthesis (`outputSchema`) and forced livecrawls (`contents.maxAgeHours: 0`) stack on top of the base `type`. See the Latency Characteristics section for details.

**Tip:** `type="auto"` works well for most queries. `outputSchema` works on every search type, so you can request structured, grounded output regardless of which type you pick.

---

## Optional: Structured Outputs (outputSchema)

Raw `results` + `highlights` should still be your default starting point for many agent workflows. Add `outputSchema` only when you want Exa to synthesize grounded JSON or a structured answer for you.

`outputSchema` works on **every** search type. Pass a JSON schema and Exa returns the synthesized answer as structured JSON in `output.content`, with field-level citations in `output.grounding`. Deep variants (`deep-lite`, `deep`, `deep-reasoning`) give higher-quality synthesis for complex queries, but the response shape is the same.

**Use `systemPrompt` and `outputSchema` together:** `systemPrompt` controls source preferences, dedupe behavior, and synthesis rules; `outputSchema` controls the exact shape of `output.content`.

**Schema controls:** `type`, `description`, `required`, `properties`, `items`. Max nesting depth 2, max total properties 10. Do NOT add citation or confidence fields to the schema — `/search` returns grounding data automatically.

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_API_KEY")

results = exa.search(
    "articles about GPUs",
    type="auto",
    system_prompt="Prefer official sources, collapse duplicate reporting, and keep the output grounded.",
    output_schema={
        "type": "object",
        "description": "Companies mentioned in articles",
        "required": ["companies"],
        "properties": {
            "companies": {
                "type": "array",
                "description": "List of companies mentioned",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the company"
                        },
                        "description": {
                            "type": "string",
                            "description": "Short description of what the company does"
                        }
                    }
                }
            }
        }
    },
    contents={"highlights": True}
)

# Access structured output
print(results.output.content)   # {"companies": [{"name": "Nvidia", "description": "..."}]}
print(results.output.grounding) # Field-level citations
```

### Response Shape

Responses with `outputSchema` include:
- `output.content` — structured JSON matching your schema (or a string for `{"type": "text"}` schemas)
- `output.grounding` — array of `{field, citations, confidence}` entries with source URLs

```json
{
  "output": {
    "content": {
      "companies": [
        {"name": "Nvidia", "description": "GPU and AI chip manufacturer"},
        {"name": "AMD", "description": "Semiconductor company producing GPUs and CPUs"}
      ]
    },
    "grounding": [
      {
        "field": "companies[0].name",
        "citations": [{"url": "https://...", "title": "Source"}],
        "confidence": "high"
      }
    ]
  }
}
```

### When to Use Structured Outputs

- **Enrichment workflows** — extract specific fields (company info, people data, product details)
- **Data pipelines** — get structured data directly instead of parsing free text
- **Grounded answers** — prefer `outputSchema` on `/search` for new structured search flows
- Prefer a deep variant (`deep-lite`/`deep`/`deep-reasoning`) when you need multi-step reasoning or synthesis across many sources

---

## Content Configuration

The generated examples request highlights by default:

```json
"contents": {
  "highlights": true
}
```

Highlights return query-relevant excerpts, which are usually the right content mode for LLM workflows because they keep token usage predictable.

Content is controlled via the `contents` object on `/search` (or top-level fields on `/contents`). Pick one of `text`, `highlights`, or `summary` by default. You can combine them, but it is usually an antipattern to do so at the start of a project.

| Mode | Config | Best For |
|------|--------|----------|
| Highlights | `"highlights": true` | Token-efficient excerpts |
| Text | `"text": {"maxCharacters": 20000}` | Full content extraction, RAG |
| Summary | `"summary": {"query": "your question"}` or `"summary": true` | LLM-written summary per result |

### Tuning knobs

- **`highlights`** — pass `true` to return query-relevant highlights for each result.
- **`summary`** — pass `true` for a generic summary, or `{"query": "..."}` to bias the summary toward a specific question. Supports a `schema` field for per-result structured output. Summary has no `verbosity` setting — verbosity lives on `text` (below).
- **`text.verbosity`** — `"compact" | "full"` (default `"compact"`). Compact returns only the main content of the page, excluding navbars, banners, footers etc.
- **`text.includeHtmlTags`** — boolean (default `false`). When `true`, preserves HTML structure (useful for code blocks, tables).
- **`text.maxCharacters`** — hard cap on extracted text length. Always set this to control token cost when requesting text.

**Case conventions:** JavaScript SDK and raw JSON use camelCase (`maxCharacters`). Python SDK uses snake_case (`max_characters`) — this applies inside nested dicts too.

**Token usage:** `text: true` with no cap can blow up context. Prefer `highlights: true` for most agent workflows, and add `text` only when downstream reasoning truly needs broad page context.

---

## Domain Filtering (Optional)

Usually not needed - Exa's neural search finds relevant results without domain restrictions.

**When to use:**
- Targeting specific authoritative sources
- Excluding low-quality domains from results

**Example:**

```json
{
  "includeDomains": ["arxiv.org", "github.com"],
  "excludeDomains": ["pinterest.com"]
}
```

**Note:** `includeDomains` and `excludeDomains` can be used together to include a broad domain while excluding specific subdomains (e.g., `"includeDomains": ["vercel.com"], "excludeDomains": ["community.vercel.com"]`).

---

## Web Search Tool

```json
{
  "query": "recent product announcements from developer tools companies",
  "numResults": 10,
  "contents": {
    "highlights": true
  }
}
```

**Tips:**
- Use `type: "auto"` for most queries
- Great for building search-powered chatbots or agents
- Combine with contents for RAG workflows

---

## Content Freshness (maxAgeHours)

`maxAgeHours` sets the maximum acceptable age (in hours) for cached content. If the cached version is older than this threshold, Exa will livecrawl the page to get fresh content.

| Value | Behavior | Best For |
|-------|----------|----------|
| 24 | Use cache if less than 24 hours old, otherwise livecrawl | Daily-fresh content |
| 1 | Use cache if less than 1 hour old, otherwise livecrawl | Near real-time data |
| 0 | Always livecrawl (ignore cache entirely) | Real-time data where cached content is unusable |
| -1 | Never livecrawl (cache only) | Maximum speed, historical/static content |
| *(omit)* | Default behavior (livecrawl as fallback if no cache exists) | **Recommended** — balanced speed and freshness |

**When LiveCrawl Isn't Necessary:**
Cached data is sufficient for many queries, especially for historical topics or educational content. These subjects rarely change, so reliable cached results can provide accurate information quickly.

See [maxAgeHours docs](https://exa.ai/docs/reference/livecrawling-contents#maxAgeHours) for more details.

---

## Other Endpoints

Beyond `/search`, the next two endpoints to know are `/contents` and `/answer`:

| Endpoint | Description | Docs |
|----------|-------------|------|
| `/contents` | Get clean, parsed content for URLs you already have | [Docs](https://exa.ai/docs/reference/get-contents) |
| `/answer` | Get a grounded answer with citations when the UI is question-first | [Docs](https://exa.ai/docs/reference/answer) |

> For new structured search flows, prefer `/search` + `outputSchema` when you want both retrieval control and grounded output. Keep `/answer` for question-first UIs where you do not need to inspect raw search results.

### /contents — Get Contents for Known URLs

Use `/contents` when you already have URLs and need their content. Unlike `/search` (which finds and optionally retrieves content), `/contents` is purely for content extraction from known URLs.

**When to use `/contents` vs `/search`:**
- URLs from another source (database, user input, RSS feeds) → `/contents`
- Need to refresh stale content for URLs you already have → `/contents` with `maxAgeHours`
- Need to find AND get content in one call → `/search` with `contents`

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_API_KEY")

results = exa.get_contents(
    ["https://example.com/article", "https://example.com/blog-post"],
    highlights=True
)

for result in results.results:
    print(result.title, result.url)
    print(result.highlights)
```

**Content retrieval options** (choose one per request):

| Option | Config | Best For |
|--------|--------|----------|
| Highlights | `"highlights": true` | Key excerpts, lower token usage |
| Text | `"text": {"max_characters": 20000}` | Full content extraction, RAG |

**Highlights example:**

```json
{
  "urls": ["https://example.com/article"],
  "highlights": true
}
```

**Freshness control:** Add `maxAgeHours` to ensure content is fresh:
- `24` — livecrawl if cached content is older than 24 hours
- `0` — always livecrawl (ignore cache)
- Omit — use cache when available, livecrawl as fallback

---

## Troubleshooting

**⚠️ COMMON PARAMETER MISTAKES — avoid these:**
- `useAutoprompt` → **deprecated**, remove it entirely
- `includeUrls` / `excludeUrls` → **do not exist**. Use `includeDomains` / `excludeDomains`
- `text`, `summary`, `highlights` at the top level of `/search` → **must be nested** inside `contents` (e.g. `"contents": {"highlights": true}`). On `/contents` they ARE top-level — don't confuse the two.
- `numSentences`, `highlightsPerUrl` → **deprecated** highlights params. Use `highlights: true` instead
- `tokensNum` → **does not exist**. Use `contents.text.maxCharacters` to limit text length
- `livecrawl: "always"` → **deprecated**. Use `contents.maxAgeHours: 0` instead
- `excludeDomains` + `category: "company" | "people"` → **400 error**. Those categories don't support `excludeDomains` or any date filters.

> **`stream: true`** switches `/search` to SSE mode (OpenAI-compatible chat-completion chunks). It's supported — just expect streaming chunks instead of one JSON response.

**Results not relevant?**
1. Try `type: "auto"` - most balanced option
2. Try `type: "deep"` - runs multiple query variations and ranks the combined results
3. Refine query - use singular form, be specific
4. Check category matches your use case

**Need structured data from search?**
1. Pass `outputSchema` on any search type — `auto` works, `deep`/`deep-reasoning` gives higher-quality synthesis
2. Define the fields you need in the schema, then add `systemPrompt` for source preferences and dedupe rules

**Results too slow?**
1. Use `type: "fast"` or `type: "instant"`
2. Reduce `numResults`
3. Skip contents if you only need URLs

**No results?**
1. Remove filters (date, domain restrictions)
2. Simplify query
3. Try `type: "auto"` - has fallback mechanisms

---

## Resources

- Docs: https://exa.ai/docs
- Dashboard: https://dashboard.exa.ai
- API Status: https://status.exa.ai"""

Here is the API key 8baf737a-7d65-4ee5-a512-7f6e83616044*

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
