# Colorado Budget Research Agent

A conversational research tool for investigating Colorado state government spending. Ask broad policy questions in plain English; the agent finds, queries, and synthesizes data from Colorado government sources.

Built as a [Strands](https://strandsagents.com) agent. The LLM is **configurable** — Claude, Devstral (Mistral), or any OpenAI-compatible endpoint (e.g. NVIDIA NIM) — via [`config.toml`](config.toml). See [Choosing a model](#choosing-a-model).

---

## Quick Start

> **Use the project venv.** This repo is managed by [uv](https://docs.astral.sh/uv/);
> the dependencies (`strands`, `openai`, …) live in `.venv`, not your system Python.
> Either activate it (`source .venv/bin/activate`) or call the interpreter directly
> (`.venv/bin/python …`). Running plain `python agent.py` with the venv inactive
> fails with `ModuleNotFoundError: No module named 'strands'`.

```bash
cd colorado_budget/src
source ../../.venv/bin/activate          # or prefix commands with ../../.venv/bin/python

# Default question (lists available datasets), default model (claude)
python agent.py

# Ask a specific question
python agent.py "What changes have been made to funding for the Colorado School for the Deaf and the Blind in the last 10 years?"

# Try a different model
python agent.py --profile devstral "How has Colorado's Medicaid spending changed since 2018?"
python agent.py --profile nvidia   "What bills passed in 2023 that affected housing affordability funding?"
```

---

## Setting up `.env`

The agent loads API keys from `colorado_budget/.env` (falling back to the repo-root `.env`, then your shell environment). **`.env` files are gitignored — never commit keys.**

Create `colorado_budget/.env` with the keys for the providers you intend to use:

```bash
# --- Required: web search (used by every run) ---
EXA_API_KEY=your-exa-key

# --- LLM provider keys (only the one(s) you use) ---
# Claude (default profile). Either name works; ANTHROPIC_API_KEY takes precedence.
ANTHROPIC_API_KEY=sk-ant-...
# OPENWEBUI_ANTHROPIC_API_KEY=sk-ant-...   # auto-mapped to ANTHROPIC_API_KEY if the above is unset

# Devstral (devstral profile)
MISTRAL_API_KEY=...

# NVIDIA NIM (nvidia profile) — free keys at https://build.nvidia.com
NVIDIA_API_KEY=nvapi-...
```

Notes:
- Each profile in `config.toml` names the env var it needs via `api_key_env`. If that variable is unset, the agent fails fast with a clear message naming the missing key.
- `EXA_API_KEY` is always required — the `web-search` MCP server uses it regardless of which LLM you pick.
- You only need the key(s) for the profile(s) you actually run.

---

## Choosing a model

Models are defined as **profiles** in [`config.toml`](config.toml). Each profile sets a provider, model ID, the env var holding its API key, and token limits. The shared `system_prompt` lives at the top of the same file (a profile may override it).

| Profile | Provider | Model | Key env | Status |
|---------|----------|-------|---------|--------|
| `claude` (default) | Anthropic | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | ✅ Verified end-to-end (full toolset) |
| `devstral` | OpenAI-compatible (`api.mistral.ai/v1`) | `devstral-small-latest` | `MISTRAL_API_KEY` | ✅ Verified end-to-end (full toolset) |
| `nvidia` | OpenAI-compatible (NIM) | `nvidia/llama-3.3-nemotron-super-49b-v1` | `NVIDIA_API_KEY` | ⚠️ Tool-calling works with few tools; returns empty on the full toolset (see caveat below) |

> **Why Devstral uses the `openai` provider, not `mistral`:** this project pins `mistralai>=2.2.0` (for langchain), which is incompatible with Strands' native `MistralModel` (it needs `mistralai<2.0.0`). Mistral's API is OpenAI-compatible, so the `devstral` profile points the `openai` provider at `https://api.mistral.ai/v1` — no `mistralai` SDK required, and tool calling works.

Select a model three ways:
- **Per run:** `python agent.py --profile devstral "question"`
- **Change the default:** set `active_profile = "devstral"` in `config.toml`
- **Add your own:** copy a `[profiles.x]` block, point `base_url`/`model_id`/`api_key_env` at any OpenAI-compatible endpoint (OpenRouter, local vLLM/Ollama, …)

```toml
[profiles.my-model]
provider    = "openai"
model_id    = "meta/llama-3.1-70b-instruct"
base_url    = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"
max_tokens  = 8096
```

> **Tool-calling matters.** This is an *agent* — it depends on the model issuing tool calls. Claude handles the full toolset well (verified on the Q8–Q10 smoke tests). Some OpenAI-compatible models return an empty response when given many tools at once: `nvidia/llama-3.3-nemotron-super-49b-v1` calls a *single* tool correctly but returns a blank completion when handed all ~16 tools the agent exposes. If a model answers without ever calling a tool (or returns nothing), suspect weak/partial tool-call support. See [tests/README.md](tests/README.md#trying-different-models) for the steps to verify a new model before trusting it.

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
├── ARCHITECTURE.md        ← design doc and roadmap
├── config.toml            ← model profiles + system prompt (edit to switch LLMs)
├── .env                   ← API keys (gitignored — create this yourself)
└── src/
    ├── agent.py           ← entry point; Strands agent wiring, --profile flag
    ├── model_config.py    ← reads config.toml, builds the selected model provider
    ├── utils.py           ← API key loading with env var fallback
    ├── servers/           ← MCP servers (open-data, web-search, legislature, ospb)
    └── tools/             ← inline tools: fetch_webpage, fetch_and_parse_pdf
```
