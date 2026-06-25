# Testing the Colorado Budget Agent

Three levels of testing, each with a different speed/cost profile.

---

## Quick start

```bash
# From the repo root
cd /home/ace/work/mcp_playground

# Unit tests — fast, no network, no API keys
pytest colorado_budget/tests/unit -v

# Integration tests — real network, real SODA API (no LLM)
pytest colorado_budget/tests/integration/test_open_data_live.py -v
pytest colorado_budget/tests/integration/test_mcp_server.py -v

# Full agent smoke test — slow, uses LLM (~$0.02 each)
pytest colorado_budget/tests/integration/test_agent_smoke.py -v -s

# All non-slow tests
pytest colorado_budget/tests -v -m "not slow"
```

---

## Test categories

### Unit tests (`tests/unit/`) — seconds, free
Fully mocked, no external calls. Test the Python logic of each tool in isolation.

| File | What it tests |
|---|---|
| `test_html_extractor.py` | `_TextExtractor` HTML parser — script/style stripping, whitespace |
| `test_open_data_tools.py` | SODA API tools — request params, empty responses, error handling |
| `test_pdf_parser.py` | PDF parser — keyword filter, page range, table extraction, truncation |

### Integration tests (`tests/integration/`) — seconds to minutes
Real network, real APIs, but no LLM costs for the server tests.

| File | What it tests | Cost |
|---|---|---|
| `test_open_data_live.py` | Real calls to data.colorado.gov | Free |
| `test_mcp_server.py` | MCP server subprocess — tools/list, initialize | Free |
| `test_agent_smoke.py` | Full agent with LLM | ~$0.05/run |

### Eval suite (`tests/evals/`) — minutes, LLM costs
LLM-as-judge quality scoring. See [evals/README.md](evals/README.md).

```bash
cd colorado_budget/tests
python evals/run_evals.py --dry-run        # preview questions
python evals/run_evals.py --cases csdb_10yr  # run one case
python evals/run_evals.py                  # all 5 cases (~10 min, ~$0.50)
```

### Manual tests (`tests/manual/`) — interactive
```bash
cd colorado_budget/src
bash ../tests/manual/questions.sh          # print question list
bash ../tests/manual/questions.sh 1        # run CSDB canonical question
```

Every manual question accepts a model profile. Pass `--profile` straight through
to `agent.py`, or run a question directly:

```bash
cd colorado_budget/src
python agent.py --profile devstral "$(bash ../tests/manual/questions.sh 1 --print 2>/dev/null || echo 'your question')"
# simplest: just call agent.py with the profile you want to compare
python agent.py --profile nvidia "What are the main fund types in Colorado's state budget?"
```

---

## Trying different models

The agent's LLM is configured in [`../config.toml`](../config.toml) (see the
[README](../README.md#choosing-a-model)). Each profile names the API-key env var it
needs; set those in `colorado_budget/.env` first.

### 1. Does the profile load and build? (free, no LLM call)

```bash
cd colorado_budget/src
python -c "
from utils import load_api_keys; load_api_keys()
from model_config import list_profiles, load_agent_config
for p in list_profiles():
    try:
        c = load_agent_config(p)
        print(f'OK   {p:10s} {type(c.model).__name__}  {c.model_id}')
    except Exception as e:
        print(f'SKIP {p:10s} {type(e).__name__}: {e}')
"
```

A missing API key surfaces here as a clear `SKIP ... requires environment variable $X`.

### 2. Does the model support tool calling? (one cheap call)

This agent depends on the model issuing tool calls. Before a full run, confirm the
endpoint works **and** the model returns a `tool_calls` finish reason. For an
OpenAI-compatible profile (e.g. `nvidia`):

```bash
cd colorado_budget/src
python -c "
import os; from utils import load_api_keys; load_api_keys()
import openai
c = openai.OpenAI(api_key=os.environ['NVIDIA_API_KEY'], base_url='https://integrate.api.nvidia.com/v1')
tools=[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
r = c.chat.completions.create(model='nvidia/llama-3.3-nemotron-super-49b-v1',
    messages=[{'role':'user','content':'Weather in Denver? Use the tool.'}], tools=tools, max_tokens=200)
print('finish_reason:', r.choices[0].finish_reason)        # want: tool_calls
print('tool_calls:', r.choices[0].message.tool_calls)
"
```

### 3. Full end-to-end run

```bash
cd colorado_budget/src
python agent.py --profile <name> "What are the main fund types in Colorado's state budget?"
```

Watch the `Tool #N:` lines in the output — they confirm the model is actually
calling tools (agent-mode) rather than answering from its own weights (LLM-mode).

> **Known caveat — many tools at once (version-sensitive).** The full agent
> exposes ~16 tools (4 MCP servers + 2 inline). Claude, Devstral, and
> `nvidia/llama-3.3-nemotron-super-49b-v1.5` handle this fine. The earlier
> `…-super-49b-v1` (no `.5`) handled a *single* tool but returned an **empty
> completion** on the full toolset — no error, no tool call, just a blank answer.
> A point release fixed it. If you see an empty answer, suspect tool-count/schema
> limits in that specific model/version: try a newer version or stronger tool-use
> model, or trim the tool list in `agent.py` for that profile.

---

## Markers

```bash
pytest -m integration    # only integration tests
pytest -m slow           # only slow (LLM) tests
pytest -m "not slow"     # everything except LLM tests
```

---

## pytest config

`pytest.ini` at `colorado_budget/` root sets default options and registers markers.
