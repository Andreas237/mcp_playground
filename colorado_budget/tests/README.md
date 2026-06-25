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
