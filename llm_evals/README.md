# llm_evals

Demo and evaluation harness for LLM models with LangSmith observability.

## What it does

- **`eval/eval.py`** — `DemoMistral` class with two pipelines:
  - `run_pipeline()` — single-shot call (English → French translation demo)
  - `run_multi_pipeline()` — interactive multi-turn conversation loop
- **`utils.py`** — loads `.env` and sets API keys as environment variables
- **`main.py`** — entry point that wires everything together

All pipeline methods are decorated with `@traceable` for LangSmith tracing.

## Running the demo

From the repo root:

```bash
uv run llm_evals/main.py
```

Or run the Mistral demo directly:

```bash
uv run llm_evals/eval/eval.py
```

## `.env` file

Create `llm_evals/.env` with the following keys:

```dotenv
# Required
MISTRAL_API_KEY=your_mistral_api_key

# Optional — enable LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=your_project_name
```

> `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` can also go here if needed by other modules in the workspace.

## Model

Uses `mistral-large-latest` via LangChain's `ChatMistralAI` integration (temperature=0, max_retries=2).
