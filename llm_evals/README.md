# LLM Evals

Runs an LLM evaluation using an LLM-as-judge pattern: a model under test answers a set of prompts, then a panel of judge models scores each response on accuracy, precision, and conciseness (1–10).

## How to run

From the repo root:

```bash
uv run llm_evals/main.py
```

## How it works

Prompts are processed one at a time — the model under test answers, then each judge scores it, before moving to the next prompt. This is simple but slow.

## Next steps

- Run prompts and judge calls concurrently to reduce wall-clock time
