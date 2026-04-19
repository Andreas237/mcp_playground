# mcp-playground

A collection of LLM and MCP experiments built with [uv](https://docs.astral.sh/uv/).

## Prerequisites

### Install uv

`uv` is the Python package manager used across all sub-projects.

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via pip: `pip install uv`

Full docs: https://docs.astral.sh/uv/getting-started/installation/

---

## Sub-projects

### voice_transcription_deepgram

Strands Agents that record live radio streams and transcribe them with Deepgram.

**Additional system dependencies:**
- `ffmpeg` — required for audio capture
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Fedora: `sudo dnf install ffmpeg`
  - macOS: `brew install ffmpeg`

**Install Python dependencies (from repo root):**
```bash
uv sync
```

**`.env` file** — create `.env` in the repo root:
```dotenv
ANTHROPIC_API_KEY=<your_key>
MISTRAL_API_KEY=<your_key>
DEEPGRAM_API_KEY=<your_key>
DEEPGRAM_DEFAULT_MODEL=<model_tag>
```

**Run:**
```bash
uv run voice_transcription_deepgram/src/agents/audioprocess.py
```

See [voice_transcription_deepgram/README.md](voice_transcription_deepgram/README.md) for full details.

---

### llm_testing_ground

Demo and evaluation harness for LLM models with LangSmith observability.

**Install Python dependencies (from repo root):**
```bash
uv sync
```

**`.env` file** — create `.env` in the `llm_testing_ground/` directory:
```dotenv
MISTRAL_API_KEY=<your_key>

# Optional — LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your_key>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=<your_project_name>
```

**Run:**
```bash
uv run llm_testing_ground/main.py
```

See [llm_testing_ground/README.md](llm_testing_ground/README.md) for full details.

---

### llm_evals

LLM-as-judge evaluation harness: a model under test answers a set of prompts, then a panel of judge models (Mistral, NVIDIA Nemotron, Claude Haiku) scores each response on accuracy, precision, and conciseness (1–10).

**Install Python dependencies (from repo root):**
```bash
uv sync
```

**`.env` file** — create `.env` in the `llm_evals/` directory:
```dotenv
ANTHROPIC_API_KEY=<your_key>
MISTRAL_API_KEY=<your_key>
NVIDIA_API_KEY=<your_key>
```

**Run:**
```bash
uv run llm_evals/main.py
```

See [llm_evals/README.md](llm_evals/README.md) for full details.