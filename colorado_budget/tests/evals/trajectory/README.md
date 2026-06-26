# Trajectory Test Plans (beyond unit testing)

Unit tests (`tests/unit/`) prove each tool's Python logic in isolation with mocked
network. Integration tests (`tests/integration/test_mcp_server.py`) prove each server
boots and advertises the right tools. **Neither proves the *agent* actually picks the
right tool for a question and chains tools in a sensible order.** That is what these
trajectory test plans cover, using the [Strands Evals SDK](https://strandsagents.com/docs/user-guide/evals-sdk/quickstart/).

## What a trajectory eval checks

For a natural-language question, we assert on the **sequence of tools the agent
called** (its *trajectory*) — not just the final text. This catches regressions like
"the agent answered from memory instead of calling a tool" or "it called the wrong
server's tool." Each case also carries lightweight **output assertions** (substrings /
facts the answer should contain).

Per the SDK, trajectory scoring offers three matchers:
- `exact_match_scorer` — the exact tool sequence, in order
- `in_order_match_scorer` — expected tools appear in order (other tools may interleave)
- `any_order_match_scorer` — expected tools all appear, order-independent

Research workflows are adaptive, so we default to **`in_order_match_scorer`** for
multi-step cases and **`any_order_match_scorer`** for cross-server cases where the
order is genuinely interchangeable.

## Install

`strands_evals` is **not** in the project venv yet. To run these:

```bash
.venv/bin/python -m pip install strands-evals    # (uv: uv add --dev strands-evals)
```

(They are written to be runnable once the package is present; until then this is a
specification. Unit + integration tests do not depend on `strands_evals`.)

## Reusable runner

Each per-server plan defines a list of `Case`s. The harness below is identical across
servers — only the cases differ. It boots the real agent (which spawns the MCP
servers), runs each question, extracts the tool trajectory from `agent.messages`, and
scores it.

```python
# tests/evals/trajectory/_runner.py  (sketch)
from strands_evals import Case, Experiment
from strands_evals.evaluators import TrajectoryEvaluator, OutputEvaluator
from strands_evals.extractors import tools_use_extractor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from agent import run_agent_collecting  # thin wrapper that returns (text, agent)

def task(case: Case) -> dict:
    text, agent = run_agent_collecting(case.input)          # uses default profile (claude)
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    return {"output": text, "trajectory": trajectory}

def run(cases):
    experiment = Experiment(cases=cases, evaluators=[
        TrajectoryEvaluator(rubric="in_order_match_scorer; partial credit for subset"),
        OutputEvaluator(rubric="answer contains the expected facts/figures"),
    ])
    return experiment.run_evaluations(task)
```

> `run_agent_collecting` is a small helper to add to `agent.py` that returns the
> `Agent` object alongside the answer (today `run_agent` returns only the string).
> Trajectory evals need `agent.messages`, hence the wrapper.

## Cost & cadence

These boot a full agent and make real LLM + network calls (~$0.02–0.10/case). Run them
**per server before a release / after touching a server's tools or the system prompt** —
not on every commit. Unit + integration stay in the fast pre-commit path.

## Plans

| Server | Plan |
|--------|------|
| `colorado-hcpf` | [hcpf.md](hcpf.md) |
| `colorado-parks-wildlife` | [cpw.md](cpw.md) |
| `colorado-agriculture` | [agriculture.md](agriculture.md) |
| `colorado-cdot` | [cdot.md](cdot.md) |
