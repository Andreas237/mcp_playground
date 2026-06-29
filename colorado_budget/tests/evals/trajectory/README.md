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

## Implemented runner — how to run

The plan in this folder is now executable. Three files implement it:

| File | What it is |
|------|-----------|
| [`dataset.py`](dataset.py) | The runnable cases — `TrajectoryCase`s consolidating the per-server `.md` plans, plus composition cases (two servers) and routing guards (forbidden servers). |
| [`_tool_server_map.py`](_tool_server_map.py) | Maps every tool → its MCP server, so scoring works at the **server** level (`servers_touched`). |
| [`_runner.py`](_runner.py) | Brings the stack up once, runs each case through a fresh `Agent`, extracts the trajectory via `agent.extract_trajectory`, scores server routing, writes `results/`. |

```bash
cd colorado_budget
python tests/evals/trajectory/_runner.py --list                 # list cases (no LLM)
python tests/evals/trajectory/_runner.py                        # all cases, default profile
python tests/evals/trajectory/_runner.py --profile claude --cases hcpf_fund_split,forecast_gap
python tests/evals/trajectory/_runner.py --limit 3              # quick subset
```

Results (per-case PASS/FAIL, servers touched vs expected, forbidden hits, tool counts)
print live and are written to `results/trajectory_<profile>_<timestamp>.{json,md}`, flushed
after each case so a partial run is never lost.

**Scoring (server-level):** a case passes when every `expected_servers` entry appears in the
trajectory, no `forbidden_servers` entry appears, and the agent actually called ≥1 tool. This
is intentionally more robust than pinning exact tool names (the agent may pick a different but
valid tool on the same server).

The runner builds on the agent's own helpers — `start_servers` / `make_mcp_clients` /
`build_agent` / `extract_trajectory` in [`src/agent.py`](../../../src/agent.py) — so it does
not depend on the `strands_evals` package. (`strands_evals` is installed and its
`Case`/`Experiment`/`TrajectoryEvaluator` can wrap this runner later for LLM-judged output
scoring; the per-server `.md` files show that `Case` form.)

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
