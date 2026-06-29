"""
Trajectory eval runner for the Colorado Budget agent.

Brings the MCP server stack up ONCE, then runs each dataset case through a fresh
Agent, extracts the tool-call trajectory (agent.extract_trajectory), and scores
it at the SERVER level: did the agent route to the expected MCP servers, avoid the
forbidden ones, and actually call tools (vs. answering from memory)?

This is the "does the agent use multiple tools correctly" check. It builds on the
helpers the agent already exposes (start_servers / make_mcp_clients / build_agent /
extract_trajectory) and the tool->server map in _tool_server_map.py.

Usage (from colorado_budget/):
    .venv/bin/python tests/evals/trajectory/_runner.py                 # all cases, default profile
    .venv/bin/python tests/evals/trajectory/_runner.py --profile claude
    .venv/bin/python tests/evals/trajectory/_runner.py --cases hcpf_fund_split,forecast_gap
    .venv/bin/python tests/evals/trajectory/_runner.py --list

Results are printed and written to tests/evals/trajectory/results/ as JSON + MD.
Each case is flushed to disk as it finishes, so a partial run is never lost.

Cost: each case is a full agent run (real LLM + live .gov calls), ~$0.02-0.10.
Run on demand / pre-release, not in CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "src"          # colorado_budget/src
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from loguru import logger

import agent as agent_mod                          # noqa: E402
from model_config import load_agent_config          # noqa: E402
from utils import load_api_keys                      # noqa: E402

from _tool_server_map import servers_touched         # noqa: E402
from dataset import CASES                            # noqa: E402

RESULTS_DIR = HERE / "results"


def _score_case(case, answer: str, trajectory: list[str], error: str | None) -> dict:
    touched = servers_touched(trajectory)
    missing = sorted(case.expected_servers - touched)
    forbidden_hit = sorted(case.forbidden_servers & touched)
    used_tools = len(trajectory) > 0
    facts_found = [f for f in case.expect_facts if f.lower() in (answer or "").lower()]
    passed = (not missing) and (not forbidden_hit) and used_tools and (error is None)
    return {
        "id": case.id,
        "passed": passed,
        "error": error,
        "used_tools": used_tools,
        "trajectory": trajectory,
        "n_tool_calls": len(trajectory),
        "servers_touched": sorted(touched),
        "expected_servers": sorted(case.expected_servers),
        "missing_servers": missing,
        "forbidden_servers": sorted(case.forbidden_servers),
        "forbidden_hit": forbidden_hit,
        "facts_expected": case.expect_facts,
        "facts_found": facts_found,
        "answer_chars": len(answer or ""),
    }


def _run_one(case, config) -> dict:
    clients = agent_mod.make_mcp_clients()
    a = agent_mod.build_agent(clients, config)
    t0 = time.time()
    answer, error = "", None
    try:
        answer = str(a(case.prompt))
    except Exception as e:  # a single case failing must not abort the suite
        error = f"{type(e).__name__}: {e}"
        logger.warning(f"[{case.id}] errored: {error}")
    trajectory = agent_mod.extract_trajectory(a)
    result = _score_case(case, answer, trajectory, error)
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def _write_results(results: list[dict], profile: str, started: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.replace(":", "").replace("-", "").replace(" ", "_")
    base = RESULTS_DIR / f"trajectory_{profile}_{stamp}"
    passed = sum(r["passed"] for r in results)
    total = len(results)
    summary = {
        "profile": profile,
        "started": started,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "no_tool_cases": [r["id"] for r in results if not r["used_tools"]],
        "forbidden_violations": [r["id"] for r in results if r["forbidden_hit"]],
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in results) / total, 1) if total else 0,
        "results": results,
    }
    base.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    base.with_suffix(".md").write_text(_render_md(summary))
    return base


def _render_md(summary: dict) -> str:
    lines = [
        f"# Trajectory eval — profile `{summary['profile']}` — {summary['started']}",
        "",
        f"**{summary['passed']}/{summary['total']} passed** "
        f"(pass rate {summary['pass_rate']}). "
        f"Avg tool calls/case: {summary['avg_tool_calls']}.",
        "",
        "| Case | Result | Expected servers | Touched | Missing | Forbidden hit | Tools | Facts |",
        "|------|--------|------------------|---------|---------|---------------|-------|-------|",
    ]
    for r in summary["results"]:
        status = "✅" if r["passed"] else "❌"
        if r["error"]:
            status = "💥"
        lines.append(
            f"| {r['id']} | {status} | {','.join(r['expected_servers'])} | "
            f"{','.join(r['servers_touched']) or '—'} | {','.join(r['missing_servers']) or '—'} | "
            f"{','.join(r['forbidden_hit']) or '—'} | {r['n_tool_calls']} | "
            f"{len(r['facts_found'])}/{len(r['facts_expected'])} |"
        )
    if summary["no_tool_cases"]:
        lines += ["", f"⚠️ Cases that called **no tools** (answered from memory): "
                      f"{', '.join(summary['no_tool_cases'])}"]
    if summary["forbidden_violations"]:
        lines += ["", f"⚠️ Cases that touched a **forbidden** server: "
                      f"{', '.join(summary['forbidden_violations'])}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Colorado Budget agent trajectory eval")
    ap.add_argument("--profile", default=None, help="model profile (default: config active_profile)")
    ap.add_argument("--cases", default="", help="comma-separated case ids to run (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="run at most N cases")
    ap.add_argument("--list", action="store_true", help="list case ids and exit")
    args = ap.parse_args()

    if args.list:
        for c in CASES:
            print(f"{c.id:28s} -> expect {sorted(c.expected_servers)}"
                  + (f" forbid {sorted(c.forbidden_servers)}" if c.forbidden_servers else ""))
        return 0

    cases = CASES
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in CASES if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases selected.")
        return 1

    load_api_keys()
    config = load_agent_config(args.profile)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Trajectory eval: {len(cases)} cases, profile "
          f"{config.profile_name} ({config.provider}/{config.model_id})\n")

    processes = agent_mod.start_servers()
    results: list[dict] = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.id} … ", end="", flush=True)
            r = _run_one(case, config)
            results.append(r)
            flag = "PASS" if r["passed"] else ("ERROR" if r["error"] else "FAIL")
            print(f"{flag}  tools={r['n_tool_calls']} "
                  f"touched={r['servers_touched']} "
                  f"missing={r['missing_servers']} forbidden={r['forbidden_hit']} "
                  f"({r['elapsed_s']}s)")
            _write_results(results, config.profile_name, started)  # flush after each case
    finally:
        agent_mod.stop_servers(processes)

    out = _write_results(results, config.profile_name, started)
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed. Results: {out.with_suffix('.md')}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
