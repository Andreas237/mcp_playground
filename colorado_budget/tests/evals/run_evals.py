"""
Eval runner for the Colorado Budget Research Agent.

Runs a subset of the eval dataset through the agent, scores each response
with the LLM judge, and writes results to evals/results/.

Usage:
    cd colorado_budget/tests
    python evals/run_evals.py                         # all cases
    python evals/run_evals.py --cases csdb_10yr       # one case
    python evals/run_evals.py --cases csdb_10yr jbc_process  # multiple cases
    python evals/run_evals.py --dry-run               # print questions, don't run

Results are written to: tests/evals/results/<timestamp>.json
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
RESULTS_DIR = Path(__file__).parent / "results"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

from dataset import EVAL_DATASET, EvalCase
from judge import judge_response


def run_agent(question: str) -> str:
    from agent import run_agent as _run
    result = _run(question)
    return str(result) if result else ""


def print_result_summary(result: dict, case_id: str) -> None:
    scores = result["scores"]
    prog = result["programmatic"]
    print(f"\n{'─'*60}")
    print(f"  Case: {case_id}")
    print(f"  Factual accuracy:  {scores['factual_accuracy']['score']:.2f}  — {scores['factual_accuracy']['reason'][:80]}")
    print(f"  Fund type accuracy:{scores['fund_type_accuracy']['score']:.2f}  — {scores['fund_type_accuracy']['reason'][:80]}")
    print(f"  Completeness:      {scores['completeness']['score']:.2f}  — {scores['completeness']['reason'][:80]}")
    print(f"  Overall:           {scores['overall']:.2f}")
    print(f"  Citations:         {prog['citation_count']} inline [{prog['has_sources_section'] and 'Sources section ✓' or 'no Sources section'}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Colorado Budget Agent evals")
    parser.add_argument("--cases", nargs="*", help="Eval case IDs to run (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print questions without running")
    args = parser.parse_args()

    from utils import load_api_keys
    load_api_keys()

    cases: list[EvalCase] = EVAL_DATASET
    if args.cases:
        ids = set(args.cases)
        cases = [c for c in EVAL_DATASET if c.id in ids]
        if not cases:
            print(f"No cases matched: {args.cases}")
            print(f"Available: {[c.id for c in EVAL_DATASET]}")
            sys.exit(1)

    if args.dry_run:
        print(f"Would run {len(cases)} eval case(s):")
        for c in cases:
            print(f"  [{c.id}] {c.question[:80]}")
        return

    client = anthropic.Anthropic()
    all_results = []
    start = time.time()

    for case in cases:
        print(f"\n{'='*60}")
        print(f"Running eval: {case.id}")
        print(f"Question: {case.question}")

        try:
            response = run_agent(case.question)
        except Exception as e:
            print(f"  AGENT ERROR: {e}")
            all_results.append({"case_id": case.id, "error": str(e)})
            continue

        result = judge_response(
            client=client,
            case_id=case.id,
            question=case.question,
            response=response,
            expected_facts=case.expected_facts,
            fund_types_expected=case.fund_types_expected,
            notes=case.notes,
        )
        result_dict = result.to_dict()
        print_result_summary(result_dict, case.id)
        all_results.append(result_dict)

    elapsed = time.time() - start

    # Write results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "case_count": len(cases),
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Eval complete — {len(all_results)} case(s) in {elapsed:.0f}s")
    print(f"Results written to: {out_path}")

    # Print aggregate
    scored = [r for r in all_results if "scores" in r]
    if scored:
        avg_overall = sum(r["scores"]["overall"] for r in scored) / len(scored)
        print(f"Average overall score: {avg_overall:.2f} / 1.00")


if __name__ == "__main__":
    main()
