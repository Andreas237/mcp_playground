#!/usr/bin/env bash
# Manual test questions for the Colorado Budget Research Agent.
# Run from the colorado_budget/src/ directory:
#
#   cd colorado_budget/src
#   bash ../tests/manual/questions.sh <question_number>
#
# Or run a single question directly:
#   cd colorado_budget/src && python agent.py "your question"

set -e
AGENT="python agent.py"
Q="${1:-}"

case "$Q" in
  1)
    echo "=== [1] CSDB canonical question ==="
    $AGENT "What changes have been made to funding for the Colorado School for the Deaf and the Blind in the last 10 years?"
    ;;
  2)
    echo "=== [2] Fund types ==="
    $AGENT "What are the different fund types in Colorado's state budget, and how do they differ in terms of legislative flexibility?"
    ;;
  3)
    echo "=== [3] CDOT structured data ==="
    $AGENT "How has CDOT payroll and contract spending changed since 2018?"
    ;;
  4)
    echo "=== [4] JBC process — web search ==="
    $AGENT "How does the Colorado Joint Budget Committee process work, and when does figure-setting happen each year?"
    ;;
  5)
    echo "=== [5] Housing permitting — web search ==="
    $AGENT "How have state permitting regulations and fees impacted the cost of building affordable housing in Colorado since 2020?"
    ;;
  6)
    echo "=== [6] Medicaid ==="
    $AGENT "How has Colorado's Medicaid (Health Care Policy and Financing) budget changed since 2019, and how much comes from federal funds versus the General Fund?"
    ;;
  7)
    echo "=== [7] Dataset discovery ==="
    $AGENT "What datasets are available on data.colorado.gov about Colorado state budget spending or appropriations?"
    ;;
  8)
    echo "=== [8] Specific bill impact ==="
    $AGENT "What fiscal impact did SB23-213 (land use reform) have on state and local government budgets?"
    ;;
  *)
    echo "Colorado Budget Research Agent — manual test questions"
    echo ""
    echo "Usage: bash questions.sh <number>"
    echo ""
    echo "  1  CSDB funding changes (10-year, canonical)"
    echo "  2  Colorado fund types — what they are and how they differ"
    echo "  3  CDOT payroll and contract spending (structured data)"
    echo "  4  JBC budget process and timeline"
    echo "  5  Affordable housing permitting costs since 2020"
    echo "  6  Medicaid GF vs federal funds breakdown"
    echo "  7  Dataset discovery (quick sanity check)"
    echo "  8  SB23-213 land use reform fiscal impact"
    echo ""
    echo "Or run directly:"
    echo "  cd colorado_budget/src && python agent.py \"your question\""
    ;;
esac
