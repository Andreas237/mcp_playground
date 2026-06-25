"""
Integration smoke tests — full agent runs (LLM + MCP servers + real APIs).

These are SLOW (30–120 seconds each) and cost API tokens. They are
deliberately minimal: we only assert a response was returned and that it
has the expected structural properties (not factual correctness — that's
the eval suite).

Run with:
    pytest tests/integration/test_agent_smoke.py -v -s
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _run(question: str) -> str:
    """Import run_agent fresh each call to avoid port conflicts."""
    from agent import run_agent
    return run_agent(question)


def test_agent_returns_non_empty_response():
    """Minimal smoke test: agent runs without raising and returns text."""
    result = _run("What datasets are available on data.colorado.gov about Colorado state budget spending?")
    assert result is not None
    text = str(result)
    assert len(text) > 100, f"Response too short: {text!r}"


def test_agent_response_contains_citation_markers():
    """Agent should use Wikipedia-style [N] citations per the system prompt."""
    result = _run("What is the Colorado School for the Deaf and Blind and how is it funded?")
    text = str(result)
    # At least one inline citation
    import re
    citations = re.findall(r"\[\d+\]", text)
    assert len(citations) >= 1, f"No [N] citations found in: {text[:500]}"


def test_agent_mentions_fund_types():
    """Agent should identify General Fund vs other fund types on a funding question."""
    result = _run("How is the Colorado Department of Education funded — what fund types does it use?")
    text = str(result).lower()
    assert "general fund" in text, f"'general fund' not in response: {text[:500]}"
