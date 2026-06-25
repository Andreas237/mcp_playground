"""
LLM-as-judge for Colorado Budget Agent evaluations.

Uses Claude Haiku (fast, cheap) to score responses on three dimensions:
  - factual_accuracy: does the response contain correct facts?
  - fund_type_accuracy: are fund types correctly identified and distinguished?
  - completeness: does the response address the full question?

Each dimension is scored 0.0–1.0 with a brief reason.
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

JUDGE_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class DimensionScore:
    score: float          # 0.0 – 1.0
    reason: str


@dataclass
class EvalResult:
    case_id: str
    question: str
    response_preview: str
    factual_accuracy: DimensionScore
    fund_type_accuracy: DimensionScore
    completeness: DimensionScore
    citation_count: int           # number of [N] inline citations found
    has_sources_section: bool     # does response have a Sources / [1] footer?
    overall: float                # simple average of the three LLM dimensions

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "scores": {
                "factual_accuracy": {"score": self.factual_accuracy.score, "reason": self.factual_accuracy.reason},
                "fund_type_accuracy": {"score": self.fund_type_accuracy.score, "reason": self.fund_type_accuracy.reason},
                "completeness": {"score": self.completeness.score, "reason": self.completeness.reason},
                "overall": round(self.overall, 3),
            },
            "programmatic": {
                "citation_count": self.citation_count,
                "has_sources_section": self.has_sources_section,
            },
        }


def _ask_judge(client: anthropic.Anthropic, prompt: str) -> str:
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _parse_score(text: str) -> tuple[float, str]:
    """Extract 'Score: X.X' and the reason from judge output."""
    match = re.search(r"Score:\s*([\d.]+)", text, re.IGNORECASE)
    score = float(match.group(1)) if match else 0.5
    score = max(0.0, min(1.0, score))
    reason_match = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip()[:300] if reason_match else text[:300]
    return score, reason


def _count_citations(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def _has_sources_section(text: str) -> bool:
    return bool(re.search(r"(?:Sources|References|Bibliography)\s*\n.*\[1\]", text, re.IGNORECASE | re.DOTALL))


FACTUAL_TEMPLATE = """You are evaluating a Colorado budget research agent's response.

QUESTION: {question}

AGENT RESPONSE:
{response}

KNOWN CORRECT FACTS (ground truth):
{notes}

TASK: Score the response for factual accuracy. Consider:
- Does the response contain correct facts about Colorado budget/spending?
- Are specific numbers, dates, and agency names accurate when verifiable?
- Does the response avoid making confident claims that contradict known facts?

Respond ONLY in this format:
Score: [0.0 to 1.0]
Reason: [one or two sentences]"""


FUND_TYPE_TEMPLATE = """You are evaluating a Colorado budget research agent's response.

QUESTION: {question}

AGENT RESPONSE:
{response}

EXPECTED FUND TYPES: {fund_types}

TASK: Score the response for fund type accuracy. Consider:
- Does the response correctly distinguish General Fund from Cash Funds from Federal Funds?
- Are fund type labels used correctly (not mixed up or applied incorrectly)?
- If the question requires fund type distinctions, are they present and accurate?
- Score 1.0 if fund types are not relevant to the question.

Respond ONLY in this format:
Score: [0.0 to 1.0]
Reason: [one or two sentences]"""


COMPLETENESS_TEMPLATE = """You are evaluating a Colorado budget research agent's response.

QUESTION: {question}

AGENT RESPONSE:
{response}

TASK: Score how completely the response addresses the question. Consider:
- Does it answer the main question asked?
- Does it cover the time range requested (if any)?
- Does it provide specific numbers or examples, not just general statements?
- Does it acknowledge gaps or limitations in available data?

Respond ONLY in this format:
Score: [0.0 to 1.0]
Reason: [one or two sentences]"""


def judge_response(
    client: anthropic.Anthropic,
    case_id: str,
    question: str,
    response: str,
    expected_facts: list[str],
    fund_types_expected: list[str],
    notes: str,
) -> EvalResult:
    resp_preview = response[:300] + "..." if len(response) > 300 else response

    factual_raw = _ask_judge(client, FACTUAL_TEMPLATE.format(
        question=question, response=response, notes=notes
    ))
    fund_raw = _ask_judge(client, FUND_TYPE_TEMPLATE.format(
        question=question, response=response,
        fund_types=", ".join(fund_types_expected) if fund_types_expected else "not required for this question"
    ))
    completeness_raw = _ask_judge(client, COMPLETENESS_TEMPLATE.format(
        question=question, response=response
    ))

    f_score, f_reason = _parse_score(factual_raw)
    ft_score, ft_reason = _parse_score(fund_raw)
    c_score, c_reason = _parse_score(completeness_raw)

    return EvalResult(
        case_id=case_id,
        question=question,
        response_preview=resp_preview,
        factual_accuracy=DimensionScore(f_score, f_reason),
        fund_type_accuracy=DimensionScore(ft_score, ft_reason),
        completeness=DimensionScore(c_score, c_reason),
        citation_count=_count_citations(response),
        has_sources_section=_has_sources_section(response),
        overall=(f_score + ft_score + c_score) / 3,
    )
