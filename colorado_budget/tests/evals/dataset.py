"""
Gold-standard evaluation dataset for the Colorado Budget Research Agent.

Each entry has:
  - question: what the user asks
  - expected_facts: specific claims the response MUST contain to be correct
  - fund_types_expected: fund types the response should identify
  - notes: context for the judge (not shown to the agent)
"""
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str
    expected_facts: list[str]
    fund_types_expected: list[str] = field(default_factory=list)
    notes: str = ""


EVAL_DATASET: list[EvalCase] = [
    EvalCase(
        id="csdb_10yr",
        question=(
            "What changes have been made to funding for the Colorado School for the Deaf "
            "and the Blind in the last 10 years?"
        ),
        expected_facts=[
            "General Fund",
            "Colorado Springs",
            "teacher salary",
            "enrollment",
        ],
        fund_types_expected=["General Fund"],
        notes=(
            "Known ground truth: GF appropriation grew from ~$10.2M (FY2014-15) to ~$15.8M "
            "(FY2025-26). On-campus enrollment fell from ~215 to ~165 students. Teacher pay "
            "tied to Colorado Springs D11 salary schedule. A $7.6M Jones/Palmer Hall renovation "
            "was approved in 2016. A $289K salary survey correction proposed for FY2026-27."
        ),
    ),
    EvalCase(
        id="general_fund_structure",
        question=(
            "What are the main fund types in Colorado's state budget, and how do they differ "
            "in terms of flexibility and political significance?"
        ),
        expected_facts=[
            "General Fund",
            "Cash Fund",
            "Federal Fund",
        ],
        fund_types_expected=["General Fund", "Cash Funds", "Federal Funds"],
        notes=(
            "Ground truth: CO has four main fund types: General Fund (income + sales tax, "
            "discretionary), Cash Funds (earmarked fees, restricted), Federal Funds (federal "
            "grants, can disappear), Reappropriated Funds (inter-agency transfers). "
            "General Fund is most politically significant. JBC sets appropriations annually."
        ),
    ),
    EvalCase(
        id="cdot_spending",
        question="How has CDOT (Colorado Department of Transportation) spending changed since 2018?",
        expected_facts=[
            "CDOT",
            "transportation",
        ],
        fund_types_expected=["Federal Funds"],
        notes=(
            "CDOT receives significant federal funds (FHWA, FTA) in addition to state highway "
            "funds (HUTF — Highway Users Tax Fund, a Cash Fund). data.colorado.gov has detailed "
            "CDOT payroll and contract data. Federal funding substantially exceeds state GF for CDOT."
        ),
    ),
    EvalCase(
        id="jbc_process",
        question=(
            "How does the Colorado Joint Budget Committee process work, and what role does it "
            "play in setting state appropriations?"
        ),
        expected_facts=[
            "Joint Budget Committee",
            "Long Bill",
            "appropriation",
        ],
        fund_types_expected=[],
        notes=(
            "JBC is a 6-member bipartisan committee (3 House, 3 Senate) that reviews every "
            "agency's budget request and drafts the annual Long Bill (HB 2xxx series). "
            "Figure-setting hearings happen Jan-Mar each year. Staff publishes briefing and "
            "figure-setting documents at content.leg.colorado.gov."
        ),
    ),
    EvalCase(
        id="housing_permitting",
        question=(
            "How have state permitting regulations and fees impacted the cost of building "
            "affordable housing in Colorado since 2020?"
        ),
        expected_facts=[
            "housing",
            "permit",
        ],
        fund_types_expected=[],
        notes=(
            "No single structured dataset covers this well. Agent should use Exa web search "
            "to find Colorado Housing Board reports, DOLA (Dept of Local Affairs) data, and "
            "policy analysis from Colorado Fiscal Institute. HB21-1271 (local government "
            "land use) and SB23-213 (zoning reform) are relevant bills. "
            "This question tests web search fallback when structured data is insufficient."
        ),
    ),
]
