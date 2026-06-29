"""
Cross-server trajectory eval dataset for the Colorado Budget agent.

Each case is a natural-language question plus expectations about which MCP
*servers* the agent should route to (and which it must NOT). This consolidates
the per-server plans (hcpf.md, cpw.md, agriculture.md, cdot.md, …) into one
runnable set, and adds composition cases (questions that should pull two servers)
and routing guards (questions that should pull one server and avoid a tempting
neighbor).

Server names match the FastMCP names in agent.MCP_SERVERS / _tool_server_map.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryCase:
    id: str
    prompt: str
    expected_servers: set[str]                     # all must appear in the trajectory
    forbidden_servers: set[str] = field(default_factory=set)  # none may appear
    expect_facts: list[str] = field(default_factory=list)     # substrings the answer should contain
    note: str = ""                                  # why this case exists


CASES: list[TrajectoryCase] = [
    # ---- single-server routing -------------------------------------------------
    TrajectoryCase(
        id="medicaid_caseload",
        prompt=("How has Colorado Medicaid (HCPF) enrollment / caseload changed "
                "over 2025? Give the monthly figures if you can."),
        expected_servers={"colorado-hcpf"},
        expect_facts=["caseload", "medicaid"],
        note="PECR caseload path; should hit the HCPF server, not generic legislature.",
    ),
    TrajectoryCase(
        id="bill_fiscal_impact",
        prompt=("What fiscal impact did SB23-213 (land use reform) have on state "
                "and local government budgets?"),
        expected_servers={"colorado-legislature"},
        expect_facts=["SB23-213", "fund"],
        note="Bill + fiscal note path.",
    ),
    TrajectoryCase(
        id="tabor_surplus",
        prompt="What is Colorado's projected TABOR surplus and refund this year?",
        expected_servers={"colorado-revenue"},
        forbidden_servers={"colorado-hcpf", "colorado-parks-wildlife", "colorado-cdot"},
        expect_facts=["TABOR"],
        note="Routing guard: broad dept descriptions must not pull a department server.",
    ),
    TrajectoryCase(
        id="school_finance_formula",
        prompt=("How is HB24-1448 changing Colorado's school finance formula, and "
                "what is the statewide per-pupil funding for FY2025-26?"),
        expected_servers={"colorado-school-finance"},
        expect_facts=["per pupil", "HB24-1448"],
    ),
    TrajectoryCase(
        id="cpw_funding",
        prompt=("How is Colorado Parks & Wildlife funded — how much comes from the "
                "General Fund versus licenses and fees?"),
        expected_servers={"colorado-parks-wildlife"},
        forbidden_servers={"colorado-agriculture"},
        expect_facts=["cash", "license", "general fund"],
        note="Fund-type contrast; guard against CPW/Agriculture confusion.",
    ),
    TrajectoryCase(
        id="agriculture_funding",
        prompt="How is the Colorado Department of Agriculture funded?",
        expected_servers={"colorado-agriculture"},
        forbidden_servers={"colorado-parks-wildlife", "colorado-cdot"},
        expect_facts=["cash funds", "agriculture"],
        note="Guard against the other small resource departments.",
    ),
    TrajectoryCase(
        id="cdot_funding",
        prompt=("How is CDOT funded — how much from the HUTF, federal funds, and "
                "the General Fund?"),
        expected_servers={"colorado-cdot"},
        expect_facts=["HUTF", "federal"],
    ),
    TrajectoryCase(
        id="federal_to_colorado",
        prompt="Which Colorado state agencies receive the most federal funding?",
        expected_servers={"colorado-federal-funds"},
        expect_facts=["federal", "HCPF"],
    ),
    TrajectoryCase(
        id="datagov_national",
        prompt=("What federal datasets about agriculture are available on data.gov, "
                "and which organizations publish them?"),
        expected_servers={"data-dot-gov"},
        forbidden_servers={"colorado-agriculture"},
        expect_facts=["data.gov"],
        note="National vs Colorado routing guard.",
    ),

    # ---- composition: should pull TWO servers ---------------------------------
    TrajectoryCase(
        id="hcpf_fund_split",
        prompt=("What share of Colorado's Medicaid program is paid by the federal "
                "government versus the state General Fund?"),
        expected_servers={"colorado-hcpf", "colorado-federal-funds"},
        expect_facts=["federal", "general fund"],
        note="~50/50 split needs both the state budget side and USAspending.",
    ),
    TrajectoryCase(
        id="forecast_gap",
        prompt=("How does the Legislative Council revenue forecast compare to the "
                "Governor's (OSPB) forecast for Colorado?"),
        expected_servers={"colorado-revenue", "colorado-ospb"},
        expect_facts=["forecast"],
        note="The two-forecast gap; needs both the LCS and OSPB servers.",
    ),
    TrajectoryCase(
        id="governor_vs_legislature_k12",
        prompt=("For FY2025-26, how did the Governor's K-12 education budget request "
                "compare to what the legislature actually appropriated?"),
        expected_servers={"colorado-ospb", "colorado-legislature"},
        expect_facts=["education", "general fund"],
        note="Request (OSPB) vs approved (legislature/JBC).",
    ),

    # ---- routing guard: budget vs actuals ------------------------------------
    TrajectoryCase(
        id="cdot_actuals_routing",
        prompt="How has CDOT's actual spending and payroll changed since 2018?",
        expected_servers={"colorado-open-data"},
        forbidden_servers={"colorado-cdot"},
        expect_facts=["CDOT"],
        note="Actuals live in the open-data datasets, not the CDOT budget server.",
    ),
]
