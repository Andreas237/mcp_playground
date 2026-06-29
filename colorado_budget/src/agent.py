"""
Colorado Budget Research Agent

Usage:
    python agent.py "your question here"
    python agent.py --profile devstral "your question"
    python agent.py --profile nvidia "your question"
    python agent.py  # runs the default dataset-discovery question

The model, API key, and system prompt come from config.toml (see model_config.py).
Switch LLMs with --profile or by changing `active_profile` in config.toml.

The agent starts the MCP servers as background subprocesses, connects to them,
then runs the query and shuts everything down cleanly.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx
from loguru import logger
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

sys.path.insert(0, str(Path(__file__).parent))
from model_config import load_agent_config
from tools.fetch_webpage import fetch_webpage
from tools.pdf_parser import fetch_and_parse_pdf
from utils import load_api_keys

SRC_DIR = Path(__file__).parent

MCP_SERVERS = [
    {"name": "colorado-open-data",   "script": SRC_DIR / "servers" / "colorado_open_data.py", "port": 8001},
    {"name": "web-search",           "script": SRC_DIR / "servers" / "web_search.py",          "port": 8002},
    {"name": "colorado-legislature",  "script": SRC_DIR / "servers" / "legislature.py",        "port": 8003},
    {"name": "colorado-ospb",        "script": SRC_DIR / "servers" / "ospb.py",                "port": 8004},
    {"name": "colorado-revenue",     "script": SRC_DIR / "servers" / "revenue.py",             "port": 8005},
    {"name": "colorado-federal-funds", "script": SRC_DIR / "servers" / "federal_funds.py",      "port": 8006},
    {"name": "colorado-school-finance", "script": SRC_DIR / "servers" / "school_finance.py",    "port": 8007},
    {"name": "colorado-hcpf",        "script": SRC_DIR / "servers" / "hcpf.py",                "port": 8008},
    {"name": "colorado-parks-wildlife", "script": SRC_DIR / "servers" / "cpw.py",             "port": 8009},
    {"name": "colorado-agriculture", "script": SRC_DIR / "servers" / "agriculture.py",        "port": 8010},
    {"name": "colorado-cdot",        "script": SRC_DIR / "servers" / "cdot.py",                "port": 8011},
    {"name": "data-dot-gov",        "script": SRC_DIR / "servers" / "data-dot-gov.py",                "port": 8012},
]


def _wait_for_server(port: int, name: str, timeout: int = 20) -> bool:
    """Wait until the server is accepting connections on port. Any HTTP response means it's up."""
    url = f"http://localhost:{port}/mcp"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.post(url, timeout=2)  # any response (even 406) means the port is live
            return True
        except httpx.ConnectError:
            time.sleep(0.4)
        except Exception:
            return True  # non-connection error means server is up
    logger.warning(f"Server {name} did not respond on port {port} within {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Reusable building blocks (used by run_agent and by the eval harness, which
# needs to bring the server stack up once and build many fresh agents).
# ---------------------------------------------------------------------------

def start_servers() -> list[subprocess.Popen]:
    """Spawn every MCP server as a subprocess and wait until each is reachable."""
    processes = []
    for srv in MCP_SERVERS:
        logger.info(f"Starting {srv['name']} on port {srv['port']}")
        processes.append(subprocess.Popen(
            [sys.executable, str(srv["script"])],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ))
    for srv in MCP_SERVERS:
        if not _wait_for_server(srv["port"], srv["name"]):
            logger.error(f"Failed to start {srv['name']} — proceeding without it")
    return processes


def stop_servers(processes: list[subprocess.Popen]) -> None:
    for p in processes:
        p.terminate()
    logger.info("MCP servers stopped")


def make_mcp_clients() -> dict[str, MCPClient]:
    """Create one MCPClient per server, keyed by server name (order = MCP_SERVERS).

    Clients are cheap and connect to the already-running server subprocesses, so
    the eval harness makes fresh clients + a fresh Agent per case (servers stay up).
    """
    clients: dict[str, MCPClient] = {}
    for srv in MCP_SERVERS:
        port = srv["port"]
        clients[srv["name"]] = MCPClient(
            lambda port=port: streamablehttp_client(f"http://localhost:{port}/mcp")
        )
    return clients


def build_agent(clients: dict[str, MCPClient], config) -> Agent:
    """Assemble an Agent from the MCP clients + inline tools for a given config."""
    return Agent(
        model=config.model,
        tools=[*clients.values(), fetch_webpage, fetch_and_parse_pdf],
        system_prompt=config.system_prompt,
    )


def extract_trajectory(agent: Agent) -> list[str]:
    """Return the ordered list of tool names the agent called, from agent.messages.

    Each tool call appears as an assistant content block with a 'toolUse' entry.
    Robust to dict- or object-shaped content across Strands versions.
    """
    trajectory: list[str] = []
    for msg in getattr(agent, "messages", []) or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        for block in content or []:
            tu = block.get("toolUse") if isinstance(block, dict) else getattr(block, "toolUse", None)
            if not tu:
                continue
            name = tu.get("name") if isinstance(tu, dict) else getattr(tu, "name", None)
            if name:
                trajectory.append(name)
    return trajectory


def run_agent(question: str, profile: str | None = None) -> str:
    load_api_keys()
    config = load_agent_config(profile)
    processes = start_servers()
    try:
        clients = make_mcp_clients()
        agent = build_agent(clients, config)
        logger.info(f"Profile: {config.profile_name} ({config.provider}/{config.model_id})")
        logger.info(f"Question: {question}")
        response = agent(question)
        answer = str(response)
        print(f"\n{'='*60}\nAnswer:\n{'='*60}\n{answer}\n")
        return answer
    finally:
        stop_servers(processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorado Budget Research Agent")
    parser.add_argument(
        "question",
        nargs="?",
        default="What datasets are available on data.colorado.gov about Colorado state budget spending or appropriations?",
        help="Policy question to research",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Model profile from config.toml (e.g. claude, devstral, nvidia). "
             "Defaults to active_profile in config.toml.",
    )
    args = parser.parse_args()
    run_agent(args.question, profile=args.profile)
