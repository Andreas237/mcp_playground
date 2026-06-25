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


def run_agent(question: str, profile: str | None = None) -> str:
    load_api_keys()
    config = load_agent_config(profile)

    # Start MCP servers as subprocesses
    processes = []
    for srv in MCP_SERVERS:
        logger.info(f"Starting {srv['name']} on port {srv['port']}")
        p = subprocess.Popen(
            [sys.executable, str(srv["script"])],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(p)

    try:
        # Wait for each server to be ready
        for srv in MCP_SERVERS:
            if not _wait_for_server(srv["port"], srv["name"]):
                logger.error(f"Failed to start {srv['name']} — proceeding without it")

        # MCPClient instances — Agent manages their lifecycle (do NOT use as context manager here)
        open_data   = MCPClient(lambda: streamablehttp_client("http://localhost:8001/mcp"))
        web_search  = MCPClient(lambda: streamablehttp_client("http://localhost:8002/mcp"))
        legislature = MCPClient(lambda: streamablehttp_client("http://localhost:8003/mcp"))
        ospb        = MCPClient(lambda: streamablehttp_client("http://localhost:8004/mcp"))

        agent = Agent(
            model=config.model,
            tools=[open_data, web_search, legislature, ospb, fetch_webpage, fetch_and_parse_pdf],
            system_prompt=config.system_prompt,
        )
        logger.info(f"Profile: {config.profile_name} ({config.provider}/{config.model_id})")
        logger.info(f"Question: {question}")
        response = agent(question)
        answer = str(response)
        print(f"\n{'='*60}\nAnswer:\n{'='*60}\n{answer}\n")
        return answer

    finally:
        for p in processes:
            p.terminate()
        logger.info("MCP servers stopped")


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
