"""
Colorado Budget Research Agent

Usage:
    python agent.py "your question here"
    python agent.py  # runs the default dataset-discovery question

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
from strands.models.anthropic import AnthropicModel
from strands.tools.mcp import MCPClient

sys.path.insert(0, str(Path(__file__).parent))
from tools.fetch_webpage import fetch_webpage
from tools.pdf_parser import fetch_and_parse_pdf
from utils import load_api_keys

SRC_DIR = Path(__file__).parent

MCP_SERVERS = [
    {"name": "colorado-open-data", "script": SRC_DIR / "servers" / "colorado_open_data.py", "port": 8001},
    {"name": "web-search",         "script": SRC_DIR / "servers" / "web_search.py",         "port": 8002},
]

SYSTEM_PROMPT = """You are a Colorado state budget research assistant. Your purpose is to help \
citizens, journalists, and policy researchers understand how Colorado state government spends \
public money — and to verify or challenge political claims with primary sources.

## Tools available

**MCP tools (structured data + web search):**
- `colorado-open-data` tools: list_datasets, get_dataset_metadata, query_dataset — Socrata SODA API
- `web-search` tools: search_web, search_colorado_government — Exa neural search

**Inline tools:**
- fetch_webpage(url) — fetch and strip HTML from any CO government page
- fetch_and_parse_pdf(url, page_range, keyword_filter) — download and extract text/tables from PDFs

## Research workflow

**Before your first tool call**, state your plan:
- What sources are most likely to have this data?
- What is your first call, and what fallback if it returns nothing?

**After each result**, note:
- What did I learn? Does this answer the question, partially, or not at all?
- What is the most useful next call?

**When a tool fails or returns empty**, reflect before retrying:
- Why did it fail? (wrong URL pattern, portal redirect, no matching records?)
- What different approach avoids the same failure?
- Do NOT retry the same URL or query more than once.

**Stop when you have enough** to give a well-sourced answer. Abandoned dead ends are fine \
— say so and move on.

## Source priority
1. data.colorado.gov (SODA API) — structured, queryable; best for CDOT and TOPS
2. leg.colorado.gov — Long Bill, JBC Appropriations History, fiscal notes, bill search
3. ospb.colorado.gov — Governor's budget requests, revenue forecasts
4. search_colorado_government — finds official CO government pages by topic
5. search_web — press coverage, policy analysis, national context

## Fund types — always distinguish
| Type | Meaning |
|------|---------|
| General Fund | State income + sales tax; discretionary; politically significant |
| Cash Funds | Earmarked fees; often cannot be redirected |
| Federal Funds | Federal grants; can disappear with federal policy changes |
| Reappropriated | Transfers between agencies; often misunderstood in political claims |

## Output format — Wikipedia-style citations

Write naturally. When you state a fact from a source, add an inline citation number like \
this[1]. At the end of your response, include a **Sources** section:

[1] Dataset name or document title — URL or dataset ID
[2] ...

Number citations sequentially. If the same source is cited multiple times, reuse its number. \
Do not add a citation for general background knowledge — only for specific facts from tool results.
"""


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


def run_agent(question: str) -> None:
    load_api_keys()

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
        open_data = MCPClient(lambda: streamablehttp_client("http://localhost:8001/mcp"))
        web_search = MCPClient(lambda: streamablehttp_client("http://localhost:8002/mcp"))

        model = AnthropicModel(model_id="claude-sonnet-4-6", max_tokens=8096)

        agent = Agent(
            model=model,
            tools=[open_data, web_search, fetch_webpage, fetch_and_parse_pdf],
            system_prompt=SYSTEM_PROMPT,
        )
        logger.info(f"Question: {question}")
        response = agent(question)
        print(f"\n{'='*60}\nAnswer:\n{'='*60}\n{response}\n")

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
    args = parser.parse_args()
    run_agent(args.question)
