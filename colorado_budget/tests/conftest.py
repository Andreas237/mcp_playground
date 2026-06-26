"""
Shared pytest fixtures for the Colorado Budget Agent test suite.
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Add src/ to path so all test files can import from it
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def api_keys():
    from utils import load_api_keys
    return load_api_keys()


@pytest.fixture(scope="session")
def open_data_server():
    """Start the colorado-open-data MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "colorado_open_data.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8001, "colorado-open-data", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def web_search_server(api_keys):
    """Start the web-search MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "web_search.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8002, "web-search", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def legislature_server(api_keys):
    """Start the colorado-legislature MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "legislature.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8003, "colorado-legislature", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def ospb_server(api_keys):
    """Start the colorado-ospb MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "ospb.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8004, "colorado-ospb", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def revenue_server(api_keys):
    """Start the colorado-revenue MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "revenue.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8005, "colorado-revenue", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def federal_funds_server(api_keys):
    """Start the colorado-federal-funds MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "federal_funds.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8006, "colorado-federal-funds", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def school_finance_server(api_keys):
    """Start the colorado-school-finance MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "school_finance.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8007, "colorado-school-finance", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def hcpf_server(api_keys):
    """Start the colorado-hcpf MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "hcpf.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8008, "colorado-hcpf", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def cpw_server(api_keys):
    """Start the colorado-parks-wildlife MCP server and yield; stop on teardown."""
    script = SRC / "servers" / "cpw.py"
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_port(8009, "colorado-parks-wildlife", timeout=20)
    yield proc
    proc.terminate()
    proc.wait()


def _wait_for_port(port: int, name: str, timeout: int = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.post(f"http://localhost:{port}/mcp", timeout=1)
            return
        except httpx.ConnectError:
            time.sleep(0.4)
        except Exception:
            return  # any non-connection error means server is up
    pytest.fail(f"Server {name} did not start on port {port} within {timeout}s")
