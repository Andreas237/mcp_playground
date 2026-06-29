"""
Small web UI for the Colorado Budget Research Agent.

A single-file chat interface (Python stdlib http.server — no extra deps) that
lets you ask budget questions in natural language and see the agent's answer
*plus the tools it called*, which is the whole point of this project: watch it
route to the right MCP servers.

Design: the MCP server stack is started ONCE when the UI boots and reused for
every request (a fresh Agent is built per question so conversations don't bleed),
so responses don't pay the ~15s server-startup cost each time. Requests are
serialized with a lock — this is a small local tool, not a production service.

Run (from colorado_budget/):
    .venv/bin/python src/ui.py                 # default profile, http://localhost:8080
    .venv/bin/python src/ui.py --profile devstral --port 8080

Then open http://localhost:8080 in a browser.
"""
from __future__ import annotations

import argparse
import atexit
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from loguru import logger

import agent as agent_mod
from model_config import load_agent_config
from utils import load_api_keys

# Populated in main()
CONFIG = None
_SERVER_PROCS: list = []
_AGENT_LOCK = threading.Lock()

# tool name -> server, for showing which servers the agent routed to.
# Kept in sync with tests/evals/trajectory/_tool_server_map.py.
_TOOL_SERVER = {
    "list_datasets": "open-data", "get_dataset_metadata": "open-data", "query_dataset": "open-data",
    "search_web": "web-search", "search_colorado_government": "web-search",
    "search_bills": "legislature", "get_bill_details": "legislature",
    "get_fiscal_note": "legislature", "find_appropriations_documents": "legislature",
    "search_ospb": "ospb", "find_governor_budget": "ospb",
    "find_revenue_forecast": "ospb", "find_budget_amendments": "ospb",
    "search_revenue": "revenue", "find_legislative_forecast": "revenue",
    "find_tax_expenditure_report": "revenue", "find_tabor_resources": "revenue",
    "colorado_federal_summary": "federal-funds", "federal_funding_by_agency": "federal-funds",
    "top_federal_recipients": "federal-funds", "search_federal_awards": "federal-funds",
    "search_school_finance": "school-finance", "find_school_finance_act": "school-finance",
    "find_per_pupil_funding": "school-finance", "find_finance_formula_resources": "school-finance",
    "search_hcpf": "hcpf", "find_hcpf_budget_request": "hcpf",
    "find_caseload_reports": "hcpf", "find_hcpf_appropriations": "hcpf",
    "search_cpw": "parks-wildlife", "find_cpw_financial_reports": "parks-wildlife",
    "get_sources_and_uses": "parks-wildlife", "find_cpw_appropriations": "parks-wildlife",
    "search_agriculture": "agriculture", "find_agriculture_budget": "agriculture",
    "find_agriculture_appropriations": "agriculture", "find_agriculture_programs": "agriculture",
    "search_cdot": "cdot", "find_cdot_budget": "cdot",
    "find_cdot_appropriations": "cdot", "find_stip": "cdot",
    "search_datasets": "data.gov", "get_keywords": "data.gov", "search_locations": "data.gov",
    "get_location_geometry": "data.gov", "get_harvest_record": "data.gov",
    "get_harvest_record_raw": "data.gov", "get_harvest_record_transformed": "data.gov",
    "fetch_webpage": "inline", "fetch_and_parse_pdf": "inline",
}


def _answer(question: str) -> dict:
    """Run one question through a fresh agent over the already-running stack."""
    with _AGENT_LOCK:
        clients = agent_mod.make_mcp_clients()
        a = agent_mod.build_agent(clients, CONFIG)
        answer = str(a(question))
        trajectory = agent_mod.extract_trajectory(a)
    servers = []
    for t in trajectory:
        s = _TOOL_SERVER.get(t, "unknown")
        if s != "inline" and s not in servers:
            servers.append(s)
    return {"answer": answer, "trajectory": trajectory, "servers": servers}


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Colorado Budget Research Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root { --co-green:#005c3a; --co-gold:#cfae5e; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f4f5f7; color: #1c2024; }
  header { background: var(--co-green); color: #fff; padding: 16px 24px; }
  header h1 { margin: 0; font-size: 18px; }
  header .model { opacity: .8; font-size: 12px; margin-top: 4px; }
  main { max-width: 860px; margin: 0 auto; padding: 24px; }
  form { display: flex; gap: 8px; }
  textarea { flex: 1; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px;
             font: inherit; resize: vertical; min-height: 56px; }
  button { background: var(--co-green); color: #fff; border: 0; border-radius: 8px;
           padding: 0 20px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .examples { margin: 10px 0 0; font-size: 13px; color: #555; }
  .examples a { color: var(--co-green); cursor: pointer; text-decoration: underline; margin-right: 12px; }
  .servers { margin: 18px 0 6px; min-height: 24px; }
  .chip { display: inline-block; background: #e7efe9; color: var(--co-green);
          border: 1px solid var(--co-gold); border-radius: 12px; padding: 2px 10px;
          font-size: 12px; margin: 0 6px 6px 0; }
  .answer { background: #fff; border: 1px solid #e2e4e8; border-radius: 10px;
            padding: 18px 22px; margin-top: 6px; line-height: 1.55; }
  .answer table { border-collapse: collapse; } .answer td, .answer th { border:1px solid #ddd; padding:4px 8px; }
  .status { color:#666; font-style: italic; }
  details { margin-top: 14px; } summary { cursor: pointer; color:#555; font-size: 13px; }
  pre { background:#f0f1f3; padding:10px; border-radius:8px; overflow:auto; font-size:12px; }
</style></head>
<body>
<header><h1>🏔️ Colorado Budget Research Agent</h1><div class="model">__MODEL__</div></header>
<main>
  <form id="f">
    <textarea id="q" placeholder="Ask about Colorado's budget — e.g. how is Parks &amp; Wildlife funded?"></textarea>
    <button id="go" type="submit">Ask</button>
  </form>
  <div class="examples">
    Try:
    <a onclick="ex(this)">How is Colorado Parks &amp; Wildlife funded vs the General Fund?</a>
    <a onclick="ex(this)">What share of Medicaid is federal vs state?</a>
    <a onclick="ex(this)">What is Colorado's projected TABOR surplus?</a>
  </div>
  <div class="servers" id="servers"></div>
  <div id="out"></div>
</main>
<script>
const f=document.getElementById('f'), q=document.getElementById('q'), go=document.getElementById('go'),
      out=document.getElementById('out'), servers=document.getElementById('servers');
function ex(a){ q.value=a.textContent.trim(); q.focus(); }
f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const question=q.value.trim(); if(!question) return;
  go.disabled=true; servers.innerHTML='';
  out.innerHTML='<div class="answer status">Researching… the agent is calling tools across Colorado budget sources. This can take 30–90s.</div>';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
    const d=await r.json();
    if(d.error){ out.innerHTML='<div class="answer">⚠️ '+d.error+'</div>'; }
    else{
      servers.innerHTML='<b style="font-size:12px;color:#555">Servers used:</b> '+
        (d.servers.length? d.servers.map(s=>'<span class="chip">'+s+'</span>').join('') : '<span class="status">none</span>');
      out.innerHTML='<div class="answer">'+marked.parse(d.answer||'(no answer)')+'</div>'+
        '<details><summary>Tool trajectory ('+d.trajectory.length+' calls)</summary><pre>'+
        d.trajectory.join('\\n')+'</pre></details>';
    }
  }catch(err){ out.innerHTML='<div class="answer">⚠️ '+err+'</div>'; }
  go.disabled=false;
});
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = PAGE.replace("__MODEL__",
                                f"{CONFIG.profile_name} · {CONFIG.provider}/{CONFIG.model_id}")
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = (payload.get("question") or "").strip()
            if not question:
                self._send(400, json.dumps({"error": "empty question"}).encode(), "application/json")
                return
            logger.info(f"UI question: {question}")
            result = _answer(question)
            self._send(200, json.dumps(result).encode("utf-8"), "application/json")
        except Exception as e:
            logger.error(f"UI /ask failed: {e}\n{traceback.format_exc()}")
            self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def log_message(self, *args):  # quiet default access logging
        pass


def main() -> int:
    global CONFIG, _SERVER_PROCS
    ap = argparse.ArgumentParser(description="Colorado Budget Agent web UI")
    ap.add_argument("--profile", default=None, help="model profile from config.toml")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    load_api_keys()
    CONFIG = load_agent_config(args.profile)
    logger.info(f"UI profile: {CONFIG.profile_name} ({CONFIG.provider}/{CONFIG.model_id})")
    logger.info("Starting MCP server stack (once)…")
    _SERVER_PROCS = agent_mod.start_servers()
    atexit.register(lambda: agent_mod.stop_servers(_SERVER_PROCS))

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  Colorado Budget Agent UI → http://localhost:{args.port}  "
          f"(profile: {CONFIG.profile_name})\n  Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        agent_mod.stop_servers(_SERVER_PROCS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
