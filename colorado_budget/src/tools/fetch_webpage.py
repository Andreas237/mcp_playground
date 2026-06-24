import re
from html.parser import HTMLParser

import httpx
from loguru import logger
from strands import tool

TIMEOUT = 30


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


@tool
def fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and return its readable text content (HTML tags stripped).

    Use for Colorado government pages that have no structured API:
    - leg.colorado.gov — bill search, Long Bill documents, JBC pages
    - ospb.colorado.gov — Governor's budget requests
    - leg.colorado.gov/offices/joint-budget-committee — Appropriations History Reports

    If this returns a truncation notice and you need more, increase max_chars (up to ~50000).

    Args:
        url: Full URL to fetch
        max_chars: Maximum characters to return (default 8000)
    """
    try:
        headers = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}
        r = httpx.get(url, headers=headers, timeout=TIMEOUT, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"FETCH_FAILED: {e}. Try a different URL or search_colorado_government for this topic."

    if "html" in r.headers.get("content-type", ""):
        parser = _TextExtractor()
        parser.feed(r.text)
        text = parser.get_text()
    else:
        text = r.text

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated — {len(text) - max_chars} more chars; pass max_chars={len(text)} to get full page]"

    logger.info(f"fetch: {url} → {len(text)} chars")
    return text
