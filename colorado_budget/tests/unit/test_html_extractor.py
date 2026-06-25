"""Unit tests for the _TextExtractor HTML parser in tools/fetch_webpage.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tools.fetch_webpage import _TextExtractor


def _parse(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return p.get_text()


def test_basic_text():
    assert _parse("<p>Hello world</p>") == "Hello world"


def test_strips_script():
    text = _parse("<p>Keep</p><script>alert('drop me')</script><p>this</p>")
    assert "Keep" in text
    assert "this" in text
    assert "alert" not in text
    assert "drop me" not in text


def test_strips_style():
    text = _parse("<style>body { color: red }</style><p>Content</p>")
    assert "Content" in text
    assert "color" not in text


def test_strips_noscript():
    text = _parse("<noscript>enable js</noscript><span>Real text</span>")
    assert "Real text" in text
    assert "enable js" not in text


def test_collapses_whitespace():
    text = _parse("<p>  lots   of   space  </p>")
    assert "  " not in text
    assert "lots of space" in text


def test_nested_skip_tags():
    # Nested script tags should still be fully skipped
    text = _parse("<script><script>inner</script></script><p>after</p>")
    assert "inner" not in text
    assert "after" in text


def test_empty_html():
    assert _parse("") == ""


def test_extracts_from_nested_divs():
    text = _parse("<div><h1>Title</h1><div><p>Body</p></div></div>")
    assert "Title" in text
    assert "Body" in text


def test_real_like_page():
    html = """
    <html>
      <head><style>.cls{color:blue}</style></head>
      <body>
        <nav><script>nav();</script></nav>
        <h1>Colorado State Budget</h1>
        <p>The General Fund appropriation was <strong>$15.8 million</strong>.</p>
        <script>trackEvent('pageview');</script>
      </body>
    </html>
    """
    text = _parse(html)
    assert "Colorado State Budget" in text
    assert "15.8 million" in text
    assert "trackEvent" not in text
    assert "color:blue" not in text
