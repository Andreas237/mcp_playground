import io
from typing import Optional

import httpx
import pdfplumber
from loguru import logger
from strands import tool

TIMEOUT = 60


@tool
def fetch_and_parse_pdf(url: str, page_range: Optional[str] = None, keyword_filter: Optional[str] = None) -> str:
    """Download a PDF and extract its text content. Use for JBC budget documents, fiscal notes, and Long Bill PDFs.

    If the PDF is large, use page_range to extract only the relevant section,
    or keyword_filter to return only pages containing a keyword.

    Args:
        url: Direct URL to the PDF file
        page_range: Page range to extract, e.g. "1-10" or "5" (default: all pages, max 30)
        keyword_filter: If set, return only pages containing this keyword (case-insensitive).
                        Useful for large PDFs — e.g. "deaf blind" or "CSDB"
    """
    try:
        headers = {"User-Agent": "ColoradoBudgetResearchAgent/1.0"}
        r = httpx.get(url, headers=headers, timeout=TIMEOUT, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"PDF_FETCH_FAILED: {e}. Check that the URL is a direct PDF link, not a portal redirect."

    content_type = r.headers.get("content-type", "")
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        return (
            f"URL did not return a PDF (content-type: {content_type}). "
            "This is likely a portal redirect. Try fetch_webpage on the listing page to find the direct PDF URL."
        )

    try:
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF: {url} — {total_pages} pages")

            # Resolve page range
            if page_range:
                parts = page_range.split("-")
                start = int(parts[0]) - 1
                end = int(parts[1]) if len(parts) > 1 else int(parts[0])
                pages = pdf.pages[start:end]
            else:
                pages = pdf.pages[:30]  # cap at 30 pages without an explicit range

            sections: list[str] = []
            for i, page in enumerate(pages):
                text = page.extract_text() or ""
                page_num = (int(page_range.split("-")[0]) - 1 + i + 1) if page_range else (i + 1)

                if keyword_filter and keyword_filter.lower() not in text.lower():
                    continue

                # Also try to extract tables as simple text
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cleaned = [str(cell or "").strip() for cell in row]
                        if any(cleaned):
                            table_text += " | ".join(cleaned) + "\n"

                combined = text
                if table_text:
                    combined += f"\n\n[Tables on page {page_num}]\n{table_text}"

                if combined.strip():
                    sections.append(f"--- Page {page_num} ---\n{combined.strip()}")

    except Exception as e:
        return f"PDF_PARSE_ERROR: {e}"

    if not sections:
        kw_note = f" containing '{keyword_filter}'" if keyword_filter else ""
        return f"No text extracted from pages{kw_note}. PDF may be scanned images (not machine-readable)."

    result = f"PDF: {url}\nTotal pages: {total_pages}\nExtracted: {len(sections)} page(s)\n\n"
    result += "\n\n".join(sections)

    if len(result) > 40000:
        result = result[:40000] + f"\n\n[truncated — use page_range to target specific pages]"

    return result
