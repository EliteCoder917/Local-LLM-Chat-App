"""Web tools — keyless internet search + page fetch via DuckDuckGo.

Used by chat "Search" mode. No API key required: we hit DuckDuckGo's HTML
endpoint and parse results, and fetch arbitrary pages with a light
tag-stripping reader. Gated behind the `network` permission.

These are deliberately dependency-light (httpx + stdlib regex) so they bundle
cleanly into the PyInstaller build with no extra wheels.
"""
from __future__ import annotations

import html
import re
import urllib.parse
from typing import List

import httpx

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
_TIMEOUT = 15.0

# DuckDuckGo's no-JS HTML endpoint. Returns server-rendered result blocks we
# can parse without a headless browser.
_DDG_HTML = "https://html.duckduckgo.com/html/"

_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.S | re.I,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snip>.*?)</a>',
    re.S | re.I,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n\s*\n+")


def _strip_html(s: str) -> str:
    """Remove tags + unescape entities + collapse runs of spaces."""
    text = _TAG_RE.sub("", s)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps result links as `//duckduckgo.com/l/?uddg=<enc>`.
    Pull the real destination out of the `uddg` query param."""
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urllib.parse.urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return qs["uddg"][0]
    except Exception:  # noqa: BLE001
        pass
    return href


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) and return a numbered list of results with
    title, URL, and snippet. Use `web_fetch` to read a result's full page."""
    q = (query or "").strip()
    if not q:
        return "Error: empty search query."
    n = max(1, min(int(max_results or 5), 10))
    try:
        resp = httpx.post(
            _DDG_HTML,
            data={"q": q},
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Error: web search failed: {type(e).__name__}: {e}"

    body = resp.text
    titles = list(_RESULT_RE.finditer(body))
    snippets = _SNIPPET_RE.findall(body)

    if not titles:
        return f"No results found for '{q}'."

    lines: List[str] = [f"Search results for '{q}':\n"]
    for i, m in enumerate(titles[:n]):
        title = _strip_html(m.group("title"))
        url = _unwrap_ddg(m.group("href"))
        snip = _strip_html(snippets[i]) if i < len(snippets) else ""
        lines.append(f"{i + 1}. {title}\n   {url}")
        if snip:
            lines.append(f"   {snip}")
    return "\n".join(lines)


def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and return its readable text (scripts/styles/markup
    stripped). Truncated to `max_chars`. Use after `web_search` to read a page."""
    u = (url or "").strip()
    if not u:
        return "Error: empty URL."
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    cap = max(500, min(int(max_chars or 8000), 40000))
    try:
        resp = httpx.get(
            u,
            headers={"User-Agent": _UA, "Accept": "text/html,*/*"},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Error: fetch failed: {type(e).__name__}: {e}"

    ctype = resp.headers.get("content-type", "")
    raw = resp.text

    if "html" in ctype or raw.lstrip()[:1] == "<":
        # Drop non-content elements wholesale, then strip remaining tags.
        cleaned = re.sub(r"(?is)<(script|style|noscript|head|svg)[^>]*>.*?</\1>", " ", raw)
        cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
        text = _strip_html(cleaned)
        text = _BLANKLINES_RE.sub("\n\n", text)
    else:
        text = raw  # plain text / json / etc. — return as-is

    text = text.strip()
    if len(text) > cap:
        text = text[:cap] + f"\n\n… [truncated at {cap} chars]"
    title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    header = f"# {_strip_html(title_m.group(1))}\n{u}\n\n" if title_m else f"{u}\n\n"
    return header + text
