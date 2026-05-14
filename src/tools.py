"""Tools available to the agent: search_info, fetch_url, code_compute.

Merged from the original ``src/tools/*`` sub-package into one module
because the three tools always travel together and the registry layer is
trivial.

Two corpus modes: ``static`` (the local Flask corpus server backing
``search_info`` / ``fetch_url`` against a Wikipedia + BrowseComp-Plus
FAISS index) and ``live`` (Desearch + direct HTTP). One-shot data
collection always uses ``static``; the ``live`` paths are retained for
parity with the main project.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sqlite3
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.config import get_env_value
from src.logging import get_logger

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_TOOL_NAMES: tuple[str, ...] = ("search_info", "fetch_url", "code_compute")


# ---------------------------------------------------------------------------
# Tool-result cache
# ---------------------------------------------------------------------------

class ToolCallCache:
    """Thread-safe (tool_name, query) -> result cache.

    Scoped per task in the one-shot pipeline so personas across rounds
    and schemas can reuse identical tool results (e.g. the same
    ``search_info`` query issued by analyst#1 and critic#2). Safe to
    share across threads.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def get(self, tool_name: str, query: str) -> str | None:
        with self._lock:
            return self._cache.get((tool_name, query))

    def set(self, tool_name: str, query: str, result: str) -> None:
        with self._lock:
            self._cache[(tool_name, query)] = result

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Base class + registry
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Base class for all tools. Unified interface: tool(query) -> str."""

    name: str
    description: str

    @abstractmethod
    def __call__(self, query: str) -> str:
        ...

    def safe_call(self, query: str, max_retries: int = 2) -> str:
        for attempt in range(max_retries):
            try:
                return self(query)
            except Exception:
                logger.warning(
                    "tool_call_failed",
                    tool=self.name,
                    query=query,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                if attempt == max_retries - 1:
                    return f"[Tool '{self.name}' failed after {max_retries} attempts]"
        return f"[Tool '{self.name}' failed]"

    def to_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to pass to the tool",
                        }
                    },
                    "required": ["query"],
                },
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[Tool]:
        result = []
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                result.append(tool)
            else:
                logger.warning("unknown_tool_requested", tool_name=name)
        return result

    @property
    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_openai_tools(
        self, names: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, Tool]]:
        schemas: list[dict[str, Any]] = []
        executors: dict[str, Tool] = {}
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                schemas.append(tool.to_openai_tool_schema())
                executors[tool.name] = tool
            else:
                logger.warning("unknown_tool_requested", tool_name=name)
        return schemas, executors


# ---------------------------------------------------------------------------
# search_info
# ---------------------------------------------------------------------------

@dataclass
class _Hit:
    title: str
    url: str
    snippet: str


def _format_hits(hits: list[_Hit], query: str) -> str:
    if not hits:
        return f"No results found for: {query}"
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] {hit.title}")
        lines.append(f"    URL: {hit.url}")
        if hit.snippet:
            lines.append(f"    Snippet: {hit.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


class _DesearchCache:
    _lock_guard = threading.Lock()
    _query_locks: dict[tuple[str, str], threading.Lock] = {}

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, query_key: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM desearch_api_cache WHERE query_key = ?",
                (query_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None

    def set(self, query_key: str, query: str, payload: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO desearch_api_cache(query_key, query, response_json)"
                " VALUES (?, ?, ?)",
                (query_key, query, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()

    def query_lock(self, query_key: str) -> threading.Lock:
        cache_key = (str(self.path.resolve()), query_key)
        with self._lock_guard:
            lock = self._query_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._query_locks[cache_key] = lock
        return lock

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS desearch_api_cache ("
            "query_key TEXT PRIMARY KEY, query TEXT NOT NULL, response_json TEXT NOT NULL)"
        )
        return conn


class SearchInfoTool(Tool):
    name = "search_info"
    description = (
        "Search the knowledge corpus (or live web at test time) and return "
        "the top hits as title/url/snippet. Use it to discover relevant "
        "documents, then call fetch_url on a promising URL for the full text."
    )

    def __init__(
        self,
        corpus_mode: str = "static",
        corpus_base_url: str = "http://localhost:8000",
        top_n: int = 5,
        timeout: int = 30,
        cache_path: Path | None = None,
    ) -> None:
        if corpus_mode not in ("static", "live"):
            raise ValueError(
                f"corpus_mode must be 'static' or 'live'; got {corpus_mode!r}"
            )
        self.corpus_mode = corpus_mode
        self.corpus_base_url = corpus_base_url.rstrip("/")
        self.top_n = top_n
        self.timeout = timeout
        self._session_local = threading.local()
        self._cache = _DesearchCache(
            cache_path
            or (Path(__file__).resolve().parent.parent / ".cache" / "desearch_cache.sqlite")
        )

    def to_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search and return the top results as a numbered list of "
                    "title/URL/snippet. Use focused queries with key entities "
                    "or constraints. Pair with fetch_url to read a result's "
                    "full text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A focused search query. Use key entities, "
                                "named topics, or distinguishing constraints."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query: str) -> str:
        cleaned = _WHITESPACE_RE.sub(" ", query).strip()
        if not cleaned:
            raise RuntimeError("search_info received empty query")
        if self.corpus_mode == "static":
            hits = self._search_static(cleaned)
        else:
            hits = self._search_live(cleaned)
        return _format_hits(hits, cleaned)

    def _search_static(self, query: str) -> list[_Hit]:
        url = f"{self.corpus_base_url}/query_search"
        try:
            resp = self._get_session().post(
                url,
                json={"query": query, "n": self.top_n},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"corpus server query failed: {exc}") from exc
        data = resp.json()
        raw_hits = data.get("hits", [])
        return [
            _Hit(
                title=str(h.get("title", "")).strip(),
                url=str(h.get("url", "")).strip(),
                snippet=str(h.get("snippet", "")).strip(),
            )
            for h in raw_hits
            if h.get("title") and h.get("url")
        ]

    def _search_live(self, query: str) -> list[_Hit]:
        query_key = self._cache_key(query)
        cached = self._cache.get(query_key)
        if cached is None:
            with self._cache.query_lock(query_key):
                cached = self._cache.get(query_key)
                if cached is None:
                    cached = self._fetch_desearch(query)
                    self._cache.set(query_key, query, cached)
        return self._parse_desearch(cached)

    def _fetch_desearch(self, query: str) -> Any:
        api_key = get_env_value("DESEARCH_API_KEY")
        if not api_key:
            raise RuntimeError("DESEARCH_API_KEY is not set")
        try:
            resp = self._get_session().get(
                "https://api.desearch.ai/web",
                params={"query": query, "num": self.top_n, "start": 0},
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"desearch query failed: {exc}") from exc

    def _parse_desearch(self, data: Any) -> list[_Hit]:
        if isinstance(data, list):
            raw = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            payload = data.get("data", data.get("results", []))
            raw = (
                [item for item in payload if isinstance(item, dict)]
                if isinstance(payload, list)
                else []
            )
        else:
            raw = []
        out: list[_Hit] = []
        for item in raw[: self.top_n]:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", item.get("link", ""))).strip()
            snippet = str(item.get("description", item.get("snippet", ""))).strip()
            if title and url:
                out.append(_Hit(title=title, url=url, snippet=snippet))
        return out

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "SchemaInductionBot/0.1",
                    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                }
            )
            self._session_local.session = session
        return session

    def _cache_key(self, query: str) -> str:
        normalized = _WHITESPACE_RE.sub(" ", query).strip().lower()
        return hashlib.sha256(
            f"desearch_search_info_v1|{normalized}".encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    html = re.sub(r"<(?:br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (
        html.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
    )
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s*\n+", "\n\n", html)
    return html.strip()


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Download the full text of a document at the given URL. Works against "
        "the local static corpus or the live web depending on run mode."
    )

    def __init__(
        self,
        corpus_mode: str = "static",
        corpus_base_url: str = "http://localhost:8000",
        max_chars: int = 5000,
        timeout: int = 30,
    ) -> None:
        if corpus_mode not in ("static", "live"):
            raise ValueError(
                f"corpus_mode must be 'static' or 'live'; got {corpus_mode!r}"
            )
        self.corpus_mode = corpus_mode
        self.corpus_base_url = corpus_base_url.rstrip("/")
        self.max_chars = max_chars
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "SchemaInductionBot/0.1",
                "Accept": "text/html,application/json,application/xhtml+xml;q=0.9,*/*;q=0.8",
            }
        )

    def to_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Fetch the full document at a URL. Pass a URL returned by "
                    "search_info, or a known URL. Returns the document text "
                    "(truncated to a few thousand characters)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The full URL to fetch. For static-corpus runs, "
                                "use a URL returned by search_info."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query: str) -> str:
        url = query.strip()
        if not url:
            raise RuntimeError("fetch_url received empty input")
        if self.corpus_mode == "static":
            text = self._fetch_static(url)
        else:
            text = self._fetch_live(url)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[Content truncated]"
        return text or f"No readable content at: {url}"

    def _fetch_static(self, url: str) -> str:
        endpoint = f"{self.corpus_base_url}/url_search"
        try:
            resp = self.session.post(endpoint, json={"url": url}, timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"corpus server fetch failed: {exc}") from exc
        if resp.status_code == 404:
            return f"URL not found in static corpus: {url}"
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"corpus server fetch failed: {exc}") from exc
        data = resp.json()
        text = data.get("text")
        if not text:
            return f"URL has no text content: {url}"
        return str(text).strip()

    def _fetch_live(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"fetch_url timed out for: {url}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"fetch_url failed for {url}: {exc}") from exc
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type or "application/xhtml" in content_type:
            return _html_to_text(resp.text)
        return resp.text


# ---------------------------------------------------------------------------
# code_compute
# ---------------------------------------------------------------------------

def _is_pure_expression(code: str) -> bool:
    try:
        ast.parse(code, mode="eval")
        return True
    except SyntaxError:
        return False


class CodeComputeTool(Tool):
    name = "code_compute"
    description = (
        "Run a Python expression or full script in a sandboxed subprocess and "
        "return its stdout/result."
    )

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def to_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Execute Python in a sandboxed subprocess and return its stdout. "
                    "Pass a bare expression for math (e.g. '2**10', 'sqrt(144)') and "
                    "the result will be printed automatically; or pass a full script "
                    "with print() for anything more involved. Supports the standard "
                    "library and runs with a 30-second timeout."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A Python expression or full script. Use print() to "
                                "emit output for multi-statement scripts. Do NOT wrap "
                                "the code in markdown fences."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query: str) -> str:
        code = query.strip()
        if not code:
            raise RuntimeError("code_compute received empty input")
        if _is_pure_expression(code):
            code = f"print({code})"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)
        try:
            result = subprocess.run(
                ["python3", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                error = result.stderr.strip()
                if error:
                    output = f"{output}\nError: {error}" if output else f"Error: {error}"
            return output if output else "(no output)"
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"code_compute timed out after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"code_compute failed: {e}") from e
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def active_tool_names(cfg: dict | None = None) -> list[str]:
    """Return enabled tool names from config (or the full default set)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    enabled = cfg.get("tools", {}).get("enabled", list(_TOOL_NAMES))
    return list(enabled)


def create_tool_registry(
    corpus_mode: str = "static",
    corpus_base_url: str | None = None,
    enabled: list[str] | None = None,
) -> ToolRegistry:
    """Create a tool registry with the enabled tool set.

    ``enabled`` defaults to the ``[tools].enabled`` list in config.toml.
    """
    if corpus_base_url is None or enabled is None:
        from src.config import load_config
        cfg = load_config()
        if corpus_base_url is None:
            corpus_base_url = cfg.get("corpus_server", {}).get(
                "base_url", "http://localhost:8000"
            )
        if enabled is None:
            enabled = active_tool_names(cfg)

    registry = ToolRegistry()
    for name in enabled:
        tool: Tool
        if name == "search_info":
            tool = SearchInfoTool(
                corpus_mode=corpus_mode, corpus_base_url=corpus_base_url
            )
        elif name == "fetch_url":
            tool = FetchUrlTool(
                corpus_mode=corpus_mode, corpus_base_url=corpus_base_url
            )
        elif name == "code_compute":
            tool = CodeComputeTool()
        else:
            continue
        registry.register(tool)
    return registry


__all__ = [
    "Tool",
    "ToolCallCache",
    "ToolRegistry",
    "SearchInfoTool",
    "FetchUrlTool",
    "CodeComputeTool",
    "active_tool_names",
    "create_tool_registry",
]
