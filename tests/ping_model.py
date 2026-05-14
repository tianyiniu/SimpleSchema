"""Ping a vLLM server (or OpenRouter) and exercise a single tool call.

Sends a prompt to the model asking it to call `search_info`, which proxies to
the local Flask corpus server's `POST /query_search` endpoint. Expects the
model to invoke the tool with the query "machine learning".

Routing is determined by ``VLLM_MODEL``:
- If it matches a known OpenRouter model id (see ``OPENROUTER_MODELS``), the
  request is sent to ``https://openrouter.ai/api/v1`` using the
  ``OPENROUTER_API_KEY`` env var (loaded from ``.env`` at the repo root).
- Otherwise, the request is sent to the local vLLM server at
  ``VLLM_BASE_URL:VLLM_PORT``.
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import requests
from openai import OpenAI
from dotenv import load_dotenv
from src.config import get_env_value
load_dotenv()


# ----- Hyperparameters --------------------------------------------------------
VLLM_BASE_URL = "localhost"
VLLM_PORT = 7472
VLLM_MODEL = "openai/gpt-oss-20b"
VLLM_API_KEY = "EMPTY"

CORPUS_BASE_URL = "localhost"
CORPUS_PORT = 7470

TOP_N = 5
REQUEST_TIMEOUT = 30
TEMPERATURE = 0.2
MAX_TOKENS = 1024
# -----------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Models routed through OpenRouter rather than the local vLLM server.
OPENROUTER_MODELS = {
    "nvidia/nemotron-3-super-120b-a12b:free",
}

SUPPORTED_MODELS = [
    "Qwen/Qwen3-4B", "google/gemma-4-E2B-it", "Qwen/Qwen3-14B", "google/gemma-4-31B-it",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "openai/gpt-oss-20b",
    *OPENROUTER_MODELS,
]


def search_info(query: str) -> str:
    """Call the corpus server's /query_search and format the top hits."""
    url = f"http://{CORPUS_BASE_URL}:{CORPUS_PORT}/query_search"
    resp = requests.post(
        url,
        json={"query": query, "n": TOP_N},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    if not hits:
        return f"No results found for: {query}"
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] {hit.get('title', '').strip()}")
        lines.append(f"    URL: {hit.get('url', '').strip()}")
        snippet = hit.get("snippet", "").strip()
        if snippet:
            lines.append(f"    Snippet: {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


SEARCH_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "search_info",
        "description": (
            "Search the knowledge corpus and return the top hits as a numbered "
            "list of title/URL/snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query.",
                },
            },
            "required": ["query"],
        },
    },
}


def _build_client(model: str) -> tuple[OpenAI, str]:
    """Return (client, endpoint_description) based on the chosen model."""
    if model in OPENROUTER_MODELS:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        return client, OPENROUTER_BASE_URL
    base_url = f"http://{VLLM_BASE_URL}:{VLLM_PORT}/v1"
    client = OpenAI(base_url=base_url, api_key=VLLM_API_KEY)
    return client, base_url


def main() -> None:
    client, endpoint = _build_client(VLLM_MODEL)

    system_prompt = (
        "You are a research assistant. When the user asks for relevant "
        "articles on a topic, call the `search_info` tool to retrieve them."
    )
    user_prompt = (
        "Find the most relevant articles to the query \"machine learning\" "
        "by calling the search_info tool."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"=> Pinging {endpoint} (model={VLLM_MODEL})")
    completion = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        tools=[SEARCH_INFO_TOOL],
        tool_choice="auto",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    msg = completion.choices[0].message
    print(msg)
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        print("Model did not produce a tool call.")
        print("Assistant content:", msg.content)
        return

    for call in tool_calls:
        print(f"\n=> Tool call: {call.function.name}({call.function.arguments})")
        if call.function.name != "search_info":
            print(f"   Unexpected tool: {call.function.name}")
            continue
        args = json.loads(call.function.arguments or "{}")
        result = search_info(args.get("query", ""))
        print("\n=> Tool result:")
        print(result)


if __name__ == "__main__":
    if VLLM_MODEL not in SUPPORTED_MODELS:
        print("Model not supported ... exiting")
        sys.exit(0)
    main()
