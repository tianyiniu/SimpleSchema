"""Ping a vLLM server (or OpenRouter) and verify its reasoning parser.

Sends a prompt that should trigger chain-of-thought and inspects the
response for a separated ``reasoning_content`` field. When vLLM is started
with ``--reasoning-parser <name>``, the model's hidden thinking is routed
into ``message.reasoning_content`` while the user-facing answer stays in
``message.content``. This script confirms that split is happening for the
configured model.

Routing follows the same convention as ping_model.py:
- If ``VLLM_MODEL`` is in ``OPENROUTER_MODELS``, the request is sent to
  ``https://openrouter.ai/api/v1`` using ``OPENROUTER_API_KEY``.
- Otherwise, it hits the local vLLM server at ``VLLM_BASE_URL:VLLM_PORT``.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from openai import OpenAI
from dotenv import load_dotenv
from src.config import get_env_value
load_dotenv()


# ----- Hyperparameters --------------------------------------------------------
VLLM_BASE_URL = "localhost"
VLLM_PORT = 7471
VLLM_MODEL = "google/gemma-4-E2B-it"
VLLM_API_KEY = "EMPTY"

TEMPERATURE = 0.7
MAX_TOKENS = 4096
# -----------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Models routed through OpenRouter rather than the local vLLM server.
OPENROUTER_MODELS = {
    "nvidia/nemotron-3-super-120b-a12b:free",
}

SUPPORTED_MODELS = [
    "Qwen/Qwen3-4B", "google/gemma-4-E2B-it", "Qwen/Qwen3-14B", "google/gemma-4-31B-it",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "openai/gpt-oss-20b",
    *OPENROUTER_MODELS,
]


def _build_client(model: str) -> tuple[OpenAI, str]:
    """Return (client, endpoint_description) based on the chosen model."""
    if model in OPENROUTER_MODELS:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        return client, OPENROUTER_BASE_URL
    base_url = f"http://{VLLM_BASE_URL}:{VLLM_PORT}/v1"
    client = OpenAI(base_url=base_url, api_key=VLLM_API_KEY)
    return client, base_url


def _reasoning_content(msg) -> str | None:
    """Pull reasoning_content off the OpenAI message, wherever the SDK put it.

    Older OpenAI SDK builds expose unknown server fields via ``model_extra``;
    newer builds attach ``reasoning_content`` directly.
    """
    direct = getattr(msg, "reasoning_content", None)
    if direct:
        return direct
    extra = getattr(msg, "model_extra", None) or {}
    return extra.get("reasoning_content")


def main() -> None:
    client, endpoint = _build_client(VLLM_MODEL)

    system_prompt = (
        "You are a careful problem solver. Think step by step before "
        "producing the final answer."
    )
    user_prompt = (
        "A train leaves station A at 60 km/h. Two hours later, a second "
        "train leaves station A on the same track at 90 km/h. How many "
        "hours after the second train departs does it catch up to the "
        "first? Reply with the final answer as a single number."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    extra_body: dict = {}

    print(f"=> Pinging {endpoint} (model={VLLM_MODEL})")
    completion = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body=extra_body or None,
    )
    msg = completion.choices[0].message
    reasoning = _reasoning_content(msg)
    content = msg.content or ""

    print("\n=> reasoning_content (separated by parser):")
    if reasoning:
        print(reasoning)
    else:
        print("<none — parser did not separate any reasoning>")

    print("\n=> content (final-answer channel):")
    print(content)

    # Heuristic: if the parser is working, raw reasoning delimiters should
    # not appear in content. <think> covers qwen3 / deepseek_r1 / nano_v3;
    # <|channel|>analysis covers gpt-oss harmony.
    has_inline_markers = "<think>" in content or "<|channel|>analysis" in content
    print("\n=> Verdict:")
    if reasoning and not has_inline_markers:
        print("OK: reasoning_content populated and final content is clean.")
    elif reasoning and has_inline_markers:
        print("PARTIAL: reasoning_content set but raw markers still leaked into content.")
    elif has_inline_markers:
        print("FAIL: reasoning markers remain in content — parser likely not active.")
    else:
        print(
            "INCONCLUSIVE: no reasoning_content and no inline markers. The "
            "model may not have emitted reasoning for this prompt; try a "
            "harder question or check that the server was started with "
            "--reasoning-parser."
        )


if __name__ == "__main__":
    if VLLM_MODEL not in SUPPORTED_MODELS:
        print("Model not supported ... exiting")
        sys.exit(0)
    main()
