"""Unit tests for LLM client helper logic."""

from __future__ import annotations

from src.llm_client import (
    LLMConfig,
    _parse_tool_arguments,
    _resolve_tool_query,
)


def test_parse_tool_arguments_json_string() -> None:
    args = _parse_tool_arguments('{"query": "capital of france"}')
    assert args["query"] == "capital of france"


def test_parse_tool_arguments_python_literal() -> None:
    args = _parse_tool_arguments("{'url': 'https://example.com'}")
    assert args["url"] == "https://example.com"


def test_parse_tool_arguments_fallback_raw_string() -> None:
    args = _parse_tool_arguments("plain search query")
    assert args["query"] == "plain search query"


def test_resolve_tool_query_with_aliases() -> None:
    fetch_query = _resolve_tool_query("fetch_url", {"url": "https://example.com"})
    code_query = _resolve_tool_query("code_compute", {"expression": "2 + 2"})

    assert fetch_query == "https://example.com"
    assert code_query == "2 + 2"


def test_llm_config_from_dict_uses_provided_model() -> None:
    cfg = LLMConfig.from_dict(
        {
            "provider": "vllm",
            "model": "google/gemma-4-26B-A4B",
            "ip": "localhost",
            "port": 6667,
        }
    )
    assert cfg.provider == "vllm"
    assert cfg.model == "google/gemma-4-26B-A4B"
    assert cfg.port == 6667
