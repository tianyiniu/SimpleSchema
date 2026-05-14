"""Integration test for the dual judge against a hardcoded execution trace.

Uses the ``gaia_3_001_gen0_seed_template_moderate.json`` fixture in the
repo root — a GAIA question about USDA 1959 dehydrated-product
standards. The ground truth is ``86``, the personas converge on ``95%``
(and one says "cannot be determined"), and no tools are used at all
despite the question demanding specific historical fact-finding.

A correctly-calibrated judge should:
  1. Mark the answer ``INCORRECT`` (95 contradicts the gold value of 86)
     -> ``answer_score == 0.0``.
  2. Give a low-to-mid trace score: no tool use, speculative reasoning,
     fabricated USDA reports, and inconsistent answers across personas.

The test makes a live OpenRouter call against ``[llm.judge]`` from
``config.toml``. It is skipped automatically if ``OPENROUTER_API_KEY``
is not available in the environment or in the repo ``.env`` file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import get_env_value, load_config
from src.judge import judge_dual
from src.llm_client import LLMClient, LLMConfig


GAIA_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "gaia_3_001_gen0_seed_template_moderate.json"
)


def _load_gaia_fixture() -> dict:
    with open(GAIA_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _trace_to_all_responses(
    trace: list[dict],
) -> list[list[dict[str, str]]]:
    """Re-bucket flat ``persona_call`` entries into per-round response lists.

    The judge expects ``all_responses[r] = [{persona, response}, ...]``;
    the gaia fixture stores them flat with a ``round`` field.
    """
    by_round: dict[int, list[dict[str, str]]] = {}
    for entry in trace:
        if entry.get("type") != "persona_call":
            continue
        round_num = int(entry["round"])
        by_round.setdefault(round_num, []).append(
            {"persona": entry["persona"], "response": entry["response"]}
        )
    return [by_round[r] for r in sorted(by_round)]


def _judge_client_or_skip() -> LLMClient:
    """Build the live judge client; skip if its backend isn't reachable."""
    cfg = load_config()
    judge_section = cfg.get("llm", {}).get("judge")
    if not judge_section:
        pytest.skip("[llm.judge] not configured in config.toml")
    shared = cfg.get("llm", {}).get("shared", {})
    merged = {**shared, **judge_section}
    provider = merged.get("provider", "")
    if provider in ("openai", "openrouter"):
        api_key_env = merged.get("api_key_env", "")
        if not api_key_env or not get_env_value(api_key_env):
            pytest.skip(
                f"judge API key env var {api_key_env!r} not set; "
                "judge-quality test requires a live model"
            )
    elif provider == "vllm":
        # Self-hosted: probe the configured host:port. Skip rather than
        # error so CI without the server up doesn't fail this test.
        import socket
        host = merged.get("ip", "localhost")
        port = int(merged.get("port", 0))
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except OSError:
            pytest.skip(
                f"judge vLLM server at {host}:{port} not reachable; "
                "start it with scripts/deploy_model.sh -m nemotron-super-120b"
            )
    return LLMClient(LLMConfig.from_dict(merged))


def test_judge_marks_incorrect_speculative_trace() -> None:
    """Live judge call: gaia_3_001 should grade INCORRECT with a low trace score."""
    fixture = _load_gaia_fixture()
    assert fixture["ground_truth"] == "86"
    assert "95" in fixture["predicted_answer"]
    assert len(fixture["trace"]) == 4

    judge_client = _judge_client_or_skip()
    try:
        judge_log: list[dict] = []
        answer_score, trace_score = judge_dual(
            question=fixture["question"],
            ground_truth=fixture["ground_truth"],
            predicted_answer=fixture["predicted_answer"],
            judge_client=judge_client,
            all_responses=_trace_to_all_responses(fixture["trace"]),
            execution_trace=fixture["trace"],  # contains no tool_call entries
            judge_log=judge_log,
        )
    finally:
        judge_client.shutdown()

    # Diagnostic — printed only on failure when pytest captures stdout.
    method = judge_log[-1].get("method") if judge_log else None
    label = judge_log[-1].get("answer_label") if judge_log else None
    print(
        f"\n[judge-quality] method={method} label={label} "
        f"answer_score={answer_score} trace_score={trace_score}"
    )

    # Distinguish a real judge verdict from a judge call that never
    # reached the model. judge_dual returns (0.0, 0.0) on API error,
    # which would otherwise satisfy the assertions below.
    assert method in ("dual_judge", "dual_judge_followup"), (
        f"judge call did not reach the model (method={method!r}). Check "
        f"that the [llm.judge] backend (vLLM server or API key) is reachable."
    )

    # 95 contradicts the gold target 86; SimpleQA-style grader must mark
    # this INCORRECT (B), which collapses to answer_score == 0.0.
    assert answer_score == 0.0, (
        f"expected INCORRECT (answer_score=0.0); got {answer_score}. "
        f"label={label}"
    )
    assert label == "INCORRECT", (
        f"expected answer_label='INCORRECT'; got {label!r}"
    )

    # Trace shows no tool use, speculative reasoning, fabricated "USDA
    # 2020 AMS report" citations, and inconsistent persona answers
    # (30 / 100 / 95 / cannot-be-determined). A well-calibrated judge
    # should give <= 3.5. The range allows some judge-side variance.
    assert 0.0 <= trace_score <= 3.5, (
        f"trace_score {trace_score} outside expected [0.0, 3.5] for a "
        f"speculative no-tool trace"
    )
