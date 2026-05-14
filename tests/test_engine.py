"""Tests for the execution engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.engine import ExecutionResult, _extract_answer_from_response, execute_schema
from src.types import (
    Task,
    InstructionType,
    PersonaType,
    Round,
    Schema,
    SynthesisMethod,
)
from src.tools import create_tool_registry


def _simple_schema() -> Schema:
    return Schema(
        max_rounds=1,
        final_synthesis=SynthesisMethod.LAST_PERSONA,
        rounds=[
            Round(
                round=1,
                personas=[PersonaType.ANALYST],
                tools=[],
                instruction=InstructionType.PRODUCE_FINAL_ANSWER,
            ),
        ],
    )


def _full_schema() -> Schema:
    return Schema(
        max_rounds=5,
        final_synthesis=SynthesisMethod.SYNTHESIZER_PERSONA,
        rounds=[
            Round(
                round=1,
                personas=[PersonaType.ANALYST, PersonaType.CRITIC, PersonaType.SYNTHESIZER],
                tools=[],
                instruction=InstructionType.THINK_AND_PLAN,
            ),
            Round(
                round=2,
                personas=[PersonaType.ANALYST, PersonaType.CRITIC],
                tools=[],
                instruction=InstructionType.INDEPENDENTLY_RESEARCH,
            ),
            Round(
                round=3,
                personas=[PersonaType.ANALYST, PersonaType.CRITIC, PersonaType.SYNTHESIZER],
                tools=[],
                instruction=InstructionType.DEBATE_AND_REFINE,
            ),
            Round(
                round=4,
                personas=[PersonaType.SYNTHESIZER],
                tools=[],
                instruction=InstructionType.PRODUCE_FINAL_ANSWER,
            ),
        ],
    )


class TestAnswerExtraction:
    def test_extract_with_answer_marker(self):
        response = "After analysis, the answer is clear.\nANSWER: 42"
        assert _extract_answer_from_response(response) == "42"

    def test_extract_without_marker(self):
        response = "The result is 42"
        assert _extract_answer_from_response(response) == "The result is 42"

    def test_extract_multiline_takes_first(self):
        response = "Some reasoning.\nANSWER: Paris\nMore text"
        assert _extract_answer_from_response(response) == "Paris"

    def test_extract_empty_response(self):
        assert _extract_answer_from_response("") == ""


class TestSchemaExecution:
    def test_execute_simple_schema(self):
        """Test execution with a mocked LLM client."""
        schema = _simple_schema()
        task = Task(
            id="test_001",
            level=1,
            question="What is 2+2?",
            ground_truth="4",
            answer_type="exact",
        )

        mock_client = MagicMock()
        mock_client.max_tokens = 16384
        mock_client.complete_batch.return_value = ["After calculation:\nANSWER: 4"]

        tool_registry = create_tool_registry()

        result = execute_schema(schema, task, mock_client, tool_registry)

        assert isinstance(result, ExecutionResult)
        assert result.num_rounds_executed == 1
        assert result.predicted_answer == "4"
        mock_client.complete_batch.assert_called_once()

    def test_execute_multi_round_schema(self):
        """Test multi-round execution with mocked LLM."""
        schema = _full_schema()
        task = Task(
            id="test_002",
            level=2,
            question="What is the capital of France?",
            ground_truth="Paris",
            answer_type="exact",
        )

        mock_client = MagicMock()
        mock_client.max_tokens = 16384
        # generate_full_seed now has 4 rounds: plan, research, debate, finalize.
        mock_client.complete_batch.side_effect = [
            # Round 1: think_and_plan (analyst + critic + synthesizer)
            ["Plan: confirm capital via search.",
             "Plan critique: verify with multiple sources.",
             "Plan synthesis: gather and confirm."],
            # Round 2: independently_research (analyst + critic)
            ["Research: Paris is the capital.\nANSWER: Paris",
             "Checking: Paris is indeed correct.\nANSWER: Paris"],
            # Round 3: debate_and_refine (all three)
            ["Paris is correct.\nANSWER: Paris",
             "Confirmed.\nANSWER: Paris",
             "Synthesizing: Paris.\nANSWER: Paris"],
            # Round 4: produce_final_answer (synthesizer)
            ["Final: Paris.\nANSWER: Paris"],
        ]

        tool_registry = create_tool_registry()

        result = execute_schema(schema, task, mock_client, tool_registry)

        assert isinstance(result, ExecutionResult)
        assert result.predicted_answer == "Paris"
