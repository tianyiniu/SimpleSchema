"""Tests for schema types and validation."""

import pytest

from src.types import (
    InstructionType,
    PersonaType,
    Round,
    Schema,
    SynthesisMethod,
    ToolType,
)
from src.validator import SchemaValidationError, is_valid_schema, validate_or_raise, validate_schema


class TestSchemaCreation:
    def test_minimal_schema(self):
        schema = Schema(
            max_rounds=1,
            final_synthesis=SynthesisMethod.LAST_PERSONA,
            rounds=[
                Round(
                    round=1,
                    personas=[PersonaType.ANALYST],
                    tools=[],
                    instruction=InstructionType.PRODUCE_FINAL_ANSWER,
                )
            ],
        )
        assert len(schema.rounds) == 1
        assert schema.max_rounds == 1

    def test_full_schema(self):
        schema = Schema(
            max_rounds=4,
            final_synthesis=SynthesisMethod.MAJORITY_VOTE,
            rounds=[
                Round(
                    round=1,
                    personas=[PersonaType.ANALYST, PersonaType.CRITIC],
                    tools=[ToolType.SEARCH_INFO],
                    instruction=InstructionType.INDEPENDENTLY_RESEARCH,
                ),
                Round(
                    round=2,
                    personas=[PersonaType.ANALYST, PersonaType.CRITIC, PersonaType.SYNTHESIZER],
                    tools=[ToolType.SEARCH_INFO, ToolType.CODE_COMPUTE],
                    instruction=InstructionType.DEBATE_AND_REFINE,
                ),
                Round(
                    round=3,
                    personas=[PersonaType.SYNTHESIZER],
                    tools=[],
                    instruction=InstructionType.PRODUCE_FINAL_ANSWER,
                ),
            ],
        )
        assert len(schema.rounds) == 3

    def test_round_requires_at_least_one_persona(self):
        with pytest.raises(Exception):
            Round(
                round=1,
                personas=[],
                tools=[],
                instruction=InstructionType.PRODUCE_FINAL_ANSWER,
            )


class TestSchemaValidation:
    def test_valid_simple_schema(self):
        schema = Schema(
            max_rounds=1,
            final_synthesis=SynthesisMethod.LAST_PERSONA,
            rounds=[
                Round(
                    round=1,
                    personas=[PersonaType.ANALYST],
                    tools=[],
                    instruction=InstructionType.PRODUCE_FINAL_ANSWER,
                )
            ],
        )
        assert is_valid_schema(schema)

    def test_max_rounds_less_than_actual(self):
        schema = Schema(
            max_rounds=1,
            final_synthesis=SynthesisMethod.LAST_PERSONA,
            rounds=[
                Round(round=1, personas=[PersonaType.ANALYST], tools=[], instruction=InstructionType.GATHER_FACTS),
                Round(round=2, personas=[PersonaType.CRITIC], tools=[], instruction=InstructionType.PRODUCE_FINAL_ANSWER),
            ],
        )
        errors = validate_schema(schema)
        assert any("max_rounds" in e for e in errors)

    def test_synthesizer_persona_synthesis_requires_synthesizer_in_last_round(self):
        schema = Schema(
            max_rounds=1,
            final_synthesis=SynthesisMethod.SYNTHESIZER_PERSONA,
            rounds=[
                Round(
                    round=1,
                    personas=[PersonaType.ANALYST],
                    tools=[],
                    instruction=InstructionType.PRODUCE_FINAL_ANSWER,
                )
            ],
        )
        errors = validate_schema(schema)
        assert any("synthesizer" in e.lower() for e in errors)

    def test_validate_or_raise_valid(self):
        schema = Schema(
            max_rounds=1,
            final_synthesis=SynthesisMethod.LAST_PERSONA,
            rounds=[
                Round(round=1, personas=[PersonaType.ANALYST], tools=[], instruction=InstructionType.PRODUCE_FINAL_ANSWER)
            ],
        )
        validate_or_raise(schema)  # Should not raise

    def test_validate_or_raise_invalid(self):
        schema = Schema(
            max_rounds=1,
            final_synthesis=SynthesisMethod.SYNTHESIZER_PERSONA,
            rounds=[
                Round(round=1, personas=[PersonaType.ANALYST], tools=[], instruction=InstructionType.PRODUCE_FINAL_ANSWER)
            ],
        )
        with pytest.raises(SchemaValidationError):
            validate_or_raise(schema)
