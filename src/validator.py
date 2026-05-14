"""Schema validation — grammar enforcement for generated/mutated schemas."""

from __future__ import annotations

from collections import Counter

from src.types import DebateMode, PersonaType, Schema, SynthesisMethod
from src.logging import get_logger

logger = get_logger(__name__)

ALL_PERSONAS = set(PersonaType)
PERSONA_MODE_PERSONAS = {
    PersonaType.ANALYST,
    PersonaType.CRITIC,
    PersonaType.SYNTHESIZER,
}

# Persona-mode rounds may repeat ANALYST and CRITIC up to 2x each to
# amplify those voices; SYNTHESIZER stays unique within a round.
PERSONA_MODE_MAX_PER_ROUND: dict[PersonaType, int] = {
    PersonaType.ANALYST: 2,
    PersonaType.CRITIC: 2,
    PersonaType.SYNTHESIZER: 1,
}


class SchemaValidationError(Exception):
    """Raised when a schema fails validation."""


def validate_schema(schema: Schema) -> list[str]:
    """Validate a schema and return a list of error messages. Empty list = valid."""
    errors: list[str] = []

    if len(schema.rounds) == 0:
        errors.append("Schema must have at least 1 round.")

    if schema.max_rounds < len(schema.rounds):
        errors.append(
            f"max_rounds ({schema.max_rounds}) < actual rounds ({len(schema.rounds)})."
        )

    for i, rnd in enumerate(schema.rounds):
        if len(rnd.personas) == 0:
            errors.append(f"Round {i + 1} has no personas.")

        for p in rnd.personas:
            if p not in ALL_PERSONAS:
                errors.append(f"Round {i + 1} references unknown persona: {p}.")

        if schema.mode == DebateMode.PERSONA:
            # Persona mode: only analyst/critic/synthesizer; n_samples must be 1.
            # Analyst and critic may each appear up to 2x; synthesizer 1x.
            persona_counts = Counter(rnd.personas)
            for p, count in persona_counts.items():
                max_allowed = PERSONA_MODE_MAX_PER_ROUND.get(p)
                if max_allowed is not None and count > max_allowed:
                    errors.append(
                        f"Round {i + 1}: persona {p.value!r} appears {count} "
                        f"times (max {max_allowed} in persona mode)."
                    )
            for p in rnd.personas:
                if p == PersonaType.GENERIC_ASSISTANT:
                    errors.append(
                        f"Round {i + 1}: generic_assistant is only valid in "
                        f"self_consistency mode."
                    )
            if rnd.n_samples != 1:
                errors.append(
                    f"Round {i + 1}: n_samples={rnd.n_samples} is only valid "
                    f"in self_consistency mode (must be 1 in persona mode)."
                )

        else:  # DebateMode.SELF_CONSISTENCY
            # Self-consistency mode: a round must use a single persona which is
            # either generic_assistant (sampling round) or synthesizer
            # (final aggregation round). Stop conditions are not allowed
            # (Option C: no early stop).
            if len(rnd.personas) != 1:
                errors.append(
                    f"Round {i + 1}: self_consistency rounds must have exactly "
                    f"one persona (got {len(rnd.personas)})."
                )
            elif rnd.personas[0] not in (
                PersonaType.GENERIC_ASSISTANT, PersonaType.SYNTHESIZER
            ):
                errors.append(
                    f"Round {i + 1}: self_consistency persona must be "
                    f"generic_assistant or synthesizer (got {rnd.personas[0].value})."
                )
            if rnd.personas == [PersonaType.SYNTHESIZER] and rnd.n_samples != 1:
                errors.append(
                    f"Round {i + 1}: aggregation rounds (synthesizer) must "
                    f"have n_samples=1."
                )

    if (
        schema.final_synthesis == SynthesisMethod.SYNTHESIZER_PERSONA
        and schema.rounds
    ):
        last_round = schema.rounds[-1]
        if PersonaType.SYNTHESIZER not in last_round.personas:
            errors.append(
                "final_synthesis is 'synthesizer_persona' but synthesizer "
                "is not in the last round."
            )

    return errors


def is_valid_schema(schema: Schema) -> bool:
    """Check if a schema is valid."""
    return len(validate_schema(schema)) == 0


def validate_or_raise(schema: Schema) -> None:
    """Validate a schema, raising SchemaValidationError if invalid."""
    errors = validate_schema(schema)
    if errors:
        msg = "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.warning("schema_validation_failed", errors=errors)
        raise SchemaValidationError(msg)
