"""Agent-proposed schema batches for the one-shot reward-model dataset.

This is a slimmed rewrite of the main repo's generator. The fixed
``simple``/``moderate``/``full`` seed templates and all evolution
infrastructure are removed. There is one entry point,
``propose_schemas_batch``, which asks the **agent** LLM to emit a JSON
array of ``n`` debate schemas in a single call. The prompt actively
pushes for diversity along multiple axes (round count, persona mix,
tool selection, instructions, synthesis method) so a single call can
return structurally distinct candidates.
"""

from __future__ import annotations

import json
import random as _random
import re

from src.types import (
    DebateMode,
    InstructionType,
    PersonaType,
    Round,
    Schema,
    SynthesisMethod,
)
from src.validator import is_valid_schema
from src.tools import active_tool_names
from src.logging import get_logger

logger = get_logger(__name__)


DEBATE_MODE_POLICIES = ("persona", "self_consistency", "mixed")


def _validate_policy(policy: str) -> str:
    if policy not in DEBATE_MODE_POLICIES:
        raise ValueError(
            f"unknown debate_mode policy {policy!r}; "
            f"expected one of {DEBATE_MODE_POLICIES}"
        )
    return policy


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_TOOL_SEMANTICS: dict[str, str] = {
    "search_info": (
        "search_info: searches the corpus and returns the top hits as "
        "title/url/snippet. Pair with fetch_url to read a specific document."
    ),
    "fetch_url": (
        "fetch_url: downloads the full text of a specific URL — typically a "
        "URL surfaced by search_info."
    ),
    "code_compute": (
        "code_compute: runs a Python expression or full script and returns "
        "stdout. Use for arithmetic, unit conversions, string ops, etc."
    ),
}


def _mode_clause(debate_mode: str) -> str:
    if debate_mode == "persona":
        return (
            '- "mode": MUST be "persona". The schema runs a debate among '
            'analyst/critic/synthesizer personas. Do NOT emit '
            '"self_consistency".'
        )
    if debate_mode == "self_consistency":
        return (
            '- "mode": MUST be "self_consistency". The schema runs N '
            'parallel samples of generic_assistant and aggregates them. '
            'Do NOT emit "persona".'
        )
    return (
        '- "mode": "persona" or "self_consistency". Persona mode runs a '
        'debate among analyst/critic/synthesizer; self_consistency runs N '
        'parallel samples of a generic_assistant and aggregates them.'
    )


def _build_batch_system_prompt(n: int, debate_mode: str) -> str:
    """System prompt that asks for ``n`` distinct schemas in one response."""
    active = active_tool_names()
    tool_list_str = ", ".join(f'"{t}"' for t in active)
    semantics_lines = [
        f"  - {_TOOL_SEMANTICS[t]}" for t in active if t in _TOOL_SEMANTICS
    ]
    semantics_block = (
        "- Tool semantics:\n" + "\n".join(semantics_lines) + "\n"
        if semantics_lines
        else ""
    )

    return f"""\
You are a debate schema designer. Your job is to produce {n} *structurally \
distinct* JSON debate schemas that could plausibly answer the user's \
question. The downstream system will execute every one of the {n} schemas \
you emit, so diversity within this single batch is critical.

DIVERSITY REQUIREMENTS (the {n} schemas must spread across these axes):
- Round count: at least one short schema (1-2 rounds) AND at least one \
longer schema (3-5 rounds).
- Persona mix: vary which personas appear, how many appear per round, and \
whether any are repeated. Do NOT submit {n} schemas that all use the same \
persona shape.
- Tool selection: vary which tools each round uses (including 0-tool \
rounds where appropriate). Avoid emitting {n} schemas that all enable the \
same tool set.
- Instructions: vary the per-round instruction across schemas — different \
schemas should emphasize different cognitive moves (planning, research, \
critique, verification, synthesis).
- Synthesis method: do not pick the same final_synthesis for every schema; \
spread across "majority_vote", "last_persona", and "synthesizer_persona" \
where the persona shape allows it.

A schema has:
{_mode_clause(debate_mode)}
- "max_rounds": integer (1-6).
- "final_synthesis": one of "majority_vote", "last_persona", "synthesizer_persona".
- "rounds": a list of round objects.

Each round has:
- "round": integer starting from 1.
- "personas": list of personas. In persona mode pick 1-5 from \
["analyst", "critic", "synthesizer"] — analyst and critic MAY each appear \
up to 2 times in the same round to amplify that voice (e.g. \
["analyst", "analyst", "critic"]); synthesizer appears at most once per \
round. In self_consistency mode use exactly one persona: \
"generic_assistant" for sampling rounds, or "synthesizer" for a final \
aggregation round.
- "n_samples": integer 1-8. In persona mode this is always 1. In \
self_consistency mode use 2-8 for sampling rounds and 1 for synthesizer \
aggregation rounds.
- "tools": list of 0+ tools from [{tool_list_str}].
- "instruction": one of "independently_research", "debate_and_refine", \
"critique_previous", "gather_facts", "produce_final_answer", \
"verify_and_check", "brainstorm", "think_and_plan".

Rules:
- At least 1 round per schema.
- Each round must have at least 1 persona.
- In persona mode, analyst and critic counts must each be <= 2 per round, \
and synthesizer count must be <= 1.
- 1-5 rounds per schema is typical.
- All defined rounds run in order — there is no early stopping mechanism, \
so do NOT emit a "stop_condition" field.
{semantics_block}
Output ONLY a JSON array of exactly {n} schema objects. No prose, no \
markdown fences, no commentary. /no_think"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list[dict] | None:
    """Find a JSON array (or single object) in the model's output.

    Strips an optional ```json fence, then takes the substring from the
    first ``[`` or ``{`` to its matching last bracket and tries to parse.
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1).strip() if fence else text.strip()
    # Trim to the outermost array/object.
    start = next((i for i, ch in enumerate(candidate) if ch in "[{"), -1)
    if start < 0:
        return None
    end_char = "]" if candidate[start] == "[" else "}"
    end = candidate.rfind(end_char)
    if end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return None


def _coerce_schemas(
    raw_items: list[dict],
    debate_mode: str,
) -> list[Schema]:
    """Validate each raw dict as a Schema; skip invalid ones with a log."""
    out: list[Schema] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            logger.warning("schema_batch_item_not_object", idx=idx)
            continue
        try:
            schema = Schema.model_validate(item)
        except Exception:
            logger.warning("schema_batch_validate_failed", idx=idx, exc_info=True)
            continue
        if not is_valid_schema(schema):
            logger.warning("schema_batch_grammar_invalid", idx=idx)
            continue
        if debate_mode == "persona" and schema.mode != DebateMode.PERSONA:
            logger.warning(
                "schema_batch_mode_policy_violation",
                idx=idx, policy=debate_mode, produced=schema.mode.value,
            )
            continue
        if (
            debate_mode == "self_consistency"
            and schema.mode != DebateMode.SELF_CONSISTENCY
        ):
            logger.warning(
                "schema_batch_mode_policy_violation",
                idx=idx, policy=debate_mode, produced=schema.mode.value,
            )
            continue
        out.append(schema)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def propose_schemas_batch(
    question: str,
    agent_client: object,
    n: int = 4,
    rng: _random.Random | None = None,
    max_attempts: int = 3,
    debate_mode: str = "mixed",
    temperature: float = 0.9,
) -> list[Schema]:
    """Ask the agent for a batch of ``n`` debate schemas in a single call.

    Args:
        question: The benchmark question to design schemas for.
        agent_client: The agent ``LLMClient`` (same client used for execution).
        n: How many schemas to request in this single call.
        rng: RNG reserved for future stochastic prompting (unused).
        max_attempts: Number of retries if the call returns invalid JSON or
            fewer than ``n`` validated schemas.
        debate_mode: One of ``persona`` / ``self_consistency`` / ``mixed``.
        temperature: Sampling temperature for this proposal call. Defaults
            high (0.9) to encourage diversity within a single response.

    Returns:
        A list of validated ``Schema`` objects. May contain fewer than ``n``
        if some entries failed validation and retries didn't recover the
        difference.
    """
    _ = rng
    debate_mode = _validate_policy(debate_mode)
    system_prompt = _build_batch_system_prompt(n=n, debate_mode=debate_mode)

    user_prompt = (
        f"Design {n} structurally distinct debate schemas to answer the "
        f"following question. Make the {n} schemas differ from each other "
        f"along round count, persona mix, tool selection, instructions, and "
        f"synthesis method — do not emit {n} near-copies.\n\nQuestion:\n{question}"
    )

    best: list[Schema] = []
    for attempt in range(max_attempts):
        try:
            raw = agent_client.complete(  # type: ignore[attr-defined]
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=4096,
                enable_thinking=False,
            )
        except Exception:
            logger.warning(
                "schema_batch_call_failed", attempt=attempt + 1, exc_info=True,
            )
            continue

        items = _extract_json_array(raw)
        if items is None:
            logger.warning(
                "schema_batch_json_parse_failed",
                attempt=attempt + 1,
                raw_preview=raw[:300],
            )
            continue

        schemas = _coerce_schemas(items, debate_mode=debate_mode)
        if len(schemas) > len(best):
            best = schemas
        if len(schemas) >= n:
            logger.info(
                "schema_batch_generated",
                attempt=attempt + 1, count=len(schemas), requested=n,
            )
            return schemas[:n]

        logger.warning(
            "schema_batch_incomplete",
            attempt=attempt + 1,
            count=len(schemas), requested=n,
        )

    if not best:
        logger.warning("schema_batch_all_attempts_failed", requested=n)
    else:
        logger.warning(
            "schema_batch_returning_partial",
            count=len(best), requested=n,
        )
    return best


# Re-export type names so existing call sites that imported them from this
# module continue to work.
__all__ = [
    "DEBATE_MODE_POLICIES",
    "DebateMode",
    "InstructionType",
    "PersonaType",
    "Round",
    "Schema",
    "SynthesisMethod",
    "propose_schemas_batch",
]
