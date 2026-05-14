"""Schema executor — runs a schema against a task."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.types import Task, Schema, SynthesisMethod
from src.debate import run_debate_round
from src.tools import ToolCallCache, ToolRegistry
from src.llm_client import LLMClient
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a schema against a task."""

    predicted_answer: str
    all_responses: list[list[dict[str, str]]] = field(default_factory=list)
    num_rounds_executed: int = 0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    execution_trace: list[dict] = field(default_factory=list)


def _extract_final_answer(
    schema: Schema,
    all_responses: list[list[dict[str, str]]],
) -> str:
    """Extract the final answer based on the schema's synthesis method."""
    if not all_responses:
        return ""

    last_round = all_responses[-1]

    if schema.final_synthesis == SynthesisMethod.LAST_PERSONA:
        if last_round:
            return _extract_answer_from_response(last_round[-1]["response"])
        return ""

    elif schema.final_synthesis == SynthesisMethod.SYNTHESIZER_PERSONA:
        for resp in reversed(last_round):
            if resp["persona"] == "synthesizer":
                return _extract_answer_from_response(resp["response"])
        # Fallback to last response
        if last_round:
            return _extract_answer_from_response(last_round[-1]["response"])
        return ""

    elif schema.final_synthesis == SynthesisMethod.MAJORITY_VOTE:
        # Collect answers from all personas in last round
        answers: list[str] = []
        for resp in last_round:
            answer = _extract_answer_from_response(resp["response"])
            if answer:
                answers.append(answer)

        if not answers:
            # Fallback: use last response
            if last_round:
                return _extract_answer_from_response(last_round[-1]["response"])
            return ""

        # Simple majority vote
        from collections import Counter
        counts = Counter(answers)
        return counts.most_common(1)[0][0]

    return ""


def _extract_answer_from_response(response: str) -> str:
    """Extract the answer from a response, looking for ANSWER: marker."""
    upper = response.upper()
    if "ANSWER:" in upper:
        idx = upper.rfind("ANSWER:")
        answer = response[idx + 7:].strip()
        # Take first line
        answer = answer.split("\n")[0].strip()
        return answer

    # Fallback: return last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else response.strip()


def execute_schema(
    schema: Schema,
    task: Task,
    agent_client: LLMClient,
    tool_registry: ToolRegistry,
    tool_cache: ToolCallCache | None = None,
) -> ExecutionResult:
    """Execute a debate schema against a benchmark task.

    Runs each round sequentially, passing previous responses forward.
    Handles early stopping via stop conditions.

    ``tool_cache`` lets callers share tool results across schemas for the
    same task (one_shot driver passes a per-task cache). When ``None``
    each ``complete_with_tools_batch`` call creates its own batch-scoped
    cache, so personas in the same round still share results.
    """
    logger.info(
        "schema_execution_start",
        task_id=task.id,
        num_rounds=len(schema.rounds),
        synthesis=schema.final_synthesis.value,
        schema_summary={
            "max_rounds": schema.max_rounds,
            "num_defined_rounds": len(schema.rounds),
            "rounds": [
                {
                    "round": r.round,
                    "personas": [p.value for p in r.personas],
                    "tools": [t.value for t in r.tools],
                    "instruction": r.instruction.value,
                }
                for r in schema.rounds
            ],
        },
    )

    all_responses: list[list[dict[str, str]]] = []
    all_trace: list[dict] = []
    total_llm_calls = 0
    total_tool_calls = 0
    rounds_executed = 0

    previous_responses: list[dict[str, str]] = []

    for round_spec in schema.rounds:
        if rounds_executed >= schema.max_rounds:
            logger.info("max_rounds_reached", max_rounds=schema.max_rounds)
            break

        responses, llm_calls, tool_calls, round_trace = run_debate_round(
            round_spec=round_spec,
            question=task.question,
            previous_round_responses=previous_responses,
            agent_client=agent_client,
            tool_registry=tool_registry,
            mode=schema.mode,
            tool_cache=tool_cache,
        )

        all_responses.append(responses)
        all_trace.extend(round_trace)
        total_llm_calls += llm_calls
        total_tool_calls += tool_calls
        rounds_executed += 1

        # Accumulate responses for next round. Schemas have no early-stop
        # mechanism — every defined round runs to completion.
        previous_responses.extend(responses)

    predicted_answer = _extract_final_answer(schema, all_responses)

    logger.info(
        "schema_execution_complete",
        task_id=task.id,
        rounds_executed=rounds_executed,
        llm_calls=total_llm_calls,
        tool_calls=total_tool_calls,
        predicted_answer=predicted_answer[:200],
    )

    return ExecutionResult(
        predicted_answer=predicted_answer,
        all_responses=all_responses,
        num_rounds_executed=rounds_executed,
        num_llm_calls=total_llm_calls,
        num_tool_calls=total_tool_calls,
        execution_trace=all_trace,
    )
