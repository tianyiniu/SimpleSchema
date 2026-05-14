"""Orchestrator supervisory logic — early stopping and runtime management."""

from __future__ import annotations

from src.types import Task, Schema
from src.engine import ExecutionResult, execute_schema
from src.tools import ToolCallCache, ToolRegistry
from src.llm_client import LLMClient
from src.logging import get_logger

logger = get_logger(__name__)


def supervised_execute(
    schema: Schema,
    task: Task,
    agent_client: LLMClient,
    tool_registry: ToolRegistry,
    max_retries: int = 3,
    tool_cache: ToolCallCache | None = None,
) -> ExecutionResult:
    """Execute a schema with supervisory retries.

    If execution fails (raises an exception), retry up to max_retries times.
    On complete failure, return an empty result.
    """
    for attempt in range(max_retries):
        try:
            return execute_schema(
                schema, task, agent_client, tool_registry, tool_cache=tool_cache,
            )
        except Exception:
            logger.warning(
                "execution_failed_retrying",
                task_id=task.id,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

    logger.error("execution_failed_all_retries", task_id=task.id)
    return ExecutionResult(
        predicted_answer="",
        all_responses=[],
        num_rounds_executed=0,
        num_llm_calls=0,
        num_tool_calls=0,
    )


