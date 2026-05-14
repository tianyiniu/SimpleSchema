"""Dual-judge scoring for the one-shot dataset.

Each schema execution is scored by a single judge LLM call that returns
BOTH a binary correctness score (SimpleQA-style 3-category grading
collapsed to 0/1) and a 0-5 Likert trace-quality score. Every record in
the dataset goes through the LLM judge — there is no regex / exact-match
shortcut on correctness anywhere in this package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.engine import ExecutionResult
from src.judge import judge_dual
from src.llm_client import LLMClient
from src.logging import get_logger
from src.supervisor import supervised_execute
from src.tools import ToolCallCache, ToolRegistry
from src.types import Task, Schema

logger = get_logger(__name__)


@dataclass
class DualJudgeResult:
    """Both scores + the execution result + per-schema judge call records.

    ``answer_score`` is binary (0.0 or 1.0); ``trace_score`` is 0-5.
    ``judge_calls`` carries the single judge entry (with the original
    CORRECT/INCORRECT/NOT_ATTEMPTED label preserved for analysis).
    ``execute_time`` and ``judge_time`` are wall-clock seconds for the
    schema execution and the judge call respectively, used by the
    one-shot driver's run profile.
    """

    answer_score: float
    trace_score: float
    execution_result: ExecutionResult
    judge_calls: list[dict]
    execute_time: float = 0.0
    judge_time: float = 0.0


def evaluate_schema_dual_judge(
    schema: Schema,
    task: Task,
    agent_client: LLMClient,
    judge_client: LLMClient,
    tool_registry: ToolRegistry,
    tool_cache: ToolCallCache | None = None,
) -> DualJudgeResult:
    """Execute a schema once and score it with the single dual judge.

    The returned ``DualJudgeResult`` carries the full ``ExecutionResult``
    (with ``execution_trace`` + ``all_responses``) and the one
    ``judge_dual`` entry that produced both scores. Pass a shared
    ``tool_cache`` to let schemas evaluated for the same task reuse each
    other's tool results.
    """
    logger.info("dual_judge_eval_start", task_id=task.id)

    t_exec = time.perf_counter()
    result = supervised_execute(
        schema, task, agent_client, tool_registry, tool_cache=tool_cache,
    )
    execute_time = time.perf_counter() - t_exec

    judge_calls: list[dict] = []
    t_judge = time.perf_counter()
    answer_score, trace_score = judge_dual(
        question=task.question,
        ground_truth=task.ground_truth,
        predicted_answer=result.predicted_answer,
        judge_client=judge_client,
        all_responses=result.all_responses,
        execution_trace=result.execution_trace,
        judge_log=judge_calls,
    )
    judge_time = time.perf_counter() - t_judge

    logger.info(
        "dual_judge_eval_complete",
        task_id=task.id,
        answer_score=answer_score,
        trace_score=trace_score,
        execute_time=execute_time,
        judge_time=judge_time,
    )
    return DualJudgeResult(
        answer_score=answer_score,
        trace_score=trace_score,
        execution_result=result,
        judge_calls=judge_calls,
        execute_time=execute_time,
        judge_time=judge_time,
    )
