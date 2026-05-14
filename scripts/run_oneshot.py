#!/usr/bin/env python3
"""One-shot data collection for reward-model training.

For each task:
  1. The **agent** LLM is asked to propose 4 *structurally distinct*
     schemas in a single call (the prompt actively pushes for diversity
     across round count, persona mix, tool selection, instructions, and
     synthesis method).
  2. All 4 schemas are executed against the task in parallel.
  3. A single judge call grades each execution on BOTH dimensions:
       - ``answer_score``  — binary correctness vs. ground truth
         (SimpleQA-style 3-category grading collapsed to 0/1).
       - ``trace_score``   — 0-5 Likert on debate-trace process quality.
  4. Each (task, schema) is persisted as two JSON files:
       results/{dataset}_{split}/{question_id}/{run_id}/schemaN.json
         — overview: question, ground truth, prediction, both scores,
           resource counts, and the schema definition.
       results/{dataset}_{split}/{question_id}/{run_id}/schemaN_trace.json
         — full execution trace, per-round responses, and the single
           dual-judge call record that scored this schema.

The frozen run configuration is written to
``results_configs/{run_id}.json``.

The orchestrator client is intentionally not used here — it will be
trained later from this dataset to predict schema quality (reward model).
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Ensure THIS package's src/ wins over any sibling/editable install of the
# main project that may also expose a top-level `src` package.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.fitness import evaluate_schema_dual_judge
from src.generator import propose_schemas_batch
from src.types import Schema, Task
from src.tools import ToolCallCache, ToolRegistry, create_tool_registry
from src.config import corpus_base_url, load_config, resolve_run_settings
from src.datasets import (
    DATASET_CHOICES,
    SPLIT_CHOICES,
    get_dataset_spec,
    load_tasks as load_dataset_tasks,
)
from src.llm_client import LLMClients
from src.logging import get_logger, setup_logging, tqdm_logging_redirect
from src.serialization import save_json_durable, schema_to_dict

logger = get_logger(__name__)


def _default_run_id(dataset: str, split: str, max_tasks: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"oneshot_{ts}_{dataset}_{split}_tasks{max_tasks}"


# ---------------------------------------------------------------------------
# Run profiling
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Thread-safe collector for per-task and per-schema timings.

    Populated from ``_process_task`` (which runs concurrently across
    tasks) and rendered into a profile summary at the end of ``main``.
    """

    proposal_times: list[float] = field(default_factory=list)
    task_times: list[float] = field(default_factory=list)
    execute_times: list[float] = field(default_factory=list)
    judge_times: list[float] = field(default_factory=list)
    tool_cache_sizes: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_task(
        self,
        proposal_time: float,
        task_time: float,
        execute_times: list[float],
        judge_times: list[float],
        tool_cache_size: int,
    ) -> None:
        with self._lock:
            self.proposal_times.append(proposal_time)
            self.task_times.append(task_time)
            self.execute_times.extend(execute_times)
            self.judge_times.extend(judge_times)
            self.tool_cache_sizes.append(tool_cache_size)


def _summary(values: list[float]) -> dict[str, float]:
    """mean / p50 / p95 / max in seconds; zeros when ``values`` is empty."""
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p50 = statistics.median(sorted_vals)
    # statistics.quantiles requires n >= 2; for n == 1 just use the single value.
    p95 = (
        statistics.quantiles(sorted_vals, n=20, method="inclusive")[18]
        if n >= 2 else sorted_vals[0]
    )
    return {
        "n": n,
        "mean": statistics.fmean(sorted_vals),
        "p50": p50,
        "p95": p95,
        "max": sorted_vals[-1],
    }


def _print_profile(
    metrics: RunMetrics,
    clients: LLMClients,
    wall_clock: float,
    num_tasks: int,
    total_records: int,
    task_workers: int,
    eval_workers: int,
) -> None:
    """Render a human-readable run profile to stdout."""
    width = 64
    print("\n" + "=" * width)
    print(f"{'Run profile':^{width}}")
    print("=" * width)

    # ---- top-line ----
    print(f"Wall-clock:                  {wall_clock:>8.1f}s")
    print(f"Tasks processed:             {num_tasks:>8d}   (parallel: {task_workers})")
    print(f"Schemas evaluated:           {total_records:>8d}   (parallel: {eval_workers}/task)")
    if metrics.tool_cache_sizes:
        max_cache = max(metrics.tool_cache_sizes)
        mean_cache = statistics.fmean(metrics.tool_cache_sizes)
        print(
            f"Tool-cache entries per task: mean={mean_cache:>5.1f}  max={max_cache:>3d}"
        )

    # ---- per-LLM-client ----
    print()
    print(f"{'LLM client':<14}{'count':>8}{'mean (s)':>12}{'total (s)':>12}")
    print("-" * width)
    for label, client in [("agent", clients.agent), ("judge", clients.judge)]:
        n = client.call_count
        mean = client.call_time_avg
        total = client.call_time_total
        print(f"  {label:<12}{n:>8d}{mean:>12.2f}{total:>12.1f}")

    # ---- per-task ----
    print()
    print(
        f"{'Per task':<14}{'n':>5}"
        f"{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}"
    )
    print("-" * width)
    for label, vals in [
        ("proposal",  metrics.proposal_times),
        ("total",     metrics.task_times),
    ]:
        s = _summary(vals)
        print(
            f"  {label:<12}{s['n']:>5d}"
            f"{s['mean']:>8.1f}s{s['p50']:>8.1f}s{s['p95']:>8.1f}s{s['max']:>8.1f}s"
        )

    # ---- per-schema ----
    print()
    print(
        f"{'Per schema':<14}{'n':>5}"
        f"{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}"
    )
    print("-" * width)
    for label, vals in [
        ("execute",     metrics.execute_times),
        ("judge wait",  metrics.judge_times),
    ]:
        s = _summary(vals)
        print(
            f"  {label:<12}{s['n']:>5d}"
            f"{s['mean']:>8.1f}s{s['p50']:>8.1f}s{s['p95']:>8.1f}s{s['max']:>8.1f}s"
        )
    print("=" * width)


def _profile_to_dict(
    metrics: RunMetrics,
    clients: LLMClients,
    wall_clock: float,
    num_tasks: int,
    total_records: int,
) -> dict:
    """JSON-serializable snapshot of the same profile, for archival."""
    return {
        "wall_clock_seconds": wall_clock,
        "tasks_processed": num_tasks,
        "schemas_evaluated": total_records,
        "tool_cache_per_task": _summary(
            [float(x) for x in metrics.tool_cache_sizes]
        ),
        "per_task": {
            "proposal": _summary(metrics.proposal_times),
            "total":    _summary(metrics.task_times),
        },
        "per_schema": {
            "execute":    _summary(metrics.execute_times),
            "judge_wait": _summary(metrics.judge_times),
        },
        "llm_clients": {
            "agent": {
                "count": clients.agent.call_count,
                "mean_seconds": clients.agent.call_time_avg,
                "total_seconds": clients.agent.call_time_total,
            },
            "judge": {
                "count": clients.judge.call_count,
                "mean_seconds": clients.judge.call_time_avg,
                "total_seconds": clients.judge.call_time_total,
            },
        },
    }


def _propose_schemas(
    question: str,
    agent_client,
    rng: random.Random,
    n: int,
    debate_mode: str,
) -> list[tuple[int, Schema]]:
    """Single-call proposal. Returns ``[(slot, schema), ...]`` (slot is 0..n-1)."""
    batch = propose_schemas_batch(
        question=question,
        agent_client=agent_client,
        n=n,
        rng=rng,
        debate_mode=debate_mode,
    )
    return list(enumerate(batch))


def _process_task(
    task: Task,
    task_idx: int,
    total_tasks: int,
    clients: LLMClients,
    tool_registry: ToolRegistry,
    split_dir: Path,
    run_id: str,
    num_schemas: int,
    debate_mode: str,
    eval_workers: int,
    rng: random.Random,
    metrics: RunMetrics,
) -> int:
    """Propose + execute + judge all schemas for a single task; return record count.

    Each task owns its own ``ToolCallCache`` so the N schemas for this
    question share corpus / fetch / compute results, but the cache does
    not leak across tasks.
    """
    logger.info(
        "oneshot_task_start",
        task_idx=task_idx + 1,
        total_tasks=total_tasks,
        task_id=task.id,
    )

    t_task_start = time.perf_counter()
    t_propose = time.perf_counter()
    proposals = _propose_schemas(
        question=task.question,
        agent_client=clients.agent,
        rng=rng,
        n=num_schemas,
        debate_mode=debate_mode,
    )
    proposal_time = time.perf_counter() - t_propose

    if not proposals:
        logger.warning("oneshot_no_schemas_proposed", task_id=task.id)
        metrics.add_task(
            proposal_time=proposal_time,
            task_time=time.perf_counter() - t_task_start,
            execute_times=[],
            judge_times=[],
            tool_cache_size=0,
        )
        return 0

    logger.info(
        "oneshot_schemas_proposed",
        task_id=task.id,
        count=len(proposals),
        expected=num_schemas,
    )

    # Stable 1..N file naming across out-of-order future completion.
    flat_index: dict[int, int] = {
        slot: i + 1 for i, (slot, _) in enumerate(proposals)
    }

    task_run_dir = split_dir / task.id / run_id
    task_run_dir.mkdir(parents=True, exist_ok=True)

    # Per-task tool cache: shared across all N schemas for this question
    # so e.g. an identical search_info("Einstein birthplace") issued by
    # schema1's analyst and schema3's critic hits the corpus once.
    task_tool_cache = ToolCallCache()

    task_records = 0
    execute_times: list[float] = []
    judge_times: list[float] = []
    with ThreadPoolExecutor(max_workers=eval_workers) as pool:
        futures = {
            pool.submit(
                evaluate_schema_dual_judge,
                schema, task, clients.agent, clients.judge, tool_registry,
                task_tool_cache,
            ): (slot, schema)
            for (slot, schema) in proposals
        }
        for future in as_completed(futures):
            slot, schema = futures[future]
            schema_index = flat_index[slot]
            try:
                dual = future.result()
            except Exception:
                logger.exception(
                    "oneshot_schema_eval_failed",
                    task_id=task.id,
                    schema_idx_in_batch=slot,
                    schema_index=schema_index,
                )
                continue

            execute_times.append(dual.execute_time)
            judge_times.append(dual.judge_time)
            exec_result = dual.execution_result

            overview = {
                "task_id": task.id,
                "level": task.level,
                "question": task.question,
                "ground_truth": task.ground_truth,
                "predicted_answer": exec_result.predicted_answer,
                "answer_score": dual.answer_score,
                "trace_score": dual.trace_score,
                "num_rounds_executed": exec_result.num_rounds_executed,
                "num_llm_calls": exec_result.num_llm_calls,
                "num_tool_calls": exec_result.num_tool_calls,
                "schema_index": schema_index,
                "schema_idx_in_batch": slot,
                "run_id": run_id,
                "schema": schema_to_dict(schema),
            }
            trace = {
                "task_id": task.id,
                "schema_index": schema_index,
                "run_id": run_id,
                "execution_trace": exec_result.execution_trace,
                "all_responses": exec_result.all_responses,
                "judge_calls": dual.judge_calls,
            }

            save_json_durable(overview, task_run_dir / f"schema{schema_index}.json")
            save_json_durable(trace, task_run_dir / f"schema{schema_index}_trace.json")
            task_records += 1

            logger.info(
                "oneshot_schema_eval_complete",
                task_id=task.id,
                schema_index=schema_index,
                schema_idx_in_batch=slot,
                answer_score=dual.answer_score,
                trace_score=dual.trace_score,
                predicted_answer=exec_result.predicted_answer[:200],
            )

    task_time = time.perf_counter() - t_task_start
    metrics.add_task(
        proposal_time=proposal_time,
        task_time=task_time,
        execute_times=execute_times,
        judge_times=judge_times,
        tool_cache_size=len(task_tool_cache),
    )

    logger.info(
        "oneshot_task_complete",
        task_id=task.id,
        records_written=task_records,
        tool_cache_entries=len(task_tool_cache),
        proposal_time=proposal_time,
        task_time=task_time,
    )
    return task_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot reward-model data collection: agent proposes 8 schemas "
            "per task; both judges score the executions."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default="nq", choices=DATASET_CHOICES,
        help="Benchmark dataset (default: nq)",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=SPLIT_CHOICES,
        help="Dataset split (default: train).",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=1000,
        help="Max tasks to process (default: 10).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID (default: auto-generated).",
    )
    parser.add_argument(
        "--config", type=str, default="config.toml",
        help="Path to config.toml (default: ./config.toml).",
    )
    parser.add_argument(
        "--debate-mode",
        choices=["persona", "self_consistency", "mixed"],
        default=None,
        help="Constrain proposed schemas to this mode (default: from [oneshot]).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Data collection runs against the static corpus; the agent + judge
    # may be live-served LLMs, but the tool corpus must be the static index.
    mode, corpus = resolve_run_settings(cfg, args.split, "static")

    oneshot_cfg = cfg.get("oneshot", {})
    num_schemas = int(oneshot_cfg.get("schemas_per_batch", 4))
    debate_mode = (
        args.debate_mode
        if args.debate_mode is not None
        else str(oneshot_cfg.get("debate_mode", "mixed"))
    )

    dataset_spec = get_dataset_spec(args.dataset, args.split)

    run_id = args.run_id or _default_run_id(
        dataset_spec.name, dataset_spec.split, args.max_tasks
    )
    split_dir = Path("results") / f"{dataset_spec.name}_{dataset_spec.split}"
    split_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = Path("results_configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Quiet console (WARNING+); full structured trace lands in the per-run log.
    log_path = configs_dir / f"{run_id}.log"
    setup_logging(log_path=log_path)
    logger.info(
        "oneshot_run_start",
        run_id=run_id,
        dataset=dataset_spec.name,
        split=dataset_spec.split,
        mode=mode,
        corpus=corpus,
        num_schemas=num_schemas,
        debate_mode=debate_mode,
    )

    clients = LLMClients.from_config(cfg)
    tool_registry = create_tool_registry(
        corpus_mode=corpus, corpus_base_url=corpus_base_url(cfg)
    )
    tasks = load_dataset_tasks(dataset_spec, max_tasks=args.max_tasks)
    logger.info("tasks_loaded", num_tasks=len(tasks))

    save_json_durable(
        {
            "run_id": run_id,
            "type": "oneshot",
            "dataset": dataset_spec.name,
            "split": dataset_spec.split,
            "mode": mode,
            "corpus": corpus,
            "num_schemas": num_schemas,
            "debate_mode": debate_mode,
            "task_ids": [t.id for t in tasks],
            "tool_names": tool_registry.available_tools,
            "llm": cfg.get("llm", {}),
        },
        configs_dir / f"{run_id}.json",
    )

    rng = random.Random(int(oneshot_cfg.get("random_seed", 42)))
    # Per-task evaluation parallelism = #schemas; bounded by judge throughput.
    eval_workers = int(oneshot_cfg.get("eval_workers", num_schemas))
    # Inter-task parallelism: pipeline propose/execute/judge across tasks
    # so the agent and judge aren't idle during the other phase. Total
    # in-flight agent calls = task_workers * eval_workers * max_personas;
    # make sure [llm.agent].max_workers is sized for that.
    task_workers = int(oneshot_cfg.get("task_workers", 1))

    logger.info(
        "oneshot_concurrency",
        task_workers=task_workers,
        eval_workers=eval_workers,
        agent_max_workers=clients.agent.config.max_workers,
        judge_max_workers=clients.judge.config.max_workers,
    )

    # Terminal-only startup banner. Everything else of value goes to the
    # log file at INFO; console handler is at WARNING so only failures and
    # explicit warnings will appear during the run.
    print(f"Run:      {run_id}")
    print(f"Dataset:  {dataset_spec.name}/{dataset_spec.split}  "
          f"({len(tasks)} task(s) x {num_schemas} schemas = "
          f"{len(tasks) * num_schemas} evaluations)")
    print(f"Mode:     {debate_mode}   corpus={corpus}")
    print(f"Parallel: tasks={task_workers}  schemas/task={eval_workers}  "
          f"agent_max_workers={clients.agent.config.max_workers}  "
          f"judge_max_workers={clients.judge.config.max_workers}")
    print(f"Log:      {log_path}")
    print()

    metrics = RunMetrics()
    wall_clock_start = time.perf_counter()

    total_records = 0
    # tqdm_logging_redirect routes any WARNING/ERROR through tqdm.write
    # so it appears on its own line above the bar instead of smearing
    # it, while preserving the WARNING console filter (the upstream
    # context manager would silently undo it).
    with tqdm_logging_redirect():
        with ThreadPoolExecutor(max_workers=task_workers) as task_pool:
            task_futures = {
                task_pool.submit(
                    _process_task,
                    task, task_idx, len(tasks),
                    clients, tool_registry,
                    split_dir, run_id,
                    num_schemas, debate_mode, eval_workers, rng,
                    metrics,
                ): task
                for task_idx, task in enumerate(tasks)
            }
            with tqdm(
                total=len(tasks), desc="Tasks", unit="task",
                dynamic_ncols=True, smoothing=0.1,
            ) as pbar:
                for fut in as_completed(task_futures):
                    task = task_futures[fut]
                    try:
                        total_records += fut.result()
                    except Exception:
                        logger.exception("oneshot_task_failed", task_id=task.id)
                    pbar.update(1)
                    pbar.set_postfix(
                        records=total_records, refresh=False,
                    )

    wall_clock = time.perf_counter() - wall_clock_start

    logger.info(
        "oneshot_run_complete",
        run_id=run_id,
        total_tasks=len(tasks),
        total_records=total_records,
    )
    _print_profile(
        metrics, clients, wall_clock,
        num_tasks=len(tasks),
        total_records=total_records,
        task_workers=task_workers,
        eval_workers=eval_workers,
    )
    profile_path = configs_dir / f"{run_id}_profile.json"
    save_json_durable(
        _profile_to_dict(metrics, clients, wall_clock, len(tasks), total_records),
        profile_path,
    )
    print(f"Results: {split_dir}")
    print(f"Profile: {profile_path}")

    clients.shutdown()


if __name__ == "__main__":
    main()
