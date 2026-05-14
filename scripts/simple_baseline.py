#!/usr/bin/env python3
"""Single-turn agent baseline over the training datasets.

Evaluates the **agent** LLM on benchmark questions using a single-turn
prompt with unlimited tool calls (no debate schemas, no synthesis, no
multi-round refinement). The agent receives the question, may issue any
number of tool calls in one extended turn, and produces a final answer.
Each answer is graded by the dual judge — only the binary correctness
score is used here (trace_score is meaningless without debate).

To enable a compute-matched comparison against run_oneshot.py (which
proposes 4 schemas per task), `--n-samples` runs N independent baseline
attempts per question (default 4). Terminal output then reports pass@1
(mean across samples) and best-of-N (max across samples), averaged
across questions, broken down by:
  - dataset (e.g. nq, hotpotqa, train_300, train_4)
  - dataset × difficulty (HotPotQA easy/medium/hard; NQ has none)

Usage:
    python scripts/simple_baseline.py
    python scripts/simple_baseline.py --datasets nq hotpotqa
    python scripts/simple_baseline.py --datasets train_4 --max-tasks 4
    python scripts/simple_baseline.py --n-samples 1   # single-attempt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

# Ensure THIS package's src/ wins on sys.path before any sibling installs.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.config import corpus_base_url, load_config, resolve_run_settings
from src.datasets import DATASETS, DATASET_CHOICES, get_dataset_spec, load_tasks
from src.engine import _extract_answer_from_response
from src.judge import judge_dual
from src.llm_client import LLMClients
from src.logging import get_logger, setup_logging
from src.tools import create_tool_registry
from src.types import Task

logger = get_logger(__name__)


BASELINE_SYSTEM_PROMPT = (
    "You are a helpful assistant answering benchmark questions accurately. "
    "Use the available tools (search, fetch URLs, code execution) to gather "
    "evidence when the answer is not already certain. Think step by step, "
    "consult tools as many times as you need, and then state your final "
    "answer. Prefix the final answer with 'ANSWER: ' on its own line so it "
    "can be extracted automatically."
)


def _build_user_prompt(question: str, available_tools: list[str]) -> str:
    parts = [
        f"Question: {question}",
        "\nReason through the problem, use any tools you need, and then "
        "produce your final answer prefixed with 'ANSWER: '.",
    ]
    tool_hints = {
        "search_info": "- search_info args: {'query': '<focused search query>'}",
        "fetch_url":   "- fetch_url args: {'query': '<full URL>'}",
        "code_compute": "- code_compute args: {'query': '<python expression or full script>'}",
    }
    if available_tools:
        parts.append("\nTool call format:")
        for t in available_tools:
            hint = tool_hints.get(t)
            if hint is not None:
                parts.append(hint)
    return "\n".join(parts)


def _eval_one_task(
    task: Task,
    dataset_name: str,
    split: str,
    sample_idx: int,
    agent_client,
    judge_client,
    tool_registry,
    max_tool_rounds: int,
    max_tokens: int,
) -> dict:
    """Run the single-turn baseline on one task; judge correctness."""
    available = list(tool_registry.available_tools)
    tool_schemas, tool_executors = tool_registry.get_openai_tools(available)
    user_prompt = _build_user_prompt(task.question, available)

    result = agent_client.complete_with_tools(
        system_prompt=BASELINE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tool_schemas,
        tool_executors=tool_executors,
        max_tool_rounds=max_tool_rounds,
        max_tokens=max_tokens,
    )
    predicted_answer = _extract_answer_from_response(result.text)

    # Wrap the single response as a degenerate one-round "trace" so the
    # dual judge can grade correctness with the same prompt template.
    # The trace_score it produces here is meaningless (no debate) and we
    # discard it.
    all_responses = [[{"persona": "baseline", "response": result.text}]]
    execution_trace = [
        {
            "type": "tool_call",
            "round": 1,
            "persona": "baseline",
            "tool": tc.tool_name,
            "arguments": tc.arguments,
            "result": tc.result,
        }
        for tc in result.tool_traces
    ]
    answer_score, _ = judge_dual(
        question=task.question,
        ground_truth=task.ground_truth,
        predicted_answer=predicted_answer,
        judge_client=judge_client,
        all_responses=all_responses,
        execution_trace=execution_trace,
    )

    return {
        "dataset": dataset_name,
        "split": split,
        "task_id": task.id,
        "sample_idx": sample_idx,
        "difficulty": task.difficulty,
        "predicted_answer": predicted_answer,
        "ground_truth": task.ground_truth,
        "answer_score": answer_score,
        "num_tool_calls": result.num_tool_calls,
        "num_llm_calls": result.num_llm_calls,
    }


def _print_summary(records: list[dict], n_samples: int) -> None:
    """Per-question reduce across the N samples, then average across questions.

    Matches the layout of scripts/analyze_results.py so baseline numbers
    line up with the 4-schema run_oneshot.py numbers visually.
    """
    if not records:
        print("No records to summarize.")
        return

    # (dataset, task_id) -> list of per-sample scores
    per_q: dict[tuple[str, str], list[float]] = defaultdict(list)
    diff_of: dict[tuple[str, str], str | None] = {}
    for r in records:
        key = (r["dataset"], r["task_id"])
        per_q[key].append(r["answer_score"])
        diff_of[key] = r.get("difficulty")

    def _agg(keys: list[tuple[str, str]]) -> tuple[int, float, float] | None:
        if not keys:
            return None
        n = len(keys)
        pass1 = sum(mean(per_q[k]) for k in keys) / n
        bestn = sum(max(per_q[k]) for k in keys) / n
        return n, pass1, bestn

    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_dataset_diff: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in per_q:
        ds = key[0]
        by_dataset[ds].append(key)
        diff = diff_of[key]
        if diff:
            by_dataset_diff[(ds, diff)].append(key)

    def _fmt(label: str, agg: tuple[int, float, float]) -> str:
        n, p1, bn = agg
        return (
            f"  {label:<28s} n={n:<5d}  "
            f"answer pass@1: {p1:.3f}  best-of-{n_samples}: {bn:.3f}"
        )

    width = 80
    header = f"BASELINE — pass@1 / best-of-{n_samples} (n_samples={n_samples})"
    print("\n" + "=" * width)
    print(f"{header:^{width}}")
    print("=" * width)

    overall = _agg(list(per_q))
    assert overall is not None
    print(_fmt("OVERALL", overall))

    print("  -- by dataset --")
    for ds in sorted(by_dataset):
        agg = _agg(by_dataset[ds])
        assert agg is not None
        print(_fmt(ds, agg))

    if by_dataset_diff:
        print("  -- by dataset / difficulty --")
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        for ds, diff in sorted(
            by_dataset_diff,
            key=lambda k: (k[0], diff_order.get(k[1], 99), k[1]),
        ):
            agg = _agg(by_dataset_diff[(ds, diff)])
            assert agg is not None
            print(_fmt(f"{ds}/{diff}", agg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=DATASET_CHOICES,
        help="Dataset names to evaluate (default: every dataset with a present split file).",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="Split to evaluate (default: train).",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Per-dataset cap on tasks (default: all).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=4,
        help=(
            "Independent baseline attempts per question for a "
            "compute-matched comparison against the 4-schema oneshot run "
            "(default: 4). Setting this to 1 disables resampling."
        ),
    )
    parser.add_argument(
        "--max-tool-rounds", type=int, default=10,
        help="Max sequential tool-call rounds per task (default: 10).",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel task evaluations (default: 8).",
    )
    parser.add_argument(
        "--config", type=str, default="config.toml",
        help="Path to config.toml (default: ./config.toml).",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help=(
            "Path to write the structured run log. Pass an explicit path, "
            "or omit to auto-write to "
            "results_configs/simple_baseline_<timestamp>.log."
        ),
    )
    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("--n-samples must be >= 1")

    cfg = load_config(args.config)
    mode, corpus = resolve_run_settings(cfg, args.split, "static")

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        configs_dir = Path("results_configs")
        configs_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = configs_dir / f"simple_baseline_{ts}.log"
    setup_logging(log_path=log_path)
    print(f"[simple_baseline] logging to {log_path}")
    logger.info(
        "simple_baseline_run_start",
        datasets=args.datasets,
        split=args.split,
        mode=mode,
        corpus=corpus,
        max_tool_rounds=args.max_tool_rounds,
        max_tasks=args.max_tasks,
        n_samples=args.n_samples,
    )

    clients = LLMClients.from_config(cfg)
    tool_registry = create_tool_registry(
        corpus_mode=corpus, corpus_base_url=corpus_base_url(cfg),
    )
    agent_max_tokens = clients.agent.max_tokens

    # Resolve which (dataset, split) targets to run. Default = every
    # registered dataset whose split file actually exists on disk.
    if args.datasets:
        targets = [(name, args.split) for name in args.datasets]
    else:
        targets = [
            (name, split)
            for (name, split), spec in DATASETS.items()
            if split == args.split and spec.path.exists()
        ]
    targets.sort()

    if not targets:
        print(f"No datasets with a present '{args.split}' split file. "
              f"Run scripts/build_train.py first.")
        clients.shutdown()
        return

    print(f"Evaluating {len(targets)} dataset(s) on split '{args.split}':")
    for name, split in targets:
        spec = get_dataset_spec(name, split)
        path_exists = spec.path.exists()
        print(f"  - {name}/{split}  ({spec.path})  exists={path_exists}")

    all_records: list[dict] = []
    for name, split in targets:
        spec = get_dataset_spec(name, split)
        tasks = load_tasks(spec, max_tasks=args.max_tasks)
        if not tasks:
            print(f"\n[skip] {name}/{split}: no tasks loaded.")
            continue
        total_attempts = len(tasks) * args.n_samples
        print(f"\n=== {name}/{split} — {len(tasks)} tasks "
              f"× {args.n_samples} sample(s) = {total_attempts} attempts ===")

        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _eval_one_task, t, name, split, sidx,
                    clients.agent, clients.judge, tool_registry,
                    args.max_tool_rounds, agent_max_tokens,
                ): (t, sidx)
                for t in tasks
                for sidx in range(args.n_samples)
            }
            for fut in as_completed(futures):
                task, sidx = futures[fut]
                try:
                    rec = fut.result()
                except Exception:
                    logger.exception(
                        "simple_baseline_task_failed",
                        dataset=name, split=split, task_id=task.id,
                        sample_idx=sidx,
                    )
                    continue
                all_records.append(rec)
                completed += 1
                if completed % 10 == 0 or completed == total_attempts:
                    print(f"  [{completed}/{total_attempts}] last={task.id}#{sidx} "
                          f"score={rec['answer_score']:.0f} "
                          f"tool_calls={rec['num_tool_calls']}")

    _print_summary(all_records, args.n_samples)

    logger.info(
        "simple_baseline_run_complete",
        total_records=len(all_records),
        n_samples=args.n_samples,
    )
    clients.shutdown()


if __name__ == "__main__":
    main()
