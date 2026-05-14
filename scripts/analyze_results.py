"""Aggregate per-run accuracy and trace scores across the 4 schemas.

Reads schemaN.json files written by run_oneshot.py and reports:
  * answer_score: pass@1 (mean across schemas) and best-of-4 (max across schemas)
  * trace_score: mean-of-4 and best-of-4
Both are first reduced per-question across the 4 schemas, then averaged across
questions. Results are broken down overall, by dataset (nq vs hotpot inferred
from task_id prefix), and by difficulty when the dataset record carries one
(hotpot only).

Usage:
    python scripts/analyze_results.py                 # all runs found
    python scripts/analyze_results.py --run-id ID     # one run (repeatable)
    python scripts/analyze_results.py --csv out.csv   # also dump raw rows
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


def load_difficulty_lookup(data_dir: Path) -> dict[str, str]:
    """Map task_id -> difficulty for every jsonl that carries the field."""
    lookup: dict[str, str] = {}
    for path in data_dir.rglob("*.jsonl"):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                diff = rec.get("difficulty")
                tid = rec.get("id")
                if tid and diff and tid not in lookup:
                    lookup[tid] = diff
    return lookup


def infer_dataset(task_id: str) -> str:
    if task_id.startswith("nq_"):
        return "nq"
    if task_id.startswith("hotpot_"):
        return "hotpot"
    return task_id.split("_", 1)[0]


def discover_runs(results_dir: Path) -> dict[str, list[Path]]:
    """run_id -> list of run directories (one per question)."""
    runs: dict[str, list[Path]] = defaultdict(list)
    for run_dir in results_dir.glob("*/*/oneshot_*"):
        if run_dir.is_dir():
            runs[run_dir.name].append(run_dir)
    return runs


def load_question_rows(run_dirs: list[Path]) -> list[dict]:
    """One row per question: aggregated answer/trace stats across schemas."""
    rows: list[dict] = []
    for qdir in run_dirs:
        schema_files = sorted(qdir.glob("schema*.json"))
        schema_files = [p for p in schema_files if not p.name.endswith("_trace.json")]
        if not schema_files:
            continue
        answers: list[float] = []
        traces: list[float] = []
        task_id = None
        for p in schema_files:
            d = json.loads(p.read_text())
            task_id = d["task_id"]
            answers.append(float(d["answer_score"]))
            traces.append(float(d["trace_score"]))
        rows.append(
            {
                "task_id": task_id,
                "n_schemas": len(answers),
                "answer_mean": mean(answers),
                "answer_best": max(answers),
                "trace_mean": mean(traces),
                "trace_best": max(traces),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        "n": len(rows),
        "answer_pass_at_1": mean(r["answer_mean"] for r in rows),
        "answer_best_of_4": mean(r["answer_best"] for r in rows),
        "trace_mean_of_4": mean(r["trace_mean"] for r in rows),
        "trace_best_of_4": mean(r["trace_best"] for r in rows),
    }


def format_summary(label: str, s: dict[str, float]) -> str:
    return (
        f"  {label:<28s} n={s['n']:<4d}  "
        f"answer pass@1: {s['answer_pass_at_1']:.3f}  "
        f"best-of-4: {s['answer_best_of_4']:.3f}   "
        f"trace mean: {s['trace_mean_of_4']:.2f}  "
        f"best-of-4: {s['trace_best_of_4']:.2f}"
    )


def report_run(run_id: str, rows: list[dict], difficulty: dict[str, str]) -> list[dict]:
    """Print a per-run report; return flat records for optional CSV export."""
    print(f"\n=== {run_id} ===")
    overall = summarize(rows)
    if overall is None:
        print("  (no schema records found)")
        return []
    print(format_summary("OVERALL", overall))

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_dataset_diff: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        ds = infer_dataset(r["task_id"])
        by_dataset[ds].append(r)
        diff = difficulty.get(r["task_id"])
        if diff:
            by_dataset_diff[(ds, diff)].append(r)

    print("  -- by dataset --")
    for ds in sorted(by_dataset):
        s = summarize(by_dataset[ds])
        assert s is not None
        print(format_summary(ds, s))

    if by_dataset_diff:
        print("  -- by dataset / difficulty --")
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        for ds, diff in sorted(
            by_dataset_diff,
            key=lambda k: (k[0], diff_order.get(k[1], 99), k[1]),
        ):
            s = summarize(by_dataset_diff[(ds, diff)])
            assert s is not None
            print(format_summary(f"{ds}/{diff}", s))

    flat: list[dict] = []
    flat.append({"run_id": run_id, "group": "overall", **overall})
    for ds, rs in by_dataset.items():
        flat.append({"run_id": run_id, "group": f"dataset={ds}", **summarize(rs)})
    for (ds, diff), rs in by_dataset_diff.items():
        flat.append(
            {"run_id": run_id, "group": f"dataset={ds};difficulty={diff}", **summarize(rs)}
        )
    return flat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", action="append", default=[],
                    help="restrict to this run_id (repeatable; default: all)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--csv", type=Path, default=None,
                    help="optional path to write a flat CSV of all summary rows")
    args = ap.parse_args()

    difficulty = load_difficulty_lookup(args.data_dir)
    runs = discover_runs(args.results_dir)
    if not runs:
        print(f"no runs found under {args.results_dir}", file=sys.stderr)
        return 1

    selected = sorted(args.run_id) if args.run_id else sorted(runs)
    missing = [r for r in selected if r not in runs]
    for r in missing:
        print(f"warning: run_id {r!r} not found", file=sys.stderr)

    all_flat: list[dict] = []
    for run_id in selected:
        if run_id not in runs:
            continue
        rows = load_question_rows(runs[run_id])
        all_flat.extend(report_run(run_id, rows, difficulty))

    if args.csv and all_flat:
        fields = ["run_id", "group", "n", "answer_pass_at_1", "answer_best_of_4",
                  "trace_mean_of_4", "trace_best_of_4"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_flat)
        print(f"\nwrote {len(all_flat)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
