#!/usr/bin/env python3
"""Build training splits from NQ Open + HotPotQA.

Downloads both datasets, normalizes each row to ``{id, question,
ground_truth, source, ...}``, then writes five independently-sampled
JSONL files:

  - ``data/nq/train.jsonl``           — per-source NQ Open pool (uniform random)
  - ``data/hotpotqa/train.jsonl``     — per-source HotPotQA pool (stratified by difficulty)
  - ``data/train_300/train.jsonl``    — 150 NQ + 150 HP (HP stratified by difficulty)
  - ``data/train_4/train.jsonl``      — 2 NQ + 2 HP (smoke-test sized)
  - ``data/train_hp_300/train.jsonl`` — 300 HP, exactly 100 each of easy/medium/hard

NQ Open (``google-research-datasets/nq_open``) is used instead of the
full ``natural_questions`` dataset because we only consume the
question + short-answer fields anyway. NQ Open is ~250x smaller on disk
(no Wikipedia documents) so the load step is seconds instead of minutes.

Each output is sampled with its own derived seed, so the mixed datasets
are independent draws — not nested subsets — from the same underlying
pools. Test splits are not produced here; they are reserved for
downstream evaluation pipelines.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

CACHE_DIR = "/nas-ssd2/tianyin4/cache/pretrained_models"


def _normalize_nq(row: dict, idx: int) -> dict | None:
    """Convert an NQ Open row to the project's task shape.

    NQ Open rows expose the question directly as ``question`` and a list
    of acceptable short answers as ``answer``. We keep the first answer
    as the gold target; the SimpleQA-style judge handles benign
    aliasing (e.g. "USA" vs. "United States") via semantic grading.
    Skip rows that are missing either field.
    """
    question = (row.get("question") or "").strip()
    answers = [a.strip() for a in (row.get("answer") or []) if a and a.strip()]
    if not question or not answers:
        return None
    return {
        "id": f"nq_train_{idx}",
        "question": question,
        "ground_truth": answers[0],
        "source": "nq_open",
    }


def _normalize_hotpot(row: dict, idx: int) -> dict | None:
    """HotPotQA exposes a ``level`` field in {easy, medium, hard}; preserve it
    under the ``difficulty`` key so downstream samplers can stratify and the
    ``Task`` model (which has ``difficulty: str | None``) loads it directly.
    """
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not question or not answer:
        return None
    difficulty = (row.get("level") or "").strip().lower() or None
    return {
        "id": f"hotpot_train_{idx}",
        "question": question,
        "ground_truth": answer,
        "source": "hotpot_qa",
        "difficulty": difficulty,
    }


def _stratified_sample(
    pool: list[dict],
    num_samples: int,
    seed: int,
    stratify_key: str | None,
) -> list[dict]:
    """Uniform random sample over the full pool, optionally balanced per bucket.

    When ``stratify_key`` is None, this is just a Fisher-Yates shuffle of
    the full pool followed by a slice — equivalent to drawing ``num_samples``
    rows uniformly at random from the entire dataset.

    When ``stratify_key`` is provided, the pool is bucketed by that field
    and the budget is divided evenly across buckets (remainder spread
    randomly). If a bucket is smaller than its share, the shortfall is
    backfilled from the remaining items so the total still hits
    ``num_samples`` whenever the pool is large enough.
    """
    rng = random.Random(seed)

    if stratify_key is None:
        rng.shuffle(pool)
        return pool[:num_samples]

    buckets: dict[object, list[dict]] = {}
    for item in pool:
        buckets.setdefault(item.get(stratify_key), []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets.keys())
    rng.shuffle(keys)
    per_bucket, remainder = divmod(num_samples, len(keys))

    sampled: list[dict] = []
    leftover: list[dict] = []
    for i, k in enumerate(keys):
        target = per_bucket + (1 if i < remainder else 0)
        take = min(target, len(buckets[k]))
        sampled.extend(buckets[k][:take])
        leftover.extend(buckets[k][take:])

    if len(sampled) < num_samples:
        rng.shuffle(leftover)
        sampled.extend(leftover[: num_samples - len(sampled)])

    rng.shuffle(sampled)

    counts: dict[object, int] = {}
    for item in sampled:
        counts[item.get(stratify_key)] = counts.get(item.get(stratify_key), 0) + 1
    print(f"Stratified sample by '{stratify_key}': {dict(sorted(counts.items(), key=lambda kv: str(kv[0])))}")

    return sampled


def _build_pool(rows, normalize_fn) -> list[dict]:
    """Normalize every row, drop rejects, return the surviving pool."""
    pool: list[dict] = []
    for i, row in enumerate(rows):
        item = normalize_fn(row, i)
        if item is not None:
            pool.append(item)
    return pool


def _write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} samples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Per-source pool cap for data/nq + data/hotpotqa (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed; per-output seeds are derived from this.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    # Derive an independent seed for each of the six sampling calls so that
    # train_300 and train_4 are truly independent draws (not nested subsets).
    master = random.Random(args.seed)
    seeds = {
        name: master.randint(0, 2**32 - 1)
        for name in (
            "nq_train",
            "hotpotqa_train",
            "train_300_nq",
            "train_300_hp",
            "train_4_nq",
            "train_4_hp",
            "train_hp_300",
        )
    }

    print("Loading nq_open train split…")
    nq_rows = load_dataset(
        "google-research-datasets/nq_open", split="train", cache_dir=CACHE_DIR
    )
    nq_pool = _build_pool(nq_rows, _normalize_nq)
    print(f"NQ pool size: {len(nq_pool)}")

    print("Loading hotpot_qa train split…")
    hp_rows = load_dataset(
        "hotpot_qa", "distractor", split="train", cache_dir=CACHE_DIR
    )
    hp_pool = _build_pool(hp_rows, _normalize_hotpot)
    print(f"HotPotQA pool size: {len(hp_pool)}")

    # Per-source pools (uniform NQ; HP stratified by difficulty).
    _write_jsonl(
        _stratified_sample(list(nq_pool), args.num_samples, seeds["nq_train"], None),
        args.data_root / "nq" / "train.jsonl",
    )
    _write_jsonl(
        _stratified_sample(list(hp_pool), args.num_samples, seeds["hotpotqa_train"], "difficulty"),
        args.data_root / "hotpotqa" / "train.jsonl",
    )

    # Mixed datasets: independent draws from the same pools.
    train_300 = (
        _stratified_sample(list(nq_pool), 150, seeds["train_300_nq"], None)
        + _stratified_sample(list(hp_pool), 150, seeds["train_300_hp"], "difficulty")
    )
    _write_jsonl(train_300, args.data_root / "train_300" / "train.jsonl")

    train_4 = (
        _stratified_sample(list(nq_pool), 2, seeds["train_4_nq"], None)
        + _stratified_sample(list(hp_pool), 2, seeds["train_4_hp"], "difficulty")
    )
    _write_jsonl(train_4, args.data_root / "train_4" / "train.jsonl")

    # HotPotQA-only mixed dataset: exactly 100 easy + 100 medium + 100 hard.
    # 300 // 3 = 100 with zero remainder, so the stratified sampler hits
    # the per-bucket target exactly.
    train_hp_300 = _stratified_sample(
        list(hp_pool), 300, seeds["train_hp_300"], "difficulty",
    )
    _write_jsonl(train_hp_300, args.data_root / "train_hp_300" / "train.jsonl")


if __name__ == "__main__":
    main()
