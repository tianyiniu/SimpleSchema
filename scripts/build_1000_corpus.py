#!/usr/bin/env python3
"""Build four fixed, reproducible 1000-question datasets.

Outputs
-------
  data/nq/train.jsonl          1000 from NQ Open train split (uniform sample)
  data/nq/test.jsonl           1000 from Natural Questions validation split
  data/hotpotqa/train.jsonl    1000 from HotPotQA train split (stratified by difficulty)
  data/hotpotqa/test.jsonl     1000 from HotPotQA validation split (stratified by difficulty)

Each file has a companion keys file:
  data/nq/train_keys.json
  data/nq/test_keys.json
  data/hotpotqa/train_keys.json
  data/hotpotqa/test_keys.json

The keys file is a JSON array of ``orig_id`` values in the order they appear
in the JSONL. ``orig_id`` is the primary key from the upstream dataset:
  - NQ Open train   — question text (no native id in the nq_open dataset)
  - NQ test         — the Google-assigned numeric id string from natural_questions
  - HotPotQA        — the hex-hash ``id`` field from hotpot_qa

These keys files let collaborators confirm they selected the exact same
questions.  See ``read_keys`` for a convenience loader.

Sampling is fully deterministic: FIXED_SEED is hard-coded; there is no CLI
knob that would change which questions are selected.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

# Hard-coded — do not parameterise.  Changing this changes the dataset.
FIXED_SEED: int = 42
NUM_SAMPLES: int = 1000
CACHE_DIR: str = "/nas-ssd2/tianyin4/cache/pretrained_models"
_DATA_ROOT = Path("data")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_nq_train(row: dict, idx: int) -> dict | None:
    """NQ Open train row → task record.

    nq_open exposes only ``question`` and ``answer``; there is no native
    primary key.  We use the question text as ``orig_id`` — it is stable
    across re-downloads and unique within the dataset.
    """
    question = (row.get("question") or "").strip()
    answers = [a.strip() for a in (row.get("answer") or []) if a and a.strip()]
    if not question or not answers:
        return None
    return {
        "id": f"nq_train_{idx}",
        "orig_id": question,
        "question": question,
        "ground_truth": answers[0],
        "source": "nq_open",
    }


def _normalize_nq_test(row: dict, idx: int) -> dict | None:
    """Natural Questions validation row → task record.

    The full natural_questions dataset exposes a top-level ``id`` (numeric
    string like ``"5225754983651766092"``), which we use as ``orig_id``.
    """
    question = ((row.get("question") or {}).get("text") or "").strip()
    if not question:
        return None
    short_answers: list[str] = []
    for sa in ((row.get("annotations") or {}).get("short_answers") or []):
        short_answers.extend(t.strip() for t in (sa.get("text") or []) if t.strip())
    if not short_answers:
        return None
    return {
        "id": f"nq_test_{idx}",
        "orig_id": str(row.get("id") or ""),
        "question": question,
        "ground_truth": short_answers[0],
        "source": "natural_questions",
    }


def _normalize_hotpot(row: dict, idx: int, split: str) -> dict | None:
    """HotPotQA row → task record.

    hotpot_qa exposes ``id`` (hex hash like ``"5a7a06935542990198eaf050"``),
    which we use as ``orig_id``.  The ``level`` field (easy/medium/hard) is
    preserved as ``difficulty`` for stratified sampling.
    """
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not question or not answer:
        return None
    difficulty = (row.get("level") or "").strip().lower() or None
    return {
        "id": f"hotpot_{split}_{idx}",
        "orig_id": str(row.get("id") or ""),
        "question": question,
        "ground_truth": answer,
        "source": "hotpot_qa",
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _build_pool(rows, normalize_fn) -> list[dict]:
    pool: list[dict] = []
    for i, row in enumerate(rows):
        item = normalize_fn(row, i)
        if item is not None:
            pool.append(item)
    return pool


def _uniform_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:n]


def _stratified_sample(
    pool: list[dict],
    n: int,
    seed: int,
    key: str,
) -> list[dict]:
    """Sample ``n`` items from ``pool``, balanced across buckets of ``key``.

    Budget is divided evenly across buckets (remainder spread across the first
    buckets in shuffled order).  If a bucket has fewer items than its share,
    the shortfall is backfilled from remaining items.
    """
    rng = random.Random(seed)

    buckets: dict[object, list[dict]] = {}
    for item in pool:
        buckets.setdefault(item.get(key), []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets.keys())
    rng.shuffle(keys)
    per_bucket, remainder = divmod(n, len(keys))

    sampled: list[dict] = []
    leftover: list[dict] = []
    for i, k in enumerate(keys):
        target = per_bucket + (1 if i < remainder else 0)
        take = min(target, len(buckets[k]))
        sampled.extend(buckets[k][:take])
        leftover.extend(buckets[k][take:])

    if len(sampled) < n:
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    rng.shuffle(sampled)

    counts: dict[object, int] = {}
    for item in sampled:
        counts[item.get(key)] = counts.get(item.get(key), 0) + 1
    print(f"  Stratified by '{key}': {dict(sorted(counts.items(), key=lambda kv: str(kv[0])))}")

    return sampled


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows → {path}")


def _write_keys(rows: list[dict], path: Path) -> None:
    keys = [item["orig_id"] for item in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(keys)} keys  → {path}")


def read_keys(jsonl_path: str | Path) -> list[str]:
    """Return the ``orig_id`` values from a dataset JSONL file, in order.

    Use this to verify that two runs selected exactly the same questions:

        assert read_keys("data/nq/train.jsonl") == read_keys("/their/data/nq/train.jsonl")
        assert read_keys("data/hotpotqa/test.jsonl") == read_keys("/their/data/hotpotqa/test.jsonl")
    """
    keys: list[str] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keys.append(json.loads(line)["orig_id"])
    return keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Derive four independent seeds from the fixed master so that each output
    # file is an independent draw — not nested subsets of each other.
    master = random.Random(FIXED_SEED)
    seeds = {
        name: master.randint(0, 2**32 - 1)
        for name in ("nq_train", "nq_test", "hotpot_train", "hotpot_test")
    }

    # -- NQ Open train --------------------------------------------------------
    print("Loading nq_open train …")
    nq_train_rows = load_dataset(
        "google-research-datasets/nq_open", split="train", cache_dir=CACHE_DIR
    )
    nq_train_pool = _build_pool(nq_train_rows, _normalize_nq_train)
    print(f"  Pool: {len(nq_train_pool)} rows")
    nq_train = _uniform_sample(nq_train_pool, NUM_SAMPLES, seeds["nq_train"])
    _write_jsonl(nq_train, _DATA_ROOT / "nq" / "train.jsonl")
    _write_keys(nq_train, _DATA_ROOT / "nq" / "train_keys.json")

    # -- Natural Questions validation (used as test) --------------------------
    print("Loading natural_questions validation …")
    nq_test_rows = load_dataset(
        "natural_questions", split="validation", cache_dir=CACHE_DIR
    )
    nq_test_pool = _build_pool(nq_test_rows, _normalize_nq_test)
    print(f"  Pool: {len(nq_test_pool)} rows")
    nq_test = _uniform_sample(nq_test_pool, NUM_SAMPLES, seeds["nq_test"])
    _write_jsonl(nq_test, _DATA_ROOT / "nq" / "test.jsonl")
    _write_keys(nq_test, _DATA_ROOT / "nq" / "test_keys.json")

    # -- HotPotQA train -------------------------------------------------------
    print("Loading hotpot_qa distractor train …")
    hp_train_rows = load_dataset(
        "hotpot_qa", "distractor", split="train", cache_dir=CACHE_DIR
    )
    hp_train_pool = _build_pool(
        hp_train_rows,
        lambda row, idx: _normalize_hotpot(row, idx, "train"),
    )
    print(f"  Pool: {len(hp_train_pool)} rows")
    hp_train = _stratified_sample(hp_train_pool, NUM_SAMPLES, seeds["hotpot_train"], "difficulty")
    _write_jsonl(hp_train, _DATA_ROOT / "hotpotqa" / "train.jsonl")
    _write_keys(hp_train, _DATA_ROOT / "hotpotqa" / "train_keys.json")

    # -- HotPotQA validation (used as test) -----------------------------------
    print("Loading hotpot_qa distractor validation …")
    hp_test_rows = load_dataset(
        "hotpot_qa", "distractor", split="validation", cache_dir=CACHE_DIR
    )
    hp_test_pool = _build_pool(
        hp_test_rows,
        lambda row, idx: _normalize_hotpot(row, idx, "test"),
    )
    print(f"  Pool: {len(hp_test_pool)} rows")
    hp_test = _stratified_sample(hp_test_pool, NUM_SAMPLES, seeds["hotpot_test"], "difficulty")
    _write_jsonl(hp_test, _DATA_ROOT / "hotpotqa" / "test.jsonl")
    _write_keys(hp_test, _DATA_ROOT / "hotpotqa" / "test_keys.json")

    print("\nDone.  To verify a collaborator's run:")
    print("  from scripts.build_train import read_keys")
    print("  assert read_keys('data/nq/train.jsonl') == read_keys('/their/data/nq/train.jsonl')")


if __name__ == "__main__":
    main()
