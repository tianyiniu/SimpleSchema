"""Dataset registry — Natural Questions and HotPotQA, each with train/test splits.

See ``scripts/build_train.py`` for how the train JSONL files are
produced. Test splits are reserved for downstream evaluation and start
out empty; ``load_tasks`` returns ``[]`` for any missing file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.types import Task


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str  # "train" or "test"
    path: Path  # JSONL: one {id, question, ground_truth, ...} per line


_DATA_ROOT = Path("data")

DATASETS: dict[tuple[str, str], DatasetSpec] = {
    ("nq", "train"): DatasetSpec(
        name="nq", split="train",
        path=_DATA_ROOT / "nq" / "train.jsonl",
    ),
    ("nq", "test"): DatasetSpec(
        name="nq", split="test",
        path=_DATA_ROOT / "nq" / "test.jsonl",
    ),
    ("hotpotqa", "train"): DatasetSpec(
        name="hotpotqa", split="train",
        path=_DATA_ROOT / "hotpotqa" / "train.jsonl",
    ),
    ("hotpotqa", "test"): DatasetSpec(
        name="hotpotqa", split="test",
        path=_DATA_ROOT / "hotpotqa" / "test.jsonl",
    ),
    # Mixed datasets built by scripts/build_train.py: NQ + HotPotQA in
    # one JSONL. train_4 is the smoke-test sized version of train_300.
    ("train_300", "train"): DatasetSpec(
        name="train_300", split="train",
        path=_DATA_ROOT / "train_300" / "train.jsonl",
    ),
    ("train_300", "test"): DatasetSpec(
        name="train_300", split="test",
        path=_DATA_ROOT / "train_300" / "test.jsonl",
    ),
    ("train_4", "train"): DatasetSpec(
        name="train_4", split="train",
        path=_DATA_ROOT / "train_4" / "train.jsonl",
    ),
    ("train_4", "test"): DatasetSpec(
        name="train_4", split="test",
        path=_DATA_ROOT / "train_4" / "test.jsonl",
    ),
    # HotPotQA-only mixed dataset: 100 easy + 100 medium + 100 hard.
    ("train_hp_300", "train"): DatasetSpec(
        name="train_hp_300", split="train",
        path=_DATA_ROOT / "train_hp_300" / "train.jsonl",
    ),
    ("train_hp_300", "test"): DatasetSpec(
        name="train_hp_300", split="test",
        path=_DATA_ROOT / "train_hp_300" / "test.jsonl",
    ),
}

DATASET_CHOICES: list[str] = sorted({name for name, _ in DATASETS})
SPLIT_CHOICES: list[str] = ["train", "test"]


def get_dataset_spec(name: str, split: str) -> DatasetSpec:
    key = (name, split)
    if key not in DATASETS:
        raise ValueError(
            f"Unknown dataset/split '{name}/{split}'. "
            f"Datasets: {DATASET_CHOICES}, splits: {SPLIT_CHOICES}"
        )
    return DATASETS[key]


def load_tasks(spec: DatasetSpec, max_tasks: int | None = None) -> list[Task]:
    """Read a dataset's JSONL file and return a list of Task objects.

    Returns an empty list (with no error) if the file is missing so the
    codebase imports cleanly before data has been built.
    """
    if not spec.path.exists():
        return []
    tasks: list[Task] = []
    with open(spec.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(Task.model_validate(json.loads(line)))
            if max_tasks is not None and len(tasks) >= max_tasks:
                break
    return tasks
