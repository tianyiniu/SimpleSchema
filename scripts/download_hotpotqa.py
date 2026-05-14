#!/usr/bin/env python3
"""Download HotPotQA validation split (distractor) as test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

CACHE_DIR = "/nas-ssd2/tianyin4/cache/pretrained_models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/test/hotpotqa.jsonl"))
    args = parser.parse_args()

    ds = load_dataset(
        "hotpot_qa", "distractor", split="validation", cache_dir=CACHE_DIR
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            item = {
                "id": f"hotpot_test_{i}",
                "question": question,
                "ground_truth": answer,
                "source": "hotpot_qa",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()
