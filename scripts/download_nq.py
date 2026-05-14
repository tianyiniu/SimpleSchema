#!/usr/bin/env python3
"""Download Natural Questions validation split as test data.

NQ's public test split lacks gold answers, so we use the validation split
(which is the standard benchmark practice). Writes JSONL to
``data/test/nq.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

CACHE_DIR = "/nas-ssd2/tianyin4/cache/pretrained_models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/test/nq.jsonl"))
    args = parser.parse_args()

    ds = load_dataset("natural_questions", split="validation", cache_dir=CACHE_DIR)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            question = row.get("question", {}).get("text", "").strip()
            short_answers = row.get("annotations", {}).get("short_answers", [])
            answers: list[str] = []
            for sa in short_answers:
                answers.extend(sa.get("text", []))
            answers = [a.strip() for a in answers if a and a.strip()]
            if not question or not answers:
                continue
            item = {
                "id": f"nq_test_{i}",
                "question": question,
                "ground_truth": answers[0],
                "source": "natural_questions",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} questions to {args.out}")


if __name__ == "__main__":
    main()
