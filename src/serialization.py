"""Schema <-> JSON helpers + a durable JSON writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.types import Schema


def schema_to_dict(schema: Schema) -> dict:
    return schema.model_dump()


def schema_from_dict(data: dict) -> Schema:
    return Schema.model_validate(data)


def save_json_durable(data: Any, path: Path, indent: int = 4) -> None:
    """Write JSON and fsync so progress survives abrupt termination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
