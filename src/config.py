"""TOML config loader + lightweight .env reader.

Reads ``config.toml`` and ``personas.toml`` from the package root and
provides ``get_env_value`` for picking up API keys without a
python-dotenv dependency.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return _REPO_ROOT / "config.toml"


def default_personas_path() -> Path:
    return _REPO_ROOT / "personas.toml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the parsed contents of config.toml."""
    cfg_path = Path(path) if path is not None else default_config_path()
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def load_personas(path: str | Path | None = None) -> dict[str, str]:
    """Return a mapping of persona name -> system_prompt."""
    p = Path(path) if path is not None else default_personas_path()
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return {name: info["system_prompt"] for name, info in data["personas"].items()}


def resolve_run_settings(
    cfg: dict[str, Any],
    mode_override: str | None = None,
    corpus_override: str | None = None,
) -> tuple[str, str]:
    """Resolve effective (mode, corpus) from CLI overrides and TOML defaults.

    Train runs are forced to ``corpus = "static"``.
    """
    run_cfg = cfg.get("run", {})
    mode = mode_override or run_cfg.get("mode", "train")
    corpus = corpus_override or run_cfg.get("corpus", "static")
    if mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train' or 'test'; got {mode!r}")
    if corpus not in ("static", "live"):
        raise ValueError(f"corpus must be 'static' or 'live'; got {corpus!r}")
    if mode == "train":
        corpus = "static"
    return mode, corpus


def corpus_base_url(cfg: dict[str, Any]) -> str:
    return cfg.get("corpus_server", {}).get("base_url", "http://localhost:8000")


def get_env_value(key: str, env_path: Path | None = None) -> str | None:
    """Return the value of ``key`` from os.environ or the ``.env`` file."""
    value = os.getenv(key)
    if value:
        return value
    path = env_path or (_REPO_ROOT / ".env")
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, env_value = stripped.split("=", 1)
        if env_key.strip() != key:
            continue
        return env_value.strip().strip("'").strip('"') or None
    return None
