"""Structured logging setup using structlog.

The console handler defaults to ``WARNING`` so terminal output stays
quiet during long runs — only failures, retries, empty-response warnings,
and other things you'd actually want to see scroll by. The optional file
handler is configured at ``INFO`` so the full structured trace is still
captured for post-hoc debugging.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import structlog
from tqdm.contrib.logging import logging_redirect_tqdm

# Module-level mirror of the configured console level so callers (e.g.
# the tqdm redirect helper) can restore it after tqdm replaces our
# stream handler with one that defaults to NOTSET.
_CONFIGURED_CONSOLE_LEVEL: int = logging.WARNING


def setup_logging(
    log_path: Path | None = None,
    file_level: int = logging.INFO,
    console_level: int = logging.WARNING,
) -> None:
    """Configure structlog → stdlib logging with split console/file levels.

    Args:
        log_path: Where to write the on-disk log. ``None`` disables the
            file handler entirely. Parent directories are created.
        file_level: Verbosity for the file handler. INFO captures the
            full structured event stream.
        console_level: Verbosity for the terminal handler. WARNING keeps
            terminal output quiet during long runs and only surfaces
            failures / retries / explicit warnings.
    """
    global _CONFIGURED_CONSOLE_LEVEL
    _CONFIGURED_CONSOLE_LEVEL = console_level

    # The structlog filter must let every event reach the handlers; each
    # handler then applies its own level. Use the looser of the two.
    filter_level = min(file_level, console_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    handlers: list[logging.Handler] = [console_handler]

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level)
        handlers.append(file_handler)

    # Reset root logger so repeated calls (e.g. in tests) don't stack
    # handlers.
    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(filter_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Render to string, then hand off to stdlib logging.
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(filter_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger."""
    return structlog.get_logger(name)  # type: ignore[return-value]


@contextlib.contextmanager
def tqdm_logging_redirect() -> Iterator[None]:
    """Run tqdm's logging redirect *and* preserve the console-level filter.

    ``tqdm.contrib.logging.logging_redirect_tqdm`` replaces the existing
    stdout StreamHandler with a ``_TqdmLoggingHandler`` so log lines
    appear above the bar instead of smearing it. That replacement is
    created at level NOTSET, which silently undoes the WARNING filter
    set in ``setup_logging`` and lets every INFO event leak to the
    terminal. This wrapper re-applies the configured console level to
    the new handler for the duration of the bar.
    """
    with logging_redirect_tqdm():
        for h in logging.getLogger().handlers:
            # The tqdm replacement is the only StreamHandler on root that
            # is not a FileHandler — file logging continues at INFO.
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(_CONFIGURED_CONSOLE_LEVEL)
        yield
