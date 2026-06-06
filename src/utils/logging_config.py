"""
Lightweight logging setup for DeepArticle.

The agents print human-friendly progress to stdout (that's the UX). This module
adds a proper logger for *diagnostics* — failed API calls, parse errors, etc. —
so problems are no longer silently swallowed by bare ``except`` blocks.

Usage:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.warning("ArXiv request failed: %s", err)

Control verbosity with the ``DEEPARTICLE_LOG_LEVEL`` env var
(DEBUG / INFO / WARNING / ERROR). Defaults to WARNING so normal runs stay quiet.
"""

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("DEEPARTICLE_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger("deeparticle")
    root.setLevel(level)
    # Avoid duplicate handlers if reconfigured.
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``deeparticle`` root."""
    _configure_root()
    short = name.split(".")[-1]
    return logging.getLogger(f"deeparticle.{short}")
