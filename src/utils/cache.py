"""
Lightweight, dependency-free disk cache for API calls.

The search tools hit external APIs (ArXiv, Semantic Scholar, CrossRef, SCImago)
repeatedly — e.g. ArXiv looks up a citation count for *every* paper, so a single
query can fire 45+ HTTP requests. Caching these responses dramatically cuts
runtime and avoids rate-limit errors, since academic metadata barely changes
between runs.

Usage:
    from utils.cache import disk_cache

    @disk_cache(namespace="s2_citations", ttl=604800)  # 7 days
    def get_citations(arxiv_id: str) -> int:
        ...

Behavior:
  * Results are stored as JSON files under ``.cache/<namespace>/<key>.json``.
  * Only JSON-serializable return values are cached; anything else falls
    through to a normal (uncached) call.
  * Disable entirely with ``DEEPARTICLE_NO_CACHE=1``.
  * Override the default TTL (seconds) with ``DEEPARTICLE_CACHE_TTL``.
"""

import functools
import hashlib
import json
import os
import time
from typing import Any, Callable, Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Cache location. Defaults to ``.cache`` next to the project root, but can be
# redirected with ``DEEPARTICLE_CACHE_DIR`` (e.g. to a mounted Docker volume).
_CACHE_ROOT = os.getenv("DEEPARTICLE_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"
)

# 7 days — academic citation/metadata changes slowly.
DEFAULT_TTL = int(os.getenv("DEEPARTICLE_CACHE_TTL", str(7 * 24 * 60 * 60)))


def _cache_disabled() -> bool:
    return os.getenv("DEEPARTICLE_NO_CACHE", "").lower() in ("1", "true", "yes")


def _make_key(args: tuple, kwargs: dict) -> str:
    payload = json.dumps([args, sorted(kwargs.items())], default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def disk_cache(namespace: str, ttl: Optional[int] = None) -> Callable:
    """
    Decorator that caches a function's JSON-serializable return value on disk.

    Args:
        namespace: Sub-folder under ``.cache`` to isolate this function's keys.
        ttl: Time-to-live in seconds. Defaults to ``DEFAULT_TTL``.
    """
    effective_ttl = ttl if ttl is not None else DEFAULT_TTL

    def decorator(func: Callable) -> Callable:
        cache_dir = os.path.join(_CACHE_ROOT, namespace)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _cache_disabled():
                return func(*args, **kwargs)

            key = _make_key(args, kwargs)
            path = os.path.join(cache_dir, f"{key}.json")

            # Try to read a fresh cache entry.
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if time.time() - entry["ts"] < effective_ttl:
                    return entry["value"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Cache read failed for %s: %s", namespace, e)

            # Miss (or stale): call the function.
            value = func(*args, **kwargs)

            # Persist only if JSON-serializable.
            try:
                os.makedirs(cache_dir, exist_ok=True)
                serialized = json.dumps({"ts": time.time(), "value": value})
                with open(path, "w", encoding="utf-8") as f:
                    f.write(serialized)
            except (TypeError, ValueError):
                logger.debug("Result for %s not JSON-serializable; not cached", namespace)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Cache write failed for %s: %s", namespace, e)

            return value

        return wrapper

    return decorator


def clear_cache(namespace: Optional[str] = None) -> int:
    """
    Delete cached files. If ``namespace`` is given, only that folder is cleared.

    Returns the number of files removed.
    """
    target = os.path.join(_CACHE_ROOT, namespace) if namespace else _CACHE_ROOT
    removed = 0
    if not os.path.isdir(target):
        return 0
    for root, _dirs, files in os.walk(target):
        for name in files:
            if name.endswith(".json"):
                try:
                    os.remove(os.path.join(root, name))
                    removed += 1
                except OSError:
                    pass
    return removed
