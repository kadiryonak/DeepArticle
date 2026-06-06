"""
Unit tests for the disk cache utility (offline, deterministic).
"""

import os
import time

import pytest

import sys
sys.path.insert(0, '..')

from utils.cache import disk_cache, clear_cache

NAMESPACE = "test_cache_ns"


@pytest.fixture(autouse=True)
def _clean():
    clear_cache(NAMESPACE)
    os.environ.pop("DEEPARTICLE_NO_CACHE", None)
    yield
    clear_cache(NAMESPACE)
    os.environ.pop("DEEPARTICLE_NO_CACHE", None)


class TestDiskCache:
    def test_caches_result_and_skips_recompute(self):
        calls = {"n": 0}

        @disk_cache(namespace=NAMESPACE)
        def compute(x):
            calls["n"] += 1
            return x * 2

        assert compute(21) == 42
        assert compute(21) == 42  # served from cache
        assert calls["n"] == 1  # function body ran only once

    def test_distinct_args_are_cached_separately(self):
        @disk_cache(namespace=NAMESPACE)
        def echo(x):
            return {"value": x}

        assert echo("a") == {"value": "a"}
        assert echo("b") == {"value": "b"}

    def test_ttl_expiry_triggers_recompute(self):
        calls = {"n": 0}

        @disk_cache(namespace=NAMESPACE, ttl=0)
        def compute():
            calls["n"] += 1
            return calls["n"]

        compute()
        time.sleep(0.01)
        compute()
        assert calls["n"] == 2  # ttl=0 means always stale

    def test_disabled_via_env(self):
        calls = {"n": 0}

        @disk_cache(namespace=NAMESPACE)
        def compute():
            calls["n"] += 1
            return calls["n"]

        os.environ["DEEPARTICLE_NO_CACHE"] = "1"
        compute()
        compute()
        assert calls["n"] == 2  # caching bypassed

    def test_non_serializable_result_not_cached(self):
        calls = {"n": 0}

        @disk_cache(namespace=NAMESPACE)
        def compute():
            calls["n"] += 1
            return object()  # not JSON-serializable

        compute()
        compute()
        # Falls through to a normal call each time; no crash.
        assert calls["n"] == 2
