"""Fixtures for the ingestor suite.

Every HTTP interaction here is a recorded payload served by `respx` — the suite
never touches a live network (CLAUDE.md §8, NEVER rule 13). Payloads live in
`tests/fixtures/` and are realistic captures, not toy XML: the RSS fixture
carries tracking params, a duplicate guid, and a linkless entry, because those
are what real feeds actually do.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from signalforge.config import SourcesConfig
from signalforge.ingest.base import HttpFetcher

FIXTURES = Path(__file__).parent.parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    """Read a recorded payload verbatim."""
    return (FIXTURES / name).read_bytes()


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Scratch stand-in for `data/http_cache/` — never the real one."""
    path = tmp_path / "http_cache"
    path.mkdir()
    return path


@pytest.fixture
async def fetcher(cache_dir: Path) -> AsyncIterator[HttpFetcher]:
    """An `HttpFetcher` pointed at a scratch cache, with retries kept fast.

    `max_attempts=2` keeps the retry path exercised without a test paying real
    exponential backoff.
    """
    instance = HttpFetcher(cache_dir=cache_dir, timeout=5.0, max_attempts=2)
    try:
        yield instance
    finally:
        await instance.aclose()


MAX_SUMMARY_CHARS = 4000
"""Test-side stand-in for `defaults.max_summary_chars`. Mirrors the shipped
`sources.yaml` value, but is deliberately a *test* constant: production reads it
from config (NEVER rule 6), so nothing under `src/` may hardcode it."""


def make_sources_config(**overrides: object) -> SourcesConfig:
    """Build a valid `SourcesConfig` in memory (no YAML file involved)."""
    data: dict[str, object] = {
        "defaults": {
            "fetch_timeout": 20,
            "min_hn_points": 80,
            "max_summary_chars": MAX_SUMMARY_CHARS,
        },
        "rss": [],
    }
    data.update(overrides)
    return SourcesConfig.model_validate(data)
