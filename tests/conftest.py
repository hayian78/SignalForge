"""Shared fixtures for the models/config/db test suite.

Scope: `models.py`, `config.py`, `db.py` only. Ingestor fixtures (recorded HTTP
payloads via respx) live under `tests/ingest/`.

Every DB here is a throwaway under `tmp_path` — a test must never touch the real
`data/signalforge.db`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signalforge.db import connect, migrate
from signalforge.models import Item, SourceType

FIXED_FETCHED_AT = datetime(2026, 7, 16, 6, 0, 0, tzinfo=UTC)
FIXED_PUBLISHED_AT = datetime(2026, 7, 15, 12, 30, 0, tzinfo=UTC)


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_config_dir() -> Path:
    """The repo's real `config/` — the YAML that actually ships.

    Read-only. This is the one fixture that touches a tracked file rather than
    `tmp_path`: a shipped config that fails validation is a 6am cron failure, and
    a parse test moves that discovery to CI.
    """
    config_dir = REPO_ROOT / "config"
    if not config_dir.is_dir():  # pragma: no cover — the repo always ships config/
        pytest.fail(f"expected the shipped config directory at {config_dir}")
    return config_dir


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path to a scratch DB. Never the real `data/signalforge.db`."""
    return tmp_path / "signalforge.db"


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """An open, migrated connection to a scratch DB."""
    connection = connect(db_path)
    migrate(connection)
    try:
        yield connection
    finally:
        connection.close()


def make_item(**overrides: object) -> Item:
    """Build a valid `Item` with deterministic timestamps.

    Timestamps are fixed rather than `now()` so a double-run test can assert
    byte-for-byte identical DB state without the clock being the difference.
    """
    fields: dict[str, object] = {
        "source_id": "simonwillison",
        "source_type": SourceType.RSS,
        "external_id": "guid-1",
        "url": "https://simonwillison.net/2026/Jul/15/mcp-sampling/",
        "title": "MCP sampling lands everywhere",
        "author": "Simon Willison",
        "published_at": FIXED_PUBLISHED_AT,
        "fetched_at": FIXED_FETCHED_AT,
        "summary": "A short feed summary.",
    }
    fields.update(overrides)
    return Item(**fields)  # type: ignore[arg-type]


def dump_table(conn: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
    """Every row of `table` as plain tuples — the byte-for-byte state snapshot."""
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()  # noqa: S608
    return [tuple(row) for row in rows]
