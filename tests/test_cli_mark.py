"""Tests for the `signalforge mark` CLI command (DESIGN §11, Phase 1 capture).

Scope: the seam where the CLI validates a verdict, checks the item exists, and
writes one `feedback` row through `db.record_feedback`. A mark is a human
action, not a pipeline run, so it must record NO `runs` row. Every DB is a
throwaway under `tmp_path` (CLAUDE.md §8).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection, get_feedback, upsert_item
from tests.conftest import make_item

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """See `tests/test_cli.py` — the CLI owns global logging config; restore it."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


def _seed_item(db_path: Path) -> int:
    with connection(db_path) as conn:
        item_id, _ = upsert_item(conn, make_item())
    return item_id


def _mark(db_path: Path, item_ref: str, verdict: str, *extra: str) -> Result:
    return runner.invoke(
        app,
        ["--log-level", "WARNING", "mark", item_ref, verdict, "--db", str(db_path), *extra],
    )


def _feedback_rows(db_path: Path, item_id: int) -> list[object]:
    with connection(db_path) as conn:
        return list(get_feedback(conn, item_id))


def _run_count(db_path: Path) -> int:
    with connection(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


def test_mark_records_one_feedback_row(db_path: Path) -> None:
    item_id = _seed_item(db_path)

    result = _mark(db_path, str(item_id), "useful")

    assert result.exit_code == 0, result.output
    assert "recorded" in result.output
    rows = _feedback_rows(db_path, item_id)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "useful"


def test_mark_writes_no_runs_row(db_path: Path) -> None:
    """A mark is a human action, not a pipeline run — no `runs` row, no tokens."""
    item_id = _seed_item(db_path)

    assert _mark(db_path, str(item_id), "useful").exit_code == 0

    assert _run_count(db_path) == 0


def test_marking_the_same_item_twice_is_idempotent_and_says_so(db_path: Path) -> None:
    item_id = _seed_item(db_path)

    first = _mark(db_path, str(item_id), "useful")
    second = _mark(db_path, str(item_id), "useful")

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "already marked" in second.output
    assert len(_feedback_rows(db_path, item_id)) == 1


def test_mark_rejects_an_unknown_verdict_with_exit_2(db_path: Path) -> None:
    item_id = _seed_item(db_path)

    result = _mark(db_path, str(item_id), "meh")

    assert result.exit_code == 2
    assert _feedback_rows(db_path, item_id) == []


def test_mark_on_a_nonexistent_item_exits_2_and_writes_nothing(db_path: Path) -> None:
    # A DB exists (migrated) but has no item 9999.
    with connection(db_path):
        pass

    result = _mark(db_path, "9999", "useful")

    assert result.exit_code == 2
    with connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_mark_stores_an_optional_note(db_path: Path) -> None:
    item_id = _seed_item(db_path)

    result = _mark(db_path, str(item_id), "missed", "--note", "should have surfaced")

    assert result.exit_code == 0, result.output
    assert _feedback_rows(db_path, item_id)[0]["note"] == "should have surfaced"
