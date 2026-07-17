"""Tests for the `signalforge digest` CLI command.

Scope: the seam where the CLI meets `db.py` reads and `report/daily.py`
rendering — not the template content itself (that's `tests/report/test_daily.py`'s
golden-file test). Every DB here is a throwaway under `tmp_path` (CLAUDE.md §8).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection, upsert_item
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


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    return tmp_path / "vault"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A minimal interests.yaml — `digest` reads `thresholds.daily_max_items`."""
    path = tmp_path / "config"
    path.mkdir()
    (path / "interests.yaml").write_text(
        "thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,"
        " daily_max_items: 15}\n",
        encoding="utf-8",
    )
    return path


def _insert_score(conn: sqlite3.Connection, item_id: int, *, triage: str = "keep") -> None:
    conn.execute(
        """
        INSERT INTO scores (
            item_id, triage, signal, relevance, novelty, reasoning,
            rubric_version, model, scored_at
        ) VALUES (?, ?, 4, 4, 3, 'A reason this item matters.', 'v1',
                  'claude-haiku-4-5', '2026-07-16T06:05:00+00:00')
        """,
        (item_id, triage),
    )


def _seed(db_path: Path, *, triage: str = "keep") -> None:
    with connection(db_path) as conn:
        item_id, _ = upsert_item(conn, make_item())
        _insert_score(conn, item_id, triage=triage)


def _invoke(db_path: Path, vault_dir: Path, config_dir: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "--log-level",
            "WARNING",
            "digest",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--vault-dir",
            str(vault_dir),
            "--date",
            "2026-07-16",
            *extra,
        ],
    )


def _runs(db_path: Path) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(conn.execute("SELECT * FROM runs ORDER BY id").fetchall())


def test_digest_writes_the_expected_file(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    path = vault_dir / "daily" / "2026-07-16.md"
    assert path.is_file()
    assert "A reason this item matters." in path.read_text(encoding="utf-8")


def test_digest_run_is_recorded_as_ok(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    _seed(db_path)

    assert _invoke(db_path, vault_dir, config_dir).exit_code == 0

    run = _runs(db_path)[-1]
    assert run["kind"] == "daily"
    assert run["status"] == "ok"
    assert run["finished_at"] is not None


def test_digest_with_no_scored_items_still_succeeds(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    path = vault_dir / "daily" / "2026-07-16.md"
    assert path.is_file()
    assert "No items cleared triage today" in path.read_text(encoding="utf-8")


def test_running_digest_twice_overwrites_the_same_file(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """The idempotency gate for `digest`, end to end through the CLI (CLAUDE.md §3)."""
    _seed(db_path)

    first = _invoke(db_path, vault_dir, config_dir)
    assert first.exit_code == 0, first.output
    path = vault_dir / "daily" / "2026-07-16.md"
    first_content = path.read_text(encoding="utf-8")

    second = _invoke(db_path, vault_dir, config_dir)
    assert second.exit_code == 0, second.output
    second_content = path.read_text(encoding="utf-8")

    assert second_content == first_content
    assert list((vault_dir / "daily").glob("2026-07-16*")) == [path]

    runs = _runs(db_path)
    assert [row["status"] for row in runs] == ["ok", "ok"]


def test_dry_run_writes_nothing(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir, "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert "A reason this item matters." in result.output
    assert not (vault_dir / "daily" / "2026-07-16.md").exists()


def test_dry_run_still_records_a_run(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    _seed(db_path)

    assert _invoke(db_path, vault_dir, config_dir, "--dry-run").exit_code == 0

    run = _runs(db_path)[-1]
    assert run["status"] == "ok"


def test_digest_defaults_to_todays_date_when_omitted(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    # `--date` is required by every other test for reproducibility; this one
    # checks the omitted-flag default lands on *some* file for today's UTC
    # date, without faking `datetime` itself — typer re-resolves `--date`'s
    # `datetime | None` annotation against `signalforge.cli`'s module globals
    # on every invocation (postponed evaluation), so swapping out `datetime`
    # there breaks its own argument parsing.
    import datetime as dt

    today = dt.datetime.now(dt.UTC).date().isoformat()

    result = runner.invoke(
        app,
        [
            "--log-level",
            "WARNING",
            "digest",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--vault-dir",
            str(vault_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (vault_dir / "daily" / f"{today}.md").is_file()


def test_killed_only_day_reports_zero_items_and_a_nonzero_killed_count(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    _seed(db_path, triage="kill")

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    content = (vault_dir / "daily" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "item_count: 0" in content
    assert "1 item(s) killed at triage" in content


def test_digest_reads_the_cap_from_interests_yaml(
    db_path: Path, vault_dir: Path, tmp_path: Path
) -> None:
    """The cap is config, not code (CLAUDE.md §4): `daily_max_items: 1` in
    interests.yaml must truncate a two-item day to one rendered item."""
    config_dir = tmp_path / "capped-config"
    config_dir.mkdir()
    (config_dir / "interests.yaml").write_text(
        "thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,"
        " daily_max_items: 1}\n",
        encoding="utf-8",
    )
    with connection(db_path) as conn:
        first_id, _ = upsert_item(conn, make_item())
        second_id, _ = upsert_item(
            conn, make_item(external_id="guid-2", url="https://example.com/second")
        )
        _insert_score(conn, first_id)
        _insert_score(conn, second_id)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    content = (vault_dir / "daily" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "item_count: 1" in content
    assert "kept_count: 2" in content
    assert "1 more kept item(s) not shown" in content


def test_digest_exits_2_on_a_missing_interests_config(
    db_path: Path, vault_dir: Path, tmp_path: Path
) -> None:
    result = _invoke(db_path, vault_dir, tmp_path / "missing")

    assert result.exit_code == 2
    assert not (vault_dir / "daily" / "2026-07-16.md").exists()


def test_digest_run_is_recorded_as_failed_when_rendering_blows_up(
    db_path: Path, vault_dir: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_path)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("template exploded")

    monkeypatch.setattr("signalforge.cli.render_digest", _boom)
    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 1
    run = _runs(db_path)[-1]
    assert run["status"] == "failed"
    errors = json.loads(run["errors"])
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["message"] == "template exploded"
