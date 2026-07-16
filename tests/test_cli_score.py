"""Tests for the `score` CLI command — the wiring in `cli.py`.

`score/__init__.py`'s own selection/persistence/failure-isolation behaviour is
covered in `tests/score/test_triage.py`; this file is about the CLI seam:
flags, exit codes, and the `runs` row it writes. The LLM boundary is faked at
`signalforge.llm.run_triage_batch` — never the real Anthropic API (CLAUDE.md
§8, NEVER rule 13).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection, upsert_item
from signalforge.llm import LlmError, TriageBatchResult, TriageResult
from signalforge.models import Item, SourceType

runner = CliRunner()

_LOG_LEVEL: list[str] = ["--log-level", "WARNING"]


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Mirrors `test_cli.py`'s fixture — the CLI owns global logging config."""
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.mkdir()
    (path / "interests.yaml").write_text(
        "priority_topics: [agents.mcp]\n"
        "interests: [python]\n"
        "stack: [python]\n"
        "learning_goals: []\n"
        "architecture_philosophy: 'Local-first.'\n"
        "ignore:\n"
        "  topics: [crypto]\n"
        "  people: []\n"
        "  repos: []\n"
        "thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,"
        " daily_max_items: 15}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


def _make_item(**overrides: object) -> Item:
    fields: dict[str, object] = {
        "source_id": "simonwillison",
        "source_type": SourceType.RSS,
        "external_id": "guid-1",
        "url": "https://simonwillison.net/post",
        "title": "MCP sampling lands everywhere",
        "summary": "A short feed summary.",
        "fetched_at": datetime(2026, 7, 16, 6, 0, tzinfo=UTC),
    }
    fields.update(overrides)
    return Item(**fields)  # type: ignore[arg-type]


def _seed_item(db_path: Path, **overrides: object) -> int:
    with connection(db_path) as conn:
        item_id, _ = upsert_item(conn, _make_item(**overrides))
    return item_id


def _invoke(config_dir: Path, db_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [*_LOG_LEVEL, "score", "--config-dir", str(config_dir), "--db", str(db_path), *extra],
    )


def _rows(db_path: Path, sql: str) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(conn.execute(sql).fetchall())


def _triage_result(**overrides: object) -> TriageResult:
    fields: dict[str, object] = {
        "triage": "keep",
        "signal": 4,
        "relevance": 4,
        "novelty": 3,
        "reasoning": "Solid working example with benchmarks.",
    }
    fields.update(overrides)
    return TriageResult(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_scores_a_pending_item_and_records_an_ok_run(
    config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item_id = _seed_item(db_path)
    batch_result = TriageBatchResult(
        results={item_id: _triage_result()}, input_tokens=500, output_tokens=80
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", lambda *a, **k: batch_result)

    result = _invoke(config_dir, db_path)

    assert result.exit_code == 0, result.output
    assert len(_rows(db_path, "SELECT * FROM scores")) == 1
    run = _rows(db_path, "SELECT * FROM runs WHERE kind = 'score'")[-1]
    assert run["status"] == "ok"
    assert (run["llm_input_tokens"], run["llm_output_tokens"]) == (500, 80)
    assert run["items_new"] == 0, "score writes no items rows"


def test_double_run_scores_nothing_new_and_spends_no_more_tokens(
    config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md §3's acceptance gate, at the CLI: a second `score` invocation
    is a true no-op — zero new `scores` rows, zero additional token spend."""
    item_id = _seed_item(db_path)
    batch_result = TriageBatchResult(
        results={item_id: _triage_result()}, input_tokens=500, output_tokens=80
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", lambda *a, **k: batch_result)

    first = _invoke(config_dir, db_path)
    assert first.exit_code == 0, first.output

    def _must_not_be_called(*args: object, **kwargs: object) -> TriageBatchResult:
        raise AssertionError("an already-scored item must never reach the LLM again")

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _must_not_be_called)
    second = _invoke(config_dir, db_path)

    assert second.exit_code == 0, second.output
    assert len(_rows(db_path, "SELECT * FROM scores")) == 1
    runs = _rows(db_path, "SELECT * FROM runs WHERE kind = 'score' ORDER BY id")
    assert [r["status"] for r in runs] == ["ok", "ok"]
    assert (runs[1]["llm_input_tokens"], runs[1]["llm_output_tokens"]) == (0, 0)


def test_dry_run_makes_no_llm_call_and_writes_no_run(
    config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_item(db_path)

    def _must_not_be_called(*args: object, **kwargs: object) -> TriageBatchResult:
        raise AssertionError("--dry-run must never call the LLM")

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _must_not_be_called)
    result = _invoke(config_dir, db_path, "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert "1 unscored" in result.output
    assert _rows(db_path, "SELECT * FROM runs") == []
    assert _rows(db_path, "SELECT * FROM scores") == []


def test_nothing_to_score_is_a_clean_ok_run(config_dir: Path, db_path: Path) -> None:
    result = _invoke(config_dir, db_path)
    assert result.exit_code == 0, result.output
    run = _rows(db_path, "SELECT * FROM runs WHERE kind = 'score'")[-1]
    assert run["status"] == "ok"


# --------------------------------------------------------------------------- #
# Failure isolation (CLAUDE.md §7, NEVER rule 12)
# --------------------------------------------------------------------------- #


def test_an_llm_batch_failure_is_recorded_and_exits_nonzero(
    config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_item(db_path)

    def _boom(*args: object, **kwargs: object) -> TriageBatchResult:
        raise LlmError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr("signalforge.llm.run_triage_batch", _boom)
    result = _invoke(config_dir, db_path)

    assert result.exit_code == 1
    run = _rows(db_path, "SELECT * FROM runs WHERE kind = 'score'")[-1]
    assert run["status"] == "failed"
    errors = json.loads(run["errors"])
    assert errors[0]["source_id"] == "*"
    assert errors[0]["error_type"] == "LlmError"
    assert _rows(db_path, "SELECT * FROM scores") == []


def test_a_partial_batch_failure_still_persists_the_good_items(
    config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_id = _seed_item(db_path)
    bad_id = _seed_item(db_path, external_id="guid-2", url="https://simonwillison.net/other")
    batch_result = TriageBatchResult(
        results={good_id: _triage_result()}, errors={bad_id: "missing from batch response"}
    )
    monkeypatch.setattr("signalforge.llm.run_triage_batch", lambda *a, **k: batch_result)

    result = _invoke(config_dir, db_path)

    assert result.exit_code == 0, result.output
    run = _rows(db_path, "SELECT * FROM runs WHERE kind = 'score'")[-1]
    assert run["status"] == "partial"
    errors = json.loads(run["errors"])
    assert errors[0]["source_id"] == str(bad_id)
    assert _rows(db_path, "SELECT * FROM scores")[0]["item_id"] == good_id


def test_a_runs_row_is_written_even_when_score_unscored_items_raises(
    config_dir: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md §3: no silent runs, even on an unexpected exception."""
    _seed_item(db_path)

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("bug")

    monkeypatch.setattr("signalforge.cli.score_unscored_items", _boom)
    result = _invoke(config_dir, db_path)

    assert result.exit_code != 0
    run = _rows(db_path, "SELECT * FROM runs WHERE kind = 'score'")[-1]
    assert run["status"] == "failed"
    assert run["finished_at"] is not None
    errors = json.loads(run["errors"])
    assert errors[0]["error_type"] == "RuntimeError"


def test_bad_config_dir_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["score", "--config-dir", str(tmp_path / "missing")])
    assert result.exit_code == 2
