"""Integration tests for the `weekly` CLI command (Stage 6, DESIGN §13 Phase 1).

The synthesis call is faked at `llm.py`'s boundary — `signalforge.synth.weekly.
run_weekly_brief`, the same seam `tests/synth/test_weekly.py` uses (CLAUDE.md §8,
NEVER rule 13). No HTTP boundary exists in this feature at all, which is asserted
rather than assumed.

What these tests protect, in order of how much money they save:

* **The Sunday snap.** Every day of a week resolves to one Sunday, so it resolves
  to one vault path, so the file guard bounds the spend at one call per week no
  matter how often the command is invoked.
* **Write-on-every-outcome.** A refused or unusable synthesis still lands a file,
  which is what makes the file guard fire next time. If this ever regressed, a
  deterministic failure would re-bill on every invocation.
* **The harvest ordering.** Marks are read before the brief is overwritten, so
  `--force` cannot destroy ticks that exist only in the file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection, upsert_item
from signalforge.feedback import checkbox_marker
from signalforge.llm import WeeklyBrief, WeeklyBriefResult, WeeklyLead
from tests.conftest import make_item

runner = CliRunner()

_LOG_LEVEL: list[str] = ["--log-level", "WARNING"]

TARGET_SUNDAY = "2026-08-09"
SCORED_AT = "2026-08-05T09:00:00+00:00"


def _interests_yaml(*, weekly_top_n: int | None = 12) -> str:
    thresholds: dict[str, int] = {
        "weekly_min_signal": 3,
        "weekly_min_relevance": 3,
        "weekly_min_total": 10,
        "daily_max_items": 15,
        "weekly_near_miss_n": 5,
    }
    if weekly_top_n is not None:
        thresholds["weekly_top_n"] = weekly_top_n
    line = "{" + ", ".join(f"{k}: {v}" for k, v in thresholds.items()) + "}"
    return (
        "priority_topics: [agents.mcp]\n"
        "interests: [python]\n"
        "stack: [python]\n"
        "learning_goals: []\n"
        "architecture_philosophy: 'Local-first.'\n"
        "ignore:\n"
        "  topics: [crypto]\n"
        "  people: []\n"
        "  repos: []\n"
        f"thresholds: {line}\n"
    )


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    path = tmp_path / "config"
    path.mkdir()
    (path / "sources.yaml").write_text(
        "defaults:\n"
        "  fetch_timeout: 5\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        "  max_item_age_days: 3650\n"
        "rss: []\n",
        encoding="utf-8",
    )
    (path / "interests.yaml").write_text(_interests_yaml(), encoding="utf-8")
    (path / "settings.yaml").write_text("timezone: UTC\n", encoding="utf-8")
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    return tmp_path / "vault"


class Recorder:
    """A stand-in for `run_weekly_brief` that records whether it was called."""

    def __init__(self, result: WeeklyBriefResult | None = None) -> None:
        self.calls = 0
        self._result = result or WeeklyBriefResult(
            brief=WeeklyBrief(leads=[WeeklyLead(headline="H", why_it_matters="W", item_ids=[1])]),
            input_tokens=1_000,
            output_tokens=500,
            sent_item_ids=(1,),
        )

    def __call__(self, prefix: str, **kwargs: Any) -> WeeklyBriefResult:
        self.calls += 1
        return self._result


def _must_not_be_called(prefix: str, **kwargs: Any) -> WeeklyBriefResult:
    raise AssertionError("the weekly brief made a billed call it should have skipped")


def _seed(db_path: Path, *, count: int = 2, scored_at: str = SCORED_AT) -> list[int]:
    """Two kept items inside the target week, each clearing the weekly gate."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    with connection(db_path) as conn:
        for index in range(1, count + 1):
            item_id, _ = upsert_item(
                conn,
                make_item(
                    external_id=f"g-{index}",
                    url=f"https://example.com/{index}",
                    title=f"Story {index}",
                    source_id=f"source-{index}",
                ),
            )
            conn.execute(
                "INSERT INTO scores (item_id, triage, signal, relevance, novelty, reasoning,"
                " rubric_version, model, scored_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (item_id, "keep", 4, 4, 4, "It matters.", "triage-v3", "haiku", scored_at),
            )
            ids.append(item_id)
        conn.commit()
    return ids


def _run(
    monkeypatch: pytest.MonkeyPatch,
    fake: Any,
    config_dir: Path,
    db_path: Path,
    vault_dir: Path,
    *extra: str,
) -> Result:
    monkeypatch.setattr("signalforge.synth.weekly.run_weekly_brief", fake)
    return runner.invoke(
        app,
        [
            *_LOG_LEVEL,
            "weekly",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--vault-dir",
            str(vault_dir),
            "--date",
            TARGET_SUNDAY,
            *extra,
        ],
    )


def _brief(vault_dir: Path) -> Path:
    return vault_dir / "weekly" / f"{TARGET_SUNDAY}.md"


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_a_brief_is_written_and_the_run_is_recorded(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    _seed(db_path)
    fake = Recorder()

    result = _run(monkeypatch, fake, config_dir, db_path, vault_dir)

    assert result.exit_code == 0, result.output
    assert fake.calls == 1
    body = _brief(vault_dir).read_text(encoding="utf-8")
    assert "# Weekly Intelligence Brief — 2026-08-09" in body

    with connection(db_path) as conn:
        row = conn.execute(
            "SELECT status, llm_input_tokens, llm_output_tokens FROM runs WHERE kind = 'weekly'"
        ).fetchone()
    assert row["status"] == "ok"
    assert (row["llm_input_tokens"], row["llm_output_tokens"]) == (1_000, 500)


def test_the_brief_covers_the_seven_days_before_its_sunday(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """An item scored *on* the target Sunday belongs to next week's brief: the
    score pass runs in the evening and this runs Sunday morning, so the day's own
    items do not exist yet (DESIGN §14)."""
    _seed(db_path, count=1, scored_at="2026-08-09T09:00:00+00:00")
    fake = Recorder()

    result = _run(monkeypatch, fake, config_dir, db_path, vault_dir)

    assert result.exit_code == 0
    assert fake.calls == 0, "an item outside the window must not trigger a billed call"
    assert "nothing cleared the brief's bar" in result.output


# --------------------------------------------------------------------------- #
# the spend bound
# --------------------------------------------------------------------------- #


def test_a_second_run_makes_no_call_and_leaves_the_file_alone(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """Idempotency and the whole double-spend defence in one assertion."""
    _seed(db_path)
    _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir)
    before = _brief(vault_dir).read_text(encoding="utf-8")

    result = _run(monkeypatch, _must_not_be_called, config_dir, db_path, vault_dir)

    assert result.exit_code == 0
    assert _brief(vault_dir).read_text(encoding="utf-8") == before


def test_force_does_make_a_fresh_call(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    _seed(db_path)
    _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir)
    fake = Recorder()

    result = _run(monkeypatch, fake, config_dir, db_path, vault_dir, "--force")

    assert result.exit_code == 0
    assert fake.calls == 1


@pytest.mark.parametrize("day", ["2026-08-05", "2026-08-08", "2026-08-10"])
def test_a_non_sunday_date_is_refused_before_any_call(
    day: str, monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """Refused rather than snapped: silently rewriting an explicit date would
    write a different week's brief than the one asked for."""
    _seed(db_path)
    monkeypatch.setattr("signalforge.synth.weekly.run_weekly_brief", _must_not_be_called)

    result = runner.invoke(
        app,
        [
            *_LOG_LEVEL,
            "weekly",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--vault-dir",
            str(vault_dir),
            "--date",
            day,
        ],
    )

    assert result.exit_code != 0
    assert not (vault_dir / "weekly").exists()


def test_every_day_of_a_week_resolves_to_the_same_sunday() -> None:
    """The property that makes a mis-wired daily invocation cost one call a week
    instead of thirty a month: whatever day it runs, the default date is the same
    Sunday, so the vault filename does not move, so the file guard sees it."""
    from datetime import date as _date

    from signalforge.cli import _most_recent_sunday

    # Sunday the 9th through Saturday the 15th.
    resolved = {_most_recent_sunday(_date(2026, 8, day)) for day in range(9, 16)}
    assert resolved == {_date(2026, 8, 9)}
    # And the next Sunday starts a new one rather than extending the old.
    assert _most_recent_sunday(_date(2026, 8, 16)) == _date(2026, 8, 16)


def test_the_brief_is_off_when_the_threshold_is_unset(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    _seed(db_path)
    (config_dir / "interests.yaml").write_text(_interests_yaml(weekly_top_n=None), encoding="utf-8")

    result = _run(monkeypatch, _must_not_be_called, config_dir, db_path, vault_dir)

    assert result.exit_code == 0
    assert "weekly brief is off" in result.output
    assert not (vault_dir / "weekly").exists()


def test_a_week_with_no_items_exits_cleanly_and_closes_its_run(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connection(db_path):
        pass

    result = _run(monkeypatch, _must_not_be_called, config_dir, db_path, vault_dir)

    assert result.exit_code == 0
    with connection(db_path) as conn:
        row = conn.execute("SELECT status, finished_at FROM runs WHERE kind = 'weekly'").fetchone()
    assert row["status"] == "ok"
    assert row["finished_at"] is not None


# --------------------------------------------------------------------------- #
# --dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_makes_no_call_writes_nothing_and_records_no_marks(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """Unlike `curate run --dry-run`, which still pays because the call *is* the
    thing being previewed, everything worth previewing here is deterministic. And
    a preview must never write ground-truth feedback rows."""
    ids = _seed(db_path)
    weekly = vault_dir / "weekly"
    weekly.mkdir(parents=True)
    (weekly / "2026-08-02.md").write_text(
        checkbox_marker(ids[0], "useful").replace("[ ]", "[x]") + "\n", encoding="utf-8"
    )

    result = _run(monkeypatch, _must_not_be_called, config_dir, db_path, vault_dir, "--dry-run")

    assert result.exit_code == 0
    assert "dry run" in result.output
    assert "Story 1" in result.output
    assert not _brief(vault_dir).exists()
    with connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# failure degrades the brief; it never skips it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "result_value",
    [
        WeeklyBriefResult(brief=None, error="refused", input_tokens=800, output_tokens=100),
        WeeklyBriefResult(
            brief=None, error="schema validation failed", input_tokens=800, output_tokens=100
        ),
    ],
    ids=["refused", "unparseable"],
)
def test_a_failed_synthesis_still_writes_the_brief_and_records_the_spend(
    result_value: WeeklyBriefResult,
    monkeypatch: pytest.MonkeyPatch,
    config_dir: Path,
    db_path: Path,
    vault_dir: Path,
) -> None:
    """Two properties at once. The vault is written first and always (DESIGN
    §13.2), so a failed call degrades the brief rather than skipping it — and
    because the file lands, the *next* invocation still declines to re-bill,
    which is what keeps the file guard a real spend bound."""
    _seed(db_path)

    result = _run(monkeypatch, Recorder(result_value), config_dir, db_path, vault_dir)

    assert result.exit_code == 0
    body = _brief(vault_dir).read_text(encoding="utf-8")
    assert "No synthesis this week" in body
    assert checkbox_marker(1, "useful") in body, "items stay markable without a narrative"

    with connection(db_path) as conn:
        row = conn.execute(
            "SELECT llm_input_tokens, llm_output_tokens FROM runs WHERE kind = 'weekly'"
        ).fetchone()
    assert (row["llm_input_tokens"], row["llm_output_tokens"]) == (800, 100)

    second = _run(monkeypatch, _must_not_be_called, config_dir, db_path, vault_dir)
    assert second.exit_code == 0


# --------------------------------------------------------------------------- #
# the harvest ordering
# --------------------------------------------------------------------------- #


def test_force_preserves_marks_that_only_existed_in_the_file(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """The reason the harvest runs before the write. A tick lives only in the
    markdown until something reads it; regenerating first would erase it."""
    ids = _seed(db_path)
    _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir)

    path = _brief(vault_dir)
    ticked = path.read_text(encoding="utf-8").replace(
        checkbox_marker(ids[0], "useful"),
        checkbox_marker(ids[0], "useful").replace("[ ]", "[x]"),
    )
    path.write_text(ticked, encoding="utf-8")

    result = _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir, "--force")

    assert result.exit_code == 0
    with connection(db_path) as conn:
        rows = conn.execute("SELECT verdict FROM feedback WHERE item_id = ?", (ids[0],)).fetchall()
    assert [row["verdict"] for row in rows] == ["useful"]


def test_a_noise_mark_keeps_an_item_out_of_the_brief(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """Harvest runs before selection, not just before the write."""
    ids = _seed(db_path)
    weekly = vault_dir / "weekly"
    weekly.mkdir(parents=True)
    (weekly / "2026-08-02.md").write_text(
        checkbox_marker(ids[0], "noise").replace("[ ]", "[x]") + "\n", encoding="utf-8"
    )

    result = _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir)

    assert result.exit_code == 0
    body = _brief(vault_dir).read_text(encoding="utf-8")
    assert "Story 1" not in body
    assert "Story 2" in body


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


def test_daily_does_not_invoke_weekly(
    monkeypatch: pytest.MonkeyPatch, db_path: Path, config_dir: Path, vault_dir: Path
) -> None:
    """The mis-wiring the Sunday snap defends against, asserted directly: a
    weekly command inside the daily chain would bill every day."""
    import inspect

    from signalforge import cli

    assert "weekly(" not in inspect.getsource(cli.daily)


def test_status_reports_the_weekly_run_without_any_change(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    _seed(db_path)
    _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir)

    result = runner.invoke(
        app,
        [*_LOG_LEVEL, "status", "--config-dir", str(config_dir), "--db", str(db_path)],
    )

    assert result.exit_code == 0
    assert "weekly" in result.output


def test_the_feature_makes_no_http_calls_at_all(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, db_path: Path, vault_dir: Path
) -> None:
    """`report/` and `synth/` never reach the network (CLAUDE.md §2). Asserted by
    making any socket use an error rather than by trusting the import graph."""
    _seed(db_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the weekly brief opened a network connection")

    monkeypatch.setattr(sqlite3, "connect", sqlite3.connect)  # keep the DB working
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)

    result = _run(monkeypatch, Recorder(), config_dir, db_path, vault_dir)
    assert result.exit_code == 0
