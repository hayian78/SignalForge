"""Tests for the CLI — the seam where config, `ingest/`, and `db.py` meet.

Every HTTP interaction is served by `respx`; the suite never touches a live
network and never opens the real `data/signalforge.db` (CLAUDE.md §8, NEVER
rules 13 and 8). Each test builds a throwaway `sources.yaml`, DB, and cache
under `tmp_path`.

What is worth testing here is not "does typer parse the flag" but the four
invariants the CLI is the *only* holder of:

* a double run adds zero rows (DESIGN §16's acceptance gate);
* a healthy source that yielded no items still earns its 304;
* one source's write failure costs only that source;
* a `runs` row exists even when the run explodes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection, upsert_item
from signalforge.ingest import IngestRun
from signalforge.ingest.base import ValidatorStore
from signalforge.models import Item

runner = CliRunner()


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Undo the CLI's `logging.basicConfig(force=True)` after every test.

    The CLI legitimately owns global logging config — it is the process entry
    point. A test invoking it must not leave that config behind for the next
    test module, so the root logger's level and handlers are snapshotted and
    put back.
    """
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


FEED_A = "https://example.com/a/feed.xml"
FEED_B = "https://example.com/b/feed.xml"

FEED_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>{title}</title>
  <entry>
    <title>{entry_title}</title>
    <link href="{link}"/>
    <id>{guid}</id>
    <updated>2026-07-15T12:30:00Z</updated>
    <summary>A short summary of the post.</summary>
  </entry>
</feed>
"""

EMPTY_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Quiet blog</title>
</feed>
"""

TWO_ENTRY_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>alpha blog</title>
  <entry>
    <title>alpha post one</title>
    <link href="https://example.com/alpha/post-1"/>
    <id>urn:alpha:1</id>
    <updated>2026-07-15T12:30:00Z</updated>
    <summary>First.</summary>
  </entry>
  <entry>
    <title>alpha post two</title>
    <link href="https://example.com/alpha/post-2"/>
    <id>urn:alpha:2</id>
    <updated>2026-07-15T13:30:00Z</updated>
    <summary>Second.</summary>
  </entry>
</feed>
"""


def _feed(name: str) -> str:
    return FEED_TEMPLATE.format(
        title=f"{name} blog",
        entry_title=f"{name} post",
        link=f"https://example.com/{name}/post-1",
        guid=f"urn:{name}:1",
    )


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A `config/` holding only RSS sources — GitHub and HN are omitted so no
    test can accidentally depend on their query shapes."""
    path = tmp_path / "config"
    path.mkdir()
    (path / "sources.yaml").write_text(
        "defaults:\n"
        "  fetch_timeout: 5\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        "rss:\n"
        f"  - id: alpha\n    url: {FEED_A}\n"
        f"  - id: beta\n    url: {FEED_B}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "http_cache"


# The CLI defaults to INFO. These tests pin WARNING because of a live bug in a
# module this suite does not own: `db.py::migrate` logs
# `extra={"version": ..., "name": ...}`, and `name` is a reserved `LogRecord`
# attribute, so stdlib logging raises
# `KeyError: "Attempt to overwrite 'name' in LogRecord"` whenever a migration is
# applied with INFO enabled — i.e. on every fresh database. Pinning WARNING here
# keeps these tests about the CLI rather than about that bug; the bug itself is
# reported against `db.py` and must be fixed there (`"name"` → `"migration"`).
_LOG_LEVEL: list[str] = ["--log-level", "WARNING"]


def _invoke(config_dir: Path, db_path: Path, cache_dir: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            *_LOG_LEVEL,
            "ingest",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--cache-dir",
            str(cache_dir),
            *extra,
        ],
    )


def _rows(db_path: Path, sql: str) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(conn.execute(sql).fetchall())


def _validator_files(cache_dir: Path) -> set[str]:
    """Committed validator sidecars, keyed by their source directory."""
    return {path.parent.parent.name for path in cache_dir.rglob("_meta/*.json")}


@pytest.fixture
def both_feeds_ok() -> Iterator[None]:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(
            return_value=httpx.Response(200, text=_feed("alpha"), headers={"etag": '"a-1"'})
        )
        mock.get(FEED_B).mock(
            return_value=httpx.Response(200, text=_feed("beta"), headers={"etag": '"b-1"'})
        )
        yield


# --------------------------------------------------------------------------- #
# The acceptance gate: run it twice, get zero new rows
# --------------------------------------------------------------------------- #


def test_double_run_adds_no_rows(
    config_dir: Path, db_path: Path, cache_dir: Path, both_feeds_ok: None
) -> None:
    """DESIGN §16's gate, end to end through the CLI.

    The second run is served an unconditional 200 with the *same* payload rather
    than a 304, which is the harder case: it proves `upsert_item` dedupes the
    items, not merely that a 304 hid them.
    """
    first = _invoke(config_dir, db_path, cache_dir)
    assert first.exit_code == 0, first.output

    before = _rows(db_path, "SELECT * FROM items")
    assert len(before) == 2

    second = _invoke(config_dir, db_path, cache_dir)
    assert second.exit_code == 0, second.output

    after = _rows(db_path, "SELECT * FROM items")
    assert [tuple(row) for row in after] == [tuple(row) for row in before]

    runs = _rows(db_path, "SELECT * FROM runs ORDER BY id")
    assert [(row["kind"], row["status"], row["items_new"]) for row in runs] == [
        ("ingest", "ok", 2),
        ("ingest", "ok", 0),
    ]


def test_second_run_sends_conditional_headers_and_a_304_is_not_an_error(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    """The committed validator is actually *used*, and a 304 run is still `ok`."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(
            return_value=httpx.Response(200, text=_feed("alpha"), headers={"etag": '"a-1"'})
        )
        mock.get(FEED_B).mock(
            return_value=httpx.Response(200, text=_feed("beta"), headers={"etag": '"b-1"'})
        )
        assert _invoke(config_dir, db_path, cache_dir).exit_code == 0

        seen: list[str | None] = []

        def _record(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("if-none-match"))
            return httpx.Response(304)

        mock.get(FEED_A).mock(side_effect=_record)
        mock.get(FEED_B).mock(side_effect=_record)
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code == 0, result.output
    assert sorted(filter(None, seen)) == ['"a-1"', '"b-1"']
    assert len(_rows(db_path, "SELECT * FROM items")) == 2
    assert _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]["status"] == "ok"


# --------------------------------------------------------------------------- #
# Validator commit
# --------------------------------------------------------------------------- #


def test_healthy_source_with_zero_items_still_commits_its_validator(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    """The trap this CLI is built to avoid.

    `beta` 200s with a perfectly good ETag and no entries, so it appears in no
    item list and is absent from `run.source_ids`. Committing off `source_ids`
    would leave it uncommitted forever and refetch it unconditionally on every
    run — silently defeating DESIGN §7's "most daily RSS fetches return 304".
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(
            return_value=httpx.Response(200, text=_feed("alpha"), headers={"etag": '"a-1"'})
        )
        mock.get(FEED_B).mock(
            return_value=httpx.Response(200, text=EMPTY_FEED, headers={"etag": '"b-empty"'})
        )
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code == 0, result.output
    assert {row["source_id"] for row in _rows(db_path, "SELECT * FROM items")} == {"alpha"}
    # beta produced nothing and failed at nothing: it is a success, and its
    # validator must be durable.
    assert _validator_files(cache_dir) == {"alpha", "beta"}
    assert ValidatorStore(cache_dir).read("beta", _only_key(cache_dir, "beta")) == {
        "etag": '"b-empty"'
    }


def _only_key(cache_dir: Path, source_id: str) -> str:
    (path,) = (cache_dir / source_id / "_meta").glob("*.json")
    return path.stem


def test_dry_run_writes_nothing_at_all(
    config_dir: Path, db_path: Path, cache_dir: Path, both_feeds_ok: None
) -> None:
    """`--dry-run` fetches and reports. No rows, no `runs` row, no validators."""
    result = _invoke(config_dir, db_path, cache_dir, "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert not db_path.exists()
    assert _validator_files(cache_dir) == set()


def test_dry_run_does_not_earn_a_304_on_the_next_real_run(
    config_dir: Path, db_path: Path, cache_dir: Path, both_feeds_ok: None
) -> None:
    """The consequence that makes the previous test matter: a dry run must not
    let the following real run 304 past items it never stored."""
    assert _invoke(config_dir, db_path, cache_dir, "--dry-run").exit_code == 0
    assert _invoke(config_dir, db_path, cache_dir).exit_code == 0
    assert len(_rows(db_path, "SELECT * FROM items")) == 2


# --------------------------------------------------------------------------- #
# Failure isolation at the persist seam
# --------------------------------------------------------------------------- #


def test_one_sources_write_failure_costs_only_that_source(
    config_dir: Path,
    db_path: Path,
    cache_dir: Path,
    both_feeds_ok: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`beta`'s upserts raise. `alpha` must still persist *and* commit.

    This is CLAUDE.md §7 / NEVER rule 12 at the persist seam: the run reports
    `partial`, records the failure, keeps alpha's 304, and withholds beta's so
    beta refetches next run rather than losing its items to a 304.
    """

    def _flaky(conn: sqlite3.Connection, item: Item) -> tuple[int, bool]:
        if item.source_id == "beta":
            raise sqlite3.OperationalError("disk I/O error")
        return upsert_item(conn, item)

    monkeypatch.setattr("signalforge.cli.upsert_item", _flaky)
    result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code == 0, result.output

    assert {row["source_id"] for row in _rows(db_path, "SELECT * FROM items")} == {"alpha"}
    assert _validator_files(cache_dir) == {"alpha"}

    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["status"] == "partial"
    assert run["items_new"] == 1
    errors = json.loads(run["errors"])
    assert [error["source_id"] for error in errors] == ["beta"]
    assert errors[0]["error_type"] == "OperationalError"


def test_a_fetch_failure_leaves_the_other_source_intact(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    """The same isolation one layer up: a 404 feed is a recorded error, not a run."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(
            return_value=httpx.Response(200, text=_feed("alpha"), headers={"etag": '"a-1"'})
        )
        mock.get(FEED_B).mock(return_value=httpx.Response(404))
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code == 0, result.output
    assert {row["source_id"] for row in _rows(db_path, "SELECT * FROM items")} == {"alpha"}
    assert _validator_files(cache_dir) == {"alpha"}
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["status"] == "partial"
    assert json.loads(run["errors"])[0]["source_id"] == "beta"


def test_a_quiet_304_day_with_one_dead_feed_is_partial_not_failed(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    """The steady state must not read as catastrophe.

    DESIGN §7's design goal is that every feed 304s and yields nothing, so a
    normal quiet morning persists zero items. Grading `failed` on "no items"
    would mark that morning as a total loss the moment any one source broke —
    cron would mail a failure daily, and the day it mattered would look
    identical. `failed` means *nothing worked*, not *nothing was new*.
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(
            return_value=httpx.Response(200, text=_feed("alpha"), headers={"etag": '"a-1"'})
        )
        mock.get(FEED_B).mock(return_value=httpx.Response(404))
        assert _invoke(config_dir, db_path, cache_dir).exit_code == 0

        # Day two: alpha is perfectly healthy and 304s; beta is still dead.
        mock.get(FEED_A).mock(return_value=httpx.Response(304))
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code == 0, result.output
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["items_new"] == 0
    assert run["status"] == "partial"


def test_a_total_failure_is_recorded_as_failed_and_exits_nonzero(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(return_value=httpx.Response(404))
        mock.get(FEED_B).mock(return_value=httpx.Response(404))
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code == 1
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["status"] == "failed"
    assert len(json.loads(run["errors"])) == 2


# --------------------------------------------------------------------------- #
# No silent runs
# --------------------------------------------------------------------------- #


def test_a_runs_row_is_written_even_when_the_run_raises(
    config_dir: Path, db_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md §3: no silent runs. A crash still closes its `runs` row.

    Faked at the `ingest_all` boundary because the point is the `finally`, not
    any particular way of exploding.
    """

    async def _boom(*args: object, **kwargs: object) -> IngestRun:
        raise RuntimeError("the network fell over")

    monkeypatch.setattr("signalforge.cli.ingest_all", _boom)
    result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code != 0
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["status"] == "failed"
    assert run["finished_at"] is not None
    errors = json.loads(run["errors"])
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["message"] == "the network fell over"


def test_keyboard_interrupt_still_closes_the_run(
    config_dir: Path, db_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C'd cron run is still a run that happened."""

    async def _interrupt(*args: object, **kwargs: object) -> IngestRun:
        raise KeyboardInterrupt

    monkeypatch.setattr("signalforge.cli.ingest_all", _interrupt)
    result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code != 0
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["status"] == "failed"
    assert json.loads(run["errors"])[0]["error_type"] == "KeyboardInterrupt"


def test_an_interrupt_mid_persist_keeps_the_errors_already_recorded(
    config_dir: Path, db_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C must not erase the failures the run had already found.

    `beta` 404s, so `ingest_all` records a `FetchError` before persistence even
    begins. The interrupt then lands mid-upsert. `runs.errors` is the only
    durable record that a source is broken (DESIGN §7), and a run that dies is
    exactly when the user most needs to know what was already broken — so beta's
    error must reach the `runs` row alongside the interrupt, not be discarded
    because the run never reached its happy path.
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(return_value=httpx.Response(200, text=_feed("alpha")))
        mock.get(FEED_B).mock(return_value=httpx.Response(404))

        def _interrupt(conn: sqlite3.Connection, item: Item) -> tuple[int, bool]:
            raise KeyboardInterrupt

        monkeypatch.setattr("signalforge.cli.upsert_item", _interrupt)
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code != 0
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    errors = json.loads(run["errors"])
    by_source = {error["source_id"]: error["error_type"] for error in errors}
    # The pre-existing fetch failure survived, and the interrupt is recorded too.
    assert by_source == {"beta": "FetchError", "*": "KeyboardInterrupt"}
    assert run["finished_at"] is not None


def test_an_interrupt_after_some_rows_landed_is_partial_and_counts_them(
    config_dir: Path, db_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows that `upsert_item` committed are real, interrupt or not.

    The same reasoning `_persist` applies to `items_new` has to reach `status`:
    a run that stored one of two items before dying accomplished something, so
    it is `partial`. Reporting `failed` would say the opposite of what the
    database holds.
    """
    calls = {"n": 0}

    def _one_then_interrupt(conn: sqlite3.Connection, item: Item) -> tuple[int, bool]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return upsert_item(conn, item)

    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(return_value=httpx.Response(200, text=TWO_ENTRY_FEED))
        mock.get(FEED_B).mock(return_value=httpx.Response(200, text=EMPTY_FEED))
        monkeypatch.setattr("signalforge.cli.upsert_item", _one_then_interrupt)
        result = _invoke(config_dir, db_path, cache_dir)

    assert result.exit_code != 0
    assert len(_rows(db_path, "SELECT * FROM items")) == 1
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert run["items_new"] == 1, "the row that landed must be counted"
    assert run["status"] == "partial"


def test_ingest_records_zero_llm_tokens(
    config_dir: Path, db_path: Path, cache_dir: Path, both_feeds_ok: None
) -> None:
    """Phase 0 ingest calls no LLM (CLAUDE.md §2) — the counters prove it."""
    assert _invoke(config_dir, db_path, cache_dir).exit_code == 0
    run = _rows(db_path, "SELECT * FROM runs ORDER BY id")[-1]
    assert (run["llm_input_tokens"], run["llm_output_tokens"]) == (0, 0)


# --------------------------------------------------------------------------- #
# --source
# --------------------------------------------------------------------------- #


def test_source_flag_runs_only_that_source(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route_a = mock.get(FEED_A).mock(
            return_value=httpx.Response(200, text=_feed("alpha"), headers={"etag": '"a-1"'})
        )
        route_b = mock.get(FEED_B).mock(return_value=httpx.Response(200, text=_feed("beta")))
        result = _invoke(config_dir, db_path, cache_dir, "--source", "alpha")

    assert result.exit_code == 0, result.output
    assert route_a.called
    assert not route_b.called
    assert {row["source_id"] for row in _rows(db_path, "SELECT * FROM items")} == {"alpha"}


def test_unknown_source_is_rejected(config_dir: Path, db_path: Path, cache_dir: Path) -> None:
    result = _invoke(config_dir, db_path, cache_dir, "--source", "nope")
    assert result.exit_code == 2
    assert not db_path.exists()


def test_bad_config_dir_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", "--config-dir", str(tmp_path / "missing")])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def _status(config_dir: Path, db_path: Path) -> Result:
    return runner.invoke(
        app, [*_LOG_LEVEL, "status", "--config-dir", str(config_dir), "--db", str(db_path)]
    )


def test_status_shouts_about_a_configured_source_that_has_never_produced_anything(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    """The dead-feed detector.

    `beta` 404s every run, so it has no `items` row and a naive `GROUP BY
    source_id` would simply omit it — a dead feed reads as a quiet one. It must
    be on the table, and it must be loud.
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(return_value=httpx.Response(200, text=_feed("alpha")))
        mock.get(FEED_B).mock(return_value=httpx.Response(404))
        _invoke(config_dir, db_path, cache_dir)

    result = _status(config_dir, db_path)

    assert result.exit_code == 0, result.output
    output = result.output
    assert "beta" in output
    assert "NEVER SEEN" in output
    assert "produced NO items" in output


def test_status_on_an_empty_db_reports_every_source_dark(config_dir: Path, db_path: Path) -> None:
    result = _status(config_dir, db_path)

    assert result.exit_code == 0, result.output
    output = result.output
    assert "no runs recorded yet" in output
    assert "2 configured source(s) have produced NO items" in output


def test_status_reports_health_freshness_and_zero_token_spend(
    config_dir: Path, db_path: Path, cache_dir: Path, both_feeds_ok: None
) -> None:
    _invoke(config_dir, db_path, cache_dir)
    result = _status(config_dir, db_path)

    assert result.exit_code == 0, result.output
    output = result.output
    assert "Last run per kind" in output
    assert "Per-source freshness" in output
    # The cost alarm shows before there is any cost to alarm about (DESIGN §8).
    assert "LLM input tokens" in output
    assert "all 2 configured sources have produced items" in output


def test_status_calls_a_low_volume_source_quiet_rather_than_broken(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    """A blog between posts is not an incident.

    `items.fetched_at` is first-seen (`db.py::_merge_item` preserves it), so the
    age shown is "last new item", not "last fetched" — a source that fetches
    cleanly every morning still shows an old timestamp if its author hasn't
    posted. Karpathy posts every few months; alarming on that trains the user to
    skim the one table that also carries NEVER SEEN.
    """
    old = "2020-01-01T00:00:00+00:00"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(return_value=httpx.Response(200, text=_feed("alpha")))
        mock.get(FEED_B).mock(return_value=httpx.Response(200, text=_feed("beta")))
        _invoke(config_dir, db_path, cache_dir)

    with connection(db_path) as conn:
        conn.execute("UPDATE items SET fetched_at = ? WHERE source_id = 'alpha'", (old,))

    output = _status(config_dir, db_path).output
    assert "quiet" in output
    assert "last new item" in output
    # The words that would make it look broken must not appear for a live source.
    assert "NEVER SEEN" not in output
    assert "⚠" not in output


def test_status_surfaces_the_last_runs_errors(
    config_dir: Path, db_path: Path, cache_dir: Path
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_A).mock(return_value=httpx.Response(200, text=_feed("alpha")))
        mock.get(FEED_B).mock(return_value=httpx.Response(404))
        _invoke(config_dir, db_path, cache_dir)

    output = _status(config_dir, db_path).output
    assert "Errors from the last 'ingest' run" in output
    assert "FetchError" in output
