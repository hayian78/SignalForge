"""Tests for delivery wired into the CLI (DESIGN §13.2's push channel).

Scope: the seam where `digest` hands an already-written digest to `deliver/`, plus
the `deliver test` command. Message rendering and the provider client are
`tests/deliver/test_email.py`'s job.

Three properties carry the weight, and all three are about *not* doing something:

* **The vault write is unconditional and comes first.** A dead mail provider must
  leave the digest on disk and the run `ok` (§7, NEVER rule 12) — the file is the
  product, the email is a mirror.
* **One send per report, ever.** Re-running `digest` for a date rewrites the file
  and mails nothing (NEVER rule 4). Two guards enforce it and one of them is
  stateless, because the other lives in a database CLAUDE.md §3 calls disposable.
* **Every refusal is named.** "No email today" with no reason is
  indistinguishable from a broken pipeline.

Every send goes through `respx` (NEVER rule 13).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner, Result

from signalforge.cli import app
from signalforge.db import connection, upsert_item
from tests.conftest import make_item

runner = CliRunner()

ENDPOINT = "https://api.resend.com/emails"

DIGEST_DATE = datetime.now(UTC).date().isoformat()
"""Today, not a frozen literal.

Delivery is gated on a freshness window measured against the wall clock, so a
hardcoded date would pass on the day it was written and silently stop exercising
the send path forever after. The digest itself is empty on most of these runs —
the seeded item is dated — which is fine: these tests are about the delivery
mechanics, not the content. `tests/report/test_daily.py` owns the content.

A midnight rollover between this line and the assertion is harmless *because*
`MAX_DELIVERY_AGE_DAYS >= 1`; if that constant ever becomes 0, this becomes a
once-a-day flake and needs a frozen clock instead.
"""

STALE_DATE = "2020-01-01"
"""Comfortably outside the window, for the refusal cases."""

API_KEY = "re_test_key_not_a_real_one"

DELIVERY_BLOCK = """
delivery:
  email:
    enabled: true
    api_key_env: RESEND_API_KEY
    from_address: SignalForge <digest@example.com>
    to:
      - reader@example.com
"""


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


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in the environment, so `get_secret` never reaches a developer's real
    `.env` during a test run."""
    monkeypatch.setenv("RESEND_API_KEY", API_KEY)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "signalforge.db"


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    return tmp_path / "vault"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Minimal config with delivery enabled."""
    path = tmp_path / "config"
    path.mkdir()
    (path / "interests.yaml").write_text(
        "thresholds: {weekly_min_signal: 3, weekly_min_relevance: 3, weekly_min_total: 10,"
        " daily_max_items: 15}\n",
        encoding="utf-8",
    )
    (path / "sources.yaml").write_text(
        "defaults:\n"
        "  fetch_timeout: 20\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        "  max_item_age_days: 7\n"
        "rss: []\n",
        encoding="utf-8",
    )
    (path / "settings.yaml").write_text(f"timezone: UTC\n{DELIVERY_BLOCK}", encoding="utf-8")
    return path


@pytest.fixture
def config_dir_no_delivery(config_dir: Path) -> Path:
    """The historical shape: no `delivery:` block at all."""
    (config_dir / "settings.yaml").write_text("timezone: UTC\n", encoding="utf-8")
    return config_dir


def _seed(db_path: Path) -> None:
    with connection(db_path) as conn:
        item_id, _ = upsert_item(conn, make_item())
        conn.execute(
            """
            INSERT INTO scores (
                item_id, triage, signal, relevance, novelty, reasoning,
                rubric_version, model, scored_at
            ) VALUES (?, 'keep', 4, 4, 3, 'A reason this item matters.', 'v1',
                      'claude-haiku-4-5', '2026-07-16T06:05:00+00:00')
            """,
            (item_id,),
        )


def _invoke(
    db_path: Path, vault_dir: Path, config_dir: Path, *extra: str, date: str = DIGEST_DATE
) -> Result:
    return runner.invoke(
        app,
        [
            "--log-level",
            "CRITICAL",
            "digest",
            "--config-dir",
            str(config_dir),
            "--db",
            str(db_path),
            "--vault-dir",
            str(vault_dir),
            "--date",
            date,
            *extra,
        ],
    )


def _deliveries(db_path: Path) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(conn.execute("SELECT * FROM deliveries ORDER BY id").fetchall())


def _last_run_errors(db_path: Path) -> list[dict[str, str]]:
    with connection(db_path) as conn:
        row = conn.execute("SELECT errors FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    errors: list[dict[str, str]] = json.loads(row["errors"] or "[]")
    return errors


def _last_run_status(db_path: Path) -> str:
    with connection(db_path) as conn:
        row = conn.execute("SELECT status FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return str(row["status"])


def _ok_response() -> httpx.Response:
    return httpx.Response(200, json={"id": "msg_test"})


# --------------------------------------------------------------------------- #
# The happy path, and the ordering that makes it safe
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_digest_is_written_and_then_mailed(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    assert (vault_dir / "daily" / f"{DIGEST_DATE}.md").is_file()
    assert route.call_count == 1
    assert "email delivered" in result.output


@respx.mock
def test_the_delivery_row_records_the_send(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    _invoke(db_path, vault_dir, config_dir)

    rows = _deliveries(db_path)
    assert len(rows) == 1
    assert rows[0]["channel"] == "email"
    assert rows[0]["report_kind"] == "daily"
    assert rows[0]["target_date"] == DIGEST_DATE
    assert rows[0]["provider_id"] == "msg_test"
    assert rows[0]["run_id"] is not None


@respx.mock
def test_the_vault_file_is_written_before_the_send_is_attempted(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """The ordering DESIGN §13.2 fixes, asserted rather than assumed.

    The provider callback checks the file already exists at the moment the POST
    happens, so a future refactor that sends first would fail here rather than
    silently make the email the only record of a report.
    """
    written_at_send_time: list[bool] = []

    def _record(request: httpx.Request) -> httpx.Response:
        written_at_send_time.append((vault_dir / "daily" / f"{DIGEST_DATE}.md").is_file())
        return _ok_response()

    respx.post(ENDPOINT).mock(side_effect=_record)
    _seed(db_path)

    _invoke(db_path, vault_dir, config_dir)

    assert written_at_send_time == [True]


# --------------------------------------------------------------------------- #
# Idempotency (NEVER rule 4)
# --------------------------------------------------------------------------- #


@respx.mock
def test_re_rendering_a_date_rewrites_the_file_and_mails_nothing(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """The whole reason the `deliveries` table exists.

    A second run is the *same morning's digest* re-rendered after late items
    landed. That is a repeat, not a correction, and a second copy is spam.
    """
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    first = _invoke(db_path, vault_dir, config_dir)
    second = _invoke(db_path, vault_dir, config_dir)

    assert (first.exit_code, second.exit_code) == (0, 0)
    assert route.call_count == 1
    assert len(_deliveries(db_path)) == 1
    assert (vault_dir / "daily" / f"{DIGEST_DATE}.md").is_file()
    assert "already delivered" in second.output


@respx.mock
def test_resend_overrides_the_send_log(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    _invoke(db_path, vault_dir, config_dir)
    result = _invoke(db_path, vault_dir, config_dir, "--resend")

    assert result.exit_code == 0
    assert route.call_count == 2
    # Still one row: the send log is keyed on the report, not on attempts.
    assert len(_deliveries(db_path)) == 1


@respx.mock
def test_resend_cannot_override_the_freshness_window(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """No flag combination may mail history.

    The window is checked *before* the send log precisely so `--resend`, which is
    allowed to overrule a recorded delivery, cannot reach past it.
    """
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir, "--resend", date=STALE_DATE)

    assert result.exit_code == 0
    assert route.call_count == 0
    assert "from today" in result.output


@respx.mock
def test_an_old_date_is_written_but_never_mailed(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """`digest --date <old>` is a legitimate repair operation.

    It must fix the file without putting month-old news in the inbox — and this
    guard is stateless, so it still holds after the database is deleted and
    rebuilt (CLAUDE.md §3).
    """
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir, date=STALE_DATE)

    assert route.call_count == 0
    assert (vault_dir / "daily" / f"{STALE_DATE}.md").is_file()
    assert _deliveries(db_path) == []
    assert "email skipped" in result.output


# --------------------------------------------------------------------------- #
# Failure isolation (§7, NEVER rule 12)
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_provider_failure_leaves_the_run_ok_and_the_digest_written(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """A dead mail provider is not a failed digest.

    The file is the product and it is already on disk; the error is recorded so
    `signalforge status` and the next digest surface it (§7's monitoring channel).
    """
    respx.post(ENDPOINT).mock(return_value=httpx.Response(500, text="upstream boom"))
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    assert (vault_dir / "daily" / f"{DIGEST_DATE}.md").is_file()
    assert _last_run_status(db_path) == "ok"
    errors = _last_run_errors(db_path)
    assert len(errors) == 1
    assert "email delivery failed" in errors[0]["message"]


@respx.mock
def test_a_failed_send_writes_no_delivery_row_so_tomorrow_can_retry(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    route = respx.post(ENDPOINT).mock(
        side_effect=[httpx.Response(400, text="nope"), _ok_response()]
    )
    _seed(db_path)

    _invoke(db_path, vault_dir, config_dir)
    assert _deliveries(db_path) == []

    _invoke(db_path, vault_dir, config_dir)

    assert route.call_count == 2
    assert len(_deliveries(db_path)) == 1


@respx.mock
def test_a_missing_api_key_is_recorded_rather_than_crashing_the_run(
    db_path: Path, vault_dir: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured on but unusable is an error, not a silent skip — and not a
    failed digest either. No HTTP call is attempted."""
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0
    assert route.call_count == 0
    assert _last_run_status(db_path) == "ok"
    assert "RESEND_API_KEY" in _last_run_errors(db_path)[0]["message"]


# --------------------------------------------------------------------------- #
# Not sending: the shapes that must stay silent-but-explained
# --------------------------------------------------------------------------- #


@respx.mock
def test_no_delivery_block_sends_nothing_and_reports_nothing(
    db_path: Path, vault_dir: Path, config_dir_no_delivery: Path
) -> None:
    """The historical behaviour, which must stay the default forever."""
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir_no_delivery)

    assert result.exit_code == 0
    assert route.call_count == 0
    assert "email" not in result.output
    assert _last_run_errors(db_path) == []


@respx.mock
def test_no_send_suppresses_delivery(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir, "--no-send")

    assert result.exit_code == 0
    assert route.call_count == 0
    assert _deliveries(db_path) == []


@respx.mock
def test_dry_run_writes_nothing_and_sends_nothing(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """A preview must have no side effect the operator did not ask for — the same
    rule that already skips the feedback harvest."""
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir, "--dry-run")

    assert result.exit_code == 0
    assert route.call_count == 0
    assert not (vault_dir / "daily" / f"{DIGEST_DATE}.md").exists()
    assert _deliveries(db_path) == []


@respx.mock
def test_an_enabled_false_channel_sends_nothing(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    (config_dir / "settings.yaml").write_text(
        f"timezone: UTC\n{DELIVERY_BLOCK.replace('enabled: true', 'enabled: false')}",
        encoding="utf-8",
    )
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    assert _invoke(db_path, vault_dir, config_dir).exit_code == 0
    assert route.call_count == 0


# --------------------------------------------------------------------------- #
# `signalforge deliver test`
# --------------------------------------------------------------------------- #


def _invoke_test_command(config_dir: Path) -> Result:
    return runner.invoke(
        app, ["--log-level", "CRITICAL", "deliver", "test", "--config-dir", str(config_dir)]
    )


@respx.mock
def test_deliver_test_sends_a_sample_without_touching_the_database(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting up a provider means a key, a verified sender and a recipient, each
    failing its own way. This surfaces those on demand, not at 06:00 tomorrow.

    "Touches no pipeline state" is asserted by making any database connection
    explode, not by checking a path that does not exist. `deliver test` has no
    `--db` flag, so an absent tmp file would stay absent even if the command
    started opening the operator's real database.
    """

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("deliver test must not open the database")

    monkeypatch.setattr("signalforge.db.connect", _explode)
    monkeypatch.setattr("signalforge.cli.connection", _explode)
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())

    result = _invoke_test_command(config_dir)

    assert result.exit_code == 0, result.output
    assert route.call_count == 1
    assert "email test sent" in result.output


@respx.mock
def test_deliver_test_reports_a_provider_refusal(config_dir: Path) -> None:
    """The realistic first failure: the sender domain is not verified yet."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(403, text="domain is not verified"))

    result = _invoke_test_command(config_dir)

    assert result.exit_code == 1
    assert "test failed" in result.output


def test_deliver_test_exits_when_no_channel_is_configured(
    config_dir_no_delivery: Path,
) -> None:
    result = _invoke_test_command(config_dir_no_delivery)

    assert result.exit_code == 2
    assert "no delivery channel is enabled" in result.output


def test_deliver_test_exits_when_the_key_is_missing(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RESEND_API_KEY", raising=False)

    result = _invoke_test_command(config_dir)

    assert result.exit_code == 2
    assert "RESEND_API_KEY is not set" in result.output


# --------------------------------------------------------------------------- #
# The one outcome that is both a success and a problem
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_send_whose_log_row_conflicts_is_reported_as_delivered_with_a_problem(
    db_path: Path, vault_dir: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message reached the inbox; only the send log disagreed.

    `record_delivery` returning False here means a duplicate copy went out. The
    operator must be told that, and must not be told "not delivered" — which would
    be the exact opposite of what happened. The run stays `ok`: the digest is on
    disk and an extra email is not a pipeline failure.
    """
    respx.post(ENDPOINT).mock(return_value=_ok_response())
    monkeypatch.setattr("signalforge.deliver.record_delivery", lambda *a, **k: False)
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    assert _last_run_status(db_path) == "ok"
    assert "duplicate message was sent" in _last_run_errors(db_path)[0]["message"]
    assert "delivered, with a problem" in result.output
    assert "not delivered" not in result.output


@respx.mock
def test_an_unexpected_render_failure_is_an_outcome_not_a_crash(
    db_path: Path, vault_dir: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`deliver/` promises a failed delivery is an outcome, not an exception.

    Only `SendError` used to be caught, so a jinja error or an httpx error outside
    the retry predicate would have escaped to the CLI backstop. The digest is
    already written either way; this pins where the failure is handled.
    """
    respx.post(ENDPOINT).mock(return_value=_ok_response())
    monkeypatch.setattr(
        "signalforge.deliver.render_email",
        lambda context: (_ for _ in ()).throw(RuntimeError("template exploded")),
    )
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    assert (vault_dir / "daily" / f"{DIGEST_DATE}.md").is_file()
    assert _last_run_status(db_path) == "ok"
    assert "template exploded" in _last_run_errors(db_path)[0]["message"]
    assert _deliveries(db_path) == []


@respx.mock
def test_an_empty_digest_is_still_mailed(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    """Deliberate: the email is proof of life.

    A quiet news day and a dead pipeline look identical from an empty inbox, and
    the message says so in its own body ("a quiet news day, or a pipeline problem
    worth checking with `signalforge status`").
    """
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    # No _seed: nothing scored, nothing kept.

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0
    assert route.call_count == 1


# --------------------------------------------------------------------------- #
# World-authored text in a terminal-formatting context
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_bracketed_provider_error_does_not_fail_the_run(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """A mail provider's error text must not be able to fail the digest.

    `SendError` carries the response body verbatim, and rich parses `[...]` as
    markup: an error body containing `[/v1]` raised `MarkupError` from inside the
    reporting loop, escaped to `digest`'s handler, and marked the run **failed** —
    a provider's prose taking down the pipeline, which NEVER rules 12 and 19 both
    forbid. The milder case, `[bold]`, silently ate the text instead.
    """
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(400, text="bad request [/v1] retired [bold]see docs")
    )
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir)

    assert result.exit_code == 0, result.output
    assert _last_run_status(db_path) == "ok"
    assert (vault_dir / "daily" / f"{DIGEST_DATE}.md").is_file()
    # The operator sees the provider's words, not a swallowed fragment.
    assert "[/v1]" in result.output
    assert "[bold]see docs" in result.output


@respx.mock
def test_status_survives_a_bracketed_error_in_runs_errors(
    db_path: Path, vault_dir: Path, config_dir: Path
) -> None:
    """`status` is the documented monitoring channel for a broken delivery.

    The same message is persisted to `runs.errors` and rendered into a rich table
    there, so an unescaped bracket would crash the one command that reports the
    failure — until that run row was superseded.
    """
    respx.post(ENDPOINT).mock(return_value=httpx.Response(400, text="nope [/x] gone"))
    _seed(db_path)
    _invoke(db_path, vault_dir, config_dir)

    result = runner.invoke(
        app,
        [
            "--log-level",
            "CRITICAL",
            "status",
            "--db",
            str(db_path),
            "--config-dir",
            str(config_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.exception is None


@respx.mock
def test_deliver_test_survives_a_bracketed_provider_error(config_dir: Path) -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(403, text="domain [unverified]"))

    result = _invoke_test_command(config_dir)

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "test failed" in result.output


# --------------------------------------------------------------------------- #
# The freshness window is symmetric
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_far_future_date_is_not_mailed(db_path: Path, vault_dir: Path, config_dir: Path) -> None:
    """`--date 2027-01-01` is a typo, not a clock artifact.

    Mailing the empty digest it renders would also burn that date's send-log row,
    suppressing the real send when the day actually arrives.
    """
    route = respx.post(ENDPOINT).mock(return_value=_ok_response())
    _seed(db_path)

    result = _invoke(db_path, vault_dir, config_dir, date="2099-01-01")

    assert result.exit_code == 0
    assert route.call_count == 0
    assert _deliveries(db_path) == []
    assert "email skipped" in result.output
