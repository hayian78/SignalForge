"""Tests for the `signalforge curate` command group (DESIGN §7.1).

`llm.run_source_scout` is faked at its boundary and every HTTP request is served by
`respx`, so nothing here reaches the real API or a live network (CLAUDE.md §8, NEVER
rules 11 and 13). One test asserts the *absence* of an LLM path: `curate apply` must
not be able to make a call, because it runs inside the 6am chain where a paid step
would be a surprise.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner, Result

from signalforge import llm
from signalforge.cli import app
from signalforge.curate.approvals import proposal_marker
from signalforge.db import connection, insert_proposal, start_run, upsert_item
from signalforge.models import ProposalKind, ProposalStatus, ProposalTier
from tests.conftest import make_item
from tests.curate.conftest import fixture_bytes, make_scout_proposal

runner = CliRunner()

FEED_URL = "https://newvoice.example.com/feed"
SURFACE_DATE = "2026-07-31"


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


def _write_config(path: Path, *, curation: bool, repo_config_dir: Path) -> Path:
    """A config dir with or without a `curation:` block — the on/off shapes."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "interests.yaml").write_text(
        (repo_config_dir / "interests.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (path / "settings.yaml").write_text(
        "timezone: Australia/Brisbane\nvault_dir: vault\n", encoding="utf-8"
    )
    block = (
        "curation:\n"
        "  max_proposals_per_run: 5\n"
        "  max_searches_per_run: 6\n"
        "  yield_window_days: 30\n"
        "  settled_display_days: 14\n"
        if curation
        else ""
    )
    (path / "sources.yaml").write_text(
        "defaults:\n"
        "  fetch_timeout: 5\n"
        "  min_hn_points: 80\n"
        "  max_summary_chars: 4000\n"
        "  max_item_age_days: 3650\n"
        "rss:\n"
        "  - id: deepmind\n"
        "    url: https://deepmind.google/blog/rss.xml\n"
        "hackernews:\n"
        "  keywords: [llm]\n" + block,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config_dir(tmp_path: Path, repo_config_dir: Path) -> Path:
    return _write_config(tmp_path / "config", curation=True, repo_config_dir=repo_config_dir)


@pytest.fixture
def config_dir_off(tmp_path: Path, repo_config_dir: Path) -> Path:
    return _write_config(tmp_path / "config-off", curation=False, repo_config_dir=repo_config_dir)


def _fake_scout(
    monkeypatch: pytest.MonkeyPatch,
    *,
    proposals: list[llm.ScoutProposal] | None = None,
    errors: list[str] | None = None,
) -> None:
    def _run(**_kwargs: object) -> llm.ScoutResult:
        return llm.ScoutResult(
            proposals=list(proposals or []),
            errors=list(errors or []),
            input_tokens=4200,
            output_tokens=1100,
            web_search_requests=6,
        )

    monkeypatch.setattr("signalforge.curate.scout.llm.run_source_scout", _run)


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any path to the Anthropic client explode.

    Patches `get_anthropic_client`, which every call in `llm.py` goes through, so
    this catches an LLM call made from anywhere rather than only the scout.
    """

    def _explode() -> object:
        raise AssertionError("this command must not touch the Anthropic client")

    monkeypatch.setattr("signalforge.llm.get_anthropic_client", _explode)


def _invoke(*args: str) -> Result:
    return runner.invoke(app, ["--log-level", "WARNING", *args])


def _curate_run(db_path: Path, config_dir: Path, cache_dir: Path, *extra: str) -> Result:
    return _invoke(
        "curate",
        "run",
        "--config-dir",
        str(config_dir),
        "--db",
        str(db_path),
        "--cache-dir",
        str(cache_dir),
        "--surface-date",
        SURFACE_DATE,
        *extra,
    )


def _runs(db_path: Path, kind: str) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(
            conn.execute("SELECT * FROM runs WHERE kind = ? ORDER BY id", (kind,)).fetchall()
        )


def _proposals(db_path: Path) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(conn.execute("SELECT * FROM proposals ORDER BY id").fetchall())


def _mock_feed(url: str = FEED_URL) -> None:
    respx.get(url).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )


# --------------------------------------------------------------------------- #
# curate run — the only paid command
# --------------------------------------------------------------------------- #


def test_curate_run_says_so_and_stops_when_the_feature_is_off(
    db_path: Path, config_dir_off: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent `curation:` block is not a default to fill in.

    It means the operator has not agreed to this feature's spend, and a command that
    costs money must not invent its own caps (CLAUDE.md §4). `_forbid_llm` proves it
    exits before reaching the client.
    """
    _forbid_llm(monkeypatch)

    result = _curate_run(db_path, config_dir_off, tmp_path / "cache")

    assert result.exit_code == 1
    assert "source curation is off" in result.output
    assert _runs(db_path, "curate") == []


@respx.mock
def test_curate_run_stores_proposals_and_says_when_they_surface(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    result = _curate_run(db_path, config_dir, tmp_path / "cache")

    assert result.exit_code == 0, result.output
    assert SURFACE_DATE in result.output
    rows = _proposals(db_path)
    assert len(rows) == 1
    assert rows[0]["surface_date"] == SURFACE_DATE
    assert rows[0]["status"] == ProposalStatus.PENDING.value


@respx.mock
def test_curate_run_records_its_spend_including_the_search_count(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tokens *and* searches: search bills per call, so tokens alone hide it (NEVER 11)."""
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    result = _curate_run(db_path, config_dir, tmp_path / "cache")

    assert result.exit_code == 0, result.output
    run = _runs(db_path, "curate")[-1]
    assert run["llm_input_tokens"] == 4200
    assert run["llm_output_tokens"] == 1100
    assert run["server_tool_requests"] == 6
    assert "$0.06" in result.output


@respx.mock
def test_a_dry_run_writes_no_proposal_but_still_records_the_spend(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opposite of `digest --dry-run`, and the difference matters.

    The API call and its searches happen either way — that is what makes the preview
    honest — so the money is spent and the `runs` row that accounts for it must be
    written. Only the `proposals` writes are skipped.
    """
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    result = _curate_run(db_path, config_dir, tmp_path / "cache", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert _proposals(db_path) == []
    run = _runs(db_path, "curate")[-1]
    assert run["server_tool_requests"] == 6
    assert run["llm_input_tokens"] == 4200


@respx.mock
def test_a_dry_run_previews_what_would_be_stored(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This preview is the operator's quality gate, so it has to show the substance."""
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    result = _curate_run(db_path, config_dir, tmp_path / "cache", "--dry-run")

    assert "add feed" in result.output
    assert FEED_URL in result.output
    assert "Cited three times" in result.output
    assert "simonwillison.net" in result.output


@respx.mock
def test_running_curate_twice_stores_each_proposal_once(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The idempotency gate for `curate`, end to end (CLAUDE.md §3, NEVER rule 4)."""
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    first = _curate_run(db_path, config_dir, tmp_path / "cache")
    before = _proposals(db_path)
    # `--force` because the cadence guard would otherwise refuse a same-day re-run;
    # what is under test here is storage idempotency, not the guard.
    second = _curate_run(db_path, config_dir, tmp_path / "cache", "--force")

    assert (first.exit_code, second.exit_code) == (0, 0)
    assert [tuple(row) for row in _proposals(db_path)] == [tuple(row) for row in before]
    assert "1 already known" in second.output


@respx.mock
def test_scout_notes_are_recorded_to_the_run_and_shown(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`runs.errors` is the monitoring channel, so a note must reach it (DESIGN §7)."""
    _fake_scout(monkeypatch, errors=["scout call failed: overloaded"])

    result = _curate_run(db_path, config_dir, tmp_path / "cache")

    assert result.exit_code == 0, result.output
    run = _runs(db_path, "curate")[-1]
    assert run["status"] == "partial"
    assert "overloaded" in str(run["errors"])
    assert "overloaded" in result.output


# --------------------------------------------------------------------------- #
# curate apply
# --------------------------------------------------------------------------- #


def _seed_pending(db_path: Path, *, dedup_key: str = FEED_URL) -> int:
    with connection(db_path) as conn:
        proposal_id = insert_proposal(
            conn,
            # A real run id, as production always has: a proposal is the audit
            # trail for why a `sources.yaml` edit happened, never run-less.
            run_id=start_run(conn, "curate", started_at=datetime(2026, 7, 30, 6, 30, tzinfo=UTC)),
            kind=ProposalKind.ADD_RSS,
            dedup_key=dedup_key,
            payload={"id": "newvoice", "url": dedup_key, "weight": 1.2},
            rationale="Cited repeatedly by items you kept.",
            evidence=[{"url": "https://example.com/post", "note": ""}],
            probe=None,
            tier=ProposalTier.WEB,
            status=ProposalStatus.PENDING,
            surface_date=date(2026, 7, 30),
            created_at=datetime(2026, 7, 30, 6, 0, tzinfo=UTC),
        )
    assert proposal_id is not None
    return proposal_id


def _apply(db_path: Path, config_dir: Path, vault_dir: Path) -> Result:
    return _invoke(
        "curate",
        "apply",
        "--config-dir",
        str(config_dir),
        "--db",
        str(db_path),
        "--vault-dir",
        str(vault_dir),
    )


def test_curate_apply_makes_no_llm_call(
    db_path: Path, config_dir: Path, vault_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It runs inside the 6am chain, where a paid step would be a surprise.

    Asserted by making every path to the Anthropic client raise, rather than by
    reading the code — this is the kind of thing a later refactor breaks quietly.
    """
    _forbid_llm(monkeypatch)
    _seed_pending(db_path)

    result = _apply(db_path, config_dir, vault_dir)

    assert result.exit_code == 0, result.output


def test_curate_apply_harvests_a_tick_and_edits_sources_yaml(
    db_path: Path, config_dir: Path, vault_dir: Path
) -> None:
    """The round trip the whole feature is for: tick in the vault, edit in the config."""
    proposal_id = _seed_pending(db_path)
    daily = vault_dir / "daily"
    daily.mkdir(parents=True)
    (daily / "2026-07-30.md").write_text(
        proposal_marker(proposal_id, "approve").replace("[ ]", "[x]") + "\n", encoding="utf-8"
    )

    result = _apply(db_path, config_dir, vault_dir)

    assert result.exit_code == 0, result.output
    sources_text = (config_dir / "sources.yaml").read_text(encoding="utf-8")
    assert "id: newvoice" in sources_text
    assert f"url: {FEED_URL}" in sources_text
    # The suggested weight rides along, visible as a line the operator can edit.
    assert "weight: 1.2" in sources_text
    assert "git diff config/sources.yaml" in result.output
    with connection(db_path) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    assert row["status"] == ProposalStatus.APPLIED.value


def test_curate_apply_twice_leaves_sources_yaml_byte_identical(
    db_path: Path, config_dir: Path, vault_dir: Path
) -> None:
    """Idempotency at the file level, which the status guard alone cannot give.

    The YAML write and the status flip are two writes; a crash between them leaves an
    approved row whose edit is already on disk. Appending again would duplicate a
    config entry, which is NEVER rule 4 against the file.
    """
    proposal_id = _seed_pending(db_path)
    daily = vault_dir / "daily"
    daily.mkdir(parents=True)
    (daily / "2026-07-30.md").write_text(
        proposal_marker(proposal_id, "approve").replace("[ ]", "[x]") + "\n", encoding="utf-8"
    )

    assert _apply(db_path, config_dir, vault_dir).exit_code == 0
    after_first = (config_dir / "sources.yaml").read_text(encoding="utf-8")
    assert _apply(db_path, config_dir, vault_dir).exit_code == 0

    assert (config_dir / "sources.yaml").read_text(encoding="utf-8") == after_first


def test_curate_apply_leaves_a_rejected_proposal_alone(
    db_path: Path, config_dir: Path, vault_dir: Path
) -> None:
    proposal_id = _seed_pending(db_path)
    daily = vault_dir / "daily"
    daily.mkdir(parents=True)
    (daily / "2026-07-30.md").write_text(
        proposal_marker(proposal_id, "reject").replace("[ ]", "[x]") + "\n", encoding="utf-8"
    )
    before = (config_dir / "sources.yaml").read_text(encoding="utf-8")

    result = _apply(db_path, config_dir, vault_dir)

    assert result.exit_code == 0, result.output
    assert (config_dir / "sources.yaml").read_text(encoding="utf-8") == before
    with connection(db_path) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    assert row["status"] == ProposalStatus.REJECTED.value


def test_both_boxes_ticked_is_reported_and_left_pending(
    db_path: Path, config_dir: Path, vault_dir: Path
) -> None:
    """Guessing which one the operator meant is how a source gets added by mistake."""
    proposal_id = _seed_pending(db_path)
    daily = vault_dir / "daily"
    daily.mkdir(parents=True)
    (daily / "2026-07-30.md").write_text(
        proposal_marker(proposal_id, "approve").replace("[ ]", "[x]")
        + "\n"
        + proposal_marker(proposal_id, "reject").replace("[ ]", "[x]")
        + "\n",
        encoding="utf-8",
    )

    result = _apply(db_path, config_dir, vault_dir)

    assert result.exit_code == 0, result.output
    assert "both boxes ticked" in result.output
    with connection(db_path) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    assert row["status"] == ProposalStatus.PENDING.value


def test_curate_apply_records_its_own_run(db_path: Path, config_dir: Path, vault_dir: Path) -> None:
    """No silent runs (CLAUDE.md §3), and a kind of its own so `status` can show both.

    Folding it into `curate` would hide a scout that has stopped running behind an
    apply that runs every morning.
    """
    assert _apply(db_path, config_dir, vault_dir).exit_code == 0

    run = _runs(db_path, "curate-apply")[-1]
    assert run["status"] == "ok"
    assert run["finished_at"] is not None
    assert run["llm_input_tokens"] == 0


# --------------------------------------------------------------------------- #
# curate list / decide / reopen
# --------------------------------------------------------------------------- #


def test_curate_list_shows_a_proposal(db_path: Path) -> None:
    _seed_pending(db_path)

    result = _invoke("curate", "list", "--db", str(db_path))

    assert result.exit_code == 0, result.output
    assert "add feed" in result.output
    assert "pending" in result.output


def test_curate_list_rejects_an_unknown_status(db_path: Path) -> None:
    _seed_pending(db_path)

    result = _invoke("curate", "list", "--db", str(db_path), "--status", "maybe")

    assert result.exit_code == 2


def test_curate_decide_records_a_rejection_with_its_reason(db_path: Path) -> None:
    """The note is the half that teaches the next scout run something."""
    proposal_id = _seed_pending(db_path)

    result = _invoke(
        "curate",
        "decide",
        str(proposal_id),
        "reject",
        "--db",
        str(db_path),
        "--note",
        "mostly product marketing",
    )

    assert result.exit_code == 0, result.output
    with connection(db_path) as conn:
        row = conn.execute(
            "SELECT status, decision_note FROM proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
    assert row["status"] == ProposalStatus.REJECTED.value
    assert row["decision_note"] == "mostly product marketing"


def test_deciding_twice_changes_nothing_the_second_time(db_path: Path) -> None:
    """`decide_proposal` guards on pending, so a re-decision is a no-op, not a flip."""
    proposal_id = _seed_pending(db_path)

    first = _invoke("curate", "decide", str(proposal_id), "approve", "--db", str(db_path))
    second = _invoke("curate", "decide", str(proposal_id), "reject", "--db", str(db_path))

    assert (first.exit_code, second.exit_code) == (0, 0)
    assert "already decided" in second.output
    with connection(db_path) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    assert row["status"] == ProposalStatus.APPROVED.value


def test_curate_decide_rejects_an_unknown_decision_word(db_path: Path) -> None:
    proposal_id = _seed_pending(db_path)

    result = _invoke("curate", "decide", str(proposal_id), "maybe", "--db", str(db_path))

    assert result.exit_code == 2


def test_curate_decide_on_an_unknown_proposal_exits_nonzero(db_path: Path) -> None:
    with connection(db_path):
        pass  # create and migrate the DB

    result = _invoke("curate", "decide", "999", "approve", "--db", str(db_path))

    assert result.exit_code == 2
    assert "unknown proposal" in result.output


def test_curate_reopen_returns_a_rejection_to_pending(db_path: Path) -> None:
    """The escape hatch: the unique index means the scout can never re-suggest it."""
    proposal_id = _seed_pending(db_path)
    _invoke("curate", "decide", str(proposal_id), "reject", "--db", str(db_path))

    result = _invoke("curate", "reopen", str(proposal_id), "--db", str(db_path))

    assert result.exit_code == 0, result.output
    with connection(db_path) as conn:
        row = conn.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    assert row["status"] == ProposalStatus.PENDING.value


def test_curate_reopen_will_not_undo_an_applied_change(db_path: Path) -> None:
    """That state lives in `sources.yaml`; undoing it means editing the file."""
    proposal_id = _seed_pending(db_path)
    with connection(db_path) as conn:
        conn.execute(
            "UPDATE proposals SET status = ? WHERE id = ?",
            (ProposalStatus.APPLIED.value, proposal_id),
        )

    result = _invoke("curate", "reopen", str(proposal_id), "--db", str(db_path))

    assert result.exit_code == 0, result.output
    assert "not reopenable" in result.output


# --------------------------------------------------------------------------- #
# the daily chain and the status readout
# --------------------------------------------------------------------------- #


def _fake_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `score` out of the way: this file tests curation, not triage."""
    monkeypatch.setattr(
        "signalforge.llm.run_triage_batch",
        lambda *_a, **_k: llm.TriageBatchResult(),
    )


@respx.mock
def test_daily_applies_approvals_before_it_ingests(
    db_path: Path,
    config_dir: Path,
    vault_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordering the whole approval loop depends on (DESIGN §7.1).

    A feed approved in yesterday's digest has to be fetched by this morning's
    ingest — which only works if `curate apply` edits the config *first*. Asserted
    by the newly-approved feed actually being fetched in the same run.
    """
    respx.get("https://deepmind.google/blog/rss.xml").mock(return_value=httpx.Response(304))
    approved = respx.get(FEED_URL).mock(
        return_value=httpx.Response(200, content=fixture_bytes("simonwillison_atom.xml"))
    )
    respx.route(host="hn.algolia.com").mock(return_value=httpx.Response(200, json={"hits": []}))
    _fake_triage(monkeypatch)

    proposal_id = _seed_pending(db_path)
    daily = vault_dir / "daily"
    daily.mkdir(parents=True)
    (daily / "2026-07-30.md").write_text(
        proposal_marker(proposal_id, "approve").replace("[ ]", "[x]") + "\n", encoding="utf-8"
    )

    result = _invoke(
        "daily",
        "--config-dir",
        str(config_dir),
        "--db",
        str(db_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--vault-dir",
        str(vault_dir),
    )

    assert result.exit_code == 0, result.output
    assert approved.called, "the approved feed should have been fetched in the same run"
    kinds = [row["kind"] for row in _runs(db_path, "curate-apply")]
    assert kinds == ["curate-apply"]


@respx.mock
def test_daily_does_not_mention_curation_when_it_is_off(
    db_path: Path,
    config_dir_off: Path,
    vault_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator who has not enabled curation should see no trace of it."""
    respx.get("https://deepmind.google/blog/rss.xml").mock(return_value=httpx.Response(304))
    respx.route(host="hn.algolia.com").mock(return_value=httpx.Response(200, json={"hits": []}))
    _fake_triage(monkeypatch)

    result = _invoke(
        "daily",
        "--config-dir",
        str(config_dir_off),
        "--db",
        str(db_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--vault-dir",
        str(vault_dir),
    )

    assert result.exit_code == 0, result.output
    assert _runs(db_path, "curate-apply") == []


def test_status_shows_the_web_search_count_and_its_dollar_figure(
    db_path: Path, config_dir: Path
) -> None:
    """Search bills per call, so this line is not derivable from the token rows.

    Without it the $30 alarm cannot see the whole invoice (DESIGN §8).
    """
    with connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (kind, started_at, finished_at, status, llm_input_tokens,
                              llm_output_tokens, server_tool_requests)
            VALUES ('curate', ?, ?, 'ok', 4200, 1100, 6)
            """,
            (datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat()),
        )

    result = _invoke("status", "--config-dir", str(config_dir), "--db", str(db_path))

    assert result.exit_code == 0, result.output
    assert "web searches" in result.output
    assert "$0.06" in result.output


def test_status_shows_a_zero_search_line_before_any_scout_run(
    db_path: Path, config_dir: Path
) -> None:
    """A cost readout that appears only once there is cost is one nobody has read."""
    with connection(db_path):
        pass

    result = _invoke("status", "--config-dir", str(config_dir), "--db", str(db_path))

    assert result.exit_code == 0, result.output
    assert "$0.00" in result.output


@respx.mock
def test_spend_is_recorded_even_when_the_run_fails_after_the_call(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid run must never record $0 (NEVER rule 11).

    `run_source_scout` already promises not to raise for an API failure so its cost
    is never lost. Everything *after* it — normalizing, probing — is free but can
    still have a bug, and the caller reads the spend off the returned value. Both
    reviews of this branch found that gap independently, so it is pinned here: the
    probe stage explodes, and the row still shows what the call cost.
    """
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    async def _explode(*_args: object, **_kwargs: object) -> object:
        raise ZeroDivisionError("a bug in the free half of the run")

    monkeypatch.setattr("signalforge.curate.scout.probe_requests", _explode)

    result = _curate_run(db_path, config_dir, tmp_path / "cache")

    assert result.exit_code == 0, result.output
    run = _runs(db_path, "curate")[-1]
    assert run["llm_input_tokens"] == 4200
    assert run["server_tool_requests"] == 6
    assert run["status"] == "partial"
    assert "after the call was billed" in str(run["errors"])


@respx.mock
def test_a_broken_sources_yaml_does_not_black_out_the_whole_daily_run(
    db_path: Path,
    config_dir: Path,
    vault_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sources.yaml` is hand-edited, so a typo in it must cost only what reads it.

    Deciding whether curation is enabled *outside* the step loop made a ConfigError
    abort `daily` before any step ran and before any `runs` row existed — no
    isolation, and no record that 6am happened. `score` needs only `interests.yaml`
    and has to still clear its backlog.
    """
    _fake_triage(monkeypatch)
    with connection(db_path) as conn:
        item_id, _ = upsert_item(conn, make_item())
        assert item_id
    (config_dir / "sources.yaml").write_text("rss: [unclosed\n", encoding="utf-8")

    result = _invoke(
        "daily",
        "--config-dir",
        str(config_dir),
        "--db",
        str(db_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--vault-dir",
        str(vault_dir),
    )

    assert result.exit_code == 2, result.output
    assert [row["kind"] for row in _runs(db_path, "score")] == ["score"], (
        "score reads only interests.yaml and must still run"
    )


@respx.mock
def test_a_second_run_inside_the_week_is_refused_unless_forced(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scout is weekly, and nothing but this stops it being run daily.

    The unique index stops a re-run *storing* duplicates; it does nothing about the
    call being billed again. Wired into `daily` by accident, that is ~$7/month
    expected instead of ~$1 — inside the $30 alarm but several times this feature's
    own budget.
    """
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])

    first = _curate_run(db_path, config_dir, tmp_path / "cache")
    second = _curate_run(db_path, config_dir, tmp_path / "cache")
    forced = _curate_run(db_path, config_dir, tmp_path / "cache", "--force")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 1
    assert "--force" in second.output
    assert forced.exit_code == 0, forced.output
    # Two billed runs, not three: the refusal happened before the call.
    assert len(_runs(db_path, "curate")) == 2


def test_the_cadence_guard_does_not_look_at_the_apply_runs(
    db_path: Path, config_dir: Path, vault_dir: Path, tmp_path: Path
) -> None:
    """`curate apply` runs every morning and costs nothing.

    Counting it as "the scout ran recently" would refuse every scout run forever,
    which is why the two have separate `runs.kind` values.
    """
    assert _apply(db_path, config_dir, vault_dir).exit_code == 0

    result = _curate_run(db_path, config_dir, tmp_path / "cache", "--dry-run")

    assert "--force" not in result.output


@respx.mock
def test_a_dry_run_is_guarded_by_the_cadence_check_too(
    db_path: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Because a dry run spends the same money as a real one.

    The guard sits before the `dry_run` branch, so this holds by construction rather
    than by a second code path — pinned so a future reorder cannot quietly make the
    preview the cheap way to spend $0.25 a day.
    """
    _mock_feed()
    _fake_scout(monkeypatch, proposals=[make_scout_proposal(target=FEED_URL)])
    assert _curate_run(db_path, config_dir, tmp_path / "cache").exit_code == 0

    result = _curate_run(db_path, config_dir, tmp_path / "cache", "--dry-run")

    assert result.exit_code == 1
    assert "too soon" in result.output
