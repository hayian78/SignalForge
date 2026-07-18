"""The typer CLI — the only place config, `ingest/`, and `db.py` meet.

Phase 0 surface, and deliberately no more (NEVER rule 15):

* `ingest` — fetch every configured source, persist the items, close a `runs` row.
* `score` — triage/score unscored items via `score/` (DESIGN §8).
* `digest` — assemble the day's markdown digest from scored items (DESIGN §13).
* `daily` — `ingest` → `score` → `digest` in sequence, the cron entry (DESIGN §14).
* `status` — last-run health, per-source freshness, month-to-date token spend
  (DESIGN §14).

LLM work stays behind the `score/` boundary: this module drives the pipeline but
must never import `llm.py` directly (NEVER rule 1 — `anthropic` lives in `llm.py`).

### Why the persist loop is grouped by source

Two invariants meet here and both are per-source:

* **Failure isolation** (CLAUDE.md §7, NEVER rule 12) — one source's writes
  blowing up must not cost the other seven their items or their 304s.
* **Validator commit** — conditional-GET validators are staged in memory by the
  fetcher and become durable only once *this* module confirms the items reached
  SQLite (`ingest/base.py::ValidatorStore`). Confirming a source whose writes
  failed would 304 next run and lose those items for good.

So items are grouped by `source_id`, each group is persisted under its own
`try/except`, and only the groups that survived get their validators committed.

### The commit set is not `run.source_ids`

`run.source_ids` is "sources that produced ≥ 1 item". A healthy feed that 200s
with no new entries stages a validator and appears in no item list — keying the
commit off `source_ids` would leave it uncommitted *forever*, so it would
refetch unconditionally on every run and DESIGN §7's "most daily RSS fetches
return 304 and cost nothing" would quietly become false.

The rule is **commit every source that did not fail**: start from the staged set
(`validators.pending_sources()`), subtract sources that errored during ingest,
subtract sources whose upserts raised. Zero items plus no error is a success.

Every unhappy path still degrades to "refetch next run", never to "skip".
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from rich.console import Console
from rich.table import Table

from signalforge.config import (
    ConfigError,
    InterestsConfig,
    SettingsConfig,
    SourcesConfig,
    load_interests,
    load_settings,
    load_sources,
)
from signalforge.db import connection, finish_run, start_run, upsert_item
from signalforge.ingest import IngestError, IngestRun, build_ingestors, ingest_all
from signalforge.ingest.base import DEFAULT_MAX_CONCURRENCY
from signalforge.ingest.hackernews import HN_SOURCE_ID
from signalforge.models import Item
from signalforge.report.daily import build_digest_context, digest_path, render_digest
from signalforge.score import ScoreOutcome, score_unscored_items

__all__ = ["app", "daily", "digest", "ingest", "score", "status"]

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR: Final = Path("config")
DEFAULT_DB_PATH: Final = Path("data/signalforge.db")
DEFAULT_CACHE_DIR: Final = Path("data/http_cache")
# The default vault location lives on `SettingsConfig.vault_dir` (config, not
# code — CLAUDE.md §4), so both commands default the `--vault-dir` flag to None
# and fall back to settings when it is not passed. No Python-side constant here.

RUN_KIND_INGEST: Final = "ingest"
RUN_KIND_SCORE: Final = "score"
RUN_KIND_DIGEST: Final = "daily"
"""Matches the `runs.kind` vocabulary in DESIGN §5 (`ingest | score | daily | weekly | monthly`)."""

_RUN_LEVEL_SOURCE_ID: Final = "*"
"""`runs.errors[].source_id` for a failure that belongs to the run, not a source."""

_QUIET_AFTER_DAYS: Final = 7
"""When `status` starts *describing* a source as quiet. Not an alarm.

A blog with no new post in a fortnight is a blog, not an incident — Karpathy
posts every few months. The alarm in this table is NEVER SEEN; fetch failures
alarm through `runs.errors`, which is where a dead URL actually shows up. A
display threshold, so it stays here rather than in YAML (CLAUDE.md §4 governs
tuning knobs that change behaviour; this one changes a word)."""

app = typer.Typer(
    name="signalforge",
    help="Local-first AI engineering intelligence pipeline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _configure_logging(level: str) -> None:
    """Send stdlib logging to stderr, leaving stdout for rich's tables."""
    resolved = logging.getLevelNamesMapping().get(level.strip().upper())
    if resolved is None:
        raise typer.BadParameter(f"unknown log level {level!r}")
    logging.basicConfig(
        level=resolved,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Logging verbosity: DEBUG shows 304s and per-fetch detail."
        ),
    ] = "INFO",
) -> None:
    """Configure logging before any command runs."""
    _configure_logging(log_level)


def _load_sources_or_exit(config_dir: Path) -> SourcesConfig:
    try:
        return load_sources(config_dir)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _load_interests_or_exit(config_dir: Path) -> InterestsConfig:
    try:
        return load_interests(config_dir)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _load_settings_or_exit(config_dir: Path) -> SettingsConfig:
    try:
        return load_settings(config_dir)
    except ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _configured_sources(config: SourcesConfig) -> list[tuple[str, str]]:
    """Every `(source_id, source_type)` this config would ingest, in run order.

    Derived from `build_ingestors` rather than re-walked from the YAML: the
    ingestors define what a `source_id` *is* (a repo slug, a feed id, `hn`), and
    a second copy of that mapping here would drift the moment one changes — and
    the drift would show up as a source silently missing from `status`, which is
    exactly what `status` exists to catch.
    """
    return [
        (ingestor.source_id, ingestor.source_type.value) for ingestor in build_ingestors(config)
    ]


def _select_source(config: SourcesConfig, source_id: str) -> SourcesConfig:
    """Narrow `config` to the single source `source_id` (for `--source`).

    Filtering the *config* rather than the ingestor list keeps `ingest_all` the
    one entry point — the debug path and the cron path run identical code.
    """
    filtered = config.model_copy(deep=True)
    filtered.rss = [source for source in filtered.rss if source.id == source_id]
    if filtered.github is not None:
        filtered.github.releases = [repo for repo in filtered.github.releases if repo == source_id]
    if filtered.hackernews is not None and source_id != HN_SOURCE_ID:
        filtered.hackernews = None
    return filtered


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PersistOutcome:
    """What the persist loop achieved, and which sources it may confirm."""

    items_new: int = 0
    items_persisted: int = 0
    failed_sources: set[str] = field(default_factory=set)
    errors: list[IngestError] = field(default_factory=list)


def _persist(
    conn: sqlite3.Connection, items: list[Item], *, outcome: _PersistOutcome, run_id: int
) -> None:
    """Upsert `items`, grouped by source, isolating each group's failures.

    A group that raises is recorded and its `source_id` added to
    `outcome.failed_sources` so the caller withholds its validators — that
    source refetches next run rather than 304ing past items it never stored.

    `items_new` counts only `is_new=True` upserts and is incremented as each row
    lands, not per group: it must describe what is actually in the database, so
    that a group that raised halfway still reports the rows it managed to write
    (CLAUDE.md §3 — the double-run gate reads this number).

    `outcome` is owned by the caller rather than returned, so that a
    `BaseException` this function does not catch — a Ctrl-C mid-loop — still
    leaves the caller holding an accurate count of the rows `upsert_item`
    already committed. A returned value would be lost on that path and
    `runs.items_new` would understate the database.
    """
    by_source: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_source[item.source_id].append(item)

    for source_id, group in by_source.items():
        try:
            for item in group:
                _, is_new = upsert_item(conn, item)
                outcome.items_persisted += 1
                if is_new:
                    outcome.items_new += 1
        except Exception as exc:
            # Never re-raised: one source's write failure is a recorded error,
            # not the end of the run (NEVER rule 12).
            logger.exception(
                "persisting a source failed; other sources continue",
                extra={"source_id": source_id, "run_id": run_id},
            )
            outcome.failed_sources.add(source_id)
            outcome.errors.append(
                IngestError.from_exception(
                    exc, source_id=source_id, source_type=group[0].source_type
                )
            )
        else:
            logger.info(
                "persisted source",
                extra={"source_id": source_id, "run_id": run_id, "item_count": len(group)},
            )


def _commit_validators(run: IngestRun, *, failed_sources: set[str], run_id: int) -> int:
    """Make durable the validators of every source that did not fail.

    Deliberately *not* keyed off `run.source_ids` — see this module's docstring.
    A source that fetched a 200 and yielded no new entries is a success and must
    earn its 304.
    """
    fetch_failures = {error.source_id for error in run.errors}
    committable = run.validators.pending_sources() - fetch_failures - failed_sources
    written = run.commit_validators(committable)
    logger.info(
        "committed conditional-get validators",
        extra={
            "run_id": run_id,
            "committed": written,
            "withheld_sources": sorted(fetch_failures | failed_sources),
        },
    )
    return written


def _run_status(
    *, errors: list[dict[str, str]], items_persisted: int, failed_sources: int, attempted: int
) -> str:
    """Map the run's outcome onto `runs.status` (`ok | partial | failed`, DESIGN §5).

    `failed` means *the run accomplished nothing*, which is not the same as "no
    new items". DESIGN §7's whole point is that the steady state is every feed
    returning 304 and yielding zero items, so grading on item count alone would
    mark a healthy quiet morning with one dead feed as `failed` — cron would
    mail a failure every day until the noise was filtered out, and the one day
    it mattered would be filtered out with it.

    So a run is `partial` if *anything* worked: rows landed, or at least one
    source came back clean. Only a run where every attempted source failed is
    `failed`.
    """
    if not errors:
        return "ok"
    if items_persisted > 0:
        return "partial"
    # No rows landed. That is normal (all 304) unless every source also failed.
    return "failed" if attempted > 0 and failed_sources >= attempted else "partial"


def _collect_error_records(
    run: IngestRun | None,
    outcome: _PersistOutcome,
    *,
    run_level_error: dict[str, str] | None,
) -> list[dict[str, str]]:
    """Every failure this run knows about, from whatever stage it got to.

    Called from the `finally`, so it must tolerate a run that died anywhere:
    `run` is None when `ingest_all` itself raised, and `outcome` carries only
    the groups persisted before an interrupt. Both are read rather than
    required, which is what lets a Ctrl-C mid-persist still record the dead feed
    `ingest_all` had already found — the case where knowing which sources were
    broken matters most, and the one where assembling this list on the happy
    path would silently drop it.
    """
    return [
        *(run.error_records() if run is not None else []),
        *(error.as_record() for error in outcome.errors),
        *([run_level_error] if run_level_error is not None else []),
    ]


def _run_level_error(exc: BaseException) -> dict[str, str]:
    """A `runs.errors` record for a failure that belongs to no single source.

    Same shape as `IngestError.as_record()` so a reader of `runs.errors` never
    has to branch on which kind of failure it is looking at.
    """
    return {
        "source_id": _RUN_LEVEL_SOURCE_ID,
        "source_type": "-",
        "error_type": exc.__class__.__name__,
        "message": str(exc) or exc.__class__.__name__,
        "occurred_at": datetime.now(UTC).isoformat(),
    }


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


def _render_ingest_report(run: IngestRun, outcome: _PersistOutcome, *, dry_run: bool) -> None:
    counts: dict[str, int] = defaultdict(int)
    for item in run.items:
        counts[item.source_id] += 1

    table = Table(title="Ingest — items fetched per source", header_style="bold")
    table.add_column("source")
    table.add_column("items", justify="right")
    for source_id in sorted(counts):
        table.add_row(source_id, str(counts[source_id]))
    if not counts:
        table.add_row("[dim]no items[/dim]", "0")
    console.print(table)

    all_errors = [*run.error_records(), *(error.as_record() for error in outcome.errors)]
    if all_errors:
        errors_table = Table(title="Errors (recorded to runs.errors)", header_style="bold red")
        errors_table.add_column("source")
        errors_table.add_column("type")
        errors_table.add_column("message", overflow="fold")
        for record in all_errors:
            errors_table.add_row(record["source_id"], record["error_type"], record["message"])
        console.print(errors_table)

    if dry_run:
        console.print(
            f"[yellow]dry run[/yellow]: fetched {len(run.items)} item(s) from "
            f"{len(run.source_ids)} source(s); nothing written, no validators committed."
        )
    else:
        console.print(
            f"[green]ingest complete[/green]: {outcome.items_new} new / "
            f"{outcome.items_persisted} persisted item(s), {len(all_errors)} error(s)."
        )


@app.command()
def ingest(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding sources.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Raw payload + validator cache.")
    ] = DEFAULT_CACHE_DIR,
    source: Annotated[
        str | None,
        typer.Option("--source", help="Ingest only this source_id (feed id, repo slug, or 'hn')."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Fetch and report only: no rows, no run, no validators."),
    ] = False,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", min=1, help="Concurrent HTTP requests.")
    ] = DEFAULT_MAX_CONCURRENCY,
) -> None:
    """Fetch every configured source and upsert its items into SQLite.

    Idempotent by construction (CLAUDE.md §3): items upsert on
    `canonical_url` / `(source_id, external_id)`, so a second run adds zero rows.
    """
    config = _load_sources_or_exit(config_dir)

    if source is not None:
        known = {source_id for source_id, _ in _configured_sources(config)}
        if source not in known:
            err_console.print(
                f"[red]unknown source[/red] {source!r}; configured: {', '.join(sorted(known))}"
            )
            raise typer.Exit(code=2)
        config = _select_source(config, source)

    cache_dir.mkdir(parents=True, exist_ok=True)
    attempted = len(_configured_sources(config))

    if dry_run:
        # No DB is opened at all — a dry run must not create the database, a
        # `runs` row, or a validator. It is an inspection tool, not a run.
        preview = asyncio.run(
            ingest_all(config, cache_dir=cache_dir, max_concurrency=max_concurrency)
        )
        _render_ingest_report(preview, _PersistOutcome(), dry_run=True)
        return

    with connection(db) as conn:
        run_id = start_run(conn, RUN_KIND_INGEST, started_at=datetime.now(UTC))
        # Everything `finish_run` needs is owned out here, and every field is
        # accumulated as it happens rather than assembled on the happy path.
        # A Ctrl-C mid-persist must not be able to discard evidence that already
        # exists: the failures `ingest_all` recorded are most worth having in
        # exactly the run that then died (CLAUDE.md §7 — errors are never
        # swallowed; DESIGN §7 — `runs.errors` is the monitoring channel).
        run: IngestRun | None = None
        outcome = _PersistOutcome()
        run_level_error: dict[str, str] | None = None
        status_value = "failed"
        try:
            run = asyncio.run(
                ingest_all(config, cache_dir=cache_dir, max_concurrency=max_concurrency)
            )
            _persist(conn, run.items, outcome=outcome, run_id=run_id)
            # Only now — after the upserts returned — may a validator become
            # durable. Sources that failed are excluded and will refetch.
            _commit_validators(run, failed_sources=outcome.failed_sources, run_id=run_id)
            status_value = _run_status(
                errors=_collect_error_records(run, outcome, run_level_error=None),
                items_persisted=outcome.items_persisted,
                failed_sources=len(
                    {error.source_id for error in run.errors} | outcome.failed_sources
                ),
                attempted=attempted,
            )
            _render_ingest_report(run, outcome, dry_run=False)
        except BaseException as exc:
            # Includes KeyboardInterrupt: a Ctrl-C'd cron run still closes its
            # `runs` row (CLAUDE.md §3 — no silent runs). Recorded, then re-raised.
            run_level_error = _run_level_error(exc)
            # Rows that `upsert_item` already committed are real, so a run that
            # died after persisting some of them is `partial`, not a total loss —
            # the same reasoning `_persist` applies to `items_new`.
            status_value = "partial" if outcome.items_persisted > 0 else "failed"
            raise
        finally:
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                items_new=outcome.items_new,
                # Phase 0 ingest calls no LLM (CLAUDE.md §2), so the token
                # counters stay 0 — `status` still surfaces them as the alarm.
                llm_input_tokens=0,
                llm_output_tokens=0,
                errors=_collect_error_records(run, outcome, run_level_error=run_level_error)
                or None,
            )

    if status_value == "failed":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def _count_unscored_items(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM items
        LEFT JOIN scores ON scores.item_id = items.id
        WHERE scores.item_id IS NULL
        """
    ).fetchone()
    return int(row["n"])


def _render_score_report(outcome: ScoreOutcome, *, dry_run: bool, pending: int = 0) -> None:
    if dry_run:
        console.print(
            f"[yellow]dry run[/yellow]: {pending} unscored item(s); "
            "nothing sent to the LLM, no run recorded."
        )
        return

    if outcome.errors:
        errors_table = Table(title="Errors (recorded to runs.errors)", header_style="bold red")
        errors_table.add_column("item/source")
        errors_table.add_column("type")
        errors_table.add_column("message", overflow="fold")
        for record in outcome.errors:
            errors_table.add_row(record["source_id"], record["error_type"], record["message"])
        console.print(errors_table)

    console.print(
        f"[green]score complete[/green]: {outcome.items_scored} item(s) scored, "
        f"{len(outcome.errors)} error(s); "
        f"{outcome.input_tokens:,} input / {outcome.output_tokens:,} output tokens."
    )


@app.command()
def score(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding interests.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Report unscored item count only: no LLM call, no run recorded."
        ),
    ] = False,
) -> None:
    """Triage + score every currently-unscored item via batched Haiku calls.

    Reads only stored items (`items` left-joined against `scores`) and calls
    the LLM only through `signalforge.llm` — this command never touches
    `anthropic` directly (CLAUDE.md §2). Idempotent by construction
    (CLAUDE.md §3): an item that already carries a `scores` row is invisible
    to the selection query, so re-running `score` twice sends nothing already
    scored to the LLM and spends zero additional tokens.
    """
    interests = _load_interests_or_exit(config_dir)

    with connection(db) as conn:
        if dry_run:
            # No `runs` row, no LLM call — an inspection tool, not a run,
            # mirroring `ingest --dry-run`'s contract.
            pending = _count_unscored_items(conn)
            _render_score_report(ScoreOutcome(), dry_run=True, pending=pending)
            return

        run_id = start_run(conn, RUN_KIND_SCORE, started_at=datetime.now(UTC))
        run_level_error: dict[str, str] | None = None
        outcome = ScoreOutcome()
        status_value = "failed"
        try:
            outcome = score_unscored_items(conn, interests)
            if not outcome.errors:
                status_value = "ok"
            elif outcome.items_scored > 0:
                status_value = "partial"
            else:
                status_value = "failed"
            _render_score_report(outcome, dry_run=False)
        except BaseException as exc:
            # Includes KeyboardInterrupt, mirroring `ingest`'s no-silent-runs
            # rule (CLAUDE.md §3): a crash mid-batch still closes its `runs`
            # row. Scores already persisted by `score_unscored_items` are
            # real, so a run that died partway through is `partial`, not a
            # total loss.
            run_level_error = _run_level_error(exc)
            status_value = "partial" if outcome.items_scored > 0 else "failed"
            raise
        finally:
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                items_new=0,  # score writes no `items` rows, only `scores` rows.
                llm_input_tokens=outcome.input_tokens,
                llm_output_tokens=outcome.output_tokens,
                errors=[*outcome.errors, *([run_level_error] if run_level_error else [])] or None,
            )

    if status_value == "failed":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# digest
# --------------------------------------------------------------------------- #


@app.command()
def digest(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding interests.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
    vault_dir: Annotated[
        Path | None,
        typer.Option(
            "--vault-dir",
            help=(
                "Vault root (digests land in <vault>/daily/). Overrides "
                "settings.yaml `vault_dir`; unset falls back to it."
            ),
        ),
    ] = None,
    target_date: Annotated[
        datetime | None,
        typer.Option(
            "--date",
            formats=["%Y-%m-%d"],
            help=(
                "Digest date, YYYY-MM-DD (default: today, UTC). "
                "Re-rendering a date overwrites its file."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render to stdout only: no file written, no validators."),
    ] = False,
) -> None:
    """Render the Daily Digest (DESIGN §13) from already-scored items.

    Reads `items`/`scores`/`runs` only — no HTTP, no LLM call (CLAUDE.md §2):
    triage/scoring is the `score` command's job, this one only assembles and
    writes markdown. Idempotent by construction (CLAUDE.md §3): re-running for
    the same `--date` overwrites `<vault>/daily/<date>.md` rather than
    appending, because the query is a pure function of that date.

    Unlike `ingest`, there is no partial/failed distinction here — assembling
    and writing one file is a single step with nothing to isolate, so the run
    is either `ok` or `failed`.
    """
    # Only the `thresholds.daily_*` knobs are used here, but the digest reads
    # them from validated config like every other tuning knob (CLAUDE.md §4).
    interests = _load_interests_or_exit(config_dir)
    settings = _load_settings_or_exit(config_dir)
    tz = settings.tzinfo
    # Precedence: an explicit --vault-dir wins; otherwise settings.yaml decides
    # (which itself defaults to `vault/`). The flag default is None precisely so
    # "not passed" (fall back to settings) is distinguishable from "passed the
    # default value".
    effective_vault_dir = vault_dir if vault_dir is not None else settings.vault_dir
    # "Today" and the digest's day boundary are the operator's local calendar
    # (settings.yaml), not UTC — storage stays UTC, presentation is local. An
    # explicit --date is already a local calendar date.
    resolved_date = target_date.date() if target_date is not None else datetime.now(tz).date()

    with connection(db) as conn:
        run_id = start_run(conn, RUN_KIND_DIGEST, started_at=datetime.now(UTC))
        run_level_error: dict[str, str] | None = None
        status_value = "failed"
        item_count = 0
        try:
            context = build_digest_context(
                conn,
                target_date=resolved_date,
                tz=tz,
                max_items=interests.thresholds.daily_max_items,
                max_per_source=interests.thresholds.daily_max_per_source,
                max_per_github_repo=interests.thresholds.daily_max_per_github_repo,
            )
            item_count = len(context.items)
            rendered = render_digest(context)

            if dry_run:
                console.print(rendered)
                console.print(
                    f"[yellow]dry run[/yellow]: {item_count} item(s) would render; nothing written."
                )
            else:
                path = digest_path(effective_vault_dir, target_date=resolved_date)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rendered, encoding="utf-8")
                console.print(f"[green]digest written[/green]: {path} ({item_count} item(s)).")
            status_value = "ok"
        except BaseException as exc:
            # Includes KeyboardInterrupt, mirroring `ingest`'s no-silent-runs rule
            # (CLAUDE.md §3): a crash mid-render still closes its `runs` row.
            run_level_error = _run_level_error(exc)
            raise
        finally:
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                items_new=0,  # digest writes no `items` rows.
                llm_input_tokens=0,
                llm_output_tokens=0,
                errors=[run_level_error] if run_level_error is not None else None,
            )

    if status_value == "failed":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# daily
# --------------------------------------------------------------------------- #


@app.command()
def daily(
    config_dir: Annotated[Path, typer.Option("--config-dir")] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
    vault_dir: Annotated[Path | None, typer.Option("--vault-dir")] = None,
    max_concurrency: Annotated[int, typer.Option("--max-concurrency", min=1)] = (
        DEFAULT_MAX_CONCURRENCY
    ),
) -> None:
    """Run `ingest`, `score`, then `digest` in sequence (DESIGN §14, cron 06:00).

    Each step keeps its own `runs` row and failure isolation — a step that
    comes back `partial`/`failed` does not stop the next one from running,
    since each downstream step only cares about rows the previous one already
    committed (ingest's new items, score's new scores). This command never
    fixes "today" before `score` runs and then hands that fixed date to
    `digest`: `digest` always resolves `--date` itself, immediately after
    `score` returns, so a triage batch that happens to straddle UTC midnight
    still lands in the digest computed *after* it finished, not one decided
    before it started.

    Exit code is the worst of the three steps' (2 > 1 > 0), so a config error
    in any step is still visible to cron even though later steps still ran.
    """
    worst_exit = 0

    steps: tuple[Callable[[], None], ...] = (
        lambda: ingest(
            config_dir=config_dir,
            db=db,
            cache_dir=cache_dir,
            source=None,
            max_concurrency=max_concurrency,
        ),
        lambda: score(config_dir=config_dir, db=db),
        lambda: digest(config_dir=config_dir, db=db, vault_dir=vault_dir),
    )
    for step in steps:
        try:
            step()
        except typer.Exit as exc:
            worst_exit = max(worst_exit, exc.exit_code)
        except Exception:
            # A step already recorded its own `runs` row and re-raised
            # (CLAUDE.md §3 — no silent runs); `daily` isolates steps from
            # each other the same way `ingest` isolates sources. Only `Exception`
            # is caught, never `BaseException`: a KeyboardInterrupt/SystemExit
            # must abort the whole `daily` run, not be downgraded to exit 1 and
            # let the next step start — a human hitting Ctrl-C means stop.
            worst_exit = max(worst_exit, 1)

    if worst_exit:
        raise typer.Exit(code=worst_exit)


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _age(value: str | None, *, now: datetime) -> str:
    """Human "how long ago", or an em dash when never."""
    parsed = _parse_iso(value)
    if parsed is None:
        return "—"
    delta = now - parsed
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _is_quiet(value: str | None, *, now: datetime) -> bool:
    parsed = _parse_iso(value)
    if parsed is None:
        return True
    return (now - parsed).days >= _QUIET_AFTER_DAYS


def _render_last_runs(conn: sqlite3.Connection, *, now: datetime) -> None:
    rows = conn.execute(
        """
        SELECT kind, id, started_at, finished_at, status, items_new,
               llm_input_tokens, llm_output_tokens, errors
        FROM runs
        WHERE id IN (SELECT MAX(id) FROM runs GROUP BY kind)
        ORDER BY kind
        """
    ).fetchall()

    table = Table(title="Last run per kind", header_style="bold")
    table.add_column("kind")
    table.add_column("run")
    table.add_column("status")
    table.add_column("when")
    table.add_column("duration", justify="right")
    table.add_column("items_new", justify="right")
    table.add_column("errors", justify="right")

    if not rows:
        console.print(table)
        console.print("[yellow]no runs recorded yet — nothing has ever ingested.[/yellow]")
        return

    decoded_errors = [(row["kind"], _decode_errors(row["errors"])) for row in rows]

    for row, (_, errors) in zip(rows, decoded_errors, strict=True):
        started = _parse_iso(row["started_at"])
        finished = _parse_iso(row["finished_at"])
        duration = (
            f"{(finished - started).total_seconds():.1f}s"
            if started is not None and finished is not None
            else "[yellow]unfinished[/yellow]"
        )
        table.add_row(
            row["kind"],
            str(row["id"]),
            _status_cell(row["status"]),
            _age(row["started_at"], now=now),
            duration,
            str(row["items_new"] or 0),
            f"[red]{len(errors)}[/red]" if errors else "0",
        )
    console.print(table)

    for kind, errors in decoded_errors:
        _render_run_errors(kind, errors)


def _decode_errors(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return (
        [record for record in decoded if isinstance(record, dict)]
        if isinstance(decoded, list)
        else []
    )


def _status_cell(status: str | None) -> str:
    match status:
        case "ok":
            return "[green]ok[/green]"
        case "partial":
            return "[yellow]partial[/yellow]"
        case "failed":
            return "[red]failed[/red]"
        case _:
            return "[yellow]unfinished[/yellow]"


def _render_run_errors(kind: str, errors: list[dict[str, Any]]) -> None:
    if not errors:
        return
    table = Table(title=f"Errors from the last '{kind}' run", header_style="bold red")
    table.add_column("source")
    table.add_column("type")
    table.add_column("message", overflow="fold")
    for record in errors:
        table.add_row(
            str(record.get("source_id", "?")),
            str(record.get("error_type", "?")),
            str(record.get("message", "")),
        )
    console.print(table)


def _render_freshness(conn: sqlite3.Connection, config: SourcesConfig, *, now: datetime) -> None:
    """Per-source freshness over *configured* sources, not over stored items.

    The join direction is the whole point. `GROUP BY source_id` on `items` can
    only report sources that produced something, so a feed whose URL 404s every
    morning has no row at all and reads as absence-of-news rather than a dead
    source. Enumerating `sources.yaml` and left-joining the DB onto it makes a
    silently dark source impossible to miss — until digests exist, this table is
    the monitoring channel (DESIGN §7).

    ### What the timestamps do and do not mean

    `MAX(items.fetched_at)` is **not** "when we last fetched this source". A
    re-ingest preserves `fetched_at` as first-seen (`db.py::_merge_item`), which
    is what makes a double run byte-for-byte identical — so the column is "when
    we last saw a *new* item" and is named that. Calling it last-fetched would
    mark every healthy low-volume blog as a fetch failure.

    Fetch health therefore lives in the last run's `runs.errors`, rendered
    above: a URL that 404s produces a `FetchError` on *every* run. That leaves
    this table one alarm — NEVER SEEN — which is the case `runs.errors` cannot
    catch on its own: a source that 200s forever and yields nothing. Quiet is
    reported, not alarmed; an alarm that fires for every blog between posts is
    an alarm the user learns to skim, and DESIGN §1 rules a noisy report out.
    """
    stored = {
        row["source_id"]: row
        for row in conn.execute(
            """
            SELECT source_id,
                   COUNT(*)           AS item_count,
                   MAX(fetched_at)    AS last_new_item,
                   MAX(published_at)  AS last_published
            FROM items
            GROUP BY source_id
            """
        ).fetchall()
    }

    configured = _configured_sources(config)
    table = Table(title="Per-source freshness", header_style="bold")
    table.add_column("source")
    table.add_column("type")
    table.add_column("items", justify="right")
    table.add_column("last new item")
    table.add_column("last published")
    table.add_column("health")

    dark: list[str] = []
    quiet: list[str] = []

    for source_id, source_type in configured:
        row = stored.pop(source_id, None)
        if row is None:
            dark.append(source_id)
            table.add_row(
                f"[red]{source_id}[/red]",
                source_type,
                "[red]0[/red]",
                "—",
                "—",
                "[bold red]NEVER SEEN[/bold red]",
            )
            continue
        source_quiet = _is_quiet(row["last_new_item"], now=now)
        if source_quiet:
            quiet.append(source_id)
        table.add_row(
            source_id,
            source_type,
            str(row["item_count"]),
            _age(row["last_new_item"], now=now),
            _age(row["last_published"], now=now),
            "[dim]quiet[/dim]" if source_quiet else "[green]ok[/green]",
        )

    # Anything left in `stored` has items but is no longer configured — a
    # renamed or removed source. Harmless, but silently orphaning rows is the
    # kind of thing this table should say out loud.
    for source_id, row in sorted(stored.items()):
        table.add_row(
            f"[dim]{source_id}[/dim]",
            "[dim]—[/dim]",
            str(row["item_count"]),
            _age(row["last_new_item"], now=now),
            _age(row["last_published"], now=now),
            "[dim]not in config[/dim]",
        )

    console.print(table)

    if dark:
        console.print(
            f"[bold red]⚠ {len(dark)} configured source(s) have produced NO items:[/bold red] "
            f"{', '.join(dark)}"
        )
        console.print(
            "[red]  A source that never appears is indistinguishable from quiet news. "
            "Check the URL — run `signalforge --log-level DEBUG ingest --source <id>`.[/red]"
        )
    else:
        console.print(
            f"[green]all {len(configured)} configured sources have produced items.[/green]"
        )
    if quiet:
        # Stated, not alarmed: a quiet blog is a blog. Fetch failures alarm
        # through the errors table above.
        console.print(
            f"[dim]{len(quiet)} source(s) have had no new item in {_QUIET_AFTER_DAYS} days: "
            f"{', '.join(quiet)}[/dim]"
        )


def _render_token_spend(conn: sqlite3.Connection, *, now: datetime) -> None:
    """Month-to-date token spend — 0 until triage lands, and shown anyway.

    DESIGN §8 makes spend the number this project lives or dies on ($30 is the
    alarm). A cost readout that only appears once there is cost to report is a
    cost readout nobody has ever looked at when it matters.
    """
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(llm_input_tokens), 0)  AS input_tokens,
               COALESCE(SUM(llm_output_tokens), 0) AS output_tokens,
               COUNT(*)                            AS run_count
        FROM runs
        WHERE started_at >= ?
        """,
        (month_start,),
    ).fetchone()

    table = Table(title=f"Month-to-date ({now:%Y-%m})", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("runs", str(row["run_count"]))
    table.add_row("LLM input tokens", f"{row['input_tokens']:,}")
    table.add_row("LLM output tokens", f"{row['output_tokens']:,}")
    console.print(table)


@app.command()
def status(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding sources.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
) -> None:
    """Show last-run health, per-source freshness, and month-to-date token spend."""
    config = _load_sources_or_exit(config_dir)
    now = datetime.now(UTC)
    with connection(db) as conn:
        _render_last_runs(conn, now=now)
        _render_freshness(conn, config, now=now)
        _render_token_spend(conn, now=now)
