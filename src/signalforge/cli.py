"""The typer CLI — the only place config, `ingest/`, and `db.py` meet.

Phase 0 surface, and deliberately no more (NEVER rule 15):

* `ingest` — fetch every configured source, persist the items, close a `runs` row.
* `score` — triage/score unscored items via `score/` (DESIGN §8).
* `digest` — assemble the day's markdown digest from scored items (DESIGN §13).
* `curate run|apply|list|decide|reopen` — adaptive source curation (DESIGN §7.1):
  propose `sources.yaml` changes, review them in the digest, apply the approved.
  `run` is the only command in this CLI that spends money on being typed.
* `podcast` — write the day's episode script and publish it to the podcast
  channel, if configured (DESIGN §13.3). Harvests vault marks first and ranks
  the episode by them: `noise` out, then exceptional → useful → unmarked.
* `daily` — `curate apply` → `ingest` → `score` → `digest` → `podcast`. Split
  across two cron entries (DESIGN §14): `daily --no-podcast` in the evening,
  `podcast --date <yesterday>` the next morning, with the gap as the operator's
  window to mark the digest.
* `status` — last-run health, per-source freshness, month-to-date token,
  web-search, and TTS spend (DESIGN §14).

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
from datetime import UTC, datetime, timedelta, tzinfo
from datetime import date as Date
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from signalforge import feedback
from signalforge.config import (
    SETTINGS_FILENAME,
    TAXONOMY_FILENAME,
    ConfigError,
    CurationConfig,
    InterestsConfig,
    SettingsConfig,
    SourcesConfig,
    TaxonomyConfig,
    get_secret,
    load_interests,
    load_settings,
    load_sources,
    load_taxonomy,
)
from signalforge.curate.apply import apply_approved_proposals
from signalforge.curate.approvals import DECISIONS, harvest_approvals
from signalforge.curate.scout import (
    Candidate,
    ScoutOutcome,
    persist_candidates,
    reprobe_invalid_proposals,
    scout_for_proposals,
    search_spend_usd,
)
from signalforge.db import (
    connection,
    decide_proposal,
    feedback_verdicts_for_items,
    finish_run,
    get_digest_items,
    get_item,
    get_proposal,
    get_proposals,
    record_feedback,
    reopen_proposal,
    start_run,
    upsert_item,
)
from signalforge.deliver import DeliveryOutcome, deliver_digest, send_test_message
from signalforge.deliver.podcast import deliver_podcast
from signalforge.deliver.tts import tts_spend_usd
from signalforge.feedback import harvest_marks
from signalforge.ingest import IngestError, IngestRun, build_ingestors, ingest_all
from signalforge.ingest.arxiv import ARXIV_SOURCE_ID
from signalforge.ingest.awesome import AWESOME_SOURCE_PREFIX
from signalforge.ingest.base import DEFAULT_MAX_CONCURRENCY, HttpFetcher
from signalforge.ingest.fullcontent import fetch_full_content
from signalforge.ingest.hackernews import HN_SOURCE_ID
from signalforge.models import Item, ProposalStatus
from signalforge.report.daily import (
    DigestContext,
    build_digest_context,
    digest_path,
    render_digest,
    select_digest_items,
    utc_day_window,
)
from signalforge.report.podcast import order_by_verdict, script_path, write_script
from signalforge.report.vaultgit import VaultCommitStatus, commit_vault
from signalforge.report.weekly import (
    WEEKLY_WINDOW_DAYS,
    WeeklySelection,
    brief_path,
    select_weekly_items,
    weekly_window,
    write_brief,
)
from signalforge.score import ScoreOutcome, score_unscored_items
from signalforge.score.taxonomy import stale_topics, tag_untagged_items
from signalforge.synth.podcast import build_podcast_stable_prefix, build_script
from signalforge.synth.weekly import build_brief, build_weekly_stable_prefix

__all__ = [
    "app",
    "curate_app",
    "daily",
    "deliver_app",
    "digest",
    "ingest",
    "mark",
    "podcast",
    "score",
    "status",
]

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR: Final = Path("config")
DEFAULT_DB_PATH: Final = Path("data/signalforge.db")
DEFAULT_CACHE_DIR: Final = Path("data/http_cache")
DEFAULT_AUDIO_DIR: Final = Path("data/audio")
"""Where synthesized episode mp3s are cached locally — regenerable plumbing,
not backed up (DESIGN §14), so it is a code default like `DEFAULT_DB_PATH`
rather than a `settings.yaml` knob."""
# The default vault location lives on `SettingsConfig.vault_dir` (config, not
# code — CLAUDE.md §4), so both commands default the `--vault-dir` flag to None
# and fall back to settings when it is not passed. No Python-side constant here.

RUN_KIND_INGEST: Final = "ingest"
RUN_KIND_SCORE: Final = "score"
RUN_KIND_DIGEST: Final = "daily"
RUN_KIND_CURATE: Final = "curate"
RUN_KIND_CURATE_APPLY: Final = "curate-apply"
RUN_KIND_PODCAST: Final = "podcast"
RUN_KIND_WEEKLY: Final = "weekly"
"""Matches the `runs.kind` vocabulary in DESIGN §5.

`curate` and `curate-apply` are separate kinds on purpose: one makes a paid Opus
call and the other edits a file for free, and `status` shows the last run of each
kind. Folding them together would hide a scout that has stopped running behind an
apply that runs every morning."""

_SUNDAY: Final = 6
"""`date.weekday()` for Sunday — Monday is 0. A brief is dated by the Sunday it is
published on, and `_resolve_target_sunday` refuses any other day."""


def _weekly_window_label(target_sunday: Date) -> str:
    """The window as the brief's prompt names it. Prose only — the actual window
    is `report.weekly.weekly_window`, and this must never become a second,
    drifting definition of it."""
    start = target_sunday - timedelta(days=WEEKLY_WINDOW_DAYS)
    end = target_sunday - timedelta(days=1)
    return f"{start.isoformat()} to {end.isoformat()}"


_MIN_CURATE_INTERVAL_DAYS: Final = 6
"""How recently a `curate` run may have happened before another is refused.

A spend backstop, not a schedule: `curate run` is the one command here that costs
money on being typed, and nothing else stops it running daily. The unique index
prevents a re-run *storing* duplicates and does nothing about the call being billed
again — wired into `daily` by mistake, that is ~$7/month expected against a $2.50
budget for this feature. 6 rather than 7 so a weekly cron that drifts an hour is
never refused. `--force` overrides it, which is what an operator tuning the prompt
wants. In code rather than YAML for the same reason `SCOUT_MAX_SEARCHES_CEILING` is
(CLAUDE.md §6): cost limits are this project's to set, not a knob to tune."""

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


def _load_taxonomy_or_none(config_dir: Path) -> TaxonomyConfig | None:
    """Load `taxonomy.yaml`, or None when there isn't one.

    The one loader here that does not exit on failure, and the asymmetry is
    deliberate. Tagging is *additive* — a digest without topic tags is the
    digest this pipeline shipped for its whole first phase — so an operator
    with no `taxonomy.yaml` should get scoring, not an exit code. A file that
    exists but does not validate is a different thing entirely: that is a typo
    silently costing the operator their topics, so it is reported loudly and
    still does not fail the scoring the run was actually for (CLAUDE.md §7).
    """
    if not (config_dir / TAXONOMY_FILENAME).is_file():
        return None
    try:
        return load_taxonomy(config_dir)
    except ConfigError as exc:
        err_console.print(f"[yellow]taxonomy config error[/yellow]: {exc}")
        err_console.print("[yellow]continuing without topic tagging[/yellow].")
        return None


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
        # Awesome lists carry a prefixed source_id (`awesome:owner/repo`), so
        # they need their own filter. Without one, debugging a single feed would
        # fetch every list, emit its items, and advance its baseline — the
        # opposite of what `--source` promises.
        filtered.github.awesome_lists = [
            repo
            for repo in filtered.github.awesome_lists
            if f"{AWESOME_SOURCE_PREFIX}{repo}" == source_id
        ]
    if filtered.hackernews is not None and source_id != HN_SOURCE_ID:
        filtered.hackernews = None
    if filtered.arxiv is not None and source_id != ARXIV_SOURCE_ID:
        filtered.arxiv = None
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
            # `message` is world-authored — a server's response body reaches here —
            # and rich reads `[...]` in a cell as markup, so an unescaped bracket
            # raises `MarkupError` mid-report. Escaped at every render site.
            errors_table.add_row(
                escape(record["source_id"]),
                escape(record["error_type"]),
                escape(record["message"]),
            )
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
        typer.Option(
            "--source", help="Ingest only this source_id (feed id, repo slug, 'hn', or 'arxiv')."
        ),
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
            errors_table.add_row(
                escape(record["source_id"]),
                escape(record["error_type"]),
                escape(record["message"]),
            )
        console.print(errors_table)

    console.print(
        f"[green]score complete[/green]: {outcome.items_scored} item(s) scored, "
        f"{len(outcome.errors)} error(s); "
        f"{outcome.input_tokens:,} input / {outcome.output_tokens:,} output tokens."
    )


def _tag_and_report(conn: sqlite3.Connection, taxonomy: TaxonomyConfig) -> list[dict[str, str]]:
    """Run the deterministic topic tagger and print what it did.

    Never raises and never changes the run's status — same posture as
    `_deliver_and_report` and `_commit_vault_and_report`, for the same reason:
    this is additive work after the thing the run was for already succeeded.
    """
    try:
        outcome = tag_untagged_items(conn, taxonomy)
    except Exception as exc:
        logger.exception("topic tagging failed; scores are unaffected")
        err_console.print(f"[yellow]topic tagging failed[/yellow]: {escape(str(exc))}")
        return [_run_level_error(exc)]

    if outcome.topics_written:
        console.print(
            f"[green]tagged[/green]: {outcome.items_tagged} item(s), "
            f"{outcome.topics_written} topic(s) across {outcome.items_examined} examined."
        )
    elif outcome.items_examined:
        console.print(
            f"[yellow]tagged[/yellow]: 0 of {outcome.items_examined} examined item(s) "
            "matched a taxonomy keyword."
        )
    return outcome.errors


@app.command()
def score(
    config_dir: Annotated[
        Path,
        typer.Option("--config-dir", help="Directory holding interests.yaml and taxonomy.yaml."),
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

    Also runs the deterministic topic tagger (DESIGN §10), which costs nothing
    and calls nothing — it is here rather than in its own command because the
    two share a trigger ("new items have arrived") and neither is worth a cron
    line of its own.
    """
    interests = _load_interests_or_exit(config_dir)
    taxonomy = _load_taxonomy_or_none(config_dir)

    with connection(db) as conn:
        if dry_run:
            # No `runs` row, no LLM call — an inspection tool, not a run,
            # mirroring `ingest --dry-run`'s contract.
            pending = _count_unscored_items(conn)
            _render_score_report(ScoreOutcome(), dry_run=True, pending=pending)
            return

        run_id = start_run(conn, RUN_KIND_SCORE, started_at=datetime.now(UTC))
        run_level_error: dict[str, str] | None = None
        tag_errors: list[dict[str, str]] = []
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
            # After scoring, and outside its status arithmetic. Tagging spends
            # nothing and is not what the run is *for*, so a tagging problem is
            # a recorded error and a printed line, never the difference between
            # an `ok` and a `failed` scoring run (CLAUDE.md §7).
            if taxonomy is not None:
                tag_errors = _tag_and_report(conn, taxonomy)
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
                errors=[
                    *outcome.errors,
                    *tag_errors,
                    *([run_level_error] if run_level_error else []),
                ]
                or None,
            )

    if status_value == "failed":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# digest
# --------------------------------------------------------------------------- #


def _deliver_and_report(
    conn: sqlite3.Connection,
    context: DigestContext,
    *,
    settings: SettingsConfig,
    today: Date,
    run_id: int | None,
    resend: bool,
) -> list[dict[str, str]]:
    """Mirror the digest to configured channels, print what happened, return errors.

    Never raises and never changes the run's status: the vault file is the product
    and it is already written by the time this is called, so a dead mail provider
    is a recorded error and a printed line, not a failed run (§7, NEVER rule 12).

    Every outcome is printed, including refusals. A silent "no email today" is
    indistinguishable from a broken pipeline.
    """
    try:
        outcomes = deliver_digest(
            conn, context, settings=settings, today=today, run_id=run_id, resend=resend
        )
        return _report_delivery_outcomes(outcomes)
    except Exception as exc:  # pragma: no cover — deliver_digest catches its own
        # A backstop, not dead code. `deliver_digest` turns a failed send into an
        # outcome, but this net means a future channel that forgets to cannot take
        # the run down with it. `BaseException` deliberately passes through, so
        # Ctrl-C during delivery still marks the run failed — the same rule
        # `ingest` and the render already follow.
        logger.exception("delivery raised unexpectedly; the digest is already written")
        return [_run_level_error(exc)]


def _report_delivery_outcomes(outcomes: list[DeliveryOutcome]) -> list[dict[str, str]]:
    """Print each outcome and return the ones that belong in `runs.errors`.

    Every `reason` here is escaped before it reaches rich, because a provider
    refusal carries the response body verbatim and rich reads `[...]` as markup.
    An error body containing `[/v1]` raised `MarkupError` from inside the printing
    loop, which escaped to `digest`'s handler and marked the run **failed** — a mail
    provider's error text failing the digest, which is exactly what NEVER rules 12
    and 19 forbid. The milder version silently ate the text.
    """
    errors: list[dict[str, str]] = []
    for outcome in outcomes:
        if outcome.error is not None:
            errors.append(_run_level_error(outcome.error))
        channel = escape(outcome.channel)
        reason = escape(outcome.reason)
        # Branch on `sent` first, not on `error`. There is one outcome that carries
        # both — a message that reached the inbox but whose send-log row conflicted,
        # meaning a duplicate went out — and telling the operator "not delivered"
        # there would be the opposite of what happened.
        if outcome.sent and outcome.error is not None:
            err_console.print(f"[yellow]{channel} delivered, with a problem[/yellow]: {reason}")
        elif outcome.sent:
            provider = escape(outcome.provider_id) if outcome.provider_id else ""
            suffix = f" (provider id {provider})" if provider else ""
            console.print(f"[green]{channel} delivered[/green]: {reason}{suffix}")
        elif outcome.error is not None:
            err_console.print(f"[red]{channel} not delivered[/red]: {reason}")
        else:
            console.print(f"[yellow]{channel} skipped[/yellow]: {reason}")
    return errors


def _commit_vault_and_report(
    *, settings: SettingsConfig, vault_dir: Path, message: str
) -> list[dict[str, str]]:
    """Snapshot the just-written report into the vault's git history, and say so.

    Called only *after* a vault write succeeded — the vault is written first and
    always (DESIGN §13.2). Returns the records that belong in `runs.errors`,
    which is at most one and only for a genuine git failure: a guarded refusal
    (vault is not its own repo) is a correct outcome, not a run error, and must
    not accumulate one every evening for an operator who never opted in.

    `detail` is escaped before printing for the same reason
    `_report_delivery_outcomes` escapes a provider body: git's stderr can
    contain `[...]`, which rich reads as markup, and a `MarkupError` raised here
    would fail the run over a *successful* report (NEVER rules 12 and 19).
    """
    outcome = commit_vault(vault_dir, message=message, enabled=settings.vault_git.enabled)
    detail = escape(outcome.detail)

    if outcome.status is VaultCommitStatus.COMMITTED:
        console.print(f"[green]vault committed[/green]: {outcome.commit_sha}")
    elif outcome.status is VaultCommitStatus.FAILED:
        err_console.print(f"[red]vault not committed[/red]: {detail}")
    elif outcome.status in (VaultCommitStatus.NOT_REPO_ROOT, VaultCommitStatus.GIT_UNAVAILABLE):
        # Worth a visible line: the operator probably meant this to work.
        err_console.print(f"[yellow]vault not committed[/yellow]: {detail}")

    if not outcome.is_error:
        return []
    # Built inline rather than through `_run_level_error`, which takes an
    # exception: nothing was raised here by design, and inventing one just to
    # unwrap it would be theatre. Same record shape either way.
    return [
        {
            "source_id": _RUN_LEVEL_SOURCE_ID,
            "source_type": "-",
            "error_type": "VaultCommitError",
            "message": outcome.detail or outcome.status.value,
            "occurred_at": datetime.now(UTC).isoformat(),
        }
    ]


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
    send: Annotated[
        bool,
        typer.Option(
            "--send/--no-send",
            help=(
                "Mirror the digest to configured delivery channels (settings.yaml "
                "`delivery:`). On by default when a channel is enabled; --dry-run "
                "always suppresses it."
            ),
        ),
    ] = True,
    resend: Annotated[
        bool,
        typer.Option(
            "--resend",
            help=(
                "Send again even if this date was already delivered. Does not "
                "override the freshness window — an old digest is never mailed."
            ),
        ),
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
    # `sources.yaml` is loaded for one field: how long a decided curation proposal
    # keeps rendering. Absent `curation:` block means the feature is off, and 0 days
    # is the honest reading of that — not a Python default standing in for config
    # (CLAUDE.md §4).
    sources = _load_sources_or_exit(config_dir)
    settled_display_days = sources.curation.settled_display_days if sources.curation else 0
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
        harvest_error: dict[str, str] | None = None
        delivery_errors: list[dict[str, str]] = []
        vault_commit_errors: list[dict[str, str]] = []
        status_value = "failed"
        item_count = 0

        # Harvest thumbs-up/down marks out of the vault *before* the render
        # overwrites today's file (DESIGN §11 harvest-then-overwrite). Skipped on
        # --dry-run: a preview must never write ground-truth feedback rows. Run
        # after start_run and non-fatally, so a broken harvest is captured into
        # this run's `runs.errors` — surfaced by `status` (DESIGN §7's monitoring
        # channel), not just cron.log — while the digest still renders (§7). Vault
        # markdown is only read here, never written (NEVER rule 8).
        if not dry_run:
            try:
                result = harvest_marks(conn, effective_vault_dir)
                if result.rows_recorded:
                    console.print(
                        f"[green]harvested feedback[/green]: {result.rows_recorded} new mark(s) "
                        f"after scanning {result.files_scanned} vault file(s)."
                    )
            except Exception as exc:
                logger.exception(
                    "harvesting vault feedback failed; continuing to the digest render"
                )
                harvest_error = _run_level_error(exc)

        try:
            context = build_digest_context(
                conn,
                target_date=resolved_date,
                tz=tz,
                max_items=interests.thresholds.daily_max_items,
                max_per_source=interests.thresholds.daily_max_per_source,
                max_per_github_repo=interests.thresholds.daily_max_per_github_repo,
                settled_display_days=settled_display_days,
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
                vault_commit_errors = _commit_vault_and_report(
                    settings=settings,
                    vault_dir=effective_vault_dir,
                    message=f"vault: daily digest {resolved_date.isoformat()}",
                )
                # Only after the vault write succeeded. The vault is canonical
                # (DESIGN §13.2); a channel is a mirror of a file that already
                # exists, never a substitute for writing it. Suppressed on
                # --dry-run for the same reason the harvest is: a preview must
                # have no side effects the operator did not ask for.
                if send:
                    delivery_errors = _deliver_and_report(
                        conn,
                        context,
                        settings=settings,
                        today=datetime.now(tz).date(),
                        run_id=run_id,
                        resend=resend,
                    )
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
                # A non-fatal harvest failure rides alongside a fatal render
                # error so both reach `runs.errors` (DESIGN §7). A harvest error
                # with a clean render leaves `status` "ok" with a visible error
                # count — the render (the product) succeeded, the side-channel
                # didn't. Delivery failures ride the same way, for the same
                # reason: the digest is on disk either way.
                errors=(
                    [e for e in (harvest_error, run_level_error) if e is not None]
                    + delivery_errors
                    + vault_commit_errors
                )
                or None,
            )

    if status_value == "failed":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# podcast (DESIGN §13.3 — the second recorded phase-gate exception)
# --------------------------------------------------------------------------- #


def _deliver_podcast_and_report(
    conn: sqlite3.Connection,
    *,
    settings: SettingsConfig,
    target_date: Date,
    vault_dir: Path,
    audio_dir: Path,
    today: Date,
    run_id: int | None,
    resend: bool,
) -> tuple[list[dict[str, str]], int]:
    """Mirror the episode script to the podcast feed; print what happened.

    Never raises (mirrors `_deliver_and_report`): the vault script is already
    written by the time this is called, so a dead R2 or TTS provider is a
    recorded error and a printed line, not a failed run (§7, NEVER rule 12,
    NEVER rule 19). Returns `(errors, tts_characters)` so the caller can fold
    both into `finish_run` regardless of which branch below produced them.
    """
    try:
        outcome = deliver_podcast(
            conn,
            settings=settings,
            target_date=target_date,
            vault_dir=vault_dir,
            audio_dir=audio_dir,
            today=today,
            run_id=run_id,
            resend=resend,
        )
    except Exception as exc:  # pragma: no cover — deliver_podcast catches its own
        # Same backstop `_deliver_and_report` keeps for `deliver_digest`: a
        # future failure mode `deliver_podcast` forgets to catch must not be
        # able to take the run down with it.
        logger.exception("podcast delivery raised unexpectedly; the script is already written")
        return [_run_level_error(exc)], 0

    if outcome is None:
        # Channel unconfigured or disabled — silent, matching `deliver_podcast`'s
        # own "nothing to report" contract; there is no channel to name.
        return [], 0

    reason = escape(outcome.reason)
    if outcome.sent:
        provider = escape(outcome.provider_id) if outcome.provider_id else ""
        suffix = f" (provider id {provider})" if provider else ""
        console.print(f"[green]{outcome.channel} delivered[/green]: {reason}{suffix}")
    elif outcome.error is not None:
        err_console.print(f"[red]{outcome.channel} not delivered[/red]: {reason}")
    else:
        console.print(f"[yellow]{outcome.channel} skipped[/yellow]: {reason}")

    errors = [_run_level_error(outcome.error)] if outcome.error is not None else []
    return errors, outcome.tts_characters


def _select_podcast_items(
    conn: sqlite3.Connection,
    *,
    interests: InterestsConfig,
    target_date: Date,
    tz: tzinfo,
    podcast_top_n: int,
) -> list[Item]:
    """The top-`podcast_top_n` kept items for `target_date`, preferring the
    operator's marks over the model's score (DESIGN §13.3).

    Three filters, in this order, and the order is the whole design:

    1. **Citable.** Items with no URL are dropped first, the same defensive
       check `report.daily._to_line` applies, since an aired segment with an
       unresolvable source is exactly what `report/podcast.py`'s citation
       discipline (NEVER rule 7) would drop at render time anyway — catching it
       here keeps it out of a top-N slot it would otherwise waste.
    2. **Marked.** `order_by_verdict` drops `noise` and re-tiers the rest
       exceptional → useful → unmarked, preserving score order inside each tier.
    3. **Crowded.** `select_digest_items` — the digest's own per-source and
       per-repo limits, capped at `podcast_top_n` rather than `daily_max_items`,
       a different Opus/TTS-priced knob (CLAUDE.md §6).

    Tiering *before* crowding is deliberate: the caps then truncate the
    operator's preferred order, so two `exceptional` items from one source still
    beat an unmarked third from elsewhere only as far as `daily_max_per_source`
    allows. Running it the other way would let the score ranking spend the
    episode's slots before a single mark was consulted.
    """
    start, end = utc_day_window(target_date, tz)
    scored_items = get_digest_items(conn, start=start, end=end)
    citable = [scored for scored in scored_items if scored.item.url]
    verdicts = feedback_verdicts_for_items(
        conn, [scored.item.id for scored in citable if scored.item.id is not None]
    )
    selected = select_digest_items(
        order_by_verdict(citable, verdicts),
        max_items=podcast_top_n,
        max_per_source=interests.thresholds.daily_max_per_source,
        max_per_github_repo=interests.thresholds.daily_max_per_github_repo,
    )
    return [scored.item for scored in selected]


async def _deep_read(
    conn: sqlite3.Connection, items: list[Item], *, cache_dir: Path, timeout: float
) -> list[dict[str, str]]:
    async with HttpFetcher(cache_dir=cache_dir, timeout=timeout) as fetcher:
        result = await fetch_full_content(fetcher, conn, items)
    if result.errors:
        console.print(
            f"[dim]deep-read: {result.fetched} item(s) backfilled, "
            f"{len(result.errors)} fetch/extract error(s) (items degrade to "
            "title+summary).[/dim]"
        )
    # Recorded into `runs.errors`, not just logged (CLAUDE.md §7's monitoring
    # channel): a paywalled or dead article must be as visible to `status` as
    # any other source failure, even though the item itself still airs on
    # title+summary alone rather than aborting the episode.
    return [error.as_record() for error in result.errors]


def _episode_date_label(target_date: Date) -> str:
    return f"{target_date:%A}, {target_date.day} {target_date:%B} {target_date.year}"


@app.command()
def podcast(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding interests.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Deep-read raw payload cache.")
    ] = DEFAULT_CACHE_DIR,
    vault_dir: Annotated[
        Path | None,
        typer.Option(
            "--vault-dir", help="Vault root (scripts land in <vault>/podcast/). See `digest`."
        ),
    ] = None,
    audio_dir: Annotated[
        Path | None,
        typer.Option(
            "--audio-dir", help="Local mp3 cache root. Overrides the data/audio/ default."
        ),
    ] = None,
    target_date: Annotated[
        datetime | None,
        typer.Option(
            "--date",
            formats=["%Y-%m-%d"],
            help="Episode date, YYYY-MM-DD (default: today, operator timezone).",
        ),
    ] = None,
    force_script: Annotated[
        bool,
        typer.Option(
            "--force-script",
            help="Regenerate the script even if today's vault file already exists "
            "(a fresh, billed Opus call).",
        ),
    ] = False,
    resend: Annotated[
        bool,
        typer.Option("--resend", help="Publish again even if this date was already delivered."),
    ] = False,
) -> None:
    """Write the day's podcast script (Opus) and publish it (TTS + R2 feed).

    The second recorded phase-gate exception (DESIGN §13.3). Item selection
    reuses `digest`'s own ranking and crowding limits, capped at
    `interests.yaml`'s `thresholds.podcast_top_n` rather than
    `daily_max_items` — a different, Opus-priced knob (CLAUDE.md §6, NEVER
    rule 9's sibling for Phase 1 synthesis spend).

    **The operator's marks outrank the score.** Vault feedback is harvested
    first (read-only, non-fatal), then `noise` items are dropped and the rest
    tiered exceptional → useful → unmarked before the cap applies — see
    `_select_podcast_items`. Unmarked is a tier, not an exclusion, so a day
    nobody ticked anything still yields a normal score-ranked episode: marking
    is upside, never a gate on the episode existing. This is why the default
    schedule records the episode the morning *after* its digest (DESIGN §14) —
    same day, and there is no window in which to have marked anything.

    Idempotent by construction
    (CLAUDE.md §3): a vault script already on disk for `--date` is reused
    rather than regenerated (`--force-script` overrides), and `deliver_podcast`
    is itself a no-op on a date already delivered.

    Presenter names live in `settings.yaml`'s `delivery.podcast:` block, the
    same block TTS/R2 read — so a script cannot be written at all until that
    block exists, even with `enabled: false` (a "configured, not yet turned
    on" state used to review Stage 3/4 scripts before Stage 5 could publish
    them). Either `thresholds.podcast_top_n` unset or no `delivery.podcast:`
    block means the channel is off: a clean "nothing to record" exit, same as
    a day with no kept items.
    """
    interests = _load_interests_or_exit(config_dir)
    settings = _load_settings_or_exit(config_dir)
    sources = _load_sources_or_exit(config_dir)
    tz = settings.tzinfo
    effective_vault_dir = vault_dir if vault_dir is not None else settings.vault_dir
    effective_audio_dir = audio_dir if audio_dir is not None else DEFAULT_AUDIO_DIR
    resolved_date = target_date.date() if target_date is not None else datetime.now(tz).date()

    podcast_top_n = interests.thresholds.podcast_top_n
    channel_config = settings.delivery.podcast

    with connection(db) as conn:
        run_id = start_run(conn, RUN_KIND_PODCAST, started_at=datetime.now(UTC))
        run_level_error: dict[str, str] | None = None
        harvest_error: dict[str, str] | None = None
        deep_read_errors: list[dict[str, str]] = []
        delivery_errors: list[dict[str, str]] = []
        vault_commit_errors: list[dict[str, str]] = []
        input_tokens = output_tokens = 0
        tts_characters = 0
        item_count = 0
        status_value = "failed"
        try:
            if podcast_top_n is None or channel_config is None:
                console.print(
                    "[yellow]podcast channel is off[/yellow]: needs both "
                    "`interests.yaml` thresholds.podcast_top_n and a `settings.yaml` "
                    "delivery.podcast: block. Nothing to record."
                )
                status_value = "ok"
            else:
                # Harvested here, not just in `digest`. The episode is built the
                # morning *after* its digest was written (DESIGN §14), so the marks
                # that decide this episode's running order were ticked since the
                # last `digest` run and live only in the vault markdown until now —
                # without this the podcast would rank against yesterday's marks and
                # the whole reorder would be a day late. Non-fatal and vault-read-
                # only, exactly as `digest` does it (NEVER rule 8): a broken harvest
                # costs the marks, not the episode, and lands in `runs.errors` where
                # `status` will surface it (CLAUDE.md §7). Re-harvesting an already
                # -stored checkbox is a no-op, so running both is safe (DESIGN §11).
                try:
                    harvested = harvest_marks(conn, effective_vault_dir)
                    if harvested.rows_recorded:
                        console.print(
                            f"[green]harvested feedback[/green]: {harvested.rows_recorded} new "
                            f"mark(s) after scanning {harvested.files_scanned} vault file(s)."
                        )
                except Exception as exc:
                    logger.exception("harvesting vault feedback failed; continuing to the episode")
                    harvest_error = _run_level_error(exc)

                items = _select_podcast_items(
                    conn,
                    interests=interests,
                    target_date=resolved_date,
                    tz=tz,
                    podcast_top_n=podcast_top_n,
                )
                if not items:
                    console.print(
                        f"[yellow]nothing to record[/yellow]: no kept items for "
                        f"{resolved_date.isoformat()}."
                    )
                    status_value = "ok"
                else:
                    item_count = len(items)
                    script_file = script_path(effective_vault_dir, date=resolved_date)
                    if script_file.is_file() and not force_script:
                        console.print(
                            f"[dim]script already exists at {script_file}; skipping the "
                            "Opus call (--force-script to regenerate).[/dim]"
                        )
                    else:
                        deep_read_errors = asyncio.run(
                            _deep_read(
                                conn,
                                items,
                                cache_dir=cache_dir,
                                timeout=sources.defaults.fetch_timeout,
                            )
                        )
                        refreshed_items = [
                            refreshed
                            for item in items
                            if item.id is not None
                            and (refreshed := get_item(conn, item.id)) is not None
                        ]
                        stable_prefix = build_podcast_stable_prefix(interests)
                        built = build_script(
                            stable_prefix,
                            date_label=_episode_date_label(resolved_date),
                            presenter_a=channel_config.presenter_a,
                            presenter_b=channel_config.presenter_b,
                            items=refreshed_items,
                        )
                        input_tokens = built.input_tokens
                        output_tokens = built.output_tokens
                        if built.script is None:
                            console.print(
                                "[yellow]script call produced nothing usable[/yellow]; "
                                "see the log for why."
                            )
                        else:
                            written = write_script(
                                conn,
                                built,
                                date=resolved_date,
                                presenter_a=channel_config.presenter_a,
                                presenter_b=channel_config.presenter_b,
                                vault_dir=effective_vault_dir,
                            )
                            console.print(f"[green]script written[/green]: {written}")
                            vault_commit_errors = _commit_vault_and_report(
                                settings=settings,
                                vault_dir=effective_vault_dir,
                                message=f"vault: podcast script {resolved_date.isoformat()}",
                            )

                    # Attempted regardless of whether the script above was fresh or
                    # reused: `deliver_podcast` is its own idempotency boundary
                    # (already-delivered dates, cached mp3s), the same posture
                    # `_deliver_and_report` takes after `digest`'s vault write.
                    delivery_errors, tts_characters = _deliver_podcast_and_report(
                        conn,
                        settings=settings,
                        target_date=resolved_date,
                        vault_dir=effective_vault_dir,
                        audio_dir=effective_audio_dir,
                        today=datetime.now(tz).date(),
                        run_id=run_id,
                        resend=resend,
                    )
                    status_value = "ok"
        except BaseException as exc:
            # Includes KeyboardInterrupt, mirroring every other command's
            # no-silent-runs rule (CLAUDE.md §3): a crash mid-episode still
            # closes its `runs` row. `build_script` itself never raises once
            # its first call has been billed (see its docstring) — an
            # `LlmError` reaching here means that first call failed before any
            # response came back, so `input_tokens`/`output_tokens` staying 0
            # is the honest figure, not a loss.
            run_level_error = _run_level_error(exc)
            raise
        finally:
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                items_new=0,  # podcast writes no `items` rows.
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
                tts_characters=tts_characters,
                errors=(
                    [
                        *([run_level_error] if run_level_error is not None else []),
                        *([harvest_error] if harvest_error is not None else []),
                        *deep_read_errors,
                        *delivery_errors,
                        *vault_commit_errors,
                    ]
                    or None
                ),
            )
        if status_value == "ok" and item_count:
            tts_model = channel_config.tts_model if channel_config is not None else ""
            dollars = tts_spend_usd(tts_characters, tts_model)
            console.print(
                f"[dim]{item_count} item(s) selected, "
                f"{input_tokens:,} input / {output_tokens:,} output tokens, "
                f"{tts_characters:,} tts character(s) (${dollars:.2f}).[/dim]"
            )

    if status_value == "failed":
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# mark
# --------------------------------------------------------------------------- #


@app.command()
def mark(
    item_ref: Annotated[
        int, typer.Argument(help="The `items.id` to mark (shown in the digest's checkbox markers).")
    ],
    verdict: Annotated[str, typer.Argument(help=f"One of: {', '.join(feedback.VERDICTS)}.")],
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
    note: Annotated[
        str | None, typer.Option("--note", help="Optional free-text note stored with the mark.")
    ] = None,
) -> None:
    """Record a human `useful|noise|exceptional|missed` mark on an item (DESIGN §11, Phase 1).

    A mark is a human action, not a pipeline run — so it writes no `runs` row and
    spends no tokens. Idempotent (CLAUDE.md §3): marking the same item with the
    same verdict twice stores one row, and the second call says "already marked".
    Nothing about scoring changes; a mark only stores ground-truth for Phase 2
    tuning (adaptation is out of scope here).
    """
    if verdict not in feedback.VERDICTS:
        raise typer.BadParameter(
            f"verdict must be one of {', '.join(feedback.VERDICTS)}, not {verdict!r}",
            param_hint="verdict",
        )

    with connection(db) as conn:
        if get_item(conn, item_ref) is None:
            err_console.print(
                f"[red]unknown item[/red] {item_ref}; nothing recorded. "
                "Check the id in the digest's checkbox marker."
            )
            raise typer.Exit(code=2)
        recorded = record_feedback(
            conn, item_id=item_ref, verdict=verdict, note=note, created_at=datetime.now(UTC)
        )

    if recorded:
        console.print(f"[green]recorded[/green]: item {item_ref} marked {verdict}.")
    else:
        console.print(f"[yellow]already marked[/yellow]: item {item_ref} is already {verdict}.")


# --------------------------------------------------------------------------- #
# curate (DESIGN §7.1)
# --------------------------------------------------------------------------- #

curate_app = typer.Typer(
    help="Adaptive source curation: propose, review, and apply sources.yaml changes.",
    no_args_is_help=True,
)
app.add_typer(curate_app, name="curate")


def _curation_or_exit(config_dir: Path) -> tuple[SourcesConfig, CurationConfig]:
    """Load `sources.yaml` and its `curation:` block, or exit saying it is off.

    An absent block is not an error and not a default — it means the operator has
    not agreed to this feature's spend, and inventing caps on their behalf is the
    one thing a command that costs money must not do (CLAUDE.md §4).
    """
    sources = _load_sources_or_exit(config_dir)
    if sources.curation is None:
        err_console.print(
            "[yellow]source curation is off[/yellow]: `config/sources.yaml` has no "
            "`curation:` block, so there are no spend caps to run under. Add one to "
            "enable the weekly scout (see DESIGN §7.1)."
        )
        raise typer.Exit(code=1)
    return sources, sources.curation


def _render_proposal_preview(candidates: list[Candidate]) -> None:
    """Print what a dry run would have stored, in the shape the digest will show."""
    if not candidates:
        console.print("[yellow]no proposals[/yellow]: the scout suggested no changes this run.")
        return
    for candidate in candidates:
        staged = " (staged)" if candidate.kind.is_staged else ""
        invalid = (
            ""
            if candidate.status is ProposalStatus.PENDING
            else f"  [red]{candidate.status.value}[/red]"
        )
        console.print(
            f"\n[bold]{candidate.kind.action_label}: {candidate.dedup_key}[/bold]{staged}{invalid}"
        )
        console.print(f"  {candidate.rationale}")
        facts = [candidate.tier.value]
        if (weight := candidate.payload.get("weight")) is not None:
            facts.append(f"weight {weight}")
        if candidate.probe is not None:
            facts.append(
                f"probe ok, {candidate.probe.items_total} entries"
                if candidate.probe.ok
                else f"probe failed: {candidate.probe.error}"
            )
        console.print(f"  [dim]{' · '.join(facts)}[/dim]")
        for entry in candidate.evidence:
            console.print(f"  [dim]- {entry['url']}[/dim]")


def _too_soon(conn: sqlite3.Connection, *, now: datetime) -> int | None:
    """Days since the last `curate` run, if that is inside the minimum interval.

    Reads only the `curate` kind. `curate-apply` runs every morning and costs
    nothing, so counting it would refuse every scout run forever — which is one of
    the reasons the two have separate kinds.
    """
    row = conn.execute(
        "SELECT MAX(started_at) AS latest FROM runs WHERE kind = ?", (RUN_KIND_CURATE,)
    ).fetchone()
    latest = _parse_iso(row["latest"] if row else None)
    if latest is None:
        return None
    days = (now - latest).days
    return days if days < _MIN_CURATE_INTERVAL_DAYS else None


@curate_app.command("run")
def curate_run(
    config_dir: Annotated[Path, typer.Option("--config-dir")] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
    surface_date: Annotated[
        datetime | None,
        typer.Option(
            "--surface-date",
            formats=["%Y-%m-%d"],
            help="Digest date the proposals first render on. Defaults to tomorrow.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the proposals; write none of them.")
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Run even if the last scout run was days ago, not a week."),
    ] = False,
) -> None:
    """Run the weekly source scout (DESIGN §7.1, cron Sunday).

    Three passes in a fixed order: re-probe the `invalid` proposals from previous
    weeks (free), make **one** Opus call with web search (the only paid step), then
    probe and store what it proposed. The re-probe goes first so a candidate the
    scout re-suggests this week already carries fresh facts.

    **`--dry-run` still spends money.** It skips every write to `proposals` — so
    nothing reaches a digest and nothing can be approved — but the API call and its
    searches happen either way, which is the whole point: the preview has to be what
    a real run would produce. The `runs` row is therefore written on a dry run too,
    because that row *is* the spend accounting (NEVER rule 11). This is the opposite
    of `digest --dry-run`, which costs nothing.

    Proposals surface on tomorrow's digest by default: this command runs after the
    morning's digest has already been written, so today's is gone.

    Refuses to run twice inside `_MIN_CURATE_INTERVAL_DAYS` unless `--force`, because
    nothing else stops a weekly command being run daily and each run is billed
    whether or not its proposals are new. `--dry-run` is guarded too: it spends the
    same money.
    """
    sources, curation = _curation_or_exit(config_dir)
    interests = _load_interests_or_exit(config_dir)
    settings = _load_settings_or_exit(config_dir)
    now = datetime.now(UTC)
    resolved_surface = (
        surface_date.date()
        if surface_date is not None
        else now.astimezone(settings.tzinfo).date() + timedelta(days=1)
    )

    with connection(db) as conn:
        if not force and (days := _too_soon(conn, now=now)) is not None:
            err_console.print(
                f"[yellow]too soon[/yellow]: the scout last ran {days} day(s) ago and this "
                f"is a weekly job. Every run is billed whether or not its proposals are "
                f"new, so pass --force if you mean it."
            )
            raise typer.Exit(code=1)
        run_id = start_run(conn, RUN_KIND_CURATE, started_at=now)
        errors: list[dict[str, str]] = []
        spent = ScoutOutcome()
        status_value = "failed"
        try:
            if not dry_run:
                # Free, and first: refreshes last week's `invalid` rows so a
                # re-proposal below finds facts already up to date.
                reprobe = asyncio.run(
                    reprobe_invalid_proposals(conn, sources=sources, cache_dir=cache_dir, now=now)
                )
                errors.extend(reprobe.errors)
                if reprobe.reopened:
                    console.print(
                        f"[green]reopened[/green]: {len(reprobe.reopened)} previously "
                        "invalid candidate(s) now fetch."
                    )

            spent = asyncio.run(
                scout_for_proposals(
                    conn,
                    sources=sources,
                    interests=interests,
                    curation=curation,
                    cache_dir=cache_dir,
                    now=now,
                )
            )
            errors.extend(spent.errors)

            if dry_run:
                _render_proposal_preview(spent.candidates)
                console.print(
                    f"[yellow]dry run[/yellow]: {len(spent.candidates)} proposal(s) would be "
                    "stored; nothing written."
                )
            else:
                stored = persist_candidates(
                    conn,
                    spent.candidates,
                    run_id=run_id,
                    surface_date=resolved_surface,
                    created_at=now,
                )
                errors.extend(stored.errors)
                console.print(
                    f"[green]stored[/green]: {len(stored.stored)} new proposal(s), "
                    f"{stored.duplicates} already known. "
                    f"They surface in the {resolved_surface.isoformat()} digest."
                )
            status_value = "partial" if errors else "ok"
        except BaseException as exc:
            # `BaseException`, matching `ingest`/`score`: a Ctrl-C'd run should still
            # close its row *with a reason*, not just close it. Re-raised, so the
            # interrupt still stops the process.
            #
            # A Ctrl-C mid-probe does lose the *spend figure* — `_shape_candidates`
            # catches `Exception`, so `KeyboardInterrupt` propagates past it and
            # `spent` stays zero. That is a deliberate trade, not an oversight: the
            # row is still written with `KeyboardInterrupt` as its reason, the amount
            # at risk is cents on a human-initiated action, and the person who hit
            # Ctrl-C knows what they just spent. **Do not "fix" it by moving the
            # accounting guarantee back into this function** — reading spend off a
            # raised exception is the contract that produced the bug two reviews
            # found, and `scout_for_proposals` now owns the promise instead.
            logger.exception("curate run failed")
            errors.append(_run_level_error(exc))
            raise
        finally:
            # In `finally` because the searches and tokens are spent before anything
            # below can fail, and a run that loses its own cost record is worse than
            # a run that failed (NEVER rule 11).
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                llm_input_tokens=spent.input_tokens,
                llm_output_tokens=spent.output_tokens,
                server_tool_requests=spent.web_search_requests,
                errors=errors or None,
            )
            searches = spent.web_search_requests
            console.print(
                f"[dim]spend: {spent.input_tokens:,} in / {spent.output_tokens:,} out tokens, "
                f"{searches} search(es) (${search_spend_usd(searches):.2f}).[/dim]"
            )
        if errors:
            _render_curate_errors(errors)


def _render_curate_errors(errors: list[dict[str, str]]) -> None:
    console.print(f"[yellow]{len(errors)} note(s) recorded to this run:[/yellow]")
    for record in errors:
        console.print(f"  [dim]{record.get('source_id', '?')}[/dim]: {record.get('message', '')}")


@curate_app.command("apply")
def curate_apply(
    config_dir: Annotated[Path, typer.Option("--config-dir")] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
    vault_dir: Annotated[Path | None, typer.Option("--vault-dir")] = None,
) -> None:
    """Harvest ticked proposals from the vault and apply the approved ones.

    Two steps, both free — **this command makes no LLM call** (see the test that
    pins it). It reads `<vault>/daily/*.md` for approve/reject ticks, records those
    decisions, then edits `config/sources.yaml` for everything now approved.

    Runs first in the `daily` chain, before ingest, so a feed approved yesterday is
    fetched this morning. The YAML change is **not committed** — it lands as a
    working-tree diff to read, which is the promise DESIGN §11 makes for every
    proposed change.
    """
    settings = _load_settings_or_exit(config_dir)
    effective_vault_dir = vault_dir if vault_dir is not None else settings.vault_dir
    today = datetime.now(settings.tzinfo).date()

    with connection(db) as conn:
        run_id = start_run(conn, RUN_KIND_CURATE_APPLY, started_at=datetime.now(UTC))
        errors: list[dict[str, str]] = []
        status_value = "failed"
        try:
            harvest = harvest_approvals(conn, effective_vault_dir)
            if harvest.decisions_recorded:
                console.print(
                    f"[green]harvested[/green]: {harvest.decisions_recorded} decision(s) from "
                    f"{harvest.files_scanned} vault file(s)."
                )
            if harvest.contradictions:
                console.print(
                    f"[yellow]{harvest.contradictions} proposal(s) had both boxes ticked[/yellow]"
                    " and were left pending — tick one."
                )

            outcome = apply_approved_proposals(conn, config_dir, today=today)
            errors.extend(outcome.errors)
            for change in outcome.changes:
                verb = "already applied" if change.already_applied else "applied"
                console.print(f"[green]{verb}[/green]: {change.kind.action_label} {change.target}")
            if outcome.text_changed:
                console.print(
                    "[bold]config/sources.yaml changed[/bold] — review it with "
                    "`git diff config/sources.yaml` (nothing was committed)."
                )
            elif not outcome.changes:
                console.print("[dim]nothing approved to apply.[/dim]")
            status_value = "partial" if errors else "ok"
        except BaseException as exc:
            # See `curate_run` — a row closed with no reason is a row that lies.
            logger.exception("curate apply failed")
            errors.append(_run_level_error(exc))
            raise
        finally:
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                errors=errors or None,
            )
        if errors:
            _render_curate_errors(errors)


@curate_app.command("list")
def curate_list(
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
    status_filter: Annotated[
        str | None,
        typer.Option("--status", help=f"One of: {', '.join(s.value for s in ProposalStatus)}."),
    ] = None,
) -> None:
    """List proposals — the terminal view of the digest's approval block."""
    statuses: list[ProposalStatus] | None = None
    if status_filter is not None:
        try:
            statuses = [ProposalStatus(status_filter)]
        except ValueError as exc:
            raise typer.BadParameter(
                f"status must be one of {', '.join(s.value for s in ProposalStatus)}",
                param_hint="--status",
            ) from exc

    with connection(db) as conn:
        proposals = get_proposals(conn, statuses=statuses)

    if not proposals:
        console.print("[dim]no proposals match.[/dim]")
        return
    table = Table(header_style="bold")
    for column in ("id", "action", "target", "status", "surfaced", "found via"):
        table.add_column(column)
    for proposal in proposals:
        table.add_row(
            str(proposal.id),
            proposal.kind.action_label,
            proposal.dedup_key,
            proposal.status.value,
            proposal.surface_date.isoformat(),
            proposal.tier.value,
        )
    console.print(table)


@curate_app.command("decide")
def curate_decide(
    proposal_id: Annotated[int, typer.Argument(help="The proposal id shown in the digest.")],
    decision: Annotated[str, typer.Argument(help=f"One of: {', '.join(DECISIONS)}.")],
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
    note: Annotated[
        str | None,
        typer.Option("--note", help="Why. Replayed to the scout so it learns from a rejection."),
    ] = None,
) -> None:
    """Approve or reject a proposal from the terminal.

    The same decision the digest checkbox records, plus a `--note` the checkbox
    cannot carry — and the note is the half that teaches the next scout run
    something, so a rejection is worth making here rather than by tick.
    """
    if decision not in DECISIONS:
        raise typer.BadParameter(
            f"decision must be one of {', '.join(DECISIONS)}, not {decision!r}",
            param_hint="decision",
        )
    status = ProposalStatus.APPROVED if decision == "approve" else ProposalStatus.REJECTED

    with connection(db) as conn:
        if get_proposal(conn, proposal_id) is None:
            err_console.print(f"[red]unknown proposal[/red] {proposal_id}; nothing recorded.")
            raise typer.Exit(code=2)
        recorded = decide_proposal(
            conn,
            proposal_id=proposal_id,
            status=status,
            decided_at=datetime.now(UTC),
            note=note,
        )

    if recorded:
        console.print(f"[green]recorded[/green]: proposal {proposal_id} {status.value}.")
    else:
        console.print(
            f"[yellow]already decided[/yellow]: proposal {proposal_id} is not pending, so "
            "nothing changed. Use `curate reopen` first if you have changed your mind."
        )


@curate_app.command("reopen")
def curate_reopen(
    proposal_id: Annotated[int, typer.Argument(help="The proposal id to return to pending.")],
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
) -> None:
    """Return a rejected or invalid proposal to pending.

    The escape hatch from a decision you regret. Without it a rejection is permanent
    from the scout's side: `ux_proposals_kind_key` means it can never re-suggest the
    candidate itself. `applied` is deliberately not reopenable — that state lives in
    `sources.yaml`, and undoing it means editing the file.
    """
    with connection(db) as conn:
        if get_proposal(conn, proposal_id) is None:
            err_console.print(f"[red]unknown proposal[/red] {proposal_id}.")
            raise typer.Exit(code=2)
        reopened = reopen_proposal(conn, proposal_id=proposal_id)

    if reopened:
        console.print(
            f"[green]reopened[/green]: proposal {proposal_id} is pending again and will "
            "appear in the next digest."
        )
    else:
        console.print(
            f"[yellow]not reopenable[/yellow]: proposal {proposal_id} is neither rejected "
            "nor invalid."
        )


# --------------------------------------------------------------------------- #
# daily
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# weekly — the Sunday brief (DESIGN §13, Phase 1's acceptance gate)
# --------------------------------------------------------------------------- #


def _most_recent_sunday(today: Date) -> Date:
    """The Sunday on or before `today`.

    Split out from `_resolve_target_sunday` so the arithmetic is testable without
    a clock: it is what makes every day of a week resolve to one vault path, and
    therefore what bounds this command's spend at one call per week.
    """
    # Monday is 0 in `weekday()`, so Sunday is 6; `(weekday + 1) % 7` is how many
    # days back the most recent Sunday is, and it is 0 on a Sunday.
    return today - timedelta(days=(today.weekday() + 1) % 7)


def _resolve_target_sunday(target_date: datetime | None, *, tz: tzinfo) -> Date:
    """The Sunday a brief is published on: `--date`, or the most recent one.

    This is the whole double-spend defence, and it lives here rather than in a
    `runs` guard on purpose. `weekly` is safe to invoke any day — every day of a
    week resolves to the same Sunday, so it resolves to the same vault path, so
    the file guard in `weekly` sees the existing file and skips the call. A
    command defaulting to "today" would instead produce a different filename
    every day, and a mis-wired daily invocation would bill thirty times a month
    while the file guard saw nothing.

    A non-Sunday `--date` is refused rather than snapped. Silently rewriting an
    operator's explicit date would make `--date 2026-08-05` quietly write the
    2026-08-02 brief, and finding that out from the filename is worse than
    finding it out from an error.
    """
    if target_date is None:
        return _most_recent_sunday(datetime.now(tz).date())

    resolved = target_date.date()
    if resolved.weekday() != _SUNDAY:
        raise typer.BadParameter(
            f"{resolved.isoformat()} is a {resolved.strftime('%A')}. A brief is dated by the "
            "Sunday it is published on, and covers the seven days before it — pass that "
            "Sunday, or omit --date for the most recent one.",
            param_hint="--date",
        )
    return resolved


def _select_weekly_items(
    conn: sqlite3.Connection,
    *,
    interests: InterestsConfig,
    target_sunday: Date,
    tz: tzinfo,
) -> WeeklySelection:
    """The week's rows, partitioned into the brief's body and its near-misses.

    All deterministic (`report/weekly.py`); the only judgement in the whole
    command is what synthesis then writes *about* these items.
    """
    start, end = weekly_window(target_sunday, tz)
    rows = get_digest_items(conn, start=start, end=end)
    verdicts = feedback_verdicts_for_items(
        conn, [scored.item.id for scored in rows if scored.item.id is not None]
    )
    return select_weekly_items(rows, verdicts, thresholds=interests.thresholds)


def _render_weekly_preview(selection: WeeklySelection, *, target_sunday: Date) -> None:
    """What `--dry-run` prints. Everything here is deterministic, which is why
    the preview needs no paid call to be a preview of anything — unlike
    `curate run --dry-run`, whose whole subject is the call (DESIGN §14)."""
    window_start = target_sunday - timedelta(days=WEEKLY_WINDOW_DAYS)
    console.print(
        f"[bold]week of {window_start.isoformat()} to "
        f"{(target_sunday - timedelta(days=1)).isoformat()}[/bold] "
        f"→ brief dated {target_sunday.isoformat()}"
    )
    for index, scored in enumerate(selection.items, start=1):
        console.print(
            f"  {index:2}. [{scored.signal}/{scored.relevance}/{scored.novelty}] "
            f"{escape(scored.item.title)}"
        )
    if selection.near_misses:
        console.print("[dim]near misses:[/dim]")
        for scored in selection.near_misses:
            console.print(
                f"  [dim]id={scored.item.id} "
                f"[{scored.signal}/{scored.relevance}/{scored.novelty}] "
                f"{escape(scored.item.title)}[/dim]"
            )
    console.print(
        f"[dim]{selection.gated_out_count} below the bar · "
        f"{selection.hidden_count} above it but not shown[/dim]"
    )


@app.command()
def weekly(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding interests.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
    vault_dir: Annotated[
        Path | None,
        typer.Option(
            "--vault-dir", help="Vault root (briefs land in <vault>/weekly/). See `digest`."
        ),
    ] = None,
    target_date: Annotated[
        datetime | None,
        typer.Option(
            "--date",
            formats=["%Y-%m-%d"],
            help="The Sunday the brief is dated (default: the most recent one).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the selection. No LLM call, no writes."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rewrite an existing brief (a fresh, billed Opus call)."),
    ] = False,
) -> None:
    """Write the Weekly Intelligence Brief — Phase 1's product and its gate.

    Covers the **seven days before** its Sunday, not including it: the score pass
    runs in the evening and this runs Sunday morning (DESIGN §14), so the
    publication day's own items do not exist yet. Consecutive Sundays tile
    exactly.

    Stricter than the daily digest by design. `interests.yaml`'s `weekly_min_*`
    thresholds have until now existed only as prompt text telling triage what bar
    to keep at; this applies the arithmetic (DESIGN §9), so items the digest
    showed can legitimately be absent. Those become the near-miss list, which is
    there to make `signalforge mark <id> missed` cheap to give.

    Idempotent, and that is what bounds the spend (CLAUDE.md §3, §6). Every day of
    a week resolves to the same Sunday and therefore the same vault path, so a
    brief already on disk is left alone and no call is made; `--force` is the only
    way to pay twice for one week.

    **The vault file is written on every outcome**, including a refused or
    unusable synthesis — the body, the checkboxes and the near-misses come from
    the deterministic selection, never from the model (DESIGN §13.2). A failed
    call therefore degrades the brief rather than skipping it, and because the
    file still lands, the next invocation still declines to re-bill.

    `thresholds.weekly_top_n` unset means the brief is off: a clean exit, no call.
    """
    interests = _load_interests_or_exit(config_dir)
    settings = _load_settings_or_exit(config_dir)
    tz = settings.tzinfo
    effective_vault_dir = vault_dir if vault_dir is not None else settings.vault_dir
    target_sunday = _resolve_target_sunday(target_date, tz=tz)

    with connection(db) as conn:
        run_id = start_run(conn, RUN_KIND_WEEKLY, started_at=datetime.now(UTC))
        run_level_error: dict[str, str] | None = None
        harvest_error: dict[str, str] | None = None
        vault_commit_errors: list[dict[str, str]] = []
        status_value = "failed"
        input_tokens = 0
        output_tokens = 0

        try:
            if interests.thresholds.weekly_top_n is None:
                console.print(
                    "[yellow]weekly brief is off[/yellow]: set `thresholds.weekly_top_n` "
                    "in interests.yaml to turn it on."
                )
                status_value = "ok"
                return

            # Harvest before selection (so `noise` applies) and before the write
            # (so `--force` cannot destroy ticks that exist only in the file).
            # One pass satisfies both. Skipped on --dry-run: a preview must never
            # write ground-truth feedback rows. Non-fatal — a broken harvest is
            # recorded to `runs.errors` and surfaced by `status`, and the brief
            # still renders (CLAUDE.md §7).
            if not dry_run:
                try:
                    harvested = harvest_marks(conn, effective_vault_dir)
                    if harvested.rows_recorded:
                        console.print(
                            f"[green]harvested feedback[/green]: "
                            f"{harvested.rows_recorded} new mark(s) after scanning "
                            f"{harvested.files_scanned} vault file(s)."
                        )
                except Exception as exc:
                    logger.exception("harvesting vault feedback failed; continuing to the brief")
                    harvest_error = _run_level_error(exc)

            selection = _select_weekly_items(
                conn, interests=interests, target_sunday=target_sunday, tz=tz
            )

            if dry_run:
                _render_weekly_preview(selection, target_sunday=target_sunday)
                console.print("[yellow]dry run[/yellow]: no LLM call, nothing written.")
                status_value = "ok"
                return

            if not selection.items:
                console.print(
                    "[yellow]nothing cleared the brief's bar this week[/yellow] — no call made."
                )
                status_value = "ok"
                return

            path = brief_path(effective_vault_dir, target_sunday=target_sunday)
            if path.is_file() and not force:
                console.print(
                    f"[dim]brief already written:[/dim] {path} "
                    "[dim](--force to rewrite; that is a fresh, billed call)[/dim]"
                )
                status_value = "ok"
                return

            built = build_brief(
                build_weekly_stable_prefix(interests),
                window_label=_weekly_window_label(target_sunday),
                items=selection.items,
            )
            input_tokens = built.input_tokens
            output_tokens = built.output_tokens
            if built.brief is None:
                console.print(
                    "[yellow]synthesis produced nothing usable[/yellow] — writing the "
                    "brief without a narrative."
                )

            # Last thing that can raise between a billed response and disk. If a
            # write failure ever became routine here the next run would re-bill,
            # which is the one repeat-spend path the Sunday snap does not close.
            written = write_brief(
                built,
                selection,
                target_sunday=target_sunday,
                vault_dir=effective_vault_dir,
            )
            console.print(
                f"[green]wrote[/green] {written} "
                f"[dim]({len(selection.items)} item(s), "
                f"{input_tokens + output_tokens} tokens)[/dim]"
            )
            vault_commit_errors = _commit_vault_and_report(
                settings=settings,
                vault_dir=effective_vault_dir,
                message=f"vault: weekly brief {target_sunday.isoformat()}",
            )
            status_value = "ok"
        except BaseException as exc:
            run_level_error = _run_level_error(exc)
            raise
        finally:
            errors = [
                record for record in (run_level_error, harvest_error) if record is not None
            ] + vault_commit_errors
            finish_run(
                conn,
                run_id,
                status=status_value,
                finished_at=datetime.now(UTC),
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
                errors=errors or None,
            )
        if status_value != "ok":
            raise typer.Exit(code=1)


@app.command()
def daily(
    config_dir: Annotated[Path, typer.Option("--config-dir")] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db")] = DEFAULT_DB_PATH,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = DEFAULT_CACHE_DIR,
    vault_dir: Annotated[Path | None, typer.Option("--vault-dir")] = None,
    audio_dir: Annotated[Path | None, typer.Option("--audio-dir")] = None,
    max_concurrency: Annotated[int, typer.Option("--max-concurrency", min=1)] = (
        DEFAULT_MAX_CONCURRENCY
    ),
    with_podcast: Annotated[
        bool,
        typer.Option(
            "--podcast/--no-podcast",
            help="Record the episode as the last step. `--no-podcast` stops after "
            "`digest`, for the split evening-digest / morning-podcast schedule.",
        ),
    ] = True,
) -> None:
    """Run `curate apply`, `ingest`, `score`, `digest`, then `podcast` (DESIGN §14).

    Split across two cron entries by default (DESIGN §14): `daily --no-podcast`
    in the evening, `podcast --date <yesterday>` the next morning. The gap is
    the point — it is the window in which the operator reads the digest and
    ticks `noise`/`useful`/`exceptional`, and `podcast` harvests those marks and
    ranks the episode by them (`_select_podcast_items`). Running the whole chain
    in one shot (the default, `--podcast`) is still correct and is what a manual
    catch-up run wants; it just records the episode against whatever marks
    already existed, which on a same-day run is none.

    Each step keeps its own `runs` row and failure isolation — a step that
    comes back `partial`/`failed` does not stop the next one from running,
    since each downstream step only cares about rows the previous one already
    committed (ingest's new items, score's new scores). This command never
    fixes "today" before `score` runs and then hands that fixed date to
    `digest`: `digest` always resolves `--date` itself, immediately after
    `score` returns, so a triage batch that happens to straddle UTC midnight
    still lands in the digest computed *after* it finished, not one decided
    before it started. `podcast` runs last and resolves its own date the same
    way, from whatever `score` just committed.

    Exit code is the worst of the steps' (2 > 1 > 0), so a config error in any step
    is still visible to cron even though later steps still ran.

    `curate apply` runs first and only when `sources.yaml` has a `curation:` block:
    applying yesterday's approvals before ingest is what lets a feed approved in
    last night's digest be fetched this morning. Its failure is isolated like every
    other step's — a broken applier costs the day's approvals, not the day's digest,
    and a `sources.yaml` that does not parse costs only the steps that read it.

    `podcast`, when it runs here at all, is the fifth and last step, isolated the
    same way: an unconfigured channel or a dead R2/TTS provider costs the day's
    episode, not the digest that already rendered before it ran (DESIGN §13.3's
    inherited invariant — a channel failure never fails the run).
    """
    worst_exit = 0

    def apply_approvals() -> None:
        """`curate apply`, but only when the feature is configured.

        The config load lives **inside** this closure, not above the loop. Reading it
        outside would make a `ConfigError` in `sources.yaml` abort `daily` before any
        step ran and before any `runs` row existed — no isolation, and no record that
        6am happened at all (CLAUDE.md §3, §7). `sources.yaml` is a file the design
        expects to be hand-edited, so a typo in it must cost the steps that need it,
        not the whole morning: `score` needs only `interests.yaml` and should still
        clear its backlog.
        """
        if _load_sources_or_exit(config_dir).curation is None:
            # An operator who has not enabled curation should see no trace of it.
            return
        curate_apply(config_dir=config_dir, db=db, vault_dir=vault_dir)

    # `curate apply` leads so a source approved in yesterday's digest is fetched by
    # this morning's ingest (DESIGN §7.1).
    steps: tuple[Callable[[], None], ...] = (
        apply_approvals,
        lambda: ingest(
            config_dir=config_dir,
            db=db,
            cache_dir=cache_dir,
            source=None,
            max_concurrency=max_concurrency,
        ),
        lambda: score(config_dir=config_dir, db=db),
        lambda: digest(config_dir=config_dir, db=db, vault_dir=vault_dir),
        *(
            (
                lambda: podcast(
                    config_dir=config_dir,
                    db=db,
                    cache_dir=cache_dir,
                    vault_dir=vault_dir,
                    audio_dir=audio_dir,
                ),
            )
            if with_podcast
            else ()
        ),
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
# deliver (DESIGN §13.2 — the push channel)
# --------------------------------------------------------------------------- #

deliver_app = typer.Typer(
    help="Outbound mirrors of a rendered report. The vault stays canonical.",
    no_args_is_help=True,
)
app.add_typer(deliver_app, name="deliver")


@deliver_app.command("test")
def deliver_test(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding settings.yaml.")
    ] = DEFAULT_CONFIG_DIR,
) -> None:
    """Send one sample message, to prove the channel works before trusting it.

    Deliberately independent of the pipeline: no database, no items, no `runs` row
    and no `deliveries` row. Setting up a mail provider means a key, a verified
    sender domain and a recipient, and each of those fails in its own way — this
    exists so those failures surface on demand rather than at 06:00 tomorrow.

    The sample is a real render of a fabricated one-item digest, so what arrives is
    what a real digest will look like.
    """
    settings = _load_settings_or_exit(config_dir)
    channel = settings.delivery.email

    # Two failure shapes, two exit codes. Misconfiguration (no channel, no key) is
    # exit 2, matching every other config error here; a provider that refused a
    # well-formed request is exit 1, because nothing about the config is wrong.
    if channel is None or not channel.enabled:
        err_console.print(
            "[red]no delivery channel is enabled[/red]: add a `delivery.email` block "
            f"to {config_dir / SETTINGS_FILENAME} (see settings.yaml.example)."
        )
        raise typer.Exit(code=2)
    api_key = get_secret(channel.api_key_env)
    if api_key is None:
        err_console.print(
            f"[red]{channel.api_key_env} is not set[/red] in the environment or .env."
        )
        raise typer.Exit(code=2)

    console.print(
        f"sending to {escape(', '.join(channel.to))} from {escape(channel.from_address)} …"
    )
    outcome = send_test_message(channel, api_key, today=datetime.now(settings.tzinfo).date())

    # Escaped for the same reason as `_report_delivery_outcomes`: `reason` carries
    # the provider's response body, and rich reads `[...]` as markup.
    if not outcome.sent:
        err_console.print(
            f"[red]{escape(outcome.channel)} test failed[/red]: {escape(outcome.reason)}"
        )
        raise typer.Exit(code=1)

    provider = escape(outcome.provider_id) if outcome.provider_id else ""
    suffix = f" (provider id {provider})" if provider else ""
    console.print(
        f"[green]{escape(outcome.channel)} test sent[/green]: {escape(outcome.reason)}{suffix}"
    )


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
            escape(str(record.get("source_id", "?"))),
            escape(str(record.get("error_type", "?"))),
            escape(str(record.get("message", ""))),
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


def _render_stale_topics(
    conn: sqlite3.Connection,
    taxonomy: TaxonomyConfig | None,
    *,
    now: datetime,
    days: int,
) -> None:
    """Name taxonomy leaves that have matched nothing lately (DESIGN §10).

    Printed only when there is something to say. A leaf going quiet is a nudge
    to edit `taxonomy.yaml`, not a fault — the reports are the monitoring
    channel (CLAUDE.md §7), and a permanent "all topics healthy" row would be
    noise in a readout whose job is to make the abnormal visible.
    """
    if taxonomy is None:
        return
    stale = stale_topics(conn, taxonomy, now=now, days=days)
    if not stale:
        return
    # `console`, not `err_console`: `_render_freshness`'s dark-source alarm is a
    # strictly louder signal and goes to stdout, so splitting streams inside one
    # readout would interleave unpredictably and hide this line under a redirect.
    console.print(
        f"[yellow]taxonomy[/yellow]: {len(stale)} leaf/leaves matched nothing in {days} days — "
        f"{', '.join(stale)}. Edit their keywords in taxonomy.yaml, or drop them."
    )


def _render_token_spend(conn: sqlite3.Connection, *, now: datetime, tts_model: str | None) -> None:
    """Month-to-date spend: tokens, plus the two lines tokens cannot express.

    DESIGN §8 makes spend the number this project lives or dies on ($30 is the
    alarm). A cost readout that only appears once there is cost to report is a
    cost readout nobody has ever looked at when it matters.
    """
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(llm_input_tokens), 0)     AS input_tokens,
               COALESCE(SUM(llm_output_tokens), 0)    AS output_tokens,
               COALESCE(SUM(server_tool_requests), 0) AS searches,
               COALESCE(SUM(tts_characters), 0)       AS tts_characters,
               COUNT(*)                               AS run_count
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
    # Web search bills per call, not per token, so this line is not derivable from
    # the two above it — without it the $30 alarm cannot see the whole invoice
    # (DESIGN §8). Shown always, including as a $0.00, for the same reason the token
    # rows are: a cost readout that appears only once there is cost to report is one
    # nobody has looked at when it mattered.
    searches = int(row["searches"])
    table.add_row("web searches", f"{searches:,} (${search_spend_usd(searches):.2f})")
    # TTS bills per character, priced at the currently configured podcast model —
    # `runs.tts_characters` alone cannot say what a past run's episode actually cost
    # if the model has since changed, but pricing zero characters costs nothing to be
    # wrong about, and `tts_spend_usd` only ever needs a model when there is spend to
    # price (see `podcast`'s own guard for the same reasoning).
    chars = int(row["tts_characters"])
    dollars = tts_spend_usd(chars, tts_model or "") if chars else 0.0
    table.add_row("TTS characters (podcast)", f"{chars:,} (${dollars:.2f})")
    console.print(table)


@app.command()
def status(
    config_dir: Annotated[
        Path, typer.Option("--config-dir", help="Directory holding sources.yaml.")
    ] = DEFAULT_CONFIG_DIR,
    db: Annotated[Path, typer.Option("--db", help="SQLite database path.")] = DEFAULT_DB_PATH,
) -> None:
    """Show last-run health, per-source freshness, and month-to-date token/TTS spend."""
    config = _load_sources_or_exit(config_dir)
    settings = _load_settings_or_exit(config_dir)
    taxonomy = _load_taxonomy_or_none(config_dir)
    tts_model = settings.delivery.podcast.tts_model if settings.delivery.podcast else None
    now = datetime.now(UTC)
    with connection(db) as conn:
        _render_last_runs(conn, now=now)
        _render_freshness(conn, config, now=now)
        _render_stale_topics(conn, taxonomy, now=now, days=settings.taxonomy_stale_days)
        _render_token_spend(conn, now=now, tts_model=tts_model)
