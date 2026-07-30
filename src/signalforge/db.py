"""Thin SQLite layer — stdlib `sqlite3`, no ORM (CLAUDE.md §3, NEVER rule 14).

This module is the *only* place the schema changes (via `MIGRATIONS`) and the
only place `datetime` objects become ISO 8601 strings.

Phase 0 tables only: `items`, `scores`, `runs`, `feedback`. The Phase 2/3
tables in DESIGN §5 (embeddings, clusters, trends, insights, …) are
deliberately absent — build them when their phase gate opens (NEVER rule 15).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Final

from signalforge.models import Item, ProposalKind, ProposalStatus, ProposalTier, SourceType

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "DigestItem",
    "KeptItem",
    "Migration",
    "Proposal",
    "RejectedProposal",
    "RunRecord",
    "SourceYield",
    "connect",
    "connection",
    "count_killed_items",
    "decide_proposal",
    "feedback_verdicts_since",
    "finish_run",
    "get_digest_items",
    "get_feedback",
    "get_item",
    "get_item_by_canonical_url",
    "get_latest_run",
    "get_proposal",
    "get_proposals",
    "insert_proposal",
    "insert_score",
    "kept_items",
    "mark_proposal_applied",
    "migrate",
    "record_feedback",
    "rejected_proposals",
    "reopen_proposal",
    "source_yield_stats",
    "start_run",
    "update_proposal_probe",
    "upsert_item",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Migration:
    """One forward-only schema step, applied inside a transaction."""

    version: int
    name: str
    statements: tuple[str, ...]


_MIGRATION_0001_PHASE0 = Migration(
    version=1,
    name="phase0_core_tables",
    statements=(
        """
        CREATE TABLE items (
            id            INTEGER PRIMARY KEY,
            source_id     TEXT NOT NULL,
            source_type   TEXT NOT NULL,
            external_id   TEXT,
            url           TEXT NOT NULL,
            canonical_url TEXT NOT NULL,
            title         TEXT NOT NULL,
            author        TEXT,
            published_at  TEXT,
            fetched_at    TEXT NOT NULL,
            summary       TEXT,
            content       TEXT,
            content_hash  TEXT NOT NULL,
            lang          TEXT DEFAULT 'en',
            raw_path      TEXT,
            UNIQUE (canonical_url),
            UNIQUE (source_id, external_id)
        )
        """,
        "CREATE INDEX idx_items_content_hash ON items (content_hash)",
        "CREATE INDEX idx_items_fetched_at ON items (fetched_at)",
        """
        CREATE TABLE scores (
            item_id        INTEGER PRIMARY KEY REFERENCES items(id),
            triage         TEXT NOT NULL,
            signal         INTEGER,
            relevance      INTEGER,
            novelty        INTEGER,
            reasoning      TEXT NOT NULL,
            rubric_version TEXT NOT NULL,
            model          TEXT NOT NULL,
            scored_at      TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE runs (
            id                INTEGER PRIMARY KEY,
            kind              TEXT NOT NULL,
            started_at        TEXT NOT NULL,
            finished_at       TEXT,
            status            TEXT,
            items_new         INTEGER DEFAULT 0,
            llm_input_tokens  INTEGER DEFAULT 0,
            llm_output_tokens INTEGER DEFAULT 0,
            errors            TEXT
        )
        """,
        """
        CREATE TABLE feedback (
            item_id    INTEGER REFERENCES items(id),
            verdict    TEXT NOT NULL,
            note       TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (item_id, created_at)
        )
        """,
    ),
)

_MIGRATION_0002_FEEDBACK_DEDUP = Migration(
    version=2,
    name="feedback_dedup_index",
    statements=(
        # Non-destructive: adds a unique index only, leaving the Phase 0
        # `feedback` table untouched (migrations are append-only — CLAUDE.md §3).
        # One row per distinct (item, verdict) is what makes harvesting the same
        # vault checkbox twice a no-op: `record_feedback` writes ON CONFLICT DO
        # NOTHING against this index (DESIGN §11 "harvest-then-overwrite").
        "CREATE UNIQUE INDEX ux_feedback_item_verdict ON feedback (item_id, verdict)",
    ),
)

_MIGRATION_0003_CURATION = Migration(
    version=3,
    name="source_curation_proposals",
    statements=(
        # Proposed `sources.yaml` edits awaiting a human tick (DESIGN §7.1). The
        # table stores the *proposal*, never the edit: applying is a text change to
        # the YAML file, so this row is the audit trail for why it happened.
        """
        CREATE TABLE proposals (
            id            INTEGER PRIMARY KEY,
            run_id        INTEGER REFERENCES runs(id),
            kind          TEXT NOT NULL,
            dedup_key     TEXT NOT NULL,
            payload       TEXT NOT NULL,
            rationale     TEXT NOT NULL,
            evidence      TEXT NOT NULL,
            probe         TEXT,
            tier          TEXT NOT NULL,
            status        TEXT NOT NULL,
            surface_date  TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            decided_at    TEXT,
            -- Why the operator said no, when they bothered to say. A digest
            -- checkbox carries no reason, so this is only ever populated from
            -- `curate decide --note`; it is what the scout's suppression list
            -- feeds back, because replaying the scout's own pitch at it teaches
            -- it nothing about the operator's taste (DESIGN §7.1).
            decision_note TEXT,
            applied_at    TEXT
        )
        """,
        # The idempotency lever, mirroring `ux_feedback_item_verdict`: one row per
        # distinct (kind, target) *ever*. Re-running the scout weekly re-suggests
        # the same obvious candidates, and `ON CONFLICT DO NOTHING` collapses those
        # to the original row — so a re-scout adds zero duplicates and a proposal
        # the operator already rejected never reappears in a digest.
        "CREATE UNIQUE INDEX ux_proposals_kind_key ON proposals (kind, dedup_key)",
        # Rendering the digest block asks for "pending, surfaced on or before D",
        # which is this index's exact shape.
        "CREATE INDEX idx_proposals_status_surface ON proposals (status, surface_date)",
        # Web search is billed per search ($10/1000), not per token, so the two
        # `llm_*_tokens` columns cannot express its cost. Without this column the
        # spend would be real and invisible — the failure NEVER rule 11 exists to
        # prevent. `signalforge status` turns it into a dollar figure (DESIGN §8).
        "ALTER TABLE runs ADD COLUMN server_tool_requests INTEGER DEFAULT 0",
    ),
)

MIGRATIONS: Final[tuple[Migration, ...]] = (
    _MIGRATION_0001_PHASE0,
    _MIGRATION_0002_FEEDBACK_DEDUP,
    _MIGRATION_0003_CURATION,
)
"""Ordered, append-only. Never edit an applied migration — add a new one."""

SCHEMA_VERSION: Final[int] = MIGRATIONS[-1].version


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a configured connection. The caller owns closing it.

    WAL keeps the daily cron writer from blocking an interactive `status` read.
    `isolation_level=None` hands transaction control to us explicitly rather
    than to sqlite3's implicit-BEGIN magic — `upsert_item` depends on it.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def connection(db_path: Path, *, migrate_on_open: bool = True) -> Iterator[sqlite3.Connection]:
    """Short-lived connection, migrated and closed. The normal entry point.

    Single writer (one cron process at a time), so a connection per command is
    simpler than a pool and costs nothing at this scale.
    """
    conn = connect(db_path)
    try:
        if migrate_on_open:
            migrate(conn)
        yield conn
    finally:
        conn.close()


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the resulting schema version.

    Version tracking uses `PRAGMA user_version` — an integer SQLite already
    stores in the file header. No bookkeeping table, nothing to get out of sync
    with the schema it describes.
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {current} is newer than this code understands "
            f"({SCHEMA_VERSION}); upgrade signalforge or restore a backup"
        )

    for migration in MIGRATIONS:
        if migration.version <= current:
            continue
        logger.info(
            "applying migration",
            extra={"version": migration.version, "migration": migration.name},
        )
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in migration.statements:
                conn.execute(statement)
            # PRAGMA does not accept a bound parameter; the value is an int
            # from our own append-only MIGRATIONS tuple, never user input.
            conn.execute(f"PRAGMA user_version = {migration.version:d}")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        current = migration.version

    return current


# --------------------------------------------------------------------------- #
# Datetime boundary
# --------------------------------------------------------------------------- #


def _to_iso(value: datetime | None) -> str | None:
    """Serialize to ISO 8601 TEXT (DESIGN §5). Models keep real datetimes."""
    return value.isoformat() if value is not None else None


def _from_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        id=row["id"],
        source_id=row["source_id"],
        source_type=SourceType(row["source_type"]),
        external_id=row["external_id"],
        url=row["url"],
        canonical_url=row["canonical_url"],
        title=row["title"],
        author=row["author"],
        published_at=_from_iso(row["published_at"]),
        fetched_at=_from_iso(row["fetched_at"]) or datetime.fromtimestamp(0),
        summary=row["summary"],
        content=row["content"],
        content_hash=row["content_hash"],
        lang=row["lang"] or "en",
        raw_path=row["raw_path"],
    )


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #

_INSERT_ITEM = """
    INSERT INTO items (
        source_id, source_type, external_id, url, canonical_url, title, author,
        published_at, fetched_at, summary, content, content_hash, lang, raw_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Deliberately touches NO identity column: `source_id`, `source_type`,
# `external_id`, `url` and `canonical_url` are absent. The unique-indexed ones
# are the keys `_find_existing_item` searched on — rewriting them here would let
# an upsert mutate a row's identity (churning it run to run) or collide with a
# *different* row that legitimately holds the incoming key, so omitting them
# also makes this statement structurally incapable of raising IntegrityError.
# `source_type` is omitted because it is functionally determined by `source_id`;
# see `_merge_item`. Identity is write-once; only payload merges.
_UPDATE_ITEM = """
    UPDATE items SET
        title         = ?,
        author        = ?,
        published_at  = ?,
        summary       = ?,
        content       = ?,
        content_hash  = ?,
        lang          = ?,
        raw_path      = ?
    WHERE id = ?
"""


def _merge_item(existing: Item, incoming: Item) -> Item:
    """Fold `incoming` into `existing`, returning the row we should store.

    Two rules:

    * **Identity is write-once.** `source_id`, `source_type`, `external_id`,
      `url`, `canonical_url` keep their first-seen values. Fields that determine
      each other move together, so the stored row is always coherent:
      `url`/`canonical_url` (`canonical_url` always equals
      `canonicalize_url(url)`), and `source_id`/`source_type` (the type is a
      property of the adapter that owns that key in `sources.yaml`, not
      free-floating payload — a row pinned to `source_id='simonwillison'` must
      never report `source_type='hn'`). A genuine URL migration or
      reclassification is rare enough to be a backfill's job, not a silent side
      effect of a daily re-ingest.
    * **Payload takes the freshest non-null value.** A re-ingest carries only
      feed-level data, so a NULL must never wipe richer data we already hold —
      above all `content`, the lazily-fetched full text a top-N deep read paid
      an LLM call to obtain (DESIGN §3).

    The write-once rule is what makes cross-source dedup safe. The same post
    reaching us from an RSS feed and from HN lands on one `canonical_url` by
    design (see `models.py`), so the second upsert is an *update by a different
    adapter*. First writer wins the attribution; the row keeps saying it came
    from the feed that actually carried it.

    `fetched_at` is preserved as first-seen, which is what makes a double run a
    byte-for-byte no-op rather than merely duplicate-free (CLAUDE.md §3).
    `content_hash` is left blank so the model recomputes it from the *merged*
    title + summary — a hash that disagreed with the stored text would silently
    poison exact dedup.
    """
    return Item(
        id=existing.id,
        source_id=existing.source_id,
        source_type=existing.source_type,
        external_id=existing.external_id,
        url=existing.url,
        canonical_url=existing.canonical_url,
        title=incoming.title,
        author=incoming.author or existing.author,
        published_at=incoming.published_at or existing.published_at,
        fetched_at=existing.fetched_at,
        summary=incoming.summary or existing.summary,
        content=incoming.content or existing.content,
        content_hash="",
        lang=incoming.lang or existing.lang,
        raw_path=incoming.raw_path or existing.raw_path,
    )


def _find_existing_item(conn: sqlite3.Connection, item: Item) -> Item | None:
    """Resolve `item` against BOTH unique constraints on `items`.

    An incoming item can collide on `canonical_url` OR on
    `(source_id, external_id)` — a single `INSERT ... ON CONFLICT` targets one
    index and would raise IntegrityError on the other. So we probe both.

    When the two probes disagree the item bridges two existing rows (e.g. a
    feed republished an old post under a new guid at a URL we already hold).
    `canonical_url` wins: it is the cross-source identity of the document,
    while `external_id` is one publisher's private bookkeeping. We log the
    ambiguity and never merge or delete — silently destroying a row that
    `scores`/`feedback` reference is far worse than leaving two.
    """
    by_url = conn.execute(
        "SELECT * FROM items WHERE canonical_url = ?", (item.canonical_url,)
    ).fetchone()

    by_ext: sqlite3.Row | None = None
    if item.external_id is not None:
        # SQLite treats NULLs as distinct in a UNIQUE index, so rows with a NULL
        # external_id do not collide with each other — only probe when we have one.
        by_ext = conn.execute(
            "SELECT * FROM items WHERE source_id = ? AND external_id = ?",
            (item.source_id, item.external_id),
        ).fetchone()

    if by_url is not None and by_ext is not None and by_url["id"] != by_ext["id"]:
        logger.warning(
            "item matches two different existing rows; keeping the canonical_url match",
            extra={
                "source_id": item.source_id,
                "canonical_url_item_id": by_url["id"],
                "external_id_item_id": by_ext["id"],
                "external_id": item.external_id,
            },
        )
        return _row_to_item(by_url)

    matched = by_url if by_url is not None else by_ext
    return _row_to_item(matched) if matched is not None else None


def _update_params(item: Item, item_id: int) -> tuple[object, ...]:
    """Bindings for `_UPDATE_ITEM` — payload only, in statement order."""
    return (
        item.title,
        item.author,
        _to_iso(item.published_at),
        item.summary,
        item.content,
        item.content_hash,
        item.lang,
        item.raw_path,
        item_id,
    )


def upsert_item(conn: sqlite3.Connection, item: Item) -> tuple[int, bool]:
    """Insert `item`, or update the row it collides with. Returns `(item_id, is_new)`.

    Idempotency is non-negotiable (CLAUDE.md §3): calling this twice with the
    same item yields exactly one row, and `is_new` is True only on the call
    that created it — so a caller summing `is_new` gets an accurate
    `runs.items_new`.

    Deduplication spans both unique constraints; see `_find_existing_item`. On a
    hit, `_merge_item` folds the incoming data into the stored row without
    touching any identity column.

    The whole resolve-then-write sequence runs inside a single IMMEDIATE
    transaction so the lookup cannot go stale before the write, and an
    IntegrityError from a lost race is retried once as an update rather than
    surfacing to the caller.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = _find_existing_item(conn, item)

        if existing is None:
            try:
                cursor = conn.execute(
                    _INSERT_ITEM,
                    (
                        item.source_id,
                        item.source_type.value,
                        item.external_id,
                        item.url,
                        item.canonical_url,
                        item.title,
                        item.author,
                        _to_iso(item.published_at),
                        _to_iso(item.fetched_at),
                        item.summary,
                        item.content,
                        item.content_hash,
                        item.lang,
                        item.raw_path,
                    ),
                )
            except sqlite3.IntegrityError:
                # Another writer inserted a colliding row between our probe and
                # this INSERT. Re-resolve and fall through to the update path.
                existing = _find_existing_item(conn, item)
                if existing is None:
                    raise
            else:
                new_id = int(cursor.lastrowid or 0)
                conn.execute("COMMIT")
                logger.debug(
                    "inserted item",
                    extra={"item_id": new_id, "source_id": item.source_id},
                )
                return new_id, True

        item_id = existing.id
        if item_id is None:  # pragma: no cover — every stored row has a primary key
            raise RuntimeError(f"item matched in {item.source_id!r} has no id")
        conn.execute(_UPDATE_ITEM, _update_params(_merge_item(existing, item), item_id))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    logger.debug("updated existing item", extra={"item_id": item_id, "source_id": item.source_id})
    return item_id, False


def get_item(conn: sqlite3.Connection, item_id: int) -> Item | None:
    """Fetch one item by primary key, or None."""
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_item(row) if row is not None else None


def get_item_by_canonical_url(conn: sqlite3.Connection, canonical_url: str) -> Item | None:
    """Fetch one item by its canonical URL (the cross-source dedup key), or None."""
    row = conn.execute("SELECT * FROM items WHERE canonical_url = ?", (canonical_url,)).fetchone()
    return _row_to_item(row) if row is not None else None


# --------------------------------------------------------------------------- #
# scores — read side for report/daily.py
#
# `score/` (another agent's work-in-progress) owns writing this table. These
# are read-only queries added for the report writer; they touch no insert path
# and must not be treated as a source of truth for scoring behaviour.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DigestItem:
    """One kept item plus its score row — the join `report/daily.py` needs.

    Not a `items`/`scores` row in its own right, just a read-side view; there
    is no writer for this shape.
    """

    item: Item
    signal: int | None
    relevance: int | None
    novelty: int | None
    reasoning: str
    model: str
    rubric_version: str
    scored_at: datetime


def _row_to_digest_item(row: sqlite3.Row) -> DigestItem:
    return DigestItem(
        item=_row_to_item(row),
        signal=row["signal"],
        relevance=row["relevance"],
        novelty=row["novelty"],
        reasoning=row["reasoning"],
        model=row["model"],
        rubric_version=row["rubric_version"],
        scored_at=_from_iso(row["scored_at"]) or datetime.fromtimestamp(0),
    )


_SELECT_DIGEST_ITEMS = """
    SELECT items.*, scores.signal, scores.relevance, scores.novelty,
           scores.reasoning, scores.model, scores.rubric_version, scores.scored_at
    FROM scores
    JOIN items ON items.id = scores.item_id
    WHERE scores.triage = 'keep'
      AND scores.scored_at >= ? AND scores.scored_at < ?
    ORDER BY (COALESCE(scores.signal, 0) + COALESCE(scores.relevance, 0)
              + COALESCE(scores.novelty, 0)) DESC,
             items.id ASC
"""


def get_digest_items(conn: sqlite3.Connection, *, start: str, end: str) -> list[DigestItem]:
    """Kept items whose `scored_at` is in the half-open UTC range `[start, end)`, ranked.

    `start`/`end` are UTC ISO-8601 strings (`YYYY-MM-DDTHH:MM:SS+00:00`) marking
    one reader-local calendar day converted to UTC — the caller
    (`report/daily.py`) owns that conversion so this module stays timezone-
    agnostic. Every `scored_at` is stored in the same UTC ISO format, so the
    string comparison is chronological: lexical order equals time order when the
    offset is always `+00:00`.

    A range (not the old `substr(scored_at,1,10)` date-prefix match) is what lets
    the digest's day track a non-UTC operator's calendar without re-storing any
    timestamp locally. Ranking is the sum of the three dimensions, highest
    first, with `item.id` as a stable tie-break, so rendering the same window
    twice is byte-for-byte identical (CLAUDE.md §3): the digest is a pure
    function of `(start, end, db state)`.
    """
    rows = conn.execute(_SELECT_DIGEST_ITEMS, (start, end)).fetchall()
    return [_row_to_digest_item(row) for row in rows]


def count_killed_items(conn: sqlite3.Connection, *, start: str, end: str) -> int:
    """How many items were triaged `kill` in the UTC range `[start, end)`.

    Shares the digest's exact window (see `get_digest_items`) so the footer's
    killed/kept/scored counts all reconcile against one calendar day.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM scores "
        "WHERE triage = 'kill' AND scored_at >= ? AND scored_at < ?",
        (start, end),
    ).fetchone()
    return int(row["n"])


# --------------------------------------------------------------------------- #
# scores — write side for score/
# --------------------------------------------------------------------------- #

_INSERT_SCORE = """
    INSERT INTO scores (
        item_id, triage, signal, relevance, novelty, reasoning,
        rubric_version, model, scored_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def insert_score(
    conn: sqlite3.Connection,
    *,
    item_id: int,
    triage: str,
    signal: int | None,
    relevance: int | None,
    novelty: int | None,
    reasoning: str,
    rubric_version: str,
    model: str,
    scored_at: datetime,
) -> None:
    """Insert one `scores` row.

    `item_id` is the table's primary key (DESIGN §5), so a second insert for an
    already-scored item raises `sqlite3.IntegrityError` rather than silently
    double-writing — idempotency is enforced by `score/`'s caller never
    selecting an already-scored item in the first place (its `WHERE
    scores.item_id IS NULL` query), not by an upsert here. `rubric_version` and
    `model` are required on every row (CLAUDE.md §3) so a later rubric change
    never leaves an ambiguous score behind.
    """
    conn.execute(
        _INSERT_SCORE,
        (
            item_id,
            triage,
            signal,
            relevance,
            novelty,
            reasoning,
            rubric_version,
            model,
            _to_iso(scored_at),
        ),
    )
    logger.debug("inserted score", extra={"item_id": item_id, "triage": triage})


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #


def start_run(conn: sqlite3.Connection, kind: str, *, started_at: datetime) -> int:
    """Open a `runs` row and return its id. Every run is logged — no silent runs."""
    cursor = conn.execute(
        "INSERT INTO runs (kind, started_at) VALUES (?, ?)",
        (kind, _to_iso(started_at)),
    )
    run_id = int(cursor.lastrowid or 0)
    logger.info("run started", extra={"run_id": run_id, "kind": kind})
    return run_id


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str,
    finished_at: datetime,
    items_new: int = 0,
    llm_input_tokens: int = 0,
    llm_output_tokens: int = 0,
    server_tool_requests: int = 0,
    errors: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Close a `runs` row.

    `status` is one of `ok | partial | failed` (DESIGN §5). `errors` is the
    per-source failure list, stored as JSON — one broken source never aborts a
    run, but it must never vanish either (CLAUDE.md §7).

    `server_tool_requests` counts Anthropic server-tool calls (web search) made
    by the run. It defaults to 0 because every existing caller makes none; the
    curation scout is the only path that passes it. Token counts alone cannot
    express that spend — web search bills per call (DESIGN §8).
    """
    encoded = json.dumps(list(errors)) if errors else None
    conn.execute(
        """
        UPDATE runs SET
            finished_at          = ?,
            status               = ?,
            items_new            = ?,
            llm_input_tokens     = ?,
            llm_output_tokens    = ?,
            server_tool_requests = ?,
            errors               = ?
        WHERE id = ?
        """,
        (
            _to_iso(finished_at),
            status,
            items_new,
            llm_input_tokens,
            llm_output_tokens,
            server_tool_requests,
            encoded,
            run_id,
        ),
    )
    logger.info(
        "run finished",
        extra={
            "run_id": run_id,
            "status": status,
            "items_new": items_new,
            "error_count": len(errors) if errors else 0,
        },
    )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One `runs` row, read back — `errors` already decoded from JSON."""

    id: int
    kind: str
    started_at: datetime
    finished_at: datetime | None
    status: str | None
    items_new: int
    errors: list[dict[str, str]]


def get_latest_run(conn: sqlite3.Connection, *, kind: str) -> RunRecord | None:
    """The most recently *started* `runs` row of `kind`, or None if there is none yet.

    Used by the digest footer to surface "yesterday's source failures": the
    digest reads the last `ingest` run's `errors` rather than re-deriving
    failure state itself, keeping `runs.errors` the single monitoring channel
    (CLAUDE.md §7, DESIGN §7).
    """
    row = conn.execute(
        "SELECT * FROM runs WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)
    ).fetchone()
    if row is None:
        return None
    raw_errors = row["errors"]
    decoded = json.loads(raw_errors) if raw_errors else []
    errors = (
        [record for record in decoded if isinstance(record, dict)]
        if isinstance(decoded, list)
        else []
    )
    return RunRecord(
        id=row["id"],
        kind=row["kind"],
        started_at=_from_iso(row["started_at"]) or datetime.fromtimestamp(0),
        finished_at=_from_iso(row["finished_at"]),
        status=row["status"],
        items_new=row["items_new"] or 0,
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# feedback — human-in-the-loop marks (DESIGN §11, Phase 1 capture)
# --------------------------------------------------------------------------- #

_INSERT_FEEDBACK = """
    INSERT INTO feedback (item_id, verdict, note, created_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(item_id, verdict) DO NOTHING
"""


def record_feedback(
    conn: sqlite3.Connection,
    *,
    item_id: int,
    verdict: str,
    note: str | None,
    created_at: datetime,
) -> bool:
    """Record one `(item_id, verdict)` mark, returning True only when it is new.

    Idempotent by construction (CLAUDE.md §3, NEVER rule 4): the
    `ux_feedback_item_verdict` unique index (migration 2) plus `ON CONFLICT DO
    NOTHING` collapse a re-harvested checkbox — or a `mark` repeated on the CLI —
    to a single stored row. `cursor.rowcount == 1` is True on the insert that
    created the row and False on the conflicting no-op, which is what lets both
    the harvester count *new* marks and the CLI say "recorded" vs "already
    marked".

    Marks are the ground-truth set Phase 2 tuning will aggregate; this function
    only *stores* one — nothing here changes scoring (adaptation is Phase 2,
    DESIGN §11).
    """
    cursor = conn.execute(_INSERT_FEEDBACK, (item_id, verdict, note, _to_iso(created_at)))
    recorded = cursor.rowcount == 1
    logger.debug(
        "recorded feedback" if recorded else "feedback already present",
        extra={"item_id": item_id, "verdict": verdict},
    )
    return recorded


def get_feedback(conn: sqlite3.Connection, item_id: int) -> list[sqlite3.Row]:
    """Every stored `feedback` row for `item_id`, ordered by verdict then time."""
    return list(
        conn.execute(
            "SELECT * FROM feedback WHERE item_id = ? ORDER BY verdict, created_at",
            (item_id,),
        ).fetchall()
    )


# --------------------------------------------------------------------------- #
# proposals — adaptive source curation (DESIGN §7.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Proposal:
    """One proposed `sources.yaml` edit, read back with its JSON fields decoded."""

    id: int
    run_id: int | None
    kind: ProposalKind
    dedup_key: str
    payload: dict[str, object]
    """The edit itself — url, source id, repo slug, keyword. Load-bearing: the
    applier writes these values into `sources.yaml`, so it is decoded strictly.
    A corrupt payload raises rather than degrading to `{}`, because a traceback
    is a better outcome than appending a malformed entry to the config."""

    rationale: str
    evidence: tuple[dict[str, str], ...]
    """`[{url, note}]` — guaranteed non-empty. `insert_proposal` refuses to store
    an uncited proposal and the read path drops any row that somehow holds one,
    so no caller can render a claim with no backing URL (CLAUDE.md §5, NEVER 7)."""

    probe: dict[str, object] | None
    """Deterministic health facts, or a recorded failure reason. Advisory detail
    for the digest block, so this is the one column decoded tolerantly."""

    tier: ProposalTier
    status: ProposalStatus
    surface_date: Date
    created_at: datetime
    decided_at: datetime | None
    decision_note: str | None
    applied_at: datetime | None


def _decode_probe(raw: str | None) -> dict[str, object] | None:
    """Decode the advisory `probe` column, tolerating anything unreadable.

    Tolerant on purpose, and only for this column: probe facts are decoration on
    a digest block, so one unreadable row must not make the whole digest
    unrenderable (CLAUDE.md §7). Every other column is either strict or typed.
    """
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("proposal probe column could not be decoded; treating as absent")
        return None
    return decoded if isinstance(decoded, dict) else None


def _decode_evidence(raw: str) -> tuple[dict[str, str], ...]:
    """Decode the evidence list, keeping only entries that carry a usable URL.

    Returns `()` for an unreadable or fully-uncited list; the read path treats
    that as a row to drop rather than a proposal to render. Re-checking here
    rather than trusting the insert is deliberate — citations are the structural
    defence against confabulation (CLAUDE.md §5), and a defence that lives in
    exactly one write path is one refactor away from being gone.
    """
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("proposal evidence could not be decoded; treating as empty")
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(
        {"url": str(entry["url"]), "note": str(entry.get("note", ""))}
        for entry in decoded
        if isinstance(entry, dict) and str(entry.get("url", "")).strip()
    )


def _decode_payload(raw: str, *, proposal_id: int) -> dict[str, object]:
    """Decode the `payload` column strictly — a corrupt edit must not proceed quietly.

    Unlike `probe`, this is what `curate/apply.py` writes into `sources.yaml`.
    Degrading it to `{}` would turn a corrupt row into a malformed config append,
    which is a worse failure than raising: the same reasoning that makes
    `surface_date` a hard `Date.fromisoformat`.
    """
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"proposal {proposal_id} has an undecodable payload: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"proposal {proposal_id} payload is not a JSON object")
    return decoded


def _row_to_proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(
        id=row["id"],
        run_id=row["run_id"],
        kind=ProposalKind(row["kind"]),
        dedup_key=row["dedup_key"],
        payload=_decode_payload(row["payload"], proposal_id=row["id"]),
        rationale=row["rationale"],
        evidence=_decode_evidence(row["evidence"]),
        probe=_decode_probe(row["probe"]),
        tier=ProposalTier(row["tier"]),
        status=ProposalStatus(row["status"]),
        surface_date=Date.fromisoformat(row["surface_date"]),
        created_at=_from_iso(row["created_at"]) or datetime.fromtimestamp(0),
        decided_at=_from_iso(row["decided_at"]),
        decision_note=row["decision_note"],
        applied_at=_from_iso(row["applied_at"]),
    )


def _cited_proposals(rows: Sequence[sqlite3.Row]) -> list[Proposal]:
    """Decode rows to `Proposal`s, dropping any that carry no usable citation.

    This is where `Proposal.evidence`'s non-empty guarantee is actually made good
    on the read side. `insert_proposal` already refuses uncited proposals, so a
    drop here means a row was hand-edited or written by an older shape — logged
    loudly and skipped, never rendered, because an uncited proposal in a digest
    is precisely the confabulation NEVER rule 7 forbids.
    """
    proposals: list[Proposal] = []
    for row in rows:
        proposal = _row_to_proposal(row)
        if not proposal.evidence:
            logger.warning(
                "dropping a proposal with no usable evidence URL",
                extra={"proposal_id": proposal.id, "kind": proposal.kind.value},
            )
            continue
        proposals.append(proposal)
    return proposals


_INSERT_PROPOSAL = """
    INSERT INTO proposals (
        run_id, kind, dedup_key, payload, rationale, evidence, probe, tier,
        status, surface_date, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (kind, dedup_key) DO NOTHING
"""


def insert_proposal(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    kind: ProposalKind,
    dedup_key: str,
    payload: Mapping[str, object],
    rationale: str,
    evidence: Sequence[Mapping[str, str]],
    probe: Mapping[str, object] | None,
    tier: ProposalTier,
    status: ProposalStatus,
    surface_date: Date,
    created_at: datetime,
) -> int | None:
    """Store one proposal, returning its id, or None when it already existed.

    Idempotent by construction (CLAUDE.md §3, NEVER rule 4): the
    `ux_proposals_kind_key` index plus `ON CONFLICT DO NOTHING` mean a weekly
    scout re-suggesting last week's candidate is a no-op, and a candidate the
    operator already rejected can never be re-surfaced for a second decision.
    None is therefore the *expected* return on a steady-state run, not an error.

    `evidence` must contain at least one entry with a non-blank `url`. That is
    checked here rather than trusted, because this is the chokepoint every
    proposal passes through, and an uncited proposal must not be storable at all
    (CLAUDE.md §5, NEVER rule 7).

    `dedup_key` must already be normalized by the caller — `canonicalize_url` for
    a feed, the bare `owner/repo` slug, or the lowercased keyword. Normalizing
    here would hide which shape each kind expects; getting it wrong shows up as a
    duplicate proposal, which the tests cover.

    Only `pending` and `invalid` are insertable. Every other state is reached
    through a guarded transition, so no caller — including a future one — can
    mint an `approved` or `applied` row and bypass the human gate that DESIGN
    §7.1 says has no bypass.
    """
    if status not in (ProposalStatus.PENDING, ProposalStatus.INVALID):
        raise ValueError(
            f"proposals are insertable only as pending or invalid, not {status.value!r}; "
            "approval and application are guarded transitions (DESIGN §7.1)"
        )
    cited = [
        {"url": str(entry["url"]).strip(), "note": str(entry.get("note", ""))}
        for entry in evidence
        if str(entry.get("url", "")).strip()
    ]
    if not cited:
        raise ValueError(
            f"proposal {kind.value} {dedup_key!r} has no evidence URL; "
            "every proposal must cite at least one source (CLAUDE.md §5)"
        )
    if not dedup_key.strip():
        raise ValueError(f"proposal {kind.value} has an empty dedup_key")

    cursor = conn.execute(
        _INSERT_PROPOSAL,
        (
            run_id,
            kind.value,
            dedup_key,
            json.dumps(dict(payload), sort_keys=True),
            rationale,
            json.dumps(cited),
            json.dumps(dict(probe), sort_keys=True) if probe is not None else None,
            tier.value,
            status.value,
            surface_date.isoformat(),
            _to_iso(created_at),
        ),
    )
    if cursor.rowcount != 1:
        logger.debug(
            "proposal already exists; not re-surfacing",
            extra={"kind": kind.value, "dedup_key": dedup_key},
        )
        return None
    proposal_id = int(cursor.lastrowid or 0)
    logger.info(
        "stored proposal",
        extra={"proposal_id": proposal_id, "kind": kind.value, "status": status.value},
    )
    return proposal_id


def get_proposal(conn: sqlite3.Connection, proposal_id: int) -> Proposal | None:
    """One proposal by primary key, or None (also None for an uncited row)."""
    row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    if row is None:
        return None
    found = _cited_proposals([row])
    return found[0] if found else None


def get_proposals(
    conn: sqlite3.Connection,
    *,
    statuses: Sequence[ProposalStatus] | None = None,
    surfaced_on_or_before: Date | None = None,
) -> list[Proposal]:
    """Proposals matching the given filters, oldest surface date first.

    Ordering is `(surface_date, id)` so a digest re-render is byte-identical
    (CLAUDE.md §3) — the block is a pure function of the filters and DB state.

    `statuses=None` means no status filter; an *empty* sequence means "no status
    is acceptable" and returns nothing. The distinction matters because a caller
    building the list dynamically would otherwise get every row — including
    rejected candidates — into a digest the moment its list computed empty.
    """
    clauses: list[str] = []
    params: list[object] = []
    if statuses is not None:
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        params.extend(status.value for status in statuses)
    if surfaced_on_or_before is not None:
        clauses.append("surface_date <= ?")
        params.append(surfaced_on_or_before.isoformat())
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        # The interpolated text is fixed clause fragments and one `?` per status;
        # no caller data reaches the SQL string itself.
        f"SELECT * FROM proposals{where} ORDER BY surface_date ASC, id ASC",  # noqa: S608
        params,
    ).fetchall()
    return _cited_proposals(rows)


def decide_proposal(
    conn: sqlite3.Connection,
    *,
    proposal_id: int,
    status: ProposalStatus,
    decided_at: datetime,
    note: str | None = None,
) -> bool:
    """Record an approve/reject decision, returning True only when it landed.

    The `status = 'pending'` guard is what makes re-harvesting the same vault
    checkbox a no-op (DESIGN §11's harvest-then-overwrite, applied to proposals):
    a proposal already approved, applied, or rejected is left exactly as it was
    and False comes back. It also means a tick added to an *old* digest after the
    fact cannot silently re-open a settled decision.

    `note` is the operator's reason, and is only ever supplied from the CLI — a
    digest checkbox carries no free text. It is what the scout's suppression list
    replays, so a rejection made with a note teaches the next run something.
    """
    if status not in (ProposalStatus.APPROVED, ProposalStatus.REJECTED):
        raise ValueError(f"decide_proposal expects approved or rejected, got {status.value!r}")
    cursor = conn.execute(
        """
        UPDATE proposals SET status = ?, decided_at = ?, decision_note = ?
        WHERE id = ? AND status = ?
        """,
        (
            status.value,
            _to_iso(decided_at),
            note,
            proposal_id,
            ProposalStatus.PENDING.value,
        ),
    )
    decided = cursor.rowcount == 1
    logger.info(
        "proposal decided" if decided else "proposal was not pending; decision ignored",
        extra={"proposal_id": proposal_id, "status": status.value},
    )
    return decided


def mark_proposal_applied(
    conn: sqlite3.Connection, *, proposal_id: int, applied_at: datetime
) -> bool:
    """Flip an approved proposal to applied, returning True only when it landed.

    Guarded on `status = 'approved'` so a second `curate apply` sees False rather
    than re-applying. **That guard protects this row, not the file.** The YAML
    edit and this status flip are two writes: a crash between them leaves the row
    `approved` with the edit already on disk, and the next run would append it
    again. The applier must therefore be idempotent against the `sources.yaml`
    *text* as well — append only when the entry is genuinely absent — because
    duplicating a config block is NEVER rule 4 at the file level, and `db.py`
    cannot prevent it from here.
    """
    cursor = conn.execute(
        "UPDATE proposals SET status = ?, applied_at = ? WHERE id = ? AND status = ?",
        (
            ProposalStatus.APPLIED.value,
            _to_iso(applied_at),
            proposal_id,
            ProposalStatus.APPROVED.value,
        ),
    )
    return cursor.rowcount == 1


_REOPENABLE_STATUSES: Final = (ProposalStatus.REJECTED, ProposalStatus.INVALID)
"""The two states `reopen_proposal` can escape.

`rejected` because an operator may change their mind, and the unique index means
the scout can never re-suggest the candidate on its own. `invalid` because probe
failures are frequently transient — a timeout, a 503, a rate limit — and one bad
Sunday must not permanently remove a good feed from the scout's reachable set.
`applied` is deliberately absent: that state lives in `sources.yaml`, and undoing
it means editing the file."""


def reopen_proposal(conn: sqlite3.Connection, *, proposal_id: int) -> bool:
    """Return a rejected or invalid proposal to pending, returning True when it moved.

    Called two ways: by the operator via `curate reopen` after a rejection they
    regret, and by the probe stage when a previously-`invalid` candidate passes
    its health check on a later run. Both matter because `ux_proposals_kind_key`
    makes every terminal state permanent from the scout's point of view.

    The reopened row keeps its original `surface_date` and its stale `probe`
    facts; the next probe pass overwrites the latter. That is why a reopen alone
    does not re-surface health numbers that could be months old.
    """
    rejected, invalid = _REOPENABLE_STATUSES
    cursor = conn.execute(
        """
        UPDATE proposals SET status = ?, decided_at = NULL, decision_note = NULL
        WHERE id = ? AND status IN (?, ?)
        """,
        (ProposalStatus.PENDING.value, proposal_id, rejected.value, invalid.value),
    )
    reopened = cursor.rowcount == 1
    logger.info(
        "proposal reopened" if reopened else "proposal is not reopenable; left as is",
        extra={"proposal_id": proposal_id},
    )
    return reopened


def update_proposal_probe(
    conn: sqlite3.Connection, *, proposal_id: int, probe: Mapping[str, object]
) -> bool:
    """Replace a proposal's health facts, returning True when a row was updated.

    The re-probe path needs this: `insert_proposal` is `ON CONFLICT DO NOTHING`,
    so a candidate that already has a row cannot have its facts refreshed through
    the insert. Without this function `probe` would be write-once, and the
    lifecycle DESIGN §7.1 describes — re-probe `invalid` rows each run, record why
    the probe failed, reopen the ones that now pass — would be undeliverable.

    Facts are replaced wholesale rather than merged: a probe result describes one
    fetch attempt, and merging last week's item count into this week's failure
    would produce a row describing a state that never existed.
    """
    cursor = conn.execute(
        "UPDATE proposals SET probe = ? WHERE id = ?",
        (json.dumps(dict(probe), sort_keys=True), proposal_id),
    )
    return cursor.rowcount == 1


@dataclass(frozen=True, slots=True)
class RejectedProposal:
    """One entry in the scout's suppression list (DESIGN §7.1)."""

    kind: ProposalKind
    dedup_key: str
    rationale: str
    """The scout's original pitch — kept so the prompt can show what was
    proposed, not just that something was."""

    decision_note: str | None
    """The operator's reason, when they gave one. This is the part that actually
    teaches taste; `rationale` alone would just replay the model's own argument
    back at it."""


def rejected_proposals(conn: sqlite3.Connection, *, limit: int) -> list[RejectedProposal]:
    """The most recently rejected proposals, oldest of those first.

    Fed into the scout prompt so it stops re-suggesting what the operator already
    turned down. The unique index would collapse a repeat suggestion anyway, but
    that wastes the searches and the proposal slot that produced it, and tells the
    scout nothing about why the answer was no.

    **Bounded, like `kept_items`, because it is prompt input on an Opus-priced
    call.** Rejections only accumulate — nothing ever removes one — so an unbounded
    list grows the weekly bill forever and eventually drowns the evidence around it.
    The bound is safe precisely because `ux_proposals_kind_key` is the real
    suppression mechanism: a candidate that falls off the end of this list can still
    never be stored a second time. Losing it from the prompt costs at most one
    wasted proposal slot, not a re-decision.

    Returned oldest-first within the window so the rendered text is stable, while
    the *selection* takes the newest — a rejection from last month says more about
    the operator's current judgment than one from a year ago.
    """
    rows = conn.execute(
        """
        SELECT kind, dedup_key, rationale, decision_note FROM (
            SELECT id, kind, dedup_key, rationale, decision_note FROM proposals
            WHERE status = ? ORDER BY id DESC LIMIT ?
        ) ORDER BY id ASC
        """,
        (ProposalStatus.REJECTED.value, limit),
    ).fetchall()
    return [
        RejectedProposal(
            kind=ProposalKind(row["kind"]),
            dedup_key=row["dedup_key"],
            rationale=row["rationale"],
            decision_note=row["decision_note"],
        )
        for row in rows
    ]


# --------------------------------------------------------------------------- #
# curation evidence — read-only aggregates for curate/gather.py (DESIGN §7.1)
#
# These deliberately return raw rows rather than verdicts about them. The
# judgment ("this source is dead") belongs to the scout, and the ordinal
# reduction of feedback belongs next to `feedback.LADDER` — encoding the ladder
# order in SQL here would be a second copy of a vocabulary that must not drift.
# --------------------------------------------------------------------------- #


def _window_start(since: datetime) -> str:
    """Normalize a window start to a UTC ISO string before comparing it in SQL.

    Every stored timestamp is UTC with a `+00:00` offset (`Item._require_utc`),
    which is what makes SQLite's lexical string comparison chronological. A
    `since` expressed in any other zone is the *same instant* written with a
    different offset, and compares wrong: `2026-07-20T10:00:00+10:00` sorts above
    `2026-07-20T00:00:00+00:00` even though they are identical moments.

    That matters concretely — the operator's zone is UTC+10 and `cli.py` already
    derives dates through `settings.tzinfo`, so a caller passing a local-midnight
    `since` would silently get a truncated or empty window. The scout would then
    read 0 items for healthy sources and propose retiring them.

    A *naive* `since` is treated as UTC, matching `Item._require_utc`'s stated
    convention for the same reason it exists there. `astimezone` alone would not
    do this: on a naive input it assumes **system local time**, and this pipeline
    runs under `TZ=Australia/Brisbane`, so a naive midnight would silently widen
    the window by ten hours. Both coercions are needed for the claim that callers
    cannot get this wrong to actually hold.
    """
    aware = since if since.tzinfo is not None else since.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class SourceYield:
    """Per-source triage outcomes over a window — the retirement evidence.

    Deliberately carries no feedback counts: an item can hold several verdicts,
    so summing them per source double-counts and deflates exactly the way DESIGN
    §11 warns about. Feedback arrives separately, per item, via
    `feedback_verdicts_since`, and the caller reduces each item to its highest
    rung before aggregating.
    """

    source_id: str
    source_type: SourceType
    items_total: int
    kept: int
    killed: int
    unscored: int


def source_yield_stats(conn: sqlite3.Connection, *, since: datetime) -> list[SourceYield]:
    """Triage outcomes per source for items fetched at or after `since`.

    Uses `fetched_at`, not `published_at`: the question is what this source has
    *delivered to us* lately, and a feed republishing old posts is still noise
    the operator had to skim past. Note the deliberate asymmetry with feedback —
    the window is on when we fetched the item, so a mark made this week on an
    item fetched last month falls outside it.
    """
    rows = conn.execute(
        """
        SELECT items.source_id                                            AS source_id,
               items.source_type                                          AS source_type,
               COUNT(*)                                                   AS items_total,
               SUM(CASE WHEN scores.triage = 'keep' THEN 1 ELSE 0 END)    AS kept,
               SUM(CASE WHEN scores.triage = 'kill' THEN 1 ELSE 0 END)    AS killed,
               SUM(CASE WHEN scores.item_id IS NULL THEN 1 ELSE 0 END)    AS unscored
        FROM items
        LEFT JOIN scores ON scores.item_id = items.id
        WHERE items.fetched_at >= ?
        GROUP BY items.source_id, items.source_type
        ORDER BY items.source_id
        """,
        (_window_start(since),),
    ).fetchall()
    return [
        SourceYield(
            source_id=row["source_id"],
            source_type=SourceType(row["source_type"]),
            items_total=int(row["items_total"]),
            kept=int(row["kept"] or 0),
            killed=int(row["killed"] or 0),
            unscored=int(row["unscored"] or 0),
        )
        for row in rows
    ]


def feedback_verdicts_since(
    conn: sqlite3.Connection, *, since: datetime
) -> list[tuple[str, int, str]]:
    """`(source_id, item_id, verdict)` for every mark on an item fetched since `since`.

    One row per stored verdict, unreduced. The caller collapses each item to its
    highest rung using `feedback.LADDER`, which is the single definition of that
    order — see this section's header comment for why the reduction is not done
    in SQL. Row volume is a handful of marks per window, so returning them
    unaggregated costs nothing.

    **Off-ladder verdicts are included.** `missed` is in the `feedback` vocabulary
    but deliberately absent from `LADDER`, so the obvious reduction
    (`max(verdicts, key=LADDER.index)`) raises `ValueError` the first time the
    operator marks something missed. Callers must filter or handle it explicitly.
    It is not noise to discard either: a source whose items keep getting marked
    `missed` is under-weighted, not dead.
    """
    rows = conn.execute(
        """
        SELECT items.source_id AS source_id, feedback.item_id AS item_id,
               feedback.verdict AS verdict
        FROM feedback
        JOIN items ON items.id = feedback.item_id
        WHERE items.fetched_at >= ?
        ORDER BY items.source_id, feedback.item_id
        """,
        (_window_start(since),),
    ).fetchall()
    return [(row["source_id"], int(row["item_id"]), row["verdict"]) for row in rows]


@dataclass(frozen=True, slots=True)
class KeptItem:
    """A kept item, reduced to what curation evidence needs.

    `content` is deliberately absent: it exists only for top-N deep reads, and
    pulling full article text here to count domains would be exactly the kind of
    quiet cost creep DESIGN §8 guards against.
    """

    source_id: str
    url: str
    title: str
    summary: str
    """As stored, which for an RSS item means **HTML already stripped** by
    `rss._clean_summary`. Anchor text survives; hrefs do not. HN items keep their
    markup because the Algolia payload carries it, so link extraction over this
    field covers HN and misses feeds — see `curate.gather` for why that gap is
    accepted rather than worked around."""

    ranking_score: int
    """signal + relevance + novelty, so the caller can show the scout the items
    the operator valued *most* rather than an arbitrary slice."""


def kept_items(conn: sqlite3.Connection, *, since: datetime, limit: int) -> list[KeptItem]:
    """The highest-ranked kept items fetched since `since`, best first.

    Bounded by `limit` because these go into a prompt: this is the one gather
    query whose output size directly moves the scout's input-token bill, so the
    caller states how many it can afford rather than discovering it later.
    """
    rows = conn.execute(
        """
        SELECT items.source_id AS source_id, items.url AS url, items.title AS title,
               COALESCE(items.summary, '') AS summary,
               (COALESCE(scores.signal, 0) + COALESCE(scores.relevance, 0)
                + COALESCE(scores.novelty, 0)) AS ranking_score
        FROM items
        JOIN scores ON scores.item_id = items.id
        WHERE scores.triage = 'keep' AND items.fetched_at >= ?
        ORDER BY ranking_score DESC, items.id ASC
        LIMIT ?
        """,
        (_window_start(since), limit),
    ).fetchall()
    return [
        KeptItem(
            source_id=row["source_id"],
            url=row["url"],
            title=row["title"],
            summary=row["summary"],
            ranking_score=int(row["ranking_score"]),
        )
        for row in rows
    ]
