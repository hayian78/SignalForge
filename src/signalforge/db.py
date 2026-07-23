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
from datetime import datetime
from pathlib import Path
from typing import Final

from signalforge.models import Item, SourceType

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "DigestItem",
    "Migration",
    "RunRecord",
    "connect",
    "connection",
    "count_killed_items",
    "finish_run",
    "get_digest_items",
    "get_feedback",
    "get_item",
    "get_item_by_canonical_url",
    "get_latest_run",
    "insert_score",
    "migrate",
    "record_feedback",
    "start_run",
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

MIGRATIONS: Final[tuple[Migration, ...]] = (
    _MIGRATION_0001_PHASE0,
    _MIGRATION_0002_FEEDBACK_DEDUP,
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
    errors: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Close a `runs` row.

    `status` is one of `ok | partial | failed` (DESIGN §5). `errors` is the
    per-source failure list, stored as JSON — one broken source never aborts a
    run, but it must never vanish either (CLAUDE.md §7).
    """
    encoded = json.dumps(list(errors)) if errors else None
    conn.execute(
        """
        UPDATE runs SET
            finished_at       = ?,
            status            = ?,
            items_new         = ?,
            llm_input_tokens  = ?,
            llm_output_tokens = ?,
            errors            = ?
        WHERE id = ?
        """,
        (
            _to_iso(finished_at),
            status,
            items_new,
            llm_input_tokens,
            llm_output_tokens,
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
