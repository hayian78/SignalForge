"""Daily Digest (DESIGN §13, Phase 0) — deterministic assembly, no LLM calls.

Reads `items`/`scores`/`runs` only (`db.py` is the only module that touches
SQL — CLAUDE.md §3); writes one markdown file per date under
`vault/daily/YYYY-MM-DD.md`. The "why it matters" line is the score row's
stored `reasoning`, lightly trimmed for a 60-second read — this module never
calls an LLM and never regenerates a claim (CLAUDE.md §2, §5, NEVER rule 2).

### What "today" means

A digest date is a pure input, not "now": `build_digest_context` asks for
every kept item whose `scores.scored_at` falls on that calendar date. This is
what makes rendering the same date idempotent (CLAUDE.md §3) — the query
depends only on `(target_date, db state)`, never on when the command happens
to run. The cron entry passes today's date; `--date` lets an operator
re-render (or backfill) any day on demand.

### Citation discipline

Every rendered line carries the item's real `url`. An item somehow missing
one (should be structurally impossible — `Item.url` is required) is logged
and dropped rather than rendered without a citation (CLAUDE.md §5, NEVER
rule 7).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import jinja2

from signalforge.db import DigestItem, count_killed_items, get_digest_items, get_latest_run

__all__ = [
    "DigestContext",
    "DigestLine",
    "SourceFailure",
    "build_digest_context",
    "digest_path",
    "render_digest",
    "write_digest",
]

logger = logging.getLogger(__name__)

_TEMPLATE_NAME = "daily.md.j2"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_WHY_IT_MATTERS_MAX_CHARS = 320
"""Truncation ceiling for the stored `reasoning` line — DESIGN §13's "60-second
read" bounds each item to a skimmable paragraph, not the full triage rationale."""

_INGEST_RUN_KIND = "ingest"
_RUN_LEVEL_SOURCE_ID = "*"
"""Mirrors `cli.py::_RUN_LEVEL_SOURCE_ID` — a failure that belongs to the run
itself (e.g. a crash), not to any one source. The digest footer reports
per-source failures, so these are excluded rather than shown as a "source"."""


@dataclass(frozen=True, slots=True)
class DigestLine:
    """One rendered item — title, why-it-matters, scores, and its citation link."""

    title: str
    url: str
    why_it_matters: str
    signal: int | None
    relevance: int | None
    novelty: int | None


@dataclass(frozen=True, slots=True)
class SourceFailure:
    """One source's failure from the last ingest run, for the digest footer."""

    source_id: str
    message: str


@dataclass(frozen=True, slots=True)
class DigestContext:
    """Everything the template needs — assembled once, rendered once."""

    date: Date
    items: tuple[DigestLine, ...]
    source_failures: tuple[SourceFailure, ...]
    killed_count: int
    scored_count: int
    """Kept + killed for this date — the denominator the footer's count is against."""
    hidden_kept_count: int
    """Kept items ranked below the `daily_max_items` cap — counted in the
    footer rather than rendered, so the digest stays a 60-second read
    (DESIGN §13) without hiding that they exist."""

    @property
    def kept_count(self) -> int:
        """Every kept item for the date — shown plus hidden. Derived here so
        the template renders it rather than doing arithmetic in jinja."""
        return len(self.items) + self.hidden_kept_count


def _trim_reasoning(text: str, *, limit: int = _WHY_IT_MATTERS_MAX_CHARS) -> str:
    """Collapse whitespace and cap length for a one-line skim, never regenerate it."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit].rsplit(" ", 1)[0]
    return f"{truncated}…"


def _to_line(scored: DigestItem) -> DigestLine | None:
    """One digest line, or None if `scored` cannot be cited.

    `Item.url` is required by the model, so this should be unreachable in
    practice — but the citation rule (NEVER rule 7) is enforced here rather
    than trusted, so a future nullable path can never slip a bare claim
    through the report writer.
    """
    if not scored.item.url:
        logger.warning(
            "kept item has no URL; dropping it rather than rendering an uncited line",
            extra={"item_id": scored.item.id},
        )
        return None
    return DigestLine(
        title=scored.item.title,
        url=scored.item.url,
        why_it_matters=_trim_reasoning(scored.reasoning),
        signal=scored.signal,
        relevance=scored.relevance,
        novelty=scored.novelty,
    )


def _source_failures(conn: sqlite3.Connection) -> tuple[SourceFailure, ...]:
    """Per-source failures from the most recent `ingest` run's `runs.errors`.

    Reads the run log rather than re-deriving failure state, keeping
    `runs.errors` the one monitoring channel (CLAUDE.md §7). Run-level errors
    (`source_id == "*"`, e.g. a crash outside any single source) are excluded:
    the footer promises *source* failures, and mislabelling a crash as a
    "source" would be its own small confabulation.
    """
    run = get_latest_run(conn, kind=_INGEST_RUN_KIND)
    if run is None:
        return ()
    return tuple(
        SourceFailure(
            source_id=str(record.get("source_id", "?")),
            message=str(record.get("message", "")),
        )
        for record in run.errors
        if record.get("source_id") != _RUN_LEVEL_SOURCE_ID
    )


def build_digest_context(
    conn: sqlite3.Connection, *, target_date: Date, max_items: int
) -> DigestContext:
    """Assemble everything `daily.md.j2` needs for `target_date`. No writes.

    `max_items` is `thresholds.daily_max_items` from `interests.yaml`
    (CLAUDE.md §4 — the cap is config, never a Python constant). The list from
    `get_digest_items` is already ranked (total score desc, stable tie-break),
    so truncating to the top N is deterministic: same date, same DB state,
    same config ⇒ the same N items in the same order, every render.
    """
    scored_date = target_date.isoformat()
    scored_items = get_digest_items(conn, scored_date=scored_date)
    lines = tuple(line for scored in scored_items if (line := _to_line(scored)) is not None)
    killed_count = count_killed_items(conn, scored_date=scored_date)

    return DigestContext(
        date=target_date,
        items=lines[:max_items],
        source_failures=_source_failures(conn),
        killed_count=killed_count,
        scored_count=len(scored_items) + killed_count,
        hidden_kept_count=max(0, len(lines) - max_items),
    )


def _template_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_digest(context: DigestContext) -> str:
    """Fill `daily.md.j2` with `context`. Pure function — no I/O beyond loading the template."""
    template = _template_env().get_template(_TEMPLATE_NAME)
    return template.render(context=context)


def digest_path(vault_dir: Path, *, target_date: Date) -> Path:
    """Where `target_date`'s digest lives — the same path every time (CLAUDE.md §3)."""
    return vault_dir / "daily" / f"{target_date.isoformat()}.md"


def write_digest(
    conn: sqlite3.Connection, *, target_date: Date, vault_dir: Path, max_items: int
) -> Path:
    """Render and write `target_date`'s digest, overwriting any existing file.

    Idempotent by construction: same date, same query, same path, `write_text`
    replaces the file's contents rather than appending (CLAUDE.md §3, NEVER
    rule 4).
    """
    context = build_digest_context(conn, target_date=target_date, max_items=max_items)
    rendered = render_digest(context)
    path = digest_path(vault_dir, target_date=target_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path
