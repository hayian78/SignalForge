"""Phase 1 feedback capture — harvest thumbs-up/down marks out of the vault.

The digest is read in Obsidian, but `mark` lives in a terminal (DESIGN §11).
So the daily template renders two GFM task-list checkboxes per item, each
self-describing via an HTML comment marker, and the next pipeline run *harvests*
the checked ones out of the vault markdown before it re-renders (overwrites)
that file — "harvest-then-overwrite" keeps the writer idempotent (DESIGN §11).

Boundaries this module holds:

* **Read-only on the vault.** `harvest_marks` only globs and reads
  `<vault>/daily/*.md`; it never writes, edits, or deletes a vault file
  (CLAUDE.md §8, NEVER rule 8). The normal render overwrites today's file, as
  it always did — that is not this module's job.
* **All SQL lives in `db.py`.** Feedback rows are written through
  `db.record_feedback` (CLAUDE.md §3); this module owns only the parsing and the
  glob.
* **`report/` stays read-only on the DB.** The harvest that *writes* feedback
  cannot live in `report/daily.py` (CLAUDE.md §2), so it lives here and is
  driven from the `digest` CLI command.

The marker format is a wire format the template and the parser must agree on,
not a tuning knob — so it is a module constant, never YAML config (CLAUDE.md §4).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from signalforge import db

__all__ = [
    "MARK_RE",
    "VERDICTS",
    "HarvestResult",
    "Mark",
    "checkbox_marker",
    "harvest_marks",
    "parse_marks",
]

logger = logging.getLogger(__name__)

VERDICTS: Final = ("useful", "noise", "missed")
"""The `feedback.verdict` vocabulary (DESIGN §5). `useful`/`noise` render as
digest checkboxes; `missed` — "this should have been surfaced" — is CLI-only,
because a rendered item was, by definition, surfaced."""

_CHECKBOX_VERDICTS: Final = ("useful", "noise")
"""The verdicts that render as a checkbox affordance. Not `missed`."""

# The marker the template emits and this module parses. Anchored to a task-list
# item so a bare mention of the comment elsewhere in prose can never be mistaken
# for a mark. Groups: the checkbox state, the id, and the verdict — the id and
# verdict are captured twice (checkbox label + comment) and cross-checked so a
# hand-edited half-marker is ignored rather than half-trusted.
MARK_RE: Final = re.compile(
    r"^\s*-\s*\[(?P<state>[ xX])\]\s*"
    r"(?P<label>useful|noise)\s*"
    r"<!--\s*sf:item=(?P<item>\d+)\s+v=(?P<verdict>useful|noise)\s*-->\s*$"
)


@dataclass(frozen=True, slots=True)
class Mark:
    """One harvested thumbs-up/down: a checked checkbox recovered from the vault."""

    item_id: int
    verdict: str


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """What a harvest pass touched — the counts the CLI/tests assert against."""

    files_scanned: int
    marks_found: int
    rows_recorded: int
    """Marks that produced a *new* `feedback` row. A re-harvested checkbox is a
    no-op (`db.record_feedback` returns False), so this is ≤ `marks_found`."""


def checkbox_marker(item_id: int, verdict: str) -> str:
    """The exact self-describing checkbox line the template renders for a mark.

    Single source of truth for the wire format so the template and `parse_marks`
    cannot drift — the template calls this via the render context, and a
    round-trip test feeds its output back through `parse_marks`.
    """
    return f"- [ ] {verdict} <!-- sf:item={item_id} v={verdict} -->"


def parse_marks(text: str) -> list[Mark]:
    """Recover every *checked* mark from `text`. Pure function — no I/O.

    Emits a `Mark` only for a checked box (`[x]`/`[X]`) whose checkbox label and
    comment verdict agree; unchecked boxes, non-mark lines, and malformed or
    self-contradictory markers are ignored rather than half-trusted. Order
    follows the text, top to bottom.
    """
    marks: list[Mark] = []
    for line in text.splitlines():
        match = MARK_RE.match(line)
        if match is None:
            continue
        if match["state"] not in ("x", "X"):
            continue  # unchecked — surfaced but not marked
        if match["label"] != match["verdict"]:
            # Label and comment disagree: a hand-edit corrupted the marker.
            logger.warning(
                "ignoring a self-contradictory feedback marker",
                extra={"item": match["item"], "label": match["label"], "verdict": match["verdict"]},
            )
            continue
        marks.append(Mark(item_id=int(match["item"]), verdict=match["verdict"]))
    return marks


def _now() -> datetime:
    """Harvest timestamp. Isolated so a test can freeze it; UTC storage (DESIGN §5)."""
    return datetime.now(UTC)


def harvest_marks(conn: sqlite3.Connection, vault_dir: Path) -> HarvestResult:
    """Read the daily vault markdown and store every checked mark. Vault-read-only.

    Globs `<vault_dir>/daily/*.md`, parses each file (never writes one — NEVER
    rule 8), and records each mark through `db.record_feedback`. A mark whose
    `item_id` is not in `items` is skipped with a warning rather than raised: a
    hand-edited or stale id must never abort the harvest (CLAUDE.md §7). The
    unique index makes a re-harvest of the same checkbox a no-op, so this is
    safe to run before every render (DESIGN §11 harvest-then-overwrite).
    """
    daily_dir = vault_dir / "daily"
    files_scanned = 0
    marks_found = 0
    rows_recorded = 0
    # Each mark gets its own `created_at`, base + microsecond offset by ordinal.
    # The migration-1 PRIMARY KEY is (item_id, created_at), so a shared timestamp
    # would make two verdicts on one item collide on the PK and silently drop the
    # second — diverging from the CLI path, which records both. Distinct stamps
    # let both persist; the UNIQUE(item_id, verdict) index still makes a
    # re-harvest a no-op regardless of timestamp, so idempotency holds.
    base = _now()

    for path in sorted(daily_dir.glob("*.md")):
        files_scanned += 1
        text = path.read_text(encoding="utf-8")
        for mark in parse_marks(text):
            created_at = base + timedelta(microseconds=marks_found)
            marks_found += 1
            if db.get_item(conn, mark.item_id) is None:
                logger.warning(
                    "feedback mark references an unknown item; skipping",
                    extra={"item_id": mark.item_id, "verdict": mark.verdict, "file": str(path)},
                )
                continue
            try:
                if db.record_feedback(
                    conn,
                    item_id=mark.item_id,
                    verdict=mark.verdict,
                    note=None,
                    created_at=created_at,
                ):
                    rows_recorded += 1
            except Exception:
                # One mark's write failing must never abort the pass (CLAUDE.md
                # §7). With distinct timestamps the expected dual-verdict path no
                # longer collides, so reaching here is a genuinely unexpected
                # error — logged loudly, not swallowed as normal flow.
                logger.exception(
                    "could not record a feedback mark; skipping it",
                    extra={"item_id": mark.item_id, "verdict": mark.verdict, "file": str(path)},
                )

    logger.info(
        "harvested vault feedback marks",
        extra={
            "files_scanned": files_scanned,
            "marks_found": marks_found,
            "rows_recorded": rows_recorded,
        },
    )
    return HarvestResult(
        files_scanned=files_scanned, marks_found=marks_found, rows_recorded=rows_recorded
    )
