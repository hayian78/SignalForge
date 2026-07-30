"""Harvesting approve/reject ticks for proposals out of the vault (DESIGN §7.1).

The operator reads the digest in Obsidian, so that is where the decision has to be
made. The digest renders two GFM checkboxes per pending proposal, each carrying a
self-describing HTML comment, and the next run harvests the ticked ones before the
render overwrites the file — the same harvest-then-overwrite shape `feedback.py`
uses for item marks (DESIGN §11), for the same reason: it keeps the writer
idempotent.

This module deliberately mirrors `feedback.py` rather than sharing code with it.
The two wire formats must be able to diverge — a proposal decision is exclusive
where an item can hold several rungs at once — and the parsers are ten lines each.
A shared abstraction over two callers with different rules is the abstraction the
simplicity bar rejects.

Boundaries held here:

* **Read-only on the vault.** Globs and reads `<vault>/daily/*.md`; never writes,
  edits, or deletes a vault file (NEVER rule 8). The normal render overwrites
  today's digest, as it always did — that is not this module's job.
* **All SQL lives in `db.py`.** Decisions are recorded through `db.decide_proposal`.

### Why a contradictory tick is ignored rather than resolved

Ticking both boxes is not a decision, and guessing which one the operator meant is
how a source gets added that they were trying to reject. Unlike an item mark —
where `noise` and `useful` on one item are both legitimately storable — approve and
reject are mutually exclusive, so both ticked is a corrupted marker and is skipped
with a warning, leaving the proposal pending for the next digest to ask again.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from signalforge import db
from signalforge.models import ProposalStatus

__all__ = [
    "DECISIONS",
    "PROPOSAL_MARK_RE",
    "ApprovalHarvest",
    "ProposalMark",
    "harvest_approvals",
    "parse_proposal_marks",
    "proposal_marker",
]

logger = logging.getLogger(__name__)

DECISIONS: Final = ("approve", "reject")
"""The tick vocabulary, in render order.

Approve first because it is the common case for a proposal the operator bothered
to read. Kept prefix-free (neither is a prefix of the other) so the regex's label
capture is unambiguous regardless of order — the same property `feedback.py`
depends on."""

_DECISION_STATUS: Final = {
    "approve": ProposalStatus.APPROVED,
    "reject": ProposalStatus.REJECTED,
}
"""Tick word to stored status. The vault says "approve"; the DB says "approved" —
mapping in one place stops those two vocabularies drifting into each other."""

# The marker the template emits and this module parses. Anchored to a task-list
# item so a mention of the comment in prose can never be read as a decision. The
# id and decision are captured twice — checkbox label and comment — and
# cross-checked, so a hand-edited half-marker is ignored rather than half-trusted.
_DECISION_ALT: Final = "|".join(DECISIONS)

PROPOSAL_MARK_RE: Final = re.compile(
    r"^\s*-\s*\[(?P<state>[ xX])\]\s*"
    rf"(?P<label>{_DECISION_ALT})\s*"
    rf"<!--\s*sf:proposal=(?P<proposal>\d+)\s+v=(?P<decision>{_DECISION_ALT})\s*-->\s*$"
)


@dataclass(frozen=True, slots=True)
class ProposalMark:
    """One harvested decision: a ticked checkbox recovered from the vault."""

    proposal_id: int
    decision: str

    @property
    def status(self) -> ProposalStatus:
        return _DECISION_STATUS[self.decision]


@dataclass(frozen=True, slots=True)
class ApprovalHarvest:
    """What a harvest pass touched — the counts the CLI and tests assert against."""

    files_scanned: int
    marks_found: int
    decisions_recorded: int
    """Marks that actually changed a proposal. A re-harvested tick is a no-op
    because `db.decide_proposal` guards on `status = 'pending'`, so this is
    ≤ `marks_found` and is 0 on every run after the first."""

    contradictions: int
    """Proposals with both boxes ticked. Skipped, left pending, and counted so the
    CLI can say so — a silently ignored tick would read as the tool losing input."""


def proposal_marker(proposal_id: int, decision: str) -> str:
    """The exact checkbox line the digest renders for one decision.

    Single source of truth for the wire format so the template and
    `parse_proposal_marks` cannot drift; the template calls this through the render
    context, and a round-trip test feeds its output back through the parser.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, not {decision!r}")
    return f"- [ ] {decision} <!-- sf:proposal={proposal_id} v={decision} -->"


def parse_proposal_marks(text: str) -> list[ProposalMark]:
    """Recover every *ticked* decision from `text`. Pure function — no I/O.

    Emits a `ProposalMark` only for a ticked box whose label and comment agree.
    Unticked boxes, non-marker lines, and self-contradictory markers are ignored
    rather than half-trusted. A proposal with **both** boxes ticked yields nothing:
    see the module docstring for why that is not resolved by preferring one.
    """
    ticked: dict[int, list[str]] = {}
    for line in text.splitlines():
        match = PROPOSAL_MARK_RE.match(line)
        if match is None:
            continue
        if match["state"] not in ("x", "X"):
            continue  # offered but not decided
        if match["label"] != match["decision"]:
            logger.warning(
                "ignoring a self-contradictory proposal marker",
                extra={
                    "proposal": match["proposal"],
                    "label": match["label"],
                    "decision": match["decision"],
                },
            )
            continue
        ticked.setdefault(int(match["proposal"]), []).append(match["decision"])

    marks: list[ProposalMark] = []
    for proposal_id, decisions in ticked.items():
        distinct = set(decisions)
        if len(distinct) > 1:
            logger.warning(
                "proposal has both approve and reject ticked; leaving it pending",
                extra={"proposal_id": proposal_id},
            )
            continue
        marks.append(ProposalMark(proposal_id=proposal_id, decision=decisions[0]))
    return marks


def _contradiction_count(text: str) -> int:
    """How many proposals in `text` carry conflicting ticks.

    Recomputed rather than returned from `parse_proposal_marks` so that function
    stays a pure list-of-marks; the count exists only so the CLI can tell the
    operator their tick was not lost, it was ambiguous.
    """
    ticked: dict[int, set[str]] = {}
    for line in text.splitlines():
        match = PROPOSAL_MARK_RE.match(line)
        if match is None or match["state"] not in ("x", "X"):
            continue
        if match["label"] != match["decision"]:
            continue
        ticked.setdefault(int(match["proposal"]), set()).add(match["decision"])
    return sum(1 for decisions in ticked.values() if len(decisions) > 1)


def _now() -> datetime:
    """Decision timestamp. Isolated so a test can freeze it; UTC storage (DESIGN §5)."""
    return datetime.now(UTC)


def harvest_approvals(conn: sqlite3.Connection, vault_dir: Path) -> ApprovalHarvest:
    """Read the daily vault markdown and record every ticked decision. Vault-read-only.

    Globs `<vault_dir>/daily/*.md` and records each decision through
    `db.decide_proposal`, whose `status = 'pending'` guard makes re-harvesting the
    same tick a no-op — which is what lets this run before every apply and every
    render without a second thought (DESIGN §11's harvest-then-overwrite).

    A tick naming a proposal that does not exist is skipped with a warning rather
    than raised: a hand-edited or stale id must never abort the pass (CLAUDE.md §7).
    Scanning every daily file rather than only today's is deliberate — the operator
    may tick a proposal days after it first surfaced, and a pending proposal keeps
    rendering until it is decided.
    """
    daily_dir = vault_dir / "daily"
    files_scanned = 0
    marks_found = 0
    decisions_recorded = 0
    contradictions = 0
    decided_at = _now()

    for path in sorted(daily_dir.glob("*.md")):
        files_scanned += 1
        text = path.read_text(encoding="utf-8")
        contradictions += _contradiction_count(text)
        for mark in parse_proposal_marks(text):
            marks_found += 1
            if db.get_proposal(conn, mark.proposal_id) is None:
                logger.warning(
                    "a tick references an unknown proposal; skipping",
                    extra={"proposal_id": mark.proposal_id, "file": str(path)},
                )
                continue
            try:
                if db.decide_proposal(
                    conn,
                    proposal_id=mark.proposal_id,
                    status=mark.status,
                    decided_at=decided_at,
                ):
                    decisions_recorded += 1
            except Exception:
                # One tick failing must never abort the pass (CLAUDE.md §7).
                logger.exception(
                    "could not record a proposal decision; skipping it",
                    extra={"proposal_id": mark.proposal_id, "file": str(path)},
                )

    logger.info(
        "harvested proposal decisions",
        extra={
            "files_scanned": files_scanned,
            "marks_found": marks_found,
            "decisions_recorded": decisions_recorded,
            "contradictions": contradictions,
        },
    )
    return ApprovalHarvest(
        files_scanned=files_scanned,
        marks_found=marks_found,
        decisions_recorded=decisions_recorded,
        contradictions=contradictions,
    )
