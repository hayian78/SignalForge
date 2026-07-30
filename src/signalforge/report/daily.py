"""Daily Digest (DESIGN §13, Phase 0) — deterministic assembly, no LLM calls.

Reads `items`/`scores`/`runs` only (`db.py` is the only module that touches
SQL — CLAUDE.md §3); writes one markdown file per date under
`vault/daily/YYYY-MM-DD.md`. The "why it matters" line is the score row's
stored `reasoning`, lightly trimmed for a 60-second read — this module never
calls an LLM and never regenerates a claim (CLAUDE.md §2, §5, NEVER rule 2).

### What "today" means

A digest date is a pure input, not "now": `build_digest_context` asks for
every kept item whose `scores.scored_at` falls on `target_date` — a calendar
date in the operator's configured zone (`settings.yaml`), which
`utc_day_window` converts to the UTC range actually queried. Storage is UTC
throughout; only this boundary is local. That makes rendering idempotent
(CLAUDE.md §3) — the query depends only on `(target_date, tz, db state)`, never
on when the command runs. The cron entry passes today's local date; `--date`
lets an operator re-render (or backfill) any day on demand.

### Citation discipline

Every rendered line carries the item's real `url`. An item somehow missing
one (should be structurally impossible — `Item.url` is required) is logged
and dropped rather than rendered without a citation (CLAUDE.md §5, NEVER
rule 7).
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, tzinfo
from datetime import date as Date
from pathlib import Path

import jinja2

from signalforge.curate.approvals import DECISIONS, proposal_marker
from signalforge.db import (
    DigestItem,
    Proposal,
    count_killed_items,
    get_digest_items,
    get_latest_run,
    get_proposals,
)
from signalforge.feedback import CHECKBOX_VERDICTS, checkbox_marker
from signalforge.models import ProposalStatus, SourceType, flatten_to_single_line

__all__ = [
    "DigestContext",
    "DigestLine",
    "ProposalLine",
    "SourceFailure",
    "build_digest_context",
    "digest_path",
    "render_digest",
    "select_digest_items",
    "utc_day_window",
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

    id: int | None
    """The `items.id`, embedded in the template's feedback-checkbox markers so a
    harvested mark can be tied back to a row (DESIGN §11). None only for a row
    with no primary key (structurally impossible for a stored item), in which
    case no checkbox renders."""
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
class ProposalLine:
    """One proposed `sources.yaml` change, ready to render (DESIGN §7.1).

    A view over `db.Proposal`, not the row itself: the template gets strings it can
    print rather than JSON blobs to traverse, and `report/` keeps doing nothing but
    read the DB and format it.
    """

    id: int
    action: str
    """What it would do, from `ProposalKind.action_label`."""

    target: str
    rationale: str
    evidence: tuple[tuple[str, str], ...]
    """`(url, note)` pairs — never empty, because `db.get_proposals` drops any row
    without a usable citation before it reaches here (CLAUDE.md §5, NEVER rule 7)."""

    tier: str
    """`corpus` or `web`: where the candidate came from, so the reader knows how much
    scrutiny it deserves."""

    pending: bool
    """True when the operator has not decided yet — the only state that renders
    checkboxes. Everything else renders as a settled one-liner."""

    settled_note: str
    """The one-line record for a decided proposal (`approved 2026-08-05`, `invalid:
    feed 404s`). Empty while pending."""

    staged: bool
    """True for a kind whose block no ingestor reads yet, tagged so the digest never
    implies an effect it does not have (NEVER rule 15)."""

    weight: float | None
    """The scout's suggested multiplier, when it proposed one. Shown because it is
    part of what a tick approves, and because seeing it is what makes it easy to
    overrule in the applied diff."""

    probe: str
    """The deterministic health facts as one skimmable line, or empty for a kind with
    nothing to fetch."""

    @property
    def heading(self) -> str:
        """The proposal's title line, staged tag included.

        Composed here rather than with inline `{% if %}` blocks in the template:
        jinja's `trim_blocks` eats the newline after a block tag, so a conditional
        that ends a content line silently joins it to the next one. Assembling the
        string in Python is both correct and testable without rendering.
        """
        staged = " *(staged — no ingestor reads this block yet)*" if self.staged else ""
        return f"{self.action}: {self.target}{staged}"

    @property
    def facts(self) -> str:
        """Provenance, suggested weight, and probe result as one line."""
        parts = [self.tier]
        if self.weight is not None:
            parts.append(
                f"**suggested weight:** {self.weight} (edit it in the diff if you disagree)"
            )
        if self.probe:
            parts.append(f"**checked:** {self.probe}")
        return " · ".join(parts)


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
    """Kept items that did not render — below the `daily_max_items` cap, or
    past their source's crowding limit. Counted in the footer rather than
    rendered, so the digest stays a 60-second read (DESIGN §13) without hiding
    that they exist."""

    proposals: tuple[ProposalLine, ...] = ()
    """Source-curation proposals for this date's block (DESIGN §7.1). Defaulted so
    every existing caller — and every render of a day before curation existed —
    behaves exactly as it did."""

    @property
    def kept_count(self) -> int:
        """Every kept item for the date — shown plus hidden. Derived here so
        the template renders it rather than doing arithmetic in jinja."""
        return len(self.items) + self.hidden_kept_count

    @property
    def pending_proposals(self) -> tuple[ProposalLine, ...]:
        """The ones still awaiting a tick, which is what the block is asking for."""
        return tuple(proposal for proposal in self.proposals if proposal.pending)

    @property
    def settled_proposals(self) -> tuple[ProposalLine, ...]:
        """Already-decided ones, kept visible so re-reading an old digest still
        shows what was approved that week."""
        return tuple(proposal for proposal in self.proposals if not proposal.pending)


def _trim_reasoning(text: str, *, limit: int = _WHY_IT_MATTERS_MAX_CHARS) -> str:
    """Collapse to one line and cap length for a skim, never regenerate it.

    The flattening is load-bearing, not cosmetic: this text renders into a vault file
    whose lines `feedback.py` harvests marks from, so a multi-line value here could
    forge a tick. It was previously safe only as a side effect of joining on
    whitespace for the "60-second read" format — an explicit call means a future
    reformat cannot quietly remove the protection.
    """
    cleaned = flatten_to_single_line(text)
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
        id=scored.item.id,
        # Not flattened here, unlike the proposal fields below: every read path
        # reconstructs an `Item`, so `Item._flatten_title` runs even for a title
        # edited directly in the database. A second call here would be genuinely
        # unreachable code, and an unreachable defence is one nobody can test.
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
            # Flattened because this text comes from an exception message, which can
            # quote a server response — and it renders into the same vault file whose
            # lines carry decisions. Nothing reachable today writes a multi-line one;
            # this is so that stays true without depending on which run kinds feed it.
            source_id=flatten_to_single_line(str(record.get("source_id", "?"))),
            message=flatten_to_single_line(str(record.get("message", ""))),
        )
        for record in run.errors
        if record.get("source_id") != _RUN_LEVEL_SOURCE_ID
    )


def _probe_summary(probe: dict[str, object] | None) -> str:
    """The health facts as one line a human can skim, or "" when there are none.

    Every value is coerced defensively because `db._decode_probe` is the one
    tolerant decoder in the read path — a probe blob is decoration on this block, so
    a malformed one must degrade to a shorter line rather than break the digest
    (CLAUDE.md §7).
    """
    if not probe:
        return ""
    if not probe.get("ok"):
        reason = str(probe.get("error") or "could not be fetched")
        status = probe.get("status_code")
        return f"probe failed: {reason}" + (f" (HTTP {status})" if status else "")

    parts = [f"{probe.get('items_total', 0)} entries"]
    if (fresh := probe.get("items_in_window")) is not None:
        parts.append(f"{fresh} recent")
    if median := probe.get("median_summary_chars"):
        parts.append(f"median {median} chars")
    if newest := probe.get("newest_published_at"):
        # Date only: the block is skimmed, and the hour a feed published is noise.
        parts.append(f"newest {str(newest)[:10]}")
    if label := probe.get("label"):
        parts.append(f"latest by {label}")
    return ", ".join(parts)


def _settled_note(proposal: Proposal, *, tz: tzinfo) -> str:
    """The one-line record shown for a proposal that is no longer pending.

    Dated in the operator's zone from `decided_at`, so the note matches the day they
    remember ticking it rather than a UTC date that can be a day off.
    """
    if proposal.status is ProposalStatus.INVALID:
        # No date: an `invalid` row was never decided, so there is nothing to date.
        reason = str((proposal.probe or {}).get("error") or "failed validation")
        return f"not shown — {reason}"
    when = proposal.decided_at or proposal.applied_at
    stamp = f" {when.astimezone(tz).date().isoformat()}" if when is not None else ""
    note = f"{proposal.status.value}{stamp}"
    return f"{note} — {proposal.decision_note}" if proposal.decision_note else note


def _proposal_lines(
    conn: sqlite3.Connection,
    *,
    target_date: Date,
    tz: tzinfo,
    settled_display_days: int,
) -> tuple[ProposalLine, ...]:
    """Proposals to render for `target_date`, pending first. Reads only.

    Two render rules, both from DESIGN §7.1:

    * **A pending proposal follows forward** into every digest until it is decided,
      so an unanswered question cannot scroll out of sight.
    * **A settled one keeps rendering** for `settled_display_days` after the day it
      first surfaced — not after it was decided — so re-reading a week-old digest
      still shows what was approved that week instead of silently losing the record
      when the file is overwritten.

    `invalid` rows are included in the settled group deliberately. They were never
    shown for approval, but "we considered this and it did not fetch" is worth one
    line: without it, a candidate the operator might have wanted disappears with no
    trace that it was ever looked at.
    """
    proposals = get_proposals(conn, surfaced_on_or_before=target_date)
    lines: list[ProposalLine] = []
    for proposal in proposals:
        pending = proposal.status is ProposalStatus.PENDING
        age = (target_date - proposal.surface_date).days
        if not pending and age > settled_display_days:
            continue
        weight = proposal.payload.get("weight")
        lines.append(
            ProposalLine(
                id=proposal.id,
                action=proposal.kind.action_label,
                target=proposal.dedup_key,
                # Flattened again here, deliberately. `db.insert_proposal` already
                # guarantees single-line text, so for anything the pipeline wrote this
                # is a no-op — but this is the boundary where text becomes *lines of a
                # vault file that a later harvest parses for decisions*, and a row
                # written by hand or by an older shape must not be able to forge one.
                # Two cheap layers beat one load-bearing one.
                rationale=flatten_to_single_line(proposal.rationale),
                evidence=tuple(
                    (entry["url"], flatten_to_single_line(entry.get("note", "")))
                    for entry in proposal.evidence
                ),
                tier=proposal.tier.value,
                pending=pending,
                settled_note="" if pending else _settled_note(proposal, tz=tz),
                staged=proposal.kind.is_staged,
                weight=float(weight) if isinstance(weight, int | float) else None,
                probe=_probe_summary(proposal.probe),
            )
        )
    # Pending first: the block exists to ask for a decision, and the settled lines
    # are a record. Stable within each group because `get_proposals` orders by
    # `(surface_date, id)`, so a re-render is byte-identical (CLAUDE.md §3).
    return tuple(
        [line for line in lines if line.pending] + [line for line in lines if not line.pending]
    )


def select_digest_items(
    scored_items: Sequence[DigestItem],
    *,
    max_items: int,
    max_per_source: int | None = None,
    max_per_github_repo: int | None = None,
) -> list[DigestItem]:
    """The ranked kept items that actually render, after crowding limits.

    `scored_items` arrives already ranked (total score desc, `item.id`
    tie-break). Both limits only ever *remove* candidates and never reorder
    them, so each keeps the top-ranked slice within its group and the digest
    stays a deterministic sub-sequence of the ranking: same rows, same config ⇒
    the same items in the same order (CLAUDE.md §3).

    Ranking — not recency — picks which of a repo's releases represents it.
    Recency reads as the obvious rule and is a trap: prereleases and betas
    publish *after* the stable release they follow, so "newest wins" hands the
    slot to `3.3.0b1` and drops the `3.2.0` that earned the score. The ranking
    already encodes which release is worth reading; defer to it.

    For a `github` source, `source_id` *is* the watched repo (`external_id` is
    `repo@tag`), so grouping releases by project needs no URL parsing and no
    version-string comparison. `max_per_github_repo` is therefore just a
    tighter `max_per_source` for release watches — a repo shipping four
    versions in one window is not four times the news that one blog posting
    four times is.
    """

    def limit_for(scored: DigestItem) -> int | None:
        """The tightest limit that applies to `scored`, or None if unlimited.

        A repo is also a source, so both knobs match a release — the tighter
        one has to win, or `daily_max_per_github_repo: 1` under a looser
        `daily_max_per_source: 2` would be a no-op.
        """
        applicable = [max_per_source]
        if scored.item.source_type == SourceType.GITHUB:
            applicable.append(max_per_github_repo)
        limits = [limit for limit in applicable if limit is not None]
        return min(limits) if limits else None

    taken: Counter[str] = Counter()
    selected: list[DigestItem] = []
    for scored in scored_items:
        limit = limit_for(scored)
        if limit is not None and taken[scored.item.source_id] >= limit:
            continue
        taken[scored.item.source_id] += 1
        selected.append(scored)
        if len(selected) == max_items:
            break

    return selected


def utc_day_window(local_date: Date, tz: tzinfo) -> tuple[str, str]:
    """The UTC ISO `[start, end)` bracketing `local_date` as one day in `tz`.

    Built from the two adjacent local midnights, each converted to UTC — never
    `start + 24h` — so a day shortened or lengthened by a DST transition is
    still exactly one calendar day in `tz`, not a fixed 24-hour slab. With
    `tz=UTC` this collapses to `[DT00:00:00+00:00, (D+1)T00:00:00+00:00)`, i.e.
    the same rows the old date-prefix match selected (backward compatible).
    """
    start_local = datetime.combine(local_date, time.min, tzinfo=tz)
    end_local = datetime.combine(local_date + timedelta(days=1), time.min, tzinfo=tz)
    return start_local.astimezone(UTC).isoformat(), end_local.astimezone(UTC).isoformat()


def build_digest_context(
    conn: sqlite3.Connection,
    *,
    target_date: Date,
    tz: tzinfo = UTC,
    max_items: int,
    max_per_source: int | None = None,
    max_per_github_repo: int | None = None,
    settled_display_days: int = 0,
) -> DigestContext:
    """Assemble everything `daily.md.j2` needs for `target_date`. No writes.

    `target_date` is a calendar date in `tz` (the operator's configured
    `settings.yaml` zone), and `tz` converts it to the UTC window actually
    queried — storage stays UTC, only the day boundary is local. `tz` defaults
    to UTC so a caller that does not care about locale gets the historical
    behaviour unchanged.

    Every limit here is `thresholds.*` from `interests.yaml` (CLAUDE.md §4 —
    caps are config, never Python constants). The list from `get_digest_items`
    is already ranked, and `select_digest_items` only filters it, so the result
    is deterministic: same date, same zone, same DB state, same config ⇒ the
    same items in the same order, every render.

    `hidden_kept_count` counts every kept item that did not render — crowded
    out of its source's slots as well as below the cap — so the footer's total
    still reconciles with `kept_count` and no item is silently dropped
    (CLAUDE.md §7).

    Citability is settled before selection, not after: an uncitable item must
    not take a slot only to be dropped at render (NEVER rule 7), which would
    silently shorten the digest.
    """
    start, end = utc_day_window(target_date, tz)
    scored_items = get_digest_items(conn, start=start, end=end)
    citable = [scored for scored in scored_items if _to_line(scored) is not None]
    selected = select_digest_items(
        citable,
        max_items=max_items,
        max_per_source=max_per_source,
        max_per_github_repo=max_per_github_repo,
    )
    lines = tuple(line for scored in selected if (line := _to_line(scored)) is not None)
    killed_count = count_killed_items(conn, start=start, end=end)

    return DigestContext(
        date=target_date,
        items=lines,
        source_failures=_source_failures(conn),
        killed_count=killed_count,
        scored_count=len(scored_items) + killed_count,
        hidden_kept_count=max(0, len(citable) - len(lines)),
        proposals=_proposal_lines(
            conn,
            target_date=target_date,
            tz=tz,
            settled_display_days=settled_display_days,
        ),
    )


def _template_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    # Render the feedback checkbox through the same function `feedback.py`
    # parses, so the template and the harvester share one wire format and cannot
    # drift (DESIGN §11). This is a pure formatting helper — no DB write crosses
    # the report/ boundary (CLAUDE.md §2).
    env.globals["checkbox_marker"] = checkbox_marker
    # Which boxes render is the vocabulary's decision, not the template's, so
    # adding a rung stays a one-line change in `feedback.py` and the parser can
    # never fall out of step with what the digest offers.
    env.globals["checkbox_verdicts"] = CHECKBOX_VERDICTS
    # Same discipline for the proposal checkboxes: the line the digest writes comes
    # from the function `curate/approvals.py` parses, so the wire format has exactly
    # one definition (DESIGN §7.1).
    env.globals["proposal_marker"] = proposal_marker
    env.globals["proposal_decisions"] = DECISIONS
    return env


def render_digest(context: DigestContext) -> str:
    """Fill `daily.md.j2` with `context`. Pure function — no I/O beyond loading the template."""
    template = _template_env().get_template(_TEMPLATE_NAME)
    return template.render(context=context)


def digest_path(vault_dir: Path, *, target_date: Date) -> Path:
    """Where `target_date`'s digest lives — the same path every time (CLAUDE.md §3)."""
    return vault_dir / "daily" / f"{target_date.isoformat()}.md"


def write_digest(
    conn: sqlite3.Connection,
    *,
    target_date: Date,
    tz: tzinfo = UTC,
    vault_dir: Path,
    max_items: int,
    max_per_source: int | None = None,
    max_per_github_repo: int | None = None,
    settled_display_days: int = 0,
) -> Path:
    """Render and write `target_date`'s digest, overwriting any existing file.

    Idempotent by construction: same date, same zone, same query, same path,
    `write_text` replaces the file's contents rather than appending
    (CLAUDE.md §3, NEVER rule 4).
    """
    context = build_digest_context(
        conn,
        target_date=target_date,
        tz=tz,
        max_items=max_items,
        max_per_source=max_per_source,
        max_per_github_repo=max_per_github_repo,
        settled_display_days=settled_display_days,
    )
    rendered = render_digest(context)
    path = digest_path(vault_dir, target_date=target_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path
