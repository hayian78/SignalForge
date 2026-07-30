"""Applying an approved proposal to `sources.yaml` (DESIGN §7.1).

This is the only code in the system that edits the operator's config, so it is
written to be boring and reversible.

### Why it edits text instead of loading and dumping YAML

`sources.yaml` is a document, not a data file. Its comments carry the reasoning
behind every past decision — which feeds were dropped and why, which candidates
turned out to have no working feed, what a knob is for. A `yaml.safe_load` /
`safe_dump` round-trip would silently delete all of it and reflow what remained.
So this module performs only two text operations, both of which the file's own
history already uses:

* **add** — append an entry at the end of its block with a dated provenance comment;
* **retire** — comment the existing lines out in place with a dated reason.

A `ruamel.yaml` round-trip loader would preserve comments and is the obvious
alternative. It is not used because it would be a new dependency for a file this
code only ever appends to, and because a round-trip that *mostly* preserves
formatting is harder to reason about than one that provably touches nothing else:
every function here returns the original text unchanged on any doubt.

### Idempotency has to hold against the file, not just the row

Writing the YAML and flipping `proposals.status` to `applied` are two separate
writes. A crash or a validation revert between them leaves an approved proposal
whose edit is already on disk, and the next run would append it a second time — a
duplicated config block, which is NEVER rule 4 at the file level. Every operation
here is therefore a no-op when the file already reflects it, checked by parsing the
current config rather than by trusting the row.

### The safety net

After every edit the result is re-validated through `config.load_sources`. If it
does not parse, the original text is restored and the failure is recorded. The file
is **not** committed to git: the change lands as a working-tree diff for the
operator to read, which is the promise DESIGN §11 makes about every dial-shift.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Final

from signalforge import db
from signalforge.config import SOURCES_FILENAME, ConfigError, SourcesConfig, load_sources
from signalforge.models import ProposalKind, ProposalStatus, canonicalize_url

__all__ = [
    "ApplyOutcome",
    "AppliedChange",
    "apply_approved_proposals",
    "apply_to_text",
]

logger = logging.getLogger(__name__)

_TOP_LEVEL_KEY: Final = re.compile(r"^[A-Za-z_][\w-]*:")
"""A line starting a new top-level block. Used to find where a block ends."""

_MAX_NOTE_CHARS: Final = 200
"""Ceiling on the rationale copied into a YAML comment. The scout's rationale is
written for a digest, and an over-long one would wrap badly in a config file."""


@dataclass(frozen=True, slots=True)
class AppliedChange:
    """One proposal that changed the file, or was found already applied."""

    proposal_id: int
    kind: ProposalKind
    target: str
    already_applied: bool
    """True when the file already reflected this edit. Still marked `applied` in
    the DB — the desired state holds, which is what `applied` means. This is the
    dual-write recovery path, and it is a success, not an error."""


@dataclass(slots=True)
class ApplyOutcome:
    """What one `curate apply` pass did."""

    changes: list[AppliedChange] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    """Per-proposal failures, shaped for `runs.errors` (CLAUDE.md §7)."""

    text_changed: bool = False
    """Whether `sources.yaml` was rewritten, so the caller can point the operator
    at `git diff` only when there is something to read."""


def _one_line_note(text: str) -> str:
    """Flatten a rationale into a single comment-safe line."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_NOTE_CHARS:
        return collapsed
    return collapsed[:_MAX_NOTE_CHARS].rsplit(" ", 1)[0] + "…"


def _provenance(proposal_id: int, today: Date, note: str) -> str:
    """The dated comment every applied change carries.

    Matches the convention already in the file by hand ("Removed 2026-07-24 as part
    of the executive-briefing rebalance: …"), because the next person reading this
    file should not be able to tell which lines a human wrote and which the scout
    proposed — only what the reason was.
    """
    return f"# {today.isoformat()} curate #{proposal_id}: {_one_line_note(note)}"


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_bounds(lines: list[str], header: str) -> tuple[int, int] | None:
    """`(start, end)` line indices of the block introduced by `header`.

    `start` is the header line; `end` is one past the block's last *content* line,
    so an append lands tight against the last entry rather than after a trailing
    blank. Returns None when the header is absent — a config without the block is
    one where this kind of proposal cannot apply.

    Membership is decided by indentation relative to the **header**, not to the
    entries: a block's items are indented *more than the header*, and a YAML list's
    items sit at the same column as each other. An earlier version compared against
    the item indent and required one space more than the items actually have, so
    every block "ended" immediately after its header — appends landed at the top of
    the list instead of the bottom, and searches for an existing entry never found
    one, which silently reported real edits as already applied.
    """
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.rstrip() == header or line.startswith(f"{header} ")
        )
    except StopIteration:
        return None

    header_indent = _line_indent(lines[start])
    end = start + 1
    last_content = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and _line_indent(line) <= header_indent:
            break
        if line.strip():
            last_content = end + 1
        end += 1
    return start, last_content


def _append_to_block(
    lines: list[str], header: str, *, indent: int, entry_lines: list[str], comment: str
) -> list[str] | None:
    """Insert `entry_lines` at the end of `header`'s block, preceded by `comment`."""
    bounds = _block_bounds(lines, header)
    if bounds is None:
        return None
    _, insert_at = bounds
    pad = " " * indent
    block = [f"{pad}{comment}", *entry_lines]
    return lines[:insert_at] + block + lines[insert_at:]


def _comment_out_rss_entry(lines: list[str], source_id: str, comment: str) -> list[str] | None:
    """Comment out the `rss:` entry whose `id` is `source_id`, and its fields.

    An entry runs from its `- id:` line to the next entry, comment, or dedent, so
    the `url` and `weight` lines travel with it. Returns None if the entry is not
    found, which is the already-retired case.
    """
    pattern = re.compile(rf"^(\s*)-\s+id:\s*{re.escape(source_id)}\s*$")
    start = next((index for index, line in enumerate(lines) if pattern.match(line)), None)
    if start is None:
        return None

    match = pattern.match(lines[start])
    assert match is not None  # noqa: S101 - guaranteed by the search above
    indent = len(match.group(1))

    end = start + 1
    while end < len(lines):
        line = lines[end]
        if not line.strip():
            break
        stripped = line.lstrip()
        # The next entry, or any line at or above this entry's indent, ends it.
        if len(line) - len(stripped) <= indent:
            break
        end += 1

    pad = " " * indent
    # Keep each line's indentation *relative to the entry*, so deleting the `# `
    # prefix by hand restores valid YAML. Flattening to `.strip()` would leave the
    # `url:` line at the entry's own indent and silently break an un-comment — and
    # a retirement the operator can reverse in an editor is the point of commenting
    # out rather than deleting.
    commented = [f"{pad}# {lines[index][indent:]}".rstrip() for index in range(start, end)]
    return lines[:start] + [f"{pad}{comment}", *commented] + lines[end:]


def _comment_out_list_item(
    lines: list[str], header: str, value: str, comment: str, *, indent: int
) -> list[str] | None:
    """Comment out a scalar list entry (`    - owner/repo`) inside `header`'s block."""
    bounds = _block_bounds(lines, header)
    if bounds is None:
        return None
    start, end = bounds
    pattern = re.compile(rf"^(\s*)-\s+{re.escape(value)}\s*$")
    index = next(
        (position for position in range(start, end) if pattern.match(lines[position])), None
    )
    if index is None:
        return None
    pad = " " * indent
    return (
        lines[:index] + [f"{pad}{comment}", f"{pad}# {lines[index].strip()}"] + lines[index + 1 :]
    )


def _edit_inline_list(
    lines: list[str], key: str, *, add: str | None, remove: str | None, comment: str
) -> list[str] | None:
    """Add to or remove from a flow-style list (`  keywords: [a, b, c]`).

    The only in-place value edit in this module, because a flow list has no per-item
    line to comment out. The whole line is rewritten and a dated comment is added
    above it, so the change is still explained in the file even though the previous
    value is only recoverable from git.
    """
    pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*\[(?P<items>.*)\]\s*$")
    index = next((position for position, line in enumerate(lines) if pattern.match(line)), None)
    if index is None:
        return None
    match = pattern.match(lines[index])
    assert match is not None  # noqa: S101 - guaranteed by the search above
    indent = match.group(1)
    items = [item.strip() for item in match.group("items").split(",") if item.strip()]

    if add is not None:
        if add in items:
            return None
        items = [*items, add]
    if remove is not None:
        if remove not in items:
            return None
        items = [item for item in items if item != remove]

    rewritten = f"{indent}{key}: [{', '.join(items)}]"
    return lines[:index] + [f"{indent}{comment}", rewritten] + lines[index + 1 :]


def _rss_entry_lines(source_id: str, url: str, weight: float | None) -> list[str]:
    entry = [f"  - id: {source_id}", f"    url: {url}"]
    if weight is not None and weight != 1.0:
        entry.append(f"    weight: {weight}")
    return entry


def apply_to_text(
    text: str, proposal: db.Proposal, *, today: Date, current: SourcesConfig
) -> str | None:
    """Return `text` with `proposal` applied, or None when it is already applied.

    Pure: no file I/O, no DB. `current` is the parsed form of `text`, passed in so
    the already-applied checks read the config rather than re-deriving membership
    from the raw lines — that is the check that makes a repeated apply a no-op even
    if the row still says `approved` (see the module docstring on dual writes).

    **None means "the config already says this", and nothing else.** Every `None`
    returned below comes from a membership check against `current`. If a membership
    check says the target *is* there but the text edit cannot find the lines, that
    is a bug or an unexpected file shape, and it raises — because the alternative is
    the worst failure this module has: reporting a no-op as success, marking the
    proposal `applied`, and quietly discarding the operator's approval while the
    file is unchanged. That is exactly what a bug in `_block_bounds` did during
    development, and it was invisible from the outside.
    """
    lines = text.splitlines()
    comment = _provenance(proposal.id, today, proposal.rationale)
    payload = proposal.payload
    target = proposal.dedup_key
    result: list[str] | None = None

    match proposal.kind:
        case ProposalKind.ADD_RSS:
            url = str(payload.get("url") or target)
            source_id = str(payload.get("id") or "")
            if not source_id:
                raise ValueError(f"proposal {proposal.id} has no source id to add")
            existing_ids = {source.id for source in current.rss}
            existing_urls = {canonicalize_url(source.url) for source in current.rss}
            if source_id in existing_ids or canonicalize_url(url) in existing_urls:
                return None
            weight = payload.get("weight")
            result = _append_to_block(
                lines,
                "rss:",
                indent=2,
                entry_lines=_rss_entry_lines(
                    source_id, url, float(weight) if isinstance(weight, int | float) else None
                ),
                comment=comment,
            )

        case ProposalKind.RETIRE_RSS:
            source_id = str(payload.get("id") or target)
            if source_id not in {source.id for source in current.rss}:
                return None
            result = _comment_out_rss_entry(lines, source_id, comment)

        case ProposalKind.ADD_GITHUB_REPO:
            if current.github is None:
                raise ValueError("cannot add a release watch: sources.yaml has no github block")
            if target in current.github.releases:
                return None
            result = _append_to_block(
                lines,
                "  releases:",
                indent=4,
                entry_lines=[f"    - {target}"],
                comment=comment,
            )

        case ProposalKind.RETIRE_GITHUB_REPO:
            if current.github is None or target not in current.github.releases:
                return None
            result = _comment_out_list_item(lines, "  releases:", target, comment, indent=4)

        case ProposalKind.ADD_HN_KEYWORD | ProposalKind.REMOVE_HN_KEYWORD:
            if current.hackernews is None:
                raise ValueError("cannot edit hn keywords: sources.yaml has no hackernews block")
            adding = proposal.kind is ProposalKind.ADD_HN_KEYWORD
            result = _edit_inline_list(
                lines,
                "keywords",
                add=target if adding else None,
                remove=None if adding else target,
                comment=comment,
            )

        case ProposalKind.ADD_ARXIV_KEYWORD:
            if current.arxiv is None:
                raise ValueError("cannot add an arxiv keyword: sources.yaml has no arxiv block")
            if target in current.arxiv.require_keywords:
                return None
            result = _append_to_block(
                lines,
                "  require_keywords:",
                indent=4,
                entry_lines=[f"    - {target}"],
                comment=comment,
            )

        case ProposalKind.REMOVE_ARXIV_KEYWORD:
            if current.arxiv is None or target not in current.arxiv.require_keywords:
                return None
            result = _comment_out_list_item(lines, "  require_keywords:", target, comment, indent=4)

    if result is None:
        # Reached only if a membership check passed and the text edit then failed to
        # locate its target. Raising is the whole point — see the docstring.
        raise ValueError(
            f"proposal {proposal.id} ({proposal.kind.value} {target!r}) is present in the "
            "parsed config but its lines could not be located in sources.yaml; the file "
            "shape is not what the applier expects, so nothing was changed"
        )
    # Preserve the file's trailing newline exactly; `splitlines` drops it.
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(result) + suffix


def apply_approved_proposals(
    conn: sqlite3.Connection,
    config_dir: Path,
    *,
    today: Date,
    applied_at_now: object = None,
) -> ApplyOutcome:
    """Apply every approved proposal to `sources.yaml`, one at a time.

    Each proposal is applied, re-validated, and marked in the DB before the next is
    considered, so a proposal that produces invalid YAML costs only itself: the file
    is restored to its state before *that* edit and the rest still apply
    (CLAUDE.md §7). Nothing is committed to git — the operator reads the diff.
    """
    from datetime import UTC, datetime  # noqa: PLC0415 - kept local so `today` stays the seam

    stamp = applied_at_now if isinstance(applied_at_now, datetime) else datetime.now(UTC)
    outcome = ApplyOutcome()
    path = config_dir / SOURCES_FILENAME
    approved = db.get_proposals(conn, statuses=[ProposalStatus.APPROVED])
    if not approved:
        return outcome

    text = path.read_text(encoding="utf-8")
    for proposal in approved:
        before = text
        try:
            current = load_sources(config_dir)
            candidate = apply_to_text(text, proposal, today=today, current=current)
        except (ConfigError, ValueError) as exc:
            outcome.errors.append({"source_id": f"proposal-{proposal.id}", "message": str(exc)})
            continue

        if candidate is None:
            # Already reflected in the file. The desired state holds, so record it
            # as applied — this is the dual-write recovery path, not a failure.
            db.mark_proposal_applied(conn, proposal_id=proposal.id, applied_at=stamp)
            outcome.changes.append(
                AppliedChange(
                    proposal_id=proposal.id,
                    kind=proposal.kind,
                    target=proposal.dedup_key,
                    already_applied=True,
                )
            )
            continue

        path.write_text(candidate, encoding="utf-8")
        try:
            load_sources(config_dir)
        except ConfigError as exc:
            # Restore and move on. A config that does not parse is a 6am failure
            # for every command, so it must never survive this function.
            path.write_text(before, encoding="utf-8")
            logger.error(
                "reverted an applied proposal: the result did not validate",
                extra={"proposal_id": proposal.id, "error": str(exc)},
            )
            outcome.errors.append(
                {
                    "source_id": f"proposal-{proposal.id}",
                    "message": f"reverted: applying produced invalid YAML: {exc}",
                }
            )
            continue

        text = candidate
        outcome.text_changed = True
        db.mark_proposal_applied(conn, proposal_id=proposal.id, applied_at=stamp)
        outcome.changes.append(
            AppliedChange(
                proposal_id=proposal.id,
                kind=proposal.kind,
                target=proposal.dedup_key,
                already_applied=False,
            )
        )
        logger.info(
            "applied a proposal to sources.yaml",
            extra={"proposal_id": proposal.id, "kind": proposal.kind.value},
        )

    return outcome
