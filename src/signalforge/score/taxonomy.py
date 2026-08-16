"""The deterministic topic tagger — keyword match against `taxonomy.yaml` (DESIGN §10).

**No LLM anywhere in this module** (NEVER rule 3). Deciding whether the words
"export control" appear in a title is not judgment; it is `str.find` with
better manners.

DESIGN §10 sketches a Haiku fallback for items the keywords miss, and it is
deliberately not built here — but **not because it is expensive**. It was
priced at ≈$0.06/month against the real corpus, so DESIGN's "marginal cost
~zero" is accurate and cost is not the argument. The actual reasons are that it
edits the triage prompt (forcing a `RUBRIC_VERSION` bump, which makes every
existing score incomparable to every new one) and that it puts an LLM surface on
a stage that can be verified by reading it. Keyword-only ships first because it
is reviewable; the fallback is a separate, cost-guarded change.

**The taxonomy is config, not code** (CLAUDE.md §4, NEVER rule 6). Not one
keyword appears below — every one is loaded from `config/taxonomy.yaml`.
Growing the tree is an operator YAML edit, the same posture `sources.yaml` and
`interests.yaml` already have.

**Idempotent** (NEVER rule 4). `tag_untagged_items` selects only items with no
current-version row and inserts with `INSERT OR IGNORE` against
`UNIQUE (item_id, topic)`, so a double run changes nothing and costs nothing.

`score/` operates on stored items only and makes no HTTP call (CLAUDE.md §2) —
that holds here trivially, since this module touches nothing but the DB and a
compiled regex.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final

from signalforge.config import TaxonomyConfig
from signalforge.db import upsert_item_topics
from signalforge.score import _error_record

__all__ = [
    "TAXONOMY_VERSION",
    "TagOutcome",
    "TopicMatch",
    "compile_taxonomy",
    "match_topics",
    "stale_topics",
    "tag_untagged_items",
]

logger = logging.getLogger(__name__)

TAXONOMY_VERSION: Final = "tax-v1"
"""Bump when `taxonomy.yaml`'s *keywords* change, exactly as `RUBRIC_VERSION`
is bumped for a prompt change. Adding a brand-new leaf does not require it —
previously-tagged items simply never carried that topic — but editing or
removing an existing leaf's keywords does, because it changes what an already
stored row means. A bump re-tags every item on the next `score` run."""

_TAG_BATCH_SIZE: Final = 500
"""Rows pulled per pass. Tagging is free, so this is only about not holding an
entire back-catalogue of titles in memory on a first run over an old database."""


@dataclass(frozen=True, slots=True)
class TopicMatch:
    """One topic a piece of text matched, and the keyword that fired.

    The keyword is carried through to storage because it is the evidence an
    operator needs to tell a precise match from a lucky substring when deciding
    whether to edit the taxonomy.
    """

    topic: str
    """`"group.leaf"`, e.g. `"industry.strategy"`."""

    keyword: str


@dataclass(slots=True)
class TagOutcome:
    """What one tagging pass achieved — folded into the CLI's report."""

    items_tagged: int = 0
    """Items that received at least one topic."""

    items_examined: int = 0
    topics_written: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)


def compile_taxonomy(taxonomy: TaxonomyConfig) -> tuple[tuple[str, str, re.Pattern[str]], ...]:
    """Pre-compile `(topic, keyword, pattern)` triples, in a stable order.

    Bounded rather than a bare substring test: a short keyword matching inside a
    longer word is what makes a keyword list quietly useless. The bounds are
    `(?<!\\w)`/`(?!\\w)` rather than `\\b`, because `\\b` is defined relative to
    the adjacent character and so misbehaves on a keyword that *ends* in
    punctuation — `\\bc\\+\\+\\b` never matches "c++". The lookaround pair
    behaves identically to `\\b` for ordinary words and correctly for the rest.

    Keywords are `re.escape`d, so an operator can write `node.js` without the
    `.` becoming a wildcard against their own corpus. Multi-word keywords match
    across any run of whitespace, so a phrase split over a line break in a
    summary still counts.

    Sorted so tagging is deterministic: the same text always yields the same
    topics in the same order, which is what makes the golden tests meaningful.
    """
    compiled: list[tuple[str, str, re.Pattern[str]]] = []
    for group in sorted(taxonomy.root):
        for leaf in sorted(taxonomy.root[group]):
            topic = f"{group}.{leaf}"
            for keyword in sorted(taxonomy.root[group][leaf].keywords):
                normalized = keyword.strip().lower()
                if not normalized:
                    continue
                pattern = r"\s+".join(re.escape(part) for part in normalized.split())
                compiled.append(
                    (topic, normalized, re.compile(rf"(?<!\w){pattern}(?!\w)", re.IGNORECASE))
                )
    return tuple(compiled)


def match_topics(
    title: str,
    summary: str | None,
    compiled: tuple[tuple[str, str, re.Pattern[str]], ...],
) -> list[TopicMatch]:
    """Topics matched by `title` + `summary`. Pure function — no I/O, no LLM.

    At most one match per topic: the first keyword to fire wins, because a
    second row for the same `(item_id, topic)` is what `UNIQUE` forbids anyway,
    and "which keyword" only needs one honest answer.
    """
    haystack = title if not summary else f"{title}\n{summary}"
    seen: set[str] = set()
    matches: list[TopicMatch] = []
    for topic, keyword, pattern in compiled:
        if topic in seen:
            continue
        if pattern.search(haystack):
            seen.add(topic)
            matches.append(TopicMatch(topic=topic, keyword=keyword))
    return matches


def _fetch_untagged(
    conn: sqlite3.Connection, *, after_id: int, limit: int
) -> list[tuple[int, str, str | None]]:
    """Items after `after_id` with no topic row at the current taxonomy version.

    `NOT EXISTS` rather than a `LEFT JOIN ... IS NULL`: an item legitimately has
    many topic rows, and the join form would have to be de-duplicated. This also
    picks up items whose rows are all at an older `taxonomy_version`, which is
    what makes a version bump re-tag rather than skip.

    **The `after_id` cursor is load-bearing, not an optimization.** An item that
    matches no keyword writes no row and so stays in this result set forever — by
    design, so a later taxonomy edit can still catch it. Re-issuing the same
    `LIMIT` query would then hand back the identical rows every iteration and
    never terminate. On the real database that is not a corner case: most items
    match nothing, so the very first run would spin forever.
    """
    rows = conn.execute(
        """
        SELECT i.id, i.title, i.summary
        FROM items AS i
        WHERE i.id > ?
          AND NOT EXISTS (
            SELECT 1 FROM item_topics AS t
            WHERE t.item_id = i.id AND t.taxonomy_version = ?
        )
        ORDER BY i.id
        LIMIT ?
        """,
        (after_id, TAXONOMY_VERSION, limit),
    ).fetchall()
    return [(int(row[0]), str(row[1]), row[2]) for row in rows]


def tag_untagged_items(conn: sqlite3.Connection, taxonomy: TaxonomyConfig) -> TagOutcome:
    """Tag every item not yet tagged at the current `TAXONOMY_VERSION`.

    Never raises. One item's write failing must not lose the rest of the pass
    (CLAUDE.md §7) — it stays untagged and is retried next run, exactly how
    `score_unscored_items` treats a failed score row.

    An item that matches nothing is a real and common outcome — the majority of
    a real corpus. It writes no row and is therefore re-examined every run. That
    is deliberate: re-matching is free, and the alternative — a sentinel "no
    topics" row — would need its own cleanup on every taxonomy edit. It is also
    why the pass below walks a cursor rather than re-issuing one `LIMIT` query
    (see `_fetch_untagged`).
    """
    outcome = TagOutcome()
    compiled = compile_taxonomy(taxonomy)
    if not compiled:
        logger.warning("taxonomy has no usable keywords; nothing to tag")
        return outcome

    tagged_at = datetime.now(UTC).isoformat()
    after_id = 0

    while True:
        batch = _fetch_untagged(conn, after_id=after_id, limit=_TAG_BATCH_SIZE)
        if not batch:
            break
        after_id = batch[-1][0]

        for item_id, title, summary in batch:
            outcome.items_examined += 1
            matches = match_topics(title, summary, compiled)
            if not matches:
                continue
            try:
                upsert_item_topics(
                    conn,
                    item_id=item_id,
                    topics=[(m.topic, m.keyword) for m in matches],
                    taxonomy_version=TAXONOMY_VERSION,
                    tagged_at=tagged_at,
                )
            except sqlite3.Error as exc:
                logger.exception("persisting topics failed", extra={"item_id": item_id})
                outcome.errors.append(_error_record(str(item_id), exc))
            else:
                outcome.items_tagged += 1
                outcome.topics_written += len(matches)

    logger.info(
        "tagged items",
        extra={
            "items_examined": outcome.items_examined,
            "items_tagged": outcome.items_tagged,
            "topics_written": outcome.topics_written,
        },
    )
    return outcome


def stale_topics(
    conn: sqlite3.Connection,
    taxonomy: TaxonomyConfig,
    *,
    now: datetime,
    days: int,
) -> list[str]:
    """Taxonomy leaves that have matched nothing inside the window (DESIGN §10).

    A leaf nobody's corpus ever hits is either a dead interest or a badly chosen
    keyword. Either way it is the operator's edit to make, so this only reports.
    A leaf that has *never* matched counts as stale — that is the loudest version
    of the same signal.
    """
    cutoff = (now - timedelta(days=days)).isoformat()
    fresh = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT topic FROM item_topics WHERE taxonomy_version = ? AND tagged_at >= ?",
            (TAXONOMY_VERSION, cutoff),
        ).fetchall()
    }
    return [
        f"{group}.{leaf}"
        for group in sorted(taxonomy.root)
        for leaf in sorted(taxonomy.root[group])
        if f"{group}.{leaf}" not in fresh
    ]
