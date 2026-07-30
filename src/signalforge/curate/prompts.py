"""The scout's prompt text — the one place it is written (DESIGN §7.1).

`llm.py` is deliberately not this place. Its docstring says it "never builds
prompt text itself", and that separation is what lets the cost guard read exactly
what gets sent without reading transport code, and lets this text be reviewed as
prose rather than as string concatenation buried in an API call.

### What the scout is and is not asked to do

The prompt gives the model evidence that was already gathered deterministically —
per-source yield, domains the operator's own kept items keep pointing at, the
current inventory — and asks only for judgment on top of it. It is explicitly told
not to count, rank, or recompute anything, because all of that is arithmetic the
pipeline already did (CLAUDE.md §2, NEVER rule 3).

Two instructions carry most of the weight:

* **Every proposal must cite a URL.** `db.insert_proposal` enforces this and will
  raise on a proposal without one, so an uncited suggestion is a wasted slot from
  a call that has already been paid for. Saying so in the prompt is cheaper than
  discarding the output.
* **Retirements matter as much as additions.** Left unsaid, a model asked to
  suggest sources suggests additions, and the source list only ever grows — which
  is half the problem §7.1 exists to fix.
"""

from __future__ import annotations

import json
from typing import Final

from signalforge.config import InterestsConfig, SourcesConfig
from signalforge.db import RejectedProposal, SourceYield
from signalforge.llm import SCOUT_PROPOSE_TOOL_NAME
from signalforge.models import ProposalKind

__all__ = [
    "SCOUT_PROMPT_VERSION",
    "build_scout_system_prompt",
    "build_scout_user_prompt",
]

SCOUT_PROMPT_VERSION: Final = "scout-v1"
"""Bumped whenever this prompt's wording changes materially.

Recorded into each proposal's `payload` rather than a column of its own, so a
suggestion can be traced to the instructions that produced it. Deliberately
weaker machinery than `rubric_version` (DESIGN §5): scores are compared to each
other across months and must be comparable, whereas every proposal is read once
by a human who then approves or rejects it."""

_STAGED_KINDS: Final = tuple(kind.value for kind in ProposalKind if kind.is_staged)


def build_scout_system_prompt(interests: InterestsConfig) -> str:
    """The scout's standing instructions plus the operator's interests.

    `interests.yaml` is injected verbatim-ish because it is the single definition
    of "relevant to me" (DESIGN §11) and the scout's whole job is judging
    candidates against it. Nothing here is a threshold or a source — those live in
    `sources.yaml` and reach the model through the user prompt as evidence.
    """
    ignore_topics = ", ".join(interests.ignore.topics) or "(none)"
    return f"""\
You advise a single engineer on which sources their personal AI-intelligence
pipeline should be reading. Once a week you review how the current sources have
actually performed and what the wider world is paying attention to, and you
propose concrete changes to a config file.

# What the operator cares about

Priority topics: {", ".join(interests.priority_topics)}
Interests: {", ".join(interests.interests)}
Current stack: {", ".join(interests.stack)}
Learning goals: {"; ".join(interests.learning_goals)}
Architecture philosophy: {interests.architecture_philosophy.strip()}
Explicitly ignored topics: {ignore_topics}

This profile is the operator's deliberate choice. Never propose changing it — you
are proposing changes to *where information comes from*, not to what they care
about.

# Your job

You are given evidence that has already been computed: per-source yield numbers,
domains the operator's own valued reading keeps pointing at, the current source
inventory, and candidates previously rejected. Do not recount, re-rank, or
recompute any of it — that arithmetic is already done and you will only introduce
errors. Your contribution is judgment the numbers cannot supply:

* Who is worth reading on these topics *now*. Use web search for this; your
  training data is not current enough to know who started publishing last month
  or who has gone quiet.
* Which existing sources have stopped earning their place.
* What the operator's own reading is pointing at that they do not yet subscribe to.

# Rules

1. **Every proposal must cite at least one URL** in `evidence` — the page you are
   proposing, the post that pointed at it, or the search result that supports your
   claim. A proposal with no citation is discarded unread, so it wastes a slot.
2. **Propose retirements, not just additions.** A source with poor yield over the
   window is a candidate for removal, and saying so is as valuable as finding
   something new. If the evidence supports no retirement this week, propose none.
3. **Prefer few, well-argued proposals.** You have a hard cap and the operator
   reads these while having coffee. Three strong suggestions beat eight weak ones.
   Proposing nothing at all is a valid and sometimes correct answer.
4. **Do not re-propose anything in the rejected list.** It will be discarded, and
   the operator has already told you what they think.
5. **`rationale` is for the operator, not for you.** One or two sentences, naming
   the specific evidence. "Cited three times this month by items you marked
   useful" is useful; "a leading voice in AI" is not.
6. Keep to sources that publish a machine-readable feed or GitHub releases. A
   Twitter account, a Discord, or a podcast without transcripts cannot be ingested
   and is not a proposal, however good it is.
7. Proposals of kind {", ".join(_STAGED_KINDS)} are staged only — the operator's
   pipeline does not read that block yet, so propose them sparingly if at all.

When you have finished searching, call the `{SCOUT_PROPOSE_TOOL_NAME}` tool
exactly once with your proposals. Call it with an empty list if the evidence does
not support any change this week.
"""


def _format_yield(rows: list[SourceYield], feedback_by_source: dict[str, dict[str, int]]) -> str:
    """One line per source: what it delivered and what the operator made of it.

    Rendered as a compact table rather than JSON because it is read by a model
    that reasons better over prose-shaped evidence, and because every column here
    is already a final number — there is nothing for the model to traverse.
    """
    if not rows:
        return "(no items ingested in the window — a first run, or a stalled pipeline)"
    lines = ["source | type | items | kept | killed | operator marks"]
    for row in sorted(rows, key=lambda item: item.source_id):
        marks = feedback_by_source.get(row.source_id, {})
        mark_text = (
            ", ".join(f"{count} {rung}" for rung, count in sorted(marks.items()) if count) or "none"
        )
        lines.append(
            f"{row.source_id} | {row.source_type.value} | {row.items_total} | "
            f"{row.kept} | {row.killed} | {mark_text}"
        )
    return "\n".join(lines)


def _format_inventory(sources: SourcesConfig) -> str:
    """The current source list, so the scout does not propose what already exists."""
    lines = [f"rss: {', '.join(source.id for source in sources.rss) or '(none)'}"]
    if sources.github:
        lines.append(f"github releases: {', '.join(sources.github.releases) or '(none)'}")
    if sources.hackernews:
        lines.append(f"hn keywords: {', '.join(sources.hackernews.keywords) or '(none)'}")
    if sources.arxiv:
        lines.append(f"arxiv keywords: {', '.join(sources.arxiv.require_keywords) or '(none)'}")
    return "\n".join(lines)


def _format_rejected(rejected: list[RejectedProposal]) -> str:
    """The suppression list, with the operator's reason when they gave one.

    The reason is the load-bearing half: replaying only the scout's own earlier
    pitch teaches it nothing about why the answer was no.
    """
    if not rejected:
        return "(nothing rejected yet)"
    lines = []
    for entry in rejected:
        reason = entry.decision_note or "no reason given"
        lines.append(f"- {entry.kind.value} {entry.dedup_key} — rejected: {reason}")
    return "\n".join(lines)


def _format_titles(titles: list[str]) -> str:
    """The kept titles, one per line.

    Load-bearing rather than decorative: `gather._outbound_domains` can only see
    hrefs where markup survived ingest, which excludes every RSS source, so for
    most of the corpus these titles are the *only* evidence of what the operator
    actually values. A scout given yield numbers alone knows which sources are
    quiet but not what subject matter earned a mark.
    """
    if not titles:
        return "(nothing kept in the window yet)"
    return "\n".join(f"- {title}" for title in titles)


def build_scout_user_prompt(
    *,
    sources: SourcesConfig,
    yield_rows: list[SourceYield],
    feedback_by_source: dict[str, dict[str, int]],
    outbound_domains: list[tuple[str, int]],
    kept_titles: list[str],
    rejected: list[RejectedProposal],
    window_days: int,
    max_proposals: int,
) -> str:
    """The week's evidence — everything volatile about this run.

    Kept separate from the system prompt for readability rather than for caching:
    at a weekly cadence there is no cache to preserve (see `llm.run_source_scout`).
    """
    domains = (
        "\n".join(f"- {domain} ({count} references)" for domain, count in outbound_domains)
        or "(none found)"
    )
    return f"""\
# How the current sources performed over the last {window_days} days

{_format_yield(yield_rows, feedback_by_source)}

A high `killed` count against a low `kept` count means the source is delivering
material that does not survive triage. Operator marks are stronger evidence than
either, because they are a human's judgment rather than the pipeline's — but there
are usually few of them, so do not read too much into one.

# Domains the operator's kept reading keeps pointing at

These are outbound links found inside items that survived triage, excluding
domains already in the inventory below. A domain appearing repeatedly here is the
strongest available signal for an addition: the operator is already reading it
second-hand.

{domains}

# What the operator's pipeline actually surfaced and they kept

The highest-ranked items that survived triage in the window. This is the concrete
picture of what earns a place in their reading — use it to judge whether a
candidate publishes this kind of material, not just whether it is well regarded.

{_format_titles(kept_titles)}

# Current inventory (do not propose adding anything already here)

{_format_inventory(sources)}

# Previously rejected — do not propose these again

{_format_rejected(rejected)}

# Your task now

Search the web for what has changed in this space recently, then call
`{SCOUT_PROPOSE_TOOL_NAME}` once with at most {max_proposals} proposals. Fewer is
better. An empty list is a valid answer for a quiet week.
"""


def proposal_payload(*, source_id: str | None, url: str | None, target: str) -> dict[str, object]:
    """The `proposals.payload` JSON for one accepted scout proposal.

    Carries `prompt_version` so a suggestion can be traced back to the wording
    that produced it without adding a column for it.

    No `weight` key: the scout does not choose scoring multipliers, so an added
    feed lands at the identity element and stays there until the operator moves it
    by hand. See `llm.ScoutProposal` for why.
    """
    payload: dict[str, object] = {"target": target, "prompt_version": SCOUT_PROMPT_VERSION}
    if source_id:
        payload["id"] = source_id
    if url:
        payload["url"] = url
    return payload


def dumps_for_prompt(value: object) -> str:
    """Stable JSON for anything embedded in a prompt.

    `sort_keys=True` because unsorted `json.dumps` is a classic silent
    cache-buster (`shared/prompt-caching.md`'s audit list). This call site does not
    cache today, but the habit costs nothing and the next prompt might.
    """
    return json.dumps(value, sort_keys=True)
