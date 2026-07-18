"""The triage/scoring system prompt — versioned, and the ONLY place it lives.

CLAUDE.md §3, NEVER rule 5: changing this prompt's text requires bumping
`RUBRIC_VERSION`. That is the *entire* required action — every `scores` row
carries the version that produced it, so two rubric texts sharing one version
tag would silently make historical scores incomparable to new ones. Do not
embed rubric text anywhere else (e.g. inline in `llm.py`); `llm.py` imports
this module's `build_triage_system_prompt` instead of building prompt text
itself, so a rubric edit is always a one-file diff.

The rendered prompt is the frozen prefix `llm.py` marks `cache_control:
ephemeral` (DESIGN §8). It carries the rubric anchors (DESIGN §9) and a
deterministic rendering of `interests.yaml` — sorted, no timestamps or run
IDs (NEVER rule 10) — so the same config always renders to the same bytes and
every batch request across every day can share one cache entry.

No taxonomy.yaml section: `taxonomy.yaml` doesn't exist yet (DESIGN §10 is
Phase 1; CLAUDE.md NEVER rule 15 — don't build ahead of the phase gate).
"""

from __future__ import annotations

import json
from typing import Final

from signalforge.config import InterestsConfig

__all__ = ["RUBRIC_VERSION", "build_triage_system_prompt"]

RUBRIC_VERSION: Final = "triage-v2"
"""Bump this constant — and only this constant — when the prompt text below
changes. Nothing else needs to change for a rubric edit to be tracked."""

_TRIAGE_INSTRUCTIONS: Final = """\
You are the triage stage of SignalForge, a personal AI-engineering \
intelligence pipeline. You are given a batch of items. Each item is a title \
and a short summary only — you are never given the full article text, so \
score on what is in front of you and do not assume more substance than the \
summary actually shows.

For each item, decide `triage` (keep or kill) and score three dimensions on a \
1-5 scale, each with a written one-sentence `reasoning`:

Signal vs hype (`signal`, 1-5):
  5 = working code, benchmarks, or a production report with real numbers
  3 = a credible announcement whose substance is thin
  1 = a press release, "game-changer" language, no artifact

Personal relevance (`relevance`, 1-5), scored against the interests below:
  5 = directly touches a priority topic or the current stack
  3 = adjacent — worth awareness, not central
  1 = an ignored topic (see `ignore` below) or an irrelevant domain

Novelty (`novelty`, 1-5):
  5 = a new capability or approach not previously possible
  3 = a meaningful increment on a known approach
  1 = a restatement of material that is already well known

Mark `triage` as "keep" when the item plausibly clears the inclusion bar given \
by the `thresholds` block below: signal >= `weekly_min_signal` AND relevance >= \
`weekly_min_relevance` AND (signal + relevance + novelty) >= `weekly_min_total` \
— the same bar the weekly brief uses for inclusion. Otherwise mark it "kill". \
When genuinely unsure between keep and kill, kill: this pipeline prizes \
precision over recall, and a noisy digest is worse than a missed item.

Always write `reasoning` — even for a "kill" — as one sentence explaining the \
scores you gave. It is stored permanently as the human-auditable record of \
why an item was scored the way it was; never leave it empty or generic.

Return exactly one result per item you were given, each carrying back the \
`item_id` it was given so results can be matched to items.
"""


def _render_interests(interests: InterestsConfig) -> str:
    """Deterministic JSON rendering of `interests.yaml` for the cached prefix.

    Every list is sorted and nothing here varies run to run — no timestamps,
    no run IDs (NEVER rule 10) — so `build_triage_system_prompt` returns
    byte-identical text for byte-identical config, which is what lets many
    batch requests across many days share a single `cache_control` entry
    instead of each writing a fresh (1.25x-priced) one.
    """
    payload = {
        "priority_topics": sorted(interests.priority_topics),
        "interests": sorted(interests.interests),
        "stack": sorted(interests.stack),
        "learning_goals": sorted(interests.learning_goals),
        "architecture_philosophy": interests.architecture_philosophy,
        "ignore": {
            "topics": sorted(interests.ignore.topics),
            "people": sorted(interests.ignore.people),
            "repos": sorted(interests.ignore.repos),
        },
        "thresholds": {
            "weekly_min_signal": interests.thresholds.weekly_min_signal,
            "weekly_min_relevance": interests.thresholds.weekly_min_relevance,
            "weekly_min_total": interests.thresholds.weekly_min_total,
        },
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def build_triage_system_prompt(interests: InterestsConfig) -> str:
    """The frozen rubric + `interests.yaml`, ready for `cache_control`.

    This is the entire stable prefix (DESIGN §8): rubric text is a Python
    constant, interests render deterministically, and nothing volatile is
    mixed in. The day's items belong after the cache breakpoint, in the
    request `llm.py` builds — never in this string.
    """
    return (
        f"{_TRIAGE_INSTRUCTIONS}\n"
        f"Interests (config/interests.yaml):\n"
        f"{_render_interests(interests)}"
    )
