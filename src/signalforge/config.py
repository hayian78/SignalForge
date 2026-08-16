"""Validated shapes for `config/*.yaml` — config is data, not code (CLAUDE.md §4).

This module defines the *shape* of the configuration. Every value — source
URLs, keyword lists, thresholds — lives in YAML. Adding a blog is a YAML edit,
never a Python edit.

Secrets never appear in YAML (CLAUDE.md §10 rule 16). They are read from the
environment (or `.env`) by `get_secret`, held as `SecretStr`, and never logged.
`sources.yaml` names the *env var* to read (`token_env: GITHUB_TOKEN`), never
the token itself.
"""

from __future__ import annotations

import logging
import os
import re
from email.utils import parseaddr
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from signalforge.models import has_control_characters

__all__ = [
    "SETTINGS_FILENAME",
    "SOURCES_FILENAME",
    "TAXONOMY_FILENAME",
    "ArxivConfig",
    "ConfigError",
    "CurationConfig",
    "DeliveryConfig",
    "EmailChannelConfig",
    "GithubConfig",
    "HackerNewsConfig",
    "IgnoreRules",
    "InterestsConfig",
    "PodcastChannelConfig",
    "RssSource",
    "SettingsConfig",
    "SourceDefaults",
    "SourcesConfig",
    "TaxonomyConfig",
    "TaxonomyLeaf",
    "Thresholds",
    "VaultGitConfig",
    "get_secret",
    "load_interests",
    "load_settings",
    "load_sources",
    "load_taxonomy",
]

logger = logging.getLogger(__name__)

SOURCES_FILENAME: Final = "sources.yaml"
INTERESTS_FILENAME: Final = "interests.yaml"
SETTINGS_FILENAME: Final = "settings.yaml"
TAXONOMY_FILENAME: Final = "taxonomy.yaml"

AWESOME_MAX_NEW_PER_RUN_CEILING: Final = 100
"""Hard ceiling on `github.awesome_max_new_per_run`.

DESIGN §8's own lesson, learned the expensive way on the scout's search budget:
a ceiling that permits values the budget forbids is not a ceiling. Every
awesome-list entry that becomes an item is billed at triage, so without this a
one-line YAML edit to `awesome_max_new_per_run: 5000` moves real spend with the
whole suite still green.

Lives here rather than beside the ingestor because `config.py` is the bottom
layer — the reverse import is a cycle, which is exactly why the scout's ceiling
has to be clamped at use instead of bounded at load. A cap that *can* be checked
at load should be.

100 items is roughly $0.03 of Haiku triage at the measured per-item rate, and
far above any real list's weekly churn, so it constrains mistakes and nothing
else."""


class ConfigError(Exception):
    """Raised when a config file is missing, unparseable, or fails validation."""


class _StrictModel(BaseModel):
    """Base for every config model: unknown keys are an error, not a shrug.

    A typo'd YAML key that silently does nothing is the worst failure mode for
    config-as-data — the user edits the file, nothing changes, and there is no
    signal. `extra="forbid"` turns that into a startup error naming the key.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# sources.yaml (DESIGN §7)
# --------------------------------------------------------------------------- #


class SourceDefaults(_StrictModel):
    """The `defaults:` block. Required — these are tuning knobs, so they live in
    YAML with no Python fallback (CLAUDE.md §10 rule 6)."""

    fetch_timeout: int = Field(gt=0, description="Per-request HTTP timeout, seconds.")
    min_hn_points: int = Field(ge=0, description="Front-page HN score floor.")
    max_summary_chars: int = Field(
        gt=0,
        description=(
            "Truncation ceiling for `items.summary`. The triage cost knob: triage reads "
            "titles + summaries only (DESIGN §8), so this bounds the per-item token spend."
        ),
    )
    max_item_age_days: int = Field(
        ge=1,
        description=(
            "Ingest freshness window, days. Items published earlier than this are skipped "
            "before they reach the DB or triage — the guard against a first run (or a newly "
            "added source) backfilling feed history. Items with no parseable published date "
            "are kept, not dropped."
        ),
    )


class RssSource(_StrictModel):
    """One feed under `rss:`."""

    id: str = Field(min_length=1, description="Stable key; becomes `items.source_id`.")
    url: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    """Score multiplier for a trusted author. 1.0 is the identity element — not a
    tuned threshold, so it is safe as a Python default."""


_GITHUB_TOKEN_PREFIXES: Final = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
"""The `<prefix>_` forms GitHub tokens carry (classic PAT, OAuth, user/server,
refresh, fine-grained). Matched against `token_env` to catch a pasted secret —
the trailing underscore is deliberate so the env-var *name* `GITHUB_PAT` passes."""


class GithubConfig(_StrictModel):
    """The `github:` block."""

    token_env: str = Field(min_length=1)
    """Name of the env var holding the PAT — never the token itself."""

    releases: list[str] = Field(default_factory=list)
    """`owner/repo` slugs polled via REST `/releases` (`/tags` fallback)."""

    awesome_lists: list[str] = Field(default_factory=list)
    """`owner/repo` slugs whose README is diffed between runs (DESIGN §7)."""

    awesome_max_new_per_run: int | None = Field(
        default=None, ge=1, le=AWESOME_MAX_NEW_PER_RUN_CEILING
    )
    """Ceiling on how many newly-added list entries become items in one run.

    Required whenever `awesome_lists` is non-empty (see the validator below),
    optional otherwise — the same principle `CurationConfig` states: a spend cap
    must never be a number buried in Python, but a cap on a feature you are not
    using has nothing to cap. A list entry carries no date, so
    `defaults.max_item_age_days` cannot bound it and this is the only thing
    between a maintainer's 300-entry merge and a 300-item triage batch."""

    @model_validator(mode="after")
    def _require_a_cap_for_awesome_lists(self) -> GithubConfig:
        """A configured awesome list must state what it may cost."""
        if self.awesome_lists and self.awesome_max_new_per_run is None:
            raise ValueError(
                "github.awesome_max_new_per_run is required when awesome_lists is set: "
                "a list entry has no date, so this is the only bound on how many new "
                "entries reach triage in one run"
            )
        return self

    @field_validator("releases", "awesome_lists")
    @classmethod
    def _validate_repo_slugs(cls, value: list[str]) -> list[str]:
        for slug in value:
            owner, _, name = slug.partition("/")
            if not owner or not name or "/" in name:
                raise ValueError(f"expected an 'owner/repo' slug, got {slug!r}")
        return value

    @field_validator("token_env")
    @classmethod
    def _reject_inline_secret(cls, value: str) -> str:
        """`token_env` must name an env var, not carry a token.

        Guards the most likely config mistake: pasting a `ghp_...` PAT straight
        into git-tracked YAML. The check matches GitHub token *shapes* — the
        `<prefix>_` form real tokens carry — not a bare name prefix, so the
        legitimate env-var name `GITHUB_PAT` (which starts with `github_pat` but
        is not `github_pat_<body>`) is accepted, while a pasted token is not.
        """
        lowered = value.lower()
        if not value.replace("_", "").isalnum() or lowered.startswith(_GITHUB_TOKEN_PREFIXES):
            raise ValueError(
                "token_env must be the NAME of an environment variable "
                "(e.g. GITHUB_TOKEN), never a token value"
            )
        return value


class ArxivConfig(_StrictModel):
    """The `arxiv:` block, read by `ingest/arxiv.py` (Phase 1, live 2026-08-07)."""

    categories: list[str] = Field(default_factory=list)
    require_keywords: list[str] = Field(default_factory=list)


class HackerNewsConfig(_StrictModel):
    """The `hackernews:` block."""

    keywords: list[str] = Field(default_factory=list)


class CurationConfig(_StrictModel):
    """The `curation:` block — adaptive source curation's knobs (DESIGN §7.1).

    Every field is required, as in `SourceDefaults` (CLAUDE.md §10 rule 6): a
    default buried in code is a spend limit nobody can see. The dividing line is
    not optional-vs-required blocks — `GithubConfig.token_env` is required inside
    an optional block too — but what the default would *be*. The defaults on
    `github`/`arxiv`/`hackernews` are empty collections and `RssSource.weight` is
    the identity element: invisible but inert. A tuned number or a spend cap is
    neither, so it has no Python fallback.

    The *block* is optional, though — omitting `curation:` turns the feature off
    rather than starting it on invented numbers. An operator who wants a weekly
    scout says so, and says what it may cost.
    """

    max_proposals_per_run: int = Field(
        ge=1,
        le=20,
        description=(
            "Change-rate cap: at most N proposals per scout run. Bounded above so a "
            "typo cannot let one week rewire the whole feed — the same discipline as "
            "DESIGN §11's ±0.1/month weight-nudge cap."
        ),
    )

    max_searches_per_run: int = Field(
        ge=0,
        description=(
            "Web-search cap per scout run. This is a money knob, not a quality one: "
            "search bills $10 per 1,000 calls on top of the tokens its results consume "
            "(DESIGN §8), so it is the first dial to turn if the monthly figure in "
            "`signalforge status` surprises you. 0 is meaningful and supported — the "
            "scout then reasons from the stored corpus alone and reaches for nothing "
            "external, which is the cheapest useful mode. No upper bound here because "
            "`llm.py` owns the hard ceiling and clamps this value to it — logging a "
            "warning and recording it so the clamp is never silent; config.py cannot "
            "import `llm.py` to check it at load (llm.py imports config)."
        ),
    )

    yield_window_days: int = Field(
        ge=1,
        le=365,
        description=(
            "Lookback for per-source yield stats. Long enough that a quiet fortnight "
            "does not read as a dead source, short enough that a source which has "
            "genuinely stopped delivering shows up while the operator still cares."
        ),
    )

    settled_display_days: int = Field(
        ge=0,
        le=90,
        description=(
            "How long a decided proposal keeps rendering in a digest as a settled "
            "one-line note, measured from `surface_date` — not from `decided_at`, so a "
            "proposal that sat pending for a week does not then linger for the full "
            "window after it is ticked. Bounds the block's growth without dropping the "
            "record the moment a decision lands: re-rendering an old digest should "
            "still show what was approved that week. 0 hides settled proposals at once."
        ),
    )


class SourcesConfig(_StrictModel):
    """Root model for `sources.yaml`."""

    defaults: SourceDefaults
    rss: list[RssSource] = Field(default_factory=list)
    github: GithubConfig | None = None
    arxiv: ArxivConfig | None = None
    hackernews: HackerNewsConfig | None = None
    curation: CurationConfig | None = None
    """None means adaptive source curation is off; `curate` says so and exits
    rather than guessing its own spend caps (DESIGN §7.1)."""

    @field_validator("rss")
    @classmethod
    def _unique_ids(cls, value: list[RssSource]) -> list[RssSource]:
        seen: set[str] = set()
        for source in value:
            if source.id in seen:
                raise ValueError(f"duplicate rss source id {source.id!r}")
            seen.add(source.id)
        return value


# --------------------------------------------------------------------------- #
# interests.yaml (DESIGN §11)
# --------------------------------------------------------------------------- #


class IgnoreRules(_StrictModel):
    """The `ignore:` block."""

    topics: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    repos: list[str] = Field(default_factory=list)


class Thresholds(_StrictModel):
    """Report tuning knobs: weekly-brief inclusion gates (DESIGN §9) and the
    daily-digest cap (DESIGN §13). Required — thresholds are the canonical
    example of what must never be hardcoded in Python."""

    weekly_min_signal: int = Field(ge=1, le=5)
    weekly_min_relevance: int = Field(ge=1, le=5)
    weekly_min_total: int = Field(ge=3, le=15)

    weekly_top_n: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Weekly Intelligence Brief item cap: the top-N kept items from the brief's "
            "7-day window, after the `weekly_min_*` gates above and the same crowding "
            "limits the digest uses, become source material for `synth/weekly.py`. "
            "Separate from `daily_max_items` for the same reason `podcast_top_n` is — a "
            "read-length knob and an Opus-cost knob are different concerns sharing one "
            "ranking. Bounded by top-N, never by how many items triage keeps, so a "
            "keep-rate improvement cannot silently become a cost multiplier. Whatever is "
            "set here, `llm.WEEKLY_MAX_ITEMS` will be the hard ceiling config can only "
            "lower. None disables the weekly brief."
        ),
    )

    weekly_near_miss_n: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many near-miss items the brief's footer lists: kept items in the window "
            "that failed the `weekly_min_*` gates, top-N in the same ranking. They exist "
            "to make `signalforge mark <id> missed` cheap to give (DESIGN §11) — `missed` "
            "is the highest-value verdict and the one the operator has no other prompt "
            "for. None omits the footer section."
        ),
    )

    daily_max_items: int = Field(
        ge=1,
        description=(
            "Daily Digest cap: only the top-N ranked kept items render (DESIGN §13's \"5–15 "
            'kept items… 60-second read"); the rest are counted in the footer. Deliberately '
            "no upper bound — a single-user tunable, not a score-range gate."
        ),
    )

    daily_max_per_source: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Crowding cap: at most N items from any one `sources.yaml` source may occupy "
            "the digest's `daily_max_items` slots. One prolific source (a link blog, a "
            "busy release watch) otherwise wins slots on volume rather than merit, "
            "crowding out the rest of the ranking. None disables the cap."
        ),
    )

    daily_max_per_github_repo: int | None = Field(
        default=None,
        ge=1,
        description=(
            "A tighter `daily_max_per_source` for release watches: at most N releases "
            "per repo (highest-ranked, not newest — a prerelease publishes after the "
            "stable release it follows). A repo shipping four versions in one window is "
            "one piece of news. None falls back to `daily_max_per_source`."
        ),
    )

    podcast_top_n: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Podcast channel item cap: the top-N ranked kept items (reusing the digest's "
            "own ranking and crowding limits) become script source material for "
            "`synth/podcast.py`. Deliberately independent of `daily_max_items` — the "
            "digest's read-length knob and the podcast's Opus-cost knob are different "
            "concerns that happen to share a ranking. Phase 1 synthesis spend must stay "
            "bounded by this top-N, not by how many items triage keeps, or a keep-rate "
            "improvement silently becomes a cost multiplier. None disables the podcast "
            "channel."
        ),
    )


class InterestsConfig(_StrictModel):
    """Root model for `interests.yaml` — the single definition of "relevant to me".

    Injected (prompt-cached) into every scoring and synthesis prompt.
    """

    priority_topics: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    stack: list[str] = Field(default_factory=list)
    learning_goals: list[str] = Field(default_factory=list)
    architecture_philosophy: str = ""
    ignore: IgnoreRules = Field(default_factory=IgnoreRules)
    thresholds: Thresholds


# --------------------------------------------------------------------------- #
# taxonomy.yaml (DESIGN §10)
#
# Modeled and validated here; not yet read by any tagger. The same staging
# posture `ArxivConfig` carried until `ingest/arxiv.py` shipped (NEVER rule
# 15) — `score/taxonomy.py`'s keyword tagger and its Haiku-triage fallback are
# a separate, larger unit of work with an open question this config alone
# does not answer (where a tagged item's topic(s) get persisted — DESIGN §5's
# schema has no `item_topics` table yet). `signalforge` does not read this
# file today; editing it has no runtime effect until that lands.
# --------------------------------------------------------------------------- #


_TAXONOMY_NAME_RE: Final = re.compile(r"[a-z0-9][a-z0-9-]*")
"""A taxonomy group or leaf name: lowercase, digits, hyphens, no leading hyphen.
Matches the `group.leaf` shape `priority_topics` writes (e.g. `industry.strategy`)
and the naming convention every other `sources.yaml`/`interests.yaml` key already
follows (`daily_max_per_source`-style snake_case aside — those are Python field
names, not user-composed identifiers like these)."""


class TaxonomyLeaf(_StrictModel):
    """One topic leaf: its match keywords for the deterministic first-pass tagger."""

    keywords: list[str] = Field(min_length=1)

    @field_validator("keywords")
    @classmethod
    def _no_blank_keywords(cls, value: list[str]) -> list[str]:
        for keyword in value:
            if not keyword.strip():
                raise ValueError("a taxonomy keyword cannot be blank")
        return value


class TaxonomyConfig(RootModel[dict[str, dict[str, TaxonomyLeaf]]]):
    """Root model for `taxonomy.yaml` — a two-level topic tree (DESIGN §10).

    Top-level keys are group names (`agents`, `frontier`, ...); each maps to a
    dict of leaf names to their match keywords. A `RootModel` over a plain dict
    rather than named attributes, because — unlike `SourcesConfig`'s fixed
    blocks (`rss`, `github`, ...) — groups and leaves here are themselves the
    config; there is no closed vocabulary to name as Python attributes.

    Access the tree via `.root["agents"]["planning"].keywords` — NOT by
    iterating the model directly. `RootModel` is still a `BaseModel`
    underneath, so `list(config)`/`for k in config` yields its *pydantic
    fields* (`[("root", {...})]`), not the group names, and `config["agents"]`
    raises `TypeError` rather than delegating to the dict. Both are easy
    mistakes for whatever consumes this next (the future `score/taxonomy.py`
    tagger) to make once, the first time it tries to loop over topics.
    """

    @field_validator("root")
    @classmethod
    def _no_empty_groups(
        cls, value: dict[str, dict[str, TaxonomyLeaf]]
    ) -> dict[str, dict[str, TaxonomyLeaf]]:
        for group, leaves in value.items():
            if not leaves:
                raise ValueError(f"taxonomy group {group!r} has no leaf topics")
        return value

    @field_validator("root")
    @classmethod
    def _valid_names(
        cls, value: dict[str, dict[str, TaxonomyLeaf]]
    ) -> dict[str, dict[str, TaxonomyLeaf]]:
        """Group and leaf names must be lowercase `[a-z0-9-]`, starting alphanumeric.

        `interests.yaml`'s `priority_topics` already writes topics as
        `group.leaf` (`industry.strategy`), and `test_config.py` checks the two
        files against each other by building that same string — see this
        file's own comment. A dot inside a name would make that join ambiguous;
        stray case or whitespace (` Agents `/`Agents`) would pass this model
        silently but never match a `priority_topics` entry again, the exact
        failure this rejects instead of a phantom, un-matchable taxonomy key.
        `str_strip_whitespace` doesn't help here — it only trims field
        *values*, never dict *keys*, and `RootModel` skips `_StrictModel`
        entirely (there is no fixed field to strip).
        """
        for group, leaves in value.items():
            if not _TAXONOMY_NAME_RE.fullmatch(group):
                raise ValueError(
                    f"taxonomy group name {group!r} must match {_TAXONOMY_NAME_RE.pattern!r}"
                )
            for leaf in leaves:
                if not _TAXONOMY_NAME_RE.fullmatch(leaf):
                    raise ValueError(
                        f"taxonomy leaf name {leaf!r} in group {group!r} "
                        f"must match {_TAXONOMY_NAME_RE.pattern!r}"
                    )
        return value


# --------------------------------------------------------------------------- #
# settings.yaml — app & locale (not relevance, not sources)
# --------------------------------------------------------------------------- #


_RESEND_KEY_PREFIX: Final = "re_"
"""The prefix a real Resend API key carries. Matched against `api_key_env` to catch a
pasted secret, the same guard `GithubConfig.token_env` applies to a PAT."""

_RESEND_KEY_MIN_LENGTH: Final = 20
"""Length below which a `re_`-prefixed string is too short to be a real Resend key.
Real ones run ~35 characters; this only has to separate them from an env-var name."""


class EmailChannelConfig(_StrictModel):
    """The `delivery.email:` block — a read-only mirror of a rendered report.

    Email is an *outbound* channel and nothing more. The vault stays canonical
    (DESIGN §13.2): marks and curation ticks are harvested from `<vault>/daily/*.md`
    only, so a digest that was emailed but not written is not a digest that
    happened. Nothing here can make the vault write conditional.
    """

    enabled: bool
    """Required rather than defaulted. Writing this block is a deliberate act, and an
    off-switch you have to delete the block to reach is not an off-switch. A missing
    `enabled` is the invisible-misconfig failure `_StrictModel` exists to catch."""

    api_key_env: str = Field(min_length=1)
    """Name of the env var holding the provider API key — never the key itself
    (CLAUDE.md §10 rule 16)."""

    from_address: str = Field(min_length=1)
    """Sender, bare (`digest@example.com`) or with a display name
    (`SignalForge <digest@example.com>`)."""

    to: list[str] = Field(min_length=1)
    """Recipients. Single-user tool, but a list costs nothing and reads honestly."""

    # Deliberately absent: a `provider:` selector and a `send_on:` list.
    #
    # `deliver/email.py` posts to one hardcoded endpoint, so a `provider` field
    # could not change behaviour — a config option nothing reads is worse than no
    # option, because it looks like a choice. `daily` is likewise the only report
    # that renders, so `send_on` would be a knob with one legal value that could
    # also be set to `[]` for an `enabled: true` channel that sends nothing.
    #
    # Both get added with the thing that needs them — a second provider, the Weekly
    # Brief — when their shape is known rather than guessed (CLAUDE.md §1: when in
    # doubt, build less).

    @field_validator("api_key_env")
    @classmethod
    def _reject_inline_secret(cls, value: str) -> str:
        """`api_key_env` must name an env var, not carry the key.

        Same guard, same reasoning as `GithubConfig.token_env`: the likely mistake is
        pasting a live `re_...` key into YAML.

        Matched case-sensitively and with a length floor, unlike the GitHub check.
        `ghp_`/`github_pat_` are distinctive enough to match case-insensitively; `re_`
        is three characters and collides with legitimate names — `RE_DIGEST_KEY`
        lowercases to something starting `re_`. Requiring the literal lowercase prefix
        *and* a body too long to be a variable name keeps that name usable while a
        pasted key, which is lowercase-prefixed and ~35 characters, still fails.
        """
        looks_like_a_key = (
            value.startswith(_RESEND_KEY_PREFIX) and len(value) >= _RESEND_KEY_MIN_LENGTH
        )
        if not value.replace("_", "").isalnum() or looks_like_a_key:
            raise ValueError(
                "api_key_env must be the NAME of an environment variable "
                "(e.g. RESEND_API_KEY), never a key value"
            )
        return value

    @field_validator("from_address", "to")
    @classmethod
    def _validate_addresses(cls, value: str | list[str]) -> str | list[str]:
        """Reject anything that is not a parseable address, and any control character.

        The control-character check is the load-bearing half. These strings are
        attacker-adjacent only in theory for a local config file, but they end up in
        message headers, where a stray `\\r\\n` is header injection. Identity fields
        are *refused* rather than repaired (DESIGN §13.1) — silently rewriting an
        address changes who the mail goes to.

        Refusal is why `parseaddr`'s output is checked for *containment* rather than
        just parsed. `parseaddr` repairs as it reads: it strips interior whitespace,
        so `a@e.com x` parses to `a@e.comx` and `a @e.com` to `a@e.com`. Validating
        the repair and then storing the original would approve a domain the operator
        never wrote and hand the provider the broken string anyway. Requiring the
        parsed address to appear verbatim in the input closes that gap while still
        accepting a display-name form, where the address is a substring.
        """
        for address in [value] if isinstance(value, str) else value:
            if has_control_characters(address):
                raise ValueError(f"address contains a control character: {address!r}")
            _, parsed = parseaddr(address)
            local, _, domain = parsed.partition("@")
            labels = domain.split(".")
            if (
                not local
                or len(labels) < 2
                or not all(labels)
                or parsed not in address  # see the docstring: refuse, never repair
            ):
                raise ValueError(
                    f"expected an email address, optionally with a display name, got {address!r}"
                )
        return value


PODCAST_PRESENTER_NAME_MAX_LENGTH: Final = 100
"""Shared bound for `PodcastChannelConfig.presenter_a`/`presenter_b` —
named rather than an inline `Field(max_length=100)` literal so
`llm.py`'s podcast worst-case cost test can price against the same number
this model enforces, instead of a smaller hardcoded stand-in drifting from
it (the exact "cap enforced in one place, priced in another" pattern an
`llm-cost-guard` review has now caught four separate times on this
feature)."""


class PodcastChannelConfig(_StrictModel):
    """The `delivery.podcast:` block — the second recorded phase-gate exception
    (DESIGN §13.3): a daily two-presenter audio show scripted by `synth/podcast.py`
    and published as a private RSS feed.

    Unlike email, this channel spends real money beyond the LLM call — TTS dollars,
    recorded to `runs.tts_characters` — so every knob here is required rather than
    defaulted (CLAUDE.md §10 rule 6): a default buried in code is a spend limit
    nobody can see.
    """

    enabled: bool
    """Required rather than defaulted, same reasoning as `EmailChannelConfig.enabled`."""

    tts_api_key_env: str = Field(min_length=1)
    """Name of the env var holding the OpenRouter API key — never the key itself."""

    tts_model: str = Field(min_length=1)
    """OpenRouter TTS model id (e.g. a Gemini Flash TTS or Kokoro-82M variant)."""

    voice_a: str = Field(min_length=1)
    voice_b: str = Field(min_length=1)
    """Provider voice ids for the two presenters."""

    presenter_a: str = Field(min_length=1, max_length=PODCAST_PRESENTER_NAME_MAX_LENGTH)
    presenter_b: str = Field(min_length=1, max_length=PODCAST_PRESENTER_NAME_MAX_LENGTH)
    """Display names rendered as dialogue speaker labels in the vault script.
    Bounded like every other string on the podcast prompt path (`llm.py`'s
    `PODCAST_MAX_ITEM_*_BYTES` family) — these two also reach
    `run_podcast_script`'s prompt and its cost worst-case."""

    r2_endpoint: str = Field(min_length=1)
    r2_bucket: str = Field(min_length=1)
    """Cloudflare R2 S3-compatible endpoint and bucket the feed and audio publish to."""

    r2_access_key_env: str = Field(min_length=1)
    r2_secret_key_env: str = Field(min_length=1)
    """Names of the env vars holding the R2 access/secret keys — never the keys
    themselves (CLAUDE.md §10 rule 16)."""

    public_base_url: str = Field(min_length=1)
    """Base URL the feed and audio are served from — the unguessable-prefix URL a
    podcast app subscribes to (DESIGN §13.3). `deliver/podcast.py` derives the
    real R2 object key prefix from this URL's path (`_r2_prefix`), so a
    malformed value here silently breaks the bucket layout rather than raising
    — see `_validate_public_base_url` for why the shape is checked at load."""

    retention_episodes: int = Field(ge=1, le=90)
    """How many episodes stay live in the feed and on R2; older ones are pruned."""

    feed_title: str = Field(min_length=1)
    """The RSS `<title>` shown in podcast apps."""

    @field_validator("tts_api_key_env")
    @classmethod
    def _reject_inline_secret(cls, value: str) -> str:
        """`tts_api_key_env` must name an env var, not carry the key.

        The likely mistake is pasting a live `sk-or-...` key into YAML. Unlike
        `EmailChannelConfig.api_key_env`'s `re_` guard, no length/prefix check is
        needed here: a real OpenRouter key contains hyphens (`sk-or-v1-...`), so
        the identifier-shape check alone already rejects it — the same guard
        `GithubConfig.token_env` uses for its non-prefixed fields.
        """
        if not value.replace("_", "").isalnum():
            raise ValueError(
                "tts_api_key_env must be the NAME of an environment variable "
                "(e.g. OPENROUTER_API_KEY), never a key value"
            )
        return value

    @field_validator("r2_access_key_env", "r2_secret_key_env")
    @classmethod
    def _reject_inline_r2_secret(cls, value: str) -> str:
        """These must name env vars, not carry the keys. The identifier-shape
        check alone (`tts_api_key_env`'s guard) is not enough here: unlike an
        OpenRouter key, an R2 access key ID is plain alnum with no
        distinctive prefix or punctuation, so it would sail through
        unchanged. Adding "the value must be SHOUTING_SNAKE_CASE" closes
        that: a real R2 access key ID (lowercase hex) fails on case, and a
        real R2 secret (base64-shaped — `+`, `/`, `=`) fails the identifier
        check outright. Not foolproof — this is a guard against the likely
        mistake, the same posture every sibling validator here takes, not a
        secret-detection engine.
        """
        if not value.replace("_", "").isalnum() or value != value.upper():
            raise ValueError(
                "must be the NAME of an environment variable "
                "(e.g. R2_ACCESS_KEY_ID), in SHOUTING_SNAKE_CASE, never a key value"
            )
        return value

    @field_validator("public_base_url")
    @classmethod
    def _validate_public_base_url(cls, value: str) -> str:
        """Must be a real `http(s)://host/...` URL with no query or fragment.

        `deliver.podcast._r2_prefix` derives the actual R2 object key prefix
        from this URL's *path* alone (`urlsplit(value).path`) — a value with
        no scheme puts the whole string in `.path` instead (objects land
        under a nonsense prefix built from the hostname), and a `?query` or
        `#fragment` would silently be ignored by every URL this module
        builds rather than flagged as the typo it almost certainly is. Both
        failure modes are invisible until an operator's phone 404s on every
        episode — caught here, at load, instead (the same
        invisible-misconfiguration posture `_StrictModel` exists for).
        """
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(
                f"public_base_url must be a full http(s):// URL with a host, got {value!r}"
            )
        if parts.query or parts.fragment:
            raise ValueError(
                f"public_base_url must not carry a ?query or #fragment, got {value!r} "
                "— every object key this channel writes is derived from the path alone"
            )
        return value


class DeliveryConfig(_StrictModel):
    """The `delivery:` block — where a rendered report is mirrored to, beyond the vault.

    Each channel is a new field here and a new module in `deliver/`, not a plugin
    registry: at this scale the explicit list is shorter than the machinery that
    would avoid it.
    """

    email: EmailChannelConfig | None = None
    podcast: PodcastChannelConfig | None = None


class VaultGitConfig(_StrictModel):
    """The `vault_git:` block — auto-commit of written reports (DESIGN §14, §16).

    Only a toggle, deliberately. *Which* paths get staged is a safety property
    rather than a preference (`report.vaultgit.COMMITTABLE_SUBDIRS`), and the
    commit is refused outright unless `vault_dir` is its own repository root —
    so there is no knob here that can widen what gets committed.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Commit each written report into the vault's own git history. Safe to "
            "leave on: when the vault is not a git repository — or is nested inside "
            "a wider one — the commit no-ops with a logged reason rather than "
            "staging anything. Nothing is ever pushed; a remote is a manual step."
        ),
    )


class SettingsConfig(_StrictModel):
    """Root model for `settings.yaml` — machine-local app and locale settings.

    Deliberately separate from `interests.yaml` (relevance) and `sources.yaml`
    (feeds): a timezone is neither what you care about nor where it comes from,
    it is who and where the operator is. This is the file that makes the tool
    portable — the pipeline stores and reasons in UTC everywhere, and *only*
    the reader-facing day boundary is resolved through this zone.
    """

    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone name (e.g. 'Australia/Sydney', 'America/New_York', 'UTC') "
            "used to resolve the reader's calendar day: which date a digest is 'for' "
            "and which items fall on it. Storage stays UTC — this is presentation only. "
            "Defaults to UTC so an operator who never sets it gets correct, if not "
            "local, behaviour."
        ),
    )

    vault_dir: Path = Field(
        default=Path("vault"),
        description=(
            "Directory the rendered reports are written to (the daily digest lands "
            "in `<vault_dir>/daily/`). `~` and `$VARS` are expanded, so a WSL "
            "operator can point straight at a Windows-side Obsidian vault, e.g. "
            "`/mnt/c/Users/<you>/Obsidian/SignalForge`. A relative path resolves "
            "against the process working directory (for cron that is the repo, via "
            "the `cd` in the crontab entry). Defaults to `vault/` inside the repo — "
            "the historical location — and the `--vault-dir` flag overrides it."
        ),
    )

    delivery: DeliveryConfig = Field(default_factory=DeliveryConfig)
    """Outbound mirrors of a rendered report (DESIGN §13.2's push channel). Absent means
    the vault is the only destination, which is the historical behaviour and stays the
    default: delivery is additive, never a substitute for the vault write."""

    taxonomy_stale_days: int = Field(
        default=60,
        ge=1,
        description=(
            "How long a `taxonomy.yaml` leaf may go without matching anything before "
            "`signalforge status` names it (DESIGN §10). A leaf nobody's corpus hits is "
            "either a dead interest or a badly chosen keyword — either way the fix is an "
            "operator edit, so this only reports. 60 days is long enough that a quiet "
            "fortnight is not an alarm."
        ),
    )

    vault_git: VaultGitConfig = Field(default_factory=VaultGitConfig)
    """Local git history for the vault. Defaults to on because the guarded no-op
    is harmless — an operator whose vault is not a repo simply never gets a commit."""

    @field_validator("vault_dir", mode="before")
    @classmethod
    def _expand_vault_dir(cls, value: object) -> object:
        """Expand `~` and `$VARS` before the value becomes a `Path`.

        Runs `before` validation so a committed `settings.yaml.example` can carry
        a portable `~/Obsidian/...` path and a real, gitignored `settings.yaml`
        can carry a machine-specific `/mnt/c/...` one — neither leaking an
        absolute home into the repo. Non-string/Path inputs pass through
        untouched for pydantic to reject with its normal type error.

        A `$VAR` that resolves to nothing is left verbatim by `expandvars`, which
        would silently yield a nonsense path and file digests somewhere
        surprising — the invisible-misconfig failure `_StrictModel` exists to
        catch (same reasoning as the timezone validator). So a residual `$` after
        expansion is rejected at load rather than deferred to the first digest.
        """
        if not isinstance(value, str | Path):
            return value
        expanded = os.path.expanduser(os.path.expandvars(os.fspath(value)))
        if "$" in expanded:
            raise ValueError(
                f"vault_dir contains an unexpanded environment variable: {expanded!r}. "
                "The referenced variable is unset — export it or use a literal path."
            )
        return expanded

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        """Reject a name `zoneinfo` cannot resolve, at load time rather than at
        the first digest. A typo'd zone silently falling back to UTC is exactly
        the invisible-misconfiguration failure `_StrictModel` exists to prevent.
        """
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"unknown IANA timezone {value!r}: {exc}. "
                "Use a name from the tz database, e.g. 'Australia/Sydney' or 'UTC'."
            ) from exc
        return value

    @property
    def tzinfo(self) -> ZoneInfo:
        """The validated zone as a `ZoneInfo`. Safe to construct — the validator
        already proved it resolves."""
        return ZoneInfo(self.timezone)


# --------------------------------------------------------------------------- #
# Secrets — environment only, never YAML
# --------------------------------------------------------------------------- #


def get_secret(env_var: str) -> SecretStr | None:
    """Read an arbitrary secret named by config (e.g. `github.token_env`).

    Checks a real environment variable first — so a value exported by cron,
    systemd, or the shell always wins — and falls back to `.env` in the
    current working directory. Reading `.env` fresh on every call (rather than
    loading it once into `os.environ`) means nothing here mutates global
    process state, which matters for tests: importing this module in a test
    run must not leak a developer's real `.env` secrets into `os.environ` for
    every subsequent test to see.

    Returns None when unset, so callers decide whether the credential is
    optional (GitHub works unauthenticated at 60 req/hr) or fatal. The value is
    wrapped in `SecretStr` and never logged — only the *name* of the missing
    variable is.
    """
    raw = os.environ.get(env_var) or dotenv_values(".env").get(env_var)
    stripped = raw.strip() if raw is not None else ""
    if not stripped:
        logger.debug("secret not set in environment", extra={"env_var": env_var})
        return None
    return SecretStr(stripped)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: could not be read: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path}: file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")
    # Keys from yaml.safe_load are arbitrary scalars; config keys must be strings.
    return {str(key): value for key, value in raw.items()}


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"{path}: {exc.error_count()} config error(s):"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def load_sources(config_dir: Path) -> SourcesConfig:
    """Load and validate `<config_dir>/sources.yaml`.

    Raises `ConfigError` with a per-field explanation on invalid config.
    """
    path = config_dir / SOURCES_FILENAME
    data = _load_yaml_mapping(path)
    try:
        config = SourcesConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    logger.debug(
        "loaded sources config",
        extra={"path": str(path), "rss_count": len(config.rss)},
    )
    return config


def load_interests(config_dir: Path) -> InterestsConfig:
    """Load and validate `<config_dir>/interests.yaml`.

    Raises `ConfigError` with a per-field explanation on invalid config.
    """
    path = config_dir / INTERESTS_FILENAME
    data = _load_yaml_mapping(path)
    try:
        config = InterestsConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    logger.debug("loaded interests config", extra={"path": str(path)})
    return config


def load_taxonomy(config_dir: Path) -> TaxonomyConfig:
    """Load and validate `<config_dir>/taxonomy.yaml`.

    Raises `ConfigError` with a per-field explanation on invalid config. Not
    yet called by any runtime path (see the module comment above
    `TaxonomyConfig`) — unlike `load_sources`/`load_interests`, nothing in
    `cli.py` calls this on every run, so a typo here is only caught by the
    test suite today, not at the command line. This loader exists so
    `tests/test_config.py` can validate the shipped file ahead of the tagger
    that will actually call it.
    """
    path = config_dir / TAXONOMY_FILENAME
    data = _load_yaml_mapping(path)
    try:
        config = TaxonomyConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    logger.debug(
        "loaded taxonomy config",
        extra={"path": str(path), "group_count": len(config.root)},
    )
    return config


def load_settings(config_dir: Path) -> SettingsConfig:
    """Load and validate `<config_dir>/settings.yaml`, or default to UTC.

    Unlike `sources.yaml`/`interests.yaml`, a *missing* settings file is not an
    error: every field has a safe default (UTC), so an operator who never
    creates one — or an existing install predating this file — gets correct
    behaviour rather than a crash. A file that is *present* is still validated
    strictly (unknown keys and bad zones raise `ConfigError`); only total
    absence is tolerated.
    """
    path = config_dir / SETTINGS_FILENAME
    if not path.is_file():
        logger.debug("no settings.yaml; using defaults", extra={"path": str(path)})
        return SettingsConfig()
    data = _load_yaml_mapping(path)
    try:
        config = SettingsConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    logger.debug("loaded settings config", extra={"path": str(path), "timezone": config.timezone})
    return config
