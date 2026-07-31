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
from email.utils import parseaddr
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from signalforge.models import has_control_characters

__all__ = [
    "SETTINGS_FILENAME",
    "SOURCES_FILENAME",
    "ArxivConfig",
    "ConfigError",
    "CurationConfig",
    "DeliveryConfig",
    "EmailChannelConfig",
    "GithubConfig",
    "HackerNewsConfig",
    "IgnoreRules",
    "InterestsConfig",
    "RssSource",
    "SettingsConfig",
    "SourceDefaults",
    "SourcesConfig",
    "Thresholds",
    "get_secret",
    "load_interests",
    "load_settings",
    "load_sources",
]

logger = logging.getLogger(__name__)

SOURCES_FILENAME: Final = "sources.yaml"
INTERESTS_FILENAME: Final = "interests.yaml"
SETTINGS_FILENAME: Final = "settings.yaml"


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
    """`owner/repo` slugs diffed between runs (Phase 1)."""

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
    """The `arxiv:` block (Phase 1 — modeled here, ingested later)."""

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


class DeliveryConfig(_StrictModel):
    """The `delivery:` block — where a rendered report is mirrored to, beyond the vault.

    One channel today. A second (Telegram, per DESIGN §13.2) is a new field here and a
    new module in `deliver/`, not a plugin registry: at this scale the explicit list
    is shorter than the machinery that would avoid it.
    """

    email: EmailChannelConfig | None = None


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
