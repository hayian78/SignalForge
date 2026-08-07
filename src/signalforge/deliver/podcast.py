"""The podcast channel — synthesize, upload, rebuild the feed, prune (DESIGN §13.3).

Own entry point, `deliver_podcast`, deliberately not folded into
`deliver_digest`'s channel loop (`deliver/__init__.py`): the two channels
mirror different vault files (`vault/daily/*.md` vs `vault/podcast/*.md`),
spend different things (nothing vs. TTS characters), and a podcast failure
must never be reported as if it were a digest-email failure.

Reads the *already-written* vault script (`report.podcast.parse_script`),
never `synth.podcast.build_script` — this module never calls an LLM
(CLAUDE.md §2), and the same "audio must never disagree with the file"
contract `report/podcast.py`'s own docstring describes. Guard chain mirrors
`deliver/__init__.py::_deliver_email_channel`'s order exactly: freshness
window, then the send log (resend overrules the log, never the window),
then secrets, then the actual work.

### Idempotency levers (CLAUDE.md §3, NEVER rule 4)

* **Audio**: a local mp3 already at `audio_path(...)` skips synthesis
  entirely — no repaying OpenRouter for a re-run. A failed upload after a
  successful synthesis costs nothing extra on retry.
* **Upload**: `put_object` overwrites a key rather than versioning it.
* **Feed**: rebuilt from `vault/podcast/*.md` ∩ local `data/audio/*.mp3`
  (`_feed_dates`) on every delivery, not accumulated — a re-run produces the
  same feed given the same files on disk.
* **Prune**: the *keep* set (`_prune_keep_dates`) is vault-only, deliberately
  not intersected with local audio — `data/audio/` is not backed up, so a
  wiped local cache must never read as "these episodes don't exist anymore"
  and cause a mass delete of real, already-paid-for R2 objects. What
  actually gets deleted is still driven by R2's own `list_object_keys`, not
  a trust that local disk mirrors it. A second prune of an already-pruned
  set deletes nothing (every delete is a no-op past the first).
* **Delivery record**: the same `deliveries` unique index / `record_delivery`
  contract `deliver/__init__.py` already documents, written only after both
  uploads succeed — a prune failure does not block it (see `deliver_podcast`).
* **Retention refusal**: before any synthesis (paid or cached), `deliver_podcast`
  refuses a `target_date` that would immediately fall outside
  `retention_episodes` — otherwise a backfill for an old date could be
  synthesized, uploaded, and pruned by this same run, spending real money to
  publish an episode that never survives the function call.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from datetime import date as Date
from email.utils import format_datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
import jinja2
from pydantic import SecretStr

from signalforge.config import PodcastChannelConfig, SettingsConfig, get_secret
from signalforge.db import delivery_exists, record_delivery
from signalforge.deliver import MAX_DELIVERY_AGE_DAYS, is_within_delivery_window
from signalforge.deliver.storage import StorageError, delete_object, list_object_keys, put_object
from signalforge.deliver.tts import SynthesisError, synthesize_episode
from signalforge.report.podcast import parse_script, script_path

__all__ = [
    "CHANNEL_NAME",
    "REPORT_KIND_PODCAST",
    "PodcastDeliveryOutcome",
    "audio_path",
    "deliver_podcast",
]

logger = logging.getLogger(__name__)

CHANNEL_NAME: Final = "podcast"
"""The `deliveries.channel` value — a constant, not an inline literal, for
the same indexing reason `deliver.email.CHANNEL_NAME` gives."""

REPORT_KIND_PODCAST: Final = "podcast"

_EPISODE_KEY_PREFIX: Final = "signalforge-"
_FEED_KEY: Final = "feed.xml"
_FEED_TEMPLATE: Final = "podcast_feed.xml.j2"
_TEMPLATES_DIR: Final = "templates"

_R2_TIMEOUT_SECONDS: Final = 30.0

_ISO_DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class PodcastDeliveryOutcome:
    """What one `deliver_podcast` attempt did, and why — the podcast
    analogue of `deliver.DeliveryOutcome`, kept as its own type rather than
    reusing that one: unlike email, this channel spends TTS characters, and
    bolting that field onto the email-shaped outcome would make the field
    meaningless for every other channel that will ever use it.
    """

    channel: str
    sent: bool
    reason: str
    provider_id: str | None = None
    error: Exception | None = None
    tts_characters: int = 0
    """Characters sent to the TTS provider *this attempt* — 0 when
    synthesis was skipped (mp3 already cached locally) or never reached.
    The caller (Stage 6's CLI) passes this straight into
    `db.finish_run(tts_characters=...)` (NEVER rule 11, applied to TTS
    spend instead of LLM tokens)."""


def audio_path(audio_dir: Path, *, date: Date) -> Path:
    """Where `date`'s local mp3 lives — same basename as `_episode_key`
    (delegated, not duplicated, so the two cannot drift), so local ↔ remote
    correspondence during retention never needs a second mapping."""
    return audio_dir / _episode_key(date)


def _episode_key(date: Date) -> str:
    """The basename both the local file and the R2 object are named after —
    never the full R2 *key*, which also needs `_r2_prefix` prepended (see
    `_remote_key`). Local files don't carry that prefix: it exists only to
    match the path segment of `public_base_url`, which the local filesystem
    has no reason to mirror."""
    return f"{_EPISODE_KEY_PREFIX}{date.isoformat()}.mp3"


def _r2_prefix(config: PodcastChannelConfig) -> str:
    """The path segment of `public_base_url`, normalized to a trailing-slash
    prefix (`""` if the feed is served from the bucket root).

    Every object this module writes must live under this prefix. R2's public
    dev-domain serving (and any custom domain proxying it) maps a URL path
    directly onto an object key with no rewriting, so `public_base_url`
    carrying an "unguessable prefix" (DESIGN §13.3 — the feed's only access
    control) is a promise about where objects actually live, not just how
    they're advertised. Writing to the bucket root while advertising a
    prefixed URL means every enclosure in the feed 404s — a code-reviewer
    pass caught this by actually replaying a delivery against mocked R2 and
    checking the resulting audio URL against the uploaded key.
    """
    path = urlsplit(config.public_base_url).path.strip("/")
    return f"{path}/" if path else ""


def _remote_key(config: PodcastChannelConfig, basename: str) -> str:
    """The full R2 object key for `basename` (an episode's `_episode_key`,
    or `_FEED_KEY`) — `_r2_prefix` plus the basename, the only place those
    two are joined."""
    return f"{_r2_prefix(config)}{basename}"


def _date_from_filename(name: str, *, prefix: str, suffix: str) -> Date | None:
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    stamp = name[len(prefix) : -len(suffix)] if suffix else name[len(prefix) :]
    if not _ISO_DATE_RE.match(stamp):
        return None
    try:
        return Date.fromisoformat(stamp)
    except ValueError:
        return None


def _vault_dates(vault_dir: Path) -> set[Date]:
    script_dir = vault_dir / "podcast"
    return {
        date
        for path in script_dir.glob("*.md")
        if (date := _date_from_filename(path.name, prefix="", suffix=".md")) is not None
    }


def _local_audio_dates(audio_dir: Path) -> set[Date]:
    return {
        date
        for path in audio_dir.glob(f"{_EPISODE_KEY_PREFIX}*.mp3")
        if (date := _date_from_filename(path.name, prefix=_EPISODE_KEY_PREFIX, suffix=".mp3"))
        is not None
    }


def _prune_keep_dates(*, vault_dir: Path, retention_episodes: int) -> list[Date]:
    """Dates that must survive pruning — vault scripts only, *not*
    intersected with local audio presence.

    `data/audio/` is not backed up (unlike the vault and the DB — DESIGN
    §14), so a wiped or freshly-cloned local cache must never be read as
    "these episodes don't exist anymore": the vault script is still proof
    an episode was published, and the R2 object behind it is real money
    already spent. A code-reviewer pass caught the earlier vault-∩-audio
    version of this set silently deleting an operator's entire published
    back catalogue off R2 the first time `data/audio/` went missing. See
    `_feed_dates` for the separate, stricter set actually rendered into the
    feed, which does need local audio (for the enclosure's byte length).
    """
    return sorted(_vault_dates(vault_dir), reverse=True)[:retention_episodes]


def _feed_dates(*, keep_dates: Sequence[Date], audio_dir: Path) -> list[Date]:
    """The dates that belong in the feed — always a *subset* of
    `keep_dates` (the caller's already-computed `_prune_keep_dates`), never
    an independent computation.

    Takes `keep_dates` as a parameter rather than recomputing
    `_prune_keep_dates` itself, and filters it down to dates with local
    audio (the enclosure's byte length needs the local file to exist),
    rather than intersecting vault ∩ audio and capping at
    `retention_episodes` independently. The two ways of computing this are
    not the same set: capping first and filtering second (this version) can
    only ever shrink what the keep set already allows, while capping an
    independently-computed intersection can select a *different* top-N —
    e.g. several recent scripts with no local audio yet (a bad synthesis
    day) push older, audio-backed dates out of the feed's own cap while the
    real keep set still protects them, so the feed would advertise dates
    `_prune` had *already deleted* in the same call. A code-reviewer pass
    reproduced that end to end (five just-deleted enclosures still listed
    as `<item>`s) against an earlier version of this function that computed
    its own intersection instead of filtering the shared keep set.
    """
    local = _local_audio_dates(audio_dir)
    return [date for date in keep_dates if date in local]


@dataclass(frozen=True, slots=True)
class _FeedEpisode:
    title: str
    description: str
    pub_date_rfc822: str
    guid: str
    audio_url: str
    audio_bytes: int


def _feed_episode(
    *, date: Date, config: PodcastChannelConfig, vault_dir: Path, audio_dir: Path, tz: ZoneInfo
) -> _FeedEpisode | None:
    """One retained date's feed entry, or `None` if its files turn out not
    to be readable right now — a defensive re-check, since `_feed_dates`
    only proved existence a moment earlier and nothing guarantees the
    filesystem hasn't changed since (CLAUDE.md §7: one bad historical file
    must not break today's delivery)."""
    audio_file = audio_path(audio_dir, date=date)
    try:
        audio_bytes = audio_file.stat().st_size
    except OSError:
        logger.warning(
            "podcast feed rebuild: retained audio file is unreadable; skipping",
            extra={"date": date.isoformat()},
        )
        return None

    try:
        context = parse_script(script_path(vault_dir, date=date).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning(
            "podcast feed rebuild: retained script is unreadable or malformed; skipping",
            extra={"date": date.isoformat()},
        )
        return None

    pub_date = datetime.combine(date, time(hour=6), tzinfo=tz)
    return _FeedEpisode(
        title=f"{config.feed_title} — {date.isoformat()}",
        description=f"{context.item_count} stories from SignalForge's daily digest.",
        pub_date_rfc822=format_datetime(pub_date),
        guid=date.isoformat(),
        audio_url=f"{config.public_base_url.rstrip('/')}/{_episode_key(date)}",
        audio_bytes=audio_bytes,
    )


@dataclass(frozen=True, slots=True)
class _Feed:
    title: str
    public_base_url: str
    episodes: tuple[_FeedEpisode, ...]


def _template_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.PackageLoader("signalforge.deliver", _TEMPLATES_DIR),
        # ON, unconditionally — unlike `deliver.email`'s per-template-name
        # split, there is only one template here and it is always XML.
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_feed(feed: _Feed) -> str:
    """Fill `podcast_feed.xml.j2` with `feed`. Pure function — no I/O beyond
    loading the template."""
    return _template_env().get_template(_FEED_TEMPLATE).render(feed=feed)


def _prune(
    client: httpx.Client,
    *,
    config: PodcastChannelConfig,
    retained_dates: set[Date],
    audio_dir: Path,
    access_key: SecretStr,
    secret_key: SecretStr,
    now: datetime,
) -> int:
    """Delete every remote and local episode file whose date fell out of
    `retained_dates` (see `_prune_keep_dates` — vault-only, so a missing
    local file is never itself a reason to delete anything). Returns the
    remote delete count, for the caller's log line. Remote state comes from
    `list_object_keys`, not a trust that local disk reflects reality — an
    object outside `retained_dates` is found and removed regardless of
    whether a local copy currently exists too.
    """
    r2_prefix = _r2_prefix(config)
    remote_keys = list_object_keys(
        client,
        endpoint=config.r2_endpoint,
        bucket=config.r2_bucket,
        prefix=f"{r2_prefix}{_EPISODE_KEY_PREFIX}",
        access_key=access_key,
        secret_key=secret_key,
        now=now,
    )
    deleted = 0
    for key in remote_keys:
        basename = key.removeprefix(r2_prefix)
        date = _date_from_filename(basename, prefix=_EPISODE_KEY_PREFIX, suffix=".mp3")
        if date is not None and date not in retained_dates:
            delete_object(
                client,
                endpoint=config.r2_endpoint,
                bucket=config.r2_bucket,
                key=key,
                access_key=access_key,
                secret_key=secret_key,
                now=now,
            )
            deleted += 1

    for local_file in audio_dir.glob(f"{_EPISODE_KEY_PREFIX}*.mp3"):
        date = _date_from_filename(local_file.name, prefix=_EPISODE_KEY_PREFIX, suffix=".mp3")
        if date is not None and date not in retained_dates:
            local_file.unlink(missing_ok=True)

    return deleted


def deliver_podcast(
    conn: sqlite3.Connection,
    *,
    settings: SettingsConfig,
    target_date: Date,
    vault_dir: Path,
    audio_dir: Path,
    today: Date,
    run_id: int | None = None,
    resend: bool = False,
) -> PodcastDeliveryOutcome | None:
    """Synthesize (if needed), upload, rebuild the feed, and prune —
    `None` when the channel is unconfigured or disabled, an outcome
    otherwise, every reason named (CLAUDE.md §7).

    `target_date`'s vault script must already exist (`report.podcast.write_script`,
    Stage 4) — this function is read-only with respect to the vault. Does
    not raise for a delivery failure: mirrors
    `deliver._deliver_email_channel`'s "never fails the run" contract, and
    for the identical reason — the script is already safely on disk.
    """
    config = settings.delivery.podcast
    if config is None or not config.enabled:
        return None

    if not is_within_delivery_window(target_date, today=today):
        reason = (
            f"episode for {target_date.isoformat()} is more than "
            f"{MAX_DELIVERY_AGE_DAYS} day(s) from today; not delivered"
        )
        logger.info(
            "skipped podcast delivery: report outside the freshness window",
            extra={"channel": CHANNEL_NAME, "run_id": run_id, "reason": reason},
        )
        return PodcastDeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=reason)

    already_sent = delivery_exists(
        conn, channel=CHANNEL_NAME, report_kind=REPORT_KIND_PODCAST, target_date=target_date
    )
    if already_sent and not resend:
        return PodcastDeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=False,
            reason=(
                f"already delivered for {target_date.isoformat()}; "
                "re-rendering does not re-send (use --resend to override)"
            ),
        )

    # Fetched once each (not re-derived), and reused below for the actual
    # calls — `get_secret` re-reads `.env` from disk every time it's
    # called, so this is also the cheaper shape, not just the tidier one.
    tts_key = get_secret(config.tts_api_key_env)
    r2_access_key = get_secret(config.r2_access_key_env)
    r2_secret_key = get_secret(config.r2_secret_key_env)
    missing = [
        name
        for name, value in (
            (config.tts_api_key_env, tts_key),
            (config.r2_access_key_env, r2_access_key),
            (config.r2_secret_key_env, r2_secret_key),
        )
        if value is None
    ]
    if missing:
        error = RuntimeError(
            "podcast delivery is enabled but "
            f"{', '.join(missing)} not set in the environment or .env"
        )
        return PodcastDeliveryOutcome(
            channel=CHANNEL_NAME, sent=False, reason=str(error), error=error
        )
    assert tts_key is not None  # noqa: S101 - just proved by `missing` above
    assert r2_access_key is not None  # noqa: S101
    assert r2_secret_key is not None  # noqa: S101

    script_file = script_path(vault_dir, date=target_date)
    if not script_file.is_file():
        reason = (
            f"no vault script for {target_date.isoformat()} at {script_file}; nothing to deliver"
        )
        return PodcastDeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=reason)
    script_text = script_file.read_text(encoding="utf-8")
    try:
        context = parse_script(script_text)
    except ValueError as exc:
        return PodcastDeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=False,
            reason=f"vault script could not be parsed: {exc}",
            error=exc,
        )
    if not context.segments:
        reason = (
            f"vault script for {target_date.isoformat()} has no episode content; nothing to deliver"
        )
        return PodcastDeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=reason)

    # Computed once, here, and reused everywhere below this point (the
    # retention pre-check, the feed rebuild, and the prune step) — vault
    # script presence cannot change during this function (only local audio
    # files get written), so a second computation would only risk two of
    # those three steps disagreeing about which dates survive.
    #
    # Checked before any synthesis (paid or cached) rather than after: with
    # a small `retention_episodes` and newer vault scripts already present,
    # a backfill for an older `target_date` could otherwise be synthesized,
    # uploaded, immediately pruned by this same run's own retention step,
    # and still recorded as delivered — spending real TTS money to publish
    # an episode that is deleted (both remotely and locally) before the
    # function returns, with `record_delivery` then suppressing any retry.
    # A code-reviewer pass demonstrated this end to end.
    keep_dates = _prune_keep_dates(
        vault_dir=vault_dir, retention_episodes=config.retention_episodes
    )
    if target_date not in keep_dates:
        reason = (
            f"{target_date.isoformat()} would fall outside retention_episodes="
            f"{config.retention_episodes} immediately; refusing to synthesize and then prune it"
        )
        return PodcastDeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=reason)

    audio_file = audio_path(audio_dir, date=target_date)
    tts_characters = 0
    if audio_file.is_file():
        logger.info(
            "podcast audio already cached locally; skipping synthesis",
            extra={"channel": CHANNEL_NAME, "run_id": run_id, "date": target_date.isoformat()},
        )
    else:
        turns = [turn for segment in context.segments for turn in segment.turns]
        try:
            result = synthesize_episode(turns, config=config, api_key=tts_key)
        except SynthesisError as exc:
            # `exc.characters_sent` carries whatever chunks already
            # succeeded before this failure (or every chunk, if only the
            # concat step failed) — NEVER rule 11 applied to TTS characters,
            # the same discipline `synth/podcast.py` gives Opus tokens.
            return PodcastDeliveryOutcome(
                channel=CHANNEL_NAME,
                sent=False,
                reason=f"synthesis failed: {exc}",
                error=exc,
                tts_characters=exc.characters_sent,
            )
        tts_characters = result.characters
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.write_bytes(result.audio)
        logger.info(
            "synthesized podcast audio",
            extra={
                "channel": CHANNEL_NAME,
                "run_id": run_id,
                "date": target_date.isoformat(),
                "characters": tts_characters,
            },
        )

    now = datetime.now(UTC)
    # `feed_dates` filters `keep_dates` (computed above) down to dates with
    # local audio — always a subset, never an independent computation (see
    # `_feed_dates`'s docstring for the bug that shape used to allow).
    feed_dates = _feed_dates(keep_dates=keep_dates, audio_dir=audio_dir)
    episodes: tuple[_FeedEpisode, ...] = ()
    try:
        with httpx.Client(timeout=_R2_TIMEOUT_SECONDS) as client:
            put_object(
                client,
                endpoint=config.r2_endpoint,
                bucket=config.r2_bucket,
                key=_remote_key(config, _episode_key(target_date)),
                body=audio_file.read_bytes(),
                content_type="audio/mpeg",
                access_key=r2_access_key,
                secret_key=r2_secret_key,
                now=now,
            )

            episodes = tuple(
                episode
                for date in feed_dates
                if (
                    episode := _feed_episode(
                        date=date,
                        config=config,
                        vault_dir=vault_dir,
                        audio_dir=audio_dir,
                        tz=settings.tzinfo,
                    )
                )
                is not None
            )
            feed_xml = render_feed(
                _Feed(
                    title=config.feed_title,
                    public_base_url=config.public_base_url,
                    episodes=episodes,
                )
            )
            put_object(
                client,
                endpoint=config.r2_endpoint,
                bucket=config.r2_bucket,
                key=_remote_key(config, _FEED_KEY),
                body=feed_xml.encode("utf-8"),
                content_type="application/rss+xml; charset=utf-8",
                access_key=r2_access_key,
                secret_key=r2_secret_key,
                now=now,
            )
    except StorageError as exc:
        return PodcastDeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=False,
            reason=f"r2 upload failed: {exc}",
            error=exc,
            tts_characters=tts_characters,
        )
    except Exception as exc:
        # Same posture as `deliver._deliver_email_channel`'s catch-all: a
        # channel failure is an outcome, not an exception crossing this line.
        logger.exception("podcast delivery raised unexpectedly", extra={"run_id": run_id})
        return PodcastDeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=False,
            reason=f"delivery failed: {exc}",
            error=exc,
            tts_characters=tts_characters,
        )

    pruned = 0
    try:
        with httpx.Client(timeout=_R2_TIMEOUT_SECONDS) as client:
            pruned = _prune(
                client,
                config=config,
                retained_dates=set(keep_dates),
                audio_dir=audio_dir,
                access_key=r2_access_key,
                secret_key=r2_secret_key,
                now=now,
            )
    except (StorageError, OSError) as exc:
        # Deliberately not fatal: the episode itself (mp3 + feed) is already
        # live by this point. A prune failure means an old episode lingers
        # one more day, not that today's delivery failed. `OSError` too
        # (e.g. `local_file.unlink()` racing a concurrent process), for the
        # same "does not raise for a delivery failure" contract this
        # function's own docstring promises — a code-reviewer pass caught
        # this handler only covering `StorageError`.
        logger.warning(
            "podcast retention prune failed; episode was still delivered",
            extra={"channel": CHANNEL_NAME, "run_id": run_id, "error": str(exc)},
        )

    remote_key = _remote_key(config, _episode_key(target_date))
    body_hash = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    recorded = record_delivery(
        conn,
        run_id=run_id,
        channel=CHANNEL_NAME,
        report_kind=REPORT_KIND_PODCAST,
        target_date=target_date,
        body_hash=body_hash,
        provider_id=remote_key,
        sent_at=now,
        expect_existing=resend,
    )
    if not recorded and not resend:
        error = RuntimeError(
            f"a delivery was already recorded for {target_date.isoformat()}; "
            "a duplicate upload occurred"
        )
        return PodcastDeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=True,
            reason=str(error),
            provider_id=remote_key,
            error=error,
            tts_characters=tts_characters,
        )

    return PodcastDeliveryOutcome(
        channel=CHANNEL_NAME,
        sent=True,
        reason=f"delivered; {len(episodes)} episode(s) in feed, {pruned} pruned remotely",
        provider_id=remote_key,
        tts_characters=tts_characters,
    )
