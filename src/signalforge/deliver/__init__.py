"""Outbound mirrors of a rendered report — DESIGN §13.2's push channel.

Module boundary (CLAUDE.md §2). `deliver/` is the one module that sends a report
somewhere other than the vault. It:

* takes an already-built `report.daily.DigestContext` and an already-rendered
  body — it never queries for items itself, so there is exactly one definition of
  what a digest contains;
* makes HTTP calls, which is why it is a sibling of `report/` and not part of it:
  `report/`'s charter is "only reads the DB and writes markdown", and the same
  reasoning keeps `feedback.py` outside it;
* never imports `llm.py`. Nothing here costs a token.

**The vault stays canonical.** DESIGN §13.2 fixes this and it is not negotiable: a
channel is a mirror of a report that has already been written to disk, never a
substitute for writing it. Two things follow, and both are load-bearing:

* Delivery runs *after* the vault write succeeds, and a channel failure is
  recorded to `runs.errors` rather than failing the run (§7). The digest is the
  product; the mirror is a side channel.
* Nothing here is an input surface. Marks and curation ticks are harvested from
  `<vault>/daily/*.md` only (DESIGN §13.1), so an emailed digest carries no
  checkboxes — it carries a count of the decisions waiting in the vault, which is
  the nudge that keeps the loop closed when the phone becomes the read surface.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as Date

from pydantic import SecretStr

from signalforge.config import EmailChannelConfig, SettingsConfig, get_secret
from signalforge.db import delivery_exists, record_delivery
from signalforge.deliver.email import CHANNEL_NAME, SendError, render_email, send_email
from signalforge.models import has_control_characters
from signalforge.report.daily import DigestContext, DigestLine

__all__ = [
    "MAX_DELIVERY_AGE_DAYS",
    "DeliveryOutcome",
    "deliver_digest",
    "send_test_message",
    "is_within_delivery_window",
]

logger = logging.getLogger(__name__)

REPORT_KIND_DAILY = "daily"
"""The `deliveries.report_kind` value for the Daily Digest. A constant for the same
reason as `email.CHANNEL_NAME`: it is half an index key."""

MAX_DELIVERY_AGE_DAYS = 1
"""How stale a report may be and still be sent.

A safety rail, not a tuning knob, which is why it lives in code rather than
`settings.yaml` (CLAUDE.md §4 governs knobs the operator tunes; this is a bound
nobody should want to widen).

It exists because the `deliveries` table cannot be the only guard. CLAUDE.md §3
calls the database regenerable plumbing and the vault the product — the operator
is explicitly permitted to delete `signalforge.db` and re-ingest. Every send
record goes with it, and a re-render of past dates would then mail weeks of old
digests. This rail is stateless, so it survives that.

It also closes a footgun that has nothing to do with the DB: `signalforge digest
--date 2026-07-01` is a legitimate thing to run when repairing an old file, and
it must not put month-old news in your inbox.

One day rather than zero because the 06:00 cron can start before midnight and
finish after it, and a digest is written for the operator's local calendar day.
"""


def is_within_delivery_window(target_date: Date, *, today: Date) -> bool:
    """Whether `target_date`'s report is fresh enough to send.

    The window is symmetric. A digest dated *just* ahead of today is a clock or
    timezone artifact, not stale news, and refusing it would be a silent failure on
    the one day it happened — but `--date 2027-01-01` is a typo, and mailing the
    empty digest it renders would also burn that date's send-log row, permanently
    suppressing the real send when it comes around.

    Two contracts the caller owes, both easy to get wrong:

    * **`today` is the operator's local calendar date** — `settings.yaml`'s zone,
      the same one that decides which date a digest is *for* — not
      `datetime.now(UTC).date()`. Storage is UTC everywhere; this is a
      reader-facing day boundary, and mixing the two makes the window off by one
      for most of the evening in Brisbane.
    * **A False must be surfaced, not silently obeyed** (§7). "No email today"
      with no explanation is indistinguishable from a broken pipeline, so the
      caller records the refusal — the reports and `signalforge status` are the
      monitoring channel.
    """
    return -MAX_DELIVERY_AGE_DAYS <= (today - target_date).days <= MAX_DELIVERY_AGE_DAYS


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    """What one channel did, and why — never a bare bool.

    Every non-send has a reason the operator can read. "No email today" with no
    explanation is indistinguishable from a broken pipeline, and the reports are
    the monitoring channel (§7).
    """

    channel: str
    sent: bool
    reason: str
    """Human-readable, always populated: why it sent, or why it did not."""

    provider_id: str | None = None
    error: Exception | None = None
    """Set only when the send was attempted and failed. The caller records this
    into `runs.errors`; it never fails the run — the vault file is the product and
    it is already written by the time this runs."""


def deliver_digest(
    conn: sqlite3.Connection,
    context: DigestContext,
    *,
    settings: SettingsConfig,
    today: Date,
    run_id: int | None = None,
    resend: bool = False,
) -> list[DeliveryOutcome]:
    """Mirror an already-written digest to every configured channel.

    Call this only after the vault write has succeeded. `context` is the same
    object `report/daily.py` rendered from, so the mirror cannot disagree with the
    file about what the day contained.

    `today` must be the operator's local calendar date (`settings.tzinfo`), not
    UTC — see `is_within_delivery_window`.

    Does not raise for a failed delivery: a channel that fails returns an outcome
    carrying the exception, because one broken channel must not abort the run and
    the digest is already safely on disk (§7, NEVER rule 12). That is a promise
    about delivery, not a promise that no exception can ever cross this line, so
    the CLI still keeps a backstop.
    """
    email = settings.delivery.email
    if email is None or not email.enabled:
        return []

    return [
        _deliver_email_channel(
            conn, context, channel=email, today=today, run_id=run_id, resend=resend
        )
    ]


def _deliver_email_channel(
    conn: sqlite3.Connection,
    context: DigestContext,
    *,
    channel: EmailChannelConfig,
    today: Date,
    run_id: int | None,
    resend: bool,
) -> DeliveryOutcome:
    """One channel's attempt, with every refusal named.

    The guards run cheapest-first and, crucially, the freshness rail runs *before*
    the send-log check: `--resend` is allowed to overrule a recorded delivery but
    never the age window, so no flag combination can mail a month of history.
    """
    if not is_within_delivery_window(context.date, today=today):
        reason = (
            f"digest for {context.date.isoformat()} is more than "
            f"{MAX_DELIVERY_AGE_DAYS} day(s) from today; not sent"
        )
        logger.info(
            "skipped delivery: report outside the freshness window",
            extra={"channel": CHANNEL_NAME, "run_id": run_id, "reason": reason},
        )
        return DeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=reason)

    already_sent = delivery_exists(
        conn,
        channel=CHANNEL_NAME,
        report_kind=REPORT_KIND_DAILY,
        target_date=context.date,
    )
    if already_sent and not resend:
        return DeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=False,
            reason=(
                f"already delivered for {context.date.isoformat()}; "
                "re-rendering does not re-send (use --resend to override)"
            ),
        )

    api_key = get_secret(channel.api_key_env)
    if api_key is None:
        # Configured on but unusable. An error, not a silent skip: the operator
        # asked for email and is not getting it.
        error = RuntimeError(
            f"email delivery is enabled but {channel.api_key_env} is not set "
            "in the environment or .env"
        )
        return DeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=str(error), error=error)

    try:
        body = render_email(context)
        provider_id = send_email(body, config=channel, api_key=api_key, run_id=run_id)
    except SendError as exc:
        return DeliveryOutcome(
            channel=CHANNEL_NAME, sent=False, reason=f"send failed: {exc}", error=exc
        )
    except Exception as exc:
        # Anything else the render or the client can throw — a jinja `TemplateError`,
        # an httpx error outside the retry predicate. Caught here rather than left to
        # the CLI because the contract this module offers is "a channel failure is an
        # outcome, not an exception" (§7), and a partial exception list is a contract
        # that quietly stops holding.
        logger.exception(
            "email delivery raised unexpectedly",
            extra={"channel": CHANNEL_NAME, "run_id": run_id},
        )
        return DeliveryOutcome(
            channel=CHANNEL_NAME, sent=False, reason=f"send failed: {exc}", error=exc
        )

    # Written only after the provider accepted the message, so a failed send leaves
    # no row and tomorrow's run may legitimately retry (`db.record_delivery`).
    recorded = record_delivery(
        conn,
        run_id=run_id,
        channel=CHANNEL_NAME,
        report_kind=REPORT_KIND_DAILY,
        target_date=context.date,
        body_hash=body.body_hash,
        # Sanitized here, before the row write, not refused there. `provider_id` is
        # world-authored — it comes back in the provider's JSON — and
        # `record_delivery` rejects a control character in it. That refusal would
        # now fire *after* the message was sent, losing the send log and letting a
        # re-render mail a second copy: the exact outcome the table exists to
        # prevent. It is an audit field, not an identity we act on, so dropping an
        # unusable value costs a traceability nicety and saves the guard.
        provider_id=_usable_provider_id(provider_id),
        sent_at=datetime.now(UTC),
        expect_existing=resend,
    )
    if not recorded and not resend:
        # `record_delivery` logs a warning; surface it as an error record too, since
        # it means a second copy reached the inbox.
        error = RuntimeError(
            f"a delivery was already recorded for {context.date.isoformat()}; "
            "a duplicate message was sent"
        )
        return DeliveryOutcome(
            channel=CHANNEL_NAME,
            sent=True,
            reason=str(error),
            provider_id=provider_id,
            error=error,
        )

    return DeliveryOutcome(
        channel=CHANNEL_NAME,
        sent=True,
        reason=f"delivered to {len(channel.to)} recipient(s)",
        provider_id=provider_id,
    )


def _usable_provider_id(provider_id: str | None) -> str | None:
    """Drop a provider id the send log would refuse — see the call site."""
    if provider_id is None or has_control_characters(provider_id):
        if provider_id is not None:
            logger.warning(
                "provider returned an id containing a control character; not recording it",
                extra={"channel": CHANNEL_NAME},
            )
        return None
    return provider_id


def send_test_message(
    channel: EmailChannelConfig, api_key: SecretStr, *, today: Date
) -> DeliveryOutcome:
    """Send one sample message, proving the channel works before trusting it.

    Takes an already-resolved channel and key: the CLI has to check both anyway to
    choose its exit code (misconfiguration is 2, a provider refusal is 1), and
    re-checking them here would be the second definition of a guard chain, not the
    single one. What lives here is the part worth sharing — the sample, the render,
    and turning a failure into an outcome.

    Deliberately touches no pipeline state: no `runs` row, no `deliveries` row, no
    database connection at all. The sample is a real render of a fabricated
    one-item digest, so what arrives is what a real digest will look like.
    """
    context = DigestContext(
        date=today,
        items=(
            DigestLine(
                id=None,
                title="SignalForge delivery test",
                url="https://github.com/signalforge",
                why_it_matters=(
                    "If you are reading this on your phone, the channel works. "
                    "This is a sample — no real items were scored for it."
                ),
                signal=5,
                relevance=5,
                novelty=5,
            ),
        ),
        source_failures=(),
        killed_count=0,
        scored_count=1,
        hidden_kept_count=0,
    )

    try:
        provider_id = send_email(render_email(context), config=channel, api_key=api_key)
    except Exception as exc:
        return DeliveryOutcome(channel=CHANNEL_NAME, sent=False, reason=str(exc), error=exc)

    return DeliveryOutcome(
        channel=CHANNEL_NAME,
        sent=True,
        reason=f"sample sent to {', '.join(channel.to)}",
        provider_id=provider_id,
    )
