"""Render a digest as email and hand it to a provider API.

Two halves, deliberately separable so the expensive one can be tested without the
network and the cheap one without a renderer:

* `render_email` — pure. `DigestContext` in, subject/HTML/text out. No I/O beyond
  loading templates.
* `send_email` — the only thing here that touches the network. One POST.

Escaping is the load-bearing detail. `report/daily.py` renders with
`autoescape=False`, which is right for markdown into Obsidian and wrong for HTML
carrying `items.title` — a field controlled outright by whoever publishes the feed
(DESIGN §13.1). This module keeps its own environment, escaping per template kind,
so the HTML part escapes and the plaintext part does not — see `_autoescape_for`
for why jinja's own `select_autoescape` is the wrong tool here.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx
import jinja2
from pydantic import SecretStr
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from signalforge.config import EmailChannelConfig
from signalforge.ingest.base import DEFAULT_USER_AGENT, parse_retry_after
from signalforge.models import is_safe_url
from signalforge.report.daily import DigestContext

__all__ = [
    "CHANNEL_NAME",
    "EmailBody",
    "SendError",
    "render_email",
    "send_email",
]

logger = logging.getLogger(__name__)

CHANNEL_NAME: Final = "email"
"""The `deliveries.channel` value. A constant, not an inline literal, because that
column is half an index key and SQLite compares it byte-for-byte — `"Email"` and
`"email"` would be two rows and the idempotency guard would quietly stop guarding."""

_RESEND_ENDPOINT: Final = "https://api.resend.com/emails"
_TEMPLATES_DIR: Final = "templates"
_HTML_TEMPLATE: Final = "email_daily.html.j2"
_TEXT_TEMPLATE: Final = "email_daily.txt.j2"

_MAX_ATTEMPTS: Final = 3
"""Fewer than the ingest fetcher's, because this runs once a day against one
well-behaved API and a failed send is recorded and retried tomorrow, not lost."""

_TIMEOUT_SECONDS: Final = 15.0

_MAX_RETRY_AFTER_SECONDS: Final = 60.0
"""Cap on the provider's own backoff hint, so a hostile or buggy `Retry-After`
cannot stall the 06:00 cron run."""

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
"""Statuses worth a second POST at a mail provider.

Deliberately local rather than reusing `ingest.base.RETRYABLE_STATUS`: that set's
membership is argued in feed terms (its comment reasons about GitHub returning 403
for an exhausted rate limit), so a future feed quirk that edits it must not
silently change what this module retries. `parse_retry_after` is shared because
RFC 9110 date parsing is genuinely the same problem; a policy set is not.
"""

_REPLAYABLE_TRANSPORT_ERRORS: Final = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)
"""Transport failures where the request provably never reached the provider.

Narrower than `httpx.TransportError` on purpose. A `ReadTimeout` means the request
was fully sent and the *response* was lost — the mail may well have been queued —
so replaying it is how an operator gets three copies of the same digest. Only
connect-phase failures are safe to retry for a POST that is not idempotent
(CLAUDE.md §3, NEVER rule 4).
"""


class SendError(Exception):
    """The provider refused the message, or retries were exhausted.

    Raised rather than returned so the caller cannot mistake a failed send for a
    successful one and write a `deliveries` row for it.
    """


class _RetryableStatus(Exception):
    """Internal: a status worth retrying, carrying any `Retry-After` hint."""

    def __init__(self, status_code: int, retry_after: float | None) -> None:
        super().__init__(f"provider returned HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class EmailBody:
    """One rendered message, in both parts a mail client may prefer."""

    subject: str
    html: str
    text: str

    @property
    def body_hash(self) -> str:
        """Stable fingerprint of what was sent, for the `deliveries` audit trail.

        Over both parts, so a change to either is visible. It answers "was what I
        read the same text the vault holds now?" after a re-render; it is
        deliberately not part of the idempotency key (see `db.record_delivery`).
        """
        digest = hashlib.sha256()
        digest.update(self.subject.encode("utf-8"))
        digest.update(self.html.encode("utf-8"))
        digest.update(self.text.encode("utf-8"))
        return digest.hexdigest()


def _autoescape_for(template_name: str | None) -> bool:
    """Escape HTML templates, leave plaintext alone.

    Hand-rolled rather than `jinja2.select_autoescape`, which matches on the file
    *extension* — and these files end in `.j2`, following the repo's convention
    (`daily.md.j2`). `select_autoescape(enabled_extensions=("html",))` therefore
    silently disables escaping for `email_daily.html.j2`, which is the worst
    possible failure: it looks configured and does nothing.
    """
    return template_name is not None and template_name.endswith((".html.j2", ".html"))


def _template_env() -> jinja2.Environment:
    """This module's own environment — see the module docstring on escaping."""
    return jinja2.Environment(
        loader=jinja2.PackageLoader("signalforge.deliver", _TEMPLATES_DIR),
        # The whole reason `deliver/` does not reuse `report/daily.py`'s env: an
        # item title is feed-controlled text, and it lands in HTML here.
        autoescape=_autoescape_for,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_email(context: DigestContext) -> EmailBody:
    """Build the message from an already-assembled `DigestContext`.

    Takes the same context object `report/daily.py` renders markdown from, so the
    email cannot disagree with the vault file about what the day contained — there
    is one query and one selection, not two.

    No checkboxes: an email is not an input surface (DESIGN §13.1). The templates
    render `pending_proposals | length` as a count instead, which is the nudge back
    to Obsidian when a decision is actually owed.
    """
    env = _template_env()
    # An item URL is feed-controlled and `Item.url` only checks it is non-empty, so
    # `javascript:` and `data:` survive to here. Escaping does not help — the URL is
    # about to sit in a live `href`, which is the one context where a well-formed
    # string is the danger. `is_safe_url` is the predicate the rest of the codebase
    # already refuses citations with (`db.insert_proposal`); the template falls back
    # to plain text when it fails, so a bad URL costs a link, not a click.
    env.globals["is_safe_url"] = is_safe_url
    subject = f"SignalForge — {context.date.isoformat()} ({len(context.items)} items)"
    return EmailBody(
        subject=subject,
        html=env.get_template(_HTML_TEMPLATE).render(context=context),
        text=env.get_template(_TEXT_TEMPLATE).render(context=context),
    )


def _wait_honoring_retry_after(state: RetryCallState) -> float:
    """Exponential backoff, floored by the provider's `Retry-After`.

    Same rule as `ingest/base.py`, and the same RFC 9110 parser: the server's hint
    wins when it is longer than our backoff, capped so a hostile value cannot stall
    the 06:00 cron run.
    """
    base: float = wait_exponential(multiplier=1, min=1, max=10)(state)
    exc = state.outcome.exception() if state.outcome is not None else None
    if isinstance(exc, _RetryableStatus) and exc.retry_after is not None:
        return max(base, min(exc.retry_after, _MAX_RETRY_AFTER_SECONDS))
    return base


def _post_once(
    client: httpx.Client, *, payload: dict[str, object], api_key: SecretStr
) -> str | None:
    """One POST. Returns the provider's message id, or None if it did not give one."""
    response = client.post(
        _RESEND_ENDPOINT,
        json=payload,
        headers={
            # `get_secret_value` is called here and nowhere else, and the result is
            # never logged or put in an exception message (CLAUDE.md §10 rule 16).
            "Authorization": f"Bearer {api_key.get_secret_value()}",
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    if response.status_code in _RETRYABLE_STATUS:
        raise _RetryableStatus(
            response.status_code, parse_retry_after(response.headers.get("retry-after"))
        )
    # An explicit success gate, not `>= 400`. `httpx` does not follow redirects by
    # default, so a 302 from a moved endpoint or a captive portal would otherwise
    # fall through every guard and be recorded as a delivered message — after which
    # the unique index suppresses the retry forever (NEVER rule 4).
    if not 200 <= response.status_code < 300:
        # The body can carry the provider's explanation, which is worth having in
        # `runs.errors`. It cannot carry the key — that only ever went out in a
        # header — but it is truncated anyway rather than trusted wholesale.
        raise SendError(f"provider returned HTTP {response.status_code}: {response.text[:300]}")

    try:
        decoded = response.json()
    except ValueError:
        return None
    # A JSON list or bare string is a 2xx we do not understand, not a failure: the
    # provider accepted the message. Reaching `.get` on it would raise
    # `AttributeError` past the `SendError` contract, and the caller would record a
    # failure for mail that was actually sent.
    if not isinstance(decoded, dict):
        return None
    identifier = decoded.get("id")
    return str(identifier) if identifier is not None else None


def send_email(
    body: EmailBody,
    *,
    config: EmailChannelConfig,
    api_key: SecretStr,
    run_id: int | None = None,
    sleep: Callable[[int | float], None] = time.sleep,
) -> str | None:
    """POST the message, returning the provider's id for the `deliveries` row.

    Raises `SendError` on a refusal or on retry exhaustion; the caller records that
    into `runs.errors` and writes no delivery row, so tomorrow's run may retry.

    `sleep` is the backoff seam. It is injected rather than having tests replace
    `_wait_honoring_retry_after`, because patching the wait function would remove
    the `Retry-After` handling from the code under test — a test suite that never
    exercises the thing the retry exists for.
    """
    payload: dict[str, object] = {
        "from": config.from_address,
        "to": list(config.to),
        "subject": body.subject,
        "html": body.html,
        "text": body.text,
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            for attempt in Retrying(
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                wait=_wait_honoring_retry_after,
                retry=retry_if_exception_type((_RetryableStatus, *_REPLAYABLE_TRANSPORT_ERRORS)),
                sleep=sleep,
                reraise=True,
            ):
                with attempt:
                    identifier = _post_once(client, payload=payload, api_key=api_key)
                    logger.info(
                        "delivered digest by email",
                        extra={
                            "channel": CHANNEL_NAME,
                            "provider_id": identifier,
                            "run_id": run_id,
                        },
                    )
                    return identifier
    except (_RetryableStatus, httpx.TransportError) as exc:
        # Say which it was. A `ReadTimeout` is deliberately not replayed — the
        # request was fully sent and the mail may already be queued — so reporting
        # it as "failed after 3 attempts" would misdescribe both what happened and
        # what the operator should expect in their inbox.
        replayed = isinstance(exc, (_RetryableStatus, *_REPLAYABLE_TRANSPORT_ERRORS))
        detail = f"after {_MAX_ATTEMPTS} attempts" if replayed else "without retrying"
        raise SendError(f"email delivery failed {detail}: {exc}") from exc

    # Unreachable: `Retrying` either returns from inside the loop or raises.
    raise SendError("email delivery produced no result")  # pragma: no cover
