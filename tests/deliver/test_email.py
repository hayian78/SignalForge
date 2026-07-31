"""Tests for `deliver/email.py` — the read-only outbound mirror (DESIGN §13.2).

Three properties carry the weight:

* **The HTML part escapes and the plaintext part does not.** An item title is
  controlled outright by whoever publishes the feed (DESIGN §13.1), and this is
  the one place in the codebase it lands in HTML.
* **The email is not an input surface.** No checkbox marker may render, in either
  part — a mail client cannot harvest one, and a forged one in the vault is the
  attack §13.1 exists to close.
* **A failed send raises.** The caller must not be able to mistake a refusal for a
  success and write a `deliveries` row for a message nobody received.
"""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import timedelta

import httpx
import pytest
import respx
from pydantic import SecretStr

from signalforge.config import EmailChannelConfig
from signalforge.deliver import MAX_DELIVERY_AGE_DAYS, is_within_delivery_window
from signalforge.deliver.email import (
    CHANNEL_NAME,
    SendError,
    _autoescape_for,
    render_email,
    send_email,
)
from signalforge.report.daily import DigestContext, DigestLine, ProposalLine, SourceFailure

ENDPOINT = "https://api.resend.com/emails"


def _no_sleep(_seconds: float) -> None:
    """Backoff seam. Injected rather than patching `_wait_honoring_retry_after`,
    so the `Retry-After` logic stays under test instead of being replaced."""


DIGEST_DATE = Date(2026, 8, 1)
API_KEY = SecretStr("re_test_key_not_a_real_one_at_all")

CHANNEL = EmailChannelConfig(
    enabled=True,
    api_key_env="RESEND_API_KEY",
    from_address="SignalForge <digest@example.com>",
    to=["reader@example.com"],
)


def make_context(
    *,
    items: tuple[DigestLine, ...] = (),
    failures: tuple[SourceFailure, ...] = (),
    proposals: tuple[ProposalLine, ...] = (),
) -> DigestContext:
    return DigestContext(
        date=DIGEST_DATE,
        items=items,
        source_failures=failures,
        killed_count=20,
        scored_count=37,
        hidden_kept_count=8,
        proposals=proposals,
    )


def make_line(*, title: str = "A real headline", url: str = "https://example.com/a") -> DigestLine:
    return DigestLine(
        id=1322,
        title=title,
        url=url,
        why_it_matters="Concrete numbers and a working demo.",
        signal=4,
        relevance=5,
        novelty=3,
    )


# --------------------------------------------------------------------------- #
# render_email
# --------------------------------------------------------------------------- #


def test_the_html_part_escapes_a_feed_controlled_title() -> None:
    """The reason `deliver/` keeps its own jinja environment.

    `report/daily.py` renders with `autoescape=False`, which is correct for
    markdown into Obsidian and would be a hole here.
    """
    body = render_email(
        make_context(items=(make_line(title="Frontier <script>alert(1)</script>"),))
    )

    assert "<script>alert(1)</script>" not in body.html
    assert "&lt;script&gt;" in body.html


def test_autoescape_keys_on_the_full_template_name_not_the_extension() -> None:
    """Pins the bug this nearly shipped with.

    The templates end in `.j2`, following the repo's convention (`daily.md.j2`),
    so `jinja2.select_autoescape(enabled_extensions=("html",))` matches neither of
    them and silently disables escaping — configuration that looks present and
    does nothing. Asserted on the predicate directly, because the symptom is
    indistinguishable from "this title happened to contain no markup".
    """
    assert _autoescape_for("email_daily.html.j2") is True
    assert _autoescape_for("email_daily.txt.j2") is False
    assert _autoescape_for(None) is False


def test_the_html_part_escapes_an_ampersand_in_a_url() -> None:
    body = render_email(make_context(items=(make_line(url="https://example.com/a?x=1&y=2"),)))
    assert "x=1&amp;y=2" in body.html


def test_the_html_part_escapes_a_source_failure_message() -> None:
    """Error text is world-authored too — it can carry a server's response body."""
    body = render_email(make_context(failures=(SourceFailure("import-ai", "HTTP 403 <b>x</b>"),)))
    assert "<b>x</b>" not in body.html
    assert "&lt;b&gt;" in body.html


def test_the_plaintext_part_is_not_escaped() -> None:
    """Escaping plaintext would show readers literal `&amp;` in their mail client."""
    body = render_email(make_context(items=(make_line(title="Tools & Frameworks"),)))
    assert "Tools & Frameworks" in body.text
    assert "&amp;" not in body.text


@pytest.mark.parametrize("part", ["html", "text"])
def test_neither_part_renders_a_feedback_checkbox(part: str) -> None:
    """An email is not an input surface (DESIGN §13.1).

    `feedback.py` and `curate/approvals.py` harvest from vault markdown only, so a
    checkbox here would be dead syntax at best — and rendering the marker wire
    format into a second channel is how it stops having exactly one definition.
    """
    body = render_email(
        make_context(
            items=(make_line(),),
            proposals=(
                ProposalLine(
                    id=7,
                    action="add feed",
                    target="https://transformernews.ai/feed",
                    rationale="Cited three times this month.",
                    evidence=(("https://example.com/post", ""),),
                    tier="web",
                    pending=True,
                    settled_note="",
                    staged=False,
                    weight=1.0,
                    probe="200 OK",
                ),
            ),
        )
    )
    rendered = getattr(body, part)

    assert "sf:item=" not in rendered
    assert "sf:proposal=" not in rendered
    assert "- [ ]" not in rendered
    assert "- [x]" not in rendered


def test_a_pending_proposal_renders_as_a_count_pointing_back_at_the_vault() -> None:
    """The nudge that keeps the loop closed once the phone is the read surface."""
    proposal = ProposalLine(
        id=7,
        action="add feed",
        target="https://transformernews.ai/feed",
        rationale="Cited three times this month.",
        evidence=(("https://example.com/post", ""),),
        tier="web",
        pending=True,
        settled_note="",
        staged=False,
        weight=None,
        probe="",
    )
    body = render_email(make_context(items=(make_line(),), proposals=(proposal,)))

    assert "1 source change(s) awaiting your decision" in body.text
    assert "1 source change(s) awaiting your decision" in body.html
    assert f"daily/{DIGEST_DATE.isoformat()}.md" in body.text


def test_an_empty_day_still_renders_both_parts() -> None:
    body = render_email(make_context())
    assert "No items cleared triage today" in body.text
    assert "No items cleared triage today" in body.html


def test_the_subject_names_the_date_and_the_item_count() -> None:
    body = render_email(make_context(items=(make_line(), make_line())))
    assert body.subject == "SignalForge — 2026-08-01 (2 items)"


def test_the_body_hash_is_stable_and_covers_every_part() -> None:
    """The `deliveries` audit trail: same render, same hash; any change moves it."""
    first = render_email(make_context(items=(make_line(),)))
    same = render_email(make_context(items=(make_line(),)))
    different = render_email(make_context(items=(make_line(title="Something else"),)))

    assert first.body_hash == same.body_hash
    assert first.body_hash != different.body_hash


# --------------------------------------------------------------------------- #
# send_email — never a live call (NEVER rule 13)
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_successful_send_returns_the_provider_message_id() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"id": "msg_abc123"}))
    body = render_email(make_context(items=(make_line(),)))

    identifier = send_email(body, config=CHANNEL, api_key=API_KEY)

    assert identifier == "msg_abc123"
    assert route.call_count == 1


@respx.mock
def test_the_request_carries_both_parts_and_the_configured_addresses() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"id": "m"}))
    body = render_email(make_context(items=(make_line(),)))

    send_email(body, config=CHANNEL, api_key=API_KEY)

    sent = route.calls.last.request
    payload = json.loads(sent.content)
    assert payload["from"] == "SignalForge <digest@example.com>"
    assert payload["to"] == ["reader@example.com"]
    assert payload["subject"] == body.subject
    assert payload["html"] == body.html
    assert payload["text"] == body.text


@respx.mock
def test_the_api_key_travels_in_the_header_and_nowhere_else() -> None:
    """NEVER rule 16 — the secret is never in the body, and never logged."""
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"id": "m"}))

    send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY)

    sent = route.calls.last.request
    assert sent.headers["authorization"] == f"Bearer {API_KEY.get_secret_value()}"
    assert API_KEY.get_secret_value() not in sent.content.decode("utf-8")


@respx.mock
def test_a_permanent_refusal_raises_without_retrying() -> None:
    """A 422 is our fault, not weather — retrying it just spends time."""
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(422, text="from_address domain is not verified")
    )

    with pytest.raises(SendError, match="422"):
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY)

    assert route.call_count == 1


@respx.mock
def test_a_permanent_refusal_does_not_leak_the_key_into_the_error() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(400, text="bad request"))

    with pytest.raises(SendError) as caught:
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY)

    assert API_KEY.get_secret_value() not in str(caught.value)


@respx.mock
def test_a_transient_status_is_retried_then_succeeds() -> None:
    route = respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"id": "msg_after_retry"}),
        ]
    )

    identifier = send_email(
        render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=_no_sleep
    )

    assert identifier == "msg_after_retry"
    assert route.call_count == 2


@respx.mock
def test_retry_exhaustion_raises_rather_than_returning_none() -> None:
    """The caller writes a `deliveries` row on success. A silent None would log a
    send that never happened."""
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(503))

    with pytest.raises(SendError, match="after 3 attempts"):
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 3


@respx.mock
def test_a_connect_error_is_retried_and_then_raises() -> None:
    """A connect-phase failure provably never reached the provider, so replaying
    it cannot duplicate a message."""
    route = respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("no route to host"))

    with pytest.raises(SendError):
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 3


@respx.mock
def test_a_read_timeout_is_not_retried() -> None:
    """The request was fully sent; only the response was lost.

    Resend may well have queued the mail, so replaying the POST is how the operator
    gets three copies of one digest (NEVER rule 4). It surfaces as a failure the
    caller records instead — an unsent digest is recoverable tomorrow, a triplicate
    inbox is not.
    """
    route = respx.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("response lost"))

    with pytest.raises(SendError, match="without retrying"):
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=_no_sleep)

    assert route.call_count == 1


@respx.mock
def test_a_success_without_an_id_is_still_a_success() -> None:
    """A provider is not obliged to return an identifier; the send still happened."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text="ok"))

    assert send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY) is None


# --------------------------------------------------------------------------- #
# The delivery window — the guard the `deliveries` table cannot be
# --------------------------------------------------------------------------- #


def test_todays_and_yesterdays_reports_are_inside_the_window() -> None:
    assert is_within_delivery_window(DIGEST_DATE, today=DIGEST_DATE)
    assert is_within_delivery_window(DIGEST_DATE, today=Date(2026, 8, 2))


def test_an_old_report_is_outside_the_window() -> None:
    """The stateless half of the idempotency guard.

    `deliveries` lives in the database, which CLAUDE.md §3 calls regenerable — the
    operator may delete it and re-ingest. Without this rail, re-rendering past
    dates afterwards would mail weeks of old digests.
    """
    assert not is_within_delivery_window(Date(2026, 7, 1), today=DIGEST_DATE)
    assert not is_within_delivery_window(DIGEST_DATE, today=Date(2026, 8, 3))


def test_a_just_future_dated_report_is_inside_the_window() -> None:
    """A clock or timezone artifact, not stale news — refusing it would be a
    silent failure on the one day it happened."""
    assert is_within_delivery_window(DIGEST_DATE + timedelta(days=1), today=DIGEST_DATE)


def test_a_far_future_dated_report_is_outside_the_window() -> None:
    """The window is symmetric: `--date 2099-01-01` is a typo, not a clock skew.

    Mailing what it renders would also burn that date's send-log row, suppressing
    the real send when the day arrives.
    """
    assert not is_within_delivery_window(Date(2099, 1, 1), today=DIGEST_DATE)
    assert not is_within_delivery_window(DIGEST_DATE + timedelta(days=2), today=DIGEST_DATE)


def test_the_window_is_exactly_the_documented_number_of_days() -> None:
    boundary = Date(2026, 8, 1 + MAX_DELIVERY_AGE_DAYS)
    assert is_within_delivery_window(DIGEST_DATE, today=boundary)
    assert not is_within_delivery_window(
        DIGEST_DATE, today=Date(boundary.year, 8, boundary.day + 1)
    )


def test_the_channel_name_is_a_constant_not_an_inline_literal() -> None:
    """It is half an index key; SQLite compares it byte-for-byte."""
    assert CHANNEL_NAME == "email"


# --------------------------------------------------------------------------- #
# Non-success responses that must not be recorded as a delivery
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_redirect_is_a_failure_not_a_silent_success() -> None:
    """`httpx` does not follow redirects by default.

    A moved endpoint or a captive portal returns 302, which is neither retryable
    nor >= 400. Treating it as delivered would write a `deliveries` row for a
    message nobody received, and the unique index would then suppress the retry
    forever (NEVER rule 4).
    """
    respx.post(ENDPOINT).mock(return_value=httpx.Response(302, text="moved"))

    with pytest.raises(SendError, match="302"):
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=_no_sleep)


@respx.mock
def test_a_non_dict_json_body_is_still_a_success() -> None:
    """A 2xx we do not understand is not a failure — the provider took the message.

    Reaching `.get("id")` on a list would raise `AttributeError` straight past the
    `SendError` contract, and the caller would record a failure for mail that was
    actually sent, then re-send it.
    """
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=["queued"]))

    assert (
        send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=_no_sleep)
        is None
    )


@respx.mock
def test_the_provider_backoff_hint_is_read_and_capped() -> None:
    """Covers the reason `parse_retry_after` was shared from `ingest/base.py`.

    Asserted on the wait the retry actually computes, not on elapsed time: the
    server's hint wins over exponential backoff when it is longer, and is capped so
    a hostile value cannot stall the 06:00 cron run.
    """
    waits: list[float] = []
    respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"id": "msg_after_backoff"}),
        ]
    )

    identifier = send_email(
        render_email(make_context()),
        config=CHANNEL,
        api_key=API_KEY,
        sleep=waits.append,
    )

    assert identifier == "msg_after_backoff"
    # Exponential backoff alone would have been 1s; the provider asked for 7.
    assert waits == [7.0]


@respx.mock
def test_an_absurd_backoff_hint_is_capped() -> None:
    waits: list[float] = []
    respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "86400"}),
            httpx.Response(200, json={"id": "m"}),
        ]
    )

    send_email(render_email(make_context()), config=CHANNEL, api_key=API_KEY, sleep=waits.append)

    assert waits == [60.0]


# --------------------------------------------------------------------------- #
# Unsafe item URLs — escaping is the wrong axis in an href
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html;base64,PHNjcmlwdD4="])
def test_an_unsafe_item_url_does_not_become_a_live_link(url: str) -> None:
    """`Item.url` only checks non-empty, so a feed can supply any scheme.

    Escaping does nothing here — the string is well-formed — so the anchor itself
    is gated on `is_safe_url`, the same predicate `db.insert_proposal` refuses
    citations with. A bad URL costs a link, not a click.
    """
    body = render_email(make_context(items=(make_line(title="Click me", url=url),)))

    assert f'href="{url}"' not in body.html
    assert "javascript:" not in body.html
    # The item is still reported; only its link is withheld.
    assert "Click me" in body.html


def test_a_safe_item_url_still_becomes_a_link() -> None:
    body = render_email(make_context(items=(make_line(url="https://example.com/a"),)))
    assert 'href="https://example.com/a"' in body.html
