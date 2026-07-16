"""Tests for `models.py` — the normalizer that decides what "the same document" means.

`canonicalize_url` populates `items.canonical_url`, which carries a UNIQUE
constraint. Every bug here is either a duplicate in the digest or a silently
dropped item, so the golden table below is the contract.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from signalforge.models import (
    TRACKING_PARAM_PREFIXES,
    TRACKING_PARAMS,
    Item,
    SourceType,
    canonicalize_url,
    compute_content_hash,
)
from tests.conftest import make_item

# --------------------------------------------------------------------------- #
# canonicalize_url — golden table (CLAUDE.md §8 requires golden tests here)
# --------------------------------------------------------------------------- #

CANONICALIZATION_CASES: list[tuple[str, str, str]] = [
    # (label, input, expected)
    (
        "plain url is untouched",
        "https://example.com/post",
        "https://example.com/post",
    ),
    (
        "scheme and host lowercased",
        "HTTPS://Example.COM/Post",
        "https://example.com/Post",
    ),
    (
        "path case is preserved — paths are case-sensitive on most servers",
        "https://example.com/A/B/CaseMatters",
        "https://example.com/A/B/CaseMatters",
    ),
    (
        "leading www. dropped",
        "https://www.example.com/post",
        "https://example.com/post",
    ),
    (
        "www. only stripped at the front, not mid-host",
        "https://blog.www.example.com/post",
        "https://blog.www.example.com/post",
    ),
    (
        "default https port removed",
        "https://example.com:443/post",
        "https://example.com/post",
    ),
    (
        "default http port removed",
        "http://example.com:80/post",
        "http://example.com/post",
    ),
    (
        "non-default port kept",
        "https://example.com:8443/post",
        "https://example.com:8443/post",
    ),
    (
        "fragment dropped — not a distinct document",
        "https://example.com/post#section-3",
        "https://example.com/post",
    ),
    (
        "trailing slash trimmed from a non-root path",
        "https://example.com/post/",
        "https://example.com/post",
    ),
    (
        "root slash preserved — '' would not be a valid URL path",
        "https://example.com/",
        "https://example.com/",
    ),
    (
        "utm_* stripped",
        "https://example.com/post?utm_source=newsletter&utm_medium=email&utm_campaign=x",
        "https://example.com/post",
    ),
    (
        "fbclid and friends stripped",
        "https://example.com/post?fbclid=abc&gclid=def&msclkid=ghi&igshid=jkl",
        "https://example.com/post",
    ),
    (
        "matomo/piwik/hubspot prefixes stripped",
        "https://example.com/post?pk_campaign=a&piwik_kwd=b&matomo_source=c&hsa_acc=d",
        "https://example.com/post",
    ),
    (
        "ref/referrer/source stripped",
        "https://example.com/post?ref=hn&referrer=twitter&source=rss",
        "https://example.com/post",
    ),
    (
        "tracking params are matched case-insensitively",
        "https://example.com/post?UTM_SOURCE=hn&FBCLID=x",
        "https://example.com/post",
    ),
    (
        "meaningful params survive",
        "https://example.com/search?q=mcp&page=2",
        "https://example.com/search?page=2&q=mcp",
    ),
    (
        "query params sorted for a stable key",
        "https://example.com/post?z=1&a=2&m=3",
        "https://example.com/post?a=2&m=3&z=1",
    ),
    (
        "tracking stripped while meaningful params are sorted and kept",
        "https://WWW.Example.com:443/post/?utm_source=hn&id=42&fbclid=z&cat=ai#top",
        "https://example.com/post?cat=ai&id=42",
    ),
    (
        "blank param value is kept — '?flag=' is not '?'",
        "https://example.com/post?flag=&x=1",
        "https://example.com/post?flag=&x=1",
    ),
    (
        "userinfo dropped — credentials are not part of a document's identity",
        "https://user:pw@example.com/post",
        "https://example.com/post",
    ),
    (
        "surrounding whitespace stripped (feeds are sloppy)",
        "  https://example.com/post  ",
        "https://example.com/post",
    ),
]


@pytest.mark.parametrize(
    ("url", "expected"),
    [pytest.param(url, expected, id=label) for label, url, expected in CANONICALIZATION_CASES],
)
def test_canonicalize_url_golden(url: str, expected: str) -> None:
    assert canonicalize_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [pytest.param(url, id=label) for label, url, _ in CANONICALIZATION_CASES],
)
def test_canonicalize_url_is_idempotent(url: str) -> None:
    # canonical_url is re-canonicalized on every Item load from the DB, so a
    # non-fixed-point normalizer would churn the UNIQUE key run to run.
    once = canonicalize_url(url)
    assert canonicalize_url(once) == once


def test_canonicalize_url_does_not_upgrade_http_to_https() -> None:
    # Deliberate: rewriting to a URL we never fetched would break citations
    # (CLAUDE.md §5). http and https are distinct documents to this pipeline.
    assert canonicalize_url("http://example.com/post") == "http://example.com/post"
    assert canonicalize_url("https://example.com/post") == "https://example.com/post"
    assert canonicalize_url("http://example.com/post") != canonicalize_url(
        "https://example.com/post"
    )


@pytest.mark.parametrize("param", sorted(TRACKING_PARAMS))
def test_every_declared_tracking_param_is_actually_stripped(param: str) -> None:
    # Guards against a dead entry in TRACKING_PARAMS: `_is_tracking_param`
    # lowercases the incoming name before the frozenset lookup, so any entry that
    # is not itself lowercase can never match and silently tracks nothing.
    assert canonicalize_url(f"https://example.com/post?{param}=x") == "https://example.com/post"


@pytest.mark.parametrize("prefix", TRACKING_PARAM_PREFIXES)
def test_every_declared_tracking_prefix_is_actually_stripped(prefix: str) -> None:
    assert (
        canonicalize_url(f"https://example.com/post?{prefix}thing=x") == "https://example.com/post"
    )


def test_canonicalize_url_collapses_the_same_post_from_two_sources() -> None:
    # The whole point: one document arriving via RSS and via HN must land on one key.
    from_rss = canonicalize_url("https://simonwillison.net/2026/Jul/15/mcp/?utm_source=feed")
    from_hn = canonicalize_url("https://www.simonwillison.net/2026/Jul/15/mcp/#comments")
    assert from_rss == from_hn == "https://simonwillison.net/2026/Jul/15/mcp"


def test_canonicalize_url_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty URL"):
        canonicalize_url("   ")


# --------------------------------------------------------------------------- #
# compute_content_hash
# --------------------------------------------------------------------------- #


def test_compute_content_hash_is_stable_across_calls() -> None:
    assert compute_content_hash("Title", "Summary") == compute_content_hash("Title", "Summary")


def test_compute_content_hash_is_sha256_of_title_plus_summary() -> None:
    import hashlib

    expected = hashlib.sha256(b"TitleSummary").hexdigest()
    assert compute_content_hash("Title", "Summary") == expected


def test_compute_content_hash_treats_none_summary_as_empty_string() -> None:
    assert compute_content_hash("Title", None) == compute_content_hash("Title", "")


def test_compute_content_hash_distinguishes_title_and_summary_changes() -> None:
    base = compute_content_hash("Title", "Summary")
    assert compute_content_hash("Title2", "Summary") != base
    assert compute_content_hash("Title", "Summary2") != base


def test_compute_content_hash_backfilled_summary_changes_the_hash() -> None:
    # Documented intent: the hash tracks the text we actually hold, so an item
    # whose summary arrives later is a different fingerprint.
    assert compute_content_hash("Title", None) != compute_content_hash("Title", "Now with summary")


# --------------------------------------------------------------------------- #
# Item — derived fields and validation
# --------------------------------------------------------------------------- #


def test_item_derives_canonical_url_from_url() -> None:
    item = make_item(url="https://WWW.Example.com/post/?utm_source=hn#x", canonical_url="")
    assert item.canonical_url == "https://example.com/post"
    assert item.url == "https://WWW.Example.com/post/?utm_source=hn#x", "url is stored verbatim"


def test_item_canonicalizes_an_explicitly_supplied_canonical_url() -> None:
    # An ingestor passing a hand-built canonical_url must not be able to smuggle
    # a non-canonical value into the UNIQUE column.
    item = make_item(canonical_url="https://www.example.com/post/?utm_source=x")
    assert item.canonical_url == "https://example.com/post"


def test_item_derives_content_hash_from_title_and_summary() -> None:
    item = make_item(title="T", summary="S", content_hash="")
    assert item.content_hash == compute_content_hash("T", "S")


def test_item_respects_an_explicit_content_hash() -> None:
    item = make_item(content_hash="deadbeef")
    assert item.content_hash == "deadbeef"


def test_item_naive_datetimes_are_assumed_utc() -> None:
    item = make_item(published_at=datetime(2026, 7, 15, 12, 0, 0))
    assert item.published_at is not None
    assert item.published_at.tzinfo is not None
    assert item.published_at == datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)


def test_item_aware_datetimes_are_converted_to_utc() -> None:
    # Mixed offsets would make ISO strings sort incorrectly in SQLite.
    aest = timezone(timedelta(hours=10))
    item = make_item(published_at=datetime(2026, 7, 15, 22, 0, 0, tzinfo=aest))
    assert item.published_at == datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)


def test_item_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="topics"):
        make_item(topics=["agents.mcp"])


def test_item_rejects_unknown_source_type() -> None:
    with pytest.raises(ValidationError):
        make_item(source_type="twitter")


@pytest.mark.parametrize("field", ["source_id", "url", "title"])
def test_item_rejects_empty_required_strings(field: str) -> None:
    with pytest.raises(ValidationError):
        make_item(**{field: ""})


def test_item_external_id_is_optional() -> None:
    # Plenty of feeds omit a guid; those items dedup on canonical_url alone.
    assert make_item(external_id=None).external_id is None


def test_source_type_vocabulary_matches_design_section_5() -> None:
    assert {member.value for member in SourceType} == {
        "rss",
        "github",
        "arxiv",
        "hn",
        "youtube",
        "newsletter",
    }


def test_item_content_defaults_to_none() -> None:
    # Full text is fetched lazily for top-N survivors only (CLAUDE.md §6); an
    # ingest-time Item must not carry it.
    assert make_item().content is None


def test_item_lang_defaults_to_en() -> None:
    assert make_item().lang == "en"


def test_item_id_is_none_before_insert() -> None:
    assert make_item().id is None


def test_item_title_whitespace_is_stripped() -> None:
    # str_strip_whitespace: a feed's leading newline must not fork the content hash.
    assert make_item(title="  MCP sampling  ").title == "MCP sampling"
    assert Item(
        source_id="s",
        source_type=SourceType.RSS,
        url="https://example.com/a",
        title="  T  ",
        summary="  S  ",
    ).content_hash == compute_content_hash("T", "S")
