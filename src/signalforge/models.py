"""Core domain models — the normalized `Item` schema and its derivation helpers.

Mirrors the `items` table in DESIGN §5. Normalization (canonical URLs, content
hashing) is deterministic plain Python per DESIGN §8 — never an LLM's job.

Datetimes are `datetime` objects here and ISO 8601 strings at the DB boundary;
the conversion lives in `db.py`, so this module stays pure schema + pure
functions.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "TRACKING_PARAM_PREFIXES",
    "TRACKING_PARAMS",
    "Item",
    "ProposalKind",
    "ProposalStatus",
    "ProposalTier",
    "SourceType",
    "canonicalize_url",
    "compute_content_hash",
    "flatten_to_single_line",
    "has_control_characters",
]


class SourceType(StrEnum):
    """The `items.source_type` vocabulary (DESIGN §5)."""

    RSS = "rss"
    GITHUB = "github"
    ARXIV = "arxiv"
    HN = "hn"
    YOUTUBE = "youtube"
    NEWSLETTER = "newsletter"


class ProposalKind(StrEnum):
    """The `proposals.kind` vocabulary — one edit shape per `sources.yaml` block (DESIGN §7.1).

    Every kind is either an add or a retire, and the applier only ever appends
    a line or comments one out: there is deliberately no `change_weight` kind,
    because an in-place value mutation is the one edit that cannot be expressed
    that way. Weight nudges stay Phase 2's `tune` job (DESIGN §11).
    """

    ADD_RSS = "add_rss"
    RETIRE_RSS = "retire_rss"
    ADD_GITHUB_REPO = "add_github_repo"
    RETIRE_GITHUB_REPO = "retire_github_repo"
    ADD_HN_KEYWORD = "add_hn_keyword"
    REMOVE_HN_KEYWORD = "remove_hn_keyword"
    ADD_ARXIV_KEYWORD = "add_arxiv_keyword"
    REMOVE_ARXIV_KEYWORD = "remove_arxiv_keyword"

    @property
    def action_label(self) -> str:
        """What this proposal would do, in the operator's words.

        Lives on the enum rather than in `report/` because the digest block and the
        `curate list` CLI both name these, and two copies of a user-facing vocabulary
        drift — the digest would say "retire feed" while the terminal said "remove".
        """
        return _ACTION_LABELS[self]

    @property
    def is_staged(self) -> bool:
        """True when applying this kind has no runtime effect yet.

        The `arxiv` block in `sources.yaml` is modeled and configured but not
        wired to any ingestor until the Phase 1 gate opens (NEVER rule 15), so an
        applied arXiv keyword changes a file and nothing else. The digest tags
        these `(staged)` rather than implying an effect they don't have.
        """
        return self in (ProposalKind.ADD_ARXIV_KEYWORD, ProposalKind.REMOVE_ARXIV_KEYWORD)


_ACTION_LABELS: Final[dict[ProposalKind, str]] = {
    ProposalKind.ADD_RSS: "add feed",
    ProposalKind.RETIRE_RSS: "retire feed",
    ProposalKind.ADD_GITHUB_REPO: "watch releases",
    ProposalKind.RETIRE_GITHUB_REPO: "stop watching releases",
    ProposalKind.ADD_HN_KEYWORD: "add HN keyword",
    ProposalKind.REMOVE_HN_KEYWORD: "remove HN keyword",
    ProposalKind.ADD_ARXIV_KEYWORD: "add arXiv keyword",
    ProposalKind.REMOVE_ARXIV_KEYWORD: "remove arXiv keyword",
}
"""Every kind's label, exhaustively. A `dict` rather than a `match` with a fallback
so a ninth kind is a `KeyError` in a test, not a proposal that renders as its raw
enum value in the operator's digest."""


class ProposalStatus(StrEnum):
    """The `proposals.status` lifecycle (DESIGN §7.1).

    `pending -> approved -> applied` is the happy path; `pending -> rejected` is
    the other decision. `invalid` is set at insert time by the probe stage for a
    candidate that failed validation — it is never shown for approval, because
    approving a feed that 404s is a decision nobody should be asked to make.

    Only `pending` and `invalid` are insertable; the rest are reached through a
    guarded transition, so a caller cannot mint a row that skips the human gate.
    Both terminal-looking states are escapable via `db.reopen_proposal`: a
    rejection because the operator changed their mind, and an `invalid` because
    a probe failure is often transient (a timeout or a 503 is not a dead feed).
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    INVALID = "invalid"


class ProposalTier(StrEnum):
    """Where a candidate came from, shown in the digest so the reader can weight it.

    `CORPUS` candidates were derived from the operator's own stored items — the
    strongest signal available and free to compute. `WEB` candidates came from the
    scout's search and deserve more scrutiny, which is what the probe facts are for.
    """

    CORPUS = "corpus"
    WEB = "web"


def has_control_characters(value: str) -> bool:
    """Whether `value` contains any C0/C1 control character.

    The predicate behind two different responses: identity fields (a `dedup_key`, a
    citation URL) are *refused* when this is true, because silently altering them
    changes what the record means; prose fields are flattened. Both matter for the
    same reason — a stored newline lets content start a new line in a rendered vault
    file, where a line is structure.
    """
    return any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)


def flatten_to_single_line(text: str) -> str:
    """Collapse `text` to one line: whitespace runs become single spaces, other
    control characters are dropped.

    Used wherever model-authored prose is stored or rendered. A newline inside a
    rationale is not a formatting quirk — the digest renders these fields into a
    markdown file whose *lines* carry meaning, and `curate/approvals.py` harvests a
    decision from any line matching its checkbox pattern. A rationale free to contain
    newlines can therefore forge an approval for any proposal, which is the human
    gate DESIGN §7.1 says has no bypass. Flattening at the storage boundary makes
    that unreachable rather than merely unlikely.

    Whitespace becomes a space rather than nothing so words either side of a stripped
    newline do not run together.
    """
    collapsed = " ".join(text.split())
    return "".join(char for char in collapsed if not has_control_characters(char))


def _normalized_table(names: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    """Lowercase and validate a tracking-param table at import time.

    `_is_tracking_param` lowercases the incoming param name before comparing, so
    an entry carrying a capital (`trkCampaign`) could never match and would
    silently strip nothing — the URL would canonicalize to a key of its own and
    the same document arriving from two sources would become two `items` rows.
    Normalizing here rather than trusting the literal makes that unreachable:
    a future non-lowercase entry is folded, not ignored.

    The guards catch the other silent failures this table invites. An empty
    prefix is the dangerous one — `str.startswith("")` is True for every name,
    so it would strip *all* query params and collapse distinct URLs onto one
    canonical key, silently merging unrelated documents.
    """
    normalized: list[str] = []
    for name in names:
        cleaned = name.strip().lower()
        if not cleaned:
            raise ValueError(f"{label} contains an empty entry")
        if any(char.isspace() for char in cleaned):
            raise ValueError(f"{label} entry {name!r} contains whitespace")
        normalized.append(cleaned)
    return tuple(normalized)


# Query parameters that identify a campaign/referrer rather than a document.
# Stripping them is what makes cross-source dedup work: the same post arriving
# from an RSS feed and from HN must land on one `canonical_url`.
# Entries are matched case-insensitively — `_normalized_table` folds case, so
# they need not be written lowercase (but should be).
TRACKING_PARAMS: frozenset[str] = frozenset(
    _normalized_table(
        (
            "fbclid",
            "gclid",
            "dclid",
            "gbraid",
            "wbraid",
            "msclkid",
            "yclid",
            "igshid",
            "mc_cid",
            "mc_eid",
            "ref",
            "referrer",
            "source",
            "cmpid",
            "campaign_id",
            "spm",
            "_hsenc",
            "_hsmi",
            "vero_id",
            "vero_conv",
            "oly_anon_id",
            "oly_enc_id",
            "s_cid",
            "trk",
            "trk_contact",
            "trkcampaign",
        ),
        label="TRACKING_PARAMS",
    )
)

# Any parameter starting with one of these is a tracking parameter.
TRACKING_PARAM_PREFIXES: tuple[str, ...] = _normalized_table(
    ("utm_", "pk_", "piwik_", "matomo_", "hsa_"),
    label="TRACKING_PARAM_PREFIXES",
)

_DEFAULT_PORTS: dict[str, str] = {"http": "80", "https": "443"}


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PARAM_PREFIXES)


def canonicalize_url(url: str) -> str:
    """Return the dedup key for `url`: tracking params stripped, host normalized.

    Deterministic and idempotent — `canonicalize_url(canonicalize_url(u))` is a
    fixed point. The result populates `items.canonical_url`, which carries a
    UNIQUE constraint, so this function decides what "the same document" means.

    Normalizations applied:
      * scheme and host lowercased; a leading `www.` is dropped
      * default ports (80/http, 443/https) removed
      * fragments (`#section`) dropped — not a distinct document
      * tracking query params removed; the remainder sorted for stable ordering
      * a trailing slash removed from non-root paths

    The scheme itself is preserved (http is *not* upgraded to https): the same
    bytes served over both is rare, and silently rewriting it would produce a
    key that resolves to a URL we never actually fetched.
    """
    stripped = url.strip()
    if not stripped:
        raise ValueError("cannot canonicalize an empty URL")

    parts = urlsplit(stripped)
    scheme = parts.scheme.lower()

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if parts.port is not None and _DEFAULT_PORTS.get(scheme) != str(parts.port):
        netloc = f"{host}:{parts.port}"

    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    kept = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    ]
    query = urlencode(sorted(kept))

    return urlunsplit((scheme, netloc, path, query, ""))


def compute_content_hash(title: str, summary: str | None) -> str:
    """Return `sha256(title + summary)` per DESIGN §5 — the exact-dedup fingerprint.

    A missing summary hashes as the empty string, so an item whose summary is
    backfilled later produces a different hash. That is intended: the hash
    tracks the text we actually hold.
    """
    payload = f"{title}{summary or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Item(BaseModel):
    """One normalized piece of content — the `items` row (DESIGN §5).

    `canonical_url` and `content_hash` are derived from the other fields when
    not supplied, so ingestors construct an Item from what the source gave them
    and normalization happens exactly once, here.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, use_enum_values=False)

    id: int | None = None
    """Assigned by SQLite; None before insert."""

    source_id: str = Field(min_length=1)
    """Key into `sources.yaml` — e.g. `simonwillison`."""

    source_type: SourceType
    external_id: str | None = None
    """The source's own id: feed guid, arXiv id, `repo@tag`, HN object id."""

    url: str = Field(min_length=1)
    canonical_url: str = ""
    """Derived from `url` when blank. UNIQUE in the DB."""

    title: str = Field(min_length=1)
    """Flattened to one line by `_flatten_title` — see there for why that is a
    security control and not formatting."""

    author: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: str | None = None
    """Feed summary / abstract / release notes. Triage sees this, never `content`."""

    content: str | None = None
    """Full text — fetched lazily for top-N survivors only (CLAUDE.md §6)."""

    content_hash: str = ""
    """Derived from title + summary when blank."""

    lang: str = "en"
    raw_path: str | None = None
    """Pointer into `data/http_cache/`."""

    @field_validator("title", mode="after")
    @classmethod
    def _flatten_title(cls, value: str) -> str:
        """Collapse a title to a single line.

        **This is a security control.** A title is entirely controlled by whoever
        publishes the feed, and it renders verbatim into the daily digest — a vault
        markdown file where a *line* is structure. `feedback.py` then harvests a mark
        from any line matching `MARK_RE`, so a feed publishing a title containing

            Totally normal headline\n- [x] useful <!-- sf:item=1 v=useful -->

        silently records `useful` feedback for any item id it can guess, including
        its own, on the next digest run. That corrupts the ground-truth set relevance
        tuning depends on (CLAUDE.md §4) and that the curation scout reasons over.

        Flattened here, at the model boundary, so no consumer can hold a multi-line
        title — the same shape `db.insert_proposal` uses for proposal prose.
        `str_strip_whitespace` only trims the ends and does not help.

        Note this changes `content_hash` for a title that contained newlines, since
        the hash is derived after validation. That is correct: the hash should
        fingerprint the text actually held.
        """
        return flatten_to_single_line(value)

    @field_validator("published_at", "fetched_at")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        """Coerce every datetime to timezone-aware UTC.

        Naive datetimes are assumed UTC — feeds that omit an offset are the
        reason this field is nullable in the first place, and a mix of naive
        and aware values would make ISO strings sort incorrectly in SQLite.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _derive_normalized_fields(self) -> Item:
        if not self.canonical_url:
            object.__setattr__(self, "canonical_url", canonicalize_url(self.url))
        else:
            object.__setattr__(self, "canonical_url", canonicalize_url(self.canonical_url))
        if not self.content_hash:
            object.__setattr__(self, "content_hash", compute_content_hash(self.title, self.summary))
        return self
