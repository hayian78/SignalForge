# Changelog

All notable changes to SignalForge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No versions are
tagged yet — Phase 0's acceptance gate was met 2026-07-23 and the project is now
in **Phase 1** (see [DESIGN §16](docs/DESIGN.md#16-roadmap)), whose gate is four
consecutive Sunday briefs. Everything below sits under *Unreleased* until then.

## [Unreleased]

### Added
- **`taxonomy.yaml`** (DESIGN §10, Phase 1). The two-level topic tree —
  a group (`industry`, `frontier`, ...) containing leaves, each carrying its
  match keywords — validated by the new `TaxonomyConfig` in `config.py`.
  Modeled and validated only, same staging posture `arxiv:` carried before its
  ingestor shipped (NEVER rule 15): `score/taxonomy.py`'s keyword tagger and
  Haiku-triage fallback are a separate, larger unit of work with an open
  schema question (DESIGN §5 has no `item_topics` table yet), so editing this
  file has no runtime effect today. Deliberately minimal — exactly the six
  `group.leaf` pairs `interests.yaml`'s `priority_topics` already names
  (`industry.strategy`, `frontier.capabilities`, `enterprise.adoption`,
  `agents.autonomy`, `policy.regulation`, `ai.research-direction`), every
  keyword traceable to operator-authored config rather than invented
  wholesale; a shipped-config test asserts the two files can't silently drift
  apart. Growing the tree past those six is an operator edit, the same
  posture `sources.yaml`/`interests.yaml` already have.
- **arXiv ingestion** (`ingest/arxiv.py`, DESIGN §7, Phase 1). `arxiv.categories`
  and `arxiv.require_keywords` fold into one `search_query` per run — a single
  arXiv Atom API request, no politeness delay needed because there is never a
  second request to space out. Reuses `feedparser` (the same Atom parser
  `ingest/rss.py` already uses) rather than a bespoke XML client. Titles and
  abstracts only, never full paper text (NEVER rule 9); the version suffix on
  an entry id is stripped before it becomes `external_id`, so a metadata-only
  revision upserts onto the same row instead of duplicating it. A malformed
  query is a `200 OK` with a synthetic `api/errors#...` entry, not an HTTP
  error — handled the same way as any other unparseable entry, so a typo'd
  keyword degrades to zero items rather than a loud failure (`--dry-run` is
  the way to notice). Closes the last gap `ProposalKind.is_staged` was tracking:
  arXiv keyword proposals from adaptive source curation are no longer tagged
  `(staged)` in the digest, because applying one now has a real effect.
- **Email delivery of the daily digest** (`deliver/`, DESIGN §13.2). A read-only
  mirror so the digest can be read away from the desk. The vault write stays
  canonical and unconditional; the email is sent only after it succeeds, carries
  no checkboxes (feedback still round-trips through vault markdown only), and a
  dead provider is an error in `runs.errors` rather than a failed run. Two
  idempotency guards: a `UNIQUE(channel, report_kind, target_date)` index on the
  new `deliveries` table (migration 4), and a *stateless* freshness window, so a
  deleted-and-rebuilt database cannot mail weeks of history. `digest` gains
  `--no-send` / `--resend`; `signalforge deliver test` sends one sample without
  touching the pipeline. Config lives in `settings.yaml` under `delivery:`; the
  API key is named there, never written there. **Zero LLM cost, zero new
  dependencies** — and shipped ahead of its phase, recorded as a deliberate
  exception with its cost in DESIGN §13.2 rather than normalised.
- **A third feedback rung, `exceptional`** (#8), above `useful` — aggregations
  read "useful or better", so an item marked only `exceptional` still counts
  toward the Phase 1 gate.
- **Adaptive source curation** (#9, DESIGN §7.1): a weekly scout proposes feed
  additions and retirements from per-source yield plus live web search, surfaces
  them as tick-boxes at the foot of the daily digest, and applies approved changes
  append-only to `sources.yaml` as an uncommitted diff. Nothing changes without a
  tick; `proposals` table + `ux_proposals_kind_key` (migration 3) make a re-scout
  a no-op and stop a rejected proposal ever coming back.
- Phase 0 ingest loop: RSS, GitHub releases, and Hacker News into SQLite, with
  per-source failure isolation and conditional GET.
- Batched Haiku triage + 3-dimension scoring (signal / relevance / novelty),
  and the daily digest writer that renders survivors into the Obsidian vault.
- `daily_max_items` — cap the digest at the top-N ranked items.
- `max_item_age_days` — skip stale items at ingest.
- Per-source and per-GitHub-repo crowding limits (`daily_max_per_source`,
  `daily_max_per_github_repo`) so one prolific source can't sweep the digest.
- Configurable reader-facing timezone in `config/settings.yaml` — the digest
  day is resolved through one IANA zone while all storage stays UTC.
- Company engineering blogs and six release watches added to `sources.yaml`.
- Phase 1 `mark` feedback capture: a `signalforge mark <item-id> useful|noise|missed`
  CLI command plus two GFM checkboxes per digest item, harvested out of the vault
  markdown before each re-render ("harvest-then-overwrite") into the `feedback`
  table via a non-destructive `UNIQUE(item_id, verdict)` index (migration 2).
  Scoring is unchanged — a mark only stores ground-truth; adaptation is Phase 2.
  Landed as Phase 0's acceptance gate closed (2026-07-23): five mornings of
  real digest use plus a verified live double-run (second `daily` added 0 rows,
  spent 0 tokens, re-rendered byte-identically).

### Changed
- Executive-briefing rebalance (2026-07-24): shifted the profile from an
  AI-engineering/framework tracker toward where AI is heading — industry
  direction & thought leadership, frontier labs, enterprise adoption, and
  policy. `interests.yaml` rewritten (new `priority_topics`/`interests`,
  trimmed `stack`, `model-release-hype` dropped from `ignore`); `sources.yaml`
  swapped five engineering/inference-infra RSS feeds for six analyst /
  thought-leadership feeds (Import AI, One Useful Thing, MIT Tech Review AI,
  Ben Evans, AI Snake Oil, Latent Space, weighted up), pruned GitHub release
  watches 13→5 (paradigm/workflow repos only), and retargeted HN keywords
  (and the Phase-1 arXiv keyword staging) to frontier/direction/policy terms.
- `RUBRIC_VERSION` bumped `triage-v2` → `triage-v3`: the `signal` dimension was
  broadened from an engineering-artifact-only scale to a 5-point substance-vs-noise
  scale that also rewards original, evidence-backed analysis, so thought-leadership
  is no longer structurally scored down. Kill-on-hype (`signal=1`) is retained,
  now the filter for contentless marketing in place of the removed `ignore` topic.
- Dropped four zero-yield GitHub release watches (`ggml-org/llama.cpp`,
  `ollama/ollama`, `BerriAI/litellm`, `pydantic/pydantic-ai`) from `sources.yaml`
  after a week of `mark` feedback: each scored 0 useful against ≥3 noise, all
  bare version-bump / CI-build tags. The 3-dimension score could not separate
  them (useful and noise items averaged the same relevance), so the source list
  — not a threshold — was the only lever. Majors still arrive via HN.
- `RUBRIC_VERSION` bumped `triage-v1` → `triage-v2`: the keep-rule now names the
  `thresholds` config keys instead of hardcoding the numeric bar, so tuning
  `interests.yaml` can no longer silently contradict the prompt.

### Fixed
- Empty digest for non-UTC operators: the digest day is now resolved in the
  configured timezone via a DST-correct half-open UTC window, instead of a naive
  UTC date prefix that hid a UTC+10 reader's items under the prior date.
- Pre-release hardening: `Ctrl-C` no longer swallowed mid-run, `GITHUB_PAT`-style
  env-var *names* accepted while pasted tokens are still rejected, and the dead
  `Secrets` config class removed.
