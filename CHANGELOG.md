# Changelog

All notable changes to SignalForge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No versions are
tagged yet — the project is in **Phase 0** (see [DESIGN §16](docs/DESIGN.md#16-roadmap)),
so everything below sits under *Unreleased* until the Phase 0 acceptance gate
is met.

## [Unreleased]

### Added
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
