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

### Changed
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
