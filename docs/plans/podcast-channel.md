# Podcast channel — daily two-presenter audio show

> **Execution doc.** Written 2026-08-07 for a future session to implement. Do not start
> before reading CLAUDE.md and DESIGN §4, §13.2, §16. Stages are strictly ordered; each
> stage ends with a blocking subagent review and hard numeric gates. Do not proceed past
> a failed gate. Stop-and-ask points are marked ⛔.

## Decisions already made (do not relitigate)

- **Format:** two-presenter dialogue show (~8–12 min/day), script written by Opus from the
  day's top-N kept articles. Not a summary read-out, not a verbatim reading.
- **TTS:** hosted, via **OpenRouter `/api/v1/audio/speech`** (OpenAI-compatible). Default
  model: Gemini Flash TTS (native 2-speaker). Cheap fallback: Kokoro-82M (per-line
  synthesis + ffmpeg concat). Local TTS ruled out (operator hardware is being downgraded).
  ElevenLabs ruled out on cost ($13–27/mo).
- **Phase gate:** DESIGN §13.2's tripwire is knowingly tripped. Operator decision
  (2026-08-07): record a **second exception** in DESIGN (§13.3) and build now. The
  exception record must be honest and must harden the tripwire (see Stage 7).

## Cost model (researched 2026-08-07)

Script ≈ 9k chars/day (≈ 270k/mo). Article input measured on 2026-08-06's real digest:
12 kept items, all fetched + trafilatura-extracted cleanly; longest 18,766 chars, total
~93k chars ≈ 25k Opus input tokens.

| Line | Monthly |
|---|---|
| Opus script (prompt-cached prefix) | ~$3 |
| Gemini Flash TTS via OpenRouter (~$0.015/min audio) | ~$4.50 |
| Kokoro via OpenRouter (fallback, ~$0.80/1M chars) | ~$0.25 |
| **New spend total** | **~$7–8** (pipeline stays under the $30 alarm) |

Caps (in code, not config — the `MAX_DELIVERY_AGE_DAYS` argument):
- Script cap **14,000 chars** (OpenRouter request cap is 15,000). Caps the *generated
  dialogue only* — never article content.
- Per-item stored content cap **25,000 chars** (2026-08-06 measurement: zero items clipped).
- `PODCAST_MONTHLY_CEILING_USD` in llm.py; TTS monthly character ceiling in deliver/tts.py.

## Architecture

```
top-N kept items (reuse digest ranking: db.get_digest_items / select_digest_items)
  → ingest/fullcontent.py   fetch + trafilatura extract where items.content IS NULL   [HTTP, deterministic, no LLM]
  → synth/podcast.py        Opus dialogue script via llm.py only — no HTTP             [judgment]
  → report/podcast.py       write vault/podcast/YYYY-MM-DD.md                          [vault first, always]
  → deliver/podcast.py      TTS → data/audio/YYYY-MM-DD.mp3 → R2 upload → feed.xml → prune
```

Boundary placements (all follow CLAUDE.md §2):
- `ingest/fullcontent.py` — the long-designed lazy deep-read (CLAUDE.md §6), bounded by
  the caller's top-N slice, never keep-count. Uses the existing `HttpFetcher` in
  `ingest/base.py` (conditional GET, tenacity, politeness, http_cache archive).
- `synth/` — activates the package DESIGN §4 reserves. Calls `llm.py` only.
- `deliver/tts.py` — sole OpenRouter caller. `deliver/storage.py` — minimal httpx SigV4
  S3 client for Cloudflare R2 (no boto3). `deliver/podcast.py` — the channel.
- Delivery = **private RSS podcast feed** on R2 under an unguessable prefix; any phone
  podcast app subscribes. Feed is read-only → never an input surface (§13.2 invariant).
  `feedback.py` globs only `vault/daily/*.md`, so `vault/podcast/` is structurally inert
  as an input surface; the template still renders no checkboxes.
- Audio lives in `data/audio/` (regenerable plumbing). The script markdown in
  `vault/podcast/` is the product and is never deleted.

Structural rule enforcement:
- **NEVER 7 (citations):** the model cites *item IDs, not URLs*. Deterministic code maps
  IDs → stored `items.url`; a segment citing an unknown ID is dropped and the drop is
  recorded to `runs.errors`. The template renders `Sources:` lists from DB URLs only.
- **NEVER 17 (flatten):** every dialogue turn and every item title passes
  `models.flatten_to_single_line` before rendering.
- **NEVER 11 analog:** TTS spend recorded in a new `runs.tts_characters` column
  (precedent: `server_tool_requests`, db.py:209-213).

## Idempotency levers (NEVER 4)

| Step | Lever |
|---|---|
| Deep-read | `UPDATE ... WHERE content IS NULL` — re-run fetches 0 pages, writes 0 rows |
| Script | vault file exists → LLM skipped (`--force-script` overrides), like "scoring skips already-scored" |
| Audio | local mp3 exists → TTS skipped; a failed upload retries free tomorrow |
| Publish | `deliveries` unique index (channel="podcast", report_kind="podcast", date) + `is_within_delivery_window`; `record_delivery` only after the feed upload succeeds |
| Prune | keep-set derived from `vault/podcast/` ∩ retention count; second run deletes nothing |

---

## Stages

### Stage 1 — Migration 5 + config schema
**Files:** `src/signalforge/db.py`, `src/signalforge/config.py`, `config/settings.yaml.example`
- Migration 5: `ALTER TABLE runs ADD COLUMN tts_characters INTEGER DEFAULT 0`, with a
  comment mirroring migration 3's rationale (per-character billing can't live in the token
  columns; invisible spend is the failure NEVER 11 exists to prevent). Extend
  `start_run`/`finish_run` to carry it (default 0). No `episodes` table — episode set is
  derivable from vault ∩ data/audio, publish-once guard is the `deliveries` index.
- `Thresholds.podcast_top_n: int | None` (interests.yaml, beside `daily_max_items`;
  `None` = feature off).
- `PodcastChannelConfig(_StrictModel)` + `DeliveryConfig.podcast: PodcastChannelConfig | None`
  (the docstring at config.py:442 mandates exactly this shape). Fields: `enabled` (required),
  `tts_api_key_env` (env-var name only; `_reject_inline_secret` for `sk-or-` keys),
  `tts_model`, `voice_a`, `voice_b`, `presenter_a`, `presenter_b`, `r2_endpoint`,
  `r2_bucket`, `r2_access_key_env`, `r2_secret_key_env`, `public_base_url`,
  `retention_episodes` (ge=1, le=90), `feed_title`.

**Review:** `code-reviewer` subagent (blocking).
**Gates:** mypy strict 0, ruff 0, all tests green; migrate-twice test (fresh DB → migrate
→ migrate → `PRAGMA user_version` = 5 both times, no error); inline-secret rejection test.

### Stage 2 — Deep-read fetch
**Files:** `src/signalforge/ingest/fullcontent.py`, `pyproject.toml` (+ `trafilatura`; add
mypy override if it ships no stubs), `tests/fixtures/` (article HTML, paywall stub, 404,
non-HTML).
- `fetch_full_content(fetcher, conn, items) -> FullContentResult`: for each item with
  `content IS NULL`, GET through `ingest/base.py`'s `HttpFetcher`, `trafilatura.extract`,
  truncate at 25,000 chars, `UPDATE items SET content=? WHERE id=? AND content IS NULL`.
- Per-item try/except → `runs.errors` records (§7). Extraction failure degrades that item
  to title+summary downstream; never fails the episode.

**Review:** `code-reviewer`.
**Gates:** run-twice test: second run performs **0 HTTP calls** (respx call count) and
**0 row deltas**; golden extraction test (fixture HTML → expected text).

### Stage 3 — Script generation ⚠️ llm-cost-guard MANDATORY
**Files:** `src/signalforge/llm.py`, `src/signalforge/synth/__init__.py`,
`src/signalforge/synth/podcast.py`
- llm.py: `PODCAST_MODEL` (Opus, same family as scout), `PODCAST_MAX_TOKENS` (≈8192),
  `PODCAST_MONTHLY_CEILING_USD` (~$6, worked worst case in docstring, mirroring
  `SCOUT_MONTHLY_CEILING_USD`), `run_podcast_script(...)`:
  - single non-batch call, structured output (schema like `_TRIAGE_OUTPUT_SCHEMA`):
    `{intro_turns, segments: [{item_ids, turns: [{speaker: "A"|"B", text}]}], outro_turns}`.
  - **prompt-cached stable prefix** = show-format brief + interests/taxonomy, with
    `cache_control` at the breakpoint (like triage, llm.py:355). Items, date, presenter
    names all AFTER the breakpoint. No timestamps/run IDs in the prefix (NEVER 10).
  - returns token counts for the runs row.
- synth/podcast.py: `PODCAST_SCRIPT_VERSION` (bump on any prompt change — NEVER 5 analog;
  recorded in vault frontmatter; `"podcast-v2"` as of the citation-discipline fix that
  constrained intro/outro to greetings-only during Stage 3 review). `build_script(...)`:
  validates total chars ≤ 14,000 (one "shorter" retry, then shrink over-length segments
  to a short teaser from the end before dropping any outright — operator call, 2026-08-07,
  after reading a live sample where the lowest-ranked stories got no airtime at all),
  drops segments citing unknown item IDs (recorded), flattens every turn.

**Reviews:** `llm-cost-guard` (blocking — PASS required) + `code-reviewer`.
**Gates:** fake `llm.py` at its boundary in all tests (NEVER 13, 0 real API calls);
prompt-prefix **byte-identity across two builds with different dates** (cache-hit
guarantee); 14,000-char cap enforced in test; confabulated-ID drop test; flatten test
with multi-line/control-char turn text.

⛔ **Stop-and-ask:** generate one real sample script (single live Opus call, operator
present), show it to the operator for a listen/read test before any TTS money is wired.

### Stage 4 — Vault script + template
**Files:** `src/signalforge/report/podcast.py`, `src/signalforge/report/templates/podcast.md.j2`
- `script_path(vault_dir, date) → vault/podcast/YYYY-MM-DD.md`; `write_script()` mirrors
  `write_digest` (overwrite, idempotent). `ScriptContext` dataclass consumed by deliver/
  so audio cannot disagree with the file (§13.2 same-context contract).
- Template: frontmatter (`kind: podcast-script`, `date`, `script_version`, `model`,
  `item_count`); `**<presenter>:** <one-line turn>` dialogue; per-segment `Sources:` list
  rendered from DB URLs; flattened titles; **no checkboxes**.
- Must round-trip: markdown → `ScriptContext` parser (used by Stage 6 to regenerate audio
  without re-paying Opus). Golden round-trip test locks the format.

**Review:** `code-reviewer`.
**Gates:** golden-file byte match; re-render byte-identical; segment with zero surviving
citations is not rendered; round-trip parse test.

### Stage 5 — TTS + storage + feed + retention
**Files:** `src/signalforge/deliver/tts.py`, `deliver/storage.py`, `deliver/podcast.py`,
`deliver/templates/podcast_feed.xml.j2`, `deliver/__init__.py` (docstring amendment),
fixtures (`tts_sample.mp3`, `podcast_feed_golden.xml`).
- tts.py: endpoint `Final` like `_RESEND_ENDPOINT`. **Build the per-line fallback first**
  (coalesce consecutive same-speaker turns → per-chunk synthesis voice_a/voice_b →
  `ffmpeg -f concat`); single-request multi-speaker (Gemini) is the optimization, since
  OpenRouter multi-speaker passthrough is unverified. `shutil.which("ffmpeg")` check →
  config-shaped error outcome, never a failed run. Retry policy copied from email.py:
  retryable statuses + connect-phase-only transport replay (a TTS POST is paid; replaying
  a ReadTimeout double-spends). `tts_spend_usd(chars, model)` + price table live HERE
  (precedent: `search_spend_usd` in curate/scout.py) — llm.py stays Anthropic-only.
  Hard refusal to synthesize >14,000 chars (double defense with Stage 3).
- storage.py: ~80-line SigV4 PUT/DELETE/LIST over httpx; deterministic keys
  `signalforge-YYYY-MM-DD.mp3`, `feed.xml`.
- Feed template: RSS 2.0 + iTunes namespace, XML autoescape ON, `<guid isPermaLink="false">`
  = date, `<pubDate>` in operator timezone, enclosure length from local file bytes.
- podcast.py channel guard chain, exactly the email order: freshness window →
  `db.delivery_exists` (resend overrules the log, never the window) → secrets present →
  mp3-exists skip → synthesize+write mp3 → upload mp3 → rebuild feed (last
  `retention_episodes` dates from vault ∩ local mp3) → upload feed → prune remote+local
  past retention → `record_delivery` (body_hash = sha256 of script markdown). Own entry
  point `deliver_podcast(...)` — NOT wired into `deliver_digest` (different vault file).
- deliver/__init__.py docstring: keep "nothing here costs a token" literally true, add
  "the podcast channel spends TTS dollars, recorded to runs.tts_characters".

**Reviews:** `code-reviewer` (+ `llm-cost-guard` over the price-table constants and
ceiling).
**Gates:** respx TTS tests (success / 429 with Retry-After / oversize refusal); SigV4
known-answer vectors; golden feed.xml; run-twice: **0 TTS chars, 0 uploads, 0 new rows**;
retention test: 3 remote episodes + retention 2 → exactly 1 DELETE, second run 0 DELETEs.

⛔ **Stop-and-ask:** first real episode end-to-end (`signalforge podcast`), operator
subscribes on phone and plays it, before Stage 6 wires it into `daily`.

### Stage 6 — CLI orchestration + spend surfacing
**Files:** `src/signalforge/cli.py`, `tests/test_daily_chain.py`
- New `podcast` command: `start_run(kind="podcast")` → resolve date like `digest` →
  select top-`podcast_top_n` via the existing digest ranking/crowding limits → deep-read
  → script (skip LLM if vault file exists, `--force-script` overrides) → `write_script`
  → only then `deliver_podcast` via a never-raising `_deliver_and_report`-style wrapper →
  `finish_run(tokens, tts_characters, errors)`. `podcast_top_n` unset / no kept items →
  clean "nothing to record" exit 0 with a runs row.
- `daily`: append podcast as the **fifth isolated step**, worst-exit-code wins.
- `status`: add `SUM(tts_characters)` + `$` line via `deliver.tts.tts_spend_usd`
  (shown even at $0.00).
- Optional `deliver test`-style `podcast test` subcommand (two-line sample, no DB writes).

**Review:** `code-reviewer`.
**Gates:** full-chain double-run with fakes at every boundary (fake llm.py, respx
sources/TTS/R2): second run **0 new rows, 0 tokens, 0 TTS chars, byte-identical vault
files**; exit-code isolation test (dead R2 → podcast step fails, digest still written).

### Stage 7 — Doc amendments
**Files:** `docs/DESIGN.md`, `CLAUDE.md`, `README.md`, `config/settings.yaml.example`
- DESIGN **§13.3 "Podcast channel — the second recorded exception"**, mirroring §13.2's
  structure: state plainly that the §13.2 tripwire was tripped and the operator chose to
  record and build (2026-08-07); what it costs (trafilatura dep, ffmpeg prerequisite on
  the fallback path, a new spend category, Weekly Brief still at zero lines); the
  inherited invariants (vault-first script; feed never an input surface; channel failure
  never fails the run; one publish per date ever); and a **hardened tripwire**: no third
  exception, period, before the Phase 1 gate; if the podcast goes unlistened 14
  consecutive days, disable the channel.
- §13.2 forward pointer; §16 Phase 1 note; §4 folder tree additions; §5 schema
  (`runs.tts_characters`); §8 cost table additions (honest note that paper total moves
  past the $5–10 target band; $30 alarm now includes TTS).
- CLAUDE.md: §2 bullets (synth/ calls llm.py only; deliver/tts.py + storage.py sole
  external speakers; ingest/fullcontent.py deterministic); §6 TTS-spend bullet; NEVER 15
  → "two recorded exceptions: §13.2, §13.3"; two new NEVER rows: "NEVER synthesize audio
  for a script not already written to the vault" and "NEVER make a TTS call that bypasses
  `runs.tts_characters` accounting". **CLAUDE.md must stay ≤ 200 lines.**
- README: ffmpeg prerequisite, R2 setup, phone subscription steps.
- settings.yaml.example: commented `delivery.podcast:` block.

**Review:** `code-reviewer` in docs-vs-code drift mode.
**Gates:** CLAUDE.md ≤ 200 lines (`wc -l`); no drift findings at blocker severity.

---

## Stage/gate summary

| # | Stage | Blocking reviews | Hard gates (all must pass) |
|---|---|---|---|
| 1 | Migration + config | code-reviewer | mypy 0 / ruff 0; migrate-twice = v5 |
| 2 | Deep-read | code-reviewer | 2nd run: 0 HTTP, 0 rows; golden extraction |
| 3 | Script LLM | **llm-cost-guard** + code-reviewer | prefix byte-identical across dates; ≤14k cap; 0 real API calls in tests |
| 4 | Vault script | code-reviewer | golden byte-match; round-trip parse |
| 5 | TTS/R2/feed | code-reviewer (+ cost-guard on prices) | 2nd run: 0 chars / 0 uploads / 0 rows; prune idempotent; SigV4 vectors |
| 6 | CLI | code-reviewer | chain double-run all-zeroes; exit-code matrix |
| 7 | Docs | code-reviewer (drift) | CLAUDE.md ≤ 200 lines |

⛔ Stop-and-ask points: after Stage 3 (operator reads a real sample script) and after
Stage 5 (operator plays the first real episode on their phone) — do not continue past
either without explicit operator sign-off.

## Verification (end of project)
1. `signalforge podcast --date <today>`: script in `vault/podcast/`, mp3 in `data/audio/`,
   feed live on R2, episode plays in a phone podcast app.
2. Re-run the same command: `status` TTS line unchanged, no uploads, no new rows.
3. After >N episodes: feed lists exactly N, R2 holds exactly N mp3s, vault holds all.

## Sources (TTS pricing research, 2026-08)
- OpenRouter TTS: https://openrouter.ai/docs/guides/overview/multimodal/tts ·
  https://openrouter.ai/collections/text-to-speech-models
- Gemini TTS pricing: https://www.nemovideo.com/blog/gemini-3-1-flash-tts-pricing
- Kokoro hosted: https://deepinfra.com/hexgrad/Kokoro-82M/api
- Fish Audio: https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits
- ElevenLabs: https://elevenlabs.io/pricing/api · comparison: https://texttolab.com/pricing
