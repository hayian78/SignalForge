# SignalForge

An AI engineering intelligence platform — a Bloomberg Terminal for AI engineers, not a news aggregator.

SignalForge ingests from high-signal AI engineering sources, strips duplicates and hype, scores what
remains against your interests and active projects, and writes daily/weekly/monthly intelligence
reports into an Obsidian-compatible markdown vault.

It answers **"what changed that matters to me"**, not "what happened".

## Status

**Phase 1 — the weekly question.** Phase 0's gate was met 2026-07-23. Later phases stay gated on the
earlier ones being *used* — not merely built.

**Built**
- [x] Ingest: RSS + GitHub releases + Hacker News → SQLite (per-source isolation, conditional GET)
- [x] Normalize + exact dedup (idempotent upserts)
- [x] Batched Haiku triage + 3-dimension scoring, on titles + summaries only
- [x] Daily digest → Obsidian vault, with per-source / per-repo crowding limits
- [x] Timezone-aware day boundary (UTC storage, configurable reader locale)
- [x] Cron installed (06:00 daily `signalforge daily`)
- [x] Adaptive source curation: weekly scout, digest-based approval
- [x] Daily digest mirrored to email, read-only (DESIGN §13.2 — shipped ahead of its phase, deliberately)
- [x] Daily two-presenter podcast, scripted from the day's top items and published as a private RSS feed (DESIGN §13.3 — a second, harder recorded exception)

**Remaining for the Phase 1 gate**
- [ ] Weekly Intelligence Brief — *the product*, not yet built
- [ ] Four consecutive Sunday briefs that answer the primary question
- [ ] ≥ 80% of brief items rated `useful` or better

Progress is logged in [`CHANGELOG.md`](CHANGELOG.md); the full roadmap lives in
[DESIGN §16](docs/DESIGN.md#16-roadmap).

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | The spec: pipeline stages, schema, phases, cost model, roadmap |
| [`CLAUDE.md`](CLAUDE.md) | Architectural rules — binding constraints for humans and AI assistants alike |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                # install dependencies
cp .env.example .env                                   # then fill in your keys
cp config/settings.yaml.example config/settings.yaml   # then set your timezone / vault path
```

Secrets live in `.env` (never committed):

- `ANTHROPIC_API_KEY` — triage and synthesis
- `GITHUB_TOKEN` — raises the GitHub API limit to 5k req/hr
- `OPENROUTER_API_KEY` — podcast TTS (only if `delivery.podcast` is configured)
- `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` — podcast feed/audio storage (same)

Machine-local settings live in `config/settings.yaml` (also gitignored; the
committed `config/settings.yaml.example` is the template): your `timezone`, and
an optional `vault_dir` to write digests straight into an Obsidian vault — e.g.
a WSL pipeline targeting a Windows vault at `/mnt/c/Users/<you>/Obsidian/...`.

## Usage

```bash
uv run signalforge ingest    # fetch from all configured sources into SQLite
uv run signalforge score     # batched Haiku triage + scoring of unscored items
uv run signalforge digest    # render today's Daily Digest into <vault_dir>/daily/
uv run signalforge podcast   # write today's episode script and publish it, if configured
uv run signalforge weekly    # the Sunday brief for the seven days before it (Opus; --dry-run is free)
uv run signalforge mark      # record how an item landed: useful | noise | exceptional | missed
uv run signalforge daily     # curate apply -> ingest -> score -> digest (cron 19:00, --no-podcast)
uv run signalforge status    # last-run health, per-source freshness, token + TTS spend

uv run signalforge curate run     # weekly source scout (paid; Fri 05:30)

uv run signalforge deliver test   # send one sample email, to prove the channel works
```

## Reading it away from the desk

The vault is the product, and it stays that way — but a digest you cannot read on
your phone is a digest you do not read. `digest` can mirror each morning's file to
an outbound channel once one is configured in `settings.yaml` (see
`settings.yaml.example`, and DESIGN §13.2 for why this shipped ahead of its phase).

Email is the one channel today. It is **read-only** by design:

- The vault file is written first and always; the email is a mirror of a file that
  already exists, never a substitute for it. A dead mail provider is a recorded
  error, not a failed run.
- Marks and source approvals still round-trip through vault markdown only, so the
  email carries no checkboxes. It carries a count of the decisions waiting and the
  filename to open — the nudge that keeps the loop closed.
- One send per digest, ever. Re-rendering a date rewrites the file and mails
  nothing (`--resend` overrides); a digest older than a day is never mailed at all,
  even if the database is deleted and rebuilt.

Costs nothing in tokens: no LLM call is involved.

## Listening to it

A second, optional channel: a daily two-presenter audio show, scripted by Opus from
the day's top items and published as a private RSS feed on Cloudflare R2 (DESIGN
§13.3 — a second recorded exception, and a harder one than email's; see that section
before enabling this). Off by default — nothing below is required to use SignalForge.

**Prerequisites**, beyond the `.env` keys above:

- **`ffmpeg` on `PATH`.** The per-line synthesis fallback stitches each spoken turn
  into one mp3 with `ffmpeg -f concat`. `apt install ffmpeg` / `brew install ffmpeg`.
  Checked before any TTS spend — a missing binary is a config error, not a failed run.
- **A Cloudflare R2 bucket**, public-read at an *unguessable* path (that path is the
  feed's only access control — do not use a short or guessable prefix). Create a
  bucket, an R2 API token scoped to it (access key + secret key), and enable public
  access at a `pub-*.r2.dev` URL or a custom domain.
- **An OpenRouter account** for TTS. `hexgrad/kokoro-82m` is the cheap, per-character
  model this channel's synthesis fallback was built and priced against.

**Enable it** by uncommenting `delivery.podcast:` in `config/settings.yaml` (see
`settings.yaml.example` for every field) and setting `interests.yaml`'s
`thresholds.podcast_top_n` — both are required together; either being unset is a
clean no-op, not an error.

**Subscribe on your phone**: run `signalforge podcast` (or wait for the next 06:00
`daily`), then open your R2 public URL + `/feed.xml` (e.g.
`https://pub-xxxxxxxx.r2.dev/some-unguessable-prefix/feed.xml`) in any podcast app's
"add by URL" / "add a show by RSS feed" option — Apple Podcasts, Pocket Casts, and
Overcast all support this on iOS; most Android podcast apps do too.

Same read-only invariants as email: the vault script is written first and always,
the feed cannot write back into SignalForge, and a dead TTS/R2 provider is a
recorded error, never a failed run. Unlike email, this one does spend real money —
`signalforge status` prices `runs.tts_characters` at your configured TTS model, and
DESIGN §8 tracks the honest paper total against the $50 alarm.

## Configuration

Sources, interests, and thresholds are **data, not code** — adding a blog is a YAML edit, never a
Python change.

| File | Purpose |
|---|---|
| `config/sources.yaml` | What to ingest |
| `config/interests.yaml` | Priorities, ignores, learning goals, scoring thresholds |
| `config/taxonomy.yaml` | Topic tree for the keyword tagger — *staged, no runtime effect yet* (DESIGN §10) |
| `config/settings.yaml` | Machine-local: timezone, vault output path, delivery channels (gitignored; see `.example`) |

Tuning relevance means editing `interests.yaml` and marking items useful/noise — never editing
prompts ad hoc.

## Layout

```
config/     YAML config — what to ingest, what you care about
src/        the pipeline: ingest → enrich → score → synth → report → deliver
vault/      frozen pre-`vault_dir` digests; the live vault is wherever settings.yaml points
data/       SQLite + HTTP cache (gitignored, regenerable)
tests/      pytest, with recorded HTTP fixtures — never live network
```

The database is regenerable plumbing; **the vault is the product**. If the DB burned down, the vault
survives; if the vault burned down, the DB could largely regenerate it.

## Development

```bash
uv run pytest              # recorded fixtures only, no live network
uv run ruff check src tests
uv run mypy src tests      # strict
```

Cost discipline is a design constraint, not an afterthought: target ≈ $5–10/month, $50 is the alarm
threshold (raised from $30 on 2026-08-16). Triage runs on titles and summaries only, batched, via
the cheapest capable model. See DESIGN §8.
