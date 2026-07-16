# SignalForge

An AI engineering intelligence platform — a Bloomberg Terminal for AI engineers, not a news aggregator.

SignalForge ingests from high-signal AI engineering sources, strips duplicates and hype, scores what
remains against your interests and active projects, and writes daily/weekly/monthly intelligence
reports into an Obsidian-compatible markdown vault.

It answers **"what changed that matters to me"**, not "what happened".

## Status

**Phase 0 — prove the loop.** RSS + GitHub releases + Hacker News → normalize → exact dedup →
batched Haiku triage → daily digest in the vault, via cron.

The Phase 0 acceptance gate (DESIGN §16): you read the digest five mornings straight and it saved
you time, and a double-run produces zero duplicates. Later phases are gated on the earlier ones
being *used* — not merely built.

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | The spec: pipeline stages, schema, phases, cost model, roadmap |
| [`CLAUDE.md`](CLAUDE.md) | Architectural rules — binding constraints for humans and AI assistants alike |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # install dependencies
cp .env.example .env     # then fill in your keys
```

Secrets live in `.env` (never committed):

- `ANTHROPIC_API_KEY` — triage and synthesis
- `GITHUB_TOKEN` — raises the GitHub API limit to 5k req/hr

## Usage

```bash
uv run signalforge ingest    # fetch from all configured sources into SQLite
uv run signalforge score     # batched Haiku triage + scoring of unscored items
uv run signalforge digest    # render today's Daily Digest into vault/daily/
uv run signalforge daily     # ingest -> score -> digest, in one call (cron 06:00)
uv run signalforge status    # last-run health, per-source freshness, token spend
```

## Configuration

Sources, interests, and thresholds are **data, not code** — adding a blog is a YAML edit, never a
Python change.

| File | Purpose |
|---|---|
| `config/sources.yaml` | What to ingest |
| `config/interests.yaml` | Priorities, ignores, learning goals, scoring thresholds |

Tuning relevance means editing `interests.yaml` and marking items useful/noise — never editing
prompts ad hoc.

## Layout

```
config/     YAML config — what to ingest, what you care about
src/        the pipeline: ingest → enrich → score → synth → report
vault/      the Obsidian vault — THE PRODUCT (git-tracked)
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

Cost discipline is a design constraint, not an afterthought: target ≈ $5–10/month, $30 is the alarm
threshold. Triage runs on titles and summaries only, batched, via the cheapest capable model. See
DESIGN §8.
