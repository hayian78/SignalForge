# CLAUDE.md — SignalForge: Architectural Rules

> **Mandatory reading for any AI coding assistant in this repo.** Every rule
> below is a hard constraint. If a proposed change conflicts with this
> document, the document wins — rework the change.

## Companion docs

Loaded on demand — read when relevant:

- `docs/DESIGN.md` — full design: pipeline stages, schema, phases, cost model. **The spec.** Read the relevant section before implementing any component.
- `.claude/conventions.md` — bash/git/tool conventions for AI assistants.
- `.claude/agents/code-reviewer.md` — architectural reviewer grounded in these rules; invoke after any non-trivial change.
- `.claude/agents/llm-cost-guard.md` — adversarial cost review; mandatory for any diff touching `llm.py`, prompts, or model selection.
- `.claude/commands/add-source.md` — `/add-source` workflow for extending `sources.yaml`.

## Maintaining this file (read before editing)

Loaded into **every** conversation; size is paid in context tokens every turn.

- ≤ **200 lines**. Past that, something moves out: subdirectory rules → nested `CLAUDE.md` in that directory; procedures → `docs/runbooks/`; assistant tool conventions → `.claude/conventions.md`; design rationale → `docs/DESIGN.md` (never duplicated here).
- New NEVER row → `See` pointer to the section holding the detail. Removing a rule → remove its NEVER row too.
- Don't restate what a companion doc says; one-line pointer instead.

## 1. What SignalForge is

A **single-user, local-first AI engineering intelligence pipeline**: ingests
RSS / GitHub / arXiv / HN, dedupes, LLM-scores against `config/interests.yaml`,
and writes daily/weekly/monthly reports into an Obsidian-compatible markdown
vault. Stack: **Python 3.12+, uv, typer, pydantic v2, SQLite (no ORM),
Anthropic SDK**. One package, one process, one database, cron-driven.
Full design: `docs/DESIGN.md`.

Current phase: **Phase 0 — prove the loop** (see DESIGN §16). Do not build
Phase 2/3 components (embeddings, clustering, MCP server, impact engine)
until the earlier phase's acceptance gate is met. When in doubt, build less.

## 2. Module boundaries (the core invariant)

Strict responsibilities — DESIGN §4. Violations are architectural regressions:

- `ingest/` **never calls an LLM** and never imports `llm.py`.
- `score/` and `synth/` **never make HTTP calls to sources** — they operate on stored items only.
- `report/` only reads the DB and writes markdown to the vault.
- `llm.py` is the **only** module that imports the `anthropic` SDK. Budget accounting, prompt caching, batching, and model selection live there and nowhere else.
- Deterministic vs LLM split (DESIGN §8): fetching, parsing, dedup, clustering math, scheduling, template assembly are plain Python. LLMs only for judgment (triage, scoring, labeling, narrative). Never "solve" a parsing/normalization problem by throwing an LLM at it.

## 3. Data & database

- SQLite via stdlib `sqlite3` + thin `db.py`. **No ORM.** WAL mode. Schema changes only through `db.py` migrations.
- **Idempotency is non-negotiable** (DESIGN principle 6): ingest upserts on `UNIQUE(canonical_url)` / `UNIQUE(source_id, external_id)`; scoring skips already-scored items; report generation overwrites the same file. Running anything twice must produce zero duplicates and zero double-spend.
- Every score row carries `rubric_version` and `model`. Changing a scoring prompt = bumping the rubric version constant in `score/rubrics.py`.
- Every run writes a `runs` row with token counts and per-source errors. No silent runs.
- The DB is regenerable plumbing; **the vault is the product**. Never treat vault files as disposable.

## 4. Config is data, not code

- Sources, interests, taxonomy, and scoring thresholds live in `config/*.yaml`, validated by pydantic models in `config.py`. ❌ Never hardcode a source URL, keyword list, or threshold in Python.
- Tuning relevance = editing `interests.yaml` + `mark useful/noise` feedback — never ad-hoc prompt edits.

## 5. Vault rules

- Reports/notes are jinja2 templates; the LLM writes only clearly-marked narrative blocks.
- **Citations mandatory:** no synthesized claim renders without at least one stored `item.url` behind it. This is the structural defense against confabulation — do not weaken it for convenience.
- Vault files are git-committed by `report/writer.py`. ❌ Never bulk-delete or rewrite vault history; it is the user's knowledge base.

## 6. LLM usage & cost (the money rules)

DESIGN §8. Target ≈ $5–10/month; $30 is the alarm threshold.

- Triage/scoring: `claude-haiku-4-5`, **Batches API**, structured outputs, batched ~25 items/request, on **titles + summaries only**. Full content is fetched lazily for top-N survivors only — never feed full content to triage.
- Synthesis/impact: `claude-opus-4-8`, prompt-cached stable prefix (rubric + interests + taxonomy), items after the breakpoint. ❌ No timestamps, run IDs, or other volatile data in the cached prefix.
- Every call goes through `llm.py`, which records tokens into `runs`.
- Any diff touching `llm.py`, prompts, model choice, or batching gets an `llm-cost-guard` review before merge (see agent file).

## 7. Failure isolation

- Per-source `try/except`; one broken source never kills a run. Errors captured into `runs.errors` and surfaced in the next digest — the reports are the monitoring channel.
- HTTP: `httpx` with conditional GET (ETag/If-Modified-Since), `tenacity` retries honoring `Retry-After`, per-source politeness limits (arXiv: 3s delay). Raw payloads archived to `data/http_cache/`.
- ❌ Never swallow an exception without recording it to `runs.errors` or logging it.

## 8. Testing

- `pytest`. Unit tests use **recorded HTTP fixtures via `respx`** — never live network calls. Golden-file tests for the normalizer and report templates.
- ❌ Never call the real Anthropic API in tests — fake `llm.py` at its boundary.
- Every ingestor gets tests with realistic captured payloads in `tests/fixtures/`, including malformed-feed and duplicate-item cases.
- Idempotency has explicit tests: run twice, assert identical DB state.

## 9. Code style

- Python 3.12+, `uv`-managed. Type hints on all signatures (no bare `Any` without a comment). Ruff (lint + format) and mypy strict, zero errors.
- CLI via `typer`, models via pydantic v2, templates via jinja2. No LangChain/LangGraph — the pipeline is deterministic Python (DESIGN §15).
- Logging: stdlib `logging` with contextual fields (`source_id`, `run_id`, `item_id`). Never `print()` outside the typer CLI's user-facing output.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`). One logical change per commit.

## 10. The NEVER table — quick reference

| # | Rule | See |
|---|---|---|
| 1 | NEVER import the `anthropic` SDK outside `llm.py` | §2 |
| 2 | NEVER call an LLM from `ingest/`, or make source HTTP calls from `score/`/`synth/` | §2 |
| 3 | NEVER use an LLM for work a deterministic function can do | §2 |
| 4 | NEVER write a non-idempotent pipeline step | §3 |
| 5 | NEVER change a scoring prompt without bumping `rubric_version` | §3 |
| 6 | NEVER hardcode sources, keywords, or thresholds in Python — YAML config only | §4 |
| 7 | NEVER emit a synthesized claim without a stored item URL citation | §5 |
| 8 | NEVER bulk-delete or rewrite vault history | §5 |
| 9 | NEVER send full item content to triage — titles + summaries only | §6 |
| 10 | NEVER put volatile data (timestamps, run IDs) in a prompt-cached prefix | §6 |
| 11 | NEVER make an LLM call that bypasses `llm.py` token accounting | §6 |
| 12 | NEVER let one source's failure abort a run | §7 |
| 13 | NEVER hit live networks or the real Anthropic API in tests | §8 |
| 14 | NEVER add an ORM, orchestration framework, or server the design explicitly rejected | §9, DESIGN §15 |
| 15 | NEVER build ahead of the current phase gate | §1, DESIGN §16 |
| 16 | NEVER commit `.env` / API keys; never log secrets | `.claude/conventions.md` |
