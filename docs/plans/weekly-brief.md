# Weekly Intelligence Brief — Phase 1's acceptance gate

> **Execution doc.** Written 2026-08-08. Do not start before reading CLAUDE.md and
> DESIGN §8, §9, §11, §13, §14, §16. Stages are strictly ordered; each stage ends with a
> blocking subagent review and hard numeric gates. Do not proceed past a failed gate.
> Stop-and-ask points are marked ⛔.

## Context

Phase 0 is done and proven. Phase 1 has four items left: the **Weekly Intelligence Brief**,
vault git auto-commit, the taxonomy tagger, and awesome-list diffing. Only the brief is what
the Phase 1 gate actually measures (DESIGN §16):

> Four consecutive Sunday briefs that answer the primary question; ≥ 80% of brief items
> rated `useful` or better.

That gate takes four calendar weeks no matter how fast we build. So the brief ships first and
alone; the other three run alongside the gate window and are **out of scope here**.

Everything the brief needs already exists in some form. `thresholds.weekly_min_signal` /
`weekly_min_relevance` / `weekly_min_total` are defined in `config.py:293-295` and are currently
read only to *tell Haiku what bar to keep at* (`score/rubrics.py:72-76`) — nothing applies them
deterministically. `run_podcast_script` (`llm.py:1326`) is a working Opus-with-cached-prefix call
to copy. `feedback.checkbox_marker` and `harvest_marks` are the mark loop the gate's 80% is
measured through. This work mostly wires existing parts together.

**Scope note.** DESIGN §13's Reports row for the brief also lists impact-engine verdicts (P3),
trend deltas (P2) and watchlist changes (P2). None of those are built (NEVER 15). The brief ships
as: lead, clusters, item list with checkboxes, near-miss footer, ops footer.

---

## Key decisions

### 1. The window is the 7 days *before* the target Sunday

Not `[target-6, target]`. `get_digest_items` buckets on `scores.scored_at` (`db.py:683`), the
score pass runs at 19:00, and the brief runs Sun 07:00 (DESIGN §14). At 07:00 the target Sunday's
bucket is still empty — its score run is 12 hours away. A `[target-6, target]` window would waste
one day *and* leave the previous Sunday's items in no window at all, permanently.

```
window = [target - 7 days, target - 1 day]   inclusive, local
```

Consecutive Sundays then tile exactly: no gap, no overlap, every day already scored.

### 2. Selection is entirely deterministic, in this order

1. `db.get_digest_items(conn, start, end)` over the window — unchanged, it already takes an
   arbitrary `[start, end)`.
2. Drop uncitable (`not item.url`) — before selection, as `daily.py:500-503` does.
3. Apply `weekly_min_signal` / `weekly_min_relevance` / `weekly_min_total`. **This will bite**:
   Haiku keeps on "plausibly clears" the bar, so `3/3/2 = 8 < 10` is currently kept and shown in
   the daily digest. The brief is stricter than the digest by design (DESIGN §9) — say so in one
   footer line, or it reads as a bug.
4. Drop items marked `noise`. A small pure `drop_noise_marked()` in `report/weekly.py` using
   `feedback.highest_rung`.
5. `report.daily.select_digest_items(..., max_items=weekly_top_n, ...)` with the existing
   per-source / per-repo caps.

**Do not reuse `report/podcast.py::order_by_verdict` wholesale.** Its tier-lift promotes items
already marked `useful`/`exceptional` to the front. The Phase 1 gate is "≥80% of brief items rated
`useful` or better", and `feedback` rows carry no provenance — a tick made on Tuesday's digest is
stored identically to one made on the brief. Lifting already-marked items makes the brief
preferentially composed of items that already satisfy the metric. The gate would measure its own
input. Keep the `noise` drop, drop the lift.

Reusing `daily_max_per_source` over a 7-day window is deliberate (tight anti-crowding is what a
brief wants), but note it in the docstring — a reader will otherwise see a copy-paste bug.

### 3. No deep read, no full content

The synthesis call sends `(item_id, title, summary, reasoning)` — a plain 4-tuple, never an `Item`,
for the reason `llm.py:1350-1352` records (an `Item` would silently carry `items.content`, which
the podcast path has already populated for several of these rows). `DigestItem` already carries
`reasoning` (`db.py:658`) so **`db.py` needs no change**.

`scores.reasoning` is unbounded LLM prose — it gets its own byte cap in `llm.py`. The 320-char
trim at `daily.py:71` is display-only.

### 4. The LLM call mirrors `run_podcast_script` exactly

Single cached `system` block from `build_weekly_stable_prefix(interests)` (interests only — no
dates, no ids, NEVER 10). Everything volatile in the user turn. `client.messages.create` with
`output_config={"effort":..., "format": {"type": "json_schema", ...}}`, re-validated with pydantic.
Never raises once billed.

| Constant | Value | Why |
|---|---|---|
| `WEEKLY_MODEL` | `"claude-opus-5"` | Same as `SCOUT_MODEL`/`PODCAST_MODEL`. DESIGN §8 still says `claude-opus-4-8`; identical pricing, and a third Opus fragments the cache. Record the drift, don't perpetuate it. |
| `WEEKLY_EFFORT` | `"high"` | `PODCAST_EFFORT = "medium"` is justified by *cadence* (`llm.py:931-940`): a daily call pays 30×/month, a weekly one 4.33×. The scout is the applicable precedent, and `SCOUT_EFFORT` is `"high"`. |
| `WEEKLY_MAX_TOKENS` | `12288` | Sized for `effort: high`, not for the output. Nothing in this repo passes `thinking`, so adaptive thinking is on, bills as output, *and* counts against `max_tokens`. Measured here: the scout at `high` consumes 78% of its budget. At 78% of 12,288 that is ~9,580 thinking tokens, leaving ~2,700 for a structured output sized at ~1,400. An 8192 budget would leave ~400 tokens of margin, and every truncation is a fully-billed call producing nothing. |
| `WEEKLY_MAX_ITEMS` | `24` | Code-enforced cap that config can only lower (the `PODCAST_MAX_ITEMS` pattern). |
| `WEEKLY_MAX_ITEM_TITLE_BYTES` | `300` | Mirrors `llm.py:989`. |
| `WEEKLY_MAX_ITEM_SUMMARY_BYTES` | `500` | Mirrors `llm.py:974`. |
| `WEEKLY_MAX_ITEM_REASONING_BYTES` | `600` | New — unbounded LLM prose. |
| `WEEKLY_MONTHLY_CEILING_USD` | `3.50` | Arithmetic below — a bound, not a forecast. |

**Worst case, both halves priced** (the rule at DESIGN §8: input-only understated the scout by 35%):

Priced at the **pessimistic 1.0 bytes-per-token**, at the **ceiling** item count (never the
shipped `weekly_top_n`), with **both halves** billed, at **5 Sundays** (a month can have five),
and with **no cache breakpoint**:

```
per item at every cap, JSON-escaped: 300 + 500 + 600 + ~60 structure = 1,460 B
× 24 items + ~300 B preamble                              = 35,340 B
stable prefix (measured: the podcast's is 2,255 B)        =  2,255 B
                                                    total = 37,595 B → 37,595 tok

input   37,595 × $5/1e6    = $0.188
output  12,288 × $25/1e6   = $0.307
per call                   = $0.495
× 5 Sundays/month          = $2.48     ceiling $3.50 (29% headroom)
```

The headroom exists because there is no `content` field and no retry. It is deliberately
wider than the podcast's: `PODCAST_MONTHLY_CEILING_USD` history records a 340-byte prefix
addition eating a third of its margin, and this prefix has lead/cluster format and citation
discipline still to be written.

**Cost band obligation.** DESIGN §8 already carries a `weekly Opus synthesis ≈ $2.40` line in its
itemized estimate, so this is a *budgeted* consumer arriving under its own published number. But
the hard ceilings paragraph (`SCOUT $13` + `PODCAST $23` + `TTS $43.40`) is new spend and must be
updated, not absorbed — §8 says the next consumer "needs the band revisited rather than absorbed".
This diff owes: a new §8 table row, an updated ceiling paragraph, and an **`llm-cost-guard` review
before merge** (CLAUDE.md §6, non-optional).

### 4a. Corrections from the Stage 2 cost review (binding on Stage 4)

The `llm-cost-guard` review of Stage 2 re-derived the arithmetic above against the
repo's own conventions and measured real payloads. Five corrections, all binding:

- **Price 5 calls a month, not 4.33.** A month can contain five Sundays. The ceiling
  is a bound, not a forecast (DESIGN §8: "an estimate that assumes good behaviour is
  a forecast, not a bound"). At 5 calls and a pessimistic 1.0 B/token the worst case
  is $1.97 — 1.5% under a $2.00 ceiling, not 28%. **Set `WEEKLY_MONTHLY_CEILING_USD` to $3.50**, sized for the `effort: high` budget below.
- **No cache breakpoint.** One request per run, seven days apart, no retry — the
  ephemeral cache can never be read, so a breakpoint is pure write premium. This is
  the scout's own recorded reasoning at the identical cadence (DESIGN §8). The
  measured podcast prefix is 2,255 B ≈ 560 tokens, below Opus's cacheable minimum
  anyway, so the breakpoint would be a silent no-op. Drop it, and drop the 1.25×
  term from the ceiling arithmetic.
- **The caps are load-bearing and must be applied after JSON escaping.** Measured on
  a real top-24 payload: 2,427 B/item uncapped versus the 1,460 B/item the ceiling
  assumes. The binding field is `summary`, whose real maximum is 4,007 B against a
  500 B cap — roughly 8× over. Use `_truncate_utf8_json_safe` with
  `ensure_ascii=False` on all three fields. Capping raw characters instead of
  escaped bytes on any one field multiplies the input side by up to 6×.
- **Clamp the list actually sent** with `items[:WEEKLY_MAX_ITEMS]`, the `llm.py:1301`
  pattern. `weekly_top_n` is `ge=1` with no upper bound, so without the clamp a YAML
  edit to 200 moves real spend with every test green. And send
  `WeeklySelection.items` only — `near_misses` must never join the payload.
- **Cap `WeeklyBrief.clusters`** at `max_length=6`. Not for dollars (`max_tokens`
  bounds those) but for truncation: nothing else tells the model to budget its own
  output length.

**On the repeat-bill risk the review raised:** it does not materialise, because of a
Stage 2 decision made after the plan was written. The vault file is written on
*every* outcome, including a refused or unusable synthesis — that is Stage 2's own
gate ("the vault is useful even when synthesis produced nothing"). So the file guard
fires on the next invocation whether or not the call succeeded, and a mis-wired daily
invocation bills once per week, not thirty times. The only surviving repeat-bill case
is a billed call followed by a *failed vault write*, which the plan already accepts at
~$0.33. **Stage 6 must therefore write the file on the failure path too** — if it
ever writes only on success, the repeat-bill risk becomes real and a `runs` guard
becomes mandatory.

### 5. Output schema, and where the checkboxes go

```
lead:     up to 3 × {headline, why_it_matters, item_ids[]}
clusters: N × {title, narrative, item_ids[]}
```

`min_length=1, max_length=3` on leads — **not** a hard `minItems: 3`. On a thin week a hard 3
forces the model to manufacture a third "thing that mattered". "The 3 things that mattered" is the
headline, not a schema constraint.

**Checkboxes attach to items, not to narrative blocks.** Every selected item renders once in its own
`## Items` section with its `checkbox_marker` line; leads and clusters link into it. If checkboxes
rendered only under cited items, the model's citation behaviour would decide the gate's denominator.
This way `item_count` in frontmatter = `len(selected)` = the denominator, a pure function of
`(target_date, tz, db state, config)`.

**Citations get checked twice** (NEVER 7):
- Build time, in `synth/weekly.py` — any block citing an id outside `sent_item_ids` is dropped whole
  and recorded. Mirrors `synth/podcast.py::_drop_unknown_segments` (`podcast.py:192-223`); never
  touches the DB.
- Render time, in `report/weekly.py` — a block whose citations all fail to resolve to a stored
  `item.url` is dropped. Mirrors `report/podcast.py::_story_segment` (`podcast.py:340-367`). Only
  `report/` can resolve a URL, so build-time checking alone is not enough.

**Flattening has a `min_length=1` trap** that `synth/podcast.py:226-269` exists to handle: a model
string of `"   "` passes `Field(min_length=1)`, then `flatten_to_single_line` reduces it to `""`,
which raises `ValidationError` on reconstruction. Flatten, drop-if-empty, then drop the block if
nothing survives.

`WEEKLY_BRIEF_VERSION = "weekly-v1"` in `synth/weekly.py` — the `rubric_version` analogue
(NEVER 5), bumped on any stable-prefix or schema change, rendered into frontmatter.

### 6. `harvest_marks` must scan `weekly/`

Without it the gate cannot be measured at all — and worse, `--force` on a brief the operator has
already ticked destroys the marks, since the DB rows only exist if something read the file first.

```python
HARVEST_DIRS: Final = ("daily", "weekly")   # podcast/ deliberately absent — no checkboxes there
```

then one loop change in `harvest_marks` (`feedback.py:197`, `:209`); the body is untouched.
`Path.glob` on a missing directory yields nothing rather than raising, so a vault without
`weekly/` is a silent no-op and the existing empty-vault test stays green unchanged. Determinism
holds: fixed tuple order × `sorted()` inside each directory keeps the `created_at` microsecond
offsets deterministic.

### 7. Near-miss footer needs no new arithmetic

Because `rubrics.py:72-76` already tells triage to keep on "plausibly clears the weekly bar", the
near-miss population is definitionally: **kept, citable, not `noise`-marked items in the window
that failed the `weekly_min_*` gate — the top `weekly_near_miss_n` in existing rank order.** No
"just under" magic number in Python (NEVER 6), a pure sub-sequence of the same ranking, disjoint
from the body by construction.

Rendered as plain lines with item id, title, url and scores, **no checkbox** — `missed` is
off-ladder and CLI-only (DESIGN §11). Emitting `<!-- sf:item=N -->` without the `- [ ]` prefix is
safe: `MARK_RE` (`feedback.py:98-102`) anchors on the task-list prefix. The id makes
`signalforge mark <id> missed` one copy-paste.

### 8. Sunday snapping kills the double-spend class

`brief_path(...).is_file() and not force` is the right primitive — the vault is the durable
artifact, `runs` is regenerable plumbing. But it is not sufficient alone: if `--date` defaulted to
"today", a mis-wired `weekly` inside the daily chain would produce a *different filename every day*
and bill 30 calls/month while the file guard saw nothing.

Fix it in date resolution, not with a second guard:
- `--date` names the **target Sunday**; default = the most recent Sunday on or before today (in tz).
- A non-Sunday `--date` is a `typer.BadParameter` — refusing is louder than silently snapping.

Every day of the week then resolves to the same file: one billed call per week, six no-ops.
Strictly stronger than a `_too_soon`-style `runs` guard, and one function instead of a query plus a
`--force` semantic. **No runs guard.**

**Partial failure** (call billed, vault write failed): accept it, visibly. `finish_run` in a
`finally:` records the tokens regardless (NEVER 11); the next run sees no file and calls again, at
~$0.33. Do **not** add atomic write-and-rename — a torn write is still a file, so the guard would
skip the next call and leave a corrupt brief in the vault permanently. Print the error, record it
to `runs.errors`.

### 9. CLI ordering is load-bearing

```
start_run(RUN_KIND_WEEKLY)
  → harvest_marks            (non-fatal; before selection AND before overwrite)
  → select                   (deterministic)
  → if file exists and not --force: print, skip the call
  → else: build_brief (one Opus call) → write vault
  → finish_run(tokens, errors) in finally:
```

Harvest must precede selection (so `noise` applies) and overwrite (so `--force` doesn't destroy
ticks). One pass does both.

`--dry-run` makes **zero** LLM calls, writes nothing, and skips the harvest (`cli.py:883`'s rule —
a preview must never write ground-truth feedback rows). This differs from `curate run --dry-run`,
which still pays, because there the paid call *is* the thing being previewed. Here the thing worth
previewing — window counts, gated-out count, the top-N with scores, the near-misses — is entirely
deterministic.

`signalforge status` picks up the new kind with no change (`_render_last_runs` is `GROUP BY kind`).

### 10. Email delivery of the brief is deferred

Structurally unblocked — `deliveries.report_kind` and its unique index are already kind-parameterised
(`db.py:230`, `:248`). Not free, though: `render_email` takes a `DigestContext` specifically
(`email.py:158`), so wiring the brief later needs its own templates and context.

---

## Cuts — things the podcast has that the brief must not copy

| Cut | Why |
|---|---|
| The `shorter` retry, `_truncate_to_fit`, teasers, `WEEKLY_RETRY_MAX_TOKENS` | `PODCAST_MAX_SCRIPT_CHARS` exists because TTS bills per character and OpenRouter caps a request at 15k. A markdown file has no downstream cap. Cutting this halves the worst-case ceiling and removes ~80 lines. |
| A `parse_script`-style round trip | `report/podcast.py::parse_script` exists because TTS voices the *file*. Nothing ever reads the brief back. ~150 lines. |
| `CoverageGap`'s two-reason taxonomy | It distinguishes "fabricated citation" from "show ran long". Without truncation there is one reason. One `dropped_item_ids` tuple, one footer line. |
| Trend deltas / impact verdicts / watchlist | P2/P3 (NEVER 15), despite appearing in DESIGN §13's Reports row. |

---

## Stages

Strictly ordered. Each stage ends with its named review subagent (blocking — do not start the
next stage on a failed gate) and its hard gates. Stop-and-ask points are marked ⛔.

Stages 0–3 are deterministic and cost nothing. **Do not touch `llm.py` until the selection is
provably right and the vault file is provably useful without it.**

Every stage carries these three baseline gates in addition to the ones listed:
`uv run ruff check` = 0, `uv run ruff format --check` clean, `uv run mypy` = 0 errors
strict, `uv run pytest` = 0 failures. **Bare `mypy`, not `mypy src`** — `pyproject.toml`
sets `files = ["src", "tests"]`, so `mypy src` silently skips every test file.

### Stage 0 — Window generalization + config knobs

Extract `utc_range_window(first_local_date, last_local_date, tz)` in `report/daily.py`;
`utc_day_window` becomes a one-line delegation. Add `weekly_top_n: int | None = None` and
`weekly_near_miss_n: int | None = None` to `Thresholds` — optional, matching `podcast_top_n`
(`config.py:328`), so no existing `interests.yaml` or test fixture breaks. Ship
`weekly_top_n: 12`, `weekly_near_miss_n: 5` in `config/interests.yaml`.

**Review:** `code-reviewer` (blocking).
**Gates:** `utc_day_window(d, tz) == utc_range_window(d, d, tz)` for every existing case; the DST
tests at `tests/report/test_daily.py:788-873` pass **unmodified**; a 7-day window spanning a DST
transition is **not** 168h; `tests/test_config.py` bounds tests for both new knobs plus a
shipped-config assertion; **zero diff** in daily-digest output (re-render an existing golden).

### Stage 1 — Deterministic selection

`report/weekly.py` part 1: `weekly_window`, `passes_weekly_gate`, `drop_noise_marked`,
`select_weekly_items`, `near_miss_items`. Pure functions — no DB handle, no jinja, no LLM.

**Review:** `code-reviewer` (blocking) — specifically on NEVER 3 (no LLM doing arithmetic),
NEVER 6 (no threshold literals in Python), and the "filter, never reorder" property.
**Gates:** gate arithmetic drops `3/3/2`, keeps `3/3/4`; uncitable dropped **before** selection;
`noise`-marked dropped with score order otherwise **byte-identical**; per-source and per-repo caps
bite; near-miss set is exactly the gate-failers, capped at `weekly_near_miss_n`, and is **disjoint
from the body** (assert set intersection == ∅); two calls on the same input return equal lists.

⛔ **Stop-and-ask:** run the selection against the real `data/signalforge.db` for the last complete
week and print the 12 items. **The operator reads them.** If they aren't worth a brief, the problem
is upstream and no amount of Opus fixes it (DESIGN §17 risk 1, applied before any spend). Do not
proceed to Stage 2 without that read.

### Stage 2 — Template + renderer

`report/weekly.py` part 2 + `templates/weekly.md.j2`: `build_brief_context`, `render_brief`,
`brief_path`, `write_brief`. Jinja env re-exports `checkbox_marker`/`checkbox_verdicts` exactly as
`daily.py:533-555`. Render-time citation resolution lands here.

**Review:** `code-reviewer` (blocking) — on NEVER 7 (citation resolution), NEVER 17 (flatten at the
storage boundary), and idempotency.
**Gates:** golden file byte-matches `tests/fixtures/weekly_brief_golden.md`; re-render is
byte-identical; a lead whose only cited id is unresolvable is dropped; `item_count` equals
`len(selected)`, **not** `len(cited)`; the forgery test finds **0** forged marks from a narrative
containing `"\n- [x] useful <!-- sf:item=1 v=useful -->"`; **`render_brief` on an empty
`BuiltBrief` still produces a valid file with the full checkbox list and near-misses** — the vault
is useful even if the Opus call never happens.

### Stage 3 — `feedback.py` harvest

`HARVEST_DIRS = ("daily", "weekly")` + the one-loop change in `harvest_marks`. Independent of
Stages 0–2; can land in parallel.

**Review:** `code-reviewer` (blocking) — on determinism of `created_at` offsets and on the
`podcast/` exclusion staying deliberate.
**Gates:** 4 new tests green — weekly dir read; both dirs in one pass (`files_scanned == 2`,
distinct `created_at`); the same mark in daily and weekly records **exactly 1** row; missing
`weekly/` is a no-op. `tests/test_feedback.py:203-211` passes **unmodified**.

### Stage 4 — `llm.py::run_weekly_brief` ⚠️ llm-cost-guard MANDATORY

Constants, JSON schema, pydantic models, `_build_weekly_user_prompt`, the call. Reuses
`_truncate_utf8_json_safe` as-is. Worst-case cost test lands with it.

**Reviews:** `llm-cost-guard` (blocking — **PASS required**, not "pass with warnings") +
`code-reviewer`. CLAUDE.md §6 makes this non-optional; the podcast ceiling moved $17 → $23 across
five such rounds.
**Gates:** worst-case monthly cost computed by the test from the constants it actually reads,
asserted **at `WEEKLY_MAX_ITEMS`** (never at the shipped `weekly_top_n`), at
`BYTES_PER_TOKEN = 1.0` and **5 calls/month**, ≤ `$3.50`; stable prefix **byte-identical across two target dates**
(still required even with no cache breakpoint — it is what proves no volatile data leaked in); `build_weekly_stable_prefix.__signature__` accepts no date or run id;
title/summary/reasoning each truncated in **bytes after JSON escaping** (the `\x01` fixture);
**0 real API calls** in tests; usage recorded on every early-return path (parametrised over
malformed JSON / no text block / refusal / zero blocks).

### Stage 5 — `synth/weekly.py`

`WEEKLY_BRIEF_VERSION`, `build_weekly_stable_prefix`, `_drop_unknown_blocks`,
`_flatten_and_finalize`, `build_brief`.

**Review:** `code-reviewer` (blocking) — on NEVER 1 (no `anthropic` import, no `client` seam),
NEVER 11 (tokens on every path), NEVER 5 (version constant present and rendered).
**Gates:** unknown-id blocks dropped whole and recorded; a brief where **every** block cites an
unknown id returns `.brief is None` with `dropped_item_ids` preserved and tokens intact; a
whitespace-only narrative drops the cluster rather than raising `ValidationError`; `>3` leads
rejected by the model, `0` leads accepted; **tokens non-zero on all 6 failure modes**.

⛔ **Stop-and-ask:** one real Opus call on the Stage 1 item set. **The operator reads the raw JSON**
before any vault write is wired. Do not proceed to Stage 6 without that read.

### Stage 6 — `cli.py::weekly`

`RUN_KIND_WEEKLY`, Sunday resolution and refusal, `_select_weekly_items`, the ordering in §9.

**Review:** `code-reviewer` (blocking) — on the harvest-before-selection-and-overwrite ordering,
failure isolation, and the `finally:` accounting.
**Gates:** second run with the LLM seam replaced by `_must_not_be_called` exits **0** with
**0 calls** and a byte-identical file; `--force` **does** call; non-Sunday `--date` exits non-zero
with **0 calls**; `--dry-run` = **0 calls, 0 files written, 0 feedback rows**; `weekly_top_n` unset
⇒ clean no-op, **0 calls**; empty week ⇒ clean exit, **0 calls**, `runs` row still closed;
`--force` on a ticked brief preserves the feedback row; `tests/test_daily_chain.py` asserts `daily`
does **not** invoke `weekly`.

⛔ **Stop-and-ask:** run the live end-to-end sequence in *Verification* below against the real DB
and vault. **The operator reads the brief in Obsidian, ticks marks, and confirms `--force`
preserves them** before Stage 7 wires the cron entry.

### Stage 7 — Docs, cron, cost band

**DESIGN §8** — the existing "Weekly brief synthesis + impact engine" row must be **split**, not
edited: the impact engine is Phase 3 and unbuilt. The new row records `claude-opus-5`, one request
per run, **no retry and no prompt cache**, `effort: high` + json_schema, a
`(item_id, title, summary, reasoning)` payload that never carries `items.content`, three byte caps
applied after JSON escaping, and `WEEKLY_MAX_ITEMS = 24` as a ceiling config can only lower.

The §8 **cost-estimate line is wrong in both terms** and must be replaced rather than left: it reads
`4 × (80k in / 8k out) ≈ $2.40`, but real input is ~14 kB, and it double-counted the unbuilt impact
engine. Replacement ≈ **$1.10/month**, which moves the itemized total **$20.30 → $19.00**. State the
reduction; do not pocket it. Mark it unmeasured until four real Sundays exist.

The **hard-ceilings paragraph** gains a fourth entry: SCOUT $13.00 + PODCAST $23.00 + TTS $43.40 +
**WEEKLY $3.50 = $82.90** (was $79.40), recorded not absorbed. Name the guarding test rather than
restating the arithmetic, and record that none of the four ceilings accounts for the SDK's default
`max_retries=2` (see `WEEKLY_MONTHLY_CEILING_USD`'s docstring).

**Three recorded drifts, not one:** §8 names `claude-opus-4-8` (ships as `claude-opus-5`); §8 also
specifies `cache_control: ephemeral` for this row (deliberately absent — weekly cadence means an
ephemeral entry always expires unread); and §8's deterministic column claims clustering math, while
`WeeklyCluster` has the model both group and narrate, as a temporary exception until Phase 2's
embeddings exist.

**Also:** DESIGN §197's `weekly/2026-W29.md` path is wrong — briefs are written as
`weekly/<Sunday ISO date>.md`, which is what makes the Sunday-snap file guard work. DESIGN §13's
Reports row marked shipped; §16 Phase 1 checklist; §11 near-miss rule recorded. `CHANGELOG.md`;
README command list; crontab `signalforge weekly` Sun 07:00.

**Review:** `code-reviewer` in docs-vs-code drift mode (blocking) + `llm-cost-guard` over the §8
numbers.
**Gates:** `wc -l CLAUDE.md` ≤ 200; no blocker-severity drift findings; every number in §8 traceable
to a constant or a test that computes it (§8's "a number written in two places will disagree with
itself"); the crontab edit touches **only** signalforge's lines.

---

## Stage/gate summary

| # | Stage | Blocking reviews | Hard gates (all must pass) |
|---|---|---|---|
| 0 | Window + config | code-reviewer | DST tests unmodified; 7-day DST window ≠ 168h; 0 diff in daily output |
| 1 | Selection | code-reviewer | near-miss ∩ body = ∅; noise-drop preserves order byte-identically; deterministic across calls |
| 2 | Template + render | code-reviewer | golden byte-match; re-render byte-identical; 0 forged marks; empty-brief file still useful |
| 3 | Harvest | code-reviewer | daily+weekly same mark = 1 row; missing dir = no-op; existing test unmodified |
| 4 | `run_weekly_brief` | **llm-cost-guard (PASS)** + code-reviewer | worst case at the ceiling ≤ $3.50 at 1.0 B/tok and 5 calls/month; prefix byte-identical across dates; 0 real API calls |
| 5 | `synth/weekly.py` | code-reviewer | all-unknown-citations ⇒ `None` + tokens intact; tokens non-zero on all 6 failure modes |
| 6 | CLI | code-reviewer | 2nd run: 0 calls, byte-identical; `--dry-run`: 0 calls / 0 files / 0 rows; non-Sunday refused |
| 7 | Docs | code-reviewer (drift) + llm-cost-guard (§8) | CLAUDE.md ≤ 200 lines; every §8 number traceable to a constant or test |

⛔ **Stop-and-ask points — do not continue past any of these without the operator:**
after Stage 1 (operator reads the selected 12 items from the real DB), after Stage 5 (operator
reads a real Opus response as raw JSON), and after Stage 6 (operator reads the brief in Obsidian
and confirms marks survive `--force`).

**Merge gate for the feature:** all seven stage gates green, `llm-cost-guard` PASS on Stage 4,
zero blocker findings outstanding.

**Phase 1's acceptance gate (weeks later, not a merge gate):** four consecutive Sunday briefs,
≥ 80% of brief items rated `useful` or better — measured off the `feedback` table via
`highest_rung`, not `verdict = 'useful'`.

---

## Files

**New**
- `src/signalforge/report/weekly.py` — window, selection, near-miss, context, render, write
- `src/signalforge/report/templates/weekly.md.j2`
- `src/signalforge/synth/weekly.py` — stable prefix, citation cleaning, `build_brief`
- `tests/test_llm_weekly.py`, `tests/synth/test_weekly.py`, `tests/report/test_weekly.py`,
  `tests/test_weekly_cli.py`, `tests/fixtures/weekly_brief_golden.md`

**Modified**
- `src/signalforge/llm.py` — `WEEKLY_*` constants, schema, `run_weekly_brief`. Reuses
  `_truncate_utf8_json_safe` (`llm.py:1226`) as-is.
- `src/signalforge/report/daily.py` — extract `utc_range_window`; `select_digest_items` reused unchanged
- `src/signalforge/cli.py` — `RUN_KIND_WEEKLY`, `weekly` command, `_select_weekly_items`
- `src/signalforge/feedback.py` — `HARVEST_DIRS`
- `src/signalforge/config.py`, `config/interests.yaml` — two new thresholds
- `docs/DESIGN.md`, `CHANGELOG.md`, `README.md`

**Read as templates, not modified**
- `llm.py:1326` `run_podcast_script`; `llm.py:925-1138` the `PODCAST_*` constant block
- `synth/podcast.py:192-223` citation cleaning, `:226-269` flatten, `:400-427` `_empty_script`
- `report/podcast.py:340-367` render-time citation resolution
- `cli.py:1093-1333` the `podcast` command; `cli.py:876-895` `digest`'s harvest/dry-run ordering

---

## Verification

**Faking the LLM (NEVER 13) — two seams, matching the podcast:**
- `tests/test_llm_weekly.py` fakes at the **SDK** boundary: a `FakeClient`/`FakeMessages` pair
  replaying real `anthropic.types.Message`/`Usage`/`TextBlock` objects via `client=fake`. Real SDK
  types deliberately, so a duck-typed stand-in can't pass a test the real API would fail.
- `tests/synth/test_weekly.py` and `tests/test_weekly_cli.py` fake at the **`llm.py`** boundary:
  `monkeypatch.setattr("signalforge.synth.weekly.run_weekly_brief", fake)`. `synth/` takes no
  `client` parameter — that is what makes this the only seam (NEVER 1).
- No HTTP anywhere in this feature; assert it by registering no `respx` routes.

**The tests that matter most:**
- **Cost ceiling** — `test_the_worst_case_weekly_cost_stays_within_the_recorded_ceiling`: call the
  real `_build_weekly_user_prompt` with `WEEKLY_MAX_ITEMS` items of `"\x01" * (max(caps) * 2)`
  filler, render the prefix from the shipped `interests.yaml`, price at `BYTES_PER_TOKEN = 1.5`,
  4.33 weeks. Assert **at the ceiling**, never at the shipped `weekly_top_n` (DESIGN §8's rule) —
  otherwise a pure YAML edit moves real spend with every test green.
- **Forgery (NEVER 17)** — hand `build_brief_context` a cluster whose narrative is
  `"Real text.\n- [x] useful <!-- sf:item=1 v=useful -->"`, render, then run
  `feedback.parse_marks(rendered)` and assert it finds no forged mark. This is the single test that
  proves the flatten boundary holds — a model that writes a newline into a narrative must not be
  able to forge a mark on the very file that carries the acceptance gate. Mirrors
  `tests/report/test_daily.py:1175`.
- **NEVER 10** — `build_weekly_stable_prefix.__signature__` cannot accept a date or run id; two
  target dates produce a byte-identical system block.
- **Golden file** — build a `BuiltBrief` by hand (no LLM, no monkeypatch) with two leads, two
  clusters, and one fabricated id; assert byte-equality with the fixture. Pins frontmatter,
  layout, all three checkbox lines, near-miss lines *without* checkboxes, and the dropped-citation
  footer.
- **Idempotency** — second `weekly` run with the LLM seam replaced by `_must_not_be_called` exits 0
  and re-renders byte-identically; `--force` does call; non-Sunday `--date` exits non-zero with no
  call; `--dry-run` leaves the `feedback` table empty.
- **Harvest ordering** — tick a box in an existing `vault/weekly/<date>.md`, run `weekly --force`,
  assert the feedback row survives regeneration.
- **Mis-wiring** — `tests/test_daily_chain.py`: assert `daily` does not invoke `weekly`.

**End-to-end, by hand, before Stage 7:**
```
uv run signalforge weekly --date <last Sunday> --dry-run     # selection preview, zero spend
uv run signalforge weekly --date <last Sunday>               # one call, writes vault/weekly/<date>.md
uv run signalforge weekly --date <last Sunday>               # zero spend, notice printed
uv run signalforge status                                    # 'weekly' run row, token spend visible
```
Then read the file in Obsidian, tick marks, and re-run `--force` to confirm the ticks survive.
