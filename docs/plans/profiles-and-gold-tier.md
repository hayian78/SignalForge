# Plan — Gold verdict tier & multi-profile digests

| | |
|---|---|
| Status | Proposed — 2026-07-28. Not built. Revised after adversarial review. |
| Scope | Two independent features, sequenced together because they overlap on the daily template and the vault layout |
| Fold into | `docs/DESIGN.md` (§5 schema, §11 personalization, §13 reports, §14 scheduling) once built — this file is the working plan, not the spec |

Two felt gaps from a week of real use (DESIGN §17 risk 1 — features must map to
a gap in a real report):

- **A.** Some items are not merely `useful`, they are exceptional and worth
  remembering. There is no way to say so.
- **B.** Crypto and hobby interests exist alongside the AI/tech lens, and
  interleaving them into one digest is wrong. Profile count will grow — a
  second hobby, months out, must not dilute the first.

---

## Feature A — a third verdict rung

### What it is

`noise < useful < gold` — an **ordinal ladder inside the existing `feedback`
vocabulary**, not a separate saved/read-later store.

A gold mark is a judgment about whether the pipeline earned its keep, exactly
like `useful`. It is not personal workflow state: no lifecycle, no un-save, no
decay, no queue. That is what makes it fit the existing append-only table and
its `UNIQUE(item_id, verdict)` index with no structural change.

**The verdict string is not final.** `gold` is the working name. It is a wire
format — checkbox label, `sf:` HTML comment, `MARK_RE`, and every stored row —
so it gets chosen once, at implementation, and then never renamed (a rename
costs a migration plus historical vault files that stop parsing).

### Changes

| Area | Change |
|---|---|
| `feedback.py` | Extend `VERDICTS` and `_CHECKBOX_VERDICTS`; widen `MARK_RE`'s two verdict alternations. No new parsing logic — the marker format already generalizes. |
| `report/templates/daily.md.j2` | A third `checkbox_marker(...)` line per item. Watch the 60-second-read budget (DESIGN §13): 3 checkboxes × 15 items is 45 lines of affordance. Consider collapsing all three onto one line. |
| `cli.py` | `signalforge mark <id> gold` works as soon as `VERDICTS` includes it. |
| Aggregations | **Every read of `feedback` must reduce to the item's highest rung**, never `verdict = 'useful'`. |
| Tests | `test_feedback.py` round-trip, `test_cli_mark.py` verdict validation, and the daily golden files all change. |

### Two traps

**The gate metric.** Phase 1's acceptance gate is "≥ 80% of brief items rated
`useful`" (DESIGN §16). Once a top rung exists, marks migrate to it — nobody
ticks `useful` *and* `gold`. Any aggregation still testing `verdict = 'useful'`
deflates the gate precisely when the system is performing best.

**"One row per item" is not a schema property.** `ux_feedback_item_verdict` is
`UNIQUE(item_id, verdict)` (`db.py:138`), which happily stores `useful` *and*
`gold` for the same item — and the microsecond-offset logic in
`feedback.py:144-156` exists specifically to let dual verdicts persist. So the
ladder is a **convention enforced at read time**: every aggregation takes the
max rung. The stored-state renderer (below) must also decide which box shows
when two rows exist — it shows the max rung.

### Also fix: rendered state does not reflect stored state

`checkbox_marker` (`feedback.py:88`) always emits `- [ ]`, and `report/daily.py`
never reads the `feedback` table. Re-rendering **today's** digest therefore
clears the ticks visually — the DB row survives, but the vault stops showing
what was marked. Older digests are never re-rendered, so exposure is same-day
only, but that is exactly the case a manual re-run hits.

Tolerable for `useful`/`noise`. Corrosive for a mark made a handful of times a
month.

**Consequence to accept:** once state renders, a mis-ticked box is permanent.
Harvest only reads *checked* boxes, so there is no un-mark path in the table,
the CLI, or the harvest. Either accept it (a wrong mark is one noisy row in a
shrinkage-smoothed aggregate) or add `signalforge unmark` as a CLI-only escape.
Accepting is the default.

### Payoff beyond the shelf

Gold marks are the highest-value Phase 2 asset available. DESIGN §11 rotates
feedback exemplars into the scoring prompt, prioritising disagreements — a
gold-marked item that only scored 3/3/3 is the most informative training example
the system will ever produce: *"this is what a 5 looks like and you missed it."*

Not built now. Phase 2, in the prompt-cached prefix, rotating at most monthly
(NEVER 10), each rotation bumping the rubric version (NEVER 5).

### Cost

Zero. Nothing in Feature A touches `llm.py`, a prompt, or a model choice, so no
`llm-cost-guard` review is required — `code-reviewer` still is (see
[Review gates](#review-gates)).

### Explicitly deferred

- **Per-item tag placeholder in digests.** Inline Obsidian tags resolve to the
  *file*, so tagging one item in a 15-item digest makes the whole digest match
  every tag it contains. Per-item tagging only pays off once gold items get
  their own note files.
- **Per-item note files** (`<vault>/<profile>/gold/<slug>.md`, created once,
  never overwritten) — the Obsidian-idiomatic upgrade, and the honest early
  version of Phase 3 `insights/` notes. Revisit after a month of real marks.
- **Full-content archiving on gold.** A stronger warrant for a lazy
  full-content fetch than any score, and it kills link rot (a risk absent from
  DESIGN §17). Deterministic, no LLM cost — but it is a *source* HTTP fetch, so
  it belongs in `ingest/`, never `report/` or `score/` (CLAUDE.md §2).

---

## Feature B — profiles

### The problem in one line

Everything in SignalForge is plural except `interests.yaml`, which DESIGN §11
calls "the single place where 'relevant to me' is defined" and which is injected
as the prompt-cached prefix on every scoring call. One definition of relevance →
one ranking → one digest.

### The invariant that keeps it cheap

> **A source belongs to exactly one profile. An item inherits its source's
> profile. Scoring stays strictly 1:1 with items.**

The naive design scores every item against every profile, which breaks `scores`'
`item_id INTEGER PRIMARY KEY` **and** multiplies the triage bill by profile
count. Under the invariant, `scores` keeps its primary key and **triage spend
scales with sources added, not with profiles owned**.

(Synthesis does *not* scale that way — see the cost section. That distinction is
the single most important correction in this revision.)

### Profile granularity — per-entry where the config already supports it

`sources.yaml` is not uniformly shaped: `rss:` entries are objects that can
carry a field (`config.py:96-104`), but `github.releases` is a bare list of
slugs (`config.py:118`) and `hackernews:` is one monolithic block with a single
keyword list and a hardcoded `source_id = "hn"` (`config.py:160`,
`ingest/hackernews.py:42`).

Rather than rework the config model up front:

| Block | Profile granularity |
|---|---|
| `rss:` | **Per entry** — `profile:` on each source. |
| `github:` | **Block-level** — the whole block belongs to one profile. |
| `hackernews:` | **Block-level** — the whole block belongs to one profile. |

Both blocks are entirely tech today, so both get `profile: tech` and nothing
else changes. Per-entry granularity for GitHub, and splitting HN into
keyword-scoped per-profile queries, are **deferred until a profile actually
needs them** — that work is a config-model rework plus ingestor changes plus new
cache keys, and buying it speculatively is what turned "turn on crypto" into a
non-config-only step in the first draft.

Crypto therefore starts **RSS-only**, which keeps step 5 a genuine YAML edit.

### The non-overlap constraint (decided)

> **No profile's `ignore.topics` may name a term that appears in another
> profile's `interests` or `priority_topics`.** Violation is a load error.

Without this, profiles actively suppress each other. Today's tech lens carries
`ignore.topics: [crypto, web3]` (`interests.yaml:56-58`), and the triage rubric
maps ignored topics to relevance 1 with an explicit kill bias
(`rubrics.py:58-73`). A crypto story reaching the DB via a *tech* source is
scored under the tech lens, killed, and — because `UNIQUE(canonical_url)`
(`db.py:85`) merges the crypto source's later fetch into the tech-owned row and
scoring is 1:1 — is permanently invisible to the crypto digest too. That is not
a routing wrinkle; it is engineered suppression of exactly the content the new
profile exists to surface.

**What tech loses.** Its blanket crypto ignore. Crypto stories reaching the tech
lens now get scored on merit and should land at low relevance, then fail the
digest thresholds. There is precedent for exactly this move in the current
config: `model-release-hype` was deliberately removed from `ignore.topics` so
the rubric's `signal` dimension could do the work instead of a blunt topic
filter. Same reasoning, same shape. Watch the first week's tech digests for
crypto bleed; reverting is a one-line YAML edit if it goes wrong.

**What the constraint does not fix.** It removes the active kill, not the
routing. A crypto story carried first by a tech source is still owned by the
tech profile and still will not appear in the crypto digest — a *passive* miss
rather than an engineered one. Two mitigations, both free:

1. **Ingest order decides ownership.** `UNIQUE(canonical_url)` means first
   writer wins, and ingest order is deterministic (config order). Listing
   specialist sources before generalist ones lets a crypto feed claim a crypto
   story before a general firehose does. Config-only, no schema change.
2. **Give each profile real source coverage.** A profile whose domain arrives
   mainly via another profile's sources is under-sourced; that is a
   `sources.yaml` problem, not an architecture problem.

**Limits of the check.** It is a literal string comparison, so it catches
`crypto` in tech's ignore against `crypto` in crypto's interests — the actual
case today — and misses semantic overlap (`web3` ignored vs `defi` in
interests). It is a lint, not a proof.

**Consequence at scale.** As N grows, every profile's ignore list must avoid
every other profile's interests, which squeezes `ignore.topics` toward empty.
That is acceptable — the ignore list is a blunt instrument and the rubric should
carry the load — and `ignore.people` / `ignore.repos` are unaffected.

**The escape hatch.** If cross-domain misses become a felt gap, the fix is
many-to-many scoring (an item scored under each profile that claims it), which
costs a `scores` PK migration and a real triage multiplier. Trigger it on
evidence — logged misses — never speculatively.

### Config layout

Adding a profile must be a **YAML-only** operation — no Python, no crontab, no
migration.

1. **No default profile.** `config/interests.yaml` → `config/profiles/tech.yaml`,
   simply the first file in the directory, not a privileged case everything else
   is defined against.
2. **Discovery by glob.** Profiles are whatever `config/profiles/*.yaml`
   contains. Accepted risk: a stray or half-finished file silently becomes a
   live profile.
3. **Sources stay in one manifest.** `sources.yaml` gains `profile:` at the
   granularity above. An **unlabelled source is a validation error**, and a
   source naming a profile with no matching file is also a load error — same
   posture as the existing `extra="forbid"`.

Each profile file carries: `interests`, `ignore`, `thresholds` (including
crowding limits), `report_cadence`, `rubric`, and a **vault subdirectory name —
not a path**. `vault_dir` is machine-local and gitignored (`settings.yaml`); a
git-tracked profile file must not contain an absolute path, or it breaks the
settings/config separation `config.py:260-292` argues for.

### Rubric selection is config; rubric text is code

| Layer | Owns |
|---|---|
| `config/profiles/*.yaml` | **which** rubric — `rubric: news` |
| `score/rubrics.py` | the prompt text **and** its version |
| `scores.rubric_version` | the **resolved** version at scoring time |

Profiles name a rubric, never a version. Pinning versions in config means
improving a prompt requires editing N profile files, and forgetting one silently
splits scoring. Naming means one constant bump lifts every profile at once, and
NEVER 5 holds exactly as written.

Most profiles share `news`. A leisure profile may eventually need its own —
punishing hype as the top dimension is probably wrong for hobby content — and
authoring that *is* correctly a code change, because a prompt is code. A profile
naming a nonexistent rubric is a **startup error**, not a fallback.

**The refactor itself bumps the version once.** The cached prefix embeds the
literal string `"Interests (config/interests.yaml):"` and a deterministic render
of the config including `thresholds` (`rubrics.py:84-123`). Renaming the file or
relocating thresholds changes prefix bytes, which invalidates the cache and —
by NEVER 5's own logic — breaks score comparability even though no prompt text
was edited. So step 4 bumps to `news-v1` once, deliberately, and freezes the
prefix bytes thereafter.

### Schema

Two append-only columns, **both with a backfill**.

| Table | Column | Why | Backfill |
|---|---|---|---|
| `scores` | `profile` | So the row explains itself, for the same reason it already carries `rubric_version` and `model`. | Derive item → source → new label; anything unresolvable → `tech`. |
| `runs` | `profile` | Backs the due-check, and carries token attribution. | Existing rows → `tech`. |

The backfill is not optional. `_SELECT_DIGEST_ITEMS` (`db.py:526`) is currently
profile-agnostic; step 4 must filter it. Filter on `scores.profile` without a
backfill and every pre-migration item — including that morning's — has NULL and
vanishes, which fails this plan's own byte-identical check on day one.

**Orphaned sources.** Items whose `source_id` is no longer in `sources.yaml`
(several were pruned on 2026-07-24) resolve to no profile. Backfilling them to
`tech` keeps them visible and keeps `_count_unscored_items` (`cli.py:504`)
draining. This is why the column is worth keeping rather than deriving profile
through a live source lookup at query time — the lookup goes wrong the moment
the source list changes, which it does routinely.

### Scheduling — one cron line, forever

Per-profile cadence **cannot** live in crontab. Ingest, the DB, and the HTTP
cache are shared: N cron lines at 06:00 means N processes against one SQLite
file re-fetching the same feeds. WAL gives one writer; that design builds lock
contention and duplicate fetching on purpose.

So a single scheduled invocation is required regardless of profile count —
`signalforge run --due`, which reads every profile, works out which are due, and
runs those. Cron gets one line, ever, which also matters because the crontab is
shared with other projects.

**Due-check specifics** (the first draft said "stays dumb" and left three things
undefined):

- **Periods resolve through `settings.tzinfo`**, not UTC. A UTC-computed period
  boundary disagrees with the digest's local-day window
  (`report/daily.py:233-244`) and re-imports the exact empty-digest failure
  Phase 0 already fixed once (DESIGN §16, "local-day boundary (resolved)").
- **"Successful"** means a `runs` row with `status IN ('ok','partial')` *and* a
  rendered file. A `partial` run that produced a digest counts as done.
- **Failure isolation between profiles.** One profile's render raising must not
  abort the others — the NEVER 12 analog at profile level, mirroring the
  per-step isolation already in `daily` (`cli.py:841-853`).

**Cutover is atomic.** `load_interests` hardcodes `INTERESTS_FILENAME`
(`config.py:421-433`) and `score`/`digest`/`daily` all call it (`cli.py:562,
661`). Renaming the file without swapping the crontab to `run --due` in the same
change means the 06:00 job exits 2 every morning. Code, config move, and the
crontab edit land together or not at all.

### Cadence applies to rendering, not to the pipeline

A weekly profile must **not** ingest weekly. RSS feeds hold only N recent
entries; fetch one weekly and anything that scrolled off is gone permanently,
taking conditional-GET efficiency with it.

| Stage | Cadence |
|---|---|
| Ingest | Daily, shared, every source, regardless of any profile's cadence |
| Score | Daily, on whatever is unscored, at its own profile's rubric |
| **Render** | **The only thing the cadence knob controls** |

Name the config key `report_cadence`, not `cadence`, so this cannot be wired up
wrong.

### Vault layout

`vault_dir` is `/mnt/c/Users/ian/OneDrive/Documents/SignalForge`
(`settings.yaml`) — **not this repo, and not a git repo**. The `vault/daily/`
directory in the repo holds two stale files predating the move, and the Phase 1
vault auto-commit does not exist yet. So "`git mv` preserves history" is a
fiction for the vault that matters; the real operation is a plain directory move
under OneDrive sync with Obsidian likely open.

Given that, **grandfather `<vault>/daily/` as the tech profile** and put new
profiles at `<vault>/<profile>/daily/`. Permanently slightly inconsistent, but it
moves nothing, breaks no Obsidian links, and touches no history (NEVER 8). If a
tidy-up is ever wanted, it is a manual operator action, not a pipeline step.

`harvest_marks` hardcodes `vault_dir / "daily"` (`feedback.py:140`) and must
become profile-aware either way — and must not strand unharvested checked boxes
in a location it stops globbing.

### `status`

Per-profile month-to-date spend requires **per-profile `runs` rows**. Tokens
live on `runs` (`cli.py:1102-1127`), not on items, so "profiles map to sources
map to items" is irrelevant to attribution. Scoring is currently one run over
all unscored items (`score/__init__.py:94-163`); step 4 splits it per profile.
Then the breakdown is a `GROUP BY`.

### Cost — synthesis is the dominant term

**Triage** scales with sources added, not profiles owned. That part holds.

**Synthesis does not.** The Weekly Intelligence Brief is one Opus call per
profile per week (~$2.40/month each, DESIGN §8). Six profiles with
Opus-rendered weeklies is ~$15/month of synthesis alone, before triage — over
half the $30 alarm, from a feature whose headline claim was "cost doesn't scale
with profiles".

**Default decision (revisitable):** a weekly render is a **deterministic
top-N template**. Opus synthesis is opt-in per profile, granted only to a
profile that has passed its acceptance gate, and bounded by an explicit top-N
cap rather than by keep-count. Restated invariant:

> Triage scales with sources. **Synthesis scales with Opus-rendered profiles**,
> and that number is capped by hand.

Second-order effects, both small at single-digit N: N distinct cached prefixes
means N cache writes per run; triage batches (~25 items) cannot span profiles,
so several small profiles mean several half-full requests. Degrades at dozens of
profiles, not at six.

**Any diff here touches scoring, prompts, and batching → `llm-cost-guard`
review is mandatory before merge (CLAUDE.md §6).**

### Discipline — each profile earns its keep

- **Cadence is the pressure valve.** Tech daily + five hobbies weekly is one
  document a morning and five across the week. Cadence decouples profile count
  from reading load.
- **Per-profile acceptance gate.** Phase 0's own bar, per profile: read it five
  times and it saved time, or delete it. This is also what gates access to Opus
  synthesis.

---

## Sequencing

Reordered after review: the Weekly Brief moves **ahead** of the profile work.
Phase 1's gate is four consecutive Sunday briefs (DESIGN §16), and spending the
risky structural steps first on a hobby lane that is months out is DESIGN §17
risk 1 by this plan's own citation. The brief is a query plus a template, not an
architecture — profile-parameterising it later is cheap.

| # | Step | Why here |
|---|---|---|
| 1 | **Gold capture** — `VERDICTS`, `MARK_RE`, third template checkbox, CLI verdict | Small, independent, zero LLM cost. Feedback cannot be captured retroactively — every day without it is signal permanently lost. |
| 2 | **Render stored state** in the daily template (max rung) | Small; fixes the trust problem before gold marks accumulate. |
| 3 | **Weekly Intelligence Brief at N=1** | Phase 1's actual deliverable and gate. Built against the current single-interests config. |
| 4 | **Profile plumbing at N=1** — `config/profiles/`, `profile:` on sources, non-overlap validation, rubric-by-name, the two columns *with backfill*, per-profile score runs, the dispatcher, crontab cutover, rubric bump to `news-v1` | The risky structural change, made verifiable: with only `tech` configured the digest must render **byte-identically**. Verify by capturing a rendered digest *before* the change and diffing after — the existing suite asserts double-run identity within one code version, not identity across a refactor. |
| 5 | **Turn on crypto (N=2)** — RSS sources + one profile file | Genuinely config-only, because GitHub/HN stay block-level tech. Two profiles proves the abstraction; three only proves it again. |
| 6 | **Gold rollup page** — derived `<vault>/<profile>/gold.md`, regenerated, idempotent | Presentation, after the vault layout settles. Optionally defer until a month of real marks exists, by the same logic as per-item notes. |

Hobbies arrive after step 6 as pure config. That is the test of whether the
design worked.

**Phase discipline:** none of this is Phase 2 or 3 work (NEVER 15). Feature A is
Phase 1 capture (DESIGN §11). Feature B is new scope justified by a felt gap,
now correctly sequenced behind the Phase 1 gate.

---

## Review gates

The repo already carries the tooling; this section binds it to the steps so
nothing lands on "it looked fine".

### Standing gates — every step, no exceptions

| Gate | Source |
|---|---|
| `ruff` (lint + format) and `mypy --strict`, **zero errors** | CLAUDE.md §9 |
| `pytest` green; new behaviour has tests before merge | CLAUDE.md §8 |
| No live network, no real Anthropic API in tests — `respx` fixtures, `llm.py` faked at its boundary | NEVER 13 |
| `code-reviewer` subagent **after the change, before commit** | CLAUDE.md, agent is explicitly "use proactively" |
| Conventional Commits, one logical change per commit | CLAUDE.md §9 |
| No secrets in config, logs, or commits | NEVER 16, `.claude/conventions.md` |

`code-reviewer` is the phase-gate check as well as the correctness check — its
brief covers NEVER-table compliance and phase discipline, so it is the standing
guard against this plan quietly building Phase 2 work.

### Per-step gates

| Step | `llm-cost-guard` | Step-specific gates |
|---|---|---|
| **1** Gold capture | Not required — no LLM path touched | Round-trip test: `checkbox_marker` output → `parse_marks` → same verdict. Dual-verdict case (`useful` + `gold` on one item) asserted to store two rows and aggregate to the max rung. Golden digest diff **read line by line**, not blind-regenerated. |
| **2** Render stored state | Not required | Re-render idempotency: render → mark → re-render → marks still shown, no duplicate rows. Golden diff read, not accepted. |
| **3** Weekly Brief (N=1) | **MANDATORY** — first Opus synthesis call in the system | **Citation discipline is the hard gate** (NEVER 7, DESIGN §5): no synthesized claim renders without a stored `item.url`. Needs an explicit test that an uncited claim fails to render. Cached prefix carries no timestamps or run IDs (NEVER 10). Top-N bound on what reaches Opus, by cap not keep-count. |
| **4** Profile plumbing (N=1) | **MANDATORY** — rubric version bump, cached-prefix bytes change, score runs split per profile (batching change) | See the dedicated list below. |
| **5** Turn on crypto (N=2) | **Advisory** — no code touches `llm.py`, but source volume drives triage spend | Sources added via `/add-source`, not hand-edited (CLAUDE.md §4 workflow). Non-overlap validation proven by a deliberately conflicting fixture profile asserting a load error. Estimate the added triage volume *before* merge. |
| **6** Gold rollup page | Not required | Idempotency: render twice, byte-identical file, zero duplicate rows. Vault write path only — the `vault-guard` hook must never need to fire. |

### Step 4 — the migration gates

The riskiest step in the plan, and the only one that touches live data.

1. **Back up `data/signalforge.db` first.** The `vault-guard` hook already
   classifies it as "regenerable but expensive — LLM re-scoring costs money".
   A failed backfill without a backup means paying to re-score history.
2. **Backfill verification, as assertions not eyeballs**: zero NULL
   `scores.profile`, zero NULL `runs.profile`, and pre/post row counts identical
   for both tables.
3. **Byte-identical digest check**: capture a rendered digest *before* the
   refactor, re-render after, diff. The existing suite asserts double-run
   identity within one code version, not identity across a refactor — this is a
   one-off manual golden, and it is the step's real acceptance test.
4. **Orphan drain**: `_count_unscored_items` must still reach zero after a run.
   Items from pruned sources must resolve to a profile, not stall forever.
5. **Cutover proof**: `run --due` produces the same digest the old `daily` cron
   line did, verified before the crontab is swapped. Code, config move, and
   crontab edit land together — a half-applied cutover exits 2 at 06:00 daily.
6. **Crontab discipline**: only SignalForge's lines may be touched; the crontab
   is shared with other projects.
7. **Rollback**: revert the code and restore the crontab line. The two columns
   are append-only and inert, so they can stay — no down-migration needed.

### Human gates (not automatable)

| Gate | When |
|---|---|
| Four consecutive Sunday briefs that answer the primary question; ≥ 80% of brief items rated **useful-or-better** | Phase 1 acceptance, after step 3 — this is the gate the resequencing exists to protect |
| Tech digest watched for one week for crypto bleed after the non-overlap constraint drops its blanket ignore | After step 4/5 |
| Per-profile acceptance: read it five times and it saved time, or delete the profile | Each new profile, and it is what grants access to Opus synthesis |

---

## Open decisions

| # | Decision | Blocks |
|---|---|---|
| 1 | **The verdict name.** `gold` is a placeholder. Chosen once at implementation, then permanent. | Step 1 |
| 2 | **Un-mark path** — accept that a mis-ticked box is permanent, or add `signalforge unmark`. Accepting is the default. | Step 2 |
| 3 | **Hobby profile count, and whether leisure needs its own rubric.** Crypto is structurally the same shape as tech and should transfer. Hobby is the real test of whether `news` generalises — if the hobby digest is bad, that is evidence, not a bug. | After step 5 |
| 4 | **Mobile read layer.** Previously deferred. A growing set of lightweight weekly digests and that idea point at each other. | Nothing — revisit after step 6 |

Resolved in this revision: weekly-render cost model (deterministic template by
default, Opus opt-in per gated profile); vault migration (grandfather, no move);
cross-domain suppression (non-overlap constraint + ingest ordering, residual
passive miss accepted).
