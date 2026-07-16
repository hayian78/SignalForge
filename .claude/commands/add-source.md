---
description: Add a new ingestion source to config/sources.yaml — validate, classify, test-fetch, and wire it in without touching Python.
---

# /add-source <url or repo or description>

Adding a source is a YAML edit, never a code change (CLAUDE.md §4). If the
requested source genuinely needs a new *source type* (a new adapter under
`src/signalforge/ingest/`), STOP and tell the user — that's a feature, not
a source addition, and it's phase-gated (DESIGN §7 table).

## Steps

1. **Classify** the input into an existing adapter type:
   - Blog / newsletter / personal site → `rss` (find the feed: try
     `/feed`, `/rss`, `/atom.xml`, `/index.xml`, `<link rel="alternate">`
     in the page head).
   - GitHub repo → `github.releases` (or `github.awesome_lists` if it's a
     curated list README).
   - Paper feed → `arxiv` keyword/category addition.
   - Keyword topic → `hackernews.keywords` addition.
   - None of the above → STOP per the rule above.

2. **Validate before editing** — fetch the candidate feed/endpoint once
   (`curl -sL --max-time 20 <url> | head -c 2000`) and confirm it parses as
   RSS/Atom/JSON and has recent entries. A dead or stale feed (< 1 post in
   90 days) gets reported back, not added.

3. **Check for duplicates** in `config/sources.yaml` (same id, same
   canonical host, or a repo already covered).

4. **Edit `config/sources.yaml`**: kebab/snake-case `id`, the feed URL, and
   `weight` only if the user asked for boosted/reduced trust (default 1.0 —
   don't invent weights).

5. **Verify**: run the config validation (`uv run signalforge status` or the
   pydantic config load in a one-liner) and, if the pipeline exists, a single
   ingest for that source (`uv run signalforge ingest --source <id>` if the
   CLI supports it). Confirm items land in the DB, then report: source id,
   type, item count from the test fetch, and any warnings.

Do NOT edit `interests.yaml` or `taxonomy.yaml` as part of this command —
if the new source implies a new topic of interest, say so and let the user
decide.
