"""Report writer — reads the DB, writes markdown to the vault. No LLM calls.

Module boundary (CLAUDE.md §2): `report/` only reads `items`/`scores`/`runs`
and writes files under the vault. It never calls an LLM itself, never makes
an HTTP call, and never regenerates the LLM's stored output — it renders
what `score/`/`synth/` already wrote. `podcast.py` (DESIGN §13.3) is the one
module here that transitively imports `llm.py`, via `synth.podcast.BuiltScript`
— `synth/` owns the Anthropic-facing payload types (`PodcastScript` and
friends), and `report/` needs their shape to render one. `report/` itself
never touches the `anthropic` SDK or calls `run_podcast_script`.

Phase 0 ships `daily.py`. Podcast (`podcast.py`) is DESIGN §13.3's separately
recorded phase-gate exception. Weekly/monthly synthesis is still Phase 1+
(NEVER rule 15).
"""

from __future__ import annotations

__all__: list[str] = []
