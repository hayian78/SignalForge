"""Judgment layer over already-scored, already-stored items (DESIGN §4).

`synth/` never makes HTTP calls to sources — the same module-boundary
discipline as `score/` (CLAUDE.md §2). It calls `llm.py` only, never the
`anthropic` SDK directly (NEVER rule 1).

DESIGN's folder tree reserves this package for Phase 2/3 cross-source
synthesis (`trends.py`, `synthesis.py`, `impact.py` — none built yet, per
CLAUDE.md §1's phase-gate discipline). `podcast.py` activates the package
early, under DESIGN §13.3's second recorded phase-gate exception: the
podcast channel's script-writing judgment belongs here, next to any future
synthesis work, rather than in `report/`, which stays "read the DB, write
markdown, no LLM calls" (CLAUDE.md §2).
"""

from __future__ import annotations
