"""Report writer — reads the DB, writes markdown to the vault. No LLM calls.

Module boundary (CLAUDE.md §2): `report/` only reads `items`/`scores`/`runs`
and writes files under the vault. It never imports `llm.py`, never makes an
HTTP call, and never regenerates the LLM's stored `reasoning` — it renders
what `score/` already wrote.

Phase 0 ships `daily.py` only. Weekly/monthly synthesis is Phase 1+ (NEVER
rule 15).
"""

from __future__ import annotations

__all__: list[str] = []
