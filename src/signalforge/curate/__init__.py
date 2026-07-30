"""Adaptive source curation (DESIGN §7.1) — proposing changes to `sources.yaml`.

The one module permitted to call both `llm.py` and an `ingest/` fetch helper, in
that fixed order: judgment first, then deterministic validation of what judgment
produced. It writes no `items` row, so nothing here can put content into the
pipeline ahead of the operator approving its source. See DESIGN §4's module table
for why that exception exists and CLAUDE.md §2 (NEVER rules 2 and 18) for the
rule it bends and the bound that keeps it safe.
"""

from __future__ import annotations

__all__: list[str] = []
