#!/usr/bin/env bash
# PreToolUse guard: blocks destructive Bash commands aimed at vault/ (the
# product — git-tracked knowledge base) and at data/signalforge.db.
# Exit 2 = block the tool call; stderr is fed back to the assistant.

payload=$(cat)

command=$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))
except Exception:
    print("")
')

[ -z "$command" ] && exit 0

# Destructive verbs targeting the vault
if printf '%s' "$command" | grep -qE '(rm|shred|unlink)\s+(-[a-zA-Z]+\s+)*[^ ]*vault/'; then
  echo "BLOCKED by vault-guard: destructive command targets vault/ — the vault is the product. If the user explicitly asked for this, have them run it themselves." >&2
  exit 2
fi

# Truncation / overwrite of the vault via shell redirection is fine only for
# report files the writer owns; block find -delete and recursive wipes.
if printf '%s' "$command" | grep -qE 'find\s+[^|;]*vault/[^|;]*-delete'; then
  echo "BLOCKED by vault-guard: bulk deletion under vault/." >&2
  exit 2
fi

# Vault git history rewrites
if printf '%s' "$command" | grep -qE 'git\s+(-C\s+[^ ]*\s+)?(push\s+.*(--force|-f)|reset\s+--hard|filter-branch|filter-repo|clean\s+-[a-zA-Z]*f)' \
   && printf '%s' "$command" | grep -q 'vault'; then
  echo "BLOCKED by vault-guard: git history rewrite touching the vault." >&2
  exit 2
fi

# Deleting or ad-hoc DDL on the live DB
if printf '%s' "$command" | grep -qE '(rm|shred|unlink)\s+(-[a-zA-Z]+\s+)*[^ ]*signalforge\.db'; then
  echo "BLOCKED by vault-guard: deleting data/signalforge.db. It is regenerable but expensive (LLM re-scoring costs money). Ask the user first." >&2
  exit 2
fi
if printf '%s' "$command" | grep -qiE 'sqlite3\s+[^ ]*signalforge\.db.*(DROP\s+TABLE|DELETE\s+FROM|ALTER\s+TABLE|UPDATE\s+)'; then
  echo "BLOCKED by vault-guard: write/DDL against the live DB via shell. Schema changes go through db.py migrations; data fixes go through the CLI." >&2
  exit 2
fi

exit 0
