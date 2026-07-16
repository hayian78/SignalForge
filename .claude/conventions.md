# Tool & shell conventions for AI assistants

Behaviour rules for AI coding assistants working in this repo. Keeps tool
output within context budgets and secrets out of the transcript.

Permission grants live in `.claude/settings.local.json` under
`permissions.allow` — this doc only covers *how* to invoke commands.

## Absolute paths + tool-native directory flags

Don't `cd` to change directory. Use absolute paths and the tool's own flag:

- Git: `git -C /home/ian/projects/signalforge <subcommand>`
- pytest / ruff / mypy: invoke from repo root with absolute or repo-relative
  paths.

`cd` is only acceptable when no native flag exists; even then, a single
`cd ... && <cmd>`.

## Dedicated tools over shell utilities

- Read files: `Read` (not `cat`/`head`/`tail`)
- Find files: `Glob` (not `find`/`ls`)
- Search content: `Grep` (not `grep`/`rg`)
- Edit files: `Edit` (not `sed`/`awk`)

Reserve Bash for: tests, `uv` commands, git, sqlite3 inspection, running the
`signalforge` CLI.

## Python invocation (uv-managed project)

- Run everything through uv: `uv run pytest ...`, `uv run ruff check .`,
  `uv run mypy src`, `uv run signalforge <command>`.
- Don't `source .venv/bin/activate` — activation doesn't persist across Bash
  tool calls. `.venv/bin/<tool>` direct is an acceptable alternative.
- Dependency changes go through `uv add` / `uv add --dev`, never hand-editing
  `pyproject.toml` version pins plus `pip install`.

## Long-running processes

- The MCP server (Phase 3) or any watch-mode process MUST use
  `run_in_background: true`. Verify with a short check, not `sleep` polling.
- Pipeline commands (`signalforge daily` etc.) are batch jobs — run them
  foreground, but pipe noisy output: `2>&1 | tail -40`.

## Database and vault discipline

- Inspect the DB read-only by default: `sqlite3 data/signalforge.db "SELECT ..."`.
  Schema changes only via `db.py` migrations — never ad-hoc `ALTER TABLE` in a
  shell.
- `data/signalforge.db` and `data/http_cache/` are regenerable; **`vault/` is
  not** — it is the product. Never `rm`, truncate, or mass-rewrite files under
  `vault/` outside `report/writer.py`'s normal overwrite-today's-file behaviour
  without explicit user instruction. (A guard hook enforces this.)

## Git

- Always `git -C <absolute-path>`.
- NEVER skip hooks (`--no-verify`) without explicit user instruction.
- NEVER force-push, `reset --hard`, `clean -f`, or `branch -D` without
  explicit approval. This applies doubly to the vault's git history.

## Secrets in tool output

The secrets here are `ANTHROPIC_API_KEY` and `GITHUB_TOKEN` (in `.env`).
Any command that would return a secret MUST NOT echo it to chat — no
`cat .env`, `env`, `printenv`, `echo $ANTHROPIC_API_KEY`.

- ✅ Pipe secrets from store to consumer without stdout:
  `GITHUB_TOKEN=$(...) uv run signalforge ingest` in **one** Bash call.
- ✅ If the user needs to see a secret, ask for confirmation before printing.
- If a secret leaks into the transcript: tell the user immediately and
  recommend rotating it.

The pattern: **secrets flow from one trusted store to one consumer, never
through stdout.**

## Permission grants

If you repeatedly trigger approval prompts for a command that belongs in the
pre-approved set, surface it for explicit addition to
`.claude/settings.local.json`. Don't reformulate a command into something
less readable to dodge the prompt.
