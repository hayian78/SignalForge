"""Vault auto-commit — snapshot a written report into the vault's own git history.

Phase 1's "vault git-committed" box (DESIGN §16). The vault is the product
(CLAUDE.md §3); this module gives it a history so an accidental overwrite,
a bad template edit, or an Obsidian sync conflict is recoverable.

**Why this lives in `report/`.** `report/` reads the DB and writes markdown to
the vault, and makes no HTTP call (CLAUDE.md §2). Committing that markdown is a
vault-write concern and stays local: there is deliberately **no `git push`**
here. DESIGN §14's "push to a private remote" line remains a manual operator
step, because a push is a network call and would break this module's boundary.

**The repo-root guard is the load-bearing part.** A `vault_dir` can sit
*inside* a larger repository — the shipped configuration points at
`/mnt/c/Users/<you>/OneDrive/...`, and a stray `git init` anywhere above it
would make `git add` reach the whole user profile. So this module commits only
when the vault directory is *itself* the repository top level. Anything else
no-ops with a reason. That makes the dangerous case unreachable by
construction rather than by remembering to check.

**Nothing here can fail a run.** The vault write already succeeded by the time
this is called; a git failure is recorded and returned, never raised (the
posture `deliver/` already takes, NEVER rule 19).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

__all__ = [
    "COMMITTABLE_SUBDIRS",
    "GIT_TIMEOUT_SECONDS",
    "VaultCommitOutcome",
    "VaultCommitStatus",
    "commit_vault",
]

logger = logging.getLogger(__name__)

COMMITTABLE_SUBDIRS: Final[tuple[str, ...]] = ("daily", "weekly", "podcast")
"""The only paths ever staged, named explicitly rather than via `git add -A`.

An operator's vault holds their own notes alongside our generated reports.
Staging everything would sweep those into our commits — and, on the shipped
OneDrive vault, whatever sync debris happens to be present that second. A
report directory that does not exist yet is skipped, not an error."""

GIT_TIMEOUT_SECONDS: Final = 30
"""Ceiling on any one git invocation. The shipped vault lives on `/mnt/c`, where
a Windows-side filesystem stall would otherwise hang the cron run indefinitely —
and a hung digest is worse than an uncommitted one."""


class VaultCommitStatus(StrEnum):
    """Why `commit_vault` did what it did — the outcome vocabulary."""

    COMMITTED = "committed"
    """A commit was created."""

    NOTHING_TO_COMMIT = "nothing_to_commit"
    """The working tree was already clean. The idempotent re-run case."""

    DISABLED = "disabled"
    """Turned off in `settings.yaml`."""

    NOT_A_REPO = "not_a_repo"
    """`vault_dir` is not inside any git repository."""

    NOT_REPO_ROOT = "not_repo_root"
    """`vault_dir` is inside a repository whose top level is somewhere else.
    The guard that stops a stray parent `.git` from capturing the vault."""

    GIT_UNAVAILABLE = "git_unavailable"
    """No `git` executable on PATH."""

    FAILED = "failed"
    """A git command errored. Recorded, never raised."""


@dataclass(frozen=True, slots=True)
class VaultCommitOutcome:
    """What happened, in a shape the CLI can print and fold into `runs.errors`.

    `detail` is always safe to log: it carries git's own stderr, which for these
    commands is repository state, never a credential (nothing here authenticates
    against anything).
    """

    status: VaultCommitStatus
    detail: str = ""
    commit_sha: str | None = None

    @property
    def is_error(self) -> bool:
        """True only for outcomes worth surfacing in `runs.errors`.

        A guard refusing to commit is a *correct* outcome, not a failure: an
        operator who never ran `git init` in their vault should not collect a
        run error every evening for a feature they did not opt into.
        """
        return self.status is VaultCommitStatus.FAILED


def _run_git(vault_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """One git invocation, scoped with `-C` and never inheriting a shell.

    `check=False`: every caller here inspects `returncode` itself, because a
    non-zero exit is frequently the *expected* answer (`rev-parse` outside a
    repo, `diff --cached --quiet` on a clean tree).
    """
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, no user-controlled binary
        ["git", "-C", str(vault_dir), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )


def _resolve_repo_root(vault_dir: Path) -> Path | None:
    """The top level of the repo containing `vault_dir`, or None if there is none."""
    result = _run_git(vault_dir, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top).resolve() if top else None


def commit_vault(
    vault_dir: Path,
    *,
    message: str,
    enabled: bool = True,
) -> VaultCommitOutcome:
    """Stage the report directories under `vault_dir` and commit them.

    Call this **after** a vault write has succeeded, never before: the vault is
    written first and always (DESIGN §13.2), and this is a snapshot of a file
    that already exists.

    Idempotent (NEVER rule 4): a second call over an unchanged vault stages the
    same paths, finds nothing different, and returns `NOTHING_TO_COMMIT` without
    creating an empty commit.

    Never raises. Every failure path — missing `git`, a timeout on a stalled
    `/mnt/c` mount, a git error — comes back as an outcome.
    """
    if not enabled:
        return VaultCommitOutcome(VaultCommitStatus.DISABLED)

    if shutil.which("git") is None:
        logger.warning("git is not on PATH; skipping the vault commit")
        return VaultCommitOutcome(
            VaultCommitStatus.GIT_UNAVAILABLE, detail="no `git` executable on PATH"
        )

    if not vault_dir.is_dir():
        return VaultCommitOutcome(
            VaultCommitStatus.NOT_A_REPO, detail=f"{vault_dir} is not a directory"
        )

    resolved_vault = vault_dir.resolve()

    try:
        repo_root = _resolve_repo_root(resolved_vault)

        if repo_root is None:
            logger.info(
                "vault is not a git repository; skipping the commit",
                extra={"vault_dir": str(resolved_vault)},
            )
            return VaultCommitOutcome(
                VaultCommitStatus.NOT_A_REPO,
                detail=f"{resolved_vault} is not inside a git repository",
            )

        # The guard. A vault nested inside someone else's repository must never
        # be staged through that repository — `git add` there reaches every
        # sibling directory under its top level.
        if repo_root != resolved_vault:
            logger.warning(
                "vault is nested inside a wider git repository; refusing to commit",
                extra={"vault_dir": str(resolved_vault), "repo_root": str(repo_root)},
            )
            return VaultCommitOutcome(
                VaultCommitStatus.NOT_REPO_ROOT,
                detail=(
                    f"{resolved_vault} sits inside the repository at {repo_root}. "
                    "Committing would stage files outside the vault. Run `git init` "
                    "in the vault directory to give it its own history."
                ),
            )

        present = [name for name in COMMITTABLE_SUBDIRS if (resolved_vault / name).is_dir()]
        if not present:
            return VaultCommitOutcome(
                VaultCommitStatus.NOTHING_TO_COMMIT,
                detail="no report directories exist in the vault yet",
            )

        staged = _run_git(resolved_vault, "add", "--", *present)
        if staged.returncode != 0:
            return _failure("git add", staged)

        # `diff --cached --quiet` exits 1 when there *are* staged changes. Any
        # other non-zero is a real error, so the two are distinguished rather
        # than both read as "something to commit".
        diff = _run_git(resolved_vault, "diff", "--cached", "--quiet")
        if diff.returncode == 0:
            return VaultCommitOutcome(VaultCommitStatus.NOTHING_TO_COMMIT)
        if diff.returncode != 1:
            return _failure("git diff --cached", diff)

        committed = _run_git(resolved_vault, "commit", "-m", message)
        if committed.returncode != 0:
            return _failure("git commit", committed)

        head = _run_git(resolved_vault, "rev-parse", "--short", "HEAD")
        sha = head.stdout.strip() if head.returncode == 0 else None
        logger.info(
            "committed the vault",
            extra={"vault_dir": str(resolved_vault), "commit": sha},
        )
        return VaultCommitOutcome(VaultCommitStatus.COMMITTED, commit_sha=sha)

    except subprocess.TimeoutExpired as exc:
        logger.exception("a git command timed out; leaving the vault uncommitted")
        return VaultCommitOutcome(
            VaultCommitStatus.FAILED,
            detail=f"git timed out after {GIT_TIMEOUT_SECONDS}s: {exc.cmd}",
        )
    except OSError as exc:
        logger.exception("running git failed; leaving the vault uncommitted")
        return VaultCommitOutcome(VaultCommitStatus.FAILED, detail=str(exc))


def _failure(label: str, result: subprocess.CompletedProcess[str]) -> VaultCommitOutcome:
    """Fold a non-zero git exit into an outcome, preferring stderr for the reason."""
    reason = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
    logger.warning("%s failed; leaving the vault uncommitted", label, extra={"reason": reason})
    return VaultCommitOutcome(VaultCommitStatus.FAILED, detail=f"{label}: {reason}")
