"""Vault auto-commit tests (DESIGN §16, `report/vaultgit.py`).

The case that matters most is the *refusal*: the shipped vault lives under
`/mnt/c/Users/<you>/OneDrive/...`, and a stray `git init` above it would let a
naive `git add` stage the whole Windows user profile. `test_nested_repo_*`
below is the regression guard for exactly that.

Every test builds real throwaway repos under `tmp_path`. No network, no vault.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from signalforge.report.vaultgit import (
    VaultCommitStatus,
    commit_vault,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture(autouse=True)
def _isolated_git_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cut these tests off from the developer's own git configuration.

    Without this, a global `commit.gpgsign = true` hangs on pinentry until the
    30s timeout and a global `core.hooksPath` runs the developer's hooks against
    a throwaway repo — so CI and a laptop disagree about whether the suite
    passes. Also keeps any real `GIT_*` in the ambient environment out of the
    tests that are specifically about `GIT_*`.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        if key not in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
            monkeypatch.delenv(key, raising=False)
    yield


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(path: Path) -> None:
    """A repo with identity configured locally, so the test never depends on
    (or writes to) the developer's global git config."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--initial-branch=main")
    _git(path, "config", "user.email", "test@signalforge.invalid")
    _git(path, "config", "user.name", "SignalForge Test")


def _write_report(vault: Path, subdir: str = "daily", name: str = "2026-08-16.md") -> Path:
    path = vault / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# digest\n", encoding="utf-8")
    return path


def _log_paths(repo: Path) -> list[str]:
    result = _git(repo, "show", "--name-only", "--pretty=format:", "HEAD")
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def _commit_count(repo: Path) -> int:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return int(result.stdout.strip()) if result.returncode == 0 else 0


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #


def test_nested_repo_refuses_and_stages_nothing(tmp_path: Path) -> None:
    """A vault inside a wider repo must not be committed *through* that repo.

    The shipped failure this prevents: `vault_dir` under a home directory that
    someone once ran `git init` in. Committing there would sweep in every
    sibling file the outer repo can see.
    """
    outer = tmp_path / "home"
    _init_repo(outer)
    unrelated = outer / "private" / "secrets.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("not ours\n", encoding="utf-8")

    vault = outer / "Documents" / "SignalForge"
    _write_report(vault)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.NOT_REPO_ROOT
    assert not outcome.is_error, "a guarded refusal is a correct outcome, not a run error"
    assert "git init" in outcome.detail, "the operator needs to be told the fix"
    # Nothing staged anywhere: the outer repo still has no commits and a clean index.
    assert _commit_count(outer) == 0
    staged = _git(outer, "diff", "--cached", "--name-only").stdout.strip()
    assert staged == ""


def test_a_leaked_git_dir_cannot_bypass_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GIT_DIR` in the environment forces git's repository discovery.

    Left inherited, `rev-parse --show-toplevel` reports the *forced* work tree —
    the vault — instead of the repository that actually contains it, the two
    sides of the guard agree, and the commit lands in the outer repo. This was a
    real, demonstrated bypass of the guard above, not a theoretical one.
    """
    outer = tmp_path / "home"
    _init_repo(outer)
    vault = outer / "Documents" / "SignalForge"
    _write_report(vault)

    monkeypatch.setenv("GIT_DIR", str(outer / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(vault))

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.NOT_REPO_ROOT
    assert _commit_count(outer) == 0


def test_not_a_repo_is_a_quiet_no_op(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write_report(vault)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.NOT_A_REPO
    assert not outcome.is_error


def test_missing_directory_is_a_no_op(tmp_path: Path) -> None:
    outcome = commit_vault(tmp_path / "absent", message="vault: nothing")

    assert outcome.status is VaultCommitStatus.NOT_A_REPO
    assert not outcome.is_error


def test_disabled_short_circuits(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16", enabled=False)

    assert outcome.status is VaultCommitStatus.DISABLED
    assert _commit_count(vault) == 0


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_commits_when_the_vault_is_its_own_repo(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.COMMITTED
    assert outcome.commit_sha
    assert _commit_count(vault) == 1
    assert _log_paths(vault) == ["daily/2026-08-16.md"]


def test_stages_only_the_report_directories(tmp_path: Path) -> None:
    """An operator's own notes live in the same vault. They are not ours to commit."""
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault, "daily")
    _write_report(vault, "weekly", "2026-08-16.md")
    _write_report(vault, "podcast", "2026-08-15.md")
    (vault / "Personal Notes").mkdir()
    (vault / "Personal Notes" / "journal.md").write_text("mine\n", encoding="utf-8")
    (vault / "desktop.ini").write_text("sync junk\n", encoding="utf-8")

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.COMMITTED
    assert _log_paths(vault) == [
        "daily/2026-08-16.md",
        "podcast/2026-08-15.md",
        "weekly/2026-08-16.md",
    ]


def test_the_operators_own_staged_files_are_left_alone(tmp_path: Path) -> None:
    """A bare `git commit -m` commits the whole index, not just what we staged.

    The sharp version of that: an operator mid-commit in their own vault, with a
    `.env` staged, finds it swept into a commit signalforge authored (NEVER
    rules 8, 16). The pathspec on both `diff` and `commit` is what prevents it.
    """
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)
    (vault / ".env").write_text("SECRET=hunter2\n", encoding="utf-8")
    (vault / "journal.md").write_text("mine\n", encoding="utf-8")
    _git(vault, "add", ".env", "journal.md")

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.COMMITTED
    assert _log_paths(vault) == ["daily/2026-08-16.md"]
    # Still staged, still uncommitted — exactly where the operator left them.
    still_staged = _git(vault, "diff", "--cached", "--name-only").stdout.split()
    assert sorted(still_staged) == [".env", "journal.md"]


def test_an_unrelated_dirty_index_does_not_fake_a_commit(tmp_path: Path) -> None:
    """The emptiness check is scoped too, or `NOTHING_TO_COMMIT` never fires."""
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)
    commit_vault(vault, message="vault: daily digest 2026-08-16")

    (vault / "journal.md").write_text("mine\n", encoding="utf-8")
    _git(vault, "add", "journal.md")
    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.NOTHING_TO_COMMIT
    assert _commit_count(vault) == 1


def test_second_run_with_no_changes_makes_no_commit(tmp_path: Path) -> None:
    """Idempotency (NEVER rule 4): re-running the digest must not pile up empty commits."""
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)

    first = commit_vault(vault, message="vault: daily digest 2026-08-16")
    second = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert first.status is VaultCommitStatus.COMMITTED
    assert second.status is VaultCommitStatus.NOTHING_TO_COMMIT
    assert not second.is_error
    assert _commit_count(vault) == 1


def test_rewritten_report_commits_again(tmp_path: Path) -> None:
    """The overwrite-today's-file path still produces a second commit."""
    vault = tmp_path / "vault"
    _init_repo(vault)
    path = _write_report(vault)
    commit_vault(vault, message="vault: daily digest 2026-08-16")

    path.write_text("# digest, now with more items\n", encoding="utf-8")
    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.COMMITTED
    assert _commit_count(vault) == 2


def test_empty_vault_repo_has_nothing_to_commit(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _init_repo(vault)

    outcome = commit_vault(vault, message="vault: nothing yet")

    assert outcome.status is VaultCommitStatus.NOTHING_TO_COMMIT
    assert not outcome.is_error


# --------------------------------------------------------------------------- #
# Failure never escapes
# --------------------------------------------------------------------------- #


def test_missing_git_executable_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("signalforge.report.vaultgit.shutil.which", lambda _: None)

    outcome = commit_vault(tmp_path, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.GIT_UNAVAILABLE
    assert not outcome.is_error


def test_a_git_failure_is_returned_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit failure must never fail the run — the digest is already on disk."""
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)

    real_run = subprocess.run

    def fail_on_commit(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "commit" in args:
            return subprocess.CompletedProcess(args, 1, "", "fatal: could not write [/objects]")
        return real_run(args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("signalforge.report.vaultgit.subprocess.run", fail_on_commit)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.FAILED
    assert outcome.is_error, "a real git failure does belong in runs.errors"
    assert "could not write" in outcome.detail


def test_undecodable_git_output_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "never raises" contract has to hold for the decode step too.

    `text=True` defaults to strict UTF-8, and git echoes paths verbatim — a
    Windows-side vault can hold a name that is not valid UTF-8. The resulting
    `UnicodeDecodeError` is a `ValueError`, so it would sail past an
    `except OSError` and fail a report that is already on disk.
    """
    vault = tmp_path / "vault"
    _init_repo(vault)
    _write_report(vault)

    def undecodable(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte")

    monkeypatch.setattr("signalforge.report.vaultgit.subprocess.run", undecodable)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.FAILED


def test_a_git_timeout_is_returned_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled `/mnt/c` mount must not hang the cron run."""
    vault = tmp_path / "vault"
    _init_repo(vault)

    def timeout(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=30)

    monkeypatch.setattr("signalforge.report.vaultgit.subprocess.run", timeout)

    outcome = commit_vault(vault, message="vault: daily digest 2026-08-16")

    assert outcome.status is VaultCommitStatus.FAILED
    assert "timed out" in outcome.detail
