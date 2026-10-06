"""Tests for local Semantic Version bumps and annotated tags."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.bump_version import ReleaseHelper


@pytest.fixture
def release_repo(tmp_path: Path) -> ReleaseHelper:
    """Create a small isolated git repository for release operations.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        Helper bound to the test repository.
    """
    helper = ReleaseHelper(tmp_path)
    helper.run("git", "init", "-q")
    helper.run("git", "config", "user.name", "Test")
    helper.run("git", "config", "user.email", "test@example.invalid")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "example"\nversion = "0.1.0"\n')
    (tmp_path / "uv.lock").write_text("test lock\n")
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n### Fixed\n\n- A fix.\n\n## [0.1.0] - 2026-10-03\n\n- Initial.\n\n"
        "[Unreleased]: https://github.com/test/example/compare/v0.1.0...HEAD\n"
        "[0.1.0]: https://github.com/test/example/releases/tag/v0.1.0\n"
    )
    helper.run("git", "add", ".")
    helper.run("git", "commit", "-qm", "Initial")
    return helper


@pytest.mark.parametrize("part,expected", [("patch", "0.1.1"), ("minor", "0.2.0"), ("major", "1.0.0")])
def test_bump(release_repo: ReleaseHelper, monkeypatch: pytest.MonkeyPatch, part: str, expected: str) -> None:
    """Verify numeric bumps, notes promotion and lock synchronization.

    Args:
        release_repo: Isolated test repository helper.
        monkeypatch: Pytest patch utility.
        part: Requested increment.
        expected: Resulting version.

    Returns:
        None.
    """
    calls: list[tuple[str, ...]] = []
    original = release_repo.run
    original("git", "tag", "v0.1.0")

    def run(*args: str) -> str:
        """Record uv calls without downloading dependencies.

        Args:
            args: Command arguments.

        Returns:
            Git output or an empty uv result.
        """
        calls.append(args)
        return "" if args[0] == "uv" else original(*args)

    monkeypatch.setattr(release_repo, "run", run)
    assert release_repo.bump(part) == expected
    assert release_repo.version() == expected
    notes = (release_repo.root / "CHANGELOG.md").read_text()
    assert "- A fix." in notes
    assert f"## [{expected}] - " in notes
    assert f"compare/v0.1.0...v{expected}" in notes
    assert f"compare/v{expected}...HEAD" in notes
    assert ("uv", "lock") in calls
    assert original("git", "tag", "--list") == "v0.1.0"


def test_empty_notes(release_repo: ReleaseHelper) -> None:
    """Reject bumps with no release notes before changing metadata.

    Args:
        release_repo: Isolated test repository helper.

    Returns:
        None.
    """
    release_repo.run("git", "tag", "v0.1.0")
    (release_repo.root / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [0.1.0] - 2026-10-03\n")
    with pytest.raises(ValueError, match="release notes"):
        release_repo.bump("patch")
    assert release_repo.version() == "0.1.0"


def test_tag(release_repo: ReleaseHelper, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify annotated tags, duplicate refusal and unrelated dirty-file tolerance.

    Args:
        release_repo: Isolated test repository helper.
        monkeypatch: Pytest patch utility.

    Returns:
        None.
    """
    original = release_repo.run
    calls: list[tuple[str, ...]] = []

    def run(*args: str) -> str:
        """Record uv validation while executing git normally.

        Args:
            args: Command arguments.

        Returns:
            Git output or empty uv output.
        """
        calls.append(args)
        return "" if args[0] == "uv" else original(*args)

    monkeypatch.setattr(release_repo, "run", run)
    (release_repo.root / "unrelated.txt").write_text("leave alone")
    assert release_repo.tag() == "v0.1.0"
    assert original("git", "cat-file", "-t", "v0.1.0") == "tag"
    assert ("uv", "lock", "--check") in calls
    assert not any("push" in call for call in calls)
    with pytest.raises(ValueError, match="already exists"):
        release_repo.tag()


def test_dirty_release(release_repo: ReleaseHelper) -> None:
    """Reject tagging staged or unstaged release changes.

    Args:
        release_repo: Isolated test repository helper.

    Returns:
        None.
    """
    with (release_repo.root / "CHANGELOG.md").open("a") as stream:
        stream.write("dirty\n")
    with pytest.raises(ValueError, match="Commit release files"):
        release_repo.tag()
    release_repo.run("git", "add", "CHANGELOG.md")
    with pytest.raises(ValueError, match="Commit release files"):
        release_repo.tag()


def test_uv_failure_does_not_tag(release_repo: ReleaseHelper, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse tags if locked metadata validation fails.

    Args:
        release_repo: Isolated test repository helper.
        monkeypatch: Pytest patch utility.

    Returns:
        None.
    """
    original = release_repo.run

    def run(*args: str) -> str:
        """Simulate a failed uv check.

        Args:
            args: Command arguments.

        Returns:
            Git output.

        Raises:
            subprocess.CalledProcessError: For uv commands.
        """
        if args[0] == "uv":
            raise subprocess.CalledProcessError(1, args)
        return original(*args)

    monkeypatch.setattr(release_repo, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        release_repo.tag()
    assert original("git", "tag", "--list") == ""


def test_commit_notes(release_repo: ReleaseHelper, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate notes from post-tag commits without requiring manual notes.

    Args:
        release_repo: Isolated repository helper.
        monkeypatch: Pytest patch utility.

    Returns:
        None.
    """
    release_repo.run("git", "tag", "v0.1.0")
    subjects = ["feat(ui): add selector", "fix: restore focus", "feat!: replace config", "plain subject"]
    for subject in subjects:
        release_repo.run("git", "commit", "--allow-empty", "-qm", subject)
    (release_repo.root / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [0.1.0] - 2026-10-03\n")
    original = release_repo.run

    def run(*args: str) -> str:
        """Stub uv while executing local history commands.

        Args:
            args: Executable and arguments.

        Returns:
            Git output or empty uv output.
        """
        return "" if args[0] == "uv" else original(*args)

    monkeypatch.setattr(release_repo, "run", run)
    assert release_repo.bump("patch") == "0.1.1"
    notes = (release_repo.root / "CHANGELOG.md").read_text()
    for subject in subjects:
        assert subject in notes
    for category in ["Added", "Fixed", "Breaking changes", "Other"]:
        assert f"### {category}" in notes
    assert "Initial (`" not in notes
    assert original("git", "rev-parse", "--short", "HEAD") in notes


def test_missing_baseline(release_repo: ReleaseHelper) -> None:
    """Refuse an ambiguous history range without modifying release metadata.

    Args:
        release_repo: Isolated repository helper.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="Missing baseline tag"):
        release_repo.bump("patch")
    assert release_repo.version() == "0.1.0"


def test_nonancestor_baseline(release_repo: ReleaseHelper) -> None:
    """Refuse a tag on a divergent commit rather than including unrelated history.

    Args:
        release_repo: Isolated repository helper.

    Returns:
        None.
    """
    initial = release_repo.run("git", "rev-parse", "HEAD")
    release_repo.run("git", "commit", "--allow-empty", "-qm", "fix: future")
    release_repo.run("git", "tag", "v0.1.0")
    release_repo.run("git", "checkout", "--detach", initial)
    with pytest.raises(ValueError, match="not an ancestor"):
        release_repo.bump("patch")
    assert release_repo.version() == "0.1.0"


def test_history_baseline(release_repo: ReleaseHelper) -> None:
    """Categorize documentation and omit commits already covered by the baseline.

    Args:
        release_repo: Isolated repository helper.

    Returns:
        None.
    """
    release_repo.run("git", "tag", "v0.1.0")
    release_repo.run("git", "commit", "--allow-empty", "-qm", "docs: clarify setup")
    notes = release_repo.release_notes("0.1.0")
    assert "### Documentation" in notes
    assert "docs: clarify setup" in notes
    assert "Initial" not in notes
