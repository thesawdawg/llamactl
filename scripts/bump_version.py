"""Prepare Semantic Version releases and create local annotated tags; never push."""
from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

try:
    from .check_release import STABLE_VERSION, check_release
except ImportError:
    from check_release import STABLE_VERSION, check_release


class ReleaseHelper:
    """Manage explicit local bump and tag operations for a repository."""

    root: Path

    def __init__(self, root: Path) -> None:
        """Set the repository location.

        Args:
            root: Repository root.

        Returns:
            None.
        """
        self.root = root

    def run(self, *args: str) -> str:
        """Run a checked command in the repository.

        Args:
            args: Executable and arguments; never interpreted by a shell.

        Returns:
            Captured standard output.

        Raises:
            subprocess.CalledProcessError: If the command fails.
        """
        return subprocess.check_output(args, cwd=self.root, text=True).strip()

    def version(self) -> str:
        """Read the authoritative stable package version.

        Args:
            None.

        Returns:
            Stable Semantic Version.

        Raises:
            ValueError: If the package version is unsupported.
        """
        with (self.root / "pyproject.toml").open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
        if not STABLE_VERSION.fullmatch(version):
            raise ValueError(f"Not a stable Semantic Version: {version}")
        return version

    def ensure_new_tag(self, version: str) -> None:
        """Refuse to overwrite an existing local release tag.

        Args:
            version: Candidate package version.

        Returns:
            None.

        Raises:
            ValueError: If the tag already exists.
        """
        if self.run("git", "tag", "--list", f"v{version}"):
            raise ValueError(f"Tag v{version} already exists; tags are never replaced")

    def release_notes(self, version: str) -> str:
        """Group committed subjects since the current release tag.

        Args:
            version: Current package version whose local tag is the baseline.

        Returns:
            Markdown notes in chronological order within each category.

        Raises:
            ValueError: If the baseline tag is missing or not an ancestor of HEAD.
        """
        tag = f"v{version}"
        if not self.run("git", "tag", "--list", tag):
            raise ValueError(f"Missing baseline tag {tag}; fetch release tags before bumping")
        try:
            self.run("git", "merge-base", "--is-ancestor", tag, "HEAD")
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Baseline tag {tag} is not an ancestor of HEAD") from error
        history = self.run("git", "log", "--reverse", "--no-merges",
                           "--format=%h%x09%s", f"{tag}..HEAD", "--")
        groups: dict[str, list[str]] = {}
        categories = {"feat": "Added", "fix": "Fixed", "perf": "Performance",
                      "refactor": "Changed", "docs": "Documentation", "test": "Tests",
                      "build": "Build and release", "ci": "Build and release",
                      "chore": "Maintenance", "style": "Maintenance", "revert": "Reverted"}
        for line in history.splitlines():
            sha, subject = line.split("\t", 1)
            match = re.match(r"([a-z]+)(?:\([^)]*\))?(!)?:\s+", subject)
            category = categories.get(match.group(1), "Other") if match else "Other"
            if match and match.group(2):
                category = "Breaking changes"
            groups.setdefault(category, []).append(f"- {subject} (`{sha}`).")
        return "\n\n".join(f"### {category}\n\n" + "\n".join(entries)
                            for category, entries in groups.items())

    def bump(self, part: str) -> str:
        """Generate commit notes, preserve Unreleased notes and update release files.

        Args:
            part: major, minor or patch (numeric SemVer increments).

        Returns:
            New version, committed locally with a chore(release) message.

        Raises:
            ValueError: If notes, metadata or the candidate tag are invalid.
            subprocess.CalledProcessError: If uv or git fails; edits remain for inspection.
        """
        paths = ("pyproject.toml", "CHANGELOG.md", "uv.lock")
        if self.run("git", "diff", "--cached", "--name-only"):
            raise ValueError("Unstage existing changes before bumping")
        if self.run("git", "status", "--porcelain", "--", *paths):
            raise ValueError("Commit existing release-file changes before bumping")
        old = self.version()
        numbers = list(map(int, old.split(".")))
        index = ("major", "minor", "patch").index(part)
        numbers[index] += 1
        numbers[index + 1:] = [0] * (2 - index)
        new = ".".join(map(str, numbers))
        self.ensure_new_tag(new)
        changelog_path = self.root / "CHANGELOG.md"
        changelog = changelog_path.read_text(encoding="utf-8")
        section = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[|\Z)", changelog, re.MULTILINE | re.DOTALL)
        if section is None:
            raise ValueError("Missing ## [Unreleased] changelog section")
        generated = self.release_notes(old)
        notes = "\n\n".join(text for text in (section.group(1).strip(), generated) if text)
        if not notes:
            raise ValueError("No release notes or commits since the current release tag")
        if re.search(rf"^## \[{re.escape(new)}\]", changelog, re.MULTILINE):
            raise ValueError(f"Changelog already contains {new}")
        metadata_path = self.root / "pyproject.toml"
        metadata = metadata_path.read_text(encoding="utf-8")
        metadata, count = re.subn(rf'^version = "{re.escape(old)}"$', f'version = "{new}"', metadata, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected exactly one version = "X.Y.Z" metadata line')
        replacement = f"## [Unreleased]\n\n## [{new}] - {date.today().isoformat()}\n" + "\n" + notes + "\n\n"
        changelog = changelog[:section.start()] + replacement + changelog[section.end():]
        changelog = re.sub(r"^(\[Unreleased\]: .*?/compare/)v[^.]+\.\d+\.\d+\.\.\.HEAD$", rf"\g<1>v{new}...HEAD", changelog, flags=re.MULTILINE)
        previous_link = re.search(rf"^\[{re.escape(old)}\]: (https://github.com/[^/]+/[^/]+)/", changelog, re.MULTILINE)
        if previous_link:
            changelog += f"[{new}]: {previous_link.group(1)}/compare/v{old}...v{new}\n"
        metadata_path.write_text(metadata, encoding="utf-8")
        changelog_path.write_text(changelog, encoding="utf-8")
        self.run("uv", "lock")
        check_release(self.root, f"v{new}")
        self.run("uv", "lock", "--check")
        self.run("git", "add", "--", *paths)
        self.run("git", "commit", "-m", f"chore(release): bump version to {new}", "--", *paths)
        return new

    def tag(self) -> str:
        """Tag HEAD after checking that release files are committed and locked.

        Args:
            None.

        Returns:
            Created annotated tag name.

        Raises:
            ValueError: If release files are dirty or a tag exists.
            subprocess.CalledProcessError: If validation or git tagging fails.
        """
        version = self.version()
        self.ensure_new_tag(version)
        paths = ["pyproject.toml", "uv.lock", "CHANGELOG.md", "scripts", "tests", ".github", "README.md", "docs", "llamactl"]
        if self.run("git", "status", "--porcelain", "--", *paths):
            raise ValueError("Commit release files before tagging; unrelated local files may remain dirty")
        check_release(self.root, f"v{version}")
        self.run("uv", "lock", "--check")
        self.run("git", "tag", "-a", f"v{version}", "-m", f"Release {version}", "HEAD")
        return f"v{version}"


def main() -> None:
    """Execute an explicitly requested bump or local tag operation.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If validation or a subprocess fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bump").add_argument("part", choices=("patch", "minor", "major"))
    commands.add_parser("tag")
    args = parser.parse_args()
    helper = ReleaseHelper(Path(__file__).resolve().parents[1])
    try:
        result = helper.bump(args.part) if args.command == "bump" else helper.tag()
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release operation failed: {error}\n")
    print(f"Prepared {result}. Nothing pushed.")


if __name__ == "__main__":
    main()
