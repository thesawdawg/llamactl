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

    def bump(self, part: str) -> str:
        """Promote Unreleased notes and update metadata and the uv lockfile.

        Args:
            part: major, minor or patch (numeric SemVer increments).

        Returns:
            New version; changes remain uncommitted for review.

        Raises:
            ValueError: If notes, metadata or the candidate tag are invalid.
            subprocess.CalledProcessError: If uv fails; inspect local changes before retrying.
        """
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
        if section is None or not section.group(1).strip():
            raise ValueError("Add release notes under ## [Unreleased] before bumping")
        if re.search(rf"^## \[{re.escape(new)}\]", changelog, re.MULTILINE):
            raise ValueError(f"Changelog already contains {new}")
        metadata_path = self.root / "pyproject.toml"
        metadata = metadata_path.read_text(encoding="utf-8")
        metadata, count = re.subn(rf'^version = "{re.escape(old)}"$', f'version = "{new}"', metadata, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected exactly one version = "X.Y.Z" metadata line')
        replacement = f"## [Unreleased]\n\n## [{new}] - {date.today().isoformat()}\n" + section.group(1)
        changelog = changelog[:section.start()] + replacement + changelog[section.end():]
        changelog = re.sub(r"^(\[Unreleased\]: .*?/compare/)v[^.]+\.\d+\.\d+\.\.\.HEAD$", rf"\g<1>v{new}...HEAD", changelog, flags=re.MULTILINE)
        previous_link = re.search(rf"^\[{re.escape(old)}\]: (https://github.com/[^/]+/[^/]+)/", changelog, re.MULTILINE)
        if previous_link:
            changelog += f"[{new}]: {previous_link.group(1)}/compare/v{old}...v{new}\n"
        metadata_path.write_text(metadata, encoding="utf-8")
        changelog_path.write_text(changelog, encoding="utf-8")
        self.run("uv", "lock")
        check_release(self.root, f"v{new}")
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
