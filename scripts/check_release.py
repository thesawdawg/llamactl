"""Validate a stable release tag against package metadata and the changelog."""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

STABLE_VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", re.ASCII)


def check_release(root: Path, tag: str) -> str:
    """Check that a release uses one consistent, stable Semantic Version.

    Args:
        root: Repository root containing pyproject.toml and CHANGELOG.md.
        tag: Release tag in vMAJOR.MINOR.PATCH form.

    Returns:
        Validated package version.

    Raises:
        ValueError: If the version, tag or changelog is inconsistent.
    """
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    if not STABLE_VERSION.fullmatch(version):
        raise ValueError(f"Expected stable MAJOR.MINOR.PATCH version, got {version!r}")
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match package version v{version}")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", re.MULTILINE)
    if not heading.search(changelog):
        raise ValueError(f"Missing dated changelog heading for {version}")
    return version


def main() -> None:
    """Validate the command-line tag and exit on an invalid release.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If usage or release validation fails.
    """
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/check_release.py vMAJOR.MINOR.PATCH")
    try:
        version = check_release(Path(__file__).resolve().parents[1], sys.argv[1])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Release validated: v{version}")


if __name__ == "__main__":
    main()
