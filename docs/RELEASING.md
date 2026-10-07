# Releases and PyPI publishing

## Version policy

`pyproject.toml` is the authoritative package version. Use stable Semantic Versioning
(`MAJOR.MINOR.PATCH`) and matching annotated tags (`v0.1.0`). The publishing workflow
rejects mismatched tags, leading zeroes, prereleases, post-releases and versions
without a dated changelog heading.

The supported public interfaces are the `llamactl` command, `python -m llamactl`,
documented configuration behavior, and documented UI operations. Internal Python
modules are not a stable library API.

- Before 1.0: bump MINOR for new features or incompatible interface/config changes;
  bump PATCH for backward-compatible fixes. Development interfaces may change.
- From 1.0: bump MAJOR for incompatible changes, MINOR for compatible features,
  and PATCH for compatible fixes.
- Documentation and test changes alone do not require a release. Packaging-only
  fixes that need republication receive a new PATCH version; never reuse a PyPI version.

The current prepared release is 0.1.3. PyPI rejected the 0.1.0, 0.1.1 and 0.1.2
wheel filenames because they had previously been used and deleted; the original
tags remain intact. Deleted filenames cannot be reused, so rerunning those tags
will fail again. The 0.1.0 changelog covers all 24 initial source commits and the
packaging implementation.

## Local release helper

Commit your changes, then choose a numeric SemVer increment:

```bash
python scripts/bump_version.py bump patch   # or minor / major
```

This generates release notes from non-merge commit subjects in `vCURRENT..HEAD`,
grouped by Conventional Commit type and including short commit hashes. Subjects
with `!` are grouped as Breaking changes; unrecognized subjects appear under Other.
Within each category, commits retain chronological order. Commit bodies (including
BREAKING CHANGE footers) are not parsed; review the generated notes yourself.
Existing Unreleased notes are preserved alongside generated notes. Manual notes
are optional; a bump with neither notes nor new commits is refused. The current
version's local tag must exist and be an ancestor of HEAD. Fetch tags yourself if
needed. Only committed changes are included; merge commits are omitted to avoid
repeating their constituent commits.

This updates `pyproject.toml`, creates a dated changelog heading, updates comparison
links and runs `uv lock` and `uv lock --check`. It then automatically creates
`chore(release): bump version to X.Y.Z`, containing only the changed release files:
`pyproject.toml`, `CHANGELOG.md` and `uv.lock`. It does not tag or push.
Existing staged changes or dirty release files are refused before any edits;
commit manual Unreleased notes before bumping. Unrelated unstaged files stay out
of the commit. Review the generated commit before tagging.
Review and update any version-specific prose in README/docs yourself. If `uv lock`
fails, the metadata and changelog edits remain for inspection; repair the lockfile
before continuing rather than rerunning the bump and incrementing twice.

After reviewing the bump commit, tests/build validation, and committing any additional release edits:

```bash
python scripts/bump_version.py tag
```

This checks release metadata, the lockfile and committed release paths, refuses
existing local tags, and creates an annotated tag on HEAD. Unrelated local changes
(such as agent configuration) may remain dirty. Remote tag names are not checked;
never force-replace an existing remote tag. The helper does not run tests/builds
or commit changes during tagging and never pushes. Push the commit and tag yourself.

## One-time setup (maintainer)

1. Confirm `llama-tui` is available on PyPI or that you control the existing project.
   No name reservation or ownership check has been performed by this implementation.
2. Choose a repository license and add its license file and package metadata.
   No license has been invented. PyPI metadata currently makes no license claim.
3. In GitHub repository `thesawdawg/llamactl`, create an environment named `pypi`.
   Add required reviewers and restrict deployment to release tags as appropriate.
4. On PyPI, add a GitHub Trusted Publisher (or pending publisher for a new project):
   - Owner: `thesawdawg`
   - Repository: `llamactl`
   - Workflow: `publish.yml`
   - Environment: `pypi`
5. Enable GitHub Actions and ensure organization/repository policy allows the
   actions used by `.github/workflows/publish.yml`.

The PyPI distribution is `llama-tui`; the command, import package, configuration
directory and GitHub repository remain `llamactl`. Configure the pending PyPI
project name as `llama-tui`, not `llamactl`.

No PyPI API token or GitHub PAT is required. OIDC permission is limited to the
publish job; the build job has read-only repository access. The publish job only
receives validated distributions via a GitHub artifact.

## Release checklist

1. Commit the changes to include in release notes and run `python scripts/bump_version.py bump patch`
   (or `minor` / `major` according to the version policy).
2. Move relevant Unreleased notes into a dated `## [X.Y.Z] - YYYY-MM-DD` section.
   Preserve the previous release notes and update comparison links.
3. Validate locally:

   ```bash
   uv sync --locked --extra dev
   uv run --locked pytest -q
   uv run --locked python scripts/check_release.py vX.Y.Z
   uv build
   uvx --from twine twine check --strict dist/*
   ```

4. Inspect wheel and source-distribution contents. Ensure the release directory
   contains only the intended version; stale artifacts must not be published.
5. Commit the release changes using Conventional Commits, then create an annotated
   `vX.Y.Z` tag on that commit. The maintainer pushes the commit and tag manually;
   the agent never pushes. No automated bump workflow writes back to the repository.
6. The tag triggers tests, version checks, a wheel/source build and metadata checks.
   Approve deployment in the `pypi` environment if reviewers are configured.
7. Verify PyPI installation in a fresh environment and create GitHub release notes
   from the corresponding changelog section. GitHub releases are not auto-created.

The workflow is adapted from `thesawdawg/dev-setup-py`, whose PyPI project is named
`devstuff`. Its `uv build` → artifact → Trusted Publishing structure is retained;
its automatic push/bump and PEP 440 `.postN` tracks are intentionally not copied.

## Reproducibility and coverage

Commit `uv.lock`. CI runs locked dependencies on Python 3.11–3.14 on Linux,
builds distributions, and checks metadata. Runtime requirements stay as compatible
lower bounds for package consumers; the lock pins development/CI resolution.
Build-backend and release-check tool versions are not locked, so this is not a
bit-for-bit reproducible build claim. Non-Linux platforms and real llama.cpp/GPU
integration require separate maintainer verification.
