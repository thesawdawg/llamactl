# Changelog

All notable changes are documented here. Versions follow [Semantic Versioning](https://semver.org/).
This history covers every existing commit from the initial import through `bb79f6c`;
there are no earlier tagged releases. Commit entries retain their original subjects.

## [Unreleased]

### Fixed

- Prepare a new PyPI release because the 0.1.0, 0.1.1 and 0.1.2 wheel
  filenames were previously used and deleted; PyPI permanently rejects reuse.

## [0.1.2] - 2026-10-06

### Added

- feat(release): generate notes from commit messages (`d1a66e0`).

## [0.1.1] - 2026-10-06

### Fixed

- Release as 0.1.1 because PyPI rejected the previously used and deleted
  `llama_tui-0.1.0` wheel filename. Preserve the original 0.1.0 tag.

### Build and release

- Add a local Semantic Version bump and annotated-tag helper; never push automatically.


## [0.1.0] - 2026-10-03

Initial development release. The date records the existing source history, not a
claim that a PyPI publication has occurred. This release provides a Textual UI for
model discovery, GGUF metadata and caching, host-aware memory recommendations,
profile editing, Hugging Face downloads, server sessions, logs and chat integration.

### Build and release

- Use `llama-tui` as the PyPI distribution name because `llamactl` is taken;
  preserve the `llamactl` command, Python module and configuration paths.

- Complete PyPI description, classifiers, project links and explicit distribution contents.
- Add a uv lockfile, multi-version CI, tag/version/changelog validation and an isolated
  build-to-publish workflow using PyPI Trusted Publishing.
- Document installation, release setup and Semantic Versioning policy.
- Add release-validator regression tests.

### Added

- feat(schema): add settings registry with hints, help and impact lines ([`18f2e98`](https://github.com/thesawdawg/llamactl/commit/18f2e98c192083e9b88ef8693a0d868f52a580eb)).
- feat(config): schema-driven Profile, version 2, command builder module ([`b9b62af`](https://github.com/thesawdawg/llamactl/commit/b9b62af1c0aba41f8120ceaf7a699f8b7af99dd0)).
- feat(gguf): GGUF header reader with mtime-keyed disk cache ([`e0bd3fb`](https://github.com/thesawdawg/llamactl/commit/e0bd3fb8c817f2d2b129bdc40d5b4b4f0471e3ab)).
- feat(recommend): memory budget and profile recommender ([`86eaf8e`](https://github.com/thesawdawg/llamactl/commit/86eaf8e7fc1c0d367fe0a21931f7c71c42a61a36)).
- feat(ui): profile editor with typed fields, live impact and help ([`7e5def5`](https://github.com/thesawdawg/llamactl/commit/7e5def50125bc7767cbba14dc673810f88410f64)).
- feat(ui): models view with metadata, budget and progressive scan ([`11bed9d`](https://github.com/thesawdawg/llamactl/commit/11bed9d7b34ff34df9ae01a2257927353bddf510)).
- feat(ui): sessions view with live states, slots, log and confirm ([`a29a7e1`](https://github.com/thesawdawg/llamactl/commit/a29a7e1975eb445666fc63012030c51d028fd080)).
- feat(ui): recommendation screen and post-download hand-off ([`45e2279`](https://github.com/thesawdawg/llamactl/commit/45e22792acdea85122bc62eb3f50a8fd71dcea0f)).
- feat(ui): hugging face view inside the shell ([`951a474`](https://github.com/thesawdawg/llamactl/commit/951a47496256d984ca94808853447a87e531586d)).
- feat(ui): settings view with validation ([`6c3427f`](https://github.com/thesawdawg/llamactl/commit/6c3427ffeb8a9ec5b2ae29b130fcf1d4519ccb1b)).
- feat(ui): loading banner with log progress while a model loads ([`bb79f6c`](https://github.com/thesawdawg/llamactl/commit/bb79f6cd83961e51793a0e4f98b4d2efd6714030)).

### Fixed

- fix(recommend): ctx is a shared KV pool, not per-slot; drop dead version read ([`be0fe41`](https://github.com/thesawdawg/llamactl/commit/be0fe411221cc9096da307db9b62b07a0f23d8ff)).
- fix(ui): F-key bindings so editor actions work while typing ([`10701e7`](https://github.com/thesawdawg/llamactl/commit/10701e754f7bc5673c5808ecfcf6e3dc6a9017b3)).
- fix(ui): focus active view, button overflow, profile summary ([`a30a7f6`](https://github.com/thesawdawg/llamactl/commit/a30a7f6871d28d27141891c99878c47135bfb28f)).
- fix(ui): hide idle HF progress bar, verdict honours headroom ([`de879dd`](https://github.com/thesawdawg/llamactl/commit/de879dd862012a3f0d3c0bd1d472a17fa15f8244)).
- fix(sessions): bind-based port check and launch off the UI thread ([`a06c664`](https://github.com/thesawdawg/llamactl/commit/a06c6646bac749f14ff1b1aa90716239731c84e5)).

### Changed

- refactor(ui): sidebar shell with views and resource gauges ([`417a3db`](https://github.com/thesawdawg/llamactl/commit/417a3db8d7804e387bdb983abeb72c80eaf457f6)).

### Tests

- test: pytest suite for schema, command, gguf, recommend, config ([`a954700`](https://github.com/thesawdawg/llamactl/commit/a9547005b2046ffc0c20d19c643eb8a57e2d4d24)).
- test: pilot tests for the profile editor ([`ab1c116`](https://github.com/thesawdawg/llamactl/commit/ab1c116c48e609559639e8d3081396547f739792)).
- test: shell and models view pilot tests ([`34ecef6`](https://github.com/thesawdawg/llamactl/commit/34ecef6b00959a86218a5e801d6769ab29746091)).
- test: sessions view, confirm screen and focus tests ([`2f657b9`](https://github.com/thesawdawg/llamactl/commit/2f657b9727d31fc833562f1bc7e00299473724c5)).
- test: recommend, prompt, settings and hf view tests ([`0fdc965`](https://github.com/thesawdawg/llamactl/commit/0fdc9651c68a42e71c61d9908a3ec62a5d8c776b)).

### Documentation

- docs: README ([`e197da5`](https://github.com/thesawdawg/llamactl/commit/e197da53d892d7439c41c4020d5bf0ae1a165057)).

### Maintenance

- chore: initial import of llamactl TUI and redesign spec ([`cb1089e`](https://github.com/thesawdawg/llamactl/commit/cb1089e36e23655c937d1de54df2b674b87e1463)).

[Unreleased]: https://github.com/thesawdawg/llamactl/compare/v0.1.2...HEAD
[0.1.0]: https://github.com/thesawdawg/llamactl/releases/tag/v0.1.0
[0.1.1]: https://github.com/thesawdawg/llamactl/compare/v0.1.0...v0.1.1
[0.1.2]: https://github.com/thesawdawg/llamactl/compare/v0.1.1...v0.1.2
