# llamactl agent notes

Textual TUI for launching and managing llama.cpp servers. Python 3.14 venv at `.venv`.

## Run

```sh
.venv/bin/python -m llamactl
```

## Test

```sh
.venv/bin/pytest -q
```

The v2 design is at [docs/SPEC.md](docs/SPEC.md); implement it in the phases it defines and tick
items in [docs/TODO.md](docs/TODO.md) in the commit that completes them. Work on the
`redesign-v2` branch.

## House rules

- Type hints and docstrings (Args / Returns / Raises) on everything.
- Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`), short and plain.
- Never `git push`. No `Co-Authored-By` trailer.
- No new runtime dependencies; tests use the `dev` extra.
- Never touch files under `~/.config/llamactl` or `~/llama.cpp` from code or tests.
