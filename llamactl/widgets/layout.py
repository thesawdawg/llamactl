"""Shared layout constants: the responsive breakpoints used app-wide.

The app's screens set these as their `HORIZONTAL_BREAKPOINTS` / `VERTICAL_BREAKPOINTS`
(a bare Harness `App` has none, so the screens carry the lists themselves): below 100
columns the screen gets `-narrow`, else `-wide`; below 32 rows it gets `-short`, else
`-tall`. Breakpoint rules in widget and screen CSS match on those classes.
"""

HORIZONTAL_BREAKPOINTS: list[tuple[int, str]] = [(0, "-narrow"), (100, "-wide")]
VERTICAL_BREAKPOINTS: list[tuple[int, str]] = [(0, "-short"), (32, "-tall")]
