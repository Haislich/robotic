---
name: stratchpad-is-users
description: "stratchpad/ (spelled so) is the user's folder for spot probes: never edit it, never treat it as tests or project code"
metadata:
  type: project
  modified: 2026-10-26
---

`stratchpad/` at the repo root (spelled with a t; committed with `a74929f`, so it syncs between machines) holds the user's spot probes, e.g. `stratchpad/t.py`. pytest ignores it (`testpaths = ["tests"]`). The user commented out its ruff and ty exclusions in `pyproject.toml` on 2026-10-26 (along with the notebook exclusion), so `uv run ruff check` reports its findings: they are the user's, not to be fixed; report them as such.

**Why:** user (2026-10-26): "scratchpad è uno scratchpad mio dove faccio prove spot e va ignorato per i test veri". Earlier its `t.py` was the only failure of the full check run.

**How to apply:** never edit or delete it; a probe the user shows from it may become a real test in `tests/`, written by Claude there. Supersedes the "leave scratchpad alone" note in [[refactor-design-state]].
