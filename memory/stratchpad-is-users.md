---
name: stratchpad-is-users
description: "stratchpad/ (spelled so) is the user's folder for spot probes: never edit it, never treat it as tests or project code"
metadata:
  type: project
  modified: 2026-10-26
---

`stratchpad/` at the repo root (spelled with a t, untracked) holds the user's spot probes, e.g. `stratchpad/t.py`. It is excluded from ruff (`extend-exclude`), ty (`[tool.ty.src] exclude`) and pytest (`testpaths = ["tests"]`) in `pyproject.toml`.

**Why:** user (2026-10-26): "scratchpad è uno scratchpad mio dove faccio prove spot e va ignorato per i test veri". Earlier its `t.py` was the only failure of the full check run.

**How to apply:** never edit or delete it; a probe the user shows from it may become a real test in `tests/`, written by Claude there. Supersedes the "leave scratchpad alone" note in [[refactor-design-state]].
