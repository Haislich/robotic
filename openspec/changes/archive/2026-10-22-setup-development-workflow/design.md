# Design

## Context

See `proposal.md` for motivation. Current state relevant to the approach:

- `pyproject.toml` lists ruff, ty, mypy, pytest as runtime dependencies, has no ruff or ty configuration, and a leftover `[[tool.mypy.overrides]]` for sympy.
- `src/robotic/` (3 modules) imports numpy, sympy, loguru, matplotlib and pandas. It is legacy code awaiting a refactor and will not pass the new checks.
- `.claude/settings.json` only enables plugins; Serena is enabled but not activated for this project.
- `openspec/config.yaml` is the untouched template; there are no specs.
- `README.md` is empty; there is no `CLAUDE.md` and no `docs/`.

Versions verified on PyPI on 2026-10-22: ruff 0.17.2, ty 0.0.91, ast-grep-cli 0.51.0, pre-commit 4.7.0, pytest 9.1.1, hypothesis 6.170.6, roboticstoolbox-python 1.4.4 (wheels up to cp314), numpy 2.5.4, sympy 1.14.0, polars 2.0.0.

## Goals / Non-Goals

**Goals:**
- Every rule that can be checked by a tool is checked by a tool, at edit time, with output proportional to the number of violations.
- One source of truth per concern: tool versions in `uv.lock`, lint policy in `pyproject.toml`, working contract in `CLAUDE.md`, artifact conduct in `openspec/config.yaml`.
- Every lint exception is visible in one place and justified here.

**Non-Goals:**
- Making legacy code pass the checks. The refactor does that.
- A project-wide gate. It would re-report legacy violations on every turn.

## Decisions

### D1. Python 3.14, not 3.15
3.15 is released, but roboticstoolbox-python has no `cp315` wheels and contains compiled extensions; building it from source is the kind of unvetted risk the dependency policy rejects. 3.14 brings lazily evaluated annotations (PEP 649/749) and template strings (PEP 750). Moving to 3.15 is a one-line change once wheels exist.

### D2. ruff: everything on, exceptions only in configuration
`select = ["ALL"]`, `preview = true` (the `DOC` pydoclint rules are preview-only), `target-version = "py314"`, pydocstyle `convention = "google"`. Setting the convention disables the mutually exclusive `D2xx` pairs automatically, so they need no manual ignore.

Inline suppression is banned, and the ban is enforced mechanically rather than stated. Verified against the pinned versions:
- ruff: `--ignore-noqa` in the hook and pre-commit neutralizes both `# noqa` and ruff's own `# ruff: ignore[...]` comments.
- ty: `respect-type-ignore-comments = false` neutralizes `# type: ignore`. ty has no switch for its own `# ty: ignore[...]` comments, so the hook and pre-commit reject files containing them.

Exceptions, all in `pyproject.toml`:

| Exception | Scope | Reason |
|---|---|---|
| `COM812` | global | The only rule ruff 0.17.2 reports at runtime as conflicting with `ruff format` under the default formatter settings. The other rules in the documented list are redundant with the formatter, not conflicting, so they stay on. |
| `CPY001` | global | Personal study repository with no license or distribution; a copyright header in every file would be noise. |
| `pep8-naming.ignore-names` = matrix symbols (`R`, `J`, `T`, ...) | global | Domain notation used in lectures and exams; renaming makes formulas harder to check against the source. This exempts the names, not the rules `N803`/`N806`. |
| `S101` | `tests/**` | pytest is built on `assert`. |
| `D1` (missing docstrings) | `tests/**` | A test's name is its contract; a docstring would restate it. |
| `TID251` | `tests/**` | Tests are where oracle libraries are allowed (D4). |
| `S404`, `S603` | `.claude/hooks/*.py` | Bandit flags every `subprocess` use regardless of arguments. The hook exists to run fixed developer tools, from the project environment, on paths given by Claude Code or pre-commit. |
| `INP001` | `.claude/hooks/*.py` | A standalone script, not part of a package. |

Considered and rejected: ignoring `PLR2004` globally. It only flags magic values in comparisons, not in arithmetic such as `pi / 2`, so it does not conflict with formulas and stays on.

### D3. ty: every rule raised to error, version pinned exactly
ty has no strict mode; strictness is per-rule severity in `[tool.ty.rules]`. The pinned version accepts `all = "error"`, so no enumeration is needed, and the Python version is inferred from `requires-python`. ty is pre-1.0, so the pin is exact and upgrades are deliberate changes.

Two analysis options, added during implementation, trade convenience for soundness: `strict-equality-semantics` (no `Literal[True]`/`Literal[False]` inference from equality checks that may be unsound) and `strict-generic-narrowing` (`isinstance(x, list)` narrows to all possible `list` specializations, not a gradual one). If either proves impractical with numpy or sympy, removing it is a recorded decision.

Rejected: keeping mypy alongside ty (duplicate diagnostics, two configurations to reconcile), and pylint (ruff covers its useful rules under `PL`).

### D4. Oracle libraries banned from `src/`
`roboticstoolbox` and `spatialmath` are listed in `flake8-tidy-imports.banned-api` (`TID251`), allowed only in `tests/`. Algorithms are implemented by hand; the oracle exists only to check them on random configurations with hypothesis. Rejected: an ast-grep rule for this, as ruff already expresses it.

### D5. Dependency layout and version policy
- Runtime: numpy, sympy; plus loguru, matplotlib, pandas because legacy code imports them, to be removed by the refactor.
- `dev`: ruff, ty, ast-grep-cli (exact pins), pytest, hypothesis, pre-commit, roboticstoolbox-python, pandas-stubs (while pandas is a dependency).
- `notebooks`: ipykernel, ipympl, polars. Notebooks are excluded from lint.
- Removed: mypy, pylint, nbqa (ruff reads notebooks natively, and notebooks are out of scope anyway), pyqt6 (ipympl covers interactive plots in notebooks), mpmath (transitive through sympy).

Tools whose behavior defines pass/fail get exact pins; libraries get lower bounds, with `uv.lock` pinning the resolved versions.

`[tool.uv] default-groups = "all"`: `uv run` syncs the environment to the default groups, so with only `dev` as default every hook run would uninstall the notebook packages. Also removed: the `robotic = "robotic:main"` script entry, which pointed to a function that does not exist.

The lock was regenerated with `--upgrade`: previously locked versions (pillow, pandas, contourpy, pyzmq, kiwisolver) had no Python 3.14 wheels and failed to build from source.

### D6. Edit-time hook: file-scoped, silent on success
A `PostToolUse` hook on `Edit|Write` runs a Python script in `.claude/hooks/` (the script is in lint scope itself). The hook command starts the script with `uv run --frozen --quiet`; the script runs each tool as a module of that same interpreter (`python -m ruff`, `python -m ty`), so every check uses the locked versions without a second resolution. For an edited `*.py` file it runs:
1. `ruff format --quiet --force-exclude` on the file,
2. `ruff check --quiet --force-exclude --ignore-noqa --output-format concise` on the file,
3. `ty check --quiet --output-format concise` on the file,
4. the `ty: ignore` comment scan (D2).

It prints nothing and exits 0 when all pass; otherwise it prints the concise diagnostics and exits 2, which returns them to Claude. `--force-exclude` makes ruff honor the configured excludes for explicitly passed paths, so edits to notebooks or excluded folders are not checked.

Cost: zero tokens when clean, one line per diagnostic otherwise. Rejected: checking the whole tree per edit (re-reports legacy failures every time); running only at end of turn (errors compound across edits before being seen).

The same script serves pre-commit: given file paths as arguments, it runs the same checks, so the checks are defined in one place. The `ty: ignore` ban is a tokenize-based scan of comment tokens, so string literals never match.

Rejected during implementation: a bash hook with the `ty: ignore` ban as an ast-grep rule. It needs no lint exceptions, but moves the checker out of ruff and ty coverage; the typed Python script with three recorded exceptions was preferred.

Writes go through `Edit`/`Write` only; Serena is used for reading and navigation. This keeps every write covered by the hook without matching Serena's tool names and parsing their inputs.

### D7. pre-commit through the locked environment
`.pre-commit-config.yaml` uses one `repo: local` hook with `language: system` and `entry: uv run --frozen --quiet python .claude/hooks/check_python.py`, which runs ruff format, ruff check, ty and the suppression ban (D6). Rejected: the upstream `ruff-pre-commit` repository, because its `rev` is a second version pin that drifts from `uv.lock`. pytest is not run at commit time; it belongs to the project-wide gate added after the refactor.

### D8. Responsibilities of each documentation home
| Home | Contains | Does not contain |
|---|---|---|
| Docstrings | Contract: summary, invariants, pre/postconditions, units and frames, `Returns` (meaning of the value, not its type; DOC201), `Raises` (DOC501), `Args` (complete whenever present; D417) | Rationale, history, alternatives |
| Comments | Rare; a function needing more comment than code is a design smell | Restating code |
| `docs/` | Wiki: theory, derivations, usage | Decisions |
| `openspec/` | Rationale, alternatives considered, rejected ideas, decisions | How-to content |
| `README.md` | What the repository is and how to set it up | Everything else |

### D9. `CLAUDE.md` and `openspec/config.yaml` split
`CLAUDE.md` governs every session: roles (user designs, Claude implements and challenges), disagreement, code style, documentation homes (D8), navigation order (Serena symbol tools, then ast-grep for structural patterns, then bounded `rg`; no whole-file dumps, quiet command flags), repository-wide consistency (search for existing equivalents before adding a symbol, check references after changing one), and the commands to run.
`openspec/config.yaml` governs artifacts and operations only, and refers to `CLAUDE.md` instead of repeating it.

### D10. The deviation loop
`operations.apply.guidance` defines a deviation as anything that invalidates a decision in `design.md`, a spec requirement, or the proposal's scope. On deviation, Claude stops, explains, proposes the artifact revision, and waits; after approval the artifacts are updated before code continues, and already-completed tasks are re-checked against the revision. Mechanical adjustments (renames, task reordering, helper extraction) are not deviations and are listed in the end-of-task summary.
`operations.archive.guidance` requires a reconciliation before archiving: the whole diff against the artifacts, a repository-wide consistency pass (naming, duplicated helpers, sibling modules), and artifacts updated to reflect what was actually built, with superseded ideas kept as rejected alternatives.
Artifact rules require `design.md` to record alternatives considered for each decision.

### D11. Documentation site generator deferred
Rejected for now: mkdocs-material + mkdocstrings (its maintainers moved development to Zensical; last mkdocs-material release July 2026) and Zensical (0.0.x). Both are a gamble today and there is no content yet. `docs/` is established as a convention in `CLAUDE.md`; the generator is chosen when there is content to publish.

### D12. Serena activation
Activate the project so `.serena/project.yml` exists and is committed; Serena's cache directory is git-ignored.

## Risks / Trade-offs

- [roboticstoolbox constrains numpy or sympy versions incompatibly with Python 3.14] → Resolution is verified by `uv lock` early in the tasks; on failure, implementation stops and the alternative is discussed (D1 is a decision, so this is a deviation).
- [ty is pre-1.0; rules and configuration keys may change] → Exact pin; upgrades are their own change.
- [ruff preview rules may change behavior between versions] → Exact pin; same policy.
- [Hook latency on every edit] → File-scoped runs of Rust tools; `uv run --frozen` skips resolution.
- [Legacy files fail checks when first touched] → Expected: touching a file means it is being refactored.
- [`ignore-names` grows into a loophole] → Every addition is a decision recorded in an OpenSpec design.
- [`CLAUDE.md` rules that tools cannot check erode over long sessions] → Keep it short; anything checkable moves into tool configuration.

## Migration Plan

Legacy code is not migrated here. Rollback: revert the commit; `uv sync` restores the previous environment from the reverted lock file.
