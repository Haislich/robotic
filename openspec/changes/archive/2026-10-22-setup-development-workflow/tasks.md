# Tasks

## 1. Python version and dependencies

- [x] 1.1 Pin Python 3.14 (`.python-version`, `requires-python = ">=3.14"`) and set the project description; verify `uv run python --version` reports 3.14
- [x] 1.2 Restructure dependencies per design D5 (runtime, `dev`, `notebooks`; remove mypy, pylint, nbqa, pyqt6, mpmath and the mypy override; exact pins for ruff, ty, ast-grep-cli); verify `uv lock` resolves. If it fails, STOP and discuss (D1 deviation)
- [x] 1.3 Sync all groups; verify `uv run --frozen python -c "import roboticstoolbox, hypothesis, numpy, sympy"` succeeds and `uv run --frozen ruff --version`, `ty --version`, `ast-grep --version` match the pins

## 2. Lint and type-check policy

- [x] 2.1 Configure ruff per design D2 and D4 (ALL, preview, py314, google convention, notebook and non-code folders excluded, formatter-conflict list taken from the pinned version's docs, `ignore-names`, per-file exceptions for `tests/**`, banned oracle imports); verify with a temporary probe file in `src/robotic/` that an oracle import raises `TID251` and a magic-value comparison raises `PLR2004`, then delete the probe
- [x] 2.2 Configure ty per design D3 (Python 3.14, every warn/ignore rule raised to error, enumerated from the pinned version); verify with a temporary probe that a previously warn-level diagnostic is now reported as error, then delete the probe
- [x] 2.3 Determine the inline-suppression enforcement mechanism for the pinned ruff and ty (design D2); verify a probe with `# noqa` and `# type: ignore` still fails the check, then delete the probe
- [x] 2.4 Record the baseline: count of ruff and ty diagnostics per legacy file, kept in the end-of-change summary (not committed) as the starting point for the refactor

## 3. Edit-time hook

- [x] 3.1 Write the hook script in `.claude/hooks/` per design D6, using the enforcement mechanism from 2.3; verify the script itself passes ruff and ty
- [x] 3.2 Register it as a `PostToolUse` hook on `Edit|Write` in `.claude/settings.json`; verify: editing a clean probe produces no output, a violating probe returns exit 2 with one concise line per diagnostic, and editing a notebook or a non-Python file is skipped; delete the probes

## 4. pre-commit

- [x] 4.1 Add `.pre-commit-config.yaml` with local `uv run --frozen` hooks per design D7 and install the git hook; verify `uv run pre-commit run --files .claude/hooks/*.py` passes

## 5. Serena

- [x] 5.1 Activate the project for Serena and git-ignore its cache (design D12); verify a symbol overview of `src/robotic/transformations.py` returns symbols and `git status` shows only `.serena/project.yml`

## 6. Working contract

- [x] 6.1 Draft `CLAUDE.md` per design D8 and D9, kept short with nothing a tool already enforces; verify by presenting the draft to the user and getting approval before considering the task done
- [x] 6.2 Fill `openspec/config.yaml` (context, artifact rules, apply and archive guidance) per design D10, referring to `CLAUDE.md` rather than repeating it; verify `openspec instructions design --change setup-development-workflow --json` returns the new rules and the user approves the text

## 7. README

- [x] 7.1 Write `README.md`: what the repository is, and setup (uv, Python 3.14, dependency groups, pre-commit install, running checks and tests); verify every documented command runs as written

## 8. Integration

- [x] 8.1 Verify the whole setup together: `uv run --frozen ruff check .claude/hooks`, `ty check .claude/hooks`, the pre-commit run and `openspec validate setup-development-workflow` all pass, and `git status` lists only the files named in the proposal's Impact section

## Workflow follow-up

- Before archiving, run the reconciliation from design D10: the diff against proposal and design, and update `design.md` with any decision that changed during implementation.
- Archive the change.
