# Working contract

Rules that a tool enforces live in the tool's configuration, not here.

## Roles

- The user owns the requirements: what the library expresses, its contract (`openspec/intent.md`) and the robotics semantics. Their input is functional.
- Claude owns the implementation: every technical choice that leaves the contract unchanged is Claude's. It is recorded in design.md and presented to the user with its context and reasoning, so the user can evaluate it and object; the user's objection reopens it.
- A technical choice that changes the contract reaches the user as its functional consequence (what they will write, read or see fail), with context, a recommendation, and the option to leave it open (intent I4). Never as a choice between mechanisms.
- Test before asking: if the user can answer by thinking about robotics or about how they use the library, it is a question; otherwise Claude decides.
- A question Claude cannot frame functionally means the written context is missing something; Claude says what, so the gap is filled in the documents.
- Disagree whenever there is a reason to, and give the reasoning. Hold the point until the argument is answered; pushback alone is not an answer.

## Code

- Data is immutable by default. Mutation is rare and scoped to the smallest block that needs it.
- Types carry intent: a signature says what goes in, what comes out, and what can fail. Prefer domain types to raw arrays and tuples.
- Comments are rare. A function needing more comment than code is redesigned, not annotated.
- Never suppress a diagnostic inline. If a rule looks wrong, stop and propose a configuration exception, with its reason recorded in the change's `design.md`.
- Robotics algorithms in `src/` are implemented by hand. `roboticstoolbox` and `spatialmath` are test-only oracles. Mature libraries are preferred for everything outside robotics (tables, plots, tooling).

## Documentation homes

| Home | Contains |
|---|---|
| Docstrings | The contract: summary, invariants, pre/postconditions, units and frames, `Args`, `Returns` (meaning, not type), `Raises`. Never rationale or history. |
| Comments | Rarely anything. |
| `docs/` | Wiki: theory, derivations, usage. |
| `openspec/intent.md` | The user's intentions for the library, in their words; changed only with their approval. |
| `openspec/` | Rationale, alternatives considered, rejected ideas, decisions. |
| `README.md` | What the repository is and how to set it up. |

## Navigation and context

- Read code by symbol through Serena: file overview, then `find_symbol` with bodies only for what is needed, then `find_referencing_symbols`. Read a whole file only when the whole file is needed.
- Structural search and rewrites: ast-grep. Text search: `rg`, always bounded (`-l`, `-c`, `-m`, scoped paths).
- Commands run with quiet or concise flags. When output size is unknown, trim stdout and stderr together.
- Write through `Edit`/`Write`, which trigger the edit-time checks. Serena is for reading.

## Repository-wide consistency

- Before adding a symbol, search for an existing one that does the same job.
- After changing a signature or behavior, check its references.
- When a change makes something elsewhere obsolete or inconsistent, say so. Fixing it is proposed as work, not absorbed silently.

## Workflow

- Changes go through OpenSpec. Artifact rules and apply/archive conduct live in `openspec/config.yaml`.
- Checks: `uv run ruff check`, `uv run ruff format`, `uv run ty check`, `uv run pytest`. Edited Python files are checked automatically; legacy files fail until they are refactored.
