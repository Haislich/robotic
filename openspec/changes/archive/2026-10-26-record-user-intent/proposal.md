# Proposal

## Why

The review of 2026-10-26 found that `refactor-math-layers` follows its written artifacts faithfully and still diverges from what the user intended. Examples: runtime checks were dropped in favor of static ones; singular configurations compile to unflagged `nan`; the numeric side was read as "secondary". The cause is in the process, not in the code. The user's intentions surfaced in discussions and were recorded only as decisions, without the intention each decision served. Claude starts every session from the written files alone, so the decisions survived and the intentions did not, and later work optimized the letter of the text.

A second gap, of the same kind: `CLAUDE.md` says the user "makes the decisions" without saying which ones. In doubt, Claude brought technical choices to the user as choices between mechanisms, which the user cannot evaluate without implementing the code themselves.

## What Changes

- New `openspec/intent.md`, owned by the user: the library's intentions I0–I7 with the user's own words as source, the declared limits, and the questions deliberately left open. The text to install is `intent.md` in this change.
- `openspec/config.yaml`:
  - its `context` points every artifact to `openspec/intent.md`;
  - new rules: every proposal names the intentions it serves, every design decision names the intention it serves, and an accepted trade-off that goes against an intention is recorded as a violation;
  - apply and archive guidance compare the design against the intentions, not only the code against the design.
- `CLAUDE.md`:
  - the Roles section separates the user's decisions (functional requirements, contract, robotics semantics) from Claude's (implementation, presented with context and reasoning);
  - questions to the user are framed as functional consequences;
  - the documentation homes table gains `openspec/intent.md`.
- `memory/`: the memories this change supersedes or corrects are updated, so a new session does not read the old wording.

## Capabilities

### New Capabilities

None. This change records intentions and working rules; it adds no behavior to the `robotic` package. The change sets `skip_specs: true`.

### Modified Capabilities

None.

## Impact

- New: `openspec/intent.md`.
- Modified: `openspec/config.yaml`, `CLAUDE.md`, `README.md` (the paths table names `openspec/intent.md`), `memory/MEMORY.md` and the superseded memory files.
- `src/`, `tests/`, `docs/` and the open change `refactor-math-layers` are not modified.

## Out of Scope / Deferred

- Revising `refactor-math-layers` against the intentions: runtime enforcement of static promises (I0, I3), a contract for `to_axis_angle` that does not depend on the simplifier, `==` / `hash` / `bool`, `is_zero`, the name `lambdify`, the meaning of floats, `Transform` built from one rotation (I7). This is the next piece of work, judged against `intent.md` once it is approved.
- Singular configurations in compiled code: open on purpose (`intent.md`, "Open on purpose").
- Editor configuration so that ty diagnostics are visible while writing, notebooks included: a tooling task Claude owns, done separately.
