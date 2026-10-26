# Design

## Context

See `proposal.md` - Why. Current state relevant to the approach:

- `openspec/config.yaml` requires every design decision to list the alternatives considered and why each was rejected; nothing asks for the intention a decision serves.
- The intentions that exist in writing are scattered across memory files (`symbolic-side-mirrors-exam`, `naming-and-immutability-intent`, `jax-decisions-delegated`, `apply-runs-autonomously`, `one-decision-at-a-time`), each recorded as a reaction to one episode. `symbolic-side-mirrors-exam` states "numeric secondary", which the user corrected on 2026-10-26 (intent I6).
- `refactor-math-layers/design.md` states as a goal "runtime checks only where the static version was measured to cost too much", the opposite of intent I3, and accepts as risks two cases that intent I0 rules out (an axis compiled alone, a normalized vector whose norm vanishes at some configuration).
- `CLAUDE.md` Roles: "The user designs the algorithms and makes the decisions" and "Do not decide on the user's behalf", with no boundary between the user's decisions and Claude's.

## Goals / Non-Goals

**Goals:**
- The user's intentions survive across sessions in the user's words, and every artifact is checked against them.
- The user's input can be functional requirements alone; technical choices reach the user with enough context to evaluate them.

**Non-Goals:**
- Changing the library or the open change `refactor-math-layers`.

## Decisions

### D1. One intent document, owned by the user, in `openspec/`

`openspec/intent.md` holds the intentions of the library: what it is for and what its contract promises, in the user's words, quoted in Italian as said. It changes only with the user's approval. It lives in `openspec/` because it outlives every change and every change is judged against it; `openspec/config.yaml` points to it from `context`, which every artifact reads.

Rejected:
- A section of `CLAUDE.md`: `CLAUDE.md` is the working contract between the user and Claude, not the contract of the library.
- The intentions inside each change's `design.md`: they would be archived with the change, which is the loss this change fixes.
- Memory files only: they are per-episode reactions, and a new session reads them as unrelated rules.
- Quotes translated to English: a translation is Claude's paraphrase, and the paraphrase is what drifted.

### D2. Decisions cite intentions; trade-offs against an intention are violations

Two artifact rules in `openspec/config.yaml`:
- A proposal names the intentions the change serves.
- A design decision names the intention it serves. A trade-off that goes against an intention is not recorded as an accepted risk. It is recorded as a violation of that intention, named, and only after the user approves it.

The second rule closes the path by which the two accepted risks of `refactor-math-layers` entered the design: each was accepted in a local discussion without a written principle to weigh it against.

Rejected: a free-form "Intent" paragraph per design (unchecked, and it repeats the intent document instead of pointing to it).

### D3. Reviews compare the design with the intentions

Apply guidance: an apply that finds a design decision contradicting `openspec/intent.md` stops, as for any deviation. Archive guidance: reconciliation compares the design with the intentions as well as the diff with the design. Code review stays as it is; the new check is on the artifacts.

Wording for `openspec/config.yaml` (D2 and D3; existing entries unchanged):

```yaml
context: |
  [existing text, then:]
  openspec/intent.md holds the user's intentions for the library (I0-I7), in
  their words. Read it before writing any artifact; it outranks design
  decisions, and a decision that contradicts it is a deviation.

rules:
  proposal:
    - Name the intentions of openspec/intent.md the change serves.
  design:
    - Every decision names the intention of openspec/intent.md it serves.
    - A trade-off that goes against an intention is recorded as a violation of that intention, named, and only after user approval; never as an accepted risk.

operations:
  apply:
    guidance:
      - A design decision that contradicts openspec/intent.md is a deviation.
  archive:
    guidance:
      - Reconciliation also compares design.md with openspec/intent.md and lists every recorded violation.
```

Decided during apply, approved by the user: the existing `context` sentence "The user designs and decides; Claude implements." contradicts D4, so it is replaced with "The user owns the requirements and the contract; Claude owns the implementation (CLAUDE.md, Roles)."

Rejected: a separate review operation in `config.yaml` (the schema has apply and archive guidance; reviews already run between them as tasks).

### D4. Roles: requirements to the user, implementation to Claude, with context

New Roles section for `CLAUDE.md` (text below). It replaces "Do not decide on the user's behalf", true only of the user's decisions, and generalizes the memories `jax-decisions-delegated` and `apply-runs-autonomously`.

```
## Roles

- The user owns the requirements: what the library expresses, its contract
  (`openspec/intent.md`) and the robotics semantics. Their input is functional.
- Claude owns the implementation: every technical choice that leaves the
  contract unchanged is Claude's. It is recorded in design.md and presented
  to the user with its context and reasoning, so the user can evaluate it
  and object; the user's objection reopens it.
- A technical choice that changes the contract reaches the user as its
  functional consequence (what they will write, read or see fail), with
  context, a recommendation, and the option to leave it open (intent I4).
  Never as a choice between mechanisms.
- Test before asking: if the user can answer by thinking about robotics or
  about how they use the library, it is a question; otherwise Claude decides.
- A question Claude cannot frame functionally means the written context is
  missing something; Claude says what, so the gap is filled in the documents.
- Disagree whenever there is a reason to, and give the reasoning. Hold the
  point until the argument is answered; pushback alone is not an answer.
```

The documentation homes table gains one row: `openspec/intent.md` | The user's intentions for the library, in their words; changed only with their approval.

Rejected: keeping "Do not decide on the user's behalf" with a list of delegated areas (the list grows one episode at a time, which is how the memories accumulated).

### D5. Superseded memories are rewritten, not left beside the new rules

- `symbolic-side-mirrors-exam`: rewritten to point to intent I6 (symbolic and numeric have distinct jobs; neither is secondary).
- `jax-decisions-delegated`, `apply-runs-autonomously`: replaced by a pointer to the Roles section, keeping the episode as their origin.
- `one-decision-at-a-time`: kept; it is about pacing design discussions, which D4 does not cover.
- `refactor-design-state`: updated with the review's outcome and the next step.
- `MEMORY.md`: index updated.

Rejected: deleting the superseded memories (their episodes are the evidence behind the rules).

## Risks / Trade-offs

- [The intent document can drift from what the user means, as the memories did] → It quotes the user verbatim, changes only with their approval, and every review reads it.
- [Citing intentions adds a line to every decision] → Accepted: the line is what lets a later session check the decision.

## Migration Plan

1. The user approves `intent.md` in this change.
2. Install it as `openspec/intent.md`, then update `config.yaml`, `CLAUDE.md`, `README.md` and `memory/`.

Rollback: revert the commit.
