---
name: jax-decisions-delegated
description: "The user delegates JAX-side decisions (compilation, pytrees, dtypes, transformations) to Claude"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dfbf9cbc-cb32-458f-96ae-f263e0cf7e05
  modified: 2026-10-25T15:40:57.103Z
---

The user said "you are in charge of the jax part" (2026-10-25, review round 3 of `refactor-math-layers`): decisions about the compiled/JAX side (pytree layout, flags, dtypes, jit/vmap/grad behaviour, `Numeric`) are Claude's to make.

**Why:** the user's focus is the robotics maths and the symbolic API; JAX is the numeric backend they do not want to arbitrate.

**How to apply:** decide JAX questions directly, record the decision and rejected alternatives in the change's `design.md`, and report the choice in one line instead of asking. Symbolic API, naming and robotics semantics stay the user's decisions ([[one-decision-at-a-time]], [[apply-runs-autonomously]]).
