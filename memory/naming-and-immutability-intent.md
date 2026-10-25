---
name: naming-and-immutability-intent
description: "Use full descriptive names (no two-letter methods like `at`); \"immutable\" means no hidden, hard-to-trace state mutation, not dogmatic frozen dataclasses"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0ad2456a-e681-4b2c-b60b-f5b7eb646e98
  modified: 2026-10-24T20:16:20.240Z
---

Name methods by what they do in full words (e.g. `evaluate`, not `at`); avoid very short cryptic method names.

No name may clash with the language (keywords, soft keywords, builtins). Escaping a keyword with a trailing underscore (`from_`) counts as a clash: find a different word instead (2026-10-24: `to=`/`from_=` became `of=`/`relative_to=`).

"Immutable by default" in CLAUDE.md means: avoid hidden state mutation that makes the data flow hard to trace. It does not require frozen dataclasses specifically — plain classes with read-only properties are fine (and were chosen because ty 0.0.91 infers frozen-dataclass generics as invariant).

**Why:** User clarified both on 2026-10-22 during the Scalar design.

**How to apply:** Pick descriptive names in API sketches; judge immutability by traceability of state, not by mechanism. Related: [[no-free-functions-tuples-dicts]], [[narrow-types-protocols-no-asserts]].
