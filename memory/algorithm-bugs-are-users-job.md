---
name: algorithm-bugs-are-users-job
description: "In the legacy refactor, fix the robotics algorithm bugs found (user reversed an earlier \"keep the bugs\" stance)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0ad2456a-e681-4b2c-b60b-f5b7eb646e98
  modified: 2026-10-22T21:56:44.948Z
---

During the legacy-code refactor, the algorithm bugs found (Euler round-trip, axis-angle precedence, mutation in round(), etc.) are fixed, not preserved. On 2026-10-22 the user first said "keep the bugs, fixing them is my job", then reversed it in the same session: "scrap the keep the bugs, solve them."

**Why:** The user decided to have the refactor deliver correct algorithms.

**How to apply:** Fix bugs as part of the refactor, verified by oracle tests, and name each fixed bug explicitly so the user sees what changed. Related: [[prefer-code-in-explanations]], [[no-free-functions-tuples-dicts]].
