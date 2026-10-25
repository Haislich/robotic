---
name: one-decision-at-a-time
description: "In design discussions, present one small topic and one decision per message; no option dumps"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0ad2456a-e681-4b2c-b60b-f5b7eb646e98
  modified: 2026-10-23T07:30:21.577Z
---

In design/explore discussions, cover one topic per message and ask for one decision. Keep messages short; don't list many options, open threads, or side observations at once.

**Why:** On 2026-10-22, during the refactor API design, the user said the flood of information ("a metric fuckton", "100 open threads") made the approach not work, and asked to build bottom-up one concept at a time (scalar, then point, then vector, then axis, ...).

**How to apply:** Pick the next concept in the agreed order, show a minimal code sketch, ask the single question that blocks it. Park other observations until their concept comes up. Related: [[prefer-code-in-explanations]].
