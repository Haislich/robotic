---
name: apply-runs-autonomously
description: "During /opsx:apply, proceed on sensible reversible defaults instead of gating on small questions"
metadata:
  node_type: memory
  type: feedback
  originSessionId: db66abd4-09fa-4526-9a5d-0fd2cd38b7a5
  modified: 2026-10-25T01:07:49.922Z
---

During implementation (`/opsx:apply`), do not stop to ask about small, reversible choices (tag names, branch, local commits, file housekeeping). Pick the sensible default, record it, report it in the summary.

**Why:** 2026-10-25 the user, after Claude asked a gating question before group 1: "Why are you asking me so much consern you should be in auto mode".

**How to apply:** still stop for real deviations from `design.md` / specs (openspec/config.yaml rule) and for irreversible actions (deleting untracked files, pushing). Everything else: decide, act, list it at the end. Overrides the "one question at a time" pacing of [[one-decision-at-a-time]] for implementation mechanics, not for design decisions.
