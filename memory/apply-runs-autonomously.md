---
name: apply-runs-autonomously
description: "Origin of the Roles rule in CLAUDE.md: during /opsx:apply, decide small reversible things and report them"
metadata:
  type: feedback
  modified: 2026-10-26
---

Superseded by the Roles section of `CLAUDE.md` (change `record-user-intent`, 2026-10-26), which is the rule; read it there. Deviations from `design.md` or `openspec/intent.md` still stop the apply (`openspec/config.yaml`), and irreversible actions (deleting untracked files, pushing, committing) still wait for the user.

**Origin:** 2026-10-25 the user, after Claude asked a gating question before group 1: "Why are you asking me so much consern you should be in auto mode".

**How to apply:** follow CLAUDE.md Roles; list what was decided in the end-of-apply summary. Related: [[jax-decisions-delegated]], [[one-decision-at-a-time]].
