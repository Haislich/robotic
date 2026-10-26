---
name: memory-always-current
description: "Keep memory/ and the latest handoff current during the session; the user moves between several PCs"
metadata:
  type: feedback
  modified: 2026-10-26
---

Update `memory/` (and `memory/handoffs/` when the work state changes) as soon as something changes, not at the end of the session. Claude's local memory folder is a symlink to the repo's `memory/` (README, setup), so memories travel with git.

**Why:** user (2026-10-26): "La memory deve essere sempre aggiornata perché faccio avanti e indietro con più pc." A stale handoff made a new session propose redoing finished work.

**How to apply:** after each approved decision or completed task, update the affected memory and the handoff's work state; include `memory/` changes in the commit. Never leave loose files outside their home (a stray folder like `robo/` "non ha alcun senso di esistere"). Related: [[refactor-design-state]].
