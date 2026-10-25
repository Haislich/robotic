---
name: refactor-design-state
description: "State of the math-layers refactor (refactor-math-layers): review rounds 1 and 2 applied and committed 2026-10-25 (44/44 tasks, d8c0b33), kept open on purpose, not archived"
metadata:
  node_type: memory
  type: project
  originSessionId: db66abd4-09fa-4526-9a5d-0fd2cd38b7a5
  modified: 2026-10-25T14:37:47.338Z
---

`refactor-math-layers` was implemented 2026-10-25 (tasks 1-8 done), then a code review opened **review round 1**: artifacts revised the same day (D11, D16, D17, D19, D30, D33, new D34), review round 1 applied the same day (9.1-9.11 done, work uncommitted at the end of that session; `r_3_downward` and `Frame3Down` removed from the notebook at the user's choice). Review round 2 (section 10) applied and committed as d8c0b33; it found a test-exposable bug, so by the proposed exit criterion a round 3 is due. Open point kept on purpose: axis-angle skew norm simplified twice (D33). Branch `refactor-math-layers`, local only. Legacy code at tag `legacy` (41d2833). `design.md` is the source of truth; this memory only records state.

The user keeps the change open on purpose and iterates: review → classify (code bug = task only; spec/design bug = `/opsx:update`) → append a "Review round N" task section → `/opsx:apply`. They fear archiving early produces an endless chain of small changes. Exit criterion was proposed ("a review round with no test-exposable bug") but never confirmed — ask before archiving.

Not done yet:
- The user's untracked `scratchpad/t.py` (a Transform sketch) fails `uv run ruff check` and must be left alone; it is the only failure of the full check run.
- Archive with `/opsx:archive` (reconcile first, per `openspec/config.yaml`); merge/PR is the user's call.
- Follow-up changes: ast-grep style-rule change (ban `from_`-style identifiers), Euler/RPY (D24), manipulator/DH/Jacobian.

**Why:** a new session should start from the current state, not the design discussion.

**How to apply:** read `design.md` and `tasks.md` before proposing changes; reopening a decision means revising design.md first (apply guidance). Related: [[apply-runs-autonomously]], [[one-decision-at-a-time]].
