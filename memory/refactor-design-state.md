---
name: refactor-design-state
description: "State of refactor-math-layers: rounds 1-3 applied 2026-10-25; the 2026-10-26 review found it diverges from the user's intentions; next: revise it against openspec/intent.md"
metadata:
  node_type: memory
  type: project
  originSessionId: db66abd4-09fa-4526-9a5d-0fd2cd38b7a5
  modified: 2026-10-26
---

`refactor-math-layers` was implemented 2026-10-25 (tasks 1-8 done), then a code review opened **review round 1**: artifacts revised the same day (D11, D16, D17, D19, D30, D33, new D34), review round 1 applied the same day (9.1-9.11 done, work uncommitted at the end of that session; `r_3_downward` and `Frame3Down` removed from the notebook at the user's choice). Review round 2 (section 10) applied and committed as d8c0b33; it found a test-exposable bug, so by the proposed exit criterion a round 3 is due. Open point kept on purpose: axis-angle skew norm simplified twice (D33). Branch `refactor-math-layers`, local only. Legacy code at tag `legacy` (41d2833). `design.md` is the source of truth; this memory only records state.

**Review of 2026-10-26** (external, against the user's goals): the code follows its artifacts faithfully yet diverges from what the user meant, because intentions were recorded only as decisions and lost between sessions. Change `record-user-intent` installed `openspec/intent.md` (I0-I7) and the CLAUDE.md Roles. Next: revise `refactor-math-layers` against intent.md, each item framed functionally: runtime enforcement of every static promise (I0, I3); a `to_axis_angle` contract independent of the simplifier; `==`/`hash`/`bool` fail loudly until decided; `is_zero` public or not; the name `lambdify`; meaning of floats; `Transform.of_rotation` (I7, candidate); replace the Goals line "runtime checks only where measured"; re-label the two accepted risks (axis compiled alone, vanishing-norm normalize) as violations or open items.

**Review round 4** (2026-10-26, section 12 of tasks.md), revisions one at a time: 1. D35 runtime enforcement (task 12.1); 2. singular configurations open on purpose, in proposal Out of Scope (task 12.2); 3. no validity condition: no symbols → exact case, symbols → regular formulas (D13, D28, task 12.3); 4. lambdify: no implicit simplify, cse on (D11, 12.4); 5. simplify strategy: trigonometric, probe trigsimp fu (D5, 12.5); 6. no equality on symbolic or compiled values: ==, hash, bool raise (D36, 12.6); 7. is_zero leaves the API, private function in symbolic.py (D31, 12.7); 8. lambdify renamed compile; naming rule: methods are namespaced by their class (D11, D16, 12.8); 9. exact numbers only on the symbolic side, floats only numeric (D4, 12.9); 10. a transform is R.homogeneous(translation: Position[Target]), property .translation, SE(3), translation = position vector of the source origin (D17, 12.10); 11. Position (alias Point) and Displacement, Base.position / Base.displacement, Direction + Base.direction (normalizes) (D37, 12.11). Contract changes are licit until this change is archived (user, 2026-10-26). Applied 2026-10-27: 61/61 tasks, full check green except the user's stratchpad/ (ruff/ty exclusions commented out by the user); committed 2026-10-27.

The user keeps the change open on purpose and iterates: review → classify (code bug = task only; spec/design bug = `/opsx:update`) → append a "Review round N" task section → `/opsx:apply`. They fear archiving early produces an endless chain of small changes. Exit criterion was proposed ("a review round with no test-exposable bug") but never confirmed — ask before archiving.

Not done yet:
- The user's untracked `stratchpad/` is excluded from ruff, ty and pytest and must be left alone ([[stratchpad-is-users]]).
- Archive with `/opsx:archive` (reconcile first, per `openspec/config.yaml`); merge/PR is the user's call.
- Follow-up changes: ast-grep style-rule change (ban `from_`-style identifiers), Euler/RPY (D24), manipulator/DH/Jacobian.

**Why:** a new session should start from the current state, not the design discussion.

**How to apply:** read `design.md` and `tasks.md` before proposing changes; reopening a decision means revising design.md first (apply guidance). Related: [[apply-runs-autonomously]], [[one-decision-at-a-time]].
