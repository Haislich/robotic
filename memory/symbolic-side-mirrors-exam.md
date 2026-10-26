---
name: symbolic-side-mirrors-exam
description: "The symbolic API reads like an exam solution; symbolic and numeric have distinct jobs, neither is secondary (intent I1, I6)"
metadata:
  type: user
  modified: 2026-10-26
---

Superseded by `openspec/intent.md` (I1 and I6), which is the source; read it there. In short: the symbolic API reads like the user's handwritten Robotics 1 exam solutions, semantics clear even if verbose; the numeric side is for simulation, a different job, not a secondary one. Typing and semantics cost is paid on the symbolic side so the numeric side is unconstrained, and numeric speed is a requirement.

**Origin:** user (2026-10-24): "what was pressing and important to me was the symbolic part ... because it mapped perfectly to how I would write my solutions for the exam". This memory then ranked the numeric side below the symbolic one; the user corrected it on 2026-10-26: "Il numerico sembra un afterthought ma non ha meno importanza, è che assolve un compito differente".

**How to apply:** cite I1/I6 in design decisions; never treat numeric-side work as optional. Related: [[refactor-design-state]].
