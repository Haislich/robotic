---
name: symbolic-side-mirrors-exam
description: The robotics refactor's priority is the symbolic API reading like an exam solution; numeric side is secondary
metadata:
  type: user
---

The symbolic API should read like the user's handwritten Robotics 1 exam solutions: semantics crystal clear, verbosity accepted. Numeric semantics (frames on JAX results) are welcome but not pressing; at the numeric step the user wants raw power.

**Why:** user (2026-10-24): "what was pressing and important to me was the symbolic part ... because it mapped perfectly to how I would write my solutions for the exam, making the semantics of the code crystal clear even if verbose in some points".

**How to apply:** when trading off, favor clarity of the symbolic API over brevity; don't spend design effort on numeric-side typing beyond the thin result. Related: [[refactor-design-state]].
