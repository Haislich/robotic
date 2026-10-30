---
name: never-simplify-implicitly
description: Simplification is costly and only about form; the library never simplifies implicitly, only when the user calls simplify
metadata:
  type: feedback
---

Simplification is never done implicitly anywhere in the library (user, 2026-10-30, generalizing D11's "compile does not simplify"). It costs a lot and serves form only; the user calls `.simplify()` explicitly. LaTeX was floated as a possible home; Claude recommended against (hidden cost on every notebook display, shown form differs from computed value) — user agreed: LaTeX does not simplify either.

**Why:** the 47 s norm simplification in `to_axis_angle` (D33) bought nothing: on Rz, Rz Rx, Rz Rx Rz Rx the simplified norm had the same op count as the raw one.

**How to apply:** no `.simplify()` inside constructors or algorithms (Direction, to_axis_angle, compile); only the explicit `simplify` methods. Related: [[refactor-design-state]], [[symbolic-side-mirrors-exam]].
