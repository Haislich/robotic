---
name: no-free-functions-tuples-dicts
description: "Prefer a functional, expression-oriented style on immutable domain types; free functions only when really needed; dicts/tuples only with judgement"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0ad2456a-e681-4b2c-b60b-f5b7eb646e98
  modified: 2026-10-23T07:52:49.978Z
---

Prefer a functional style: immutable values, expressions over imperative step-by-step code, operations that return new values (methods on domain types, composition). Free (module-level) functions only when really needed — the user finds them counter-intuitive and hard to read. No lambdas, no `Any`.

Dicts and tuples: not banned outright, but Claude tends to spray them everywhere. Use a dict only when the keys are genuinely not known a priori (e.g. binding variables to values); use a tuple only as private storage for a homogeneous variable-length collection inside a domain type. Never as anonymous records in signatures — use named frozen types.

**Why:** User said (2026-10-22) the codebase otherwise degrades into ~50 free functions calling each other; imperative code reads as "difficult for its own sake", and functional style reads better in math-heavy code. They clarified the dict rule exists because of abuse, not because dicts are never right.

**How to apply:** Behavior lives on the type it concerns; no static-only "utils" classes. Before adding a dict/tuple, check the keys/shape are truly dynamic; otherwise a named type. Wrap untyped libraries (sympy) inside domain types so `Any`/`Unknown` doesn't leak. Related: [[narrow-types-protocols-no-asserts]], [[prefer-code-in-explanations]].
