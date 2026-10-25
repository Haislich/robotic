---
name: narrow-types-protocols-no-asserts
description: "Narrow types as far as possible, favor Protocols, never narrow with assert"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0ad2456a-e681-4b2c-b60b-f5b7eb646e98
  modified: 2026-10-22T22:51:41.229Z
---

Narrow types as much as possible (specific domain types, generics bound to Protocols, union results handled with `match`). The user likes Protocols. Do not use `assert` / `assert isinstance(...)` to narrow types or check invariants.

**Why:** User request on 2026-10-22 during the API design for the refactor ("narrow down the types as much as possible, protocols are really cool, and avoid fiddling with asserts").

**How to apply:** Design so the type system carries the narrowing (bounded generics, Protocols, TypeIs/TypeGuard where unavoidable, exhaustive `match`). Related: [[no-free-functions-tuples-dicts]].
