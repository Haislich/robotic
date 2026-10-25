# active-memory handoff: robotic math-layers refactor, review rounds 1-3

**Handoff #5** · 2026-10-25 · Lineage: #1 (2026-10-24): explore-phase design, affine points/vectors, typing principles; #2 (2026-10-24): sympy-only behind a typed boundary, numeric only via `lambdify` to JAX, receiver-first rule, API stub; #3 (2026-10-24/25): created OpenSpec change `refactor-math-layers` (D1-D28), open questions resolved; #4 (2026-10-25): implemented tasks 1-8, ran review round 1, revised artifacts (tasks 9.x, D34) (file: `handoff-robotic-review.md`); #5 (2026-10-25): applied review round 1 (9.1-9.11), ran and applied review rounds 2 (section 10) and 3 (section 11), all committed; JAX decisions delegated to Claude

## 0. Instructions for Claude (read first)

You are continuing work from a previous chat. That chat is gone; this file is the complete context and the source of truth.

1. Read this whole file before replying.
2. Follow sections 3 (Style), 4 (Hard rules) and 5 (Corrections) in every reply, for the rest of this chat. They override your defaults.
3. Use the values in section 8 exactly. Never round, re-estimate, or "correct" them.
4. Do not suggest anything listed in section 7 (Changed / rejected) again unless the user brings it up.
5. Code word: None active. If the user types /amcodeword, start every reply with the phrase they choose (default "Yes Boss!").
6. Your first reply: at most 5 lines covering the goal, the current state, and the next step (section 11). Mention any files from section 13 that were not attached. Ask the questions in section 12 if there are any. End with "Ready to continue with <next step>?" Then wait for the user's go.

## 1. Mission
- **Goal:** Keep the OpenSpec change `refactor-math-layers` open and iterate: review (`/code-review src/robotic`) → triage with the user → record in design.md/tasks.md as "Review round N" → `/opsx:apply` → commit, until the code reads like the maths (hand-implemented robotics library in `src/robotic/`, used by the user to learn robotics).
- **Done looks like:** Not confirmed. Claude proposed: "a review round with no bug a test can expose" (alternative: exam notebooks match official solutions). Then reconcile and archive once.
- **Why it matters / context:** User: "The code serves me to learn robotics so this is the main goal."

## 2. About the user (as relevant to this work)
- Robotics student; repo `/home/haislich/Programming/robotic` (Python 3.14, uv, ruff, ty 0.0.91 pinned, sympy 1.14.0, jax 0.12.0.1).
- Designs the algorithms and makes the decisions; Claude implements, accelerates, challenges (CLAUDE.md).
- Reads a rotation as a mapping: ^T R_S maps a vector expressed in the Source S to the Target T.
- Fears archiving early: "I don't want to archive it or it will cause a series of infinite specs."
- Delegated the JAX side: "you are in charge of the jax part".

## 3. Style & communication
- **Language:** English.
- **Tone:** direct; disagree with reasons and hold the point until answered (CLAUDE.md).
- **Reply length:** short; one concept and one question per message.
- **Formatting:** code sketches over prose; small tables for triage; end with one clear question.
- **Working style:** recommend one option, then ask; never decide for the user except on JAX-side matters; probe with ty/runtime before claiming a behaviour (probe files in the scratchpad dir, not the repo); explain plainly when the user asks "how did we end up…" (they asked twice in a row about the norm issue until the wording was precise).
- **Avoid:** option dumps; deciding on the user's behalf; asserts in library code; inline diagnostic suppressions; loose wording (say "the norm is simplified twice", not "normalized twice").

## 4. Hard rules (word for word)
1. CLAUDE.md in the repo is the working contract (roles, code, documentation homes, Serena navigation, ast-grep/rg bounded, OpenSpec workflow, checks `uv run ruff check`, `uv run ruff format`, `uv run ty check`, `uv run pytest`).
2. "Robotics algorithms in `src/` are implemented by hand. `roboticstoolbox` and `spatialmath` are test-only oracles."
3. "Never suppress a diagnostic inline. If a rule looks wrong, stop and propose a configuration exception, with its reason recorded in the change's `design.md`."
4. Naming: no clashes with the language; escaping a keyword with a trailing underscore (`from_`) counts as a clash.
5. Data immutable; no hidden mutation (the builder returns new objects).
6. Apply guidance (`openspec/config.yaml`): a deviation from design/spec/proposal → stop, explain, propose the artifact revision, wait; mechanical adjustments are listed in the task summary; decisions made during implementation are recorded in design.md; during `/opsx:apply` decide small reversible things autonomously.
7. "leave scratchpad alone" (the untracked repo folder `scratchpad/`, containing the user's `t.py`, a Transform sketch that fails ruff and uses old keywords; never edit, delete or commit it).
8. "you are in charge of the jax part": decide JAX questions, record them in design.md, report in one line.
9. Commit only when the user says so; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 5. Corrections log
| # | Claude did | The user wanted |
|---|---|---|
| 1 | Proposed `relative_to`/`of` everywhere | Mapping vocabulary, coherent → Target/Source |
| 2 | (User first proposed `Transform[Source, Target]`) Claude showed composition reads backwards | "Target first is better" |
| 3 | Recommended `rotation.with_translation(v)` | "we're not applying a translation to a rotation, is misleading" |
| 4 | Recommended `Transform(rotation, translation)` with a known inline miss | "I prefer redundancy" → builder from frames |
| 5 | Kept task 10.2's "norm simplified once" wording with a stop condition instead of correcting it; then said "normalization twice" | A plain explanation (the norm ‖s‖ is computed and simplified twice: for sin θ and inside `normalize`; normalization happens once), then: "keep em, but change the wording so we can revisit at a later date" |
| 6 | Round 3 triage rejected #3 (`Rotation @ point` returning `None` at runtime) as covered by ty | "3 should be not implemented" → `case _: return NotImplemented` in `Rotation.__matmul__` and `Transform.__matmul__` |
| 7 | Round 2 rejected a runtime fallback in `Scalar._expression_of` as duplicating ty | Self-corrected in round 3: operators must return `NotImplemented` (Python protocol), which is what lets `l1 * x_axis` reach `Vector.__rmul__` |

## 6. Decisions
| Decision | Why |
|---|---|
| Frames Target/Source at every layer; keywords `target=`/`source=`, written target first in calls; method-level type params `A`, `B` (D16, D33) | Mapping reading; mirrors ^A R_B ^B R_C = ^A R_C |
| `Frame.notation` (D19); `Point(x, y, z, frame)`, `Vector(x, y, z, frame)` (D30) | User's |
| `Transform(target=, source=).with_rotation(R).with_translation(v)`, `.translation: Vector[Target]`; `with_origin` and 3-number constructor removed (D17) | User prefers explicit redundant frames |
| `Rotation` and `Transform` are `@final`; `simplify`/`with_*` return the concrete type (D33) | ty 0.0.91 rejects the private factory as `Self` |
| `Layout.condition` holds the `Symbolic` condition; `Compiled` reads its `_layout()` in the boundary (D33) | `self._valid_when._layout()` from rotation.py is ruff `private-member-access` |
| Half-turn `ValidityCondition` = chosen column's `r_kk + 1` (D33) | Round 1 bug (axis (a,1,0) at a=0 gave NaN with valid True) |
| `ParameterShapeError(ValueError)` raised in the traced kernel; documented in `Compiled.__call__` Args, not Raises (D11, D31) | ruff DOC502 rejects Raises for an exception raised elsewhere |
| Removed `r_3_downward` and `Frame3Down` from the notebook | User: "remove it" |
| `jax_enable_x64` stays at import, process-wide, documented in `docs/compilation.md` and README (D12) | User accepted; float32 would show false mismatches |
| Skew norm simplified twice in `to_axis_angle` kept as an open point to revisit (D33, proposal deferred list) | User: "keep em, but change the wording so we can revisit at a later date"; D20 forbids a unit vector built from a supplied length |
| Scalar operators return `NotImplemented` for operands outside `ScalarLike`; `Vector.__rmul__(factor: Scalar \| float)` (D31) | `l1 * x_axis` was a ty error and a runtime crash |
| `@` returns `NotImplemented` for other operands (D23) | User's decision |
| `Symbol.simplify()` returns `self`, typed `Self` (D31) | Self contract |
| Validity flag stays on every `Numeric`; derivatives taken of `.components` (D12) | Claude's (JAX delegated): a differentiated `Numeric[Rotation]` would hold dR/dq, not a rotation |
| `Numeric._unflatten` keeps `Numeric[Symbolic](...)` (D12) | Claude's: plain `Numeric(...)` is ty `unsound-return-statement`; `Numeric[Any]` banned by ruff `any-type` |
| `Numeric.__repr__` shows a tracer as JAX prints it (D12) | Round 2 bug: `.tolist()` crashed under `jax.jit` |
| `UnitVector.__neg__` and `.simplify` share private `_of_unit_components`, no re-normalization (D20, D33) | Rounds 1-2 |
| One `_Matrix3.column_product` and one `Symbolic._bmatrix` (used by points/vectors, rotations, transforms) | Round 2 duplication |

## 7. Changed / rejected
- `of=`/`relative_to=`, `To`/`From`, `T`/`S` → Target/Source
- ❌ `[Source, Target]` order; ❌ `Frame.latex`, `label`, `name`, `symbol` as the frame keyword
- `Transform(rotation, x, y, z)` + `rotation.with_origin(p)` → builder from frames; ❌ `Transform(rotation, translation)`; ❌ `rotation.with_translation(v)`; ❌ mutating builder; ❌ copying rotation constructors onto Transform; ❌ builder pattern everywhere
- ❌ Broadcasting direct arrays (second batching path)
- Half turn valid everywhere (`1 != 0`) → `r_kk + 1 != 0`
- ❌ Keeping `r_3_downward` (Claude recommended keeping; user removed it)
- Round 2 rejected: float half turn sent to the regular branch (by design, D4/D28); runtime raise for non-`ScalarLike` values (revised in round 3 to `NotImplemented`); default for the half-turn `next()` (unreachable, D33)
- ❌ x64 scoped inside `Compiled.__call__` (unprobed, expected to fail under the caller's jit/vmap); ❌ caller enables x64
- Round 3 rejected: float half turn again (D4); `StopIteration` again (D33); norm simplified 3x (open point); runtime frame comparison (Goals: static checks first, D6); `Rotation(...)` vs `about_axis` two spellings (D34)
- ❌ Dropping the flag child from flagless results; ❌ differentiating `Numeric` as a whole
- Ways to revisit the double norm, not chosen: `UnitVector` from a known length (reopens D20); `UnitVector` exposing its length; memoizing simplify in the boundary (hidden state)

## 8. Data & facts (exact)
- Branch `refactor-math-layers` (local, not pushed), main `main`; legacy tag `legacy` = `41d2833`.
- Commits this chat: `24554c2` Drop the unused pydantic dependency; `c2318d1` Revise refactor-math-layers for review round 1 (D34); `2b5ce7b` Apply review round 1; `d8c0b33` Apply review round 2; `631fb96` Apply review round 3 (HEAD).
- Tasks: 50/50 complete (sections 1-8, 9 = round 1, 10 = round 2, 11 = round 3). `openspec validate refactor-math-layers --strict` passes. Design D1-D34 (no new D numbers; round notes added to D4, D11, D12, D17, D20, D23, D31, D33).
- Checks at HEAD: ruff clean except `scratchpad/t.py` (`uv run ruff check --exclude scratchpad` clean), `ruff format --check` clean, ty clean, 113 tests pass, notebook runs with no errors.
- Notebook verification: chained ^0R_1…^0R_6 and the wrist point identical to the pre-round-1 values, e.g. `base_6 Rotation[Frame0, Frame6]([0, 0, 1], [0, -1, 0], [1, 0, 0])`; `wrist Point[Frame0]((a_1 + a_2*cos(q_2) + d_4*sin(q_2 + q_3))*cos(q_1), (a_1 + a_2*cos(q_2) + d_4*sin(q_2 + q_3))*sin(q_1), a_2*sin(q_2) + d_1 - d_4*cos(q_2 + q_3))`.
- Notebook execution needs a temporary nbconvert: `uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace exams/2024_09_19.ipynb` (nbconvert is not a project dependency).
- Float half turn: `Rotation.about_z(math.pi, ...)` skew norm 2.4e-16, angle `-1.22464679914735e-16 + pi`, flag false.
- jacrev doctest: dR/dq of Rz at q = 0 → shape `(3, 3)`, `derivative[1, 0] = 1.0`, `derivative[0, 1] = -1.0`.
- `ParameterShapeError` message: `field 'q1' holds an array of shape (2,); every field must be a scalar of shape (), batch configurations with jax.vmap`.
- Default tolerance `1e-9`.

## 9. People, terms & names
- **People:** the user (only).
- **Terms & nicknames:** "review round N" = task section appended to tasks.md per review loop; "known miss" = negative-fixture line that must stay silent; "open point" = deliberately deferred item in D33 (skew norm simplified twice); `^A R_B` = rotation mapping B (Source) coordinates to A (Target).
- **Names in use:** `Rotation`, `Transform`, `Frame`, `Point`, `Vector`, `UnitVector`, `FreeAxis`, `Scalar`, `Symbol`, `Symbolic`, `Compiled`, `Numeric`, `Layout`, `ValidityCondition`, `AxisAngle`, `AxisAngleSolutions`, `ParameterTypeError`, `MissingParameterError`, `ParameterShapeError`, `ZeroVectorError`, `Index3`, `Index4`; private helpers `_Matrix3.column_product`, `Symbolic._bmatrix`, `Scalar._combined`, `UnitVector._of_unit_components`, `Rotation._of_matrix`, `Transform._of_parts`; negative fixture `tests/type_mistakes.py.txt`; notebook `exams/2024_09_19.ipynb` (frames `Frame0`-`Frame6`, `Joint1`-`Joint4` with notations `1'`-`4'`).

## 10. Work state
| Item | Status | Version / location | Notes |
|---|---|---|---|
| `refactor-math-layers` tasks | done 50/50 | `openspec/changes/refactor-math-layers/tasks.md` | Round 3 found test-exposable bugs → by the proposed criterion a round 4 is due |
| design.md / proposal.md / specs | current | change dir | proposal deferred list includes the skew-norm open point |
| Source | committed | `src/robotic/{symbolic,geometry,rotation,transform,__init__}.py` @ `631fb96` | |
| Docs | committed | `docs/*.md` (doctested), `README.md` | |
| Notebook | committed, runs | `exams/2024_09_19.ipynb` | |
| `scratchpad/t.py` | user's, untracked | repo `scratchpad/` | leave alone; fails ruff |
| Memory | updated | `refactor-design-state.md`, new `jax-decisions-delegated.md` | |

## 11. Next steps
1. **Next action:** ask the user whether to run review round 4 (`/code-review src/robotic`) or to settle the exit criterion first.
2. Triage each finding against D1-D34 and the section 7 rejections before presenting; one triage table, one question.
3. Record accepted items as "## 12. Review round 4" in tasks.md plus design notes, validate strict, then `/opsx:apply`, verify as in 8.2, commit on the user's go.
4. When the exit criterion is met: reconcile (per `openspec/config.yaml` archive guidance) and `/opsx:archive`.

## 12. Open questions ⚠️
- Exit criterion for the review loop: (a) "a review round with no test-exposable bug" (Claude's suggestion), (b) exam notebooks match official solutions, (c) other.

## 13. Re-attach checklist
- Nothing to attach. The new chat reads the repo directly (`openspec/changes/refactor-math-layers/`, `CLAUDE.md`, `openspec/config.yaml`).

---
<sub>Audit: 13/13 sections · 9 rules · 7 corrections · 16 data points · secrets removed: none found · generated by active-memory</sub>
