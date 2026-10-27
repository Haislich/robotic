# active-memory handoff: robotic, external review and the record-user-intent proposal

**Handoff #6** · 2026-10-26 · Lineage: #1-#3 (2026-10-24/25): explore and design of `refactor-math-layers` (D1-D28); #4 (2026-10-25): tasks 1-8, review round 1; #5 (2026-10-25, `memory/handoffs/handoff-robotic-rounds.md`): review rounds 1-3 applied, JAX delegated; #6 (2026-10-26, this file, cloud session): external review of the code against the goals, root cause of the divergence found, OpenSpec change `record-user-intent` proposed (not pushed, GitHub 403)

## 0. Instructions for Claude (read first)

You are continuing work from a previous chat. That chat is gone; this file is the complete context and the source of truth.

1. Read this whole file before replying.
2. Follow sections 3 (Style), 4 (Hard rules) and 5 (Corrections) in every reply, for the rest of this chat. They override your defaults.
3. Use the values in section 8 exactly. Never round, re-estimate, or "correct" them.
4. Do not suggest anything listed in section 7 (Changed / rejected) again unless the user brings it up.
5. Code word: None active.
6. Your first reply: at most 5 lines covering the goal, the current state, and the next step (section 11). Mention any files from section 13 that were not attached. Ask the questions in section 12 if there are any. End with "Ready to continue with <next step>?" Then wait for the user's go.

## 1. Mission
- **Goal:** Make the user's intentions survive across sessions, then re-judge `refactor-math-layers` against them. Step 1 is the OpenSpec change `record-user-intent` (artifacts written, not applied). Step 2 is revising `refactor-math-layers` against `openspec/intent.md`.
- **Done looks like:** `record-user-intent` applied (intent.md installed, config.yaml, CLAUDE.md, README, memory updated) and committed; then a revision of `refactor-math-layers` whose design decisions each cite an intention.
- **Why it matters:** The review found the code follows its written artifacts faithfully yet diverges from what the user meant. Root cause (user's words): "tu sei stateless mentre io ricordo, quindi ad ogni cambio di sessione io do per scontato delle cose che non lo sono e questo causa divergenza."

## 2. About the user (as relevant to this work)
- Robotics student; the library serves to learn robotics and to simulate.
- Wants to give only functional requirements and have the written context be enough for Claude to implement; still wants to be told about technical decisions, with context, to evaluate them.
- Does not want to decide things whose direction is not yet clear ("rischio di dare una risposta che poi mi si ritorce contro").
- sympy and JAX idiosyncrasies are Claude's job: "non ho tempo né voglia di occuparmi".

## 3. Style & communication
- **Language:** Italian in chat (this session); repository files in English, user quotes in Italian.
- **Tone:** direct; disagree with reasons and hold the point until answered.
- **Reply length:** short; one concept, one decision per message.
- **Formatting:** code sketches over prose; small tables; end with one question.
- **Working style:** frame every question as its functional consequence (what the user will write, read or see fail), never as a choice between mechanisms; test before asking: "can the user answer by thinking about robotics or about how they use the library?" If not, decide, record, explain. Offer "leave it open" when the use is not clear yet. Verify claims with probes (ty, runtime) before stating them.
- **Avoid:** technical option tables (e.g. `checkify` vs `jax_debug_nans`) put to the user; paraphrasing intentions instead of quoting; treating the numeric side as secondary.

## 4. Hard rules (word for word)
1. CLAUDE.md in the repo is the working contract (unchanged until `record-user-intent` is applied).
2. "Robotics algorithms in `src/` are implemented by hand. `roboticstoolbox` and `spatialmath` are test-only oracles."
3. "Never suppress a diagnostic inline."
4. Tasks producing text the user owns (intent, CLAUDE.md, config.yaml) are done only after the user approves the wording.
5. Commit only when the user says so; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
6. "leave scratchpad alone" (the user's untracked repo folder `scratchpad/`, if present).

## 5. Corrections log
| # | Claude did | The user wanted |
|---|---|---|
| 1 | Called the 8 hand-written `simplify` implementations an extensibility problem | Internals are free; only the contract must be fixed ("l'API rimane la stessa ed è fissata, quello che cambia è il comportamento sottostante") |
| 2 | Proposed an escape hatch to raw sympy for notebooks | The boundary is a line of defense; extend it with typed methods later, never puncture it |
| 3 | Proposed making cost visible in the contract | Costs and implementation details are hidden in the wrapper (I5) |
| 4 | Proposed "mathematical laws" as the contract, without types | The contract is the mathematics with its meaning in the types; the type checker stays the oracle ("rotazioni di chi, rispetto a chi") |
| 5 | Proposed quantity types (Angle/Length, then Angle only) | Scalars stay dimensionless, a declared permanent limit: the return is marginal |
| 6 | Wrote "the library need not compute fast" | Numeric speed is a requirement; only the symbolic side may be slow |
| 7 | Proposed a symbolic time budget, then "never simplify" | No time limit on symbolic; its cost is paid once at setup; symbolic and numeric have distinct jobs (I6) |
| 8 | Said a vanishing-norm singularity cannot be caught at runtime | It is caught at substitution time (q = 0 gives 0/0); numeric behavior left open on purpose |
| 9 | Asked the user to choose between JAX mechanisms (flag / checkify / debug nans) | Questions framed functionally; the user did not want to answer yet → recorded as "Open on purpose" |
| 10 | Read memory `symbolic-side-mirrors-exam` as "numeric is secondary" | Numeric is less *constrained*, not less important: typing and semantics cost is paid on the symbolic side so numeric is free |

## 6. Decisions
| Decision | Why |
|---|---|
| Write the user's intentions in `openspec/intent.md` (I0-I7), user-owned, quotes in Italian | Intentions were recorded only as decisions and lost between sessions |
| I0: wrong situations unrepresentable (Rust mantra); numeric correct by construction | User's guiding principle |
| I3: static first; runtime always backs every static promise, at the failing operation; runtime alone only where types cannot reach; never runtime instead of an available static guarantee | User: "volevo che coesistessero, ma che non fosse un modo per abusare del runtime" |
| I4: contract fixed, implementation free; undecided things are left open and promised nothing, so later answers are additions | "piastrelle" / "bagno" metaphor |
| Scalars dimensionless, permanently | User's cost/benefit |
| Singular configurations in compiled code: open on purpose | User does not know the use yet |
| Design decisions cite the intention they serve; trade-offs against an intention are recorded as violations, never as accepted risks | Two "accepted risks" in `refactor-math-layers` contradict I0 |
| New CLAUDE.md Roles: user owns requirements/contract/semantics; Claude owns implementation and presents choices with context; contract-changing choices reach the user as functional consequences | User: "il mio unico input siano requisiti funzionali"; "rendimi comunque partecipe delle decisioni tecniche" |
| `.vscode`: configure ty as language server (notebooks too) | Claude's tooling task, separate from the change |

## 7. Changed / rejected
- ❌ Escape hatch to sympy; ❌ cost visibility in the contract; ❌ quantity types (Angle/Length/Ratio, closed or runtime-dimension variants); ❌ symbolic setup time budget; ❌ "never simplify" principle; ❌ unifying `simplify` as a review criterion
- `Transform.of_rotation(R).with_translation(t)` (one rotation fixes both frames) probed clean in ty 0.0.91, all 4 mistakes caught: a candidate for I7, not decided

## 8. Data & facts (exact)
- Repo `Haislich/robotic`; base commit `473611d` ("Keep Claude Code's project memory and the latest handoff in the repository"); branch for this work `claude/code-overview-v1jahm`; local commit `da64cc9` "Propose record-user-intent" not pushed (GitHub App 403). Patch: `0001-Propose-record-user-intent.patch`.
- Checks at `473611d`: 113 tests pass, ruff clean, ty clean.
- Runtime probes at `473611d` (no checks exist at runtime): `A.vector(1,0,0) + B.vector(1,0,0)` → `Vector[A](2, 0, 0)`; `A.point(1,0,0) + A.point(1,0,0)` → `Point[A](2, 0, 0)`; `r_ab @ r_ab` → `Rotation[A, B]`; `A.vector(1,0,0) * A.vector(1,0,0)` → `Vector[A](None, None, None)`; `Scalar("x")` → `None`; `Scalar.symbol("q") == Scalar.symbol("q")` → `False`; `Scalar(1) == 1` → `False`.
- ty checks `.ipynb` when run (`uv run ty check` finds errors in notebooks); pre-commit (`types: [python]`) and the edit hook do not; `.vscode/settings.json` has no ty configuration.
- Symbolic timings, alternating Rz/Rx chains: 2 joints simplify 0.23 s / to_axis_angle 0.74 s; 3 joints 0.38 / 1.59 s; 4 joints 1.2 / 15.75 s; 6 joints simplify did not finish in 200 s.
- 6R forward kinematics, numeric: build 0.11 s, lambdify 0.08 s, first call (JIT) 2.64 s, per call from Python 83.9 µs, vmap 10k configurations 18.48 ms.
- Specs: ~15 error scenarios end with "the type checker reports an error"; only zero-vector normalization and compile errors specify runtime errors. `docs/` holds 134 `>>>` doctest examples. `hypothesis` is used by 2 tests only.

## 9. People, terms & names
- **People:** the user (only).
- **Terms:** "piastrelle" (tiles) = the contract: robotics mathematics with meaning in types; "bagno" (bathroom) = the implementation, free to change; "Open on purpose" = deliberately undecided, promised nothing; I0-I7 = intentions in `intent.md`.
- **Names:** change `record-user-intent`; open change `refactor-math-layers`; candidate `Transform.of_rotation`; candidate exception `FrameMismatchError`.

## 10. Work state
Updated 2026-10-26, local session on branch `refactor-math-layers` (the user works there; no separate branch, `claude/code-overview-v1jahm` and the patch are obsolete).

| Item | Status | Location | Notes |
|---|---|---|---|
| `record-user-intent` | applied, committed (`e5e8a56`), archived | `openspec/changes/archive/2026-10-26-record-user-intent/` | User approved intent.md (I4 rewritten: "The contract grows by addition"), Roles, config.yaml (incl. the `context` sentence replaced, recorded in design D3) |
| `refactor-math-layers` | review round 4 in progress (12.1 D35, 12.2 singular cases, 12.3 no validity condition, 12.4 cse, 12.5 simplify strategy, 12.6 no equality, 12.7 is_zero private, 12.8 compile, 12.9 exact numbers only, 12.10 homogeneous, 12.11 Position/Displacement; written, not applied) | `openspec/changes/refactor-math-layers/` | To be revised against intent.md |
| `.vscode` ty setup | done by the user (committed in `a74929f`) | `.vscode/settings.json` | ty and ruff from the environment |

## 11. Next steps
1. Revise `refactor-math-layers` against intent.md. Items, each framed functionally: runtime enforcement of every static promise (I0, I3; fixture lines also executed at runtime); `to_axis_angle` contract independent of the simplifier; `==`/`hash`/`bool` fail loudly until decided; `is_zero` public or not; the name `lambdify`; meaning of floats; `Transform.of_rotation` (I7); replace the Goals line "runtime checks only where measured"; re-label the two accepted risks (axis compiled alone, vanishing-norm normalize) as violations or open items.

## 12. Open questions ⚠️
- None.

## 13. Re-attach checklist
- None: everything is in the repository.

---
<sub>Audit: 13/13 sections · 6 rules · 10 corrections · 9 data points · secrets removed: none found · generated by active-memory</sub>
