# Design

## Context

See `proposal.md` - Why. Current state relevant to the approach:

- Tooling (from `setup-development-workflow`): Python 3.14, uv, ruff 0.17.2 (`select = ["ALL"]`, preview), ty 0.0.91 pinned exactly with every rule at error, inline suppressions neutralized, edit-time hook on every edited `.py` file.
- Installed: sympy 1.14.0, jax 0.12.0.1 (CPU; CUDA jaxlib not installed).
- Legacy `src/robotic/`: `transformations.py` (1185 lines), `manipulator.py` (449 lines), `utils.py` (`draw_frame`); 420 ruff errors, 85 ty diagnostics, 3 of 22 tests failing. Nothing in it is reused.
- sympy is untyped for ty: `Symbol`, `Float`, `Integer`, `Rational`, `cos(q)` and `simplify(...) -> Expr` are typed, but `q + 1`, `q * q`, `Integer(1) / 10` and `sympy.pi` infer `Unknown` and pass silently.
- `exams/2024_09_19.ipynb` builds six frames from elementary rotations (`sympy.pi/2`, `-sympy.pi/2`), chains them, and then uses `Manipulator`, DH and the Jacobian, which are out of scope.

## Goals / Non-Goals

**Goals:**
- Code reads like a handwritten exam solution: frames visible in every type (^A R_B, ^A T_B, ^F v), exact values by default, verbose where that makes the semantics clear.
- Every rule the design relies on is verified by a tool (ty, ruff, a test), never protected by documentation alone.
- Static checks first, and every static promise is also enforced at runtime, at the operation that breaks it (intent I0, I3; D35). Runtime checks alone only where types cannot express the rule.

**Non-Goals:**
- A numeric matrix algebra. Numbers exist only as the output of a compiled function.
- Static tracking of which symbols an expression contains (D38 tracks whether a value has symbols, not which).
- Catching statically the wider types the user writes explicitly (`v: Displacement[Scalar, Frame] = ...`, user generic helpers); the runtime check of D35 catches their first misuse. See Risks.
- Euler angles and RPY (D24).
- A general matrix type (D21).

## Decisions

Names are settled in D18.

### D1. sympy is the only algebra; numbers only through compilation

One symbolic representation for every quantity. Numeric work is a function compiled from a symbolic quantity (D11); there is no numeric matrix class. `evaluate` (D38) substitutes exact values and returns a constant, never a number.

Rejected:
- Two worlds, `Symbolic` and `Numeric` (names since reused with another meaning, see D18), with an `S` parameter on every class (`Matrix[S]`, `Vector[F, S]`, `Rotation[To, From, S]`) and world-gated methods (`def evaluate(self: Matrix[Symbolic]) -> Matrix[Numeric]`, probed working in ty 0.0.91): it worked, but solved a problem that does not exist. Simulation needs a compiled function, not a numeric algebra. Reopened in another form in review round 5 (D38): a kind parameter separating constants from expressions, for a problem intent I8 makes real (which questions have meaning), not for numbers.
- Separate `SymbolicMatrix` / `NumericMatrix` / `SymbolicPoint` classes: two APIs for one concept.
- Backend wrapper classes holding whole arrays: once compilation exists they are a burden on both sides.
- Partial evaluation, then full-only `evaluate`, then an exact partial-binding operation: each added a second way to get numbers and confused what a matrix is.
- Static symbol tracking (`Expression[V]` with `Literal` unions, `Symbol[N].replace_in`, `numeric()` on `Expression[Never]`): possible in ty, rejected for cost.
- Set-based symbol removal `evaluate(set[str])`: ty infers `Unknown`; `Symbolic("q1", ...)` widens to `str` and falsely reaches `Numeric`.
- A schema class (`Joints(Variables)` + descriptor + `dataclass_transform` + field specifier, matrices naming their schema): worked in ty, rejected for time and effort.
- CasADi, SymEngine, SymForce, Rust, Julia (chat 0): not reconsidered.

### D2. Complete typed boundary around sympy (the "unsafe" rule)

The scalar class `Scalar` wraps a `sympy.Expr`. No public signature accepts or returns a sympy type. `isinstance` checks and `Unknown` values exist only inside the boundary modules, the way Rust confines `unsafe`. Probed: `q + 1`, `1 + q`, `q * q`, `q.cos()`, `Scalar.pi() / 2` all infer `Scalar`; `Vector(q + 1, Scalar.pi() / 2, 0.1)` gives `[q + 1, pi/2, 0.100000000000000]`.

Arithmetic accepts `ScalarLike = Scalar | int` on both sides, and `Expression` besides wherever the result is an expression (D38; floats are rejected, D4).

Rejected:
- Entries typed raw `sympy.Expr`: arithmetic on it is `Unknown`, so every formula silently loses its type.
- A Scalar Protocol shared by symbolic and numeric scalars: no numeric scalar exists after D1.
- An expression-only scalar or per-scalar JAX values (chat 0): measured 299 µs for an entry-by-entry 3x3 product with 0-d JAX arrays versus 6.8 µs under `jit`.

### D3. sympy confined by ruff `banned-api` (configuration exception)

`q + sympy.Symbol("p")` passes ty silently through sympy's untyped `__radd__`. ty cannot see it, so the linter closes it: `[tool.ruff.lint.flake8-tidy-imports.banned-api] "sympy"` bans the import everywhere, with a per-file exception for the boundary modules.

| Exception | Scope | Reason |
|---|---|---|
| `TID251` ignored | `src/robotic/symbolic.py` only (D29) | This module is the boundary of D2: the only code allowed to touch untyped sympy values. |
| `SLF001` ignored | `src/robotic/symbolic.py` only (D29) | `Compiled(quantity, parameters)` reads the quantity's private layout and lambdifies the private sympy expressions of its scalars together, across classes. Probed with ruff 0.17.2: private access inside a class on a parameter annotated with that class (`other: Scalar`, `other: Rotation[From, Inner]`, `other: Self`) passes; access from another class, a module function or an unannotated loop variable is flagged. Privacy in the boundary is module-wide, the way `unsafe` is. |
| `PLR0904` (`too-many-public-methods`) ignored | `src/robotic/symbolic.py` only | The rule counts dunders: `Scalar` has 23 public methods, all fixed by the spec (11 arithmetic operators for both sides, D2; 6 functions, D5; `pi`, `simplify`, `latex`, `is_close`, `__init__`, `__repr__`; `is_zero` left the API in review round 4, D31; `symbol` became the `Symbol` constructor and `is_close` was added in review round 5, D38). `Expression` is in the same case (the operators, the functions, `simplify`, `latex`, `evaluate`). Rejected: raising `max-public-methods` globally (loosens the check everywhere); splitting `Scalar` (contradicts D5). |
| `S404`, `S603` ignored | `tests/test_negative_typing.py` only | The negative type test (D10, D32) runs ty in a subprocess with fixed arguments and no shell, the same exception `.claude/hooks/` has. |

The configuration names rules by name (`banned-api`, `private-member-access`), as ruff's preview rule `rule-codes-in-selectors` requires; codes are used in this document for brevity.

Scope (user's decision): the ban holds in `src/`, where it is fully verifiable and where the non-boundary layers (geometry, rotation, transform) would otherwise drift into raw sympy unnoticed. It is lifted in tests and notebooks, a limit of the tool accepted as such: `tests/**` already ignores `TID251` for the oracle libraries and the ignore is per rule, not per banned module; notebooks are excluded from ruff (`extend-exclude = ["*.ipynb"]`). No configuration is added for either.

Rejected: lifting the ban everywhere because it cannot reach tests and notebooks (gives up the part that is verifiable); a documented "never import sympy" rule (a rule only docs protect); an ast-grep rule (ruff already has the mechanism).

### D4. Symbols, constants and floats

- Only the library creates symbols (`Symbol(name)`, `class Symbol(Expression)`, D38; `Scalar.symbol(name)` until review round 5), with fixed assumptions. Probed: `Symbol("q2") + Symbol("q2", real=True)` is `q2 + q2` with two distinct free symbols; one creation path closes it.
- Exact constants are methods: `Scalar.pi()`, `.sqrt()`, `.exp()`. A module-level `pi` is shadowed silently by `math.pi`.
- The symbolic side takes exact numbers only (review round 4, user's decision; intent I0, I2): `int`, `Scalar` constants (`Scalar.pi()`, `Scalar(3) / 10`, `.sqrt()`) and symbols. A `float` is a type error and raises at runtime (D35). Floats exist on the numeric side only: parameter fields, the values given to a compiled function, its results. User (2026-10-27): "non voglio vedere scritto 6.28: 2*pi ha un senso concreto per me e risulta facilmente semplificabile per sympy se necessario"; "se impongo che posso usare solo scalari esatti, non float, ho una visualizzazione pulita sempre".

Why (probed): a float is a number near the value meant, not the value. `Rz(math.pi).to_axis_angle()` gave first `(0, 0, 1.0)` with angle `π − 1.22e-16` and second `(0, 0, −1.0)` with angle `−π + 1.22e-16`: the half turn was not recognized, so the second solution is not the course's (−r, π), and the printed result carries the noise; `Scalar.pi()` gives (r, π), (−r, π). Float `Rz(q) Rz(pi/2)[0,0]` simplified to `-1.0*sin(q) + 6.12323399573677e-17*cos(q)` against the exact `-sin(q)`; `solve(cos q = 0.5)` gave `[1.04719755119660, 5.23598775598299]` against `[pi/3, 5*pi/3]`.

Typing: `int` is the annotation, not `float`, because a `float` annotation also accepts `int` while an `int` annotation rejects `float`. `bool` is a subclass of `int`; the runtime check of D35 rejects it.

Rejected: floats accepted as written (rounds 1-3: "writing `0.3` means not caring about exactness"; it let near values pass as exact ones silently, against intent I0); converting floats to fractions (`nsimplify`: `0.3` becomes `3/10`, but `math.pi` is not π, and a guessed conversion hides what was written); raw float typed as `Numeric` with a documented trap table (only docs protected it).

### D5. sympy functions as methods

`.simplify()`, `.latex()`, `.compile(...)`, trigonometric functions and `atan2` are methods on the value. Rejected: free functions (they need docs to be found; style rule: free functions only when really needed); a static-only utils class.

What `simplify` promises (review round 4; intent I4, I5): an equal expression, in a simpler form when one is found; the form is not promised, so a better strategy later changes no contract. Which sympy strategy runs is Claude's choice, inside the boundary. sympy's own documentation says `simplify` "is not a well defined term and the exact strategies this function tries can change in the future versions of SymPy" and recommends the specific functions. Probed with sympy 1.14 on entry (2,2) of the user's `monster` (identity): `simplify` 0.22 s and `trigsimp` (default method) 0.53 s both leave a polynomial in sin q and cos q; `trigsimp(method="fu")` gives `1` in 0.05 s, `method="groebner"` in 0.38 s. On a norm (‖Rz(q) x‖) and on entry (1,1) of 3-, 4- and 6-joint alternating chains, `trigsimp` gave the same result as `simplify`, 1.5 to 4 times faster (6 joints: 0.21 s versus 0.30 s for one entry). The library's expressions are products and sums of sin and cos of symbols, with square roots from norms and `atan2` from axis-angle, so the strategy is chosen among the trigonometric ones and verified on the whole test suite (task 12.5); the earlier figure "6 joints simplify did not finish in 200 s" was measured through the library on whole rotations and is re-measured there.

Decided during apply (task 12.5): `factor_terms(trigsimp(e, method="fu"))`. `fu` alone gave the same results as `simplify` on every expression the tests check (the 2R tip, norms, `Abs(sin q)`, `R R⁻¹`, `1/2` for cos(π/3)) and the exam notebook's values, but printed the notebook's wrist expanded (`a_1*cos(q_1) + a_2*cos(q_1)*cos(q_2) + …`) where `simplify` printed it factored (`(a_1 + a_2*cos(q_2) + d_4*sin(q_2 + q_3))*cos(q_1)`), the exam's form (intent I1); `factor_terms` after `fu` restores it, and all 13 notebook outputs print as before. The values agree to 8.9e-16 over 200 random configurations.

Nothing simplifies implicitly (review round 6, user's decision, 2026-10-30): no constructor, operation, `to_axis_angle`, `compile` or `latex` calls `simplify`; it runs only when the code calls it. Simplification is costly and serves form only: on Rz, Rz Rx and Rz Rx Rz Rx the simplified skew norm had as many operations as the raw one, and on the last it took 47 s inside `to_axis_angle`.

Rejected: `sympy.simplify` (heuristics that sympy itself says may change between versions, slower on every probe); `trigsimp` with its default method (misses the `monster`); `fu` without `factor_terms` (the wrist above).

### D6. Frames are types, stored at runtime, invariant

`Position[K, F: Frame]`, `Displacement[K, F]`, `Direction[K, F]`, `FreeAxis[K, F]` (subtype of `Direction[K, F]`); names of D37, kind `K` of D38. The value stores its frame at runtime. Probed: with the stored frame and methods taking a displacement in `F`, ty infers `F` invariant; a phantom `F` was covariant, which let a displacement in `Tool` widen to one in `Frame` and mix with one in `Base`.

A frame creates its own positions, displacements and directions (`Tool.position(...)`, `Tool.displacement(...)`, `Tool.direction(...)`, `Tool.origin()`): a frame gives a position origin and axes, a displacement axes only.

Confirmed by the user against the principle "data is bare components, meaning is in the type": the components stay bare data (a sympy matrix, a JAX array); the frame is a label beside them, like `^{Tool}` beside the numbers of a handwritten solution. Three things read it at runtime: LaTeX (D19), `compile`, which copies it into the compiled result's aux data (D12), and the type checker's inference. Re-probed in ty 0.0.91, bound values: a stored frame makes `Stored[Tool]` not assignable to `Stored[Frame]`; a phantom `F` widens silently even with `__add__(self, other: Phantom[F])`, and a constructor without a frame argument gives `Phantom[Unknown]`.

Rejected: opt-in frames; a `NoFrame` default; free points and vectors; `Point3(Tool, ...)` constructors; Protocol frames; an explicit old-style invariant `TypeVar` (not needed, and D10 forbids it).

### D7. Affine arithmetic tracks the homogeneous w

P − P → V, P ± V → P, V ± V → V, k·V → V; P + P and k·P are type errors. Reason: with [p; 1] and [v; 0], 1 − 1 = 0, 1 ± 0 = 1, 0 ± 0 = 0, and 1 + 1 = 2 is not a point. Coordinates as a vector are explicit: `p - Tool.origin()`.

Rejected: allowing `2 * p` (it came from a misread of "2 * vector").

### D8. Rotations and transforms carry both frames

`Rotation[Target, Source]` is ^A R_B; `Transform[Target, Source]` is ^A T_B (names and order in D16). Composition `@` chains inner frames; inverse swaps them. `Displacement[K, F]` is an arrow resolved in F: changing the frame of resolution is symbolic only (rotation or transform applied to a vector or point); numbers come from compiling the resolution already chosen. `T @ point` is R p + t; `T @ vector` is R v.

Rejected: frames on vectors only; numeric frame changes on compiled results.

### D9. Receiver-first rule

No function or constructor in the API solves a frame from two arguments. A frame is fixed by the receiver (or an explicit frame argument) and every other argument is checked against it.

Probed in ty 0.0.91 (with the stub's `to=` / `from_=` spelling, replaced by D16): invariance stops widening existing values, but inline values are built with an invented union. `Transform(Rotation.about_z(q, to=Base, from_=Link1), Link1.origin())` silently gives `Transform[Base | Link1, Link1]`, and applying it to a `Link1` point gives `Point[Base | Link1]`. The same happens with `Self` classmethods on frames (`Base.rotation_about_z(...)`) and with an inline invariant key `Key(Base)`. Receiver-first `Rotation.about_z(q, to=Base, from_=Link1).with_origin(Link1.origin())` reports "Expected `Point[Base]`, found `Point[Link1]`"; inline `@` with mismatched frames reports an error.

Consequence: the stub's `Transform(rotation, origin)` constructor is replaced by a transform built from its rotation, which fixes both frames, and the origin checked against them (D17).

Rejected: fixing inline widening through variance alone (`Transform` is already invariant); pre-bound frame keys (`BASE = Key(Base)`) as the user's responsibility (only docs protect it).

### D10. PEP 695 generics, gated by a negative type test

PEP 695 syntax only (ruff UP040, UP046, UP047 enforce it). Because ty is pre-1.0 and inference decides variance, a negative fixture of mistakes (one per line) is run by pytest through ty, which asserts every line is an error. The test gates every ty upgrade.

The stub's fixture caught 8 of 10 (`tip + tip`, `2 * tip`, `Tool` + `Base` vectors, transform applied to a wrong-frame point, `Base←Link1 @ Base←Link1`, `q1 + "a"`, `compiled(0.5)`, `Parameters()` missing `q1`); the two misses are the inline cases closed by D9 and the explicit widening listed in Risks.

Rejected: old-style `TypeVar` with explicit variance; characterization tests or `xfail(strict)` (chat 0).

### D11. Compilation: `compile` against a parameters dataclass

Named `lambdify` until review round 4, sympy's word for it (intent I5: the tools stay out of the contract); renamed `compile` by the user, a method name the naming rule allows (D16). The code review of round 4 flagged it as a clash with the builtin `compile`; closed with no change (user, 2026-10-30): a method never shadows a builtin.

`compile[P: DataclassInstance](self, parameters: type[P]) -> Compiled[P, Self]`, `Compiled.__call__(values: P) -> Numeric[Self]`, on symbolic quantities; a constant compiles with no parameters (D38). `P` is a user-written frozen dataclass (one for the whole robot, e.g. `Joints`). Fields are matched to symbols by name; ordering is internal; extra fields are allowed. Type hints survive, so `compiled(0.5)` and `Parameters()` with a missing field are type errors.

Validation is at runtime, once, at compile time: every field annotated `float` or `jax.typing.ArrayLike` (string annotations resolved with `typing.get_type_hints`; `dataclasses.fields` alone returns the string `'float'`), every free symbol has a field, otherwise raise. `float` is for single-configuration code (the exam notebook); `ArrayLike` is for code that batches: JAX transformations keep the wrapped function's signature, so `jax.vmap(f)(batch)` with `def f(q: float)` is a ty error, while a field and a parameter annotated `ArrayLike` accept floats and arrays and still reject `Joint(q1="a")` and `compiled(0.5)` (probed in ty 0.0.91). The quantity and the dataclass are linked only by names; a static link needs the schema rejected in D1.

`jax.vmap` is the only way to batch (D34). `ArrayLike` is the annotation that lets `vmap` type-check, not a promise that the compiled function takes arrays: every field value must be a scalar of shape `()`, which is what `vmap` hands in. A field holding an array raises `ParameterShapeError(ValueError)`, naming the field and pointing to `jax.vmap`. The check sits inside the traced kernel, where shapes are static, so it runs when JAX traces (once per input shape), not on each call. Found in review round 1: an array passed directly reached `jnp.stack`, which failed with "Cannot stack arrays with different numbers of dimensions" for any quantity with constant entries (every rotation and transform), because the constant entries stay of shape `()`.

Rejected (review round 1): broadcasting the entries so a direct array call works (a second batching path beside `vmap`, and it needs rules for fields of different lengths, (N,) with (M,)); leaving the direct call unhandled (an internal stacking error for a user mistake).

JAX is the only backend: `sympy.lambdify(..., modules="jax")` under `jax.jit`, with `jax_enable_x64`. Autodiff is part of the numeric path (whitelisted).

Compilation does not simplify (a case of D5: nothing simplifies implicitly), and it eliminates common subexpressions (review round 4; the first is the user's decision, the second Claude's; intent I5, I6). `simplify` stays explicit, for reading a result: it gives no numeric gain and can block. Probed on the user's `monster` (eight rotations whose product is the identity): simplify 0.85 s, after which sympy still could not reduce two diagonal entries to 1; max error against I over q in [0.1, 10] 4.4e-16 with and without simplify; 11.9 µs versus 14.2 µs per call. Float64 error grows about linearly with the chain, ~1e-16 per rotation (R R⁻¹ chains of 4, 8, 16 rotations: 3.3e-16, 6.7e-16, 1.4e-15). Unsimplified products grow exponentially as expressions (entry (1,1) of a 24-rotation chain: 8609 characters, ×2.7 every 4 rotations), and XLA's compile time with them; `sympy.lambdify(..., cse=True)` computes each repeated piece once: 24 rotations 8.65 s → 0.18 s, 32 rotations > 90 s → 0.50 s, 48 rotations > 90 s → 17.5 s (the cse pass itself). sympy leaves `cse` off by default because the pass has its own cost, which small expressions do not repay; the library's quantities are products of rotations, where it pays.

Rejected: an implicit `simplify` inside `compile` (no precision or speed gain measured; a 6-joint alternating chain did not simplify in 200 s, so compiling could never finish); compiling without `cse` (sympy's default; the exponential compile time above).

Because validation reads the annotations at runtime, the annotation's name must exist at runtime. ruff's `typing-only-third-party-import` (TC002) treats a bare `ArrayLike` in a dataclass field as typing-only and asks to move its import under `TYPE_CHECKING`, which would make `get_type_hints` fail; the tests and docs write `jax.typing.ArrayLike` through the runtime-used `jax` binding instead. (ruff's `runtime-evaluated-decorators = ["dataclasses.dataclass"]` would state this in configuration; not added, the qualified spelling needs no exception.)

Rejected: `float` fields only (batched code is a ty error); `ArrayLike` fields only (the exam code would write `q1: ArrayLike` for one configuration); caller-ordered symbols `compile((q1, q2))`; sorted-name order (`"q10" < "q2"`); creation-order counters; runtime-generated parameter classes (`type(...)`, `make_dataclass`, metaclass: ty sees `Unknown`/`Any`); a `backend` parameter, `Backend(StrEnum)` and per-backend overloads (until a second backend exists); `Piecewise` branching inside compiled code.

### D12. Thin frame-carrying compiled result

A compiled point, vector, rotation or transform returns a thin result `Numeric[T]` in the natural shape (`(3,)`, `(3, 3)`, `(4, 4)`, not sympy's `(3, 1)`), registered as a JAX pytree: frame as static aux data, components as the only child. It has no frame operations; simulation code reads `.components`. `vmap` leading axes are configurations and are not modeled in the type.

Principle (the user's): data is bare components; meaning (the frame) is in the type, on both sides of the boundary. `(0, 0, 1)` only makes sense while you know where it was described.

Measured: a plain class returned from `jax.jit` raises `TypeError`; registered, it passes `jit`, `vmap` (frame kept, components `(2, 3)`) and `grad` (0.8775825618903728). Cost under `jit`: single call 8.3 µs raw versus 9.6 µs wrapped; `vmap` over 1000 configurations 18.2 µs versus 19.1 µs.

Rejected: a bare `jax.Array` (frame lost).

`jax_enable_x64` is set when `robotic` is imported, for the whole process (user's decision, review round 2). The library checks exam values against symbolic results, where float32 rounding would show as false mismatches. The side effect is stated in `docs/compilation.md` and `README.md`. Rejected: enabling x64 only inside `Compiled.__call__` (not probed; expected to fail under the caller's own `jit` / `vmap`, whose input dtypes are fixed when the outer trace runs, outside the scope); requiring the caller to enable it (a forgotten flag silently gives float32).

The printed form of a `Numeric` lists concrete components and shows a tracer as JAX prints it, so printing works under tracing (review round 2: `.tolist()` raised `ConcretizationTypeError` inside `jax.jit`).

`Numeric._unflatten` builds `Numeric[Symbolic](...)` through the subscripted alias (review round 3, Claude's decision). Probed in ty 0.0.91: a plain `Numeric(...)` infers `Numeric[Unknown]` and is an `unsound-return-statement` (and, bound to an annotated local, an `unsound-assignment`); annotating the return `Numeric[Any]` is banned by ruff's `any-type`. The alias dispatch costs microseconds and runs when JAX unflattens at a trace boundary, not on each compiled call.

Derivatives are taken of `.components`, not of the `Numeric` (review round 3, Claude's decision; the user delegated JAX-side decisions): `jax.jacrev(lambda q: compiled(P(q)).components)`. Rejected: making `Numeric` differentiable as a whole (the derivative would come back labelled `Numeric[Rotation[A, B]]` while holding dR/dq, which is not a rotation: the type would claim a meaning the numbers lack, against this decision's principle).

### D13. Axis-angle is symbolic

`R.to_axis_angle()` returns an `AxisAngleSolutions` with `.first` and `.second` (each an `AxisAngle`: `.axis` and `.angle`), without evaluating anything; it compiles like any quantity, and so does an axis alone. Content (course convention): angle in (−π, π]; regular (r, θ), (−r, −θ); half turn (r, π), (−r, π); identity (a, 0), (−a, 0) with `FreeAxis`, default z. The axis of ^A R_B satisfies R r = r, so it has the same components in A and B.

Which case applies (review round 5, intent I8; D38): a constant rotation, `Rotation[Scalar, A, B]`, is decided on its values with `is_close` at the default tolerance: regular when the norm of the skew part is not close to zero; otherwise the identity when the cosine is close to 1, and the half turn when it is not. A rotation whose skew part vanishes has θ = 0 or π, so the cosine is ±1 and no undecided branch is left (a skew norm 2 sin θ within 1e-9 puts cos θ within 1e-18 of ±1). A symbolic rotation, `Rotation[Expression, A, B]`, always gets the regular formulas, whatever values the symbols will take. The symbolic side does not predict where those formulas break, because it does not have the values; what happens there is open on purpose (intent.md; proposal, Out of Scope). User (2026-10-26): "Nel lato simbolico ho due possibili funzioni che variano a seconda del valore dei simboli, che non ho e quindi non riesco a determinare, quindi in quel momento per quanto ne so io l'indeterminazione è valida e rimangono entrambe le funzioni." The exact cases of a symbolic rotation are reached by evaluating first: `R.evaluate({...}).to_axis_angle()`.

Until review round 5 the case was chosen on the simplified entries (no symbol left: exact), so `R @ R.inverse()` was recognized as the identity when the simplifier removed the symbols and missed it when it did not (the user's `monster`, eight rotations whose product is the identity); which answer the user got depended on the simplifier. Now it depends on the type.

Consequence: a symbolic rotation whose skew part vanishes at some or all configurations (a half turn about a symbolic axis (a, 1, 0); `R @ R.inverse()` with `R = Rx(q)`) gets an axis that is undefined there, as at any singular configuration; where sympy evaluates the skew part to zero itself, the axis prints `nan` (0/0). Normalizing an expression never raises (D20). User (2026-10-30): "il nan mi sembra quantifichi una cosa che avviene realmente, cioè le singolarità". Evaluating those solutions with exact values where the skew part vanishes raises `UndefinedValueError` (D38, review round 7); only compiled code stays open.

Measured in review round 5 (task 13.5; finding #6 of the code review of round 4): on the 4-joint alternating chain Rz Rx Rz Rx, `to_axis_angle` took 133 s before and 89 s after the zero tests were removed. Simplifying the rotation takes 0.4 s; one simplification of the skew part's norm takes 47 s, and the regular case does it twice (D33, open point), so the two account for the rest. Review round 6 removed both, with the simplification of the rotation (D5: nothing simplifies implicitly): `to_axis_angle` uses the rotation and ‖s‖ as they are. Re-measured (task 14.1): 0.014 s on the same chain; `R.simplify().to_axis_angle()`, the explicit path to the compact form, 0.57 s.

Probed (the singular case, open on purpose): for `Rz(q)` sympy gives axis `[0, 0, sin(q)/Abs(sin(q))]` and angle `atan2(Abs(sin(q)), cos(q))`; compiled at q = 0.5 it gives `[0 0 1]`, at q = 0 `[0 0 nan]`, at q = π `[0 0 1]` only by luck (float sin(π) = 1.2e-16). `Rx(π/2) Rz(q)` at q = π gives `[nan -inf inf]`.

Rejected: raising for the identity; `NoRotation` / `HalfTurn` / `Undecided` union arms; an "all cases" object with mandatory `.evaluate()`; `FreeAxis` as a non-subtype; the case chosen on the simplified entries (rounds 1-4, above). Evaluating first, then extracting, was rejected until review round 5, when no exact `evaluate` existed (D1); it is now the path to the exact cases.

### D14. Error principle

Raise when no answer exists (normalizing a zero vector, a free symbol without a field at compile time), and when an operation breaks a static promise (D35). Return a typed result when a singular case still carries information (axis-angle of the identity).

### D15. The exam notebook is the benchmark

`exams/2024_09_19.ipynb` keeps only the cells the new API can express, rewritten against it; the cells that need `Manipulator`, DH or the Jacobian are deleted. The notebook is the change's benchmark, and later changes add cells back as the API grows. The deleted cells stay reachable at the `legacy` tag.

Rejected: keeping the out-of-scope cells non-runnable until their change lands (a notebook that does not run top to bottom is no benchmark).

### D16. Frame keywords `target=` / `source=`

^A R_B maps coordinates expressed in the source B to the target A. That mapping reading names the frame pair at every layer: type parameters `[Target, Source]`, attributes `_target` / `_source`, properties `.target` / `.source`, keywords `target=` / `source=`. Constructors that need frames their arguments cannot fix take them as keywords: `Rotation.about_z(q1, source=Link1, target=Base)` is `Rotation[Base, Link1]`; `Rotation.identity(source=Tool, target=Wrist)`; `Rotation.about_axis(axis, q, source=Link1)`, where the axis's frame fixes the target. Each frame is fixed by its own keyword, so D9 holds. Target comes first so the types read like the composition ^A R_B ^B R_C = ^A R_C: `Rotation[A, B] @ Rotation[B, C] -> Rotation[A, C]`. Positions and displacements keep the "expressed in" reading (`Displacement[K, F]`, `.frame`).

Naming rule (user's): a name that can shadow one of the language's (a module-level function, class or constant, a variable, a parameter) never takes the name of a keyword, soft keyword or builtin, and escaping a keyword with a trailing underscore counts as a clash. Methods and attributes are always reached through their class or value, which acts as the namespace Python lacks, so they cannot clash and may take a builtin's name (clarified in review round 4; user: "la classe fa da sorta di namespace, dato che Python non lo ha, è impossibile avere dei clash in generale di nome"); this is also why free functions are avoided. Builtin shadowing is already a ruff error (`A001`–`A005` under `ALL`); the escape is not caught by any rule today, so the separate ast-grep style-rule change bans identifiers ending in a single `_`.

Probed in ty 0.0.91 (with the earlier `of=` / `relative_to=` spelling; keywords bind the same way), bound and inline: `assert_type(Rotation.about_z(0.5, of=Link1, relative_to=Base), Rotation[Base, Link1])` passes, and a receiver-first method taking a point of the target frame reports an inline `Link1` point ("Expected `Point[Base]`, found `Point[Link1]`").

Rejected:
- `of=` / `relative_to=` (the course's orientation sentence, "the orientation of B relative to A"; mixed with mapping-named type parameters `To` / `From`, it gave one frame pair five vocabularies: A/B, `To`/`From`, `relative_to`/`of`, `_target`/`_source`, `T`/`S`).
- `relative_to` / `of` at every layer (loses the mapping reading, which is how the user reads a rotation).
- `[Source, Target]` order (reads source to target, but composition types read backwards against ^A R_B: `Rotation[B, A] @ Rotation[C, B] -> Rotation[C, A]`).
- `to=` / `from_=` (the stub's spelling; `from_` escapes the keyword `from`).
- A builder from the frames, `Link1.relative_to(Base).about_z(q1)` (adds an intermediate class whose only job is to carry two frames).

### D17. A transform is the homogeneous matrix of a rotation and a position

Revised in review round 4 (user's decision; intent I0, I2, I7). A transform ^A T_B is an element of SE(3), [R p; 0 1]: its fourth column is [p; 1], the position vector of B's origin in A. It is built only from its rotation:

```python
Rotation.about_z(q1, target=Base, source=Link1).homogeneous(
    Base.position(0, 0, d1)
)  # Transform[Base, Link1]
Rotation.identity(target=Base, source=Link1).homogeneous(
    Base.position(0, 0, d1)
)  # pure translation
Rotation.about_z(q1, target=Base, source=Link1).homogeneous()  # coincident origins
```

`Rotation.homogeneous(self, translation: Position[Target] | None = None) -> Transform[Target, Source]` (D37 names): the rotation fixes both frames (D9), and the translation is the source's origin expressed in the target, a position; `None` means `Target.origin()`. Named `origin` when first applied; renamed `translation` by the user (2026-10-27), the parameter and the property alike. The argument is a point because the fourth column is a position vector, which carries 1 in homogeneous coordinates; a free vector ([v; 0]) would give a last row of zeros, a matrix outside SE(3). In this library the position vector is a `Point` (stored as its coordinates from the frame's origin), while `Vector` is a free vector (`T @ vector` is R v, D8). So a transform that is not in SE(3) cannot be written: a vector is a type error, a point in the wrong frame is a type error, and both raise at runtime (D35). The transform exposes `.rotation` and `.translation` (a `Position[Target]`).

User (2026-10-27): "transformation per me è SE(3)"; the earlier spelling carried the translation as a `Vector`, a free vector, though it is a position.

Python has no private constructors, so `Transform.__init__` stays public. Decided during apply (Claude's): it takes a private `_Parts` (the rotation and the origin) that only `homogeneous` builds, after checking both; anything else raises `TypeError` pointing to `homogeneous`, and passing a rotation is a type error. The design first said `Transform(rotation)` with `homogeneous` setting the origin through a private factory, but ruff's `private-member-access` flags a call from `Rotation` to `Transform`'s private factory or attributes (probed with ruff 0.17.2: a private classmethod call, an assignment through an annotated local and a module function were all flagged), while importing a private class from a sibling module passes. A constructor taking rotation and origin together would let ty invent a frame union inline (probed, below). `transform.py` imports `Rotation` only for annotations, so `rotation.py` imports `transform.py` without a cycle (D29).

Probed in ty 0.0.91 with `Point`, `Vector` and `Rotation` invariant (earlier rounds):
- `__init__(self, rotation: Rotation[To, From], origin: Point[To])` and the self-typed variant: inline silent (`Transform[Base | Link1, Link1]`), bound caught.
- `__init__(self, rotation: Rotation[Target, Source], translation: Vector[Target])` (re-probed in review round 1): inline infers `Transform[Base | Link1, Link1]`; the mistake surfaces only at an annotation, explicit `Transform[Base, Link1](...)`, or the first operation with a `Base` value.
- Receiver-first methods: an inline wrong-frame argument is an error (D16).
- `Transform.of_rotation(R).with_translation(t)` (handoff, 2026-10-26): all four frame mistakes of the fixture caught.

Open on purpose (user, review round 4: "non voglio chiudere quella strada ancora"): a builder starting from the rotation, as `Transform.of_rotation(R)` with a step setting the origin, or the frames-first builder of rounds 1-3 (`Transform(target=, source=).with_rotation(R).with_translation(t)`, whose translation was typed as a free vector). Both were safe in ty; neither is implemented now, and adding one later is an addition (intent I4).

Rejected:
- `vector.homogeneous(R)` beside `R.homogeneous(p)`: two spellings of one transform (D34), and a vector has one frame, so the source would need a keyword.
- Composition of elementary transforms, `Trans(t) @ Rot(R)`: needs an intermediate frame per transform (D25); kept for the DH change, where the course composes elementary transforms.
- `Transform(rotation, x, y, z)`; `Transform(rotation, translation)` (the inline hole above); `rotation.with_translation(v)` / `rotation.with_origin(p)` (reads as applying a translation to a rotation; `homogeneous` reads as the course's homogeneous matrix of R); a mutating builder.
### D18. Names

Rule (user's, D16): no clash with the language for names that can shadow one; a keyword escape counts as a clash; methods and attributes are namespaced by their class. Every candidate was swept against keywords, soft keywords, builtins, every importable stdlib module, and the top level of sympy, jax, jax.numpy and numpy.

| Name | Meaning | Rejected |
|---|---|---|
| `Scalar` | a rank-0 constant, built from integers and exact constants only (D38); what `norm` and `dot` of constants return | `Constant`, `Exact` (as names of a kind tag on the scalar, review round 5: the classes make the tag unnecessary, and exactness is true of both sides since D4), `Real` (`numbers.Real`, and `float` is registered as one), `Value` (`multiprocessing.Value`), `Term` (one part of a sum) |
| `Expression` | a rank-0 value built from at least one symbol (D38) | the user's name (2026-10-30). Rejected for `Scalar` in rounds 1-4 because of `ast.Expression`: the sweep above counts stdlib modules, while the naming rule (D16) covers keywords, soft keywords and builtins only, so the name is allowed |
| `Symbol` | one variable, a subclass of `Expression` (D38) | `Symbol` for every expression (a derivative could not require a variable) |
| `ScalarLike` | the alias `Scalar \| int` accepted by every scalar operation; the `-Like` convention of `numpy.typing.ArrayLike` and `jax.typing.ArrayLike` | `Number` (`numbers.Number`, `tokenize.Number`, `sympy.Number`; says "plain number" but accepts symbolic scalars), `Operand` (says nothing), `Numeral` (a digit string) |
| `Symbolic` | the shared base of `Scalar`, `Expression`, `Point`, `Vector`, `Rotation`, `Transform`: `simplify`, `latex`, `compile` | `Quantity` (`sympy.physics.units.Quantity`, pint: implies units), `Expression` (`ast`; a rotation is not "an expression" in course language), `Compilable` (covers only `compile`) |
| `Numeric[T]` | the thin compiled result of a `T` (D12); pairs with `Symbolic` | `Evaluated` (names the process), `Numerical` (same meaning, longer), `Sample` (random or time-series sampling) |
| `Compiled[P, T: Symbolic]` | the compiled function; `__call__(values: P) -> Numeric[T]` | `Compiled[P, R]` with the full result type (`R` was always `Numeric[...]`, and an unbounded `R` let `Compiled[Parameters, int]` pass; probed: the bound rejects it) |
| `AxisAngle[F]` | one axis-angle representation: `.axis`, `.angle` | `AxisRotation` (the stub's name for it: reads as a rotation object, confusable with `Rotation.about_axis`); in the stub `AxisAngle` named the outer result, which is not an axis-angle |
| `AxisAngleSolutions[F]` | the result of `to_axis_angle()`: `.first`, `.second`; the course's "two solutions"; a `Symbolic`, so it compiles as a whole | `AxisAnglePair` (says how many, not why; invites tuple unpacking) |
| `Rotation.homogeneous`, `Transform.translation` | the course's homogeneous matrix of R with the source's origin as fourth column (D17); `translation` is that position (user's name, 2026-10-27) | `Transform.with_rotation` / `with_translation` (the builder of rounds 1-3, D17), `Rotation.with_origin` (reads as giving a rotation an origin), `placed_at` (vague), `origin` (the name first applied) |
| `target`, `source` | the frames of a rotation or transform, in the mapping reading (D16) | `relative_to` / `of`, `To` / `From` (D16) |
| `Frame.notation` | how the course writes the frame, e.g. `0` or `3'` (D19) | see D19 |

`ValidityCondition` (a symbolic statement true where a result is valid, compiled to a flag) was removed in review round 4 with the flag (D28).

`Symbolic` here is not the rejected two-worlds scalar of D1: there is one symbolic base and one compiled result type, no `S` parameter and no gated methods.

### D19. A frame's notation is a class keyword

`class Base(Frame, notation="0")` renders `{}^{0}`; without the keyword the class name is used (`class Tool(Frame)` renders `{}^{Tool}`). Implemented with `__init_subclass__(cls, *, notation: str | None = None)` storing `Frame.notation: ClassVar[str]`. Reason: the exam numbers its frames (^0 R_1, ^3 T_6) and a class cannot be named `0`.

Probed in ty 0.0.91 and at runtime (with the earlier keyword `latex`): a misspelled keyword is a ty `unknown-argument` error and a runtime `TypeError`; a non-string value is a ty error (expected `str | None`).

Rejected: the keyword `latex` (the earlier name: a string on frames, while `latex()` is the render method on every quantity, so `rotation.target.latex()` failed and `point.latex` returned a bound method); `label` and `name` (both already used in `symbolic.py`; `name` also reads as the Python name); a class attribute `notation = "0"` (a misspelled attribute passes ty silently and the class name is printed); the keyword `symbol` (collides in meaning with `Scalar.symbol`, which creates a variable); always the class name (cannot express numbered frames).

### D20. Unit vectors come only from normalization

`Direction(displacement: Displacement[K, F])` (`UnitVector` before D37) normalizes its argument: it divides the components by the norm as computed, not simplified (review round 6, D5). A constant displacement raises `ZeroVectorError` when its norm is close to zero (`is_close` at the default tolerance; D14, D38). A symbolic displacement never raises: its direction is undefined where the norm vanishes, at some configurations or at all, a singular configuration (open on purpose). `d.normalize()` is the spelling (D34); it calls `Direction(d)`. As with D30, Python has no private constructors, so the public constructor itself guarantees the invariant: no `Direction` can hold components not divided by their own norm, and no length check exists.

Probed with sympy 1.14, real symbols, norm not simplified: (1, 1, 0) gives (sqrt(2)/2, sqrt(2)/2, 0); (3, 4, 0) gives (3/5, 4/5, 0); (q, 0, 0) gives (q/Abs(q), 0, 0), sympy reducing these square roots itself; (cos q, sin q, 0) gives (`cos(q)/sqrt(sin(q)**2 + cos(q)**2)`, `sin(q)/sqrt(sin(q)**2 + cos(q)**2)`, 0), and `.simplify()` gives (cos(q), sin(q), 0). Until review round 6 the norm was simplified first, so the last printed (cos(q), sin(q), 0) directly.

A norm that vanishes only at some configurations (q/Abs(q) at q = 0), or at all of them, does not raise symbolically; evaluating it there with exact values raises `UndefinedValueError` (D38, review round 7); its compiled value there is unspecified, open on purpose (intent.md; proposal, Out of Scope). Probed: it compiles to `nan` today, which is not a promise.

Rejected: the symbolic zero test of rounds 1-4 (simplify, then compare with 0): `(q + 1)**2 - q**2 - 2*q - 1` was not proved zero and was divided anyway, so whether normalizing raised depended on the simplifier (code review of round 4; superseded by D38); a constructor taking components with a unit-length check (proving x² + y² + z² = 1 symbolically can fail and would raise for a vector that is unit, against D14); a constructor taking components without a check (the type would promise what nothing guarantees).

### D21. No public matrix type

Rotations store their entries in an internal typed 3x3 class built from `Scalar` arithmetic only, so it needs no sympy and lives in the rotation module, outside the boundary (D29). A transform stores no matrix: its 4x4 entries are read from its rotation and origin. No public signature returns the storage. `Rotation` and `Transform` are the only public matrix-like concepts (D1: one class per concept) and expose their entries as `Scalar`s through a typed accessor (D26) and as LaTeX.

Rejected: a public frameless `Matrix` (a free value that does everything `Rotation[To, From]` does while bypassing the frame types, the hole D6 closed for points and vectors; nothing in this change needs one, and the Jacobian change decides about a general matrix when it needs one).

### D22. A rotation acts on a vector with `@`

`R @ v` is the only spelling: ^A v = ^A R_B ^B v, consistent with `Rotation @ Rotation` and `Transform @` positions and displacements (D8). Probed in ty 0.0.91: `base_link1 @ link1_tool @ approach` and `base_link1 @ (link1_tool @ approach)` both infer a displacement in `Base`; `(base_link1 @ link1_tool).inverse() @ Base.displacement(1, 0, 0)` infers one in `Tool`; a wrong-frame displacement at the end of a chain is an error.

Rejected: `.rotate(v)` (the active reading, the arrow moves, contradicts D8's passive one, the arrow stays and its frame of resolution changes); a second, named spelling (if one is ever added, its name must describe the passive meaning).

### D23. A rotation does not act on points

`Rotation @ Point` is a type error: a rotation relates axes and says nothing about origins, so with offset origins the result would have the right type and the wrong value (identity rotation, origin of B at (5, 0, 0) in A: the tip (1, 0, 0) in B is (6, 0, 0) in A, not (1, 0, 0)). The two honest readings stay explicit: `R.homogeneous() @ p` writes the coincident-origin assumption down; `R @ (p - B.origin())` resolves the grounded vector (D7). Probed in ty 0.0.91: both infer a position and a displacement in `Base`; `R @ tip` is an error.

At runtime, `Rotation.__matmul__` and `Transform.__matmul__` return `NotImplemented` for any other operand, so `R @ point` raises Python's `TypeError` instead of returning `None` (review round 3, user's decision).

Rejected: allowing it under an implicit coincident-origin assumption.

### D24. Euler angles and RPY are deferred

The benchmark (D15) needs none: every surviving rotation in `exams/2024_09_19.ipynb` has at most one nonzero angle (identity, `Rx(pi/2)`, `Rz(pi/2)`, `Rx(-pi/2)`; the `(pi/2, 0, pi)` line in cell 9 is overwritten on the next line), and frames 0 to 6 all have names. Euler brings its own design, so it gets its own change. Facts gathered for it: fixed-axes sequence ABC with angles (a, b, c) equals moving-axes sequence CBA with angles (c, b, a); 12 sequences cover everything; course naming: Euler = moving, RPY = fixed (rejected chat-0 names: `EulerPair`, `GimbalLockSum`).

Rejected: Euler and RPY in this change (enlarges it with a design the benchmark does not exercise).

### D25. Intermediate frames are declared by the user

A frame a product of elementary rotations passes through is an ordinary frame the user declares, named with D19 as the course writes it (`class Prime(Frame, notation="'")`, RF', RF''). Probed in ty 0.0.91: `first @ second @ third` through `Prime` and `Second` infers `Rotation[Base, Tool]`; skipping a factor (`first @ third`) is an error.

Rejected: library-provided anonymous frames (ty cannot create a fresh type per call, so every intermediate frame would share one class and a chain that skips or swaps them would type-check).

### D26. Entries are read with `entry(row, column)`, counted from 1

`Rotation.entry(row: Index3, column: Index3) -> Scalar` with `type Index3 = Literal[1, 2, 3]`; `Transform.entry` uses `Index4 = Literal[1, 2, 3, 4]`. Counting from 1 follows the course (r₁₃ is `entry(1, 3)`); a named method signals that this is not Python indexing; the index range is in the type, so an out-of-range index is a type error, not an `IndexError`.

Probed in ty 0.0.91: `entry(1, 3)` infers `Scalar`; `entry(0, 2)`, `entry(4, 1)` and an `int` index are errors; `for row in (1, 2, 3)` passes (ty infers `Literal[1, 2, 3]`); `for row in range(1, 4)` is an error (ty sees `int`). Accepted: user loops over entries use a literal tuple; library code walks the internal storage (D21), not `entry`.

Rejected: `R[0, 2]` for r₁₃ (an off-by-one on every transcription from a solution); `R[1, 3]` (square brackets promise Python's 0-based indexing); one property per entry, `r13` and so on (9 + 16 cryptic names).

### D27. No frame accessor on compiled results in this change

`Numeric[T]` exposes `.components` only. The meaning is the type `T`; the runtime frame, kept in the pytree aux data (D12), appears in the printed form (`Numeric[Point[Base]]([1.0, 0.5, 0.0])`). Nothing in this change reads a compiled frame: simulation reads `.components`, the benchmark compares numbers.

Additive path for a later change (probed in ty 0.0.91, one class, no new types): self-annotated methods offered only for the matching `T`, `frame[F](self: Numeric[Point[F]]) -> type[F]` and `target[A, B](self: Numeric[Rotation[A, B]]) -> type[A]`; `scalar.frame()` and `rotation.frame()` are errors.

Rejected: a plain `.frame` property (no single return type fits a point, a rotation and a scalar); per-concept result classes (reopens D18's single `Numeric[T]`); adding the frame accessors now (no reader in this change). The validity flag of earlier rounds took this gated path (`valid[F](self: Numeric[AxisAngleSolutions[F]])`) until review round 4 removed it (D28).

### D28. No validity condition: the symbolic side does not predict singularities

Serves intent I4 and I6; review round 4, user's decision. `to_axis_angle` returns the two solutions only; there is no `ValidityCondition`, no `.valid_when`, no compiled flag and no `tolerance` argument. Where a symbolic result breaks is a question about values, which the symbolic side does not have (D13); the behavior at singular configurations in compiled code is open on purpose (intent.md; proposal, Out of Scope), so any later answer (raise, flag, debug mode) is an addition. User: "Cercare di prevedere tutto quello che causa problemi ci porta a una spirale in cui perdiamo il controllo"; "sembra comunque un problema che deve risolvere il lato numerico".

Evidence kept for the open item, `Rz(q)` regular formulas compiled with JAX under `jit`:

| q | axis z | angle | exact flag `sin θ ≠ 0` | tolerance flag `\|sin θ\| > 1e-9` |
|---|---|---|---|---|
| 0.0 | nan | 0.000e+00 | False | False |
| 1e-12 | 1.0 | 1.000e-12 | True | False |
| 1e-06 | 1.0 | 1.000e-06 | True | True |
| 0.5 | 1.0 | 5.000e-01 | True | True |
| π | 1.0 | 3.142e+00 | True | False |

Without a flag, q = 1e-12 and q = π give plausible numbers with no warning (at π the axis is right only because float sin(π) = 1.2e-16). This is what the open item must weigh.

Rejected (review round 4): the compiled flag of rounds 1-3 (a `ValidityCondition` compiled with a tolerance, read through a gated `valid()`, carried as a second pytree child by every result): which case and which condition a rotation got depended on what the simplifier could prove, so a sympy upgrade could change the contract (intent I4), and it predicted on the symbolic side what only values decide. Rejected earlier: an exact `≠ 0` flag (the misses in the table); `Piecewise` branching in compiled code; forbidding an axis from compiling alone (reverses D13).

### D29. One boundary module

`src/robotic/symbolic.py` holds `Symbolic`, `Scalar`, `Expression`, `Symbol`, `ScalarLike`, `Compiled` and `Numeric`; it is the only module importing sympy and calling `sympy.lambdify`. `geometry.py` (frames, points, vectors), `rotation.py` (rotations, their storage, axis-angle) and `transform.py` build on the public `Scalar` API only. `Rotation.homogeneous` returns a `Transform` and `Transform` holds a `Rotation` (D17); how the two modules import each other without a cycle is an implementation choice of task 12.10.

Rejected: a separate boundary module per concern (scalar, matrix, compilation): every pair would need private access across modules; public accessors for raw entries (a "give me your sympy" method on every quantity) or a `Compiled` constructor taking pre-built parts (an explicit specialization could then attach any frame type to any numbers, against D20's principle that the public constructor guarantees the invariant).

### D30. Points and vectors have a public frame-taking constructor

`Position(x, y, z, frame)` and `Displacement(x, y, z, frame)` (`Point` and `Vector` before D37) exist because Python has no private constructors (as in D17 and D20); `Tool.position(...)` and `Tool.displacement(...)` call them and stay the spelling (D34). The frame comes last: the numbers, then the frame they are read in. It has one source, the `frame` argument, so D9 holds: `Position(1, 0, 0, Tool)` infers `Position[Scalar, Tool]`.

The `Point3(Tool, ...)` rejection (D6) stands as an API choice: docs and examples use the frame methods only.

Rejected: building points through hidden attribute assignment after `object.__new__` (hidden mutation); a constructor taking an opaque components object (still public, adds a type with no meaning).

### D31. Scalar construction, zero test and compile errors

- `Scalar(value: ScalarLike)` builds a constant: `Scalar(2).sqrt()` is the spec's "square root of the scalar 2". Results of sympy operations are wrapped by a private factory inside the boundary, which picks `Scalar` or `Expression` from the operands' classes (D38); the public constructor never takes a sympy type (D2).
- `Symbol(name: str)` creates a variable and is the spelling (review round 5, D34, D38; before, `Scalar.symbol(name)` called it, which would now return a value that is not a `Scalar`), with one fixed assumption (real), so D4's guarantee holds.
- There is no zero test (review round 5, intent I8, D38). The private `_is_zero` and `_has_symbols` of review round 4 are removed, and with them the declared exception to the style rule "free functions only when really needed" that they needed. Constants answer with the public `is_close`; whether a value has symbols is its class. Every helper lives on its type: the exactness check and the simplification are private methods of the scalar base; the frame checks are in D35.
- Superseded in review round 5, kept as history: the zero test was not public (review round 4, user's decision; intent I0, I4). It answers "simplification proved this scalar identically zero", so `False` means "not proved", not "nonzero" (the user's `monster` entry minus 1 is zero and is not proved), the same flaw as equality (D36); the user never needed it. Only the library asks it: D20 (a zero norm raises) and the exact cases of axis-angle (D13). It is a module-level private function of the boundary, `_is_zero(value: Scalar) -> bool` in `symbolic.py`, imported by `geometry.py` and `rotation.py`. It expands the simplified form before comparing with 0 (code review of round 4): `fu` leaves polynomial identities unexpanded, so `(q + 1)**2 - q**2 - 2*q - 1` was not proved zero and its normalization divided by a zero norm (a 0/0 direction instead of D20's error); `expand` proves it in about 1 ms. The expansion stays inside the zero test, because `simplify` must keep the factored form (D5, the wrist). Probed with ruff 0.17.2 and the project configuration: importing a private function from a sibling module passes, while calling a private method (`value._is_zero()`) from another module is `private-member-access`, so a private method would need a configuration exception. Rejected: keeping `Scalar.is_zero()` public (a two-valued answer to a three-valued question, in the contract); a private method (the exception above); a three-valued public test (no use). `_has_symbols(value: Scalar) -> bool` (D13's test "contains a symbol") follows the same route. These two are a declared exception to the style rule "free functions only when really needed" (user, 2026-10-27, "lascialo così"): the rule, ruff's `private-member-access` and the zero test staying private cannot all hold, and two private functions of the boundary are the narrowest exception. Every other helper lives on its type: the exactness check and the simplification are private methods of `Scalar`; the frame checks are private methods of `_Resolved` and `Rotation`, and `Transform` relies on `Rotation`'s (review round 4, after the user pointed out a free function). Rejected: private methods with a per-file `private-member-access` exception for `geometry.py` and `rotation.py` (it would stop the check for every private access in those files); public methods (reopens the zero test's removal).
- Reverse operators take `int` (review round 4: floats are rejected, D4; before, `float`, which also admitted `int` under ruff PYI041). `ScalarLike` is `Scalar | int`; operations that can return an expression also take `Expression` (D38). `Vector.__rmul__` takes `Scalar | Expression | int`, so a symbolic factor written on the left, `l1 * x_axis` as the exam writes it, is a vector (review round 3: it was a ty error and a runtime crash).
- `Symbol.simplify()` returns the symbol itself, typed `Self`: a symbol is already simplest, and `Expression.simplify -> Expression` (`Scalar.simplify` before D38) would otherwise give `Symbol` a `Scalar` result against `Symbolic.simplify -> Self` (review round 3).
- Compile-time failures are domain exceptions: `ParameterTypeError(TypeError)` for a field not annotated `float`, `MissingParameterError(ValueError)` for a free symbol without a field, each naming the class and the fields or symbols. `ParameterShapeError(ValueError)` is raised at trace time for a field holding an array outside `vmap` (D11).

Rejected: a public `Scalar(expression: sympy.Expr)` constructor (a sympy type in a public signature); `Symbol` without its own constructor (the `Symbol(3)` hole above); a runtime check that raises for values outside `ScalarLike`, such as `np.float32` (review round 2: ty already rejects them; reversed in review round 4 by D35, since intent I3 asks every static promise to hold at runtime too). Revised in review round 3: every operator returns `NotImplemented` for an operand outside `ScalarLike`, as Python's operator protocol requires; this is not a duplicate of the type check, it is what lets `l1 * x_axis` reach `Vector.__rmul__` (before, `Scalar.__mul__` wrapped `None` and crashed).

### D32. Verification mechanisms

- The negative fixture is `tests/type_mistakes.py.txt`, copied to a temporary `.py` and checked by ty from the project root. Lines end in `# mistake:` (must be an error) or `# known miss:` (must stay silent); every other line must be clean. A `.py` fixture in the tree would need exclusions in ty, the edit-time hook and pre-commit, three configurations for one file.
- Docs pages are doctests (`>>>` inside fenced blocks), run by `tests/test_docs.py` with `doctest.testfile`: "runs as written" also checks the printed output, with no pytest configuration.

Rejected: executing extracted code blocks with `exec` (ruff S102, and outputs unchecked).

### D33. Rotation and transform construction

- `Rotation(axis, angle, *, source)` is the public constructor: the rotation about a unit axis by Rodrigues' formula, which `about_axis` calls. Like D20, the public constructor itself keeps the invariant, so no public path builds a `Rotation` from nine arbitrary entries. Elementary rotations, the identity, products, inverses and simplification build the internal storage (D21) through one private factory of the class (`object.__new__`, attributes set on the fresh instance before it is returned); no method repeats its body.
- `Rotation.target` and `Rotation.source` expose the two frames, named after the D16 keywords; `Transform` needs them for its LaTeX.
- `Transform` stores its rotation and translation only (D21); `entry(row, column)` reads the 4x4 matrix from them, with t = translation − `Target.origin()`. Composition is (R₁R₂, t₁ + R₁t₂), the action on a point is R p + t (as `translation + R @ (p - B.origin())`), the inverse is (Rᵀ, −Rᵀt).
- `Point` is `@final`. Points and vectors share a private storage base, so ty considered a class that is both; `final` rules it out and narrows `Point | Vector` in `match`.
- `AxisAngle[F]` is a frozen dataclass with `axis` and `angle`; it is not `Symbolic` (the stub's choice): its axis and angle compile on their own (D13).
- The skew-part axis of the regular case is `skew.normalize()`, so the normalization of D20 gives `sin(q)/Abs(sin(q))` for `Rz(q)` directly; the second solution is `(-first.axis, -first.angle)`, as the course writes it, with `Direction.__neg__` returning a `Direction` (negation keeps unit length, so nothing is normalized twice).
- The half-turn axis is the normalized column k of R + I whose diagonal entry r_kk + 1 is not close to zero (`is_close`, D38). Column k is 2 r r_k and r_kk + 1 = 2 r_k², so the column vanishes exactly where r_kk + 1 = 0. The half turn and the identity are reached only by constant rotations (D13), where the choice is made on values.
- `Rotation` and `Transform` are `@final`, like `Point`. ty 0.0.91 does not accept the private factory's `Rotation[A, B]` as `Self`, even on a final class; so `simplify` and `homogeneous` are annotated with the concrete type (`Rotation[Target, Source]`, `Transform[Target, Source]`), which final makes equal to `Self`. Probed: the `Self` return with the factory is an `invalid-return-type` error; the concrete return on the final class is clean.
- Method-level frame parameters are `A`, `B` (`about_z[A: Frame, B: Frame]`), the letters of ^A R_B; the class parameters stay `Target`, `Source`. Calls write `target=` before `source=`, the order of the type parameters.
- Closed in review round 6 (open point of review round 2): the regular case computes the norm ‖s‖ of the skew part twice, once for sin θ = ½‖s‖ and once inside `normalize`, because D20 lets no direction be built from a supplied length. Since nothing simplifies implicitly (D5), each computation is a square root of a sum of squares, built in microseconds, so none of the ways considered then is needed: a `Direction` built from a known length (reopens the hole D20 closes); `Direction` exposing its length (a public accessor on a type whose length is one); memoizing simplification in the boundary (hidden mutable state).

The column search of the half turn always finds a column: r_kk + 1 = 2 r_k², and a unit axis has some r_k² ≥ 1/3, so r_kk + 1 ≥ 2/3 there, far from the tolerance the search skips. Rejected (review round 2): a default for the search or a separate error (it cannot be reached by a `Rotation`, whose public constructors guarantee a rotation).

Rejected (review round 1): the half turn as valid everywhere (`1 != 0`, the earlier text): with the axis (a, 1, 0) normalized and the angle π, column 1 is chosen because r₁₁ + 1 = 2a²/(a² + 1) is not identically zero, and compiled at a = 0 the axis is NaN while `valid()` is true. Superseded in review round 4: a rotation with a symbol gets the regular formulas (D13), and the condition is gone (D28).

Rejected: a public constructor from entries (a matrix that is not a rotation would type as one); a public `_Matrix3`-taking constructor (the same hole behind a private type); rotations storing their frames only as phantom types (D6).

### D34. One public spelling per concept

Each concept has exactly one spelling in docs, tests, notebooks and library code outside its own module:
- The constructor, when the arguments are the concept's parts: `Scalar(2)` (a constant from its value). A transform was the other case until review round 4; it is now built by `R.homogeneous(translation)` (D17).
- A factory or method, when it adds meaning the arguments do not carry: `Base.position(x, y, z)` (numbers read in a frame), `Rotation.about_axis(axis, q, source=...)` (names the construction), `v.normalize()` (an operation on `v`). `Symbol("q")` is a constructor whose argument is the variable's name (review round 5; `Scalar.symbol("q")` before, D31).

Public constructors that exist only because Python has no private ones (`Position(...)`, `Displacement(...)`, `Direction(d)`, `Rotation(axis, angle, source=...)`) are called only by the spelling they serve; elsewhere the class name is a type, used in annotations and `match`. The classes stay exported for that reason.

Rejected: a factory for everything (`with_origin` duplicated the constructor); a builder pattern for every type (D17 replaced the only builder, `Transform`'s, in review round 4); leaving both spellings in use (a learner sees the same math written two ways and must check that they are equivalent).

### D35. Every static promise also holds at runtime

Serves intent I0 and I3. Review round 4, against `openspec/intent.md`: at the end of review round 3 no operation checked anything at runtime. Probed: `A.vector(1,0,0) + B.vector(1,0,0)` gave `Vector[A](2, 0, 0)`; `A.vector(1,0,0) * A.vector(1,0,0)` gave `Vector[A](None, None, None)`; `Scalar("x")` gave `None`; `r_ab @ r_ab` gave `Rotation[A, B]`. The user's case (`stratchpad/t.py`): with `v1: Vector[B2] = Vector(0, 0, 1, B1)`, ty reports the assignment, and at runtime `v1 + v2` and `v2 + v1` returned vectors instead of raising.

Every mistake the type checker rejects raises when executed, at the operation that commits it, with a message naming what is wrong:
- Operands in different frames (positions, displacements, rotations, transforms, the translation given to `homogeneous`): `FrameMismatchError(TypeError)`, naming the operation and both frames, e.g. "cannot add Displacement[A] and Displacement[B]: frame B where A is expected". It subclasses `TypeError` because it is the runtime image of a type error. Every operand is named by its kind, `Displacement[Tool]` or `Rotation[Base, Tool]`, never by its components (review round 7, task 15.4; finding #8): a method names its own receiver with its private `_kind()`, and an operand of another class with `type(x).__name__` and its public frame, since calling another class's `_kind()` is ruff's `private-member-access`. The check reads the frames stored in the values (D6), so it holds whatever the annotations say.
- Operands outside an operation's accepted types (`q + "a"`, `q + 0.3`, `vector * vector`, `2 * point`, `rotation @ point`, `Scalar("x")`, `compiled(0.5)`): `TypeError`, through Python's operator protocol (`NotImplemented`) for operators and an explicit check for constructors and calls.
- An entry index outside its range: `IndexError` naming the range ("rotation entries are counted from 1 to 3"; D26 still makes it a type error first).
- A kind mistake (D38): a constant-only method called on an expression (`q.is_close(0)`) raises `AttributeError`, since expressions do not have it; an expression passed where a constant is required (`a.is_close(q)`) and `compile()` without parameters on a symbolic quantity raise `TypeError`. The kind is read from the components, so it holds whatever the annotations say.

Each method that compares frames raises `FrameMismatchError` directly, `if ...: raise FrameMismatchError(...)`, so its docstring lists the exception, as ruff's docstring rules require of a raise in the body (user, 2026-10-27). The shared frame-check helpers of `geometry.py` and `rotation.py` are gone, and each class writes its own description once, in `_kind()`, which its layout label and printed form also use (code review of round 4, #9: the check was duplicated, and the description of a `Rotation` built in three places). The message names the operation and the operands the user wrote: `Transform @ Position` names the transform and the position, not the rotation and displacement it is computed through (#3). Outside dunder methods ruff's `private-member-access` flags `other._kind()`, so `dot`, `cross` and the free-axis check name the other operand by its printed form.

Limit, declared: an annotation is not executed, so `v: Displacement[Scalar, Base] = approach` with `approach` in `Tool` raises nothing at the assignment; the mistake surfaces at the first operation with a `Base` value, where the stored frame differs. Nothing in Python can intercept the assignment itself. User (2026-10-26): "Questo è uno di quei casi in cui Python ci ricorda che i tipi sono solo dei suggerimenti e sta a noi renderli validi. In questo caso abbiamo fatto il massimo e il runtime è l'unica linea di difesa che ci potrebbe salvare."

Cost (intent I6): the checks run on the symbolic side, when an expression is built, never inside a compiled function; `Compiled.__call__` checks its argument's type once per call in plain Python and once per trace under the caller's `jit`.

Verification (as built): the negative fixture (D32) marks each line `# mistake:` (ty error, raises), `# static only:` (ty error, an annotation Python does not check), `# runtime only:` (ty silent, raises: the user's `v1: Displacement[Scalar, B2] = Displacement(0, 0, 1, B1)` followed by `v1 + v2`), or `# known miss:` (ty silent today). ty runs once on the whole fixture; Python runs one module per raising line, made of every silent line in order plus the line under test, loaded with `importlib` (no `exec`), and a further module of the silent lines alone must run. With the frame check disabled, the 14 frame-mismatch lines fail; restored, all pass. Runtime mistakes are tested only there: ty rejects them in ordinary tests too (`operator.add(q, 0.1)` is a ty error), and an inline suppression is banned.

Rejected: runtime checks only where the static check was measured to cost too much (the earlier Goals line; it lets an ignored type error produce a wrong result much later, against intent I3); `assert` statements (removed under `python -O`, and the project bans them, `narrow-types-protocols-no-asserts`); a debug flag enabling the checks (a mistake must fail the same way whether or not a flag is set).

### D36. Symbolic and compiled values have no equality

Serves intent I0 and I4; review round 4, user's decision. At the end of review round 3 no class defined `__eq__`, so every value inherited Python's identity comparison: `A.point(1, 0, 0) == A.point(1, 0, 0)`, `Rotation.about_z(q, ...) == Rotation.about_z(q, ...)` and `Scalar.symbol("q") == Scalar.symbol("q")` were all `False`, and `AxisAngle`, a dataclass, compared its fields the same way. Nobody decided it; Python provides it, so a wrong answer could pass unnoticed. User: "il fatto che python lo implementa significa che posso avere delle risposte sbagliate senza accorgermene."

`==`, `!=`, `hash` and `bool` raise `TypeError` on every `Symbolic` and on `Numeric`. The message says what to do instead: for constants, `is_close` (D38); for expressions, evaluate them or compile them and compare at a configuration; for compiled values, compare `.components` with a tolerance (`jnp.allclose`). The symbolic side cannot know whether two values are equal on the whole domain; a configuration can. User: "ti posso dire solo se puntualmente è vero, non posso dirti se è vero per ogni elemento del dominio, che è una cosa onesta, ma troppo onerosa per ==." Frames are classes and keep Python's class comparison, which is the right one for a type.

Built as a private mixin of the boundary, `_Incomparable`, inherited by `Symbolic`, `Numeric` and `AxisAngle` (a frozen dataclass with `eq=False`; its generated `__eq__` would compare fields, and tuple comparison short-circuits on identity, so `a == a` would answer `True`). No equality is implemented: none has a use yet, and one implemented badly would give answers that look right. When a use appears, adding a comparison is an addition to the contract (intent I4).

Rejected: `==` on constants with a tolerance (the tolerance must be visible where it is used: `is_close`, D38); a three-valued comparison of symbolic values (proved equal, proved different, unknown; proposed in the code review of round 4 and superseded by intent I8: on expressions the question has no meaning); keeping Python's identity comparison (a silent wrong answer, against intent I0); equality of the written form (`False` may mean "written differently", not "different"); mathematical equality through simplification (sympy can fail to prove a true identity, as on the user's `monster`, and can hang; no `hash` consistent with it); an exact canonical form for polynomials in sine and cosine (probed: exact and fast on the robotics cases, including the `monster` and a 6R chain, but outside its class, e.g. q and q/2, it needs further rules; cost without a use); probabilistic equality at random points (not a proof).

### D37. Positions and displacements

Serves intent I2; review round 4, user's decision. The geometry layer's two kinds of three-component values are named for what they are:

| Concept | Class | Created by | Uses of its frame | Under a change of frame | Homogeneous form |
|---|---|---|---|---|---|
| where something is, measured from the frame's origin | `Position[F]` (was `Point`) | `Base.position(x, y, z)`, `Base.origin()` | axes and origin | R p + t | [p; 1] |
| how much and which way to move, from nowhere in particular | `Displacement[F]` (was `Vector`) | `Base.displacement(x, y, z)`, `Position − Position` | axes only | R v | [v; 0] |

The affine rules of D7 read as sentences: `Position − Position → Displacement`, `Position ± Displacement → Position`, `Displacement ± Displacement → Displacement`, `k · Displacement → Displacement`; `Position + Position` and `k · Position` are type errors. A displacement's frame says which axes its components are read on, not where it is anchored; it has no anchor.

User (2026-10-27), after the discussion of position vectors and free vectors: "quindi sono una posizione e uno spostamento?"; on the names: "l'alternativa mi piace molto di più perché sono più chiari". The earlier names misled: both a point and a vector are "vectors" in the course's sense (position vector, free vector), and `Vector` did not say which.

Unit length (user's decision): `UnitVector` becomes `Direction[F]`, a displacement of length one; `FreeAxis` (the identity's arbitrary axis, D13) stays, a subtype of `Direction`. Two spellings, each with its own use (D34): `Base.direction(x, y, z)` for a direction written from components, and `d.normalize()` for one computed from a displacement (the skew part in axis-angle). `Base.direction(x, y, z)` normalizes its components, as D20 requires of every unit value: `Base.direction(1, 1, 0)` is (√2/2, √2/2, 0), and `Base.direction(0, 0, 0)` raises (D14); a symbolic direction never raises (D20, D38). `Direction` and its constructor keep D20's guarantee: no direction holds components not divided by their own norm. Normalizing instead of checking is Claude's choice, accepted by the user (2026-10-27): proving symbolically that given components have length one can fail when they do (the simplifier's limits, D5), while dividing by the norm is always defined; rejected: a unit-length check that raises (it could reject a true unit direction).

Alias, a declared exception to D34: `Point = Position`, a plain assignment, so it is the same class everywhere (annotations, `isinstance`, `match`, construction); printed forms and runtime errors name `Position`. User: "per quanto ci riguarda un punto è un vettore posizione"; it is kept while it helps and removed if superfluous, decided before the change is archived (afterwards removal would break code that uses it). Probed: a PEP 695 `type Point = Position` is not a class at runtime (`isinstance`, construction and `match` raise `TypeError`); a `NewType` makes a distinct type (a `Position` is not accepted where `Point` is expected) and cannot be specialized by frame (`Point[Base]` is an error in ty).

Earlier decisions keep the names of their round (`Point`, `Vector`); this decision maps them. The transform's translation (D17) is a `Position[Target]`.

Rejected: `PositionVector` / `FreeVector` (the course's terms, chosen first; the user preferred the plainer words); `Displacement` as the base class of `Position` (a position used where a displacement is expected would make `p + p` and `2 * p` legal, against D7); creating displacements only as `Position − Position` (a displacement is not anchored, so writing it from its components is meaningful: "3 east, 4 north" anywhere); an alias through `type` or `NewType` (probed above).

### D38. Constants and expressions

Serves intent I8, and I0 and I2; review round 5, user's decision (2026-10-30). Zero and equality have meaning only on fixed values (I8). Every attempt to answer them on symbolic values was a proof procedure that misses true zeros: `simplify`, `expand`, a canonical form for polynomials in sine and cosine, all probed and fuzzed in the code review of round 4, which found zeros each one misses (half angles, constants in integer radians, nested trigonometric functions); and simplification can hang. The type now says whether the question has meaning:

| Class | Is | Example |
|---|---|---|
| `Scalar` | a constant, built from integers and exact constants only | `Scalar(2)`, `Scalar.pi() / 3`, `Scalar(2).sqrt()` |
| `Expression` | a value built from at least one symbol | `q + 1`, `2 * q.sin()`, `q - q` |
| `Symbol(Expression)` | one variable | `Symbol("q")` |

- Promotion is syntactic, statically and at runtime: an operation returns a `Scalar` only when every operand is a `Scalar` or an `int`, and an `Expression` otherwise, whatever symbols remain after sympy's own evaluation. `q - q` is an `Expression` printing `0`. Choosing the runtime class from the free symbols left would make the runtime contradict the type (D35).
- `Symbol` is a subclass of `Expression`: a variable is usable wherever an expression is, while an operation that needs a variable (a derivative, in the Jacobian change) can require `Symbol`. `Scalar` and `Expression` are unrelated, so neither passes where the other is required; their shared operations live in a private base of the boundary.
- Composite quantities take the kind as their first parameter, `K: (Scalar, Expression)` (a constrained type parameter): `Position[K, F]`, `Displacement[K, F]`, `Direction[K, F]`, `FreeAxis[K, F]`, `Rotation[K, Target, Source]`, `Transform[K, Target, Source]`, `AxisAngle[K, F]`, `AxisAngleSolutions[K, F]`. A composite is `Scalar` only when built from constants only: `Position(a, a, a, B1)` is `Position[Scalar, B1]`; `Position(a, a, b, B1)`, `Position(a, a, a + b, B1)` and `Position(a, a, b1 + b2, B1)` are `Position[Expression, B1]`. No composite is `[Symbol, ...]`: no operation needs a position made of bare variables. Constructors and binary operations use two overloads: all constants give `Scalar`, any other combination `Expression`.
- At runtime the kind is read from the components, as the frame is (D6): a symbolic composite stores its constant components as `Expression`, so `Position(a, a, b, B1).x` is an `Expression` printing `2`, at runtime as in the type. User: "se almeno un elemento è un simbolo, allora tutto il vettore è simbolico".
- Constants compare with a tolerance: `Scalar.is_close(other: ScalarLike, tolerance: float = 1e-9) -> bool` is true when |a − b| ≤ tolerance, with a − b evaluated to 30 significant digits so that the tolerance is the only margin. Absolute, because the library's values are of order one; the caller passes another. There is no separate zero test: `a.is_close(0)`. User's tolerance check (review round 7): constants within 1e-9 of zero count as zero, so 1e-10 rad is the identity, π − 1e-10 a half turn, and a 1e-12 displacement has no direction; kept, because such values are zero numerically and passing them is a misuse of the API.
- A constant always has a real value (review round 7, user's decision, 2026-10-30; intent I0): an operation or evaluation whose constant result is not a real number (`nan` from 0/0 or atan2(0, 0), an infinity from 1/0 or tan(π/2), the imaginary unit from √−1) raises `UndefinedValueError(ArithmeticError)` at once, naming the operation or the values substituted. `Rz(q).to_axis_angle().evaluate({"q": 0})` raises instead of returning a `nan` axis, and `is_close` and D20's zero check never see `nan` (until round 7 they did, and `is_close` answered false to avoid sympy's "Invalid NaN comparison"). The check has two tiers (review round 8): one pass over the constant's sympy form rejects `nan`, `zoo`, `oo` and `-oo`; only a value holding `I` or a fractional power of a negative number atom, such as (−8)^(1/3) = 2·(−1)^(1/3) whose principal value is complex, is expanded into real and imaginary parts, and rejected if `I` remains, so ((√−1 + 1)(√−1 − 1)) = −2 passes. sympy's semantic query (`is_extended_real`, `is_finite`) on every operation was probed at about 80 times the cost (40 ms against 0.5 ms on the entries of an 8-rotation constant chain) and still answers "unknown" on `nan` and on (−1 + i)(1 + i). Known miss: a fractional power of a negative constant that is not a number atom, such as (1 − √3)^(1/3); asking sympy the sign of every base costs seconds on long constant chains (the user's `monster` test went from 0.7 s to 6 s), and no quantity of the library raises a composite negative constant to a fractional power. `UndefinedValueError` says in words why the value is not real (undefined, infinite, complex) beside sympy's printed value, which alone shows tokens such as `zoo` (review round 8). Since ruff's `docstring-extraneous-exception` lists only exceptions a method raises itself, `Expression.evaluate` raises it directly and lists it; the operators, the functions and the composites' `evaluate`, which raise it through their components, state it in their class docstrings instead. Expressions are not checked: sympy may print `nan` inside one (D13), and evaluating it raises. Complex values are rejected, not designed out: no robotics quantity of the library needs them, and allowing them later is an addition (intent I4). Rejected: returning `nan` (a wrong number reaching the user silently); leaving it with the singular configurations (those are open in compiled code; on constants the values are fixed, so the question has meaning, I8); a complete check by numeric evaluation (about 0.1 s per entry of an 8-rotation constant chain, paid on every operation).
- The library's own decisions on values use `is_close` at the default tolerance, on constants only: normalization (D20) and the cases of axis-angle (D13). On expressions the library decides nothing.
- `evaluate(values)` on every `Expression` and every symbolic composite substitutes exact values and returns the constant: `Expression -> Scalar`, `Position[Expression, F] -> Position[Scalar, F]`, and so on. `values` maps symbol names to `ScalarLike` and may be omitted; a float raises (D4); a symbol without a value raises `MissingParameterError` naming it; values for symbols that do not occur are allowed, so one dictionary of joint values serves every quantity, as extra fields do in `compile` (D11). User: `Position(a, a, b, B1).x.evaluate()` is 2 and `.z.evaluate({"b": 1})` is 1. It is the only way from an expression to a constant, and its check is at runtime: the type does not record which symbols a value contains (Non-Goals), so a static check is impossible.
- Constants compile with no parameters: `compile()` on a constant quantity returns its `Numeric` value directly, since nothing varies. On a symbolic quantity, `compile()` without parameters is a ty error and raises. The compiled value of a constant is the nearest double (√2 is 1.4142135623730951); the exact value stays symbolic.
- `==`, `hash` and `bool` stay banned on every value, constants included (D36).

Probed in ty 0.0.91 (scratch prototype): with overloads, `a + a` infers the constant type and `a + q`, `q + a` the expression type; a constant-only method called on an expression is an error ("Expected `Scalar[Exact]`, found `Scalar[Symbolic]`" in the tag version, the same with classes); `evaluate` returns the constant type; `Position(a, a, b).x` infers `Expression`; a function taking `variable: Symbol` rejects an `Expression` ("Expected `Symbol`, found `Expression`").

Decided during apply (task 13.1-13.6, Claude's; probed in ty 0.0.91):
- `Expression(value)` is a public constructor, the counterpart of `Scalar(value)`: it takes a constant or an expression as an expression. A symbolic composite needs it to hold its constant components as expressions, and `rotation.py` needs it to promote a constant translation; a private conversion would be `private-member-access` from those modules. An addition to the contract (intent I4), reported to the user.
- Composites are built through `__new__` overloads and a private generic factory per class (`object.__new__`, fields set on the fresh instance, as D33's rotation factory), because ty does not let an `__init__` with an annotated `self` use the class's type parameters, and does not narrow a constrained `K` when a runtime branch picks the kind. Two private classes of the boundary, `_Constants` and `_Expressions`, carry the components of one kind each; a `match` on them picks the factory's specialization.
- ty preserves `K` through operators (`x + y`, `x / 2`, `-x` on values typed `K`) but not through method calls, which it answers per constraint as `Scalar | Expression`. So `simplify` returns `Self` on constants and expressions (`Symbol` keeps returning itself), the norm is written `squares ** (Scalar(1) / 2)` (sympy's `sqrt` is that power), and methods that branch on the kind annotate `self` as the union of the two specializations and narrow with `TypeIs` predicates. Probed: a method-level type parameter bound to `self` gives `Unknown` on a `K` receiver and is ruff's `custom-type-var-for-self`.
- A free axis passed to `to_axis_angle` must be a constant direction: it only appears in the identity of a constant rotation.
- Copying (review round 7, task 15.2; finding #6): the constructors of round 5 are `__new__` with required arguments, which `copy` and `pickle` call with none. `Symbolic.__reduce__` rebuilds every quantity from its stored fields through `object.__new__`, as the private factories do, so no constructor runs again (a direction is not normalized twice). `copy` and `deepcopy` are tested on every composite; the `pickle` round trip was probed (a rotation and a direction come back with the same printed form) and is not tested, because ruff bans importing `pickle` (`suspicious-pickle-import`) and pickling is not part of the contract.
- `Displacement`, `Direction` and `FreeAxis` each have a private factory `_of` (review round 8, task 16.3), so `evaluate` keeps the class of a direction or a free axis without normalizing again; `evaluate` calls `self._of`, which dispatches. One factory built from `cls` (finding #4) was probed and rejected by ty 0.0.91: `object.__new__(cls)` is `Self`, which ty does not accept as the instance of another kind (`Displacement[J, G]`), the limit D33 records for `Rotation`.
- Components already of the composite's kind are stored as they are, not wrapped again (review round 7, task 15.5; finding #9): sharing them is safe because values are immutable. Operations keep going through the kind check of `_Constants.or_expressions`, one `isinstance` per component, so the kind of a result is decided in one place; bypassing it per operation would repeat that decision in every operator for no measurable gain. Measured: 2000 pairs of `R @ v` and `v + v` on symbolic values, 75 ms before, 69 ms after; the rest is sympy's arithmetic.
- `UndefinedValueError` (review round 7, task 15.1) names the value obtained (`nan is not a real number`) and, from `evaluate`, the values substituted (`… at q = 0`); for an operation the traceback points at the line. The check sits in the factory every constant result goes through, so the operation itself is not named: naming it would thread a description through each of the twelve call sites for a line the traceback already shows. The spec scenario evaluates `q.sin() / q`, not `q.sin() / q.sin()`, which sympy reduces to 1 before any value is substituted.

Consequence (user-approved, 2026-10-30): a symbolic quantity that equals a constant stays symbolic. `R @ R.inverse()` with `R = Rx(q)` is `Rotation[Expression, ...]`; its axis-angle uses the regular formulas, whose axis is undefined where the skew part vanishes (D13). The exact case is reached by evaluating first.

Rejected: a kind tag on the scalar itself (`Scalar[Exact]`, `Scalar[Symbolic]`; the classes already say it, and the user asked why a wrapper was needed); the names `Exact` or `Constant` for the kind (exactness is true of both sides since D4; with the component classes as the parameter, `Position[Scalar, B1]` reads as the user wrote it); `Symbol` for every expression (a derivative could not require a variable); the runtime class chosen from the free symbols left after sympy's evaluation (contradicts the type for `q - q`); deciding zero on expressions by any proof procedure (above); numeric comparison of expressions at random points (not a proof, and not a question with meaning on undetermined values); `is_close` on composites (no use yet; an addition, intent I4).

## Risks / Trade-offs

- [Explicit widening is not caught statically: `w: Displacement[Scalar, Frame] = Tool.displacement(1, 0, 0)` inline, and user generic helpers `[G: Frame]` called with inline mixed arguments, pass ty] → The stored frame makes the first operation mixing frames raise (D35, intent I3: runtime is the only line where types cannot reach); the negative fixture records them as known misses so a ty upgrade that starts catching them is visible.
- [ty 0.0.91 is pre-1.0; variance inference and union construction may change] → Exact pin; the negative type test (D10) gates every upgrade.
- [Runtime validation in `compile` (D11) happens later than a type error] → It runs once per compilation, at definition time in a notebook, not inside simulation loops.
- [Singular configurations: an axis compiled alone (D13), a symbolic direction whose norm vanishes at some or all configurations (D20, D38), which prints `nan` symbolically where sympy evaluates the norm to zero itself] → Not an accepted risk: open on purpose (intent.md, "Open on purpose"; proposal, Out of Scope). The contract promises no value there in compiled code; today they compile to `nan` or ±inf, and no doc or test states it. With exact values, evaluating there raises (D38, review round 7). The validity flag that marked some of these cases was removed (D28), whose probe table is the evidence for the open item. Review round 4 re-labelled the earlier "accepted" entry, which contradicted intent I0.
- [Rodrigues' (1 − cos θ) loses precision near θ = 0] → Oracle tests sample near singularities.

## Migration Plan

1. Tag `41d2833` (e.g. `legacy`) so the old code stays reachable.
2. Delete the legacy library, its tests and the five midterm notebooks; drop `loguru`, `pandas`, `pandas-stubs`, `matplotlib`.
3. Build bottom-up: scalar, geometry, rotation, transform, compilation; each layer with its tests and its negative fixture lines.
4. Rewrite the in-scope cells of `exams/2024_09_19.ipynb` and delete the rest (D15).

Rollback: check out the `legacy` tag.
