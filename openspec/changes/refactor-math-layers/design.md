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
- Static checks first; runtime checks only where the static version was measured to cost too much.

**Non-Goals:**
- A numeric matrix algebra. Numbers exist only as the output of a compiled function.
- Static tracking of which symbols an expression contains.
- Catching wider types the user writes explicitly (`v: Vector[Frame] = ...`, user generic helpers); see Risks.
- Euler angles and RPY (D24).
- A general matrix type (D21).

## Decisions

Names are settled in D18.

### D1. sympy is the only algebra; numbers only through compilation

One symbolic representation for every quantity. Numeric work is a function compiled from a symbolic quantity (D11); there is no numeric matrix class and no `evaluate`.

Rejected:
- Two worlds, `Symbolic` and `Numeric` (names since reused with another meaning, see D18), with an `S` parameter on every class (`Matrix[S]`, `Vector[F, S]`, `Rotation[To, From, S]`) and world-gated methods (`def evaluate(self: Matrix[Symbolic]) -> Matrix[Numeric]`, probed working in ty 0.0.91): it worked, but solved a problem that does not exist. Simulation needs a compiled function, not a numeric algebra.
- Separate `SymbolicMatrix` / `NumericMatrix` / `SymbolicPoint` classes: two APIs for one concept.
- Backend wrapper classes holding whole arrays: once compilation exists they are a burden on both sides.
- Partial evaluation, then full-only `evaluate`, then an exact partial-binding operation: each added a second way to get numbers and confused what a matrix is.
- Static symbol tracking (`Expression[V]` with `Literal` unions, `Symbol[N].replace_in`, `numeric()` on `Expression[Never]`): possible in ty, rejected for cost.
- Set-based symbol removal `evaluate(set[str])`: ty infers `Unknown`; `Symbolic("q1", ...)` widens to `str` and falsely reaches `Numeric`.
- A schema class (`Joints(Variables)` + descriptor + `dataclass_transform` + field specifier, matrices naming their schema): worked in ty, rejected for time and effort.
- CasADi, SymEngine, SymForce, Rust, Julia (chat 0): not reconsidered.

### D2. Complete typed boundary around sympy (the "unsafe" rule)

The scalar class `Scalar` wraps a `sympy.Expr`. No public signature accepts or returns a sympy type. `isinstance` checks and `Unknown` values exist only inside the boundary modules, the way Rust confines `unsafe`. Probed: `q + 1`, `1 + q`, `q * q`, `q.cos()`, `Scalar.pi() / 2` all infer `Scalar`; `Vector(q + 1, Scalar.pi() / 2, 0.1)` gives `[q + 1, pi/2, 0.100000000000000]`.

Arithmetic accepts `ScalarLike = Scalar | int | float` on both sides.

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
| `PLR0904` (`too-many-public-methods`) ignored | `src/robotic/symbolic.py` only | The rule counts dunders: `Scalar` has 24 public methods, all fixed by the spec (11 arithmetic operators for both sides, D2; 6 functions, D5; `pi`, `symbol`, `simplify`, `latex`, `is_zero`, `__init__`, `__repr__`). Rejected: raising `max-public-methods` globally (loosens the check everywhere); splitting `Scalar` (contradicts D5). |
| `S404`, `S603` ignored | `tests/test_negative_typing.py` only | The negative type test (D10, D32) runs ty in a subprocess with fixed arguments and no shell, the same exception `.claude/hooks/` has. |

The configuration names rules by name (`banned-api`, `private-member-access`), as ruff's preview rule `rule-codes-in-selectors` requires; codes are used in this document for brevity.

Scope (user's decision): the ban holds in `src/`, where it is fully verifiable and where the non-boundary layers (geometry, rotation, transform) would otherwise drift into raw sympy unnoticed. It is lifted in tests and notebooks, a limit of the tool accepted as such: `tests/**` already ignores `TID251` for the oracle libraries and the ignore is per rule, not per banned module; notebooks are excluded from ruff (`extend-exclude = ["*.ipynb"]`). No configuration is added for either.

Rejected: lifting the ban everywhere because it cannot reach tests and notebooks (gives up the part that is verifiable); a documented "never import sympy" rule (a rule only docs protect); an ast-grep rule (ruff already has the mechanism).

### D4. Symbols, constants and floats

- Only the library creates symbols (`Scalar.symbol(name) -> Symbol`, `class Symbol(Scalar)`), with fixed assumptions. Probed: `Symbol("q2") + Symbol("q2", real=True)` is `q2 + q2` with two distinct free symbols; one creation path closes it.
- Exact constants are methods: `Scalar.pi()`, `.sqrt()`, `.exp()`. A module-level `pi` is shadowed silently by `math.pi`.
- Floats are accepted inside expressions as written. Shown before deciding: float `Rz(q) Rz(pi/2)[0,0]` simplifies to `-1.0*sin(q) + 6.12323399573677e-17*cos(q)` against the exact `-sin(q)`; `solve(cos q = 0.5)` gives `[1.04719755119660, 5.23598775598299]` against `[pi/3, 5*pi/3]`. The leak is intentional: writing `0.3` means not caring about exactness.

Consequence kept (review round 2): a rotation built from floats does not reach the exact identity or half-turn branch of axis-angle. `Rz(math.pi)` has a skew part of norm 2.4e-16, so it takes the regular formulas, its second angle is −π, and its compiled flag is false. The tolerance flag (D28) reports exactly this near-singular case; the exact branches are for exact angles (`Scalar.pi()`).

Rejected: rejecting floats at every entry point; raw float typed as `Numeric` with a documented trap table (only docs protected it); explicit exact fractions required (`Symbolic(3) / 10`); converting floats with `nsimplify` in the printer.

### D5. sympy functions as methods

`.simplify()`, `.latex()`, `.lambdify(...)`, trigonometric functions and `atan2` are methods on the value. Rejected: free functions (they need docs to be found; style rule: free functions only when really needed); a static-only utils class.

### D6. Frames are types, stored at runtime, invariant

`Vector[F: Frame]`, `Point[F: Frame]`, `UnitVector[F]`, `FreeAxis[F]` (subtype of `UnitVector[F]`). The value stores its frame at runtime. Probed: with the stored frame and methods taking `Vector[F]`, ty infers `F` invariant; a phantom `F` was covariant, which let `Vector[Tool]` widen to `Vector[Frame]` and mix with `Vector[Base]`.

A frame creates its own points and vectors (`Tool.point(...)`, `Tool.vector(...)`, `Tool.origin()`): a frame gives a point origin and axes, a vector axes only.

Confirmed by the user against the principle "data is bare components, meaning is in the type": the components stay bare data (a sympy matrix, a JAX array); the frame is a label beside them, like `^{Tool}` beside the numbers of a handwritten solution. Three things read it at runtime: LaTeX (D19), `lambdify`, which copies it into the compiled result's aux data (D12), and the type checker's inference. Re-probed in ty 0.0.91, bound values: a stored frame makes `Stored[Tool]` not assignable to `Stored[Frame]`; a phantom `F` widens silently even with `__add__(self, other: Phantom[F])`, and a constructor without a frame argument gives `Phantom[Unknown]`.

Rejected: opt-in frames; a `NoFrame` default; free points and vectors; `Point3(Tool, ...)` constructors; Protocol frames; an explicit old-style invariant `TypeVar` (not needed, and D10 forbids it).

### D7. Affine arithmetic tracks the homogeneous w

P − P → V, P ± V → P, V ± V → V, k·V → V; P + P and k·P are type errors. Reason: with [p; 1] and [v; 0], 1 − 1 = 0, 1 ± 0 = 1, 0 ± 0 = 0, and 1 + 1 = 2 is not a point. Coordinates as a vector are explicit: `p - Tool.origin()`.

Rejected: allowing `2 * p` (it came from a misread of "2 * vector").

### D8. Rotations and transforms carry both frames

`Rotation[Target, Source]` is ^A R_B; `Transform[Target, Source]` is ^A T_B (names and order in D16). Composition `@` chains inner frames; inverse swaps them. `Vector[F]` is an arrow resolved in F: changing the frame of resolution is symbolic only (rotation or transform applied to a vector or point); numbers come from compiling the resolution already chosen. `T @ point` is R p + t; `T @ vector` is R v.

Rejected: frames on vectors only; numeric frame changes on compiled results.

### D9. Receiver-first rule

No function or constructor in the API solves a frame from two arguments. A frame is fixed by the receiver (or an explicit frame argument) and every other argument is checked against it.

Probed in ty 0.0.91 (with the stub's `to=` / `from_=` spelling, replaced by D16): invariance stops widening existing values, but inline values are built with an invented union. `Transform(Rotation.about_z(q, to=Base, from_=Link1), Link1.origin())` silently gives `Transform[Base | Link1, Link1]`, and applying it to a `Link1` point gives `Point[Base | Link1]`. The same happens with `Self` classmethods on frames (`Base.rotation_about_z(...)`) and with an inline invariant key `Key(Base)`. Receiver-first `Rotation.about_z(q, to=Base, from_=Link1).with_origin(Link1.origin())` reports "Expected `Point[Base]`, found `Point[Link1]`"; inline `@` with mismatched frames reports an error.

Consequence: the stub's `Transform(rotation, origin)` constructor is replaced by a transform built from its frames, whose parts are checked against them (D17).

Rejected: fixing inline widening through variance alone (`Transform` is already invariant); pre-bound frame keys (`BASE = Key(Base)`) as the user's responsibility (only docs protect it).

### D10. PEP 695 generics, gated by a negative type test

PEP 695 syntax only (ruff UP040, UP046, UP047 enforce it). Because ty is pre-1.0 and inference decides variance, a negative fixture of mistakes (one per line) is run by pytest through ty, which asserts every line is an error. The test gates every ty upgrade.

The stub's fixture caught 8 of 10 (`tip + tip`, `2 * tip`, `Tool` + `Base` vectors, transform applied to a wrong-frame point, `Base←Link1 @ Base←Link1`, `q1 + "a"`, `compiled(0.5)`, `Parameters()` missing `q1`); the two misses are the inline cases closed by D9 and the explicit widening listed in Risks.

Rejected: old-style `TypeVar` with explicit variance; characterization tests or `xfail(strict)` (chat 0).

### D11. Compilation: `lambdify` against a parameters dataclass

`lambdify[P: DataclassInstance](self, parameters: type[P]) -> Compiled[P, Self]`, `Compiled.__call__(values: P) -> Numeric[Self]`. `P` is a user-written frozen dataclass (one for the whole robot, e.g. `Joints`). Fields are matched to symbols by name; ordering is internal; extra fields are allowed. Type hints survive, so `compiled(0.5)` and `Parameters()` with a missing field are type errors.

Validation is at runtime, once, at compile time: every field annotated `float` or `jax.typing.ArrayLike` (string annotations resolved with `typing.get_type_hints`; `dataclasses.fields` alone returns the string `'float'`), every free symbol has a field, otherwise raise. `float` is for single-configuration code (the exam notebook); `ArrayLike` is for code that batches: JAX transformations keep the wrapped function's signature, so `jax.vmap(f)(batch)` with `def f(q: float)` is a ty error, while a field and a parameter annotated `ArrayLike` accept floats and arrays and still reject `Joint(q1="a")` and `compiled(0.5)` (probed in ty 0.0.91). The quantity and the dataclass are linked only by names; a static link needs the schema rejected in D1.

`jax.vmap` is the only way to batch (D34). `ArrayLike` is the annotation that lets `vmap` type-check, not a promise that the compiled function takes arrays: every field value must be a scalar of shape `()`, which is what `vmap` hands in. A field holding an array raises `ParameterShapeError(ValueError)`, naming the field and pointing to `jax.vmap`. The check sits inside the traced kernel, where shapes are static, so it runs when JAX traces (once per input shape), not on each call. Found in review round 1: an array passed directly reached `jnp.stack`, which failed with "Cannot stack arrays with different numbers of dimensions" for any quantity with constant entries (every rotation and transform), because the constant entries stay of shape `()`.

Rejected (review round 1): broadcasting the entries so a direct array call works (a second batching path beside `vmap`, and it needs rules for fields of different lengths, (N,) with (M,)); leaving the direct call unhandled (an internal stacking error for a user mistake).

JAX is the only backend: `lambdify(..., modules="jax")` under `jax.jit`, with `jax_enable_x64`. Autodiff is part of the numeric path (whitelisted).

Because validation reads the annotations at runtime, the annotation's name must exist at runtime. ruff's `typing-only-third-party-import` (TC002) treats a bare `ArrayLike` in a dataclass field as typing-only and asks to move its import under `TYPE_CHECKING`, which would make `get_type_hints` fail; the tests and docs write `jax.typing.ArrayLike` through the runtime-used `jax` binding instead. (ruff's `runtime-evaluated-decorators = ["dataclasses.dataclass"]` would state this in configuration; not added, the qualified spelling needs no exception.)

Rejected: `float` fields only (batched code is a ty error); `ArrayLike` fields only (the exam code would write `q1: ArrayLike` for one configuration); caller-ordered symbols `lambdify((q1, q2))`; sorted-name order (`"q10" < "q2"`); creation-order counters; runtime-generated parameter classes (`type(...)`, `make_dataclass`, metaclass: ty sees `Unknown`/`Any`); a `backend` parameter, `Backend(StrEnum)` and per-backend overloads (until a second backend exists); `Piecewise` branching inside compiled code.

### D12. Thin frame-carrying compiled result

A compiled point, vector, rotation or transform returns a thin result `Numeric[T]` in the natural shape (`(3,)`, `(3, 3)`, `(4, 4)`, not sympy's `(3, 1)`), registered as a JAX pytree: frame as static aux data, components as the only child. It has no frame operations; simulation code reads `.components`. `vmap` leading axes are configurations and are not modeled in the type.

Principle (the user's): data is bare components; meaning (the frame) is in the type, on both sides of the boundary. `(0, 0, 1)` only makes sense while you know where it was described.

Measured: a plain class returned from `jax.jit` raises `TypeError`; registered, it passes `jit`, `vmap` (frame kept, components `(2, 3)`) and `grad` (0.8775825618903728). Cost under `jit`: single call 8.3 µs raw versus 9.6 µs wrapped; `vmap` over 1000 configurations 18.2 µs versus 19.1 µs.

Rejected: a bare `jax.Array` (frame lost).

`jax_enable_x64` is set when `robotic` is imported, for the whole process (user's decision, review round 2). The library checks exam values against symbolic results, where float32 rounding would show as false mismatches. The side effect is stated in `docs/compilation.md` and `README.md`. Rejected: enabling x64 only inside `Compiled.__call__` (not probed; expected to fail under the caller's own `jit` / `vmap`, whose input dtypes are fixed when the outer trace runs, outside the scope); requiring the caller to enable it (a forgotten flag silently gives float32).

The printed form of a `Numeric` lists concrete components and shows a tracer as JAX prints it, so printing works under tracing (review round 2: `.tolist()` raised `ConcretizationTypeError` inside `jax.jit`).

`Numeric._unflatten` builds `Numeric[Symbolic](...)` through the subscripted alias (review round 3, Claude's decision). Probed in ty 0.0.91: a plain `Numeric(...)` infers `Numeric[Unknown]` and is an `unsound-return-statement` (and, bound to an annotated local, an `unsound-assignment`); annotating the return `Numeric[Any]` is banned by ruff's `any-type`. The alias dispatch costs microseconds and runs when JAX unflattens at a trace boundary, not on each compiled call.

Derivatives are taken of `.components`, not of the `Numeric` (review round 3, Claude's decision; the user delegated JAX-side decisions): `jax.jacrev(lambda q: compiled(P(q)).components)`. `jax.jacrev` and `jax.jacfwd` of the whole result fail on its boolean `valid` child (D28). Rejected: dropping the flag child from results without a condition (the runtime branch D28 avoids, and axis-angle results would still fail); making `Numeric` differentiable as a whole (the derivative would come back labelled `Numeric[Rotation[A, B]]` while holding dR/dq, which is not a rotation: the type would claim a meaning the numbers lack, against this decision's principle).

### D13. Axis-angle is symbolic

`R.to_axis_angle()` returns an `AxisAngleSolutions` with `.first` and `.second` (each an `AxisAngle`: `.axis` and `.angle`) and `.valid_when` (a `ValidityCondition`), without evaluating anything; an axis compiles like any vector. Content (course convention): angle in (−π, π]; regular (r, θ), (−r, −θ); half turn (r, π), (−r, π); identity (a, 0), (−a, 0) with `FreeAxis`, default z. Default `tolerance=1e-9`. The axis of ^A R_B satisfies R r = r, so it has the same components in A and B.

Probed (the singular case, see D28): for `Rz(q)` sympy gives axis `[0, 0, sin(q)/Abs(sin(q))]` and angle `atan2(Abs(sin(q)), cos(q))`; compiled at q = 0.5 it gives `[0 0 1]`, at q = 0 `[0 0 nan]`, at q = π `[0 0 1]` only by luck (float sin(π) = 1.2e-16). `Rx(π/2) Rz(q)` at q = π gives `[nan -inf inf]`.

Rejected: evaluating first, then extracting (`R.evaluate({...}).to_axis_angle()`); raising for the identity; `NoRotation` / `HalfTurn` / `Undecided` union arms; an "all cases" object with mandatory `.evaluate()`; `FreeAxis` as a non-subtype.

### D14. Error principle

Raise when no answer exists (normalizing a zero vector, a free symbol without a field at compile time). Return a typed result when a singular case still carries information (axis-angle of the identity).

### D15. The exam notebook is the benchmark

`exams/2024_09_19.ipynb` keeps only the cells the new API can express, rewritten against it; the cells that need `Manipulator`, DH or the Jacobian are deleted. The notebook is the change's benchmark, and later changes add cells back as the API grows. The deleted cells stay reachable at the `legacy` tag.

Rejected: keeping the out-of-scope cells non-runnable until their change lands (a notebook that does not run top to bottom is no benchmark).

### D16. Frame keywords `target=` / `source=`

^A R_B maps coordinates expressed in the source B to the target A. That mapping reading names the frame pair at every layer: type parameters `[Target, Source]`, attributes `_target` / `_source`, properties `.target` / `.source`, keywords `target=` / `source=`. Constructors that need frames their arguments cannot fix take them as keywords: `Rotation.about_z(q1, source=Link1, target=Base)` is `Rotation[Base, Link1]`; `Rotation.identity(source=Tool, target=Wrist)`; `Rotation.about_axis(axis, q, source=Link1)`, where the axis's frame fixes the target. Each frame is fixed by its own keyword, so D9 holds. Target comes first so the types read like the composition ^A R_B ^B R_C = ^A R_C: `Rotation[A, B] @ Rotation[B, C] -> Rotation[A, C]`. Points and vectors keep the "expressed in" reading (`Vector[F]`, `.frame`).

Naming rule (user's): no name clashes with the language (keywords, soft keywords, builtins), and escaping a keyword with a trailing underscore counts as a clash. Builtin shadowing is already a ruff error (`A001`–`A005` under `ALL`); the escape is not caught by any rule today, so the separate ast-grep style-rule change bans identifiers ending in a single `_`.

Probed in ty 0.0.91 (with the earlier `of=` / `relative_to=` spelling; keywords bind the same way), bound and inline: `assert_type(Rotation.about_z(0.5, of=Link1, relative_to=Base), Rotation[Base, Link1])` passes, and a receiver-first method taking a point of the target frame reports an inline `Link1` point ("Expected `Point[Base]`, found `Point[Link1]`").

Rejected:
- `of=` / `relative_to=` (the course's orientation sentence, "the orientation of B relative to A"; mixed with mapping-named type parameters `To` / `From`, it gave one frame pair five vocabularies: A/B, `To`/`From`, `relative_to`/`of`, `_target`/`_source`, `T`/`S`).
- `relative_to` / `of` at every layer (loses the mapping reading, which is how the user reads a rotation).
- `[Source, Target]` order (reads source to target, but composition types read backwards against ^A R_B: `Rotation[B, A] @ Rotation[C, B] -> Rotation[C, A]`).
- `to=` / `from_=` (the stub's spelling; `from_` escapes the keyword `from`).
- A builder from the frames, `Link1.relative_to(Base).about_z(q1)` (adds an intermediate class whose only job is to carry two frames).

### D17. A transform is built from its frames

`Transform(*, target: type[Target], source: type[Source])` is the identity transform between the two frames. `.with_rotation(rotation: Rotation[Target, Source])` and `.with_translation(translation: Vector[Target])` each return a new transform with that part replaced; the frames never change:

```python
Transform(target=Base, source=Link1)
    .with_rotation(Rotation.about_z(q1, source=Link1, target=Base))
    .with_translation(Base.vector(0, 0, d1))
```

The frames are written although the parts repeat them. User's reason: it states what the rotation and the translation are about, and the frames cannot change after construction. The receiver fixes both frames, so D9 holds inline. The translation is ^A p, the origin of B expressed in A: a vector, so `T @ point` is point + vector (D7).

Python has no private constructors, so the public `__init__` must itself satisfy D9; the keywords fix one frame each.

Probed in ty 0.0.91 with `Point`, `Vector` and `Rotation` invariant:
- `__init__(self, rotation: Rotation[To, From], origin: Point[To])` and the self-typed variant: inline silent (`Transform[Base | Link1, Link1]`), bound caught.
- `__init__(self, rotation: Rotation[Target, Source], translation: Vector[Target])` (re-probed in review round 1): inline infers `Transform[Base | Link1, Link1]`; the mistake surfaces only at an annotation, explicit `Transform[Base, Link1](...)`, or the first operation with a `Base` value.
- Receiver-first methods: an inline wrong-frame argument is an error (D16).

Rejected:
- `Transform(rotation, x, y, z)` (the earlier decision: safe, but a transform is built from parts, not coordinates).
- `Transform(rotation, translation)` (no redundancy, but the inline hole above; a known miss in place of a type error).
- `rotation.with_translation(v)` / `rotation.with_origin(p)` (reads as applying a translation to a rotation).
- `vector.homogeneous().with_rotation(R)` (the intermediate value has no source frame, so it needs its own class; the homogeneous form of a vector is [v; 0], not a transform).
- Copying the rotation constructors onto the builder (`Transform(...).about_z(q)`): two spellings of every rotation (D34).
- A mutating builder (`self._rotation = ...; return self`): two names for one transform would change together.

### D18. Names

Rule (user's, D16): no clash with the language; a keyword escape counts as a clash. Every candidate was swept against keywords, soft keywords, builtins, every importable stdlib module, and the top level of sympy, jax, jax.numpy and numpy.

| Name | Meaning | Rejected |
|---|---|---|
| `Scalar` | the rank-0 symbolic value: what `norm` and `dot` return and what scales a vector | `Expression` (`ast.Expression`; "symbolic" is true of every value since D1), `Real` (`numbers.Real`, and `float` is registered as one), `Value` (`multiprocessing.Value`), `Term` (one part of a sum) |
| `ScalarLike` | the alias `Scalar \| int \| float` accepted by every scalar operation; the `-Like` convention of `numpy.typing.ArrayLike` and `jax.typing.ArrayLike` | `Number` (`numbers.Number`, `tokenize.Number`, `sympy.Number`; says "plain number" but accepts symbolic scalars), `Operand` (says nothing), `Numeral` (a digit string) |
| `Symbolic` | the shared base of `Scalar`, `Point`, `Vector`, `Rotation`, `Transform`: `simplify`, `latex`, `lambdify` | `Quantity` (`sympy.physics.units.Quantity`, pint: implies units), `Expression` (`ast`; a rotation is not "an expression" in course language), `Compilable` (covers only `lambdify`) |
| `Numeric[T]` | the thin compiled result of a `T` (D12); pairs with `Symbolic` | `Evaluated` (names the process), `Numerical` (same meaning, longer), `Sample` (random or time-series sampling) |
| `Compiled[P, T: Symbolic]` | the compiled function; `__call__(values: P) -> Numeric[T]` | `Compiled[P, R]` with the full result type (`R` was always `Numeric[...]`, and an unbounded `R` let `Compiled[Parameters, int]` pass; probed: the bound rejects it) |
| `AxisAngle[F]` | one axis-angle representation: `.axis`, `.angle` | `AxisRotation` (the stub's name for it: reads as a rotation object, confusable with `Rotation.about_axis`); in the stub `AxisAngle` named the outer result, which is not an axis-angle |
| `AxisAngleSolutions[F]` | the result of `to_axis_angle()`: `.first`, `.second`, `.valid_when`; the course's "two solutions"; a `Symbolic`, so it compiles as a whole (D28) | `AxisAnglePair` (says how many, not why; invites tuple unpacking) |
| `ValidityCondition` | a symbolic statement over the variables, true where a result is valid ("valid when sin q ≠ 0"); renders LaTeX; a `Symbolic` that compiles to a flag (D28) | `Condition` (`threading`, `asyncio`, `multiprocessing`), `Predicate` (`sympy.Predicate`), `Constraint` (something imposed, as in optimization), `Guard` (programming jargon) |
| `Transform.with_rotation`, `Transform.with_translation` | the builder steps of D17; each pairs with its property (`T.with_rotation(R).rotation` is `R`) and returns the same type, as `Path.with_suffix` does | `Rotation.with_origin(origin: Point[To])` (the earlier name: reads as giving a rotation an origin; replaced with D17), `placed_at` (vague about which part is set) |
| `target`, `source` | the frames of a rotation or transform, in the mapping reading (D16) | `relative_to` / `of`, `To` / `From` (D16) |
| `Frame.notation` | how the course writes the frame, e.g. `0` or `3'` (D19) | see D19 |

`Symbolic` here is not the rejected two-worlds scalar of D1: there is one symbolic base and one compiled result type, no `S` parameter and no gated methods.

### D19. A frame's notation is a class keyword

`class Base(Frame, notation="0")` renders `{}^{0}`; without the keyword the class name is used (`class Tool(Frame)` renders `{}^{Tool}`). Implemented with `__init_subclass__(cls, *, notation: str | None = None)` storing `Frame.notation: ClassVar[str]`. Reason: the exam numbers its frames (^0 R_1, ^3 T_6) and a class cannot be named `0`.

Probed in ty 0.0.91 and at runtime (with the earlier keyword `latex`): a misspelled keyword is a ty `unknown-argument` error and a runtime `TypeError`; a non-string value is a ty error (expected `str | None`).

Rejected: the keyword `latex` (the earlier name: a string on frames, while `latex()` is the render method on every quantity, so `rotation.target.latex()` failed and `point.latex` returned a bound method); `label` and `name` (both already used in `symbolic.py`; `name` also reads as the Python name); a class attribute `notation = "0"` (a misspelled attribute passes ty silently and the class name is printed); the keyword `symbol` (collides in meaning with `Scalar.symbol`, which creates a variable); always the class name (cannot express numbered frames).

### D20. Unit vectors come only from normalization

`UnitVector(direction: Vector[F])` normalizes its argument: it divides the components by the simplified norm and raises when the norm simplifies to zero (D14). `v.normalize()` is the spelling (D34); it calls `UnitVector(v)`. As with D30, Python has no private constructors, so the public constructor itself guarantees the invariant: no `UnitVector` can hold components not divided by their own norm, and no length check exists.

Probed with sympy 1.14, norm simplified first: (cos q, sin q, 0) stays (cos(q), sin(q), 0); (1, 1, 0) gives (sqrt(2)/2, sqrt(2)/2, 0); (3, 4, 0) gives (3/5, 4/5, 0); (q, 0, 0) gives (q/Abs(q), 0, 0). Without simplifying the norm the first prints `cos(q)/sqrt(sin(q)**2 + cos(q)**2)`.

A norm that vanishes only at some configurations (q/Abs(q) at q = 0) does not raise symbolically and compiles to `nan` there; that is hole 2 (Q2), not a unit-vector question.

Rejected: a constructor taking components with a unit-length check (proving x² + y² + z² = 1 symbolically can fail and would raise for a vector that is unit, against D14); a constructor taking components without a check (the type would promise what nothing guarantees).

### D21. No public matrix type

Rotations store their entries in an internal typed 3x3 class built from `Scalar` arithmetic only, so it needs no sympy and lives in the rotation module, outside the boundary (D29). A transform stores no matrix: its 4x4 entries are read from its rotation and origin. No public signature returns the storage. `Rotation` and `Transform` are the only public matrix-like concepts (D1: one class per concept) and expose their entries as `Scalar`s through a typed accessor (D26) and as LaTeX.

Rejected: a public frameless `Matrix` (a free value that does everything `Rotation[To, From]` does while bypassing the frame types, the hole D6 closed for points and vectors; nothing in this change needs one, and the Jacobian change decides about a general matrix when it needs one).

### D22. A rotation acts on a vector with `@`

`R @ v` is the only spelling: ^A v = ^A R_B ^B v, consistent with `Rotation @ Rotation` and `Transform @` points and vectors (D8). Probed in ty 0.0.91: `base_link1 @ link1_tool @ approach` and `base_link1 @ (link1_tool @ approach)` both infer `Vector[Base]`; `(base_link1 @ link1_tool).inverse() @ Base.vector(1, 0, 0)` infers `Vector[Tool]`; a wrong-frame vector at the end of a chain is an error.

Rejected: `.rotate(v)` (the active reading, the arrow moves, contradicts D8's passive one, the arrow stays and its frame of resolution changes); a second, named spelling (if one is ever added, its name must describe the passive meaning).

### D23. A rotation does not act on points

`Rotation @ Point` is a type error: a rotation relates axes and says nothing about origins, so with offset origins the result would have the right type and the wrong value (identity rotation, origin of B at (5, 0, 0) in A: the tip (1, 0, 0) in B is (6, 0, 0) in A, not (1, 0, 0)). The two honest readings stay explicit: `Transform(target=Base, source=Tool).with_rotation(R) @ p` writes the coincident-origin assumption down; `R @ (p - B.origin())` resolves the grounded vector (D7). Probed in ty 0.0.91: both infer `Point[Base]` and `Vector[Base]`; `R @ tip` is an error.

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

`Numeric[T]` exposes `.components`, and one self-gated accessor (D28): `valid[F](self: Numeric[AxisAngleSolutions[F]]) -> jax.Array`. The meaning is the type `T`; the runtime frame, kept in the pytree aux data (D12), appears in the printed form (`Numeric[Point[Base]]([1.0, 0.5, 0.0])`). Nothing in this change reads a compiled frame: simulation reads `.components`, the benchmark compares numbers.

Additive path for a later change (probed in ty 0.0.91, one class, no new types): self-annotated methods offered only for the matching `T`, `frame[F](self: Numeric[Point[F]]) -> type[F]` and `target[A, B](self: Numeric[Rotation[A, B]]) -> type[A]`; `scalar.frame()` and `rotation.frame()` are errors.

Rejected: a plain `.frame` property (no single return type fits a point, a rotation and a scalar); per-concept result classes (reopens D18's single `Numeric[T]`); adding the frame accessors now (no reader in this change). The validity flag does have a reader (D28), so it takes the gated path; folding it into the components as a fifth float column was rejected (a boolean posing as a float, read by index).

### D28. Singular configurations surface as a compiled flag

`ValidityCondition` is a `Symbolic` and compiles to a boolean of shape `()`, applying the tolerance of `to_axis_angle` (`|expression| > tolerance`, default `1e-9`) instead of `≠ 0`. `AxisAngleSolutions` is a `Symbolic` too; compiling it as a whole returns both solutions and the flag in one result, so on the main path the flag cannot be left out: `.components` has shape `(2, 4)`, rows the first and second solution, columns r_x, r_y, r_z, θ; `.valid()` is the boolean flag of shape `()` (D27), a second pytree child. Every compiled result carries a flag child (true for quantities without a condition), so the accessor needs no runtime branch; only `Numeric[AxisAngleSolutions[F]]` exposes it. When the flag is false the caller decides; nothing is substituted.

Probed, `Rz(q)` regular-case formulas compiled with JAX under `jit` (axis z component, angle, exact flag `sin θ ≠ 0`, tolerance flag `|sin θ| > 1e-9`):

| q | axis z | angle | exact | tolerance |
|---|---|---|---|---|
| 0.0 | nan | 0.000e+00 | False | False |
| 1e-12 | 1.0 | 1.000e-12 | True | False |
| 1e-06 | 1.0 | 1.000e-06 | True | True |
| 0.5 | 1.0 | 5.000e-01 | True | True |
| π | 1.0 | 3.142e+00 | True | False |

The exact flag misses the near-identity case (meaningless axis) and the half turn (right only because float sin(π) = 1.2e-16); the tolerance flag catches both.

Rejected: an exact `≠ 0` flag (the two misses above); `Piecewise` branching in compiled code; forbidding an axis from compiling alone (reverses D13; see Risks).

### D29. One boundary module

`src/robotic/symbolic.py` holds `Symbolic`, `Scalar`, `Symbol`, `ScalarLike`, `Compiled` and `Numeric`; it is the only module importing sympy and calling `sympy.lambdify`. `geometry.py` (frames, points, vectors), `rotation.py` (rotations, their storage, axis-angle) and `transform.py` build on the public `Scalar` API only. `transform.py` imports `rotation.py` (a transform starts from `Rotation.identity`, D17); `rotation.py` does not import `transform.py`.

Rejected: a separate boundary module per concern (scalar, matrix, compilation): every pair would need private access across modules; public accessors for raw entries (a "give me your sympy" method on every quantity) or a `Compiled` constructor taking pre-built parts (an explicit specialization could then attach any frame type to any numbers, against D20's principle that the public constructor guarantees the invariant).

### D30. Points and vectors have a public frame-taking constructor

`Point(x, y, z, frame)` and `Vector(x, y, z, frame)` exist because Python has no private constructors (as in D17 and D20); `Tool.point(...)` and `Tool.vector(...)` call them and stay the spelling (D34). The frame comes last: the numbers, then the frame they are read in. It has one source, the `frame` argument, so D9 holds: `Point(1, 0, 0, Tool)` infers `Point[Tool]`.

The `Point3(Tool, ...)` rejection (D6) stands as an API choice: docs and examples use the frame methods only.

Rejected: building points through hidden attribute assignment after `object.__new__` (hidden mutation); a constructor taking an opaque components object (still public, adds a type with no meaning).

### D31. Scalar construction, zero test and compile errors

- `Scalar(value: ScalarLike)` builds a constant: `Scalar(2).sqrt()` is the spec's "square root of the scalar 2". Results of sympy operations are wrapped by a private factory inside the boundary; the public constructor never takes a sympy type (D2).
- `Symbol(name)` is the constructor `Scalar.symbol(name)` calls, with the same fixed assumption (real). It exists because `Symbol` inherits `Scalar(value)`, which would otherwise let `Symbol(3)` wrap a constant typed as a symbol; one assumption set keeps D4's guarantee.
- `Scalar.is_zero() -> bool`: true only when simplification proves the scalar identically zero. The layers outside the boundary need it for D20 (zero norm raises) and for the singular cases of axis-angle (D13); it is the only three-valued sympy question exposed, collapsed to "proved zero".
- Reverse operators take `float` (ruff PYI041: `int | float` is redundant); `int` is still accepted by the type checker and stays exact at runtime. `ScalarLike` keeps `Scalar | int | float` as the documented alias. `Vector.__rmul__` takes `Scalar | float`, so a symbolic factor written on the left, `l1 * x_axis` as the exam writes it, is a vector (review round 3: it was a ty error and a runtime crash).
- `Symbol.simplify()` returns the symbol itself, typed `Self`: a symbol is already simplest, and `Scalar.simplify -> Scalar` would otherwise give `Symbol` a `Scalar` result against `Symbolic.simplify -> Self` (review round 3).
- Compile-time failures are domain exceptions: `ParameterTypeError(TypeError)` for a field not annotated `float`, `MissingParameterError(ValueError)` for a free symbol without a field, each naming the class and the fields or symbols. `ParameterShapeError(ValueError)` is raised at trace time for a field holding an array outside `vmap` (D11).

Rejected: a public `Scalar(expression: sympy.Expr)` constructor (a sympy type in a public signature); `Symbol` without its own constructor (the `Symbol(3)` hole above); a runtime check that raises for values outside `ScalarLike`, such as `np.float32` (review round 2: ty already rejects them). Revised in review round 3: every operator returns `NotImplemented` for an operand outside `ScalarLike`, as Python's operator protocol requires; this is not a duplicate of the type check, it is what lets `l1 * x_axis` reach `Vector.__rmul__` (before, `Scalar.__mul__` wrapped `None` and crashed).

### D32. Verification mechanisms

- The negative fixture is `tests/type_mistakes.py.txt`, copied to a temporary `.py` and checked by ty from the project root. Lines end in `# mistake:` (must be an error) or `# known miss:` (must stay silent); every other line must be clean. A `.py` fixture in the tree would need exclusions in ty, the edit-time hook and pre-commit, three configurations for one file.
- Docs pages are doctests (`>>>` inside fenced blocks), run by `tests/test_docs.py` with `doctest.testfile`: "runs as written" also checks the printed output, with no pytest configuration.

Rejected: executing extracted code blocks with `exec` (ruff S102, and outputs unchecked).

### D33. Rotation and transform construction

- `Rotation(axis, angle, *, source)` is the public constructor: the rotation about a unit axis by Rodrigues' formula, which `about_axis` calls. Like D20, the public constructor itself keeps the invariant, so no public path builds a `Rotation` from nine arbitrary entries. Elementary rotations, the identity, products, inverses and simplification build the internal storage (D21) through one private factory of the class (`object.__new__`, attributes set on the fresh instance before it is returned); no method repeats its body.
- `Rotation.target` and `Rotation.source` expose the two frames, named after the D16 keywords; `Transform` needs them for its LaTeX.
- `Transform` stores its rotation and translation only (D21); `entry(row, column)` reads the 4x4 matrix from them. Composition is (R₁R₂, t₁ + R₁t₂), the action on a point is R p + t (as `A.origin() + t + R @ (p - B.origin())`), the inverse is (Rᵀ, −Rᵀt).
- `Point` is `@final`. Points and vectors share a private storage base, so ty considered a class that is both; `final` rules it out and narrows `Point | Vector` in `match`.
- `AxisAngle[F]` is a frozen dataclass with `axis` and `angle`; it is not `Symbolic` (the stub's choice): its axis and angle compile on their own (D13).
- The skew-part axis of the regular case is `skew.normalize()`, so the normalization of D20 gives `sin(q)/Abs(sin(q))` for `Rz(q)` directly; the second solution is `(-first.axis, -first.angle)`, as the course writes it, with `UnitVector.__neg__` returning a `UnitVector` (negation keeps unit length, so nothing is normalized twice).
- The half-turn axis is the normalized column k of R + I whose diagonal entry does not simplify to zero. Column k is 2 r r_k and r_kk + 1 = 2 r_k², so the column vanishes exactly where r_kk + 1 = 0: the half turn's `ValidityCondition` is `r_kk + 1 != 0` for the chosen k. Only the identity is valid everywhere (`1 != 0`). A skew part that simplifies to zero with a trace proving neither case raises (D14: no answer can be given).
- `AxisAngleSolutions` compiles its condition through the condition's own layout; the layout of a `ValidityCondition` is defined once. `Layout.condition` holds the condition itself, and `Compiled` (inside the boundary) reads its layout: calling `self._valid_when._layout()` from `AxisAngleSolutions` is private access across classes, which ruff's `private-member-access` flags outside `symbolic.py`.
- `Rotation` and `Transform` are `@final`, like `Point`. ty 0.0.91 does not accept the private factory's `Rotation[A, B]` as `Self`, even on a final class; so `simplify`, `with_rotation` and `with_translation` are annotated with the concrete type (`Rotation[Target, Source]`, `Transform[Target, Source]`), which final makes equal to `Self`. Probed: the `Self` return with the factory is an `invalid-return-type` error; the concrete return on the final class is clean.
- Method-level frame parameters are `A`, `B` (`about_z[A: Frame, B: Frame]`), the letters of ^A R_B; the class parameters stay `Target`, `Source`. Calls write `target=` before `source=`, the order of the type parameters.
- Open point, kept for a later revisit (user's decision, review round 2): the regular case simplifies the norm ‖s‖ of the skew part twice, once for sin θ = ½‖s‖ and once inside `normalize`, which divides s by its own simplified norm. Normalization happens once; the norm is computed twice because D20 lets no unit vector be built from a supplied length. The cost is one extra `sympy.simplify` of a square root per extraction. Ways to revisit, not chosen: a `UnitVector` built from a known length (reopens the hole D20 closes); `UnitVector` exposing its length (a public accessor on a type whose length is one); memoizing simplification in the boundary (hidden mutable state).

The column search of the half turn always finds a column: r_kk + 1 = 2 r_k², and a unit axis has some r_k ≠ 0, while the search skips only entries proved zero. Rejected (review round 2): a default for the search or a separate error (it cannot be reached by a `Rotation`, whose public constructors guarantee a rotation).

Rejected (review round 1): the half turn as valid everywhere (`1 != 0`, the earlier text): with the axis (a, 1, 0) normalized and the angle π, column 1 is chosen because r₁₁ + 1 = 2a²/(a² + 1) is not identically zero, and compiled at a = 0 the axis is NaN while `valid()` is true.

Rejected: a public constructor from entries (a matrix that is not a rotation would type as one); a public `_Matrix3`-taking constructor (the same hole behind a private type); rotations storing their frames only as phantom types (D6).

### D34. One public spelling per concept

Each concept has exactly one spelling in docs, tests, notebooks and library code outside its own module:
- The constructor, when the arguments are the concept's parts or frames: `Transform(target=Base, source=Link1)` (D17).
- A factory or method, when it adds meaning the arguments do not carry: `Base.point(x, y, z)` (numbers read in a frame), `Rotation.about_axis(axis, q, source=...)` (names the construction), `v.normalize()` (an operation on `v`), `Scalar.symbol("q")`.

Public constructors that exist only because Python has no private ones (`Point(...)`, `Vector(...)`, `UnitVector(v)`, `Symbol(name)`, `Rotation(axis, angle, source=...)`) are called only by the spelling they serve; elsewhere the class name is a type, used in annotations and `match`. The classes stay exported for that reason.

Rejected: a factory for everything (a transform's parts are already its arguments; `with_origin` duplicated the constructor); a builder pattern for every type (only `Transform` is assembled from framed parts, D17); leaving both spellings in use (a learner sees the same math written two ways and must check that they are equivalent).

## Risks / Trade-offs

- [Explicit widening is not caught: `w: Vector[Frame] = Tool.vector(1, 0, 0)` inline, and user generic helpers `[G: Frame]` called with inline mixed arguments, pass ty] → Only the user can write these; the negative fixture records them as known misses so a ty upgrade that starts catching them is visible.
- [ty 0.0.91 is pre-1.0; variance inference and union construction may change] → Exact pin; the negative type test (D10) gates every upgrade.
- [Runtime validation in `lambdify` (D11) happens later than a type error] → It runs once per compilation, at definition time in a notebook, not inside simulation loops.
- [NaN or ±inf at singular configurations after compiling] → `AxisAngleSolutions` compiles with its validity flag (D28).
- [An axis compiled alone (D13) gives `nan` at q = 0 and an unmarked, meaningless axis near the identity; only the documentation points to the flagged path] → Accepted by the user: forbidding axes from compiling alone would reverse D13. The docs of task 6.5 show the flagged path first.
- [Rodrigues' (1 − cos θ) loses precision near θ = 0] → Oracle tests sample near singularities.
- [Float leaks make simplified expressions noisy (`6.12323399573677e-17*cos(q)`)] → Accepted (D4); exact constants are one method call away.

## Migration Plan

1. Tag `41d2833` (e.g. `legacy`) so the old code stays reachable.
2. Delete the legacy library, its tests and the five midterm notebooks; drop `loguru`, `pandas`, `pandas-stubs`, `matplotlib`.
3. Build bottom-up: scalar, geometry, rotation, transform, compilation; each layer with its tests and its negative fixture lines.
4. Rewrite the in-scope cells of `exams/2024_09_19.ipynb` and delete the rest (D15).

Rollback: check out the `legacy` tag.
