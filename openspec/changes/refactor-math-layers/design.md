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
| `TID251` ignored | the boundary modules only (exact paths fixed in task 2.1) | These modules are the boundary of D2: the only code allowed to touch untyped sympy values. |

Scope (user's decision): the ban holds in `src/`, where it is fully verifiable and where the non-boundary layers (geometry, rotation, transform) would otherwise drift into raw sympy unnoticed. It is lifted in tests and notebooks, a limit of the tool accepted as such: `tests/**` already ignores `TID251` for the oracle libraries and the ignore is per rule, not per banned module; notebooks are excluded from ruff (`extend-exclude = ["*.ipynb"]`). No configuration is added for either.

Rejected: lifting the ban everywhere because it cannot reach tests and notebooks (gives up the part that is verifiable); a documented "never import sympy" rule (a rule only docs protect); an ast-grep rule (ruff already has the mechanism).

### D4. Symbols, constants and floats

- Only the library creates symbols (`Scalar.symbol(name) -> Symbol`, `class Symbol(Scalar)`), with fixed assumptions. Probed: `Symbol("q2") + Symbol("q2", real=True)` is `q2 + q2` with two distinct free symbols; one creation path closes it.
- Exact constants are methods: `Scalar.pi()`, `.sqrt()`, `.exp()`. A module-level `pi` is shadowed silently by `math.pi`.
- Floats are accepted inside expressions as written. Shown before deciding: float `Rz(q) Rz(pi/2)[0,0]` simplifies to `-1.0*sin(q) + 6.12323399573677e-17*cos(q)` against the exact `-sin(q)`; `solve(cos q = 0.5)` gives `[1.04719755119660, 5.23598775598299]` against `[pi/3, 5*pi/3]`. The leak is intentional: writing `0.3` means not caring about exactness.

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

`Rotation[To, From]` is ^A R_B; `Transform[To, From]` is ^A T_B. Composition `@` chains inner frames; inverse swaps them. `Vector[F]` is an arrow resolved in F: changing the frame of resolution is symbolic only (rotation or transform applied to a vector or point); numbers come from compiling the resolution already chosen. `T @ point` is R p + t; `T @ vector` is R v.

Rejected: frames on vectors only; numeric frame changes on compiled results.

### D9. Receiver-first rule

No function or constructor in the API solves a frame from two arguments. A frame is fixed by the receiver (or an explicit frame argument) and every other argument is checked against it.

Probed in ty 0.0.91 (with the stub's `to=` / `from_=` spelling, replaced by D16): invariance stops widening existing values, but inline values are built with an invented union. `Transform(Rotation.about_z(q, to=Base, from_=Link1), Link1.origin())` silently gives `Transform[Base | Link1, Link1]`, and applying it to a `Link1` point gives `Point[Base | Link1]`. The same happens with `Self` classmethods on frames (`Base.rotation_about_z(...)`) and with an inline invariant key `Key(Base)`. Receiver-first `Rotation.about_z(q, to=Base, from_=Link1).with_origin(Link1.origin())` reports "Expected `Point[Base]`, found `Point[Link1]`"; inline `@` with mismatched frames reports an error.

Consequence: the stub's `Transform(rotation, origin)` constructor is replaced by a constructor taking the origin's components (D17) and, for an origin that is already a point, a method on the rotation, `with_origin` (D18).

Rejected: fixing inline widening through variance alone (`Transform` is already invariant); pre-bound frame keys (`BASE = Key(Base)`) as the user's responsibility (only docs protect it).

### D10. PEP 695 generics, gated by a negative type test

PEP 695 syntax only (ruff UP040, UP046, UP047 enforce it). Because ty is pre-1.0 and inference decides variance, a negative fixture of mistakes (one per line) is run by pytest through ty, which asserts every line is an error. The test gates every ty upgrade.

The stub's fixture caught 8 of 10 (`tip + tip`, `2 * tip`, `Tool` + `Base` vectors, transform applied to a wrong-frame point, `Base←Link1 @ Base←Link1`, `q1 + "a"`, `compiled(0.5)`, `Parameters()` missing `q1`); the two misses are the inline cases closed by D9 and the explicit widening listed in Risks.

Rejected: old-style `TypeVar` with explicit variance; characterization tests or `xfail(strict)` (chat 0).

### D11. Compilation: `lambdify` against a parameters dataclass

`lambdify[P: DataclassInstance](self, parameters: type[P]) -> Compiled[P, Self]`, `Compiled.__call__(values: P) -> Numeric[Self]`. `P` is a user-written frozen dataclass (one for the whole robot, e.g. `Joints`). Fields are matched to symbols by name; ordering is internal; extra fields are allowed. Type hints survive, so `compiled(0.5)` and `Parameters()` with a missing field are type errors.

Validation is at runtime, once, at compile time: every field annotated `float` (string annotations resolved with `typing.get_type_hints`; `dataclasses.fields` alone returns the string `'float'`), every free symbol has a field, otherwise raise. The quantity and the dataclass are linked only by names; a static link needs the schema rejected in D1.

JAX is the only backend: `lambdify(..., modules="jax")` under `jax.jit`, with `jax_enable_x64`. Autodiff is part of the numeric path (whitelisted).

Rejected: caller-ordered symbols `lambdify((q1, q2))`; sorted-name order (`"q10" < "q2"`); creation-order counters; runtime-generated parameter classes (`type(...)`, `make_dataclass`, metaclass: ty sees `Unknown`/`Any`); a `backend` parameter, `Backend(StrEnum)` and per-backend overloads (until a second backend exists); `Piecewise` branching inside compiled code.

### D12. Thin frame-carrying compiled result

A compiled point, vector, rotation or transform returns a thin result `Numeric[T]` in the natural shape (`(3,)`, `(3, 3)`, `(4, 4)`, not sympy's `(3, 1)`), registered as a JAX pytree: frame as static aux data, components as the only child. It has no frame operations; simulation code reads `.components`. `vmap` leading axes are configurations and are not modeled in the type.

Principle (the user's): data is bare components; meaning (the frame) is in the type, on both sides of the boundary. `(0, 0, 1)` only makes sense while you know where it was described.

Measured: a plain class returned from `jax.jit` raises `TypeError`; registered, it passes `jit`, `vmap` (frame kept, components `(2, 3)`) and `grad` (0.8775825618903728). Cost under `jit`: single call 8.3 µs raw versus 9.6 µs wrapped; `vmap` over 1000 configurations 18.2 µs versus 19.1 µs.

Rejected: a bare `jax.Array` (frame lost).

### D13. Axis-angle is symbolic

`R.to_axis_angle()` returns an `AxisAngleSolutions` with `.first` and `.second` (each an `AxisAngle`: `.axis` and `.angle`) and `.valid_when` (a `ValidityCondition`), without evaluating anything; an axis compiles like any vector. Content (course convention): angle in (−π, π]; regular (r, θ), (−r, −θ); half turn (r, π), (−r, π); identity (a, 0), (−a, 0) with `FreeAxis`, default z. Default `tolerance=1e-9`. The axis of ^A R_B satisfies R r = r, so it has the same components in A and B.

Probed (the singular case, see D28): for `Rz(q)` sympy gives axis `[0, 0, sin(q)/Abs(sin(q))]` and angle `atan2(Abs(sin(q)), cos(q))`; compiled at q = 0.5 it gives `[0 0 1]`, at q = 0 `[0 0 nan]`, at q = π `[0 0 1]` only by luck (float sin(π) = 1.2e-16). `Rx(π/2) Rz(q)` at q = π gives `[nan -inf inf]`.

Rejected: evaluating first, then extracting (`R.evaluate({...}).to_axis_angle()`); raising for the identity; `NoRotation` / `HalfTurn` / `Undecided` union arms; an "all cases" object with mandatory `.evaluate()`; `FreeAxis` as a non-subtype.

### D14. Error principle

Raise when no answer exists (normalizing a zero vector, a free symbol without a field at compile time). Return a typed result when a singular case still carries information (axis-angle of the identity).

### D15. The exam notebook is the benchmark

`exams/2024_09_19.ipynb` keeps only the cells the new API can express, rewritten against it; the cells that need `Manipulator`, DH or the Jacobian are deleted. The notebook is the change's benchmark, and later changes add cells back as the API grows. The deleted cells stay reachable at the `legacy` tag.

Rejected: keeping the out-of-scope cells non-runnable until their change lands (a notebook that does not run top to bottom is no benchmark).

### D16. Frame keywords `of=` / `relative_to=`

Constructors that need frames their arguments cannot fix take them as keywords read from the course's sentence for ^A R_B, "the orientation of frame B relative to frame A": `Rotation.about_z(q1, of=Link1, relative_to=Base)` is `Rotation[Base, Link1]`; `Rotation.identity(of=Tool, relative_to=Wrist)`; `Rotation.about_axis(axis, q, of=Link1)`, where the axis's frame fixes the reference frame. Each frame is fixed by its own keyword, so D9 holds.

Naming rule (user's): no name clashes with the language (keywords, soft keywords, builtins), and escaping a keyword with a trailing underscore counts as a clash. Builtin shadowing is already a ruff error (`A001`–`A005` under `ALL`); the escape is not caught by any rule today, so the separate ast-grep style-rule change bans identifiers ending in a single `_`.

Probed in ty 0.0.91, bound and inline: `assert_type(Rotation.about_z(0.5, of=Link1, relative_to=Base), Rotation[Base, Link1])` and the `with_origin(Base.origin())` result as `Transform[Base, Link1]` pass; `...with_origin(Link1.origin())` is an error both inline and bound ("Expected `Point[Base]`, found `Point[Link1]`").

Rejected: `to=` / `from_=` (the stub's spelling; `from_` escapes the keyword `from`); a builder from the frames, `Link1.relative_to(Base).about_z(q1)` (passes the same probes, but adds an intermediate class whose only job is to carry two frames).

### D17. Transform constructor takes the origin's components

`Transform(rotation: Rotation[To, From], x: ScalarLike, y: ScalarLike, z: ScalarLike)`: the origin's components are read in the rotation's target frame, so both frames come from the rotation alone and D9 holds, inline included. It reads like the exam's [R | t] with t written as numbers, and like `Frame.point(x, y, z)`: one source for the frame, numbers read in it. An origin that is already a point goes through `rotation.with_origin(point)`, which checks the point's frame.

Python has no private constructors, so the public `__init__` must itself satisfy D9; the stub hid this by declaring none.

Probed in ty 0.0.91 with `Point` and `Rotation` invariant, bound and inline:
- `__init__(self, rotation: Rotation[To, From], origin: Point[To])`: inline silent (`Transform[Base | Link1, Link1]`), bound caught.
- Self-typed `__init__[T, S](self: Transform[T, S], rotation: Rotation[T, S], origin: Point[T])`: same as above.
- Components: `assert_type(Transform(Rotation.about_z(0.5, of=Link1, relative_to=Base), 0, 0, 1), Transform[Base, Link1])` passes; `Transform(Rotation.about_z(...), Link1.origin())` is an error.

A first probe run showed all three shapes silent because the probe classes were covariant; the result above is with invariance added.

Rejected: the naive and the self-typed constructors (both reopen hole 1 inline).

Accepted limit: numbers taken from a point in another frame (`Transform(R, p.x, p.y, p.z)`) pass silently, the same limit as `Frame.point(p.x, p.y, p.z)`; plain numbers carry no frame.

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
| `Rotation.with_origin(origin: Point[To]) -> Transform[To, From]` | the checked path from a rotation and an origin point to a transform (D17); pairs with `Transform.origin`: `R.with_origin(p).rotation` is `R`, `.origin` is `p`. Accepted: Python's `with_*` methods usually return the same type (`Path.with_suffix`); a rotation has no origin, so "a rotation with an origin" can only be a pose | `placed_at` (breaks the pairing with `.origin`; vaguer about which part is set), `with_translation` (a translation is a vector, w = 0; the argument is a point, D7) |

`Symbolic` here is not the rejected two-worlds scalar of D1: there is one symbolic base and one compiled result type, no `S` parameter and no gated methods.

### D19. A frame's LaTeX name is a class keyword

`class Base(Frame, latex="0")` renders `{}^{0}`; without the keyword the class name is used (`class Tool(Frame)` renders `{}^{Tool}`). Implemented with `__init_subclass__(cls, *, latex: str | None = None)` storing a `ClassVar[str]`. Reason: the exam numbers its frames (^0 R_1, ^3 T_6) and a class cannot be named `0`.

Probed in ty 0.0.91 and at runtime: `class Typo(Frame, latx="1")` is a ty `unknown-argument` error and a runtime `TypeError` ("Did you mean 'latex'?"); `latex=1` is a ty error (expected `str | None`).

Rejected: a class attribute `latex = "0"` (a misspelled attribute passes ty silently and the class name is printed); the keyword `symbol` (collides in meaning with `Scalar.symbol`, which creates a variable); the keyword `name` (reads as the Python name); always the class name (cannot express numbered frames).

### D20. Unit vectors come only from normalization

`UnitVector(direction: Vector[F])` normalizes its argument: it divides the components by the simplified norm and raises when the norm simplifies to zero (D14). `v.normalize()` is `UnitVector(v)`. As with D17, Python has no private constructors, so the public constructor itself guarantees the invariant: no `UnitVector` can hold components not divided by their own norm, and no length check exists.

Probed with sympy 1.14, norm simplified first: (cos q, sin q, 0) stays (cos(q), sin(q), 0); (1, 1, 0) gives (sqrt(2)/2, sqrt(2)/2, 0); (3, 4, 0) gives (3/5, 4/5, 0); (q, 0, 0) gives (q/Abs(q), 0, 0). Without simplifying the norm the first prints `cos(q)/sqrt(sin(q)**2 + cos(q)**2)`.

A norm that vanishes only at some configurations (q/Abs(q) at q = 0) does not raise symbolically and compiles to `nan` there; that is hole 2 (Q2), not a unit-vector question.

Rejected: a constructor taking components with a unit-length check (proving x² + y² + z² = 1 symbolically can fail and would raise for a vector that is unit, against D14); a constructor taking components without a check (the type would promise what nothing guarantees).

### D21. No public matrix type

Rotations and transforms store their entries in an internal typed square-matrix class in a boundary module (D2, D3: the rotation and transform modules may not import sympy). No public signature returns it. `Rotation` and `Transform` are the only public matrix-like concepts (D1: one class per concept) and expose their entries as `Scalar`s through a typed accessor (D26) and as LaTeX.

Rejected: a public frameless `Matrix` (a free value that does everything `Rotation[To, From]` does while bypassing the frame types, the hole D6 closed for points and vectors; nothing in this change needs one, and the Jacobian change decides about a general matrix when it needs one).

### D22. A rotation acts on a vector with `@`

`R @ v` is the only spelling: ^A v = ^A R_B ^B v, consistent with `Rotation @ Rotation` and `Transform @` points and vectors (D8). Probed in ty 0.0.91: `base_link1 @ link1_tool @ approach` and `base_link1 @ (link1_tool @ approach)` both infer `Vector[Base]`; `(base_link1 @ link1_tool).inverse() @ Base.vector(1, 0, 0)` infers `Vector[Tool]`; a wrong-frame vector at the end of a chain is an error.

Rejected: `.rotate(v)` (the active reading, the arrow moves, contradicts D8's passive one, the arrow stays and its frame of resolution changes); a second, named spelling (if one is ever added, its name must describe the passive meaning).

### D23. A rotation does not act on points

`Rotation @ Point` is a type error: a rotation relates axes and says nothing about origins, so with offset origins the result would have the right type and the wrong value (identity rotation, origin of B at (5, 0, 0) in A: the tip (1, 0, 0) in B is (6, 0, 0) in A, not (1, 0, 0)). The two honest readings stay explicit: `Transform(R, 0, 0, 0) @ p` writes the coincident-origin assumption down; `R @ (p - B.origin())` resolves the grounded vector (D7). Probed in ty 0.0.91: both infer `Point[Base]` and `Vector[Base]`; `R @ tip` is an error.

Rejected: allowing it under an implicit coincident-origin assumption.

### D24. Euler angles and RPY are deferred

The benchmark (D15) needs none: every surviving rotation in `exams/2024_09_19.ipynb` has at most one nonzero angle (identity, `Rx(pi/2)`, `Rz(pi/2)`, `Rx(-pi/2)`; the `(pi/2, 0, pi)` line in cell 9 is overwritten on the next line), and frames 0 to 6 all have names. Euler brings its own design, so it gets its own change. Facts gathered for it: fixed-axes sequence ABC with angles (a, b, c) equals moving-axes sequence CBA with angles (c, b, a); 12 sequences cover everything; course naming: Euler = moving, RPY = fixed (rejected chat-0 names: `EulerPair`, `GimbalLockSum`).

Rejected: Euler and RPY in this change (enlarges it with a design the benchmark does not exercise).

### D25. Intermediate frames are declared by the user

A frame a product of elementary rotations passes through is an ordinary frame the user declares, named with D19 as the course writes it (`class Prime(Frame, latex="'")`, RF', RF''). Probed in ty 0.0.91: `first @ second @ third` through `Prime` and `Second` infers `Rotation[Base, Tool]`; skipping a factor (`first @ third`) is an error.

Rejected: library-provided anonymous frames (ty cannot create a fresh type per call, so every intermediate frame would share one class and a chain that skips or swaps them would type-check).

### D26. Entries are read with `entry(row, column)`, counted from 1

`Rotation.entry(row: Index3, column: Index3) -> Scalar` with `type Index3 = Literal[1, 2, 3]`; `Transform.entry` uses `Index4 = Literal[1, 2, 3, 4]`. Counting from 1 follows the course (r₁₃ is `entry(1, 3)`); a named method signals that this is not Python indexing; the index range is in the type, so an out-of-range index is a type error, not an `IndexError`.

Probed in ty 0.0.91: `entry(1, 3)` infers `Scalar`; `entry(0, 2)`, `entry(4, 1)` and an `int` index are errors; `for row in (1, 2, 3)` passes (ty infers `Literal[1, 2, 3]`); `for row in range(1, 4)` is an error (ty sees `int`). Accepted: user loops over entries use a literal tuple; library code walks the internal storage (D21), not `entry`.

Rejected: `R[0, 2]` for r₁₃ (an off-by-one on every transcription from a solution); `R[1, 3]` (square brackets promise Python's 0-based indexing); one property per entry, `r13` and so on (9 + 16 cryptic names).

### D27. No frame accessor on compiled results in this change

`Numeric[T]` exposes `.components` only. The meaning is the type `T`; the runtime frame, kept in the pytree aux data (D12), appears in the printed form (`Numeric[Point[Base]]([1.0, 0.5, 0.0])`). Nothing in this change reads a compiled frame: simulation reads `.components`, the benchmark compares numbers.

Additive path for a later change (probed in ty 0.0.91, one class, no new types): self-annotated methods offered only for the matching `T`, `frame[F](self: Numeric[Point[F]]) -> type[F]` and `relative_to[A, B](self: Numeric[Rotation[A, B]]) -> type[A]`; `scalar.frame()` and `rotation.frame()` are errors.

Rejected: a plain `.frame` property (no single return type fits a point, a rotation and a scalar); per-concept result classes (reopens D18's single `Numeric[T]`); adding the gated accessors now (no reader in this change).

### D28. Singular configurations surface as a compiled flag

`ValidityCondition` is a `Symbolic` and compiles to a boolean of shape `()`, applying the tolerance of `to_axis_angle` (`|expression| > tolerance`, default `1e-9`) instead of `≠ 0`. `AxisAngleSolutions` is a `Symbolic` too; compiling it as a whole returns both solutions and the flag in one result, so on the main path the flag cannot be left out. When the flag is false the caller decides; nothing is substituted.

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
