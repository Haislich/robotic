# Tasks

## 1. Legacy removal

- [x] 1.1 Tag commit `41d2833` (name agreed with the user, e.g. `legacy`); verify `git rev-parse <tag>` prints `41d2833...`
- [x] 1.2 Delete `src/robotic/transformations.py`, `manipulator.py`, `utils.py`, `tests/test_axis.py`, `tests/test_rotation.py` and the five `midterms/midterm_*.ipynb`; ask the user about untracked `src/robotic/p.py` and the deleted `src/robotic/py.typed` before touching either; verify `git status` lists exactly these deletions
- [x] 1.3 Remove `loguru`, `pandas`, `pandas-stubs`, `matplotlib` from `pyproject.toml`; verify `uv lock` resolves, `uv sync` succeeds and `rg -l "loguru|pandas|matplotlib" src tests` finds nothing

## 2. Scalar layer

- [x] 2.1 Add the `sympy` entry to `banned-api` with the per-file exception for the boundary modules in `src/` (design D3; tests and notebooks stay uncovered by decision), after user approval of the configuration text; verify a probe module in `src/` outside the boundary importing sympy reports `TID251`, a probe inside the boundary does not, then delete the probes
- [x] 2.2 Build the negative type test (design D10): a fixture file with one mistake per line and a pytest test that runs ty on it and asserts every line is an error; verify by adding a correct line temporarily and seeing the test fail
- [x] 2.3 Implement the scalar, symbol creation, exact constants and elementary functions (design D2, D4, D5); verify the `scalar` spec scenarios pass as tests, `q + "a"` is in the negative fixture and caught, and `uv run ruff check` and `uv run ty check` are clean
- [x] 2.4 Add the test that walks every public signature and annotation and fails on any sympy type (spec `scalar`, "No untyped algebra in public signatures"); verify it fails on a temporary public method returning `sympy.Expr`, then remove that method
- [x] 2.5 Document the scalar layer in `docs/` (symbols, exact constants, the float policy with the `6.12323399573677e-17` example); verify every code block runs as written

## 3. Geometry layer

- [x] 3.1 Implement frames, points and vectors with affine arithmetic and runtime frames (design D6, D7); verify the `geometry` spec scenarios pass and `tip + tip`, `2 * tip`, `-tip`, Tool + Base vectors and the bound widening are in the negative fixture and caught
- [x] 3.2 Implement dot, cross, norm, `UnitVector` built only by normalization, `normalize` and `FreeAxis` (design D6, D14, D20); verify the spec scenarios, including the zero vector raising and the simplified norm
- [x] 3.3 Implement simplification, the frame's `latex=` class keyword and LaTeX with the frame as left superscript (design D19); verify the default-name and declared-name scenarios, and the misspelled keyword caught by the negative fixture
- [x] 3.4 Document frames and affine arithmetic in `docs/` (two-fingertip example, homogeneous w table); verify every code block runs as written

## 4. Rotation layer

- [x] 4.1 Implement the internal typed matrix storage and `entry(row, column)` counted from 1 (design D21, D26), then elementary, axis and identity rotations with the `of=` / `relative_to=` keywords, composition and inverse under the receiver-first rule (design D8, D9, D16); verify the `rotation` spec scenarios, matrices against the spatialmath oracle at sampled angles, and inline wrong-frame arguments, mismatched chains and an out-of-range entry index caught by the negative fixture
- [x] 4.2 Implement the rotation acting on vectors with `@` (design D22); verify the spec scenarios, and the wrong-source-frame vector, the rotation applied to a point (design D23) and the skipped intermediate frame (design D25) caught by the negative fixture
- [x] 4.3 Implement symbolic axis-angle as `AxisAngleSolutions` (`.first`, `.second` as `AxisAngle`, `.valid_when` as `ValidityCondition`, design D13, D18), tolerance `1e-9` and the free axis (design D13); verify the regular, half-turn, identity and symbolic scenarios, and oracle agreement at samples near θ = 0 and θ = π
- [x] 4.4 Document rotations and axis-angle in `docs/` (^A R_B notation, course convention for the two solutions); verify every code block runs as written

## 5. Transform layer

- [x] 5.1 Implement transforms built by the constructor from a rotation and three numbers (design D17) and by `with_origin` on the rotation (design D9, D18), with rotation, origin and 4x4 matrix; verify the matrix-layout and three-numbers scenarios, and an inline wrong-frame origin given to the method and an inline point given to the constructor both caught by the negative fixture
- [x] 5.2 Implement composition, inverse and action on points and vectors (design D8); verify the planar 2R tip scenario, the translation-free vector scenario, the mismatched-chain line in the negative fixture, and matrices against the spatialmath oracle
- [x] 5.3 Document transforms in `docs/` (planar 2R forward kinematics end to end); verify every code block runs as written

## 6. Compilation layer

- [x] 6.1 Implement `lambdify` returning `Compiled[P, T]` against a parameters dataclass with compile-time validation (design D11; fields `float` or `ArrayLike`); verify the missing-field, non-float field, string-annotation, batchable-field and extra-field scenarios, and `compiled(0.5)` and a missing constructor argument caught by the negative fixture
- [x] 6.2 Implement the thin frame-carrying result `Numeric[T]` with no frame accessor and the frame in its printed form (design D27), as a registered pytree in natural shapes, with `jax_enable_x64` (design D12); verify shapes and float64 dtype for each quantity, and `jit`, `vmap` (components `(2, 3)`, frame kept) and `grad` (0.8775825618903728)
- [x] 6.3 Add property tests comparing compiled results with the symbolic quantity substituted and with the oracle, sampling near singular configurations; verify they pass
- [x] 6.4 Make `ValidityCondition` compile to a flag with the tolerance and `AxisAngleSolutions` compile to `(2, 4)` components with its flag read by the gated `valid()` (design D27, D28); verify, for `Rz(q)`, the flag is true at q = 0.5 and q = 1e-6, and false at q = 0, q = 1e-12 and q = π, and `valid()` on a compiled point caught by the negative fixture
- [x] 6.5 Document compilation in `docs/` (parameters dataclass, validation errors, `.components` in simulation, `vmap`, axis-angle compiled with its flag shown before an axis compiled alone); verify every code block runs as written

## 7. Acceptance: exam 2024-09-19

- [x] 7.1 Rewrite the cells of `exams/2024_09_19.ipynb` the new API can express and delete the cells that need `Manipulator`, DH or the Jacobian (design D15); verify the notebook runs top to bottom in the project kernel and the chained frames and wrist position match the values the legacy notebook produced at tag `legacy`

## 8. Integration

- [x] 8.1 Update `README.md` to describe the new library (no Jacobians or manipulator until their change); verify every documented command runs as written
- [x] 8.2 Verify the whole change together: `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, `uv run pytest` (negative type test included) all pass, `openspec validate refactor-math-layers --strict` passes, and `git status` lists only the files named in the proposal's Impact section

## 9. Review round 1

Completed tasks 3.3, 4.1 and 5.1 are superseded where they name `latex=`, `of=` / `relative_to=`, the three-number constructor or `with_origin`; this section redoes those parts.

- [x] 9.1 Rename the frame vocabulary to Target/Source in `src/`, tests, docs, the negative fixture and the notebook (design D16); verify `rg -nw "relative_to|To|From" src tests docs` and `rg -n "\bof=" src tests docs` find no API use, and the inline wrong-frame lines of the negative fixture are still errors
- [x] 9.2 Rename `Frame.latex` to `Frame.notation` (design D19); verify the "Declared notation" and "Misspelled declaration" scenarios, and `rg -n 'latex="' src tests docs` finds nothing
- [x] 9.3 Put the frame last in `Point(x, y, z, frame)` and `Vector(x, y, z, frame)` (design D30); verify ty and the geometry tests pass
- [x] 9.4 Build transforms from their frames: `Transform(target=, source=)`, `with_rotation`, `with_translation`, `.translation`; remove the three-number constructor and `Rotation.with_origin`; composition, inverse and action through the translation; `rotation.py` no longer imports `transform.py` (design D17, D29, D33); verify the transform spec scenarios (identity, immutability, both inline wrong-frame lines added to the negative fixture) and the spatialmath oracle tests
- [x] 9.5 One spelling per concept (design D34): `UnitVector(...)` outside `geometry.py` becomes `.normalize()`, `Symbol(...)` outside `symbolic.py` becomes `Scalar.symbol(...)`; verify with `rg` that neither constructor is called outside its module
- [x] 9.6 Fix the half-turn validity condition to `r_kk + 1 != 0` (design D33); verify the new scenario fails first, then passes (axis (a, 1, 0) normalized, angle π, flag false at a = 0)
- [x] 9.7 Axis-angle cleanups (design D33): `UnitVector.__neg__` returns a `UnitVector`, the second solution is `-first.axis`, the norm is simplified once; `AxisAngleSolutions` compiles its condition through the condition's layout; `Rotation.simplify` uses the private factory; verify the axis-angle tests pass and `assert_type(-axis, UnitVector[F])` is clean
- [x] 9.8 Batching through vmap (design D11, D31): `ParameterShapeError` raised at trace time; correct the "floats and arrays alike" sentence in `docs/compilation.md`; verify both new scenarios, the direct-array one failing first
- [x] 9.9 Update every `docs/` page and `README.md` to the new spellings; verify `tests/test_docs.py` passes
- [x] 9.10 Rewrite `exams/2024_09_19.ipynb` against the new API, give `Frame3Down` a notation distinct from `Joint3`'s `3'`, and remove the dead `r_3_downward` derivation (user approves the notation and the removal); verify the notebook runs top to bottom with the same values as before
- [x] 9.11 Verify the whole change as in 8.2

## 10. Review round 2

- [x] 10.1 A `Numeric` prints under tracing (design D12): a test that prints a compiled result inside `jax.jit` fails first, then passes; the concrete printed form is unchanged (`Numeric[Point[Base]]([1.5, 0.0, 0.0])` in `docs/compilation.md`)
- [x] 10.2 Record that the regular case of `to_axis_angle` simplifies the norm of the skew part twice, once for sin θ and once inside `normalize` (design D33, open point); the code is unchanged; verify the axis-angle tests pass
- [x] 10.3 `UnitVector.simplify` simplifies the components without normalizing again, as `__neg__` does (design D20); verify `FreeAxis` stays `FreeAxis` and the geometry tests pass
- [x] 10.4 One matrix-vector product and one bmatrix rendering: `Rotation @ Vector` uses `_Matrix3`, and `Transform.latex` shares the row rendering; verify the LaTeX and oracle tests pass unchanged
- [x] 10.5 State the process-wide `jax_enable_x64` in `docs/compilation.md` and `README.md` (design D12); verify `tests/test_docs.py` passes
- [x] 10.6 Verify the whole change as in 8.2

## 11. Review round 3

- [x] 11.1 A symbolic factor on the left scales a vector (design D31): every `Scalar` operator returns `NotImplemented` for an operand outside `ScalarLike`, and `Vector.__rmul__` takes `Scalar | float`; verify a test of `l1 * x_axis` (ty-clean, `Vector[A](l1, 0, 0)`) fails first, then passes, and `x_axis * l1` is unchanged
- [x] 11.2 `Rotation.__matmul__` and `Transform.__matmul__` return `NotImplemented` for other operands (design D23); verify a test that `R @ point` raises `TypeError` at runtime fails first, then passes, and the negative fixture still reports it
- [x] 11.3 `Symbol.simplify` returns the symbol, typed `Self` (design D31); verify `assert_type(q.simplify(), Symbol)`
- [x] 11.4 Keep `Numeric._unflatten` on the explicit `Numeric[Symbolic](...)` and record why (design D12); verify ty is clean and the jit, vmap and grad tests pass
- [x] 11.5 Document that derivatives are taken of `.components` in `docs/compilation.md`, with a `jax.jacrev` doctest of a compiled rotation (design D12); verify `tests/test_docs.py` passes
- [x] 11.6 Verify the whole change as in 8.2

## 12. Review round 4: against `openspec/intent.md`

- [x] 12.1 Every static promise also holds at runtime (design D35): `FrameMismatchError(TypeError)` for operands in different frames, `TypeError` for operands and arguments outside the accepted types, `IndexError` for an entry out of range, `TypeError` for `.valid()` without a flag; mark annotation-only fixture lines `# static only:`; verify a test executing every `# mistake:` line of `tests/type_mistakes.py.txt` fails first, then passes, and a test of the `v1: Vector[B2] = Vector(0, 0, 1, B1)` case (`v1 + v2`, `v2 + v1` raise naming B1 and B2; `v1 + v1` is (0, 0, 2) in B1)
- [x] 12.2 Singular configurations in compiled code promise nothing (design D20, Risks; proposal, Out of Scope): remove the doctest line printing `[0.0, 0.0, nan]` from `docs/compilation.md` and state that the behavior at singular configurations is not decided yet; verify `rg -n "nan" docs README.md` finds no promised value and `tests/test_docs.py` passes
- [x] 12.3 No validity condition (design D13, D28): a rotation whose simplified entries have no symbol is resolved exactly, one with a symbol always gets the regular formulas; remove `ValidityCondition`, `.valid_when`, `tolerance`, `Numeric.valid()` and the flag pytree child; remove the `.valid()` lines from the negative fixture and the flag from `docs/rotation.md`, `docs/compilation.md` and the exam notebook if present; verify the rotation and compilation spec scenarios (the symbolic half turn raising first fails, then passes), `rg -n "valid_when|ValidityCondition|valid\(\)" src tests docs exams` finds nothing, and the whole check as in 8.2
- [x] 12.4 Compile with common subexpression elimination and without implicit simplification (design D11); verify a 32-rotation R R⁻¹ chain compiles and runs its first call in under 5 s, the compiled results of the existing tests are unchanged within 1e-12, and the 6R forward kinematics compile time does not grow
- [x] 12.5 Choose the simplification strategy (design D5): replace `sympy.simplify` with the trigonometric strategy that passes the whole test suite (probe `trigsimp(method="fu")` first), and use it in `is_zero` too; verify the full check as in 8.2, the `monster` identity simplifying to the identity, and re-measure simplify time on 4- and 6-joint alternating chains through the library, recorded in D5
- [x] 12.6 No equality on symbolic and compiled values (design D36): `__eq__`, `__ne__`, `__hash__` and `__bool__` raise `TypeError` on `Symbolic` and `Numeric`, `AxisAngle` included; verify the spec scenarios fail first, then pass, `rg` shows no test, doc or notebook relying on `==` between these values, and the full check as in 8.2
- [x] 12.7 Remove `Scalar.is_zero()` from the API (design D31): the zero test becomes the private function `_is_zero` in `symbolic.py`, imported by `geometry.py` and `rotation.py`, using the strategy of 12.5; `tests/test_scalar.py` tests it through `normalize` and `to_axis_angle` instead; verify `rg -n "\.is_zero\(" src tests docs exams` finds nothing, the public-signature test passes, and the full check as in 8.2
- [x] 12.8 Rename the method `lambdify` to `compile` (design D11, D16) in `src/`, tests, the negative fixture, `docs/`, `README.md` and the exam notebook; `sympy.lambdify` inside the boundary keeps its name; verify `rg -n "\.lambdify\(" src tests docs exams README.md` finds only the boundary's `sympy.lambdify`, the notebook runs, and the full check as in 8.2
- [x] 12.9 The symbolic side takes exact numbers only (design D4, D31): `ScalarLike` is `Scalar | int`, reverse operators and `Vector.__rmul__` take `int`, every entry point rejects `float` (and `bool`) at runtime; add `q + 0.1` and a rotation by `math.pi` to the negative fixture; rewrite symbolic floats in tests and `docs/scalar.md` (the float policy section becomes the exact-only rule) with exact values; numeric values in `tests/test_compilation.py` and `docs/compilation.md` stay floats; verify the scalar spec scenarios fail first, then pass, and the full check as in 8.2
- [x] 12.10 A transform is the homogeneous matrix of a rotation (design D17): add `Rotation.homogeneous(translation: Point[Target] | None = None)`, `Transform.translation` (first applied as `origin`, renamed by the user); `Transform.__init__` takes the rotation only; remove `Transform(target=, source=)`, `with_rotation`, `with_translation`; resolve the import between `rotation.py` and `transform.py` without a cycle; update the negative fixture (origin in the wrong frame inline, a vector as origin), tests, `docs/transform.md`, `README.md` and the exam notebook; verify the transform spec scenarios, the spatialmath oracle tests, the notebook values unchanged, `rg -n "with_rotation|with_translation" src tests docs exams README.md` finds nothing, and the full check as in 8.2
- [x] 12.11 Positions and displacements (design D37): rename `Point` to `Position` and `Vector` to `Displacement`, `Frame.point` to `Frame.position`, `Frame.vector` to `Frame.displacement`, `UnitVector` to `Direction` (`FreeAxis` stays, a subtype); add `Frame.direction(x, y, z)`, which normalizes and raises on zero; add the alias `Point = Position` (plain assignment, exported); `Rotation.homogeneous` takes a `Position[Target]`; update the negative fixture, tests, `docs/`, `README.md` and the exam notebook; verify `isinstance(Base.position(0, 0, 0), Point)` and `match` on `Point()` work, the geometry spec scenarios pass under the new names, `rg -nw "Vector|UnitVector|\.vector\(|\.point\(" src tests docs exams README.md` finds no old spelling, and the full check as in 8.2

## 13. Review round 5: constants and expressions (intent I8)

From the code review of round 4 (findings #1-#10) and the decisions of 2026-10-30 (design D38). Completed tasks that name removed API (3.2 `UnitVector` and the zero vector simplifying to zero, 4.3 `.valid_when` and the tolerance `1e-9` of the flag, 6.4 `ValidityCondition`, 12.5 and 12.7 the zero test) are superseded where they do; this section and section 12 redo those parts. Finding #7 (`compile` and the builtin) is closed with no change (design D11). Finding #8 is closed by this revision of the artifacts.

- [x] 13.0 `to_axis_angle` checks that `free_axis` is a direction in its target frame before any branch (finding #2); verified by two negative-fixture lines and the rotation tests
- [x] 13.1 Split the scalar (design D38, D31, D34): `Scalar` for constants, `Expression` for values built from a symbol, `Symbol(Expression)` created by `Symbol(name)`; promotion by operand class, statically (overloads) and at runtime; remove `Scalar.symbol`; the `==` message points to `is_close` for constants; verify the scalar spec scenarios "Typed scalar arithmetic", "Elementary functions", "A variable is required" and "Equal constants" fail first, then pass, `assert_type(q - q, Expression)`, negative-fixture lines for a `Symbol`-requiring function given `q + 1`, and `rg -n "Scalar\.symbol" src tests docs exams README.md` finds nothing
- [x] 13.2 `Scalar.is_close(other, tolerance=1e-9)`, absolute, on the difference evaluated to 30 digits (design D38); verify the "Constants compare with a tolerance" scenarios fail first, then pass, and `q.is_close(0)` is a `# mistake:` line of the negative fixture
- [x] 13.3 `evaluate(values)` on `Expression` and on every symbolic composite (design D38), raising `MissingParameterError` naming the missing symbols and rejecting floats; verify the "Evaluation substitutes exact values" scenarios fail first, then pass, and a float value is a `# mistake:` line
- [x] 13.4 The kind parameter on every composite, `Position[K, F]` through `AxisAngleSolutions[K, F]`, with two overloads per constructor and binary operation, and constant components stored as `Expression` in symbolic composites (design D38, D35); verify the "Constant or symbolic is in the type" scenarios, the transform scenario "A symbolic translation makes a symbolic transform", every existing fixture line still behaves as marked, and ty is clean on `src/` and tests
- [x] 13.5 Normalization and axis-angle without a zero test (design D13, D20, D31, D33): constants decide with `is_close`, the identity and the half turn are told apart by the sign of the cosine, expressions never raise; remove `_is_zero`, `_has_symbols` and `tests/test_zero_norm_properties.py`; verify the geometry scenarios "Normalizing a constant zero vector raises" and the rotation axis-angle scenarios fail first where changed, then pass, `rg -n "_is_zero|_has_symbols" src tests` finds nothing, and re-measure `to_axis_angle` on a 4-joint alternating chain (finding #6: 133 s before), recorded in D13
- [x] 13.6 Constants compile with no parameters, and `compile` rejects an instance of the parameters dataclass (design D38, D11; finding #5: `is_dataclass` accepts instances); verify the compilation scenarios "Constants compile without parameters" and "An instance instead of the class" fail first, then pass
- [x] 13.7 Frame checks raise `FrameMismatchError` directly in each method, with the message built by the exception from the operation and both values, and the docstrings list it (design D35; finding #9); verify the 14 frame-mismatch fixture lines still raise, ruff's docstring rules are clean on `src/`, and `rg` finds the `Rotation` description built in one place
- [x] 13.8 `Transform @ Position` in the wrong frame names the transform and the position (design D35; finding #3); verify the transform scenario "Point in the wrong frame" fails first on the message, then passes
- [x] 13.9 `Displacement * Displacement` returns `NotImplemented`, so Python raises its unsupported-operand `TypeError` instead of the scalar's float hint (design D35; finding #4); verify a test of the message fails first, then passes
- [x] 13.10 The exactness check of `Scalar` runs once per construction path (finding #10); verify every float entry point still raises (the D4 fixture lines) and the scalar tests pass
- [x] 13.11 Update `docs/`, `README.md` and `exams/2024_09_19.ipynb` to the new classes, the kind parameter, `Symbol("q")`, `is_close`, `evaluate` and constant compilation; verify `tests/test_docs.py` passes and the notebook runs top to bottom with the same values
- [x] 13.12 Verify the whole change as in 8.2

## 14. Review round 6: nothing simplifies implicitly

From the user's decision of 2026-10-30 (design D5): `simplify` runs only when the code calls it. Closes D33's open point and finding #6 of the code review of round 4.

- [x] 14.1 `Direction` divides by the norm as computed; `to_axis_angle` simplifies neither the rotation nor the skew norm (design D5, D13, D20, D33); verify the geometry scenario "The norm is not simplified" fails first, then passes, `rg -n "\.simplify\(\)" src` finds calls only inside `simplify` methods, and re-measure `to_axis_angle` on the 4-joint alternating chain (89 s before), recorded in D13
- [x] 14.2 Verify the scalar scenario "LaTeX does not simplify" (expected to pass already, so no fail-first)
- [x] 14.3 Update tests, `docs/`, `README.md` and `exams/2024_09_19.ipynb` where they expect a simplified form, calling `.simplify()` explicitly; verify `tests/test_docs.py` passes and the notebook runs top to bottom with the same values
- [x] 14.4 Verify the whole change as in 8.2

## 15. Review round 7

From `/code-review 08fef31..HEAD` (2026-10-30) and the user's decisions of the same day (design D38). Findings #1, #2 and #4 (constants within the tolerance of zero) are kept, recorded in D38.

- [x] 15.1 Constants are real numbers (design D38): add `UndefinedValueError(ArithmeticError)`, exported; every computed constant and every `evaluate` result is checked for `nan`, `zoo`, `oo`, `-oo` and `I`, and `evaluate` names the values substituted; verify the scalar scenarios "Constants are real numbers" and the rotation scenario "Evaluating at a singular configuration raises" fail first, then pass
- [x] 15.2 `copy.copy` and `copy.deepcopy` work on every composite value again (finding #6: broken by the `__new__` constructors of round 5); verify a test copying a position, displacement, direction, free axis, rotation, transform and axis-angle result fails first, then passes, and `pickle` still round-trips
- [x] 15.3 `Direction.evaluate` returns a `Direction` without normalizing again (finding #7); verify the evaluated axis of `Rz(q)` at `q = 1` holds no square root and is typed `Direction[Scalar, F]`, failing first
- [x] 15.4 Frame-mismatch messages share one wording across `geometry.py`, `rotation.py` and `transform.py`, the checks staying in each method (design D35; finding #8); verify a test of the `add`, `dot` and `cross` messages fails first, then passes, and the 14 frame-mismatch fixture lines still raise
- [x] 15.5 Operations whose operands have the same kind store their components without re-checking and re-wrapping them, if it stays simple; otherwise record why not in D38 (finding #9); verify the full test suite
- [x] 15.6 Document `UndefinedValueError` in `docs/scalar.md` and `docs/rotation.md`; verify `tests/test_docs.py` passes
- [x] 15.7 Verify the whole change as in 8.2

## 16. Review round 8

From `/code-review 92ff8f3..HEAD` (2026-10-30). Finding #7 (the operand label repeated in seven places) is kept, as recorded in D35.

- [x] 16.1 The real-number check of constants in two tiers (design D38; findings #1, #2, #8): one pass rejects `nan`, `zoo` and the infinities; a value holding `I` or a fractional power of a negative number is expanded into real and imaginary parts and rejected only if `I` remains; verify `Scalar(-8) ** (Scalar(1) / 3)` raises and `((q.sqrt() + 1) * (q.sqrt() - 1)).evaluate({"q": -1})` gives -2, both failing first
- [x] 16.2 The message of `UndefinedValueError` says undefined, infinite or complex in words, beside the value (finding #9); verify the scalar tests and `docs/` pass with the new wording
- [x] 16.3 `FreeAxis.evaluate` keeps `FreeAxis` (finding #3), through a private factory on each displacement class; one factory built from `cls` (finding #4) is not possible in ty 0.0.91, see D38; verify a test evaluating a symbolic free axis fails first, then passes
- [x] 16.4 Document `UndefinedValueError` (findings #5, #6): `Expression.evaluate` raises it directly and lists it under `Raises`; `Scalar` and the composites (`Displacement`, `Position`, `Rotation`, `Transform`, `AxisAngleSolutions`), which raise it through their components, state it in their class docstrings, since ruff's `docstring-extraneous-exception` rejects a `Raises` entry a method does not raise itself (design D38); verify ruff's docstring rules pass
- [x] 16.5 Verify the whole change as in 8.2

## 17. Review round 9

From `/code-review 1692b14..HEAD` (2026-10-30) and the user's decision of the same day: complex values are not supported (design D38; findings #1, #2, #3, #7, #9 go with the complex check). Finding #8 (three private factories) is kept, as recorded in D38.

- [x] 17.1 Remove the complex tier of the constant check: one pass for `nan` and the infinities, the reason is `undefined` or `infinite`, and the docstrings, tests and `docs/` no longer promise anything on complex values (design D38); verify the scalar scenarios "Constants are never undefined or infinite" pass and `rg -n "complex" src docs` finds no promise about complex values
- [x] 17.2 `FreeAxis.evaluate` is typed `FreeAxis[Scalar, F]` (finding #5); verify `assert_type` on an evaluated symbolic free axis, failing first under ty
- [x] 17.3 Verify the whole change as in 8.2

## 18. Final cleanup

From `/code-review 9378122..HEAD` (2026-10-30). No finding is a bug against the contract: #1 and #2 (values holding `I`) are out of scope by the decision of round 9 (user, 2026-10-30). The rest are names and wording left by the earlier "constants are real" rule.

- [x] 18.1 `UndefinedValueError` says "the value is undefined (nan)" or "infinite (zoo)", without "not a real number"; `_real` and `NotReal` become `_undefined` and `Undefined`; `is_close` loses its unreachable `nan` branch and the promise about it; two scalar tests take the requirement's name; `Displacement.evaluate` says a free axis stays a free axis; the broken sentence of `docs/scalar.md` is fixed (findings #3-#8); verify the full check as in 8.2

## Workflow follow-up

- Archive the change after reconciling the diff with the proposal and design, per `openspec/config.yaml`.
- Propose the separate ast-grep style-rule change.
