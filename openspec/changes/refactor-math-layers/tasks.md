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

## Workflow follow-up

- Archive the change after reconciling the diff with the proposal and design, per `openspec/config.yaml`.
- Propose the separate ast-grep style-rule change.
