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

- [ ] 7.1 Rewrite the cells of `exams/2024_09_19.ipynb` the new API can express and delete the cells that need `Manipulator`, DH or the Jacobian (design D15); verify the notebook runs top to bottom in the project kernel and the chained frames and wrist position match the values the legacy notebook produced at tag `legacy`

## 8. Integration

- [ ] 8.1 Update `README.md` to describe the new library (no Jacobians or manipulator until their change); verify every documented command runs as written
- [ ] 8.2 Verify the whole change together: `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, `uv run pytest` (negative type test included) all pass, `openspec validate refactor-math-layers --strict` passes, and `git status` lists only the files named in the proposal's Impact section

## Workflow follow-up

- Archive the change after reconciling the diff with the proposal and design, per `openspec/config.yaml`.
- Propose the separate ast-grep style-rule change.
