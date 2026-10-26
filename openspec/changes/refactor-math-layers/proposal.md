# Proposal

## Why

The legacy library (`src/robotic/transformations.py`, `manipulator.py`, `utils.py`) is buggy (3 of 22 tests fail), fails the project checks (420 ruff errors, 85 ty diagnostics), and its types do not say what a value means: a 3-vector carries no frame, sympy values flow untyped through every signature, and numeric and symbolic paths are separate code. This change replaces its mathematical foundation with layers whose code reads like a handwritten exam solution, where the type checker rejects frame mistakes and affine mistakes before anything runs.

## What Changes

- **BREAKING**: delete the legacy library and its tests; tag the last legacy commit `41d2833` first so the old code stays reachable.
- **BREAKING**: delete the five midterm notebooks (`midterms/midterm_2020-11-20.ipynb` ... `midterm_2024-11-22.ipynb`); keep `exams/2024_09_19.ipynb` as the running example.
- New scalar layer: a typed class wrapping sympy expressions. sympy is the only algebra; no public signature takes or returns a sympy type, and sympy imports are banned outside the boundary modules.
- New geometry layer: frames as types; points and vectors resolved in a frame, with affine arithmetic enforced by the type checker.
- New rotation layer: `Rotation[Target, Source]`, composition, inverse, elementary rotations, action on vectors, symbolic axis-angle extraction following the course convention.
- New transform layer: `Transform[Target, Source]` built from its two frames, composition, inverse, action on points and vectors.
- New compilation layer: any quantity compiles through `lambdify` to a JAX function of a user-written parameters dataclass, returning a thin result that keeps the frame in its type.
- Negative type test: a fixture of mistakes that must each be a ty error, run by pytest, gating ty upgrades.
- Drop dependencies `loguru`, `pandas`, `pandas-stubs`, `matplotlib`.

## Capabilities

### New Capabilities

- `scalar`: the typed scalar wrapping sympy expressions, symbol creation, exact constants, float policy, sympy confinement.
- `geometry`: frames, points and vectors resolved in a frame, affine arithmetic, vector algebra.
- `rotation`: rotations between frames, their composition and action on vectors, axis-angle extraction.
- `transform`: homogeneous transformations between frames and their action on points and vectors.
- `compilation`: compiling a symbolic quantity to a JAX function of a parameters dataclass, its runtime validation and its frame-carrying result.

### Modified Capabilities

None. `openspec/specs/` is empty.

## Impact

- Code: `src/robotic/` rewritten from empty (`symbolic.py`, `geometry.py`, `rotation.py`, `transform.py`, `__init__.py`); `tests/test_axis.py` and `tests/test_rotation.py` replaced by one test module per capability plus the negative type test (`tests/type_mistakes.py.txt`), the public-signature test and the docs test. `src/robotic/p.py` (untracked scratch) was deleted at the user's choice.
- Configuration: `pyproject.toml` gains a `banned-api` entry for `sympy` and per-file exceptions for the boundary module and the negative type test (design D3), names its rules by name, and loses four dependencies; `uv.lock` follows.
- Documentation: `docs/` gains one doctested page per layer (scalar, geometry, rotation, transform, compilation); `README.md` describes the new library.
- Notebooks: `exams/2024_09_19.ipynb` is the benchmark. Its cells the new API can express are rewritten against it; its cells that need `Manipulator`, DH or the Jacobian are deleted and come back with later changes (they stay reachable at the `legacy` tag). The five `midterms/` notebooks are deleted.

## Out of Scope / Deferred

- Manipulator, robot definition, DH tables, Jacobians (note for later: the angular part of the geometric Jacobian is ω = vee(Ṙ Rᵀ)), inverse kinematics.
- Plotting and workspace analysis (`draw_frame`, TODO.md items).
- URDF or real-robot I/O.
- Free points and vectors (without a frame) and a frame graph.
- Euler angles and RPY (deferred to their own change; design D24).
- Numeric backends other than JAX.
- Static tracking of which symbols an expression contains.
- Simplifying the skew norm of axis-angle once instead of twice (design D33, open point).
- Singular configurations in compiled code (an axis compiled alone, a normalized vector whose norm vanishes at some configuration): whether they raise, carry a validity flag, or are caught by a debug mode. Open on purpose (`openspec/intent.md`); this change promises no value there, not even `nan`. The flagged path of axis-angle (design D28) is unaffected.
- The ast-grep style-rule change (separate, small, planned), including a ban on identifiers ending in a single `_` (keyword escapes such as `from_`).
