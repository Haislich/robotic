# robotic

Hand-implemented robot kinematics for studying the robotics course and verifying exam exercises, written so the code reads like a handwritten solution: frames are types, ^A R_B is `Rotation[A, B]`, and mixing frames or adding two points is a type error before anything runs.

Everything is symbolic (sympy, behind a typed boundary): scalars, points and vectors resolved in a frame, rotations with their axis-angle solutions, and homogeneous transformations. Numbers come from compiling a quantity to a JAX function of a parameters dataclass, which works under `jax.jit`, `jax.vmap` and `jax.grad`. Euler angles, manipulators, DH tables and Jacobians come with later changes.

```python
from robotic import Frame, Rotation, Scalar


class Base(Frame, notation="0"):
    """Fixed frame."""


class Tool(Frame):
    """Frame of the tool."""


q = Scalar.symbol("q")
base_tool = Rotation.about_z(q, target=Base, source=Tool)
approach = base_tool @ Tool.vector(1, 0, 0)  # Vector[Base](cos(q), sin(q), 0)
```

| Path | Contents |
|---|---|
| `src/robotic/` | The library |
| `tests/` | Tests, including the negative type test (`tests/type_mistakes.py.txt`) |
| `docs/` | Usage and theory, one page per layer; every example runs as a doctest |
| `exams/`, `formulas/` | Notebooks and reference material |
| `openspec/` | Design rationale and decisions |
| `CLAUDE.md` | Working contract for Claude Code |

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.14 if it is missing.

```bash
uv sync
uv run pre-commit install
```

`uv sync` installs every dependency group: the library, development tools (`dev`) and notebook support (`notebooks`). In the notebooks, select the project's `.venv` as the kernel.

## Checks

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```

The pre-commit hook runs formatting, linting and type checking on staged Python files.
