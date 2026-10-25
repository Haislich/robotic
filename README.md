# robotic

Hand-implemented robot kinematics for studying the robotics course and verifying exam exercises: rotation representations (axis-angle, Euler angles), homogeneous transformations and manipulator Jacobians, computed both symbolically and numerically.

| Path | Contents |
|---|---|
| `src/robotic/` | The library |
| `tests/` | Tests |
| `exams/`, `midterms/`, `formulas/` | Notebooks and reference material |
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
