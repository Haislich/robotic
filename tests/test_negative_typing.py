"""The negative type test: the fixture's mistakes are ty errors and raise when run.

Each line of the fixture carries at most one marker (see its docstring). ty runs once
on the whole fixture. Python runs one module per raising line: every line that is
silent at runtime, in order, plus the line under test.
"""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).with_name("type_mistakes.py.txt")
PROJECT = Path(__file__).parents[1]
DIAGNOSTIC = re.compile(r"^.+?:(?P<line>\d+):\d+: error\[")
LINES = FIXTURE.read_text().splitlines()


def marked_lines(*markers: str) -> frozenset[int]:
    return frozenset(
        number
        for number, line in enumerate(LINES, start=1)
        if any(f"  # {marker}:" in line for marker in markers)
    )


STATIC_ERRORS = marked_lines("mistake", "static only")
RAISING = marked_lines("mistake", "runtime only")


@pytest.fixture(scope="module")
def error_lines(tmp_path_factory: pytest.TempPathFactory) -> frozenset[int]:
    source = tmp_path_factory.mktemp("negative") / "type_mistakes.py"
    shutil.copyfile(FIXTURE, source)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--output-format",
            "concise",
            str(source),
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    matches = (DIAGNOSTIC.match(line) for line in completed.stdout.splitlines())
    return frozenset(int(match["line"]) for match in matches if match)


def test_every_static_mistake_is_an_error(error_lines: frozenset[int]) -> None:
    assert STATIC_ERRORS - error_lines == frozenset()


def test_only_static_mistakes_are_errors(error_lines: frozenset[int]) -> None:
    assert error_lines - STATIC_ERRORS == frozenset()


def test_known_misses_stay_silent(error_lines: frozenset[int]) -> None:
    assert marked_lines("known miss") & error_lines == frozenset()


def run_module(path: Path, kept: frozenset[int]) -> None:
    kept_lines = (line for number, line in enumerate(LINES, start=1) if number in kept)
    path.write_text("\n".join(kept_lines), encoding="utf-8")
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)


SILENT = frozenset(range(1, len(LINES) + 1)) - RAISING


def test_silent_lines_run(tmp_path: Path) -> None:
    run_module(tmp_path / "silent.py", SILENT)


@pytest.mark.parametrize("number", sorted(RAISING))
def test_every_runtime_mistake_raises(tmp_path: Path, number: int) -> None:
    with pytest.raises((TypeError, IndexError, AttributeError)):
        run_module(tmp_path / f"line_{number}.py", SILENT | {number})
