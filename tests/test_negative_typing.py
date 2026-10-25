"""The negative type test: the fixture's mistakes are ty errors, nothing else is."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).with_name("type_mistakes.py.txt")
PROJECT = Path(__file__).parents[1]
DIAGNOSTIC = re.compile(r"^.+?:(?P<line>\d+):\d+: error\[")


def marked_lines(marker: str) -> frozenset[int]:
    lines = FIXTURE.read_text().splitlines()
    return frozenset(
        number for number, line in enumerate(lines, start=1) if marker in line
    )


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


def test_every_mistake_is_an_error(error_lines: frozenset[int]) -> None:
    assert marked_lines("  # mistake:") - error_lines == frozenset()


def test_only_mistakes_are_errors(error_lines: frozenset[int]) -> None:
    assert error_lines - marked_lines("  # mistake:") == frozenset()


def test_known_misses_stay_silent(error_lines: frozenset[int]) -> None:
    assert marked_lines("  # known miss:") & error_lines == frozenset()
