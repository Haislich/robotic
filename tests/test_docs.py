"""Every code block in docs/ runs as written and prints what the page shows."""

import doctest
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"


@pytest.mark.parametrize("page", sorted(DOCS.glob("*.md")), ids=str)
def test_page_runs_as_written(page: Path) -> None:
    results = doctest.testfile(str(page), module_relative=False, verbose=False)
    assert results.attempted > 0
    assert results.failed == 0
