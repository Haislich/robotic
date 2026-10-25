"""Format, lint and type-check Python files, reporting diagnostics only on failure.

Runs as a Claude Code ``PostToolUse`` hook, reading the hook payload from stdin,
or as a pre-commit hook, reading file paths from the command line. Exits with
``BLOCKING_EXIT_CODE`` when any check fails, which Claude Code feeds back to the
model and pre-commit treats as a failed hook.
"""

import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

BLOCKING_EXIT_CODE = 2
TY_SUPPRESSION = re.compile(r"\bty:\s*ignore\b")


def python_files(candidates: Iterable[str]) -> tuple[Path, ...]:
    """Select the existing Python source files among candidate paths.

    Args:
        candidates: Paths as received from the caller, possibly not Python files.

    Returns:
        The candidates that are existing ``.py`` files, in input order.
    """
    return tuple(
        path
        for path in map(Path, candidates)
        if path.suffix == ".py" and path.is_file()
    )


def hook_payload_paths(payload: str) -> tuple[str, ...]:
    """Extract the edited file path from a Claude Code hook payload.

    Args:
        payload: The JSON document Claude Code writes to the hook's stdin.

    Returns:
        The edited file path, or nothing when the tool input carries none.
    """
    file_path = json.loads(payload).get("tool_input", {}).get("file_path")
    return (file_path,) if isinstance(file_path, str) else ()


def run_tool(arguments: Iterable[str | Path]) -> list[str]:
    """Run a Python tool module from the project environment.

    Args:
        arguments: Module name followed by its command-line arguments.

    Returns:
        The tool's output lines when it fails, or nothing when it succeeds.
    """
    completed = subprocess.run(
        [sys.executable, "-m", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return []
    return (completed.stdout + completed.stderr).splitlines()


def ty_suppressions(path: Path) -> list[str]:
    """Find ``ty: ignore`` comments, which ty offers no setting to disregard.

    Files that cannot be tokenized yield nothing: the syntax error itself is
    reported by ruff and ty.

    Args:
        path: Python source file to scan.

    Returns:
        One diagnostic line per suppression comment.
    """
    try:
        with tokenize.open(path) as source:
            tokens = tuple(tokenize.generate_tokens(source.readline))
    except tokenize.TokenError, SyntaxError:
        return []
    return [
        f"{path}:{token.start[0]}:{token.start[1] + 1}: inline ty suppression "
        "is banned; fix the cause instead"
        for token in tokens
        if token.type == tokenize.COMMENT and TY_SUPPRESSION.search(token.string)
    ]


def diagnostics(paths: tuple[Path, ...]) -> list[str]:
    """Format the files in place, then collect every lint and type diagnostic.

    Args:
        paths: Python source files to check.

    Returns:
        All diagnostic lines; empty when every check passes.
    """
    formatting = run_tool(["ruff", "format", "--quiet", "--force-exclude", *paths])
    linting = run_tool(
        [
            "ruff",
            "check",
            "--quiet",
            "--force-exclude",
            "--ignore-noqa",
            "--output-format",
            "concise",
            *paths,
        ]
    )
    typing = run_tool(["ty", "check", "--quiet", "--output-format", "concise", *paths])
    suppressions = [line for path in paths for line in ty_suppressions(path)]
    return formatting + linting + typing + suppressions


def main() -> int:
    """Check the files named on the command line, or the file in the hook payload.

    Returns:
        Zero when every check passes, ``BLOCKING_EXIT_CODE`` otherwise.
    """
    candidates = sys.argv[1:] or hook_payload_paths(sys.stdin.read())
    paths = python_files(candidates)
    if not paths:
        return 0
    found = diagnostics(paths)
    if not found:
        return 0
    sys.stderr.write("\n".join(found) + "\n")
    return BLOCKING_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
