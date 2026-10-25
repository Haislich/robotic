"""A parameters dataclass whose annotations are strings, as under PEP 563."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stringified:
    """One joint value, annotated with the string `"float"`."""

    q1: float
