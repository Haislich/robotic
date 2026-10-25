"""No public signature or annotation of the library names a sympy type."""

import annotationlib
import inspect
import typing
from typing import TYPE_CHECKING, TypeAliasType

import robotic

if TYPE_CHECKING:
    from collections.abc import Iterator


def public_members(owner: type) -> Iterator[object]:
    for name, member in vars(owner).items():
        if not name.startswith("_") or (name.startswith("__") and name.endswith("__")):
            match member:
                case property():
                    yield member.fget
                case classmethod() | staticmethod():
                    yield member.__func__
                case _:
                    yield member


def public_objects() -> Iterator[object]:
    for name in robotic.__all__:
        exported = getattr(robotic, name)
        yield exported
        if inspect.isclass(exported):
            for owner in exported.__mro__:
                if owner.__module__.startswith("robotic"):
                    yield from public_members(owner)


def annotations_of(target: object) -> Iterator[object]:
    match target:
        case TypeAliasType():
            yield target.__value__
        case _ if callable(target) or inspect.isclass(target):
            try:
                found = annotationlib.get_annotations(
                    target, format=annotationlib.Format.FORWARDREF
                )
            except TypeError:
                return
            yield from (
                annotation
                for name, annotation in found.items()
                if not name.startswith("_")
            )


def mentions_sympy(annotation: object) -> bool:
    match annotation:
        case annotationlib.ForwardRef():
            return "sympy" in annotation.__forward_arg__
        case TypeAliasType():
            return mentions_sympy(annotation.__value__)
        case _:
            module = str(getattr(annotation, "__module__", ""))
            arguments = typing.get_args(annotation)
            origin = typing.get_origin(annotation)
            return (
                module.startswith("sympy")
                or any(mentions_sympy(argument) for argument in arguments)
                or (origin is not None and mentions_sympy(origin))
            )


def test_public_signatures_are_sympy_free() -> None:
    offenders = [
        (target, annotation)
        for target in public_objects()
        for annotation in annotations_of(target)
        if mentions_sympy(annotation)
    ]
    assert offenders == []
