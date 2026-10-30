# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import difflib
import io
import logging
import types
import typing
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, TypeAlias

MAX_LISTED = 5

# A value read by ``yaml.safe_load``. Its type is not known in advance: finding it out is the
# purpose of the checks, which narrow it with ``isinstance``.
YamlValue: TypeAlias = object
# The settings of a study case (or a section of it) in ``config.yaml``
YamlMapping: TypeAlias = dict[str, YamlValue]
# A type hint like ``int``, ``float | None`` or ``dict[str, float]``, as returned by ``get_type_hints``
Annotation: TypeAlias = object


def type_name(annotation: Annotation) -> str:
    """Human readable name of an annotation, where ``int`` and ``float`` are both "number"."""
    if annotation is None or annotation is type(None):
        return "null"
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) in (
        typing.Union,
    ):
        return " | ".join(type_name(a) for a in typing.get_args(annotation))
    if annotation in (int, float):
        return "number"
    return getattr(annotation, "__name__", str(annotation))


def type_ok(value: YamlValue, annotation: Annotation) -> bool:
    """Check a yaml value against an annotation, where ``int`` and ``float`` both mean "number"."""
    if annotation is None or annotation is Any:
        return True
    if annotation is type(None):
        return value is None
    origin = typing.get_origin(annotation)
    if isinstance(annotation, types.UnionType) or origin is typing.Union:
        return any(type_ok(value, a) for a in typing.get_args(annotation))
    if annotation in (int, float):
        return isinstance(value, int | float) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    if annotation is bool:
        return isinstance(value, bool)
    target = origin or annotation
    if isinstance(target, type):
        return isinstance(value, target)
    return True  # callables, protocols, ... cannot be expressed in yaml anyway


def describe_wrong_type(key: str, value: YamlValue, annotation: Annotation) -> str:
    return f"'{key}' must be {type_name(annotation)}, got {value!r} ({type(value).__name__})"


def hint(name: object, known: Iterable[object]) -> str:
    """Return " (did you mean 'x'?)" if ``name`` looks like a typo of something in ``known``."""
    close = difflib.get_close_matches(str(name), [str(k) for k in known], n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def listed(items: Iterable[object]) -> str:
    """Join items for a message, shortened to the first few."""
    names = [str(i) for i in items]
    more = f" and {len(names) - MAX_LISTED} more" if len(names) > MAX_LISTED else ""
    return ", ".join(names[:MAX_LISTED]) + more


@contextmanager
def quiet_logging() -> Iterator[None]:
    """Silence log output and prints (the loader and its libraries are chatty) while verifying."""
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(logging.NOTSET)
