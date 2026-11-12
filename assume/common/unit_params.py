# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Base for pydantic parameter models of units.

A unit class can expose a model of its plain numeric/typed constructor parameters as the class
attribute ``Params``. The model is the single place for range and consistency rules:

- the unit constructor builds it with :func:`validate_params` and raises one
  :class:`~assume.common.exceptions.ValidationError` listing all violations,
- the scenario verifier validates every csv row with it (see :func:`describe_errors`).
"""

import math
import numbers
from typing import Annotated, Any, TypeVar

import numpy as np
import pandas as pd
import pydantic

from assume.common.exceptions import ValidationError

P = TypeVar("P", bound="UnitParams")


# ints stay ints and floats stay floats, as the constructors always accepted both
Number = int | float


# a constant or a time series, as accepted by FastSeries
NumberOrSeries = Number | pd.Series | np.ndarray | list


def _unit_interval(value: Number) -> Number:
    if not 0 <= value <= 1:
        raise ValueError("must be between 0 and 1")
    return value


# a fraction, e.g. an efficiency or a state of charge
UnitInterval = Annotated[Number, pydantic.AfterValidator(_unit_interval)]


class FieldValueError(ValueError):
    """
    Raised by cross-field validators of a :class:`UnitParams` model.

    Args:
        field (str): The parameter which is reported as the offending one.
        message (str): What is wrong, naming all involved parameters, e.g. ``min_power=20 must be <= max_power=10``.
    """

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message


class UnitParams(pydantic.BaseModel):
    """
    Base class of the parameter models of units.

    Empty values (``None`` or NaN, as pandas delivers empty csv cells) are dropped before
    validation, so the default of the field applies. Keys the model does not know are ignored,
    because units forward further keyword arguments to their base classes.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _drop_empty(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {
            key: value
            for key, value in data.items()
            if not (
                value is None or (isinstance(value, numbers.Real) and math.isnan(value))
            )
        }


def _shown(value: Any) -> Any:
    """Unwrap numpy scalars so that messages read ``-10.0`` instead of ``np.float64(-10.0)``."""
    return value.item() if isinstance(value, np.generic) else value


def describe_errors(error: pydantic.ValidationError) -> list[tuple[str, str]]:
    """
    Turn a pydantic error into ``(field, problem)`` pairs, e.g. ``("max_power", "max_power=-10 must be >= 0")``.

    Args:
        error (pydantic.ValidationError): The error raised by a :class:`UnitParams` model.

    Returns:
        list[tuple[str, str]]: One entry per violation, in the order of the model's fields.
    """
    problems = []
    seen = set()
    for err in error.errors():
        cause = err.get("ctx", {}).get("error")
        if isinstance(cause, FieldValueError):
            problems.append((cause.field, cause.message))
            continue
        field = str(err["loc"][0]) if err["loc"] else ""
        if field in seen:  # unions report once per member type
            continue
        seen.add(field)
        if err["type"] == "missing":
            problems.append((field, f"{field} is required"))
        else:
            detail = cause if cause is not None else err["msg"]
            problems.append((field, f"{field}={_shown(err['input'])!r}: {detail}"))
    return problems


def validate_params(model: type[P], unit_id: str, **values: Any) -> P:
    """
    Build a parameter model and convert pydantic errors to a :class:`ValidationError`.

    Args:
        model (type[UnitParams]): The parameter model of the unit.
        unit_id (str): The id of the unit, used in the messages.
        **values: The constructor arguments of the unit.

    Returns:
        UnitParams: The validated parameters.

    Raises:
        ValidationError: With ``id`` set to ``unit_id``, ``field`` set to the first offending
            parameter and a message listing all violations.
    """
    try:
        return model(**values)
    except pydantic.ValidationError as e:
        problems = describe_errors(e)
        raise ValidationError(
            message="; ".join(
                f"{problem} for unit {unit_id}" for _, problem in problems
            ),
            id=unit_id,
            field=problems[0][0],
        ) from None
