# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import numpy as np
import pydantic
import pytest
from pydantic import Field

from assume.common.exceptions import ValidationError
from assume.common.unit_params import (
    Number,
    UnitInterval,
    UnitParams,
    describe_errors,
    validate_params,
)


class Rules(UnitParams):
    positive: Number = Field(default=1, gt=0)
    capped: Number = Field(default=0, le=10)
    below: Number = Field(default=0, lt=5)
    at_least: Number = Field(default=0, ge=2)
    fraction: UnitInterval = 0.5
    required: Number


def problems(**values):
    with pytest.raises(pydantic.ValidationError) as exc:
        Rules(**values)
    return dict(describe_errors(exc.value))


def test_describe_errors_covers_all_kinds():
    found = problems(
        positive=0, capped=11, below=5, at_least=1, fraction=1.5, extra_key=1
    )
    assert found == {
        "required": "required is required",
        "positive": "positive=0: Input should be greater than 0",
        "capped": "capped=11: Input should be less than or equal to 10",
        "below": "below=5: Input should be less than 5",
        "at_least": "at_least=1: Input should be greater than or equal to 2",
        "fraction": "fraction=1.5: must be between 0 and 1",
    }


def test_describe_errors_not_a_number_and_numpy_values():
    assert problems(required="x")["required"].startswith("required='x': ")
    assert (
        problems(required=1, capped=np.float64(11.5))["capped"]
        == "capped=11.5: Input should be less than or equal to 10"
    )


def test_empty_values_are_dropped_and_unknown_keys_ignored():
    rules = Rules(required=np.float64(1), capped=np.nan, below=None, other="x")
    assert (rules.capped, rules.below) == (0, 0)


def test_validate_params_raises_one_validation_error():
    with pytest.raises(ValidationError) as exc:
        validate_params(Rules, "u1", positive=-1, capped=11, required=1)
    assert exc.value.id == "u1"
    assert exc.value.field == "positive"
    assert str(exc.value) == (
        "positive=-1: Input should be greater than 0 for unit u1; capped=11: Input should be less than or equal to 10 for unit u1"
    )
