# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Pydantic models of ``config.yaml``: the single declaration of its structure.

A ``config.yaml`` maps the name of a study case to a :class:`StudyCaseModel`, which holds
:class:`MarketModel` s (with :class:`ProductModel` s) and a learning configuration. The models say

* which keys exist, which are required, which type they have and which default the loader applies,
* which formats are valid. The validators call the parsers the loader uses (``pd.Timestamp``,
  ``to_offset``, ``convert_to_rrule_freq``, ``pd.Timedelta``), so a value is valid exactly
  if the loader can read it,
* which combinations are valid (``end_date`` after ``start_date``, ...).

The models are strict: a yaml ``1`` is no ``str``, ``true`` is no number. A violation which a
validator reports itself is raised as :class:`~assume.common.unit_params.FieldValueError`, so that
its message is kept when the error is translated by the
:class:`~assume.scenario.validation.config.ConfigChecker`.

What a model cannot know stays in the checker: registries (market mechanisms), the file system
and unknown keys, which are warnings in a study case and a market but errors elsewhere.
:func:`config_json_schema` generates the JSON Schema of the file from these models.
"""

import math
import types
import typing
from dataclasses import fields
from functools import reduce
from operator import or_
from typing import Annotated, Any, ClassVar

import pandas as pd
import pydantic
from pydantic import AfterValidator, BeforeValidator, WithJsonSchema

from assume.common.base import LearningConfig
from assume.common.unit_params import FieldValueError
from assume.common.utils import convert_to_rrule_freq
from assume.scenario.validation.schema import FILE_KEYS

# ints stay ints and floats stay floats; a bool is no number
Number = int | float


# -- parsers shared with the loader -----------------------------------------------------------
def _field(info: pydantic.ValidationInfo) -> str:
    return str(info.field_name)


def parse_date(value: Any) -> pd.Timestamp:
    """Parse a date like the loader does; raises ValueError if it is none."""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        stamp = pd.NaT
    # null and "" give NaT instead of raising
    if pd.isna(stamp):
        raise ValueError(f"not a valid date: {value!r}")
    return stamp


def parse_time_step(value: Any) -> Any:
    """Parse a time step like the loader does (``pd.date_range``); raises ValueError if it is none."""
    try:
        return pd.tseries.frequencies.to_offset(value)
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"not a valid frequency: {value!r}") from None


def _date(value: Any, info: pydantic.ValidationInfo) -> pd.Timestamp:
    try:
        return parse_date(value)
    except ValueError:
        raise FieldValueError(
            _field(info), f"'{_field(info)}' is not a valid date: {value!r}"
        ) from None


def _optional_date(value: Any, info: pydantic.ValidationInfo) -> Any:
    # the loader falls back to the dates of the study case for a missing or empty date
    if value is not None:
        try:
            pd.Timestamp(value)
        except (ValueError, TypeError):
            raise FieldValueError(
                _field(info), f"'{_field(info)}' is not a valid date: {value!r}"
            ) from None
    return value


def _time_step(value: Any, info: pydantic.ValidationInfo) -> Any:
    try:
        return parse_time_step(value)
    except ValueError:
        raise FieldValueError(
            _field(info),
            f"'{_field(info)}' is not a valid frequency like '1h' or '15min': {value!r}",
        ) from None


def _duration(value: Any, info: pydantic.ValidationInfo) -> pd.Timedelta:
    try:
        return pd.Timedelta(value)
    except (ValueError, TypeError):
        raise FieldValueError(
            _field(info),
            f"'{_field(info)}' is not a valid duration like '1h' or '15min': {value!r}",
        ) from None


def _rrule_frequency(value: str, info: pydantic.ValidationInfo) -> str:
    try:
        convert_to_rrule_freq(value)
    except (ValueError, KeyError) as e:
        raise FieldValueError(_field(info), f"'{_field(info)}': {e}") from None
    return value


def _file_name(value: Any, info: pydantic.ValidationInfo) -> Any:
    if value is not None and not isinstance(value, str):
        raise FieldValueError(
            _field(info), f"'{_field(info)}' must be a file name or null, got {value!r}"
        )
    return value


def _count(value: Any, info: pydantic.ValidationInfo) -> Any:
    if not (isinstance(value, int) and value > 0):
        raise FieldValueError(
            _field(info), f"{_field(info)} must be a positive integer, got {value!r}"
        )
    return value


# fields whose value is parsed by pandas, which accepts more than the JSON Schema can express
Date = Annotated[
    Any,
    BeforeValidator(_date),
    WithJsonSchema({"description": "A date like '2019-01-01 00:00'."}),
]
OptionalDate = Annotated[
    Any,
    BeforeValidator(_optional_date),
    WithJsonSchema({"description": "A date like '2019-01-01 00:00' or null."}),
]
TimeStep = Annotated[
    Any,
    BeforeValidator(_time_step),
    WithJsonSchema(
        {"type": "string", "description": "A frequency like '1h' or '15min'."}
    ),
]
Duration = Annotated[
    Any,
    BeforeValidator(_duration),
    WithJsonSchema(
        {"type": "string", "description": "A duration like '1h' or '15min'."}
    ),
]
RruleFrequency = Annotated[
    str,
    AfterValidator(_rrule_frequency),
    WithJsonSchema(
        {"type": "string", "description": "A frequency like '1h', '24h', '1d' or '1w'."}
    ),
]
OpeningDuration = Annotated[
    str,
    AfterValidator(_duration),
    WithJsonSchema(
        {"type": "string", "description": "A duration like '1h' or '15min'."}
    ),
]
FileName = Annotated[str | None, BeforeValidator(_file_name)]
PositiveCount = Annotated[
    Any,
    BeforeValidator(_count),
    WithJsonSchema({"type": "integer", "minimum": 1}),
]


class ConfigModel(pydantic.BaseModel):
    """
    Base of the models of ``config.yaml``.

    Keys the model does not know are ignored. The wording of the two messages which depend on
    the kind of section can be adapted by the subclasses.
    """

    model_config = pydantic.ConfigDict(strict=True, extra="ignore")

    missing_wording: ClassVar[str] = "missing required key '{}'"
    unknown_wording: ClassVar[str] = "unknown key '{}'"


class ProductModel(ConfigModel):
    """A product of a market: what is traded (``duration``), how many (``count``) and when it is delivered."""

    model_config = pydantic.ConfigDict(extra="forbid")
    missing_wording: ClassVar[str] = "missing '{}'"
    unknown_wording: ClassVar[str] = "has unknown key '{}'"

    duration: Duration
    count: PositiveCount
    first_delivery: Duration


class MarketModel(ConfigModel):
    """
    The settings of one market. The defaults are those the loader applies to a key
    which is missing in ``config.yaml`` (they differ from the ones of ``MarketConfig``).
    """

    operator: str
    product_type: str = "energy"
    products: list[ProductModel]
    opening_frequency: RruleFrequency
    opening_duration: OpeningDuration
    market_mechanism: str
    start_date: OptionalDate = None
    end_date: OptionalDate = None
    maximum_bid_volume: Number | None = 1e6
    maximum_bid_price: Number | None = 3000
    minimum_bid_price: Number = -3000
    max_gradient: Number | None = None
    volume_unit: str = "MW"
    volume_tick: Number | None = None
    price_unit: str = "€/MWh"
    price_tick: Number | None = None
    additional_fields: list = pydantic.Field(default_factory=list)
    supports_get_unmatched: bool = False
    param_dict: dict = pydantic.Field(default_factory=dict)


def _learning_hint(hint: Any) -> Any:
    """An int or float annotation accepts any number (yaml does not tell ``1`` from ``1.0``)."""
    if hint in (int, float):
        return Number
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        return reduce(or_, (_learning_hint(arg) for arg in typing.get_args(hint)))
    return hint


def _continue_needs_load_path(value: Any, info: pydantic.ValidationInfo) -> Any:
    if info.data.get("continue_learning") and not value:
        raise FieldValueError(
            _field(info), "continue_learning needs trained_policies_load_path"
        )
    return value


def _learning_model() -> type[ConfigModel]:
    """
    Derive the model of ``learning_config`` from the :class:`~assume.common.base.LearningConfig`
    dataclass, so that it cannot drift from it.
    """
    hints = typing.get_type_hints(LearningConfig)
    model_fields: dict[str, Any] = {}
    for f in fields(LearningConfig):
        options: dict[str, Any] = {"default": f.default}
        if f.name == "trained_policies_load_path":
            options["validate_default"] = (
                True  # continue_learning needs it, even if absent
            )
        model_fields[f.name] = (
            _learning_hint(hints[f.name]),
            pydantic.Field(**options),
        )
    validators = {
        "_continue_needs_load_path": pydantic.field_validator(
            "trained_policies_load_path"
        )(_continue_needs_load_path)
    }
    base = type(
        "LearningModelBase",
        (ConfigModel,),
        {
            "model_config": pydantic.ConfigDict(extra="forbid"),
            "__doc__": "The settings of the reinforcement learning, see LearningConfig.",
        },
    )
    return pydantic.create_model(
        "LearningConfigModel", __base__=base, __validators__=validators, **model_fields
    )


LearningConfigModel = _learning_model()


def _moved_learning_mode(value: Any, info: pydantic.ValidationInfo) -> Any:
    raise FieldValueError(
        _field(info),
        "'learning_mode' moved into 'learning_config'. Please adjust your config",
    )


class _StudyCaseBase(ConfigModel):
    # time_step is declared before end_date, because the end is only compared to the start
    # if the time line can be built (the validators see the fields declared before them)
    start_date: Date
    time_step: TimeStep
    end_date: Date
    markets_config: dict[Any, MarketModel]

    seed: Number | None = 42
    simulation_id: str = None  # type: ignore[assignment]  # the loader derives it from the folder
    save_frequency_hours: Number | None = 48
    learning_config: LearningConfigModel = pydantic.Field(  # type: ignore[valid-type]
        default_factory=dict, validate_default=True
    )
    bidding_strategy_params: dict = pydantic.Field(default_factory=dict)
    forecast_algorithms: dict = pydantic.Field(default_factory=dict)
    use_forecasts_df: bool = True
    save_forecasts: bool = False
    forecast_save_file: str = "saved_forecasts.csv"
    learning_mode: Annotated[
        Any,
        BeforeValidator(_moved_learning_mode),
        WithJsonSchema(
            {"deprecated": True, "description": "Moved into learning_config."}
        ),
    ] = None

    @pydantic.field_validator("end_date")
    @classmethod
    def _after_start(cls, end: pd.Timestamp, info: pydantic.ValidationInfo) -> Any:
        start = info.data.get("start_date")
        if start is not None and "time_step" in info.data and end <= start:
            raise FieldValueError("end_date", "'end_date' must be after 'start_date'")
        return end

    @pydantic.field_validator("markets_config", mode="before")
    @classmethod
    def _is_mapping(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise FieldValueError(
                "markets_config",
                "'markets_config' must be a mapping of market ids to their settings",
            )
        return value

    @pydantic.field_validator("markets_config")
    @classmethod
    def _not_empty(cls, value: dict) -> dict:
        if not value:
            raise FieldValueError(
                "markets_config", "'markets_config' must contain at least one market"
            )
        return value

    @pydantic.field_validator("save_frequency_hours")
    @classmethod
    def _whole_hours(cls, value: Number | None) -> Number | None:
        if value is not None and not (
            math.isfinite(value) and value >= 0 and value == int(value)
        ):
            raise FieldValueError(
                "save_frequency_hours",
                f"'save_frequency_hours' must be null or a positive integer, got {value!r}",
            )
        return value


# The keys naming a csv file can be a file name or null; without them the loader reads
# ``<key>.csv``. The keys are those of ``schema`` that tables and folder need as well.
StudyCaseModel = pydantic.create_model(
    "StudyCaseModel",
    __base__=_StudyCaseBase,
    __doc__="One study case of ``config.yaml``.",
    **{key: (FileName, f"{key}.csv") for key in FILE_KEYS},
)


def config_json_schema() -> dict[str, Any]:
    """
    The JSON Schema of ``config.yaml``: a mapping of study case names to study cases.

    Returns:
        dict[str, Any]: A JSON serialisable schema, generated from the models of this module.
    """
    schema = pydantic.TypeAdapter(dict[str, StudyCaseModel]).json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ASSUME config.yaml"
    return schema
