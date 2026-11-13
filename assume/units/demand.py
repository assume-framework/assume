# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import ValidationInfo, field_validator, model_validator

from assume.common.base import SupportsMinMax
from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import DemandForecaster
from assume.common.unit_params import (
    FieldValueError,
    Number,
    UnitParams,
    validate_params,
)


class DemandParams(UnitParams):
    """
    The numeric and typed parameters of a :class:`Demand`, with their ranges.

    The elastic demand parameters are no named arguments of the constructor, they arrive as
    keyword arguments. ``max_price`` defaults to ``price``.
    """

    max_power: Number
    min_power: Number
    price: Number = 3000.0
    max_price: Number | None = None
    elasticity: Number = 0
    elasticity_model: str | Number = 0
    num_bids: int = 0

    @field_validator("max_power", "min_power")
    @classmethod
    def _not_positive(cls, value: Number, info: ValidationInfo) -> Number:
        # the message says "< 0" although 0 is allowed, as it always did
        if value > 0:
            raise FieldValueError(
                info.field_name, f"{info.field_name} must be <= 0 but is {value}"
            )
        return value

    @field_validator("num_bids", mode="before")
    @classmethod
    def _truncate_num_bids(cls, value: Any) -> Any:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value  # reported as "must be a number"

    @model_validator(mode="after")
    def _check_power_range(self):
        if self.max_power > self.min_power:
            raise FieldValueError(
                "max_power",
                f"max_power={self.max_power!r} must be <= min_power={self.min_power!r}",
            )
        return self

    @model_validator(mode="after")
    def _check_elastic_demand(self):
        if self.max_price is None:
            self.max_price = self.price
        if self.elasticity_model == 0:
            return self
        if self.elasticity_model not in ("linear", "isoelastic"):
            raise FieldValueError(
                "elasticity_model",
                f"Invalid elasticity_model '{self.elasticity_model}'. Choose 'linear' or 'isoelastic'.",
            )
        # the message says ">= 1" although 1 is rejected, as it always did
        if self.num_bids <= 1:
            raise FieldValueError(
                "num_bids", "'num_bids' parameter must be > 1 for elastic demand"
            )
        if self.elasticity_model == "isoelastic" and self.elasticity >= 0.0:
            raise FieldValueError(
                "elasticity",
                "'elasticity' parameter must be given and negative for isoelastic demand",
            )
        if self.elasticity_model == "linear":
            max_power = max(abs(self.min_power), abs(self.max_power))
            if max_power == 0 or (self.max_price / max_power <= 0.0):
                raise FieldValueError(
                    "max_price",
                    "Invalid slope of demand curve. Slope must be negative for linear demand. Set 'max_price' positive.",
                )
        return self


class Demand(SupportsMinMax):
    """
    A demand unit.

    Attributes:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        index (pandas.DatetimeIndex): The index of the unit.
        max_power (float): The maximum power input (negative) capacity of the power plant in MW.
        min_power (float, optional): The minimum power input (negative) capacity of the power plant in MW. Defaults to 0.0 MW.
        node (str, optional): The node of the unit. Defaults to "node0".
        price (float): The bidding price of the unit.
        location (tuple[float, float]): Geographical location.
        elasticity, elasticity_model, max_price, num_bids: Optional parameters for elastic demand modeling.
    """

    Params = DemandParams

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        max_power: float,
        min_power: float,
        forecaster: DemandForecaster,
        node: str = "node0",
        price: float = 3000.0,
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        if not isinstance(forecaster, DemandForecaster):
            raise ValueError(f"forecaster must be of type {DemandForecaster.__name__}")

        params = validate_params(
            self.Params,
            self.id,
            max_power=max_power,
            min_power=min_power,
            price=price,
            **kwargs,
        )

        self.max_power = params.max_power
        self.min_power = params.min_power

        self.price = FastSeries(index=self.index, value=params.price)

        # Elastic demand parameters
        self.max_price = params.max_price
        self.elasticity = params.elasticity
        self.elasticity_model = params.elasticity_model
        self.num_bids = params.num_bids

    def execute_current_dispatch(
        self,
        start: datetime,
        end: datetime,
    ) -> np.ndarray:
        """
        Execute the current dispatch of the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.

        Returns:
            np.ndarray: The volume of the unit for the given time range.
        """

        return self.outputs["energy"].loc[start:end]

    def calculate_min_max_power(
        self, start: datetime, end: datetime, product_type="energy"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the minimum and maximum power output of the unit and returns the bid volume as both the minimum and maximum power output of the unit.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            end (pandas.Timestamp): The end time of the dispatch (exclusive).
            product_type (str, optional): The product type. Defaults to "energy".

        Returns:
            tuple[np.ndarray, np.ndarray]: The bid volume as both the minimum and maximum power output of the unit.
        """

        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq
        bid_volume = (
            self.forecaster.demand.loc[start:end_excl]
            - self.outputs[product_type].loc[start:end_excl]
        )

        return bid_volume, bid_volume

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculate the marginal cost of the unit returns the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        return self.price.at[start]

    def calculate_generation_cost(
        self, start: datetime, end: datetime, product_type: str
    ) -> None:
        """
        Demand has no generation costs, the procurement is part of the cashflow.

        Args:
            start (datetime.datetime): The start time for the calculation.
            end (datetime.datetime): The end time for the calculation.
            product_type (str): The type of product.
        """
        self.outputs[f"{product_type}_generation_costs"].loc[start:end] = 0

    def as_dict(self) -> dict:
        """
        Returns the unit as a dictionary.

        Returns:
            dict: The unit as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "unit_type": "demand",
            }
        )

        return unit_dict
