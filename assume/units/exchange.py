# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from assume.common.base import BaseUnit
from assume.common.forecaster import ExchangeForecaster


class Exchange(BaseUnit):
    """
    An exchange unit represents a unit that can import or export energy.

    Attributes:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        forecaster: The forecaster containing the import and export volume time series
        node (str, optional): The node of the unit. Defaults to "node0".
        price_import (float): The price for import of the unit.
        price_export (float): The price for export of the unit.
        location (tuple[float, float], optional): The location of the unit. Defaults to (0.0, 0.0).

    Methods
    -------
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: ExchangeForecaster,
        node: str = "node0",
        price_import: float = 0.0,
        price_export: float = 2999.0,
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology="exchange",
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        if not isinstance(forecaster, ExchangeForecaster):
            raise ValueError(
                f"forecaster must be of type {ExchangeForecaster.__name__}"
            )

        # import is positive
        self.volume_import = abs(forecaster.volume_import)
        # export is negative
        self.volume_export = -abs(forecaster.volume_export)

        self.price_import = price_import
        self.price_export = price_export

    def as_dict(self) -> dict:
        """
        Returns the unit as a dictionary.

        Returns:
            dict: The unit as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "exchange",
                "price_import": self.price_import,
                "price_export": self.price_export,
            }
        )

        return unit_dict
