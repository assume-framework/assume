# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

from assume.common import UnitsOperator
from assume.common.market_objects import (
    MarketConfig,
)
from assume.strategies import UnitOperatorStrategy
from assume.units import BaseUnit

logger = logging.getLogger(__name__)


class RLUnitsOperator(UnitsOperator):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        portfolio_strategies: dict[str, UnitOperatorStrategy] = {},
    ):
        super().__init__(available_markets, portfolio_strategies)

    def on_ready(self):
        super().on_ready()

    def add_unit(
        self,
        unit: BaseUnit,
    ) -> None:
        """
        Create a unit.

        Args:
            unit (BaseUnit): The unit to be added.
        """
        self.units[unit.id] = unit
