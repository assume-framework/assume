# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import dateutil.rrule as rr

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.naive_strategies import NaiveSingleBidStrategy


class OTCStrategy(BaseStrategy):
    """
    Strategy for OTC (over the counter trading) markets
    """

    def __init__(self, *args, scale_firm_power_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale_firm_power_capacity

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and defines how it is dispatched to the market.

        Returns a list of bids that the unit operator will submit to the market.

        Args:
            unit (SupportsMinMax): Unit to dispatch.
            market_config (MarketConfig): Market configuration.
            product_tuples (list[Product]): List of products to dispatch.
            **kwargs (dict): Additional arguments.

        Returns:
            Orderbook: Orderbook.
        """
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]

            _, max_power = unit.calculate_min_max_power(
                start, end
            )  # max_power describes the maximum power output of the unit
            current_power = unit.outputs[
                "energy"
            ].at[
                start
            ]  # current power output describes the power output at the start of the product
            volume = max_power[0]
            if "OTC" in market_config.market_id:
                volume *= self.scale
            price = unit.calculate_marginal_cost(start, current_power + volume)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                    "node": unit.node,
                }
            )

        bids = self.remove_empty_bids(bids)

        return bids


def is_co2emissionless(units):
    requirement = lambda x: x in ["demand", "nuclear", "wind", "solar", "biomass"]
    return all([requirement(info["technology"]) for info in units])


class SupportStrategy(NaiveSingleBidStrategy):
    """
    Strategy for support markets.
    A list of allowed `contract_types` is given, as well as a value which is used to bid contracts.
    As often not everything which is possible should be bid on contracts, this can be defined through
    `contract_amount_fraction` - as well as an execution schedule.
    """

    def __init__(
        self,
        contract_types: list[str] = [],
        contract_value=0,
        contract_amount_fraction=1,
        evaluation_frequency=rr.WEEKLY,
        *args,
        **kwargs,
    ):
        """
        Init function of the support strategy.
        Pass a list of contract_types for which this strategy creates individual bids each.

        Args:
            contract_types (list[str], optional): List of contract types of available_contracts. Defaults to [].
            contract_amount_fraction (float, optional): a fraction of how much of the maximum capacity should be bid at max on this contract. Defaults to 1.
            contract_value (float, optional): the value used as a price for the given contract
            evaluation_frequency (int, optional): the evaluation frequency as dateutil FREQ
        """
        super().__init__(*args, **kwargs)
        self.contract_types = contract_types
        self.contract_amount_fraction = contract_amount_fraction
        self.contract_value = contract_value
        self.evaluation_frequency = evaluation_frequency
        self.eligible_lambda = is_co2emissionless

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): Unit to dispatch.
            market_config (MarketConfig): Market configuration.
            product_tuples (List[Product]): List of products to dispatch.
            **kwargs (dict): Additional arguments.

        Returns:
            Orderbook: The orderbook.
        """
        if "evaluation_frequency" not in market_config.additional_fields:
            # seems like we do not have a contract market here
            return super().calculate_bids(unit, market_config, product_tuples, **kwargs)

        bids = []
        # demand is the other way around
        # TODO should be generic without asking for Demand name
        power = unit.min_power if type(unit).__name__ == "Demand" else unit.max_power
        # bid a fraction as support for showcase reasons - not everything should be bid on contracts
        power *= self.contract_amount_fraction
        for product in product_tuples:
            start = product[0]
            end = product[1]
            current_power = unit.outputs["energy"].at[start]
            if not self.contract_value:
                price = unit.calculate_marginal_cost(start, current_power)
            else:
                price = self.contract_value
            price = min(price, market_config.maximum_bid_price)

            for contract_type in self.contract_types:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": product[2],
                        "price": price,
                        "volume": power,
                        "sender_id": unit.id,
                        "contract": contract_type,
                        # by default only bid on co2 emissionless contracts
                        "eligible_lambda": self.eligible_lambda,
                        # lambda u: u.technology in ["nuclear"],
                        "evaluation_frequency": self.evaluation_frequency,
                        "node": unit.node,
                    }
                )

        return bids


class MarkupStrategy(BaseStrategy):
    """
    Strategy for Markup (over the counter trading) markets
    """

    def __init__(self, *args, abs_markup=0, rel_markup=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.abs_markup = abs_markup
        self.rel_markup = rel_markup

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Returns a list of bids that the unit operator will submit to the market.

        Args:
            unit (SupportsMinMax): Unit to dispatch.
            market_config (MarketConfig): Market configuration.
            product_tuples (list[Product]): List of products to dispatch.
            **kwargs (dict): Additional arguments.

        Returns:
            Orderbook: The orderbook.
        """
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]

            # max_power describes the maximum power output of the unit
            _, max_power = unit.calculate_min_max_power(start, end)
            current_power = unit.outputs[
                "energy"
            ].at[
                start
            ]  # current power output describes the power output at the start of the product
            volume = max_power[0]
            price = unit.calculate_marginal_cost(start, current_power + volume)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product[2],
                    "price": price * self.rel_markup + self.abs_markup,
                    "volume": volume,
                    "node": unit.node,
                }
            )

        bids = self.remove_empty_bids(bids)

        return bids
