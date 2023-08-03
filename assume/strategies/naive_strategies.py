from datetime import datetime

from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMax
from assume.common.market_objects import MarketConfig, Product


class PowerPlantStrategy(BaseStrategy):
    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end_all)

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            marginal_cost = unit.calculate_marginal_cost(start, previous_power)
            volume = unit.calculate_ramp_up(
                previous_power, max_power[start], current_power
            )
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": volume,
                }
            )

            previous_power = volume + current_power

        return bids


class NaiveStrategy(PowerPlantStrategy):
    pass


class NaivePosReserveStrategy(PowerPlantStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(
            start, end_all, market_config.product_type
        )

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            volume = unit.calculate_ramp_up(
                previous_power, max_power[start], current_power
            )
            price = 0
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                }
            )
            previous_power = volume + current_power
        return bids


class NaiveNegReserveStrategy(PowerPlantStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(
            start, end_all, market_config.product_type
        )

        bids = []
        for product in product_tuples:
            start = product[0]
            previous_power = unit.get_output_before(start)
            current_power = unit.outputs["energy"].at[start]
            volume = unit.calculate_ramp_down(
                previous_power, min_power[start], current_power
            )
            price = 0
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                }
            )
            previous_power = volume + current_power
        return bids
