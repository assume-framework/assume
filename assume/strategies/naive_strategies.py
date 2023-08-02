from datetime import datetime

from assume.common.base import BaseStrategy, BaseUnit, SupportsMinMax
from assume.common.market_objects import MarketConfig, Product

# from assume.units import PowerPlant


class PowerPlantStrategy(BaseStrategy):
    def calculate_simple(
        self,
        unit: SupportsMinMax,
        start: datetime,
        end: datetime,
        **kwargs,
    ):
        min_power, max_power = unit.calculate_min_max_power(start, end)
        previous_power = unit.get_output_before(start)
        marginal_cost = unit.calculate_marginal_cost(start, previous_power)
        current_power = unit.outputs["energy"].at[start]
        max_power = unit.calculate_ramp_up(previous_power, max_power, current_power)

        return marginal_cost, max_power

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        bids = []
        for product in product_tuples:
            price, volume = self.calculate_simple(unit, product[0], product[1])
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                }
            )

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
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]
            previous_power = unit.get_output_before(start)
            min_power, max_power = unit.calculate_min_max_power(
                start, end, market_config.product_type
            )
            current_power = unit.outputs["energy"].at[start]
            volume = unit.calculate_ramp_up(previous_power, max_power, current_power)
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
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]
            previous_power = unit.get_output_before(start)
            min_power, max_power = unit.calculate_min_max_power(
                start, end, market_config.product_type
            )
            current_power = unit.outputs["energy"].at[start]
            volume = unit.calculate_ramp_down(previous_power, min_power, current_power)
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
        return bids
