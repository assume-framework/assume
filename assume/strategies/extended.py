from assume.common.market_objects import MarketConfig, Product
from assume.strategies.base_strategy import BaseStrategy
from assume.units.base_unit import SupportsMinMax


class OTCStrategy(BaseStrategy):
    def __init__(self, *args, scale_firm_power_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale_firm_power_capacity

    def calculate_bids(
        self,
        unit: SupportsMinMax,
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
        for product_tuple in product_tuples:
            start = product_tuple[0]
            end = product_tuple[1]

            min_power, max_power = unit.calculate_min_max_power(start, end)
            current_power = unit.outputs["energy"].at[start]
            min_price = unit.calculate_marginal_cost(start, current_power - min_power)
            max_price = unit.calculate_marginal_cost(start, current_power + max_power)

            price = max_price
            volume = max_power
            if "OTC" in market_config.name:
                volume *= self.scale
            bids.append({"price": price, "volume": volume})
        return bids
