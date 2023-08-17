from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product


class NaiveStrategy(BaseStrategy):
    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end_all)

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            marginal_cost = unit.calculate_marginal_cost(start, previous_power)
            volume = unit.calculate_ramp(
                previous_power, max_power[start], current_power
            )
            order: Order = {
                "start_time": product[0],
                "end_time": product[1],
                "only_hours": product[2],
                "price": marginal_cost,
                "volume": volume,
            }
            order.update({field: None for field in market_config.additional_fields})
            bids.append(order)

            if "bid_type" in market_config.additional_fields:
                bids[-1]["bid_type"] = "SB"

            previous_power = volume + current_power

        return bids
    
class NaiveDAStrategy(BaseStrategy):
    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        
        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end_all)

        bids = []
        current_power = unit.outputs["energy"].at[start]
        marginal_cost = unit.calculate_marginal_cost(start, previous_power)
        volume = unit.calculate_ramp(
            previous_power, max_power[start], current_power
        )
        bids.append(
            {
                "start_time": start,
                "end_time": product_tuples[0][1],
                "only_hours": product_tuples[0][2],
                "price": marginal_cost,
                "profile": {product[0]: volume for product in product_tuples},
            }
        )

        return bids


class NaivePosReserveStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            volume = unit.calculate_ramp(
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


class NaiveNegReserveStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            volume = unit.calculate_ramp(
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
