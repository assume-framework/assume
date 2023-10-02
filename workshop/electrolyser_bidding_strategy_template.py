from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product


class NaiveStrategyElectrolyser(BaseStrategy):
    """
    A naive strategy that bids the marginal cost of the electrolyser on the market.
    """

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

        :param unit: the unit to be dispatched
        :type unit: SupportsMinMax
        :param market_config: the market configuration
        :type market_config: MarketConfig
        :param product_tuples: list of all products the unit can offer
        :type product_tuples: list[Product]
        :return: the bids
        :rtype: Orderbook
        """
        start = product_tuples[0][0]  # start time of the first product

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal revenue of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """

            start = None  # Co-code
            end = None  # Co-code

            hydrogen_demand = unit.forecaster[f"{unit.id}_h2demand"].loc[start]
            hydrogen_price = unit.forecaster[f"{unit.id}_h2price"].loc[start]

            # Co code: Determine bid volume (Call the power)

            # Co code: Determine bid price

            order: Order = {
                "start_time": None,  # Co-code,
                "end_time": None,  # Co-code,
                "volume": None,  # Co-code,
                "price": None,  # Co-code,
            }

            bids.append(order)

        return bids
