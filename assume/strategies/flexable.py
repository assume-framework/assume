import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Product


class flexableEOM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains eom_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))
        self.current_time = None

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        # TODO only works for single bids for now
        # should work with all other bids too
        start = product_tuples[0][0]
        end = product_tuples[0][1]

        # not adjusted for ramp up speed
        min_power, max_power = unit.calculate_min_max_power(start, end)
        previous_power = unit.get_output_before(start)
        current_power = unit.outputs["energy"].at[start]

        # adjust for ramp down speed
        max_power = unit.calculate_ramp_up(previous_power, max_power, current_power)
        # adjust for ramp up speed
        min_power = unit.calculate_ramp_down(previous_power, min_power, current_power)

        bid_quantity_inflex = min_power

        # =============================================================================
        # Powerplant is either on, or is able to turn on
        # Calculating possible bid amount and cost
        # =============================================================================

        marginal_cost_mr = unit.calculate_marginal_cost(
            start, current_power + bid_quantity_inflex
        )
        marginal_cost_flex = unit.calculate_marginal_cost(
            start, current_power + max_power
        )
        self.current_time = start

        # =============================================================================
        # Calculating possible price
        # =============================================================================
        if unit.current_status:
            bid_price_inflex = self.calculate_EOM_price_if_on(
                unit, marginal_cost_mr, bid_quantity_inflex
            )
        else:
            bid_price_inflex = self.calculate_EOM_price_if_off(
                unit, marginal_cost_flex, bid_quantity_inflex
            )

        if unit.outputs["heat"][self.current_time] > 0:
            power_loss_ratio = (
                unit.outputs["power_loss"][self.current_time]
                / unit.outputs["heat"][self.current_time]
            )
        else:
            power_loss_ratio = 0.0

        # Flex-bid price formulation
        if unit.current_status:
            bid_quantity_flex = max_power - bid_quantity_inflex
            bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

        bids = [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price_inflex,
                "volume": bid_quantity_inflex,
            },
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price_flex,
                "volume": bid_quantity_flex,
            },
        ]

        return bids

    def calculate_EOM_price_if_off(self, unit, marginal_cost_mr, bid_quantity_inflex):
        # The powerplant is currently off and calculates a startup markup as an extra
        # to the marginal cost
        # Calculating the average uninterrupted operating period
        av_operating_time = max(
            unit.mean_market_success, unit.min_operating_time, 1
        )  # 1 prevents division by 0

        starting_cost = self.get_starting_costs(time=unit.current_down_time, unit=unit)
        if bid_quantity_inflex == 0:
            markup = starting_cost / av_operating_time
        else:
            markup = starting_cost / av_operating_time / bid_quantity_inflex

        bid_price_inflex = min(marginal_cost_mr + markup, 3000.0)

        return bid_price_inflex

    def calculate_EOM_price_if_on(self, unit, marginal_cost_flex, bid_quantity_inflex):
        """
        Check the description provided by Thomas in last version, the average downtime is not available
        """
        if bid_quantity_inflex == 0:
            return 0

        t = self.current_time
        min_down_time = max(unit.min_down_time, 1)

        starting_cost = self.get_starting_costs(time=min_down_time, unit=unit)

        price_reduction_restart = starting_cost / min_down_time / bid_quantity_inflex

        if unit.outputs["heat"][t] > 0:
            heat_gen_cost = (
                unit.outputs["heat"][t] * (unit.fuel_price["natural gas"][t] / 0.9)
            ) / bid_quantity_inflex
        else:
            heat_gen_cost = 0.0

        possible_revenue = get_specific_revenue(
            unit=unit,
            marginal_cost=marginal_cost_flex,
            current_time=self.current_time,
            foresight=self.foresight,
        )
        if possible_revenue >= 0 and unit.price_forecast[t] < marginal_cost_flex:
            marginal_cost_flex = 0

        bid_price_inflex = max(
            -price_reduction_restart - heat_gen_cost + marginal_cost_flex,
            -499.00,
        )

        return bid_price_inflex

    def get_starting_costs(self, time, unit):
        if time < unit.downtime_hot_start:
            return unit.hot_start_cost

        elif time < unit.downtime_warm_start:
            return unit.warm_start_cost

        else:
            return unit.cold_start_cost


class flexablePosCRM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        self.current_time = None

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        start = product_tuples[0][0]
        end = product_tuples[0][1]
        self.current_time = start
        min_power, max_power = unit.calculate_min_max_power(start, end)
        marginal_cost = 0

        previous_power = unit.get_output_before(start)
        # calculate pos reserve volume
        current_power = unit.outputs["energy"].at[start]
        # max_power + current_power < previous_power + unit.ramp_up
        bid_quantity = unit.calculate_ramp_up(previous_power, max_power, current_power)

        if bid_quantity == 0:
            return []

        # Specific revenue if power was offered on the energy market
        specific_revenue = get_specific_revenue(
            unit=unit,
            marginal_cost=marginal_cost,
            current_time=self.current_time,
            foresight=self.foresight,
        )

        if specific_revenue >= 0:
            capacity_price = specific_revenue
        else:
            capacity_price = abs(specific_revenue) * unit.min_power / bid_quantity

        energy_price = marginal_cost

        if market_config.product_type == "capacity_pos":
            price = capacity_price
        elif market_config.product_type == "energy_pos":
            price = energy_price
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": price,
                "volume": bid_quantity,
            },
        ]


class flexableNegCRM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        self.current_time = None

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ):
        start = product_tuples[0][0]
        end = product_tuples[0][1]
        self.current_time = start
        min_power, max_power = unit.calculate_min_max_power(start, end)
        marginal_cost = 0

        previous_power = unit.get_output_before(start)
        current_power = unit.outputs["energy"].at[start]

        # min_power + current_power > previous_power - unit.ramp_down
        bid_quantity = unit.calculate_ramp_down(
            previous_power, min_power, current_power
        )
        if bid_quantity == 0:
            return []

        marginal_cost = unit.calculate_marginal_cost(start, previous_power - min_power)

        # Specific revenue if power was offered on the energy marke
        specific_revenue = get_specific_revenue(
            unit=unit,
            marginal_cost=marginal_cost,
            current_time=self.current_time,
            foresight=self.foresight,
        )

        if specific_revenue < 0:
            capacity_price = (
                abs(specific_revenue) * (unit.min_power + bid_quantity) / bid_quantity
            )
        else:
            capacity_price = 0.0

        energy_price = marginal_cost * (-1)

        if market_config.product_type == "capacity_neg":
            price = capacity_price
        elif market_config.product_type == "energy_neg":
            price = energy_price
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": price,
                "volume": bid_quantity,
            },
        ]


def get_specific_revenue(
    unit,
    marginal_cost,
    current_time,
    foresight,
):
    t = current_time
    price_forecast = []

    if t + foresight > unit.price_forecast.index[-1]:
        price_forecast = unit.price_forecast.loc[t:]
    else:
        price_forecast = unit.price_forecast.loc[t : t + foresight]

    possible_revenue = (price_forecast - marginal_cost).sum()

    return possible_revenue
