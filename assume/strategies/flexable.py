import pandas as pd

from assume.common.market_objects import MarketConfig
from assume.strategies.base_strategy import BaseStrategy, OperationalWindow
from assume.units.base_unit import BaseUnit


class flexableEOM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains eom_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))
        self.current_time = None

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
        **kwargs,
    ):
        bid_quantity_inflex, bid_price_inflex = 0, 0
        bid_quantity_flex, bid_price_flex = 0, 0

        self.current_time = operational_window["window"][0]
        # =============================================================================
        # Powerplant is either on, or is able to turn on
        # Calculating possible bid amount
        # =============================================================================
        bid_quantity_inflex = operational_window["states"]["min_power"]["volume"]

        marginal_cost_mr = operational_window["states"]["min_power"]["cost"]
        marginal_cost_flex = operational_window["states"]["max_power"]["cost"]
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
            bid_quantity_flex = (
                operational_window["states"]["max_power"]["volume"] - bid_quantity_inflex
            )
            bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

        bids = [
            {"price": bid_price_inflex, "volume": bid_quantity_inflex},
            {"price": bid_price_flex, "volume": bid_quantity_flex},
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
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        self.current_time = operational_window["window"][0]

        bid_quantity = operational_window["states"]["pos_reserve"]["volume"]
        if bid_quantity == 0:
            return []

        marginal_cost = operational_window["states"]["pos_reserve"]["cost"]

        # Specific revenue if power was offered on the energy marke
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
            bids = [
                {"price": capacity_price, "volume": bid_quantity},
            ]
        elif market_config.product_type == "energy_pos":
            bids = [
                {"price": energy_price, "volume": bid_quantity},
            ]
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return bids


class flexableNegCRM(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        self.current_time = None

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        self.current_time = operational_window["window"][0]

        bid_quantity = operational_window["states"]["neg_reserve"]["volume"]
        if bid_quantity == 0:
            return []

        marginal_cost = operational_window["states"]["neg_reserve"]["cost"]

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
            bids = [
                {"price": capacity_price, "volume": bid_quantity},
            ]
        elif market_config.product_type == "energy_neg":
            bids = [
                {"price": energy_price, "volume": bid_quantity},
            ]
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return bids


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
