import pandas as pd

from assume.strategies import BaseStrategy
from assume.units.base_unit import BaseUnit


class PowerPlant(BaseUnit):
    """A class for a power plants.

    Attributes
    ----------
    id : str
        The ID of the power plant.
    technology : str
        The technology of the power plant.
    node : str
        The node of the power plant.
    max_power : float
        The maximum power output of the power plant in MW.
    min_power : float
        The minimum power output of the power plant in MW.
    efficiency : float
        The efficiency of the power plant.
    fuel_type : str
        The fuel type of the power plant.
    fuel_price : list
        The fuel type specific fuel price in €/MWh.
    co2_price : list
        The co2 price in €/t.
    emission_factor : float
        The emission factor of the power plant.
    ramp_up : float, optional
        The ramp up rate of the power plant in MW/15 minutes.
    ramp_down : float, optional
        The ramp down rate of the power plant in MW/15 minutes.
    fixed_cost : float, optional
        The fixed cost of the power plant in €/MW.
    hot_start_cost : float, optional
        The hot start cost of the power plant in €/MW.
    warm_start_cost : float, optional
        The warm start cost of the power plant in €/MW.
    cold_start_cost : float, optional
        The cold start cost of the power plant in €/MW.
    min_operating_time : float, optional
        The minimum operating time of the power plant in hours.
    min_down_time : float, optional
        The minimum down time of the power plant in hours.
    heat_extraction : bool, optional
        If the power plant can extract heat.
    max_heat_extraction : float, optional
        The maximum heat extraction of the power plant in MW.
    availability : dict, optional
        The availability of the power plant in MW for each time step.
    is_active: bool
        Defines whether or not the unit bids itself or is portfolio optimized.
    bidding_startegy: str
        In case the unit is active it has to be defined which bidding strategy should be used
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    reset()
        Reset the power plant.
    calculate_operational_window()
        Calculate the operation window for the next time step.
    calc_marginal_cost(power_output, partial_load_eff)
        Calculate the marginal cost of the power plant.
    """

    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        max_power: float or pd.Series,
        min_power: float or pd.Series = 0.0,
        efficiency: float = 1,
        fuel_type: str = None,
        fuel_price: float or pd.Series = 0.0,
        co2_price: float or pd.Series = 0.0,
        price_forecast: pd.Series = None,
        emission_factor: float = 0.0,
        ramp_up: float = -1,
        ramp_down: float = -1,
        fixed_cost: float = 0,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        heat_extraction: bool = False,
        max_heat_extraction: float = 0,
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = None,
        node: str = None,
        **kwargs
    ):
        super().__init__(
            id=id,
            technology=technology,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
        )

        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.fuel_type = fuel_type
        self.fuel_price = fuel_price
        self.co2_price = co2_price
        self.price_forecast = (
            price_forecast if price_forecast is not None else pd.Series(0, index=index)
        )
        self.emission_factor = emission_factor

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time if min_operating_time > 0 else 1
        self.min_down_time = min_down_time if min_down_time > 0 else 1
        self.downtime_hot_start = downtime_hot_start
        self.downtime_warm_start = downtime_warm_start

        self.fixed_cost = fixed_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.location = location

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_power_output = pd.Series(0.0, index=self.index)
        self.total_power_output.iloc[:2] = self.min_power

        self.total_heat_output = pd.Series(0.0, index=self.index)
        self.power_loss_chp = pd.Series(0.0, index=self.index)

        self.pos_capacity_reserve = pd.Series(0.0, index=self.index)
        self.neg_capacity_reserve = pd.Series(0.0, index=self.index)

        self.mean_market_success = 0
        self.market_success_list = [0]

    def calculate_operational_window(
        self,
        product_type: str,
        current_time: pd.Timestamp,
    ) -> dict:
        """Calculate the operation window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """

        self.current_time = current_time

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        current_power = self.total_power_output.at[current_time]

        min_power = (
            self.min_power[current_time]
            if type(self.min_power) is pd.Series
            else self.min_power
        )
        if self.ramp_down != -1:
            min_power = max(current_power - self.ramp_down, min_power)
        else:
            min_power = min_power

        max_power = (
            self.max_power[current_time]
            if type(self.max_power) is pd.Series
            else self.max_power
        )
        if self.ramp_up != -1:
            max_power = min(current_power + self.ramp_up, max_power)
        else:
            max_power = max_power

        operational_window = {
            "current_power": {
                "power": current_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=current_power,
                    current_time=current_time,
                    partial_load_eff=True,
                ),
            },
            "min_power": {
                "power": min_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=min_power,
                    current_time=current_time,
                    partial_load_eff=True,
                ),
            },
            "max_power": {
                "power": max_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=max_power,
                    current_time=current_time,
                    partial_load_eff=True,
                ),
            },
        }

        return operational_window

    def calculate_bids(
        self,
        product_type,
        operational_window,
    ):
        return super().calculate_bids(
            unit=self,
            product_type=product_type,
            operational_window=operational_window,
        )

    def get_dispatch_plan(self, dispatch_plan, current_time):
        if dispatch_plan["total_capacity"] > self.min_power:
            self.market_success_list[-1] += 1
            self.current_status = 1
            self.current_down_time = 0
            self.total_power_output.at[current_time] = dispatch_plan["total_capacity"]

        elif dispatch_plan["total_capacity"] < self.min_power:
            self.current_status = 0
            self.current_down_time += 1
            self.total_power_output.at[current_time] = 0

            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)

    def calc_marginal_cost(
        self,
        power_output: float,
        current_time: pd.Timestamp,
        partial_load_eff: bool = False,
    ) -> float:
        fuel_price = (
            self.fuel_price.at[current_time]
            if type(self.fuel_price) is pd.Series
            else self.fuel_price
        )
        co2_price = (
            self.co2_price.at[current_time]
            if type(self.co2_price) is pd.Series
            else self.co2_price
        )

        # Partial load efficiency dependent marginal costs
        if not partial_load_eff:
            marginal_cost = (
                fuel_price / self.efficiency
                + co2_price * self.emission_factor / self.efficiency
                + self.fixed_cost
            )

            return marginal_cost

        capacity_ratio = power_output / self.max_power

        if self.fuel_type in ["lignite", "hard coal"]:
            eta_loss = (
                0.095859 * (capacity_ratio**4)
                - 0.356010 * (capacity_ratio**3)
                + 0.532948 * (capacity_ratio**2)
                - 0.447059 * capacity_ratio
                + 0.174262
            )

        elif self.fuel_type == "combined cycle gas turbine":
            eta_loss = (
                0.178749 * (capacity_ratio**4)
                - 0.653192 * (capacity_ratio**3)
                + 0.964704 * (capacity_ratio**2)
                - 0.805845 * capacity_ratio
                + 0.315584
            )

        elif self.fuel_type == "open cycle gas turbine":
            eta_loss = (
                0.485049 * (capacity_ratio**4)
                - 1.540723 * (capacity_ratio**3)
                + 1.899607 * (capacity_ratio**2)
                - 1.251502 * capacity_ratio
                + 0.407569
            )

        else:
            eta_loss = 0

        efficiency = self.efficiency - eta_loss

        marginal_cost = (
            fuel_price / efficiency
            + co2_price * self.emission_factor / efficiency
            + self.fixed_cost
        )

        return marginal_cost
