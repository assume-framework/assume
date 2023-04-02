from ..strategies import BaseStrategy
from .base_unit import BaseUnit


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
        node: str,
        max_power: float,
        min_power: float,
        efficiency: float,
        fuel_type: str,
        fuel_price: float,  # should be list later
        co2_price: float,
        emission_factor: float,
        ramp_up: float = -1,
        ramp_down: float = -1,
        fixed_cost: float = 0,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = -1,
        min_down_time: float = -1,
        heat_extraction: bool = False,
        max_heat_extraction: float = 0,
        availability: dict = None,
        location: tuple[float, float] = None,
        **kwargs
    ):

        super().__init__(id, technology, node)

        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.fuel_type = fuel_type
        self.fuel_price = fuel_price * 1000
        self.co2_price = co2_price * 1000
        self.emission_factor = emission_factor

        self.ramp_up = ramp_up if ramp_up > 0 else max_power
        self.ramp_down = ramp_down if ramp_down > 0 else max_power
        self.min_operating_time = max(min_operating_time, 0)
        self.min_down_time = max(min_down_time, 0)

        self.fixed_cost = fixed_cost
        self.hot_start_cost = hot_start_cost * max_power
        self.warm_start_cost = warm_start_cost * max_power
        self.cold_start_cost = cold_start_cost * max_power

        self.heat_extraction = heat_extraction
        self.max_heat_extraction = max_heat_extraction

        self.availability = availability

        self.location = location

        self.bidding_strategy = None

    def reset(self):
        """Reset the unit to its initial state."""

        self.current_time_step = 0
        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_power_output = [0.5 * self.max_power]
        self.total_heat_output = [0.0]
        self.pos_capacity_reserve = [0.0]
        self.neg_capacity_reserve = [0.0]

    def calculate_operational_window(self) -> dict:

        """Calculate the operation window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """

        t = self.current_time_step

        current_power = self.total_power_output[t - 1]

        if self.availability is None:
            min_power = max(
                self.total_power_output[t - 1] - self.ramp_down, self.min_power
            )
            max_power = min(
                self.total_power_output[t - 1] + self.ramp_up, self.max_power
            )

        elif self.availability[t] >= self.min_power:
            min_power = max(
                self.total_power_output[t - 1] - self.ramp_down, self.min_power
            )
            max_power = min(
                self.total_power_output[t - 1] + self.ramp_up, self.availability[t]
            )

        else:
            min_power = 0.0
            max_power = 0.0

        operational_window = {
            "current_power": {
                "power": current_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=current_power, partial_load_eff=True
                ),
            },
            "min_power": {
                "power": min_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=min_power, partial_load_eff=True
                ),
            },
            "max_power": {
                "power": max_power,
                "marginal_cost": self.calc_marginal_cost(
                    power_output=max_power, partial_load_eff=True
                ),
            },
        }

        return operational_window

    def calc_marginal_cost(
        self, power_output: float, partial_load_eff: bool = False
    ) -> float:

        """Calculate the marginal cost of the unit.

        Parameters
        ----------
        power_output : float
            Power output of the unit in MW.
        partial_load_eff : bool, optional
            If True, the partial load efficiency is considered. The default is False.

        Returns
        -------
        marginal_cost : float
            Marginal cost of the unit in €/MWh.
        """

        t = self.current_time_step

        fuel_price = 10  # self.fuel_price[t]
        co2_price = 12  # self.co2_price[t]

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
