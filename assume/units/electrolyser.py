import pandas as pd

from assume.strategies import OperationalWindow
from assume.units.base_unit import BaseUnit


class Electrolyser(BaseUnit):
    def __init__(
        self,
        id: str,
        technology: str,
        bidding_strategies: dict,
        max_hydrogen_output: float or pd.Series,
        min_hydrogen_output: float or pd.Series,
        efficiency: float or pd.Series,
        volume: float or pd.Series = 1000,
        electricity_price: float or pd.Series = 3000,
        ramp_up: float = -1,
        ramp_down: float = 1,
        fixed_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 0.001,  # hours
        downtime_warm_start: int = 0.005,  # hours
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = None,
        node: str = None,
        dr_factor=None,
        **kwargs,
    ):
        super().__init__(
            id=id,
            technology=technology,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
        )

        self.max_hydrogen_output = max_hydrogen_output
        self.min_hydrogen_output = min_hydrogen_output
        self.efficiency = efficiency
        self.volume = volume
        self.electricity_price = (
            electricity_price
            if electricity_price is not None
            else pd.Series(0, index=index)
        )
        self.fixed_cost = fixed_cost
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start
        self.location = location
        self.dr_factor = dr_factor

    def reset(self):
        """Reset the unit to its initial state."""
        self.current_status = 1
        self.current_down_time = self.min_down_time

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[float]:
        """Calculate the operational window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """
        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        max_power = self.max_hydrogen_output.at[start]
        min_power = self.min_hydrogen_output.at[start]

        # Adjust for ramp down speed
        if self.ramp_down != -1:
            min_power = max(0, min_power - self.ramp_down)
        else:
            min_power = min_power

        current_power_input = self.outputs["hydrogen"].at[start]

        # Adjust min_power if sold negative reserve capacity on control reserve market
        min_power = min_power + self.neg_capacity_reserve.at[start]

        # Adjust for ramp up speed
        max_power = min(current_power_input + self.ramp_up, max_power)

        # Adjust max_power if sold positive reserve capacity on control reserve market
        max_power = max_power - self.outputs["pos_capacity"].at[start]

        return min_power, max_power

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        check if the total dispatch plan is feasible
        This checks if the market feedback is feasible for the given unit.
        And sets the closest dispatch if not.
        The end param should be inclusive.
        """
        end_excl = end - self.index.freq

        if self.outputs["energy"][start:end_excl].min() < self.min_power:
            self.outputs["energy"].loc[start:end_excl] = 0
            self.outputs["hydrogen"].loc[start:end_excl] = 0
            self.current_status = 0
            self.current_down_time += 1
        else:
            self.current_status = 1
            self.current_down_time = 0
            self.outputs["hydrogen"].loc[start:end_excl] = self.outputs["energy"][
                start:end_excl
            ]
        return self.outputs["energy"][start:end_excl]

    def calc_marginal_cost(
        self,
        timestep: pd.Timestamp,
    ) -> float or pd.Series:
        """
        Calculate the marginal cost for the electrolyser at the given time step.

        Parameters
        ----------
        timestep : pd.Timestamp
            The current time step.

        Returns
        -------
        bid_price : float or pd.Series
            The calculated bid price.
        """
        efficiency_t = self.efficiency.loc[timestep]

        if isinstance(self.electricity_price, pd.Series):
            bid_price = self.electricity_price.at[timestep] / efficiency_t
        else:
            bid_price = self.electricity_price / efficiency_t

        return bid_price
