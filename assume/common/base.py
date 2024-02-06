# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import defaultdict
from datetime import datetime, timedelta
from typing import TypedDict, Union

import pandas as pd

from assume.common.forecasts import Forecaster
from assume.common.market_objects import MarketConfig, Orderbook, Product


class BaseStrategy:
    pass


class BaseUnit:
    """
    A base class for a unit. This class is used as a foundation for all units.

    Parameters:
        id (str): The ID of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict[str, BaseStrategy]): The bidding strategies of the unit.
        index (pd.DatetimeIndex): The index of the unit.
        node (str, optional): The node of the unit. Defaults to "".
        forecaster (Forecaster, optional): The forecast of the unit. Defaults to None.
        **kwargs: Additional keyword arguments.

    Args:
        id (str): The ID of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict[str, BaseStrategy]): The bidding strategies of the unit.
        index (pd.DatetimeIndex): The index of the unit.
        node (str, optional): The node of the unit. Defaults to "".
        forecaster (Forecaster, optional): The forecast of the unit. Defaults to None.
        location (tuple[float, float], optional): The location of the unit. Defaults to (0.0, 0.0).
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict[str, BaseStrategy],
        index: pd.DatetimeIndex,
        node: str = "",
        forecaster: Forecaster = None,
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        self.id = id
        self.unit_operator = unit_operator
        self.technology = technology
        self.node = node
        self.location = location
        self.bidding_strategies: dict[str, BaseStrategy] = bidding_strategies
        self.index = index
        self.outputs = defaultdict(lambda: pd.Series(0.0, index=self.index))
        # series does not like to convert from tensor to float otherwise
        self.outputs["rl_actions"] = pd.Series(0.0, index=self.index, dtype=object)
        self.outputs["rl_observations"] = pd.Series(0.0, index=self.index, dtype=object)
        self.outputs["reward"] = pd.Series(0.0, index=self.index, dtype=object)
        self.outputs["learning_mode"] = pd.Series(False, index=self.index, dtype=bool)
        self.outputs["rl_exploration_noise"] = pd.Series(
            0.0, index=self.index, dtype=object
        )
        if forecaster:
            self.forecaster = forecaster
        else:
            self.forecaster = defaultdict(lambda: pd.Series(0.0, index=self.index))

    def calculate_bids(
        self,
        market_config: MarketConfig,
        product_tuples: list[tuple],
    ) -> Orderbook:
        """
        Calculates the bids for the next time step.

        Args:
            market_config (MarketConfig): The market configuration.
            product_tuples (list[tuple]): The product tuples.

        Returns:
            Orderbook: The bids.

        Raises:
            KeyError: If the product type is not found in the bidding strategies.

        """

        if market_config.product_type not in self.bidding_strategies:
            return []

        bids = self.bidding_strategies[market_config.product_type].calculate_bids(
            unit=self,
            market_config=market_config,
            product_tuples=product_tuples,
        )
        # TODO one should make sure to use valid bidding strategies
        for i, _ in enumerate(bids):
            bids[i].update(
                {
                    field: None
                    for field in market_config.additional_fields
                    if field not in bids[i].keys()
                }
            )

        return bids

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculates the marginal cost for the given power.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost for the given power.

        """
        return 0

    def set_dispatch_plan(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        Iterates through the orderbook, adding the accepted volumes to the corresponding time slots
        in the dispatch plan. It then calculates the cashflow and the reward for the bidding strategies.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.

        """

        product_type = marketconfig.product_type
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            if isinstance(order["accepted_volume"], dict):
                added_volume = list(order["accepted_volume"].values())
            else:
                added_volume = order["accepted_volume"]
            self.outputs[product_type].loc[start:end_excl] += added_volume
        self.calculate_cashflow(product_type, orderbook)

        self.bidding_strategies[product_type].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )

    def calculate_generation_cost(
        self,
        start: datetime,
        end: datetime,
        product_type: str,
    ) -> None:
        """
        Calculates the generation cost for a specific product type within the given time range.

        Args:
            start (datetime.datetime): The start time for the calculation.
            end (datetime.datetime): The end time for the calculation.
            product_type (str): The type of product for which the generation cost is to be calculated.

        """
        if start not in self.index:
            start = self.index[0]
        product_type_mc = product_type + "_marginal_costs"
        for t in self.outputs[product_type_mc][start:end].index:
            mc = self.calculate_marginal_cost(
                start, self.outputs[product_type].loc[start]
            )
            self.outputs[product_type_mc][t] = mc * self.outputs[product_type][start]

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        """
        Checks if the total dispatch plan is feasible.

        This method checks if the market feedback is feasible for the given unit and sets the closest dispatch if not.
        The end parameter should be inclusive.

        Args:
            start: The start time of the dispatch.
            end: The end time of the dispatch.

        Returns:
            The volume of the unit within the given time range.
        """
        return self.outputs["energy"][start:end]

    def get_output_before(self, dt: datetime, product_type: str = "energy") -> float:
        """
        Returns output before the given datetime.

        If the datetime is before the start of the index, 0 is returned.

        Args:
            dt: The datetime.
            product_type: The product type (default is "energy").

        Returns:
            The output before the given datetime.
        """
        if dt - self.index.freq < self.index[0]:
            return 0
        else:
            return self.outputs[product_type].at[dt - self.index.freq]

    def as_dict(self) -> dict[str, Union[str, int]]:
        """
        Returns a dictionary representation of the unit.

        Returns:
            A dictionary representation of the unit.
        """
        return {
            "id": self.id,
            "technology": self.technology,
            "unit_operator": self.unit_operator,
            "node": self.node,
            "unit_type": "base_unit",
        }

    def calculate_cashflow(self, product_type: str, orderbook: Orderbook):
        """
        Calculates the cashflow for the given product type.

        Args:
            product_type: The product type.
            orderbook: The orderbook.
        """
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq

            if isinstance(order["accepted_volume"], dict):
                cashflow = [
                    float(order["accepted_price"][i] * order["accepted_volume"][i])
                    for i in order["accepted_volume"].keys()
                ]
                self.outputs[f"{product_type}_cashflow"].loc[start:end_excl] += (
                    cashflow * self.index.freq.n
                )
            else:
                cashflow = float(
                    order.get("accepted_price", 0) * order.get("accepted_volume", 0)
                )
                hours = (end - start) / timedelta(hours=1)
                self.outputs[f"{product_type}_cashflow"].loc[start:end_excl] += (
                    cashflow * hours
                )

    def get_starting_costs(self, op_time: int) -> float:
        """
        Returns the costs if start-up is planned.

        Args:
            op_time: Operation time in hours running from get_operation_time.

        Returns:
            Start-up costs.
        """
        return 0

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculates the marginal cost for the given power.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost for the given power.
        """
        pass


class SupportsMinMax(BaseUnit):
    """
    Base class used for units supporting continuous dispatch and without energy storage.
    This class is best to be used as foundation for classes of power plants and similar units.

    Parameters:
        min_power (float): The minimum power output of the unit.
        max_power (float): The maximum power output of the unit.
        ramp_down (float): How much power can be decreased in one time step.
        ramp_up (float): How much power can be increased in one time step.
        efficiency (float): The efficiency of the unit.
        emission_factor (float): The emission factor of the unit.
        min_operating_time (int): The minimum time the unit has to be on.
        min_down_time (int): The minimum time the unit has to be off.

    Methods
    -------
    """

    min_power: float
    max_power: float
    ramp_down: float
    ramp_up: float
    efficiency: float
    emission_factor: float
    min_operating_time: int = 0
    min_down_time: int = 0

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type: str = "energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculates the min and max power for the given time period.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.
            product_type (str): The product type of the unit.

        Returns:
            tuple[pd.Series, pd.Series]: The min and max power for the given time period.
        """
        pass

    def calculate_ramp(
        self,
        op_time: int,
        previous_power: float,
        power: float,
        current_power: float = 0,
    ) -> float:
        """
        Corrects the possible power to offer according to ramping restrictions.

        Args:
            op_time (int): The operation time.
            previous_power (float): The previous power output of the unit.
            power (float): The planned power offer of the unit.
            current_power (float): The current power output of the unit.

        Returns:
            float: The corrected possible power to offer according to ramping restrictions.
        """

        # was off before, but should be on now and min_down_time is not reached
        if power > 0 and op_time < 0 and op_time > -self.min_down_time:
            power = 0
        # was on before, but should be off now and min_operating_time is not reached
        elif power == 0 and op_time > 0 and op_time < self.min_operating_time:
            power = self.min_power

        if power == 0:
            # if less than min_power is required, we run min_power
            # we could also split at self.min_power/2
            return power
        # ramp up constraint
        # max_power + current_power < previous_power + unit.ramp_up
        power = min(
            power,
            previous_power + self.ramp_up - current_power,
            self.max_power - current_power,
        )
        # ramp down constraint
        # min_power + current_power > previous_power - unit.ramp_down
        power = max(
            power,
            previous_power - self.ramp_down - current_power,
            self.min_power - current_power,
        )
        return power

    def get_clean_spread(self, prices: pd.DataFrame) -> float:
        """
        Returns the clean spread for the given prices.

        Args:
            prices (pd.DataFrame): The prices.

        Returns:
            float: The clean spread for the given prices.
        """
        emission_cost = self.emission_factor * prices["co"].mean()
        fuel_cost = prices[self.technology.replace("_combined", "")].mean()
        return (fuel_cost + emission_cost) / self.efficiency

    def get_operation_time(self, start: datetime) -> int:
        """
        Returns the time the unit is operating (positive) or shut down (negative).

        Args:
            start (datetime.datetime): The start time.

        Returns:
            int: The operation time.
        """
        before = start - self.index.freq

        max_time = max(self.min_operating_time, self.min_down_time)
        begin = start - self.index.freq * max_time
        end = before
        arr = self.outputs["energy"][begin:end][::-1] > 0
        if len(arr) < 1:
            # before start of index
            return max_time
        is_off = not arr.iloc[0]
        runn = 0
        for val in arr:
            if val == is_off:
                break
            runn += 1
        return (-1) ** is_off * runn

    def get_average_operation_times(self, start: datetime) -> tuple[float, float]:
        """
        Calculates the average uninterrupted operation and down time.

        Args:
            start (datetime.datetime): The current time.

        Returns:
            tuple[float, float]: Tuple of the average operation time avg_op_time and average down time avg_down_time.

        Note:
            down_time in general is indicated with negative values
        """
        op_series = []

        before = start - self.index.freq
        arr = self.outputs["energy"][self.index[0] : before][::-1] > 0

        if len(arr) < 1:
            # before start of index
            return max(self.min_operating_time, 1), min(-self.min_down_time, -1)

        op_series = []
        status = arr.iloc[0]
        runn = 0
        for val in arr:
            if val == status:
                runn += 1
            else:
                op_series.append(-((-1) ** status) * runn)
                runn = 1
                status = val
        op_series.append(-((-1) ** status) * runn)

        op_times = [operation for operation in op_series if operation > 0]
        if op_times == []:
            avg_op_time = self.min_operating_time
        else:
            avg_op_time = sum(op_times) / len(op_times)

        down_times = [operation for operation in op_series if operation < 0]
        if down_times == []:
            avg_down_time = self.min_down_time
        else:
            avg_down_time = sum(down_times) / len(down_times)

        return max(1, avg_op_time, self.min_operating_time), min(
            -1, avg_down_time, -self.min_down_time
        )

    def get_starting_costs(self, op_time: int) -> float:
        """
        Returns the start-up cost for the given operation time.
        If operation time is positive, the unit is running, so no start-up costs are returned.
        If operation time is negative, the unit is not running, so start-up costs are returned
        according to the start-up costs of the unit and the hot/warm/cold start times.

        Args:
            op_time (int): The operation time.

        Returns:
            float: The start-up costs depending on the down time.
        """
        if op_time > 0:
            # unit is running
            return 0

        if self.downtime_hot_start is not None and self.hot_start_cost is not None:
            if -op_time <= self.downtime_hot_start:
                return self.hot_start_cost
        if self.downtime_warm_start is not None and self.warm_start_cost is not None:
            if -op_time <= self.downtime_warm_start:
                return self.warm_start_cost
        if self.cold_start_cost is not None:
            return self.cold_start_cost

        return 0


class SupportsMinMaxCharge(BaseUnit):
    """
    Base Class used for units with energy storage.

    Parameters:
        initial_soc (float): The initial state of charge of the storage.
        min_power_charge (float): How much power must be charged at least in one time step.
        max_power_charge (float): How much power can be charged at most in one time step.
        min_power_discharge (float): How much power must be discharged at least in one time step.
        max_power_discharge (float): How much power can be discharged at most in one time step.
        ramp_up_discharge (float): How much power can be increased in discharging in one time step.
        ramp_down_discharge (float): How much power can be decreased in discharging in one time step.
        ramp_up_charge (float): How much power can be increased in charging in one time step.
        ramp_down_charge (float): How much power can be decreased in charging in one time step.
        max_volume (float): The maximum volume of the storage.
        efficiency_charge (float): The efficiency of charging.
        efficiency_discharge (float): The efficiency of discharging.

    Methods
    -------
    """

    initial_soc: float
    min_power_charge: float
    max_power_charge: float
    min_power_discharge: float
    max_power_discharge: float
    ramp_up_discharge: float
    ramp_down_discharge: float
    ramp_up_charge: float
    ramp_down_charge: float
    max_volume: float
    efficiency_charge: float
    efficiency_discharge: float

    def calculate_min_max_charge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculates the min and max charging power for the given time period.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.
            product_type (str, optional): The product type of the unit. Defaults to "energy".

        Returns:
            tuple[pd.Series, pd.Series]: The min and max charging power for the given time period.
        """
        pass

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculates the min and max discharging power for the given time period.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.
            product_type (str, optional): The product type of the unit. Defaults to "energy".

        Returns:
            tuple[pd.Series, pd.Series]: The min and max discharging power for the given time period.
        """
        pass

    def get_soc_before(self, dt: datetime) -> float:
        """
        Returns the State of Charge (SoC) before the given datetime.
        If datetime is before the start of the index, the initial SoC is returned.
        The SoC is a float between 0 and 1.

        Args:
            dt (datetime.datetime): The current datetime.

        Returns:
            float: The SoC before the given datetime.
        """
        if dt - self.index.freq <= self.index[0]:
            return self.initial_soc
        else:
            return self.outputs["soc"].at[dt - self.index.freq]

    def get_clean_spread(self, prices: pd.DataFrame) -> float:
        """
        Returns the clean spread for the given prices.

        Args:
            prices (pd.DataFrame): The prices.

        Returns:
            float: The clean spread for the given prices.
        """
        emission_cost = self.emission_factor * prices["co"].mean()
        fuel_cost = prices[self.technology.replace("_combined", "")].mean()
        return (fuel_cost + emission_cost) / self.efficiency_charge

    def calculate_ramp_discharge(
        self,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
    ) -> float:
        """
        Adjusts the discharging power to the ramping constraints.

        Args:
            previous_power (float): The previous power output of the unit.
            power_discharge (float): The discharging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.

        Returns:
            float: The discharging power adjusted to the ramping constraints.
        """
        if power_discharge == 0:
            return power_discharge

        # if storage was charging before and ramping for charging is defined
        if previous_power < 0 and self.ramp_down_charge != None:
            power_discharge = max(
                previous_power - self.ramp_down_charge - current_power, 0
            )
        else:
            # Assuming the storage is not restricted by ramping charging down
            previous_power = max(previous_power, 0)

            power_discharge = min(
                power_discharge,
                max(0, previous_power + self.ramp_up_discharge - current_power),
            )
            # restrict only if ramping defined
            if self.ramp_down_discharge != None:
                power_discharge = max(
                    power_discharge,
                    previous_power - self.ramp_down_discharge - current_power,
                    0,
                )
        return power_discharge

    def calculate_ramp_charge(
        self,
        previous_power: float,
        power_charge: float,
        current_power: float = 0,
    ) -> float:
        """
        Adjusts the charging power to the ramping constraints.

        Args:
            previous_power (float): The previous power output of the unit.
            power_charge (float): The charging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.

        Returns:
            float: The charging power adjusted to the ramping constraints.
        """

        if power_charge == 0:
            return power_charge

        # assuming ramping down discharge restricts ramp up of charge
        # if storage was discharging before and ramp_down_discharge is defined
        if previous_power > 0 and self.ramp_down_discharge != 0:
            power_charge = min(
                previous_power - self.ramp_down_discharge - current_power, 0
            )
        else:
            previous_power = min(previous_power, 0)

            power_charge = max(
                power_charge,
                min(previous_power + self.ramp_up_charge - current_power, 0),
            )
            # restrict only if ramping defined
            if self.ramp_down_charge != 0:
                power_charge = min(
                    power_charge,
                    previous_power - self.ramp_down_charge - current_power,
                    0,
                )

        return power_charge


class BaseStrategy:
    """
    A base class for a bidding strategy.

    Args:
        *args (list): The arguments.
        **kwargs (dict): The keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        pass

    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates the bids for the next time step.

        Args:
            unit (BaseUnit): The unit.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The product tuples.

        Returns:
            Orderbook: The bids.
        """

    def calculate_reward(
        self,
        unit: BaseUnit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the given unit.

        Args:
            unit (BaseUnit): The unit.
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        pass


class LearningStrategy(BaseStrategy):
    """
    A strategy which provides learning functionality, has a method to calculate the reward.

    Parameters:
        obs_dim (int): The observation dimension.
        act_dim (int): The action dimension.

    Args:
        *args (list): The arguments.
        **kwargs (dict): The keyword arguments.
    """

    obs_dim: int
    act_dim: int

    def __init__(self, *args, **kwargs):
        """
        Initializes the learning strategy.
        """
        super().__init__(*args, **kwargs)
        self.obs_dim = kwargs.get("observation_dimension", 50)
        self.act_dim = kwargs.get("action_dimension", 2)


class LearningConfig(TypedDict):
    """
    A class for the learning configuration.

    Parameters:
        observation_dimension (int): The observation dimension.
        action_dimension (int): The action dimension.
        continue_learning (bool): Whether to continue learning.
        max_bid_price (float): The maximum bid price.
        learning_mode (bool): Whether to use learning mode.
        algorithm (str): The algorithm to use.
        learning_rate (float): The learning rate.
        training_episodes (int): The number of training episodes.
        episodes_collecting_initial_experience (int): The number of episodes collecting initial experience.
        train_freq (int): The training frequency.
        gradient_steps (int): The number of gradient steps.
        batch_size (int): The batch size.
        gamma (float): The discount factor.
        device (str): The device to use.
        noise_sigma (float): The standard deviation of the noise.
        noise_scale (int): Controls the initial strength of the noise.
        noise_dt (int): Determines how quickly the noise weakens over time.
        trained_policies_save_path (str): The path to the learned model to save.
        trained_policies_load_path (str): The path to the learned model to load.
    """

    observation_dimension: int
    action_dimension: int
    continue_learning: bool
    max_bid_price: float
    learning_mode: bool
    algorithm: str
    learning_rate: float
    training_episodes: int
    episodes_collecting_initial_experience: int
    train_freq: int
    gradient_steps: int
    batch_size: int
    gamma: float
    device: str
    noise_sigma: float
    noise_scale: int
    noise_dt: int
    trained_policies_save_path: str
