# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import defaultdict
from datetime import datetime, timedelta
from typing import TypedDict

import numpy as np

from assume.common.fast_pandas import FastSeries, TensorFastSeries
from assume.common.forecasts import Forecaster
from assume.common.market_objects import MarketConfig, Orderbook, Product


class BaseStrategy:
    pass


class BaseUnit:
    """
    A base class for a unit. This class is used as a foundation for all units.

    Args:
        id (str): The ID of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict[str, BaseStrategy]): The bidding strategies of the unit.
        index (FastIndex): The index of the unit.
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
        forecaster: Forecaster,
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        self.id = id
        self.unit_operator = unit_operator
        self.technology = technology
        self.bidding_strategies: dict[str, BaseStrategy] = bidding_strategies
        self.forecaster = forecaster
        self.index = forecaster.index

        self.node = node
        self.location = location

        self.outputs = defaultdict(lambda: FastSeries(value=0.0, index=self.index))
        # series does not like to convert from tensor to float otherwise

        self.avg_op_time = 0
        self.total_op_time = 0

        # some data is stored as series to allow to store it in the outputs
        # check if any bidding strategy is using the RL strategy
        if any(
            isinstance(strategy, LearningStrategy)
            for strategy in self.bidding_strategies.values()
        ):
            self.outputs["actions"] = TensorFastSeries(value=0.0, index=self.index)
            self.outputs["exploration_noise"] = TensorFastSeries(
                value=0.0,
                index=self.index,
            )
            self.outputs["reward"] = FastSeries(value=0.0, index=self.index)
            self.outputs["regret"] = FastSeries(value=0.0, index=self.index)

            # RL data stored as lists to simplify storing to the buffer
            self.outputs["rl_observations"] = []
            self.outputs["rl_actions"] = []
            self.outputs["rl_rewards"] = []

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

        if market_config.market_id not in self.bidding_strategies:
            return []

        bids = self.bidding_strategies[market_config.market_id].calculate_bids(
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

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculates the marginal cost for the given power.`

        Args:
            start (datetime.datetime): The start time of the dispatch.
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
        in the dispatch plan.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.

        """

        product_type = marketconfig.product_type
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            # end includes the end of the last product, to get the last products' start time we deduct the frequency once
            end_excl = end - self.index.freq

            # Determine the added volume
            if isinstance(order["accepted_volume"], dict):
                added_volume = list(order["accepted_volume"].values())
            else:
                added_volume = order["accepted_volume"]
            self.outputs[product_type].loc[start:end_excl] += added_volume

            # Get the accepted price and store it in the outputs
            if isinstance(order["accepted_price"], dict):
                accepted_price = list(order["accepted_price"].values())
            else:
                accepted_price = order["accepted_price"]
            self.outputs[f"{product_type}_accepted_price"].loc[start:end_excl] = (
                accepted_price
            )

    def calculate_cashflow_and_reward(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        Calculates the cashflow and the reward for the given unit.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """

        product_type = marketconfig.product_type
        self.calculate_cashflow(product_type, orderbook)

        self.bidding_strategies[marketconfig.market_id].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )

    def calculate_generation_cost(
        self, start: datetime, end: datetime, product_type: str
    ) -> None:
        """
        Calculates the generation cost for a specific product type within the given time range,
        but only if the end is the last index in the time series.

        Args:
            start (datetime.datetime): The start time for the calculation.
            end (datetime.datetime): The end time for the calculation.
            product_type (str): The type of product for which the generation cost is to be calculated.
        """

        if start not in self.index:
            start = self.index[0]

        # Adjusted code for accessing product data and mapping over the index
        product_data = self.outputs[product_type].loc[start:end]

        marginal_costs = [
            self.calculate_marginal_cost(t, product_data[idx])
            for idx, t in enumerate(self.index[start:end])
        ]
        generation_costs = np.abs(marginal_costs * product_data)
        self.outputs[f"{product_type}_generation_costs"].loc[start:end] = (
            generation_costs
        )

    def execute_current_dispatch(
        self,
        start: datetime,
        end: datetime,
    ) -> np.array:
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
        return self.outputs["energy"].loc[start:end]

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

    def as_dict(self) -> dict[str, str | int]:
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
            # end includes the end of the last product, to get the last products' start time we deduct the frequency once
            end_excl = end - self.index.freq

            if isinstance(order["accepted_volume"], dict):
                cashflow = np.array(
                    [
                        float(order["accepted_price"][i] * order["accepted_volume"][i])
                        for i in order["accepted_volume"].keys()
                    ]
                )
            else:
                cashflow = float(
                    order.get("accepted_price", 0) * order.get("accepted_volume", 0)
                )

            elapsed_intervals = (end - start) / self.index.freq
            self.outputs[f"{product_type}_cashflow"].loc[start:end_excl] += (
                cashflow * elapsed_intervals
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

    def reset_saved_rl_data(self):
        """
        Resets the saved RL data. This deletes all data besides the observation and action where we do not yet have calculated reward values.
        """
        values_len = len(self.outputs["rl_rewards"])

        self.outputs["rl_observations"] = self.outputs["rl_observations"][values_len:]
        self.outputs["rl_actions"] = self.outputs["rl_actions"][values_len:]
        self.outputs["rl_rewards"] = []


class SupportsMinMax(BaseUnit):
    """
    Base class used for units supporting continuous dispatch and without energy storage.
    This class is best to be used as foundation for classes of power plants and similar units.
    """

    min_power: float
    max_power: float
    ramp_down: float = None
    ramp_up: float = None
    efficiency: float
    emission_factor: float
    min_operating_time: int = 0
    min_down_time: int = 0

    def calculate_min_max_power(
        self, start: datetime, end: datetime, product_type: str = "energy"
    ) -> tuple[np.array, np.array]:
        """
        Calculates the min and max power for the given time period.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.
            product_type (str): The product type of the unit.

        Returns:
            tuple[np.array, np.array]: The min and max power for the given time period.
        """

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
        if self.ramp_down is None and self.ramp_up is None:
            return power

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
        if self.ramp_up is not None:
            power = min(
                power,
                previous_power + self.ramp_up - current_power,
                self.max_power - current_power,
            )
        # ramp down constraint
        # min_power + current_power > previous_power - unit.ramp_down
        if self.ramp_down is not None:
            power = max(
                power,
                previous_power - self.ramp_down - current_power,
                self.min_power - current_power,
            )
        return power

    def get_operation_time(self, start: datetime) -> int:
        """
        Returns the time the unit is operating (positive) or shut down (negative).

        Args:
            start (datetime.datetime): The start time.

        Returns:
            int: The operation time as a positive integer if operating, or negative if shut down.
        """
        # Set the time window based on max of min operating/down time
        max_time = max(self.min_operating_time, self.min_down_time, 1)
        begin = max(start - self.index.freq * max_time, self.index[0])
        end = start - self.index.freq

        if start <= self.index[0]:
            # before start of index
            return max_time

        # Check energy output in the defined time window, reversed for most recent state
        arr = (self.outputs["energy"].loc[begin:end] > 0)[::-1]

        # Determine initial state (off if the first period shows zero energy output)
        is_off = not arr[0]
        run = 0

        # Count consecutive periods with the same status, break on change
        for val in arr:
            if val != (not is_off):  # Stop if the state changes
                break
            run += 1

        # Return positive time if operating, negative if shut down
        return -run if is_off else run

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
            # The unit is running, no start-up cost is needed
            return 0

        downtime = abs(op_time)

        # Check and return the appropriate start-up cost
        if downtime <= self.downtime_hot_start:
            return self.hot_start_cost

        if downtime <= self.downtime_warm_start:
            return self.warm_start_cost

        # If it exceeds warm start threshold, return cold start cost
        return self.cold_start_cost


class SupportsMinMaxCharge(BaseUnit):
    """
    Base Class used for units with energy storage.

    The volume is always the amount of energy which is put on (if positive) the market or (if negative) taken from the market.
    A demand does always have a negative volume - as it buys/consumes energy .
    A powerplant does always have a positive volume - as it produces energy.

    All charging related params are negative, as charging is a demand.
    This is special for SupportsMinMaxCharge as both charge and discharge is available.
    """

    initial_soc: float
    min_power_charge: float
    # negative float - if this storage is charging, what is the minimum charging power (the negative non-zero power closest to zero) (resulting in negative current power)
    max_power_charge: float
    # negative float - if this storage is charging, what is the maximum charging power (resulting in negative current power)
    min_power_discharge: float
    # positive float - if this storage is discharging, what is the minimum output power
    max_power_discharge: float
    # positive float - if this storage is discharging, what is the maximum output power
    ramp_up_discharge: float
    # positive float - when discharging,
    ramp_down_discharge: float
    # positive float
    ramp_up_charge: float
    # negative
    ramp_down_charge: float
    # ramp_down_charge is negative
    max_soc: float
    efficiency_charge: float
    efficiency_discharge: float

    def calculate_min_max_charge(
        self, start: datetime, end: datetime, soc: float = None
    ) -> tuple[np.array, np.array]:
        """
        Calculates the min and max charging power for the given time period.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.
            soc (float, optional): The current state-of-charge. Defaults to None.

        Returns:
            tuple[np.array, np.array]: The min and max charging power for the given time period.
        """

    def calculate_min_max_discharge(
        self, start: datetime, end: datetime, soc: float = None
    ) -> tuple[np.array, np.array]:
        """
        Calculates the min and max discharging power for the given time period.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.
            soc (float, optional): The current state-of-charge. Defaults to None.

        Returns:
            tuple[np.array, np.array]: The min and max discharging power for the given time period.
        """

    def calculate_ramp_discharge(
        self,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
    ) -> float:
        """
        Adjusts the discharging power to the ramping constraints.
        Given previous_power in the step before, we want to offer power_discharge (positive) to some market,
        while we already sold current_power on some other market.

        power_discharge is a power delta to our current output, this function checks if this amount is still feasible to sell.

        Args:
            previous_power (float): The previous power output of the unit.
            power_discharge (float): The discharging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.

        Returns:
            float: The discharging power adjusted to the ramping constraints.
        """
        # previously charging -800 MW wants to go to 200 MW - but must first check if charging is even possible
        # - 800 MW to 0 with charge ramp down and then 200 MW with discharge ramp up
        # if storage was charging before we need to check if we can ramp back to zero
        if (
            previous_power < 0
            and self.calculate_ramp_charge(previous_power, 0, current_power) < 0
        ):
            # if we can not ramp back to 0, we can not discharge anything
            return self.calculate_ramp_charge(previous_power, 0, current_power)
        else:
            # as we can ramp the charging to 0, we can assume that the previous_power = 0
            previous_power = max(previous_power, 0)

            power_discharge = min(
                power_discharge,
                # what I had + how much I could - what I already sold
                max(0, previous_power + self.ramp_up_discharge - current_power),
                self.max_power_discharge - current_power,
            )
            # restrict only if ramping defined
            if self.ramp_down_discharge and power_discharge != 0:
                power_discharge = max(
                    power_discharge,
                    # what I had - ramp down = minimum_required
                    # as I already provide current_power,
                    # need to at least offer minimum_required - current_power
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

        if (
            previous_power > 0
            and self.calculate_ramp_discharge(previous_power, 0, current_power) > 0
        ):
            # if we can not ramp back to 0, we can not charge anything
            return self.calculate_ramp_discharge(previous_power, 0, current_power)
        else:
            # as we can ramp the charging to 0, we can assume that the previous_power = 0
            previous_power = min(previous_power, 0)

            power_charge = max(
                power_charge,
                # what I had + how much I could - what I already sold
                min(0, previous_power + self.ramp_up_charge - current_power),
                self.max_power_charge - current_power,
            )
            # restrict only if ramping defined
            if self.ramp_down_charge and power_charge != 0:
                power_charge = min(
                    power_charge,
                    # what I had - ramp down = minimum_required
                    # as I already provide current_power,
                    # need to at least offer minimum_required - current_power
                    previous_power - self.ramp_down_charge - current_power,
                    0,
                )
        return power_charge

    def set_dispatch_plan(
        self, marketconfig: MarketConfig, orderbook: Orderbook
    ) -> None:
        """Updates the SOC for storage units."""
        super().set_dispatch_plan(marketconfig, orderbook)

        if not orderbook:
            return

        # also update the SOC when setting the dispatch plan
        start = min(order["start_time"] for order in orderbook)
        end = max(order["end_time"] for order in orderbook)
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq
        time_delta = self.index.freq / timedelta(hours=1)

        for t in self.index[start:end_excl]:
            next_t = t + self.index.freq
            # continue if it is the last time step
            if next_t not in self.index:
                continue
            current_power = self.outputs["energy"].at[t]

            # calculate the change in state of charge
            delta_soc = 0
            soc = self.outputs["soc"].at[t]

            # discharging
            if current_power > 0:
                max_soc_discharge = self.calculate_soc_max_discharge(soc)

                if current_power > max_soc_discharge:
                    current_power = max_soc_discharge

                delta_soc = -current_power * time_delta / self.efficiency_discharge

            # charging
            elif current_power < 0:
                max_soc_charge = self.calculate_soc_max_charge(soc)

                if current_power < max_soc_charge:
                    current_power = max_soc_charge

                delta_soc = -current_power * time_delta * self.efficiency_charge

            # update the values of the state of charge and the energy
            self.outputs["soc"].at[next_t] = soc + delta_soc
            self.outputs["energy"].at[t] = current_power


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

    def remove_empty_bids(self, bids: list) -> list:
        """
        Removes empty bids from the orderbook. Use this method to clean the bids before submitting
        them to the market to speed up the market clearing process, and if zero volume bids are not
        required for the specific market.

        Args:
            bids (list): The bids.

        Returns:
            list: The cleaned bids.
        """

        cleaned_bids = []
        for bid in bids:
            if isinstance(bid["volume"], dict):
                if all(volume == 0 for volume in bid["volume"].values()):
                    continue
            elif bid["volume"] == 0:
                continue
            cleaned_bids.append(bid)

        return cleaned_bids


class LearningStrategy(BaseStrategy):
    """
    A strategy which provides learning functionality, has a method to calculate the reward.

    It is important to keep in mind, that the DRL method and the centralized critic relies on
    unique observations of individual units. The algorithm is designed in such a way, that
    the unique observations are always placed at the end of the observation space. Please follow this
    convention when designing your create_observation method and the observation space.

    Attributes:
        obs_dim (int): The observation dimension.
        act_dim (int): The action dimension.
        unique_obs_dim (int): The unique observation dimension.
        num_timeseries_obs_dim (int): The number of observation timeseries dimension.

    Args:
        *args (list): The arguments.
        **kwargs (dict): The keyword arguments.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        unique_obs_dim: int = 0,
        num_timeseries_obs_dim: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initializes the learning strategy.
        """
        super().__init__(*args, **kwargs)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # this defines the number of unique observations, which are not the same for all units
        # this is used by the centralised critic and will return an error if not matched
        self.unique_obs_dim = unique_obs_dim

        # defines the number of provided timeseries, this is necessary for correctly splitting
        # them into suitable format for recurrent neural networks
        self.num_timeseries_obs_dim = num_timeseries_obs_dim


class LearningConfig(TypedDict):
    """
    A class for the learning configuration.
    """

    continue_learning: bool
    max_bid_price: float
    learning_mode: bool
    algorithm: str
    actor_architecture: str
    learning_rate: float
    learning_rate_schedule: str
    training_episodes: int
    episodes_collecting_initial_experience: int
    train_freq: str
    gradient_steps: int
    batch_size: int
    gamma: float
    device: str
    noise_sigma: float
    noise_scale: int
    noise_dt: int
    action_noise_schedule: str
    trained_policies_save_path: str
    trained_policies_load_path: str
    early_stopping_steps: int
    early_stopping_threshold: float
