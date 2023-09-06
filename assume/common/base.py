from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from assume.common.forecasts import Forecaster
from assume.common.market_objects import MarketConfig, Orderbook, Product


class BaseStrategy:
    pass


class BaseUnit:
    """
    A base class for a unit.

    :param id: The ID of the unit.
    :type id: str
    :param unit_operator: The operator of the unit.
    :type unit_operator: str
    :param technology: The technology of the unit.
    :type technology: str
    :param bidding_strategies: The bidding strategies of the unit.
    :type bidding_strategies: dict[str, BaseStrategy]
    :param index: The index of the unit.
    :type index: pd.DatetimeIndex
    :param outputs: The output of the unit.
    :type outputs: dict[str, pd.Series]
    :param forecaster: The forecast of the unit.
    :type forecaster: Forecaster

    Methods
    -------
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
        **kwargs,
    ):
        self.id = id
        self.unit_operator = unit_operator
        self.technology = technology
        self.node = node
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
        Calculate the bids for the next time step.

        :param market_config: The market configuration.
        :type market_config: MarketConfig
        :param product_tuples: The product tuples.
        :type product_tuples: list[tuple]
        :return: The bids.
        :rtype: Orderbook
        """

        if market_config.product_type not in self.bidding_strategies:
            return []

        bids = self.bidding_strategies[market_config.product_type].calculate_bids(
            unit=self,
            market_config=market_config,
            product_tuples=product_tuples,
        )
        for i, _ in enumerate(bids):
            bids[i].update(
                {
                    field: None
                    for field in market_config.additional_fields
                    if field not in bids[i].keys()
                }
            )

        return bids

    def set_dispatch_plan(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        adds dispatch plan from current market result to total dispatch plan

        :param marketconfig: The market configuration.
        :type marketconfig: MarketConfig
        :param orderbook: The orderbook.
        :type orderbook: Orderbook
        """
        product_type = marketconfig.product_type
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            if isinstance(order["accepted_volume"], dict):
                self.outputs[product_type].loc[start:end_excl] += [
                    order["accepted_volume"][key]
                    for key in order["accepted_volume"].keys()
                ]
            else:
                self.outputs[product_type].loc[start:end_excl] += order[
                    "accepted_volume"
                ]

        self.calculate_cashflow(product_type, orderbook)

        self.outputs[product_type + "_marginal_costs"].loc[start:end_excl] = (
            self.calculate_marginal_cost(start, self.outputs[product_type].loc[start])
            * self.outputs[product_type].loc[start:end_excl]
        )

        self.bidding_strategies[product_type].calculate_reward(
            unit=self,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        """
        check if the total dispatch plan is feasible
        This checks if the market feedback is feasible for the given unit.
        And sets the closest dispatch if not.
        The end param should be inclusive.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :return: the volume of the unit within the given time range
        :rtype: pd.Series
        """
        end_excl = end - self.index.freq
        energy = self.outputs["energy"][start:end_excl]
        return energy

    def get_output_before(self, dt: datetime, product_type: str = "energy") -> float:
        """
        return output before the given datetime.
        If datetime is before the start of the index, 0 is returned.

        :param dt: the datetime
        :type dt: datetime
        :param product_type: the product type
        :type product_type: str
        :return: the output before the given datetime
        :rtype: float
        """
        if dt - self.index.freq < self.index[0]:
            return 0
        else:
            return self.outputs["energy"].at[dt - self.index.freq]

    def as_dict(self) -> dict:
        """
        Returns a dictionary representation of the unit.

        :return: a dictionary representation of the unit
        :rtype: dict
        """
        return {
            "id": self.id,
            "technology": self.technology,
            "unit_operator": self.unit_operator,
            "unit_type": "base_unit",
        }

    def calculate_cashflow(self, product_type: str, orderbook: Orderbook):
        """
        calculates the cashflow for the given product_type

        :param product_type: the product type
        :type product_type: str
        :param orderbook: The orderbook.
        :type orderbook: Orderbook
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
                cashflow = float(order["accepted_price"] * order["accepted_volume"])
                hours = (end - start) / timedelta(hours=1)
                self.outputs[f"{product_type}_cashflow"].loc[start:end_excl] += (
                    cashflow * hours
                )


class SupportsMinMax(BaseUnit):
    """
    Base Class used for Powerplant derived classes

    :param min_power: The minimum power output of the unit.
    :type min_power: float
    :param max_power: The maximum power output of the unit.
    :type max_power: float
    :param ramp_down: How much power can be decreased in one time step.
    :type ramp_down: float
    :param ramp_up: How much power can be increased in one time step.
    :type ramp_up: float
    :param efficiency: The efficiency of the unit.
    :type efficiency: float
    :param emission_factor: The emission factor of the unit.
    :type emission_factor: float
    :param min_operating_time: The minimum time the unit has to be on.
    :type min_operating_time: int
    :param min_down_time: The minimum time the unit has to be off.
    :type min_down_time: int

    Methods
    -------
    """

    min_power: float
    max_power: float
    ramp_down: float
    ramp_up: float
    efficiency: float
    emission_factor: float
    min_operating_time: int
    min_down_time: int

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculates the min and max power for the given time period

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the min and max power for the given time period
        :rtype: tuple[pd.Series, pd.Series]
        """
        pass

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculates the marginal cost for the given power

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param power: the power output of the unit
        :type power: float
        :return: the marginal cost for the given power
        :rtype: float
        """
        pass

    def calculate_ramp(
        self, previous_power: float, power: float, current_power: float = 0
    ) -> float:
        """
        Calculates the ramp for the given power

        :param previous_power: the previous power output of the unit
        :type previous_power: float
        :param power: the power output of the unit
        :type power: float
        :param current_power: the current power output of the unit
        :type current_power: float
        :return: the ramp for the given power
        :rtype: float
        """
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

    def get_clean_spread(self, prices: pd.DataFrame):
        """
        returns the clean spread for the given prices

        :param prices: the prices
        :type prices: pd.DataFrame
        :return: the clean spread for the given prices
        :rtype: float
        """
        emission_cost = self.emission_factor * prices["co"].mean()
        fuel_cost = prices[self.technology.replace("_combined", "")].mean()
        return (fuel_cost + emission_cost) / self.efficiency

    def get_operation_time(self, start: datetime):
        """
        returns the operation time
        if unit is on since 4 hours, it returns 4
        if the unit is off since 4 hours, it returns -4
        The value at start is not considered

        :param start: the start time
        :type start: datetime
        :return: the operation time
        :rtype: int

        """
        before = start - self.index.freq
        # before = start
        max_time = max(self.min_operating_time, self.min_down_time)
        begin = before - self.index.freq * max_time
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


class SupportsMinMaxCharge(BaseUnit):
    """
    Base Class used for Storage derived classes

    :param initial_soc: The initial state of charge of the storage.
    :type initial_soc: float
    :param min_power_charge: How much power must be charged at least in one time step.
    :type min_power_charge: float
    :param max_power_charge: How much power can be charged at most in one time step.
    :type max_power_charge: float
    :param min_power_discharge: How much power must be discharged at least in one time step.
    :type min_power_discharge: float
    :param max_power_discharge: How much power can be discharged at most in one time step.
    :type max_power_discharge: float
    :param ramp_up_discharge: How much power can be increased in discharging in one time step.
    :type ramp_up_discharge: float
    :param ramp_down_discharge: How much power can be decreased in discharging in one time step.
    :type ramp_down_discharge: float
    :param ramp_up_charge: How much power can be increased in charging in one time step.
    :type ramp_up_charge: float
    :param ramp_down_charge: How much power can be decreased in charging in one time step.
    :type ramp_down_charge: float
    :param max_volume: The maximum volume of the storage.
    :type max_volume: float
    :param efficiency_charge: The efficiency of charging.
    :type efficiency_charge: float
    :param efficiency_discharge: The efficiency of discharging.
    :type efficiency_discharge: float

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
        calculates the min and max charging power for the given time period

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the min and max charging power for the given time period
        :rtype: tuple[pd.Series, pd.Series]
        """
        pass

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        calculates the min and max discharging power for the given time period

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the min and max discharging power for the given time period
        :rtype: tuple[pd.Series, pd.Series]
        """
        pass

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        calculates the marginal cost for the given power

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param power: the power output of the unit
        :type power: float
        :return: the marginal cost for the given power
        :rtype: float
        """
        pass

    def get_soc_before(self, dt: datetime) -> float:
        """
        return SoC before the given datetime.
        If datetime is before the start of the index, the initial SoC is returned.
        The SoC is a float between 0 and 1.

        :param dt: the datetime
        :type dt: datetime
        :return: the SoC before the given datetime
        :rtype: float
        """
        if dt - self.index.freq < self.index[0]:
            return self.initial_soc
        else:
            return self.outputs["soc"].at[dt - self.index.freq]

    def get_clean_spread(self, prices: pd.DataFrame):
        """
        returns the clean spread for the given prices

        :param prices: the prices
        :type prices: pd.DataFrame
        :return: the clean spread for the given prices
        :rtype: float
        """
        emission_cost = self.emission_factor * prices["co"].mean()
        fuel_cost = prices[self.technology.replace("_combined", "")].mean()
        return (fuel_cost + emission_cost) / self.efficiency_charge

    def calculate_ramp_discharge(
        self,
        previous_soc: float,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
        min_power_discharge: float = 0,
    ) -> float:
        """
        calculates the ramp for the given discharging power

        :param previous_power: the previous power output of the unit
        :type previous_power: float
        :param power_discharge: the discharging power output of the unit
        :type power_discharge: float
        :param current_power: the current power output of the unit
        :type current_power: float
        :return: the ramp for the given discharging power
        :rtype: float
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
            if previous_power < 0:
                previous_power = 0

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
        previous_soc: float,
        previous_power: float,
        power_charge: float,
        current_power: float = 0,
    ) -> float:
        """
        calculates the ramp for the given charging power

        :param previous_power: the previous power output of the unit
        :type previous_power: float
        :param power_charge: the charging power output of the unit
        :type power_charge: float
        :param current_power: the current power output of the unit
        :type current_power: float
        :return: the ramp for the given charging power
        :rtype: float
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
            if previous_power > 0:
                previous_power = 0

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
    """A base class for a bidding strategy.

    :param args: The arguments.
    :type args: list
    :param kwargs: The keyword arguments.
    :type kwargs: dict

    Methods
    -------

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

        :param unit: The unit.
        :type unit: BaseUnit
        :param market_config: The market configuration.
        :type market_config: MarketConfig
        :param product_tuples: The product tuples.
        :type product_tuples: list[Product]
        :return: The bids
        :rtype: Orderbook
        """
        raise NotImplementedError()

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the given unit

        :param unit: the unit
        :type unit: BaseUnit
        :param marketconfig: The market configuration.
        :type marketconfig: MarketConfig
        :param orderbook: The orderbook.
        :type orderbook: Orderbook
        """

    pass


class LearningStrategy(BaseStrategy):
    """
    A strategy which provides learning functionality, has a method to calculate the reward.

    :param args: The arguments.
    :type args: list
    :param kwargs: The keyword arguments.
    :type kwargs: dict

    Methods
    -------
    """

    obs_dim: int
    act_dim: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_dim = kwargs.get("observation_dimension", 50)
        self.act_dim = kwargs.get("action_dimension", 2)
