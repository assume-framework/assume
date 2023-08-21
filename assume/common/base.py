from collections import defaultdict
from datetime import datetime

import pandas as pd

from assume.common.forecasts import Forecaster
from assume.common.market_objects import MarketConfig, Orderbook, Product


class BaseStrategy:
    pass


class BaseUnit:
    """A base class for a unit.

    Attributes
    ----------
    id : str
        The ID of the unit.
    technology : str
        The technology of the unit.
    node : str
        The node of the unit.

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
        self.outputs["rl_rewards"] = pd.Series(0.0, index=self.index, dtype=object)
        if forecaster:
            self.forecaster = forecaster
        else:
            self.forecaster = defaultdict(lambda: pd.Series(0.0, index=self.index))

    def reset(self):
        """Reset the unit to its initial state."""
        raise NotImplementedError()

    def calculate_bids(
        self,
        market_config: MarketConfig,
        product_tuples: list[tuple],
    ) -> Orderbook:
        """Calculate the bids for the next time step."""

        if market_config.product_type not in self.bidding_strategies:
            return []

        return self.bidding_strategies[market_config.product_type].calculate_bids(
            unit=self,
            market_config=market_config,
            product_tuples=product_tuples,
        )

    def set_dispatch_plan(
        self,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ) -> None:
        """
        adds dispatch plan from current market result to total dispatch plan
        """
        product_type = marketconfig.product_type
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            end_excl = end - self.index.freq
            self.outputs[product_type].loc[start:end_excl] += order["volume"]

        self.calculate_cashflow(product_type, orderbook)

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
        """
        end_excl = end - self.index.freq
        return self.outputs["energy"][start:end_excl]

    def get_output_before(self, dt: datetime, product_type: str = "energy") -> float:
        """
        return output before the given datetime.
        """
        if dt - self.index.freq < self.index[0]:
            return 0
        else:
            return self.outputs["energy"].at[dt - self.index.freq]

    def as_dict(self) -> dict:
        """
        Returns a dictionary representation of the unit.
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
        """
        pass


class SupportsMinMax(BaseUnit):
    """
    Base Class used for Powerplant derived classes
    """
    # minimum power output 
    min_power: float
    # maximum power output
    max_power: float 
     # how much power can be reduced in one time step
    ramp_down: float 
    # how much power can be increased in one time step
    ramp_up: float  
    # percentage of how much output power is provided
    efficiency: float 
    # how much kg/kWh of CO2 emissions is needed
    emission_factor: float
    # minimum time the unit has to be on
    min_operating_time: int  
    # minimum time the unit has to be off
    min_down_time: int  

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        calculates the min and max power for the given time period  
        """
        pass

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        calculates the marginal cost for the given power
        """
        pass

    def calculate_ramp(
        self, previous_power: float, power: float, current_power: float = 0
    ) -> float:
        """
        calculates the ramp for the given power
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
        """
        emission_cost = self.emission_factor * prices["co"].mean()
        fuel_cost = prices[self.technology.replace("_combined", "")].mean()
        return (fuel_cost + emission_cost) / self.efficiency

    def get_operation_time(self, start: datetime):
        """returns the operation time
        if unit is on since 4 hours, it returns 4
        if the unit is off since 4 hours, it returns -4
        """
        max_time = max(self.min_operating_time, self.min_down_time)
        begin = start - self.index.freq * max_time
        end = start
        arr = self.outputs["energy"][begin:end][::-1] > 0
        is_off = not arr[0]
        runn = 0
        for val in arr:
            if val == is_off:
                break
            runn += 1
        return (-1) ** is_off * runn


class SupportsMinMaxCharge(BaseUnit):
    """
    Base Class used for Storage derived classes
    """
    # initial state of charge
    initial_soc: float
    # how much power must be charged at least in one time step
    min_power_charge: float
    # how much power can be charged at most in one time step
    max_power_charge: float
    # how much power must be discharged at least in one time step
    min_power_discharge: float
    # how much power can be discharged at most in one time step
    max_power_discharge: float
    # how much power can be increased in discharging in one time step
    ramp_up_discharge: float
    # how much power can be decreased in discharging in one time step
    ramp_down_discharge: float
    # how much power can be increased in charging in one time step
    ramp_up_charge: float
    # how much power can be decreased in charging in one time step
    ramp_down_charge: float
    # maximum volume of the storage
    max_volume: float
    # efficiency of charging
    efficiency_charge: float
    # efficiency of discharging 
    efficiency_discharge: float

    def calculate_min_max_charge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        calculates the min and max charging power for the given time period 
        """
        pass

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        calculates the min and max discharging power for the given time period
        """
        pass

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        calculates the marginal cost for the given power
        """
        pass

    def get_soc_before(self, dt: datetime) -> float:
        """
        return SoC before the given datetime.
        If datetime is before the start of the index, the initial SoC is returned.
        The SoC is a float between 0 and 1.
        """
        if dt - self.index.freq < self.index[0]:
            return self.initial_soc
        else:
            return self.outputs["soc"].at[dt - self.index.freq]

    def get_clean_spread(self, prices: pd.DataFrame):
        """
        returns the clean spread for the given prices
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
        calculates the ramp for the given discharging power
        """
        if power_discharge == 0:
            return power_discharge
        power_discharge = min(
            power_discharge,
            previous_power + self.ramp_up_discharge - current_power,
        )

        power_discharge = max(
            power_discharge,
            previous_power - self.ramp_down_discharge - current_power,
        )
        return power_discharge

    def calculate_ramp_charge(
        self, previous_power: float, power_charge: float, current_power: float = 0
    ) -> float:
        """
        calculates the ramp for the given charging power
        """
        if power_charge == 0:
            return power_charge
        power_charge = max(
            power_charge, previous_power + self.ramp_up_charge - current_power
        )
        power_charge = min(
            power_charge, previous_power - self.ramp_down_charge - current_power
        )
        return power_charge


class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


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
        raise NotImplementedError()

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        pass


class LearningStrategy(BaseStrategy):
    """
    A strategy which provides learning functionality, has a method to calculate the reward.
    """

    obs_dim: int
    act_dim: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_dim = kwargs.get("observation_dimension", 50)
        self.act_dim = kwargs.get("action_dimension", 2)
