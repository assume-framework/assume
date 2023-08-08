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
        node: str,
        forecaster: Forecaster = None,
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

    def get_output_before(
        self, datetime: datetime, product_type: str = "energy"
    ) -> float:
        if datetime - self.index.freq < self.index[0]:
            return 0
        else:
            return self.outputs["energy"].at[datetime - self.index.freq]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "technology": self.technology,
            "unit_operator": self.unit_operator,
            "unit_type": "base_unit",
        }

    def calculate_cashflow(self, product_type: str, orderbook: Orderbook):
        pass


class SupportsMinMax(BaseUnit):
    """
    Base Class used for Powerplant derived classes
    """

    min_power: float
    max_power: float
    ramp_down: float
    ramp_up: float

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        pass

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        pass

    def calculate_ramp(
        self, previous_power: float, power: float, current_power: float = 0
    ) -> float:
        if power == 0:
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


class SupportsMinMaxCharge(BaseUnit):
    """
    Base Class used for Storage derived classes
    """

    min_power_charge: float
    max_power_charge: float
    min_power_discharge: float
    max_power_discharge: float
    ramp_up_discharge: float
    ramp_down_discharge: float
    ramp_up_charge: float
    ramp_down_charge: float

    def calculate_min_max_charge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        pass

    def calculate_min_max_discharge(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        pass

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        pass

    def calculate_ramp_discharge(
        self,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
    ) -> float:
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
