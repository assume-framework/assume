# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    Objective,
    Reals,
    Var,
    maximize,
    quicksum,
)
from pyomo.opt import SolverFactory, check_available_solvers

from assume.common.base import BaseStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product


def shift(prc, type_: str = "first"):
    """
    Shifts the price curve up or down

    Args:
      prc(numpy.ndarray): price curve
      type_(str): type of shift. default is 'first'
      type_: str:  (Default value = "first")

    Returns:
      np.array: shifted price curve

    """
    new_prices, num_prices = [], len(
        prc
    )  # new_prices is the list of prices after the shift

    for p, index in zip(prc, range(num_prices)):
        if type_ == "first":
            new_prices += [
                p * (0.9 + index * 0.2 / num_prices)
            ]  # calculate the new price by multiplying the old price with a factor
        else:
            new_prices += [p * (1.1 - index * 0.2 / num_prices)]

    return np.asarray(new_prices)


def shaping(prc, type_: str = "peak"):
    """
    Shifts the price curve up or down

    Args:
      prc(numpy.ndarray): price curve
      type_(str): type of shift. default is 'peak'. other options are 'pv' and 'demand'.
      type_: str:  (Default value = "peak")

    Returns:
      np.array: shifted price curve

    """
    if type_ == "peak":
        prc[
            8:20
        ] *= 1.1  # between 8 and 20 hours, the price is multiplied by 1.1 (general production is high)
    elif type_ == "pv":
        prc[
            10:14
        ] *= 0.9  # between 10 and 14 hours, the price is multiplied by 0.9 (Pv production is high)
    elif type_ == "demand":
        prc[
            6:9
        ] *= 1.1  # between 6 and 9 hours, the price is multiplied by 1.1 (demand is high)
        prc[17:20] *= 1.1  # between 17 and 20 hours, the price is multiplied by 1.1
    return prc


PRICE_FUNCS = {
    "left": lambda prc: np.roll(prc, -1),
    "right": lambda prc: np.roll(prc, 1),
    "normal": lambda prc: prc,
    # 'first': lambda prc: shift(prc, type_='first'),
    # 'last': lambda prc: shift(prc, type_='last'),
    # 'peak_off_peak': lambda prc: shaping(prc, type_='peak'),
    "pv_sink:": lambda prc: shaping(prc, type_="pv"),
    "demand": lambda prc: shaping(prc, type_="demand"),
}


def get_solver_factory(solvers_str=["glpk", "cbc", "gurobi", "cplex"]) -> SolverFactory:
    """
    Returns the first available solver from the list of solvers

    Args:
        solvers_str (list, optional): default solvers. Defaults to ["glpk", "cbc", "gurobi", "cplex"].

    Raises:
        Exception: if no solvers available

    Returns:
        SolverFactory: the first working solver
    """
    solvers = check_available_solvers(*solvers_str)
    if len(solvers) < 1:
        raise Exception(f"None of {solvers_str} are available")
    return SolverFactory(solvers[0])


class DmasStorageStrategy(BaseStrategy):
    """Strategy for a storage unit that uses DMAS to optimize its operation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = ConcreteModel("storage")
        self.opt = get_solver_factory()

    def build_model(self, unit: SupportsMinMaxCharge, start: datetime, hour_count: int):
        """
        Builds the optimization model

        Args:
            unit (SupportsMinMaxCharge): unit to dispatch
            start (datetime.datetime): start time
            hour_count (int): number of hours to optimize

        Returns:
            np.array: power

        """
        self.model.clear()
        time_range = range(hour_count)

        self.model.p_plus = Var(
            time_range, within=Reals, bounds=(0, -unit.max_power_charge)
        )
        self.model.p_minus = Var(
            time_range, within=Reals, bounds=(0, unit.max_power_discharge)
        )
        self.model.volume = Var(time_range, within=Reals, bounds=(0, unit.max_volume))

        self.power = [
            -self.model.p_minus[t] / unit.efficiency_discharge
            + self.model.p_plus[t] * unit.efficiency_charge
            for t in time_range
        ]

        self.model.vol_con = ConstraintList()
        soc0 = unit.get_soc_before(start)
        v0 = unit.max_volume * soc0

        for t in time_range:
            if t == 0:
                self.model.vol_con.add(self.model.volume[t] == v0 + self.power[t])
            else:
                self.model.vol_con.add(
                    self.model.volume[t] == self.model.volume[t - 1] + self.power[t]
                )

        # always end with half full SoC
        self.model.vol_con.add(self.model.volume[hour_count - 1] == unit.max_volume / 2)
        return self.power

    def optimize(
        self,
        unit: SupportsMinMaxCharge,
        market_id: str,
        start: datetime,
        hour_count: int,
    ):
        """
        Optimizes the unit operation

        Args:
            unit(SupportsMinMaxCharge): unit to dispatch
            start(datetime.datetime): start time
            hour_count(int): number of hours to optimize

        Returns:
          dict: optimization results
        """
        opt_results = {key: np.zeros(hour_count) for key in PRICE_FUNCS.keys()}
        time_range = range(hour_count)

        base_price = unit.forecaster[f"price_{market_id}"][
            start : start + timedelta(hours=hour_count)
        ]

        for key, func in PRICE_FUNCS.items():
            prices = func(base_price.values)
            self.power = self.build_model(unit, start, hour_count)
            profit = [-self.power[t] * prices[t] for t in time_range]
            self.model.obj = Objective(
                expr=quicksum(profit[t] for t in time_range), sense=maximize
            )
            r = self.opt.solve(self.model)
            power = np.asarray(
                [
                    -self.model.p_minus[t].value * unit.efficiency_discharge
                    + self.model.p_plus[t].value
                    for t in time_range
                ]
            )
            profit = [-power[t] * prices[t] for t in time_range]
            volume = np.asarray([self.model.volume[t].value for t in time_range])
            opt_results[key] = power
            if key == "normal":
                end = start + unit.index.freq * (hour_count - 1)

                # unit.outputs["total"][self.power < 0] = -self.power[self.power < 0]
                # unit.outputs["demand"][self.power > 0] = self.power[self.power > 0]
                # unit.outputs["storage"] = self.power
                unit.outputs["volume"].loc[start:end] = volume
                unit.outputs["generation"].loc[start:end] = power
                unit.outputs["profit"].loc[start:end] = profit
        return opt_results

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Returns a list of bids that the unit operator will submit to the market

        Args:
            unit (SupportsMinMaxCharge): unit to dispatch
            market_config (MarketConfig): market configuration
            product_tuples (list[Product]): list of products to dispatch
            **kwargs (dict): additional arguments

        Returns:
            Orderbook: bids

        """
        assert "exclusive_id" in market_config.additional_fields
        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        hour_count = (end - start) // timedelta(hours=1)
        opt_results = self.optimize(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            hour_count=hour_count,
        )
        total_orders = {}
        block_id = 0
        power_prices = unit.forecaster[f"price_{market_config.market_id}"][
            start : start + timedelta(hours=hour_count)
        ]
        for key, power in opt_results.items():
            prc = np.zeros(hour_count)
            bid_hours = np.argwhere(power < 0).flatten()
            ask_hours = np.argwhere(power > 0).flatten()
            if len(bid_hours) > 1:
                max_charging_price = power_prices.values[bid_hours].max()
            else:
                max_charging_price = 0
            min_discharging_price = max_charging_price / (
                unit.efficiency_discharge * unit.efficiency_discharge
            )
            prc[ask_hours] = (power_prices.iloc[ask_hours] + min_discharging_price) / 2
            prc[bid_hours] = power_prices.values[bid_hours]
            add = True
            for orders in total_orders.values():
                if any(prc != orders["price"]) or any(power != orders["volume"]):
                    add = False
            if add:
                total_orders[block_id] = dict(price=prc, volume=power)
                block_id += 1
        dfs = []
        time_range = range(hour_count)
        for block_id, bids in total_orders.items():
            df = pd.DataFrame(data=bids)
            df["bid_id"] = unit.id
            df["exclusive_id"] = block_id
            df["hour"] = time_range
            df = df.set_index(["exclusive_id", "hour", "bid_id"])
            dfs.append(df)

        bids = pd.concat(dfs)

        if not bids.empty:
            bids = bids.reset_index()
            bids["start_time"] = bids.apply(
                lambda o: start + timedelta(hours=o["hour"]), axis=1
            )
            bids["end_time"] = bids.apply(
                lambda o: start + timedelta(hours=o["hour"]) + unit.index.freq, axis=1
            )
            del bids["hour"]
        return bids.to_dict("records")
