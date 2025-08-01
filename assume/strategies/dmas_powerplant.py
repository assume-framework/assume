# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    Reals,
    Var,
    maximize,
    quicksum,
    value,
)
from pyomo.opt import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
)

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_supported_solver

logger = logging.getLogger(__name__)


class DmasPowerplantStrategy(BaseStrategy):
    def __init__(self, steps=[-10, -1, 0, 1, 10], *args, **kwargs):
        """
        Initializes the strategy

        Args:
            steps (list): list of steps to optimize
            *args (list): additional arguments
            **kwargs (dict): additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.model = ConcreteModel("powerplant")
        self.opt = SolverFactory(get_supported_solver())
        self.steps = steps
        self.T = 24
        self.opt_results = {
            step: dict(
                power=np.zeros(self.T, float),
                emission=np.zeros(self.T, float),
                fuel=np.zeros(self.T, float),
                start=np.zeros(self.T, float),
                profit=np.zeros(self.T, float),
                obj=0,
            )
            for step in steps
        }

        self.prevented_start = dict(
            prevent=False, hours=np.zeros(self.T, float), delta=0
        )
        self.reduction_next_day = {}

    def build_model(
        self,
        unit: SupportsMinMax,
        start: datetime,
        hour_count: int,
        emission_prices,
        fuel_prices,
        power_prices,
        runtime: int = None,
        p0: float = None,
    ) -> None:
        """
        Builds the optimization model and returns the cashflow.

        Args:
            unit (SupportsMinMax): Unit to optimize.
            start (datetime.datetime): Start time.
            hour_count (int): Number of hours to optimize.
            emission_prices (numpy.ndarray): Emission prices.
            fuel_prices (numpy.ndarray): Fuel prices.
            power_prices (numpy.ndarray): Power prices.
            runtime (int, optional): Runtime of the unit. Defaults to None.
            p0 (float, optional): Initial power. Defaults to None.

        Returns:
            np.array: Cashflow.
        """
        runtime = runtime or unit.get_operation_time(start)
        p0 = p0 or unit.get_output_before(start)
        self.model.clear()
        tr = np.arange(hour_count)

        delta = unit.max_power - unit.min_power

        self.model.p_out = Var(tr, bounds=(0, unit.max_power), within=Reals)
        self.model.p_model = Var(tr, bounds=(0, delta), within=Reals)

        # states (on, ramp up, ramp down)
        self.model.z = Var(tr, within=Binary)
        self.model.v = Var(tr, within=Binary, initialize=False)
        self.model.w = Var(tr, within=Binary, initialize=False)

        # define constraint for output power
        self.model.real_power = ConstraintList()
        self.model.real_max = ConstraintList()
        # define constraint for model power
        self.model.model_min = ConstraintList()
        self.model.model_max = ConstraintList()
        # define constraint ramping
        self.model.ramping_up = ConstraintList()
        self.model.ramping_down = ConstraintList()
        # define constraint for run- and stop-time
        self.model.stop_time = ConstraintList()
        self.model.run_time = ConstraintList()
        self.model.states = ConstraintList()
        self.model.initial_on = ConstraintList()
        self.model.initial_off = ConstraintList()

        for t in tr:  # iterate over hours to optimize
            # output power of the plant
            self.model.real_power.add(
                self.model.p_out[t]
                == self.model.p_model[t] + self.model.z[t] * unit.min_power
            )
            if t < hour_count - 1:  # only the next day
                self.model.real_max.add(
                    self.model.p_out[t]
                    <= unit.min_power
                    * (self.model.z[t] + self.model.v[t + 1] + self.model.p_model[t])
                )
            # model power for optimization
            self.model.model_min.add(0 <= self.model.p_model[t])
            self.model.model_max.add(self.model.z[t] * delta >= self.model.p_model[t])
            # ramping (gradients)
            if t == 0:
                self.model.ramping_up_0 = Constraint(
                    expr=self.model.p_out[0] <= p0 + unit.ramp_up
                )
                self.model.ramping_down_0 = Constraint(
                    expr=self.model.p_out[0] >= p0 - unit.ramp_down
                )
            else:
                self.model.ramping_up.add(
                    self.model.p_model[t] - self.model.p_model[t - 1]
                    <= unit.ramp_up * self.model.z[t - 1]
                )
                self.model.ramping_down.add(
                    self.model.p_model[t - 1] - self.model.p_model[t]
                    <= unit.ramp_down * self.model.z[t]
                )
            # minimal run and stop time
            if t > unit.min_down_time:
                self.model.stop_time.add(
                    1 - self.model.z[t]
                    >= quicksum(
                        self.model.w[k] for k in range(t - unit.min_down_time, t)
                    )
                )
            if t > unit.min_operating_time:
                self.model.run_time.add(
                    self.model.z[t]
                    >= quicksum(
                        self.model.v[k] for k in range(t - unit.min_operating_time, t)
                    )
                )
            if t > 0:
                self.model.states.add(
                    self.model.z[t - 1]
                    - self.model.z[t]
                    + self.model.v[t]
                    - self.model.w[t]
                    == 0
                )

            if runtime > 0 and t < unit.min_operating_time - runtime:
                self.model.initial_on.add(self.model.z[t] == 1)
            elif runtime < 0 and t < unit.min_down_time - (-runtime):
                self.model.initial_off.add(self.model.z[t] == 0)

        # -> fuel costs
        fuel_cost = [
            (self.model.p_out[t] / unit.efficiency) * fuel_prices[t] for t in tr
        ]
        # -> emission costs
        emission_cost = [
            (self.model.p_out[t] / unit.efficiency * unit.emission_factor)
            * emission_prices[t]
            for t in tr
        ]
        # -> start costs
        start_cost = [self.model.v[t] * unit.cold_start_cost for t in tr]

        # -> profit and resulting cashflow
        profit = [self.model.p_out[t] * power_prices[t] for t in tr]
        cashflow = [
            profit[t] - (fuel_cost[t] + emission_cost[t] + start_cost[t]) for t in tr
        ]

        return cashflow

    def _set_results(
        self,
        unit: SupportsMinMax,
        emission_prices,
        fuel_prices,
        power_prices,
        start: datetime,
        step: int,
        hour_count: int,
    ) -> None:
        """
        sets the results of the optimization

        Args:
            unit(SupportsMinMax): unit to optimize
            emission_prices(numpy.ndarray): emission prices
            fuel_prices(numpy.ndarray): fuel prices
            power_prices(numpy.ndarray): power prices
            start(datetime.datetime): start time
            step(int): step
            hour_count(int): number of hours to optimize
        Returns:
            None: None

        """
        # -> output power
        tr = np.arange(hour_count)
        power = np.asarray([self.model.p_out[t].value for t in tr])
        self.opt_results[step]["power"] = power
        # TODO rounding really needed?
        self.opt_results[step]["power"][power < 0.1] = 0

        # -> emission costs
        self.opt_results[step]["emission"] = (
            power / unit.efficiency * unit.emission_factor * emission_prices
        )
        # -> fuel costs
        self.opt_results[step]["fuel"] = power / unit.efficiency * fuel_prices
        # -> start costs
        self.opt_results[step]["start"] = np.asarray(
            [self.model.v[t].value * unit.cold_start_cost for t in tr]
        )
        # -> profit
        self.opt_results[step]["profit"] = power_prices * power
        # -> sum cashflow
        self.opt_results[step]["obj"] = value(self.model.obj)

        if step == 0:
            end = start + unit.index.freq * (hour_count - 1)
            unit.outputs["fuel"].loc[start:end] = self.opt_results[step]["fuel"]
            unit.outputs["emission"].loc[start:end] = self.opt_results[step]["emission"]
            unit.outputs["start_ups"].loc[start:end] = self.opt_results[step]["start"]
            unit.outputs["profit"].loc[start:end] = self.opt_results[step]["profit"]
            unit.outputs["generation"].loc[start:end] = self.opt_results[step]["power"]
            # self.generation[
            #    str(unit.fuel_type).replace("_combined", "")
            # ] = self.opt_results[step]["power"]
            # self.generation["total"] = self.opt_results[step]["power"]

    def optimize(
        self,
        unit: SupportsMinMax,
        start: datetime,
        hour_count: int,
        prices: pd.DataFrame = None,
        steps: tuple = None,
    ) -> np.array:
        """
        optimizes the unit

        Args:
          unit(SupportsMinMax): unit to optimize
          start(datetime.datetime): start time
          hour_count(int): number of hours to optimize
          prices(pd.DataFrame): prices
          steps(tuple): steps to optimize

        Returns:
          np.array: generation

        """
        base_price = prices.copy()
        self.prevented_start = dict(
            prevent=False, hours=np.zeros(self.T, float), delta=0
        )
        hour_count2 = 2 * hour_count
        steps = steps or self.steps
        prices_24h = prices[:hour_count].copy()
        prices_48h = prices[:hour_count2].copy()
        try:
            fuel_prices = unit.forecaster.get_price(unit.fuel_type)
            emission_prices = unit.forecaster.get_price("co2")
        except KeyError:
            logger.error(f"no price for {unit.fuel_type=} with {steps=} and {start=}")
            raise Exception(f"No Fuel prices given for fuel {unit.fuel_type}")

        for step in steps:
            adjusted_price = base_price[:hour_count] + step
            cashflow = self.build_model(
                unit,
                start,
                hour_count,
                emission_prices.iloc[:hour_count],
                fuel_prices.iloc[:hour_count],
                adjusted_price,
            )
            self.model.obj = Objective(expr=quicksum(cashflow), sense=maximize)
            r = self.opt.solve(self.model)
            if (r.solver.status == SolverStatus.ok) and (
                r.solver.termination_condition == TerminationCondition.optimal
            ):
                logger.debug("find optimal solution in step: %s", step)

                self._set_results(
                    unit,
                    emission_prices.iloc[:hour_count],
                    fuel_prices.iloc[:hour_count],
                    adjusted_price,
                    start=start,
                    step=step,
                    hour_count=hour_count,
                )

                if self.opt_results[step]["power"][-1] == 0 and step == 0:
                    all_off = np.argwhere(
                        self.opt_results[step]["power"] == 0
                    ).flatten()
                    last_on = np.argwhere(self.opt_results[step]["power"] > 0).flatten()
                    last_on = last_on[-1] if len(last_on) > 0 else 0
                    prevented_off_hours = list(all_off[all_off > last_on])

                    # simulate powerplant is turned off
                    runtime = -len(prevented_off_hours)
                    p0 = 0
                    cashflow = self.build_model(
                        unit,
                        start,
                        hour_count,
                        emission_prices.iloc[:hour_count],
                        fuel_prices.iloc[:hour_count],
                        prices_24h,
                        runtime,
                        p0,
                    )
                    self.model.obj = Objective(expr=quicksum(cashflow), sense=maximize)
                    self.opt.solve(self.model)
                    total_obj_single = self.opt_results[step]["obj"] + value(
                        self.model.obj
                    )
                    tr = np.arange(hour_count)
                    power_day1 = list(self.opt_results[step]["power"])
                    power_day2 = [self.model.p_out[t].value for t in tr]
                    total_single_power = np.asarray(power_day1 + power_day2)

                    all_off = np.argwhere(total_single_power == 0).flatten()
                    prevented_off_hours = np.asarray(list(all_off[all_off > last_on]))

                    self.build_model(
                        unit,
                        start,
                        hour_count2,
                        emission_prices.iloc[:hour_count2],
                        fuel_prices.iloc[:hour_count2],
                        prices_48h,
                        runtime,
                        p0,
                    )
                    tr = np.arange(hour_count2)
                    self.model.obj = Objective(expr=quicksum(cashflow), sense=maximize)
                    self.opt.solve(self.model)
                    power_check = np.asarray([self.model.p_out[t].value for t in tr])
                    prevent_start = all(power_check[prevented_off_hours] > 0)
                    delta = value(self.model.obj) - total_obj_single
                    if prevent_start and delta > 0:
                        delta /= sum(power_check[prevented_off_hours])
                        prevent_start_today = prevented_off_hours[
                            prevented_off_hours < self.T
                        ]
                        self.prevented_start = dict(
                            prevent=True, hours=prevent_start_today, delta=delta
                        )
                        prevent_start_tomorrow = (
                            prevented_off_hours[prevented_off_hours >= self.T] - self.T
                        )
                        self.reduction_next_day[start.date()] = (
                            delta,
                            prevent_start_tomorrow,
                        )

            else:
                if r.solver.termination_condition == TerminationCondition.infeasible:
                    logger.error(f"infeasible model in step: {step}")
                else:
                    logger.error(f"{step} - {r.solver}")
                for key in ["power", "emission", "fuel", "start", "profit"]:
                    self.opt_results[step][key] = np.zeros(self.T)
                self.opt_results[step]["obj"] = 0
        return unit.outputs["generation"].loc[start:]

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market
        Returns a list of bids that the unit operator will submit to the market

        This works by optimizing multiple times for different pricing levels.
        From the optimization result, the relevant block bid scenarios are created.
        For the efficiency, a good price forecast is required.

        The power plant assumes, that it is started in the first hour as a base block if possible.
        As it is not possible to combine block bids with exclusive bids, a few decisions on the behavior have to be made.
        One of these is the decision of to which block hours the startup cost should be added.
        Ideally, the difference is negligible, as the market takes care of the result optimization.
        Yet this can have a significant impact if the decision has to be made if the startup cost
        should be added to the last hour of the current clearing or to the first hours of the following.

        TODO: ramp_down constraints are not yet respected by adding an additional block for the next day

        Args:
          unit(SupportsMinMax): unit to dispatch
          market_config(MarketConfig): market configuration
          product_tuples(list[Product]): list of products to dispatch
          kwargs(dict): additional arguments

        Returns:
          Orderbook: orderbook

        """
        if "block_id" not in market_config.additional_fields:
            raise Exception("Block Id missing from Marketconfig")
        if "link" not in market_config.additional_fields:
            raise Exception("link missing from Marketconfig")
        start = product_tuples[0][0]
        hour_count = len(product_tuples)
        hour_count2 = hour_count * 2

        base_price = unit.forecaster[f"price_{market_config.market_id}"][
            start : start + timedelta(hours=hour_count2 - 1)
        ]
        e_price = unit.forecaster.get_price("co2")[
            start : start + timedelta(hours=hour_count2 - 1)
        ]
        fuel_price = unit.forecaster.get_price(unit.fuel_type)[
            start : start + timedelta(hours=hour_count2 - 1)
        ]
        self.optimize(unit, start, hour_count, base_price)

        def get_cost(p: float, t: int):
            f = fuel_price[t]
            e = e_price[t]
            return (p / unit.efficiency) * (f + e * unit.emission_factor)

        def get_marginal(p0: float, p1: float, t: int):
            if p0 == p1:
                return 1, 0
            marginal = (get_cost(p=p0, t=t) - get_cost(p=p1, t=t)) / (p0 - p1)
            return marginal, p1 - p0

        def get_maximal_profit_hours(base_price, start=0):
            """
            decides when to turn the powerplant on if it is currently off

            we calculate the sliding window and see when the most profitable time is to turn on.
            This does search the most profitable hours, even if the first hour might be sufficient to match our marginal costs
            """
            max_profit, start_hour = 0, 0
            run_time = unit.min_operating_time
            for t in range(start, hour_count - run_time):
                p = np.sum(unit.min_power * base_price[t : t + run_time])

                if p > max_profit:
                    max_profit = p
                    start_hour = t
            return list(
                range(
                    start_hour,
                    min(start_hour + run_time, hour_count),
                )
            )

        order_book = {}
        last_power = np.zeros(hour_count)
        block_number = 0
        tr = np.arange(hour_count)
        links = {i: None for i in tr}

        max_hours = get_maximal_profit_hours(base_price)
        start_cost = unit.cold_start_cost / max(unit.min_power, 10) ** 2

        yesterday = start.date() - timedelta(days=1)

        index = 0
        runtime = unit.get_operation_time(start)

        # TODO add ramping constraints from
        # unit.calculate_ramp(runtime, last_power, unit.min_power)
        # unit.ramp_down

        while index < len(self.steps):
            step = self.steps[index]

            # -> get optimization result for key (block) and step
            result = self.opt_results[step]
            # if we are in hour 0
            if any(result["power"] > 0) and block_number == 0:
                # pwp is on and must runtime is reached
                if result["power"][0] > 0 and runtime > 0:
                    reduction = 0
                    hours_needed_to_run = unit.min_operating_time - runtime
                    hours = (
                        list(range(hours_needed_to_run))
                        if hours_needed_to_run > 0
                        else [0]
                    )
                    # -> a start is prevented
                    if yesterday in self.reduction_next_day.keys():
                        reduction, hours = self.reduction_next_day[yesterday]
                        self.reduction_next_day = dict()
                elif runtime < 0:
                    # pwp is off
                    hours = max_hours
                    reduction = -start_cost
                else:
                    # pwp was on but turned off in first hour
                    hours = max_hours
                    reduction = -start_cost

                for hour in hours:
                    price, power = get_marginal(
                        p0=last_power[hour], p1=unit.min_power, t=hour
                    )
                    order_book.update(
                        {
                            (block_number, hour, unit.id): (
                                price - reduction,
                                power,
                                -1,
                            )
                        }
                    )
                    links[hour] = block_number
                    last_power[hour] += unit.min_power

                block_number += 1  # -> increment block number

            # -> stack on top
            # XXX bitwise and operator
            hours = np.argwhere(
                (result["power"] - last_power > 0.1) & (last_power > 0)
            ).flatten()
            for hour in hours:
                price, power = get_marginal(
                    p0=last_power[hour], p1=result["power"][hour], t=hour
                )
                order_book.update(
                    {(block_number, hour, unit.id): (price, power, links[hour])}
                )
                last_power[hour] += power
                links[hour] = block_number
                block_number += 1
            # -> stack before
            hours = np.argwhere(
                (result["power"] - last_power > 0.1) & (last_power == 0)
            ).flatten()
            first_on_hour = (
                np.argwhere(last_power == 0)[-1][0] + 1
                if len(np.argwhere(last_power == 0))
                else 0
            )
            first_on_hour = 0 if first_on_hour > 23 else first_on_hour
            first_hours = list(hours[hours < first_on_hour])
            # -> no gap between current (mother) block and new block on the left side
            if first_hours:
                set_hours = []
                delta_hour = first_on_hour - first_hours[-1]
                if delta_hour == 1:
                    first_hours.reverse()
                    for hour in first_hours:
                        if links.get(hour + 1) is not None:
                            price, power = get_marginal(
                                p0=last_power[hour], p1=result["power"][hour], t=hour
                            )
                            order_book.update(
                                {
                                    (block_number, hour, unit.id): (
                                        price,
                                        power,
                                        links[hour + 1],
                                    )
                                }
                            )
                            last_power[hour] += power
                            links[hour] = block_number
                            block_number += 1
                            set_hours += [hour]
                        else:
                            break
                first_hours = list(set(first_hours) - set(set_hours))
                if delta_hour > 1 or len(first_hours) > 0:
                    total_start_cost = result["start"][first_hours[0]]
                    result["start"][first_hours] = total_start_cost / (
                        unit.min_power * len(first_hours)
                    )
                    # -> add new mother block before another mother block
                    # -> this means that a new start is added before a start in a previous step
                    for hour in first_hours:
                        price, power = get_marginal(
                            p0=last_power[hour],
                            p1=unit.min_power,
                            t=hour,
                        )
                        order_book.update(
                            {(block_number, hour, unit.id): (price, power, -1)}
                        )
                        last_power[hour] += unit.min_power
                        links[hour] = block_number
                    block_number += 1

                    for hour in first_hours:
                        if result["power"][hour] > last_power[hour]:
                            price, power = get_marginal(
                                p0=last_power[hour], p1=result["power"][hour], t=hour
                            )
                            order_book.update(
                                {
                                    (block_number, hour, unit.id): (
                                        price,
                                        power,
                                        links[hour - 1],
                                    )
                                }
                            )
                            last_power[hour] += power
                            links[hour] = block_number
                            block_number += 1

            # -> stack behind
            last_on_hour = (
                np.argwhere(last_power > 0)[-1][0]
                if len(np.argwhere(last_power > 0))
                else tr[-1]
            )
            last_on_hours = list(hours[hours > last_on_hour])
            for hour in last_on_hours:
                if links[hour - 1] is None:
                    # we need to start mid day
                    # first update the next efficient start
                    max_hours = get_maximal_profit_hours(base_price, hour)

                    if all(result["power"][max_hours] > 0):
                        # pwp is on and must runtime is not reached or it is turned off and started later
                        for t in max_hours:
                            price, power = get_marginal(
                                p0=last_power[t], p1=unit.min_power, t=t
                            )
                            order_book.update(
                                {
                                    (block_number, t, unit.id): (
                                        price + start_cost,
                                        power,
                                        -1,
                                    )
                                }
                            )
                            links[t] = block_number
                            last_power[t] += unit.min_power
                    block_number += 1
                    index -= 1
                    break
                else:
                    if result["power"][hour] > last_power[hour]:
                        price, power = get_marginal(
                            p0=last_power[hour], p1=result["power"][hour], t=hour
                        )
                        order_book.update(
                            {
                                (block_number, hour, unit.id): (
                                    price,
                                    power,
                                    links[hour - 1],
                                )
                            }
                        )
                        last_power[hour] += power
                        links[hour] = block_number
                        block_number += 1
            index += 1
        if order_book:
            df = pd.DataFrame.from_dict(order_book, orient="index")
        else:
            # if nothing in self.portfolio.energy_systems
            df = pd.DataFrame(columns=["price", "volume", "link"])

        df.columns = ["price", "volume", "link"]
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["block_id", "hour", "bid_id"]
        )

        if self.prevented_start["prevent"]:
            hours = self.prevented_start["hours"]

            def get_marginals(x):
                return get_marginal(p0=0, p1=unit.min_power, t=x)

            min_price = (
                np.mean([price for price, _ in map(get_marginals, hours)])
                - self.prevented_start["delta"]
            )

            # -> volume and price which is already in orderbook
            normal_volume = df.loc[:, df.index.get_level_values("hour").isin(hours), :][
                "volume"
            ]
            normal_price = df.loc[:, df.index.get_level_values("hour").isin(hours), :][
                "price"
            ]
            # -> drop volume and price in these hours and build new orders
            df = df.loc[~df.index.get_level_values("hour").isin(hours)]
            # -> get last block to link on
            last_block = (
                max(df.index.get_level_values("block_id").values) if len(df) > 0 else -1
            )

            # -> build new orders
            prev_order = {}
            block_number = last_block + 1
            # -> for each hour build one block with minPower
            for hour in hours:
                prev_order[(block_number, hour, unit.id)] = (
                    min_price,
                    unit.min_power,
                    last_block,
                    "generation",
                )
            last_block = block_number
            block_number += 1
            for index in normal_volume.index:
                vol = normal_volume.loc[index] - unit.min_power
                prc = normal_price.loc[index]
                _, hour, _ = index
                if vol > 0:
                    prev_order[(block_number, hour, unit.id)] = (
                        prc,
                        vol,
                        last_block,
                        "generation",
                    )
                    block_number += 1

            df_prev = pd.DataFrame.from_dict(prev_order, orient="index")
            df_prev.columns = ["price", "volume", "link"]
            df_prev.index = pd.MultiIndex.from_tuples(
                df_prev.index, names=["block_id", "hour", "bid_id"]
            )
            # -> limit to market price range
            df = pd.concat([df, df_prev], axis=0)
            df.loc[df["price"] < -500 / 1e3, "price"] = -500 / 1e3
        if not df.empty:
            df = df.reset_index()
            df["start_time"] = df.apply(
                lambda o: start + timedelta(hours=o["hour"]), axis=1
            )
            df["end_time"] = df.apply(
                lambda o: start + timedelta(hours=o["hour"]) + unit.index.freq, axis=1
            )
            del df["hour"]
            df["exclusive_id"] = None
        df["unit_id"] = unit.id
        return df.to_dict("records")
