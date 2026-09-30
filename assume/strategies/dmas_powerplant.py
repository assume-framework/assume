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

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_supported_solver_pyomo

logger = logging.getLogger(__name__)


class EnergyOptimizationDmasStrategy(MinMaxStrategy):
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
        self.opt = SolverFactory(get_supported_solver_pyomo())
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
    ) -> np.ndarray:
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
            np.ndarray: Cashflow.
        """
        runtime = runtime or unit.get_operation_time(start)
        p0 = p0 or unit.get_output_before(start)
        self.model.clear()
        tr = np.arange(hour_count)

        fuel_prices = np.asarray(fuel_prices, dtype=float)
        emission_prices = np.asarray(emission_prices, dtype=float)
        power_prices = np.asarray(power_prices, dtype=float)

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
                if unit.ramp_up is not None:
                    self.model.ramping_up_0 = Constraint(
                        expr=self.model.p_out[0] <= p0 + unit.ramp_up
                    )
                if unit.ramp_down is not None:
                    self.model.ramping_down_0 = Constraint(
                        expr=self.model.p_out[0] >= p0 - unit.ramp_down
                    )
            else:
                if unit.ramp_up is not None:
                    self.model.ramping_up.add(
                        self.model.p_model[t] - self.model.p_model[t - 1]
                        <= unit.ramp_up * self.model.z[t - 1]
                    )
                if unit.ramp_down is not None:
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
            if t == 0:
                z_initial = 1 if runtime > 0 else 0
                self.model.states.add(
                    z_initial - self.model.z[0] + self.model.v[0] - self.model.w[0] == 0
                )
            else:
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

        emission_prices = np.asarray(emission_prices, dtype=float)
        fuel_prices = np.asarray(fuel_prices, dtype=float)
        power_prices = np.asarray(power_prices, dtype=float)

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

    def _check_avoided_start_costs(
        self,
        unit: SupportsMinMax,
        start: datetime,
        hour_count: int,
        hour_count2: int,
        prices_48h,
        fuel_prices,
        emission_prices,
        step0_obj: float,
    ) -> None:
        """
        Evaluates a 48-hour rolling horizon to detect whether running at a slight loss
        in the final hours of the first day avoids a costly restart on the second day.
        """
        power_step0 = self.opt_results[0]["power"]
        if power_step0[-1] != 0:
            return

        all_off = np.argwhere(power_step0 == 0).flatten()
        last_on = np.argwhere(power_step0 > 0).flatten()
        last_on = last_on[-1] if len(last_on) > 0 else 0
        prevented_off_hours = list(all_off[all_off > last_on])
        if not prevented_off_hours:
            return

        # simulate powerplant is turned off on Day 2
        runtime_day2 = -len(prevented_off_hours)
        p0_day2 = 0
        day2_start = start + unit.index.freq * hour_count
        day2_prices = (
            prices_48h.iloc[hour_count:hour_count2]
            if hasattr(prices_48h, "iloc")
            else prices_48h[hour_count:hour_count2]
        )
        day2_emissions = (
            emission_prices.iloc[hour_count:hour_count2]
            if hasattr(emission_prices, "iloc")
            else emission_prices[hour_count:hour_count2]
        )
        day2_fuels = (
            fuel_prices.iloc[hour_count:hour_count2]
            if hasattr(fuel_prices, "iloc")
            else fuel_prices[hour_count:hour_count2]
        )

        cashflow_day2 = self.build_model(
            unit,
            day2_start,
            hour_count,
            day2_emissions,
            day2_fuels,
            day2_prices,
            runtime_day2,
            p0_day2,
        )
        self.model.obj = Objective(expr=quicksum(cashflow_day2), sense=maximize)
        self.opt.solve(self.model)
        total_obj_single = step0_obj + value(self.model.obj)
        tr = np.arange(hour_count)
        power_day1 = list(self.opt_results[0]["power"])
        power_day2 = [self.model.p_out[t].value for t in tr]
        total_single_power = np.asarray(power_day1 + power_day2)

        all_off = np.argwhere(total_single_power == 0).flatten()
        prevented_off_hours = np.asarray(list(all_off[all_off > last_on]))

        # 48h horizon starting from initial state at `start`
        runtime_day1 = unit.get_operation_time(start)
        p0_day1 = unit.get_output_before(start)
        cashflow_48h = self.build_model(
            unit,
            start,
            hour_count2,
            emission_prices.iloc[:hour_count2]
            if hasattr(emission_prices, "iloc")
            else emission_prices[:hour_count2],
            fuel_prices.iloc[:hour_count2]
            if hasattr(fuel_prices, "iloc")
            else fuel_prices[:hour_count2],
            prices_48h,
            runtime_day1,
            p0_day1,
        )
        tr_48 = np.arange(hour_count2)
        self.model.obj = Objective(expr=quicksum(cashflow_48h), sense=maximize)
        self.opt.solve(self.model)
        power_check = np.asarray([self.model.p_out[t].value for t in tr_48])
        delta = value(self.model.obj) - total_obj_single

        prevent_start = len(prevented_off_hours) > 0 and all(
            power_check[prevented_off_hours] > 0
        )
        off_power_sum = (
            sum(power_check[prevented_off_hours]) if len(prevented_off_hours) > 0 else 0
        )
        if prevent_start and delta > 0 and off_power_sum > 0:
            delta /= off_power_sum
            prevent_start_today = np.asarray(
                [h for h in prevented_off_hours if h < hour_count]
            )
            self.prevented_start = dict(
                prevent=True, hours=prevent_start_today, delta=delta
            )
            prevent_start_tomorrow = np.asarray(
                [h - hour_count for h in prevented_off_hours if h >= hour_count]
            )
            self.reduction_next_day[start.date()] = (
                delta,
                prevent_start_tomorrow,
            )

    def optimize(
        self,
        unit: SupportsMinMax,
        start: datetime,
        hour_count: int,
        prices: pd.DataFrame = None,
        steps: tuple = None,
    ) -> np.ndarray:
        """
        optimizes the unit

        Args:
          unit(SupportsMinMax): unit to optimize
          start(datetime.datetime): start time
          hour_count(int): number of hours to optimize
          prices(pd.DataFrame): prices
          steps(tuple): steps to optimize

        Returns:
          np.ndarray: generation

        """
        hour_count2 = 2 * hour_count
        if hasattr(prices, "iloc"):
            prices_24h = prices.iloc[:hour_count].copy()
            prices_48h = prices.iloc[:hour_count2].copy()
        else:
            prices_24h = prices[:hour_count].copy()
            prices_48h = prices[:hour_count2].copy()

        self.prevented_start = dict(
            prevent=False, hours=np.zeros(hour_count, float), delta=0
        )
        steps = steps or self.steps
        try:
            fuel_prices = unit.forecaster.fuel_prices[unit.fuel_type]
            emission_prices = unit.forecaster.fuel_prices["co2"]
        except KeyError:
            logger.error(f"no price for {unit.fuel_type=} with {steps=} and {start=}")
            raise Exception(f"No Fuel prices given for fuel {unit.fuel_type}")

        fuel_slice = (
            fuel_prices.iloc[:hour_count]
            if hasattr(fuel_prices, "iloc")
            else fuel_prices[:hour_count]
        )
        emission_slice = (
            emission_prices.iloc[:hour_count]
            if hasattr(emission_prices, "iloc")
            else emission_prices[:hour_count]
        )

        for step in steps:
            adjusted_price = prices_24h + step
            cashflow = self.build_model(
                unit,
                start,
                hour_count,
                emission_slice,
                fuel_slice,
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

                if step == 0:
                    self._check_avoided_start_costs(
                        unit,
                        start,
                        hour_count,
                        hour_count2,
                        prices_48h,
                        fuel_prices,
                        emission_prices,
                        step0_obj=self.opt_results[step]["obj"],
                    )

            else:
                if r.solver.termination_condition == TerminationCondition.infeasible:
                    logger.error(f"infeasible model in step: {step}")
                else:
                    logger.error(f"{step} - {r.solver}")
                for key in ["power", "emission", "fuel", "start", "profit"]:
                    self.opt_results[step][key] = np.zeros(hour_count)
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

        base_price = unit.forecaster.price[market_config.market_id][
            start : start + timedelta(hours=hour_count2 - 1)
        ]
        e_price = unit.forecaster.fuel_prices["co2"][
            start : start + timedelta(hours=hour_count2 - 1)
        ]
        fuel_price = unit.forecaster.fuel_prices[unit.fuel_type][
            start : start + timedelta(hours=hour_count2 - 1)
        ]
        self.optimize(unit, start, hour_count, base_price)

        def get_cost(p: float, t: int):
            f = fuel_price.iloc[t] if hasattr(fuel_price, "iloc") else fuel_price[t]
            e = e_price.iloc[t] if hasattr(e_price, "iloc") else e_price[t]
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
            max_profit, start_hour = -float("inf"), start
            run_time = max(unit.min_operating_time, 1)
            limit = max(start + 1, hour_count - run_time + 1)
            for t in range(start, limit):
                slice_prc = (
                    base_price.iloc[t : t + run_time]
                    if hasattr(base_price, "iloc")
                    else base_price[t : t + run_time]
                )
                p = np.sum(unit.min_power * np.asarray(slice_prc))

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
                    )
                    block_number += 1

            if prev_order:
                df_prev = pd.DataFrame.from_dict(prev_order, orient="index")
                df_prev.columns = ["price", "volume", "link"]
                df_prev.index = pd.MultiIndex.from_tuples(
                    df_prev.index, names=["block_id", "hour", "bid_id"]
                )
                df = pd.concat([df, df_prev], axis=0)
            min_bid = (
                market_config.minimum_bid_price
                if hasattr(market_config, "minimum_bid_price")
                and market_config.minimum_bid_price is not None
                else -500 / 1e3
            )
            df.loc[df["price"] < min_bid, "price"] = min_bid
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


PowerplantDmasStrategy = EnergyOptimizationDmasStrategy
