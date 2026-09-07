# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""EV aggregation through the existing UnitsOperator portfolio interface."""

import math

import pandas as pd
import pyomo.environ as pyo

from assume.common.base import MinMaxChargeStrategy
from assume.common.utils import timestamp2datetime
from assume.strategies.portfolio_strategies import UnitOperatorStrategy


class EVPortfolioMemberStrategy(MinMaxChargeStrategy):
    """Registration and tariff-feedback hooks for operator-controlled EVs."""

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        raise RuntimeError(
            "EV portfolio members require units_operator_ev on their operator"
        )

    def calculate_reward(self, unit, marketconfig, orderbook):
        if marketconfig.product_type == "grid_fee":
            for order in orderbook:
                unit.outputs["grid_fee_announced"].loc[
                    order["start_time"] : order["end_time"] - unit.index.freq
                ] = 1


class EVPortfolioStrategy(UnitOperatorStrategy):
    """Price-taking charging/V2G optimisation for an EV operator.

    ``perfect_foresight`` uses the remaining simulation; ``rolling_horizon``
    uses ``ev_look_ahead_horizon``. Both replan on each opening from accepted
    energy and fix already-cleared products. Terminal SOC defaults to initial
    SOC to avoid liquidating batteries at the horizon boundary. Prices are the
    unit's EOM forecast, including configured forecast updates such as tariffs.

    Bids retain EV ids and nodes for standard clearing/settlement. This is an
    EV-only portfolio. Optional limits apply to net portfolio import/export,
    including an exogenous background load (positive for consumption). Grid
    fees apply to positive net portfolio imports, without export credits.
    Peak pricing covers the optimisation horizon; rolling callers must pass
    the peak already observed in the same billing period.
    """

    def __init__(
        self,
        *args,
        ev_horizon_mode="rolling_horizon",
        ev_look_ahead_horizon="48h",
        ev_terminal_soc=None,
        ev_import_limit_mw=None,
        ev_export_limit_mw=None,
        ev_peak_price=0.0,
        ev_observed_peak_mw=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ev_horizon_mode not in ("perfect_foresight", "rolling_horizon"):
            raise ValueError("Unknown EV horizon mode")
        self.horizon_mode = ev_horizon_mode
        self.look_ahead = pd.Timedelta(ev_look_ahead_horizon).to_pytimedelta()
        if self.look_ahead.total_seconds() <= 0:
            raise ValueError("EV look-ahead must be positive")
        self.terminal_soc = ev_terminal_soc
        self.import_limit = ev_import_limit_mw
        self.export_limit = ev_export_limit_mw
        self.peak_price = float(ev_peak_price)
        self.observed_peak = float(ev_observed_peak_mw)
        for value in (
            self.import_limit,
            self.export_limit,
            self.peak_price,
            self.observed_peak,
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise ValueError("EV limits and peak prices must be finite/nonnegative")

    def optimize(
        self,
        units,
        start,
        end,
        market_id="EOM",
        *,
        grid_fees=None,
        background_load=None,
    ):
        """Return {unit_id: {timestamp: signed MW}} without changing dispatch."""
        from assume.units.electric_vehicle import ElectricVehicleUnit

        if not units:
            return {}
        if not all(isinstance(u, ElectricVehicleUnit) for u in units):
            raise TypeError(
                "EV portfolio strategies require only ElectricVehicleUnit units"
            )
        times = list(units[0].index[start : end - units[0].index.freq])
        if not times:
            return {}
        if any(
            u.index.freq != units[0].index.freq or any(t not in u.index for t in times)
            for u in units
        ):
            raise ValueError("EV portfolio indices must align")
        hours = units[0].index.freq.total_seconds() / 3600

        def profile(values, name, nonnegative=False):
            result = (
                [0.0] * len(times)
                if values is None
                else [float(values[t]) for t in times]
            )
            if any(not math.isfinite(v) or (nonnegative and v < 0) for v in result):
                raise ValueError(
                    f"{name} must be finite" + ("/nonnegative" if nonnegative else "")
                )
            return result

        fees = profile(grid_fees, "Grid fees", nonnegative=True)
        background = profile(background_load, "Background load")
        model = pyo.ConcreteModel()
        model.U = pyo.RangeSet(0, len(units) - 1)
        model.T = pyo.RangeSet(0, len(times) - 1)
        model.charge = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals)
        model.discharge = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals)
        model.energy = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals)
        model.charging = pyo.Var(model.U, model.T, domain=pyo.Binary)
        model.constraints = pyo.ConstraintList()
        cost = 0
        for i, unit in enumerate(units):
            terminal = (
                unit.initial_soc
                if self.terminal_soc is None
                else float(self.terminal_soc)
            )
            if not unit.min_soc <= terminal <= unit.max_soc:
                raise ValueError("EV terminal SOC must lie within SOC bounds")
            initial_energy = unit.energy_at(start)
            for k, t in enumerate(times):
                c, d, e = model.charge[i, k], model.discharge[i, k], model.energy[i, k]
                availability = unit.forecaster.availability.at[t]
                model.constraints.add(
                    c <= -unit.max_power_charge * availability * model.charging[i, k]
                )
                model.constraints.add(
                    d
                    <= unit.max_power_discharge
                    * availability
                    * (1 - model.charging[i, k])
                )
                previous = initial_energy if k == 0 else model.energy[i, k - 1]
                model.constraints.add(
                    e
                    == previous
                    + hours
                    * (c * unit.efficiency_charge - d / unit.efficiency_discharge)
                    - unit.trip_energy_consumption.at[t]
                )
                model.constraints.add(
                    pyo.inequality(
                        unit.min_soc * unit.capacity, e, unit.max_soc * unit.capacity
                    )
                )
                if unit.outputs["energy_committed"].at[t]:
                    model.constraints.add(d - c == unit.outputs["energy"].at[t])
                price = unit.forecaster.price[market_id].at[t]
                cost += hours * (
                    price * (c - d)
                    + unit.additional_cost_charge * c
                    + unit.additional_cost_discharge * d
                )
            model.constraints.add(
                model.energy[i, len(times) - 1] >= terminal * unit.capacity
            )
        model.imports = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.peak = pyo.Var(
            domain=pyo.NonNegativeReals, bounds=(self.observed_peak, None)
        )
        for k in model.T:
            withdrawal = sum(
                model.charge[i, k] - model.discharge[i, k] for i in model.U
            )
            model.constraints.add(model.imports[k] >= withdrawal)
            model.constraints.add(model.peak >= model.imports[k])
            if self.import_limit is not None:
                model.constraints.add(
                    withdrawal + background[k] <= float(self.import_limit)
                )
            if self.export_limit is not None:
                model.constraints.add(
                    withdrawal + background[k] >= -float(self.export_limit)
                )
            cost += hours * fees[k] * model.imports[k]
        cost += self.peak_price * (model.peak - self.observed_peak)
        model.objective = pyo.Objective(expr=cost)
        result = pyo.SolverFactory("appsi_highs").solve(model, load_solutions=False)
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"EV portfolio infeasible or unsolved at {start}: {result.solver.termination_condition}"
            )
        model.solutions.load_from(result)
        return {
            u.id: {
                t: pyo.value(model.discharge[i, k] - model.charge[i, k])
                for k, t in enumerate(times)
            }
            for i, u in enumerate(units)
        }

    def calculate_bids(self, units_operator, market_config, product_tuples, **kwargs):
        if not product_tuples or not units_operator.units:
            return []
        if market_config.product_type not in ("energy", "grid_fee"):
            raise ValueError("EV portfolio supports energy and grid_fee products")
        units = list(units_operator.units.values())
        index = units[0].index
        now = timestamp2datetime(units_operator.context.current_timestamp)
        start = next((t for t in index if t > now), None)
        if start is None:
            return []
        end = index[-1] + index.freq
        # Markets can announce products beyond the simulation's data boundary.
        product_tuples = [p for p in product_tuples if p[0] < end]
        if not product_tuples:
            return []
        if self.horizon_mode == "rolling_horizon":
            end = min(end, start + self.look_ahead)
            # Do not impose the terminal SOC in the middle of a trip or before
            # there has been enough time to recharge after its return.
            nominal_end = end
            hours = index.freq.total_seconds() / 3600
            for unit in units:
                away = [
                    t
                    for t in index
                    if start <= t < nominal_end
                    and not unit.forecaster.availability.at[t]
                ]
                if not away or unit.max_power_charge == 0:
                    continue
                target = (
                    unit.initial_soc
                    if self.terminal_soc is None
                    else float(self.terminal_soc)
                )
                recharge = max(0, target - unit.min_soc) * unit.capacity
                for t in index:
                    if t <= away[-1]:
                        continue
                    if unit.forecaster.availability.at[t]:
                        recharge -= (
                            -unit.max_power_charge * unit.efficiency_charge * hours
                        )
                    else:
                        recharge += unit.trip_energy_consumption.at[t]
                    if recharge <= 0:
                        end = max(end, t + index.freq)
                        break
                else:
                    end = index[-1] + index.freq
        for left, right, only_hours in product_tuples:
            if (
                only_hours is not None
                or right - left != index.freq
                or left not in index
                or left < start
                or right > end
            ):
                raise ValueError(
                    "EV products must be full single steps within the planning horizon"
                )
        for unit in units:
            if unit.forecaster._registries is not None:
                unit.forecaster.update(unit=unit)
        plan = self.optimize(units, start, end)
        bids = []
        for unit in units:
            for t, power in plan[unit.id].items():
                unit.outputs["ev_planned_energy"].at[t] = power
            for k, (left, right, only_hours) in enumerate(product_tuples):
                power = plan[unit.id][left]
                if market_config.product_type == "grid_fee":
                    power = -max(-power, 1e-9)
                elif unit.outputs["energy_committed"].at[left]:
                    continue
                price = (
                    market_config.maximum_bid_price
                    if power <= 0
                    else market_config.minimum_bid_price
                )
                bids.append(
                    {
                        "start_time": left,
                        "end_time": right,
                        "only_hours": only_hours,
                        "volume": round(power / market_config.volume_tick)
                        if market_config.volume_tick
                        else power,
                        "price": round(price / market_config.price_tick)
                        if market_config.price_tick
                        else price,
                        "unit_id": unit.id,
                        "node": unit.node,
                        "agent_addr": units_operator.context.addr,
                        "bid_id": f"{unit.id}_{k + 1}",
                    }
                )
        return bids
