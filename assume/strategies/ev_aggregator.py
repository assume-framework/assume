# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""EV aggregation through the existing UnitsOperator portfolio interface.

An EV aggregator is modelled as a :class:`UnitsOperator` whose units are all
:class:`~assume.units.electric_vehicle.ElectricVehicleUnit` instances and whose
portfolio strategy is :class:`EVPortfolioStrategy`.  The individual vehicles
carry :class:`EVPortfolioMemberStrategy`, which only registers them on the
markets and routes tariff feedback back into their forecasts - they never bid
on their own.

The optimisation
----------------

:meth:`EVPortfolioStrategy.optimize` builds one **MILP** over the whole
portfolio and the whole planning horizon:

=========================  =================================================
Variable                   Meaning
=========================  =================================================
``charge[i, k]``           grid-side charging power of vehicle *i*, MW >= 0
``discharge[i, k]``        grid-side discharging power (V2G), MW >= 0
``energy[i, k]``           battery energy at the end of step *k*, MWh
``charging[i, k]``         binary: 1 = charging, 0 = discharging
``unmet[i, k]``            driving energy the battery could not supply, MWh
``shortfall[i]``           terminal-SOC miss at the horizon end, MWh
``imports[k]``             net withdrawal at the connection point, MW >= 0
``peak``                   largest ``imports`` over the horizon, MW
=========================  =================================================

Everything except ``charging`` is continuous, so the model is an LP with one
binary per vehicle and step.  That binary encodes the charge/discharge
disjunction.  It matters only when the *effective* price of energy - the market
price plus the grid fee - can go negative: a relaxed model would then run
``charge`` and ``discharge`` together to dissipate energy, freeing SOC headroom
to withdraw yet more energy it is paid to take.  Whenever the effective price
stays non-negative the LP relaxation is exactly tight, and the relaxation
solves two to three times faster, so :meth:`EVPortfolioStrategy.optimize`
solves the relaxation first and only re-solves as a MILP if the relaxed
solution actually charges and discharges the same vehicle at the same time.
This is a pure speed-up: the answer is identical either way, and the check is
on the solution rather than on a sufficient condition, so it cannot be fooled
by an unforeseen combination of fees, peak prices and connection limits.
``ev_force_milp=True`` skips the relaxation and goes straight to the MILP.

Soft constraints
----------------

Driving demand and the terminal SOC are enforced through penalised slacks
(``unmet``, ``shortfall``) rather than as hard constraints.  A hard constraint
turns a fleet that is merely short of energy into an unsolvable model, and an
aggregator that raises where the physics would simply arrive with a flatter
battery is not a useful model of an aggregator - the vehicle itself degrades
gracefully and books the gap in ``unmet_driving_energy``.  The penalty is far
above any plausible arbitrage value, so the slacks stay at zero whenever a
feasible schedule exists, and a nonzero slack is reported through
``outputs["ev_planned_unmet_energy"]`` and a warning.

Grid fees and the connection point
----------------------------------

The tariff is a property of the *connection*, not of a vehicle, so it is
modelled on ``imports``: the net withdrawal of the whole portfolio including
any exogenous ``background_load``.  This is what makes the fee asymmetric -
levied on import, never credited on export - and what makes the capacity charge
bill the peak that the transformer actually sees.  Folding the fee into each
vehicle's price forecast instead (:func:`price_plus_grid_fee`) cannot express
either property, so :meth:`calculate_bids` reads the announced fee itself and
takes the price from :func:`grid_fee_free_price` to avoid charging it twice.
"""

import logging
import math

import pandas as pd
import pyomo.environ as pyo

from assume.common.base import MinMaxChargeStrategy
from assume.common.forecast_algorithms import (
    effective_grid_fee,
    grid_fee_free_price,
)
from assume.common.utils import timestamp2datetime
from assume.strategies.grid_tariff import ANNOUNCED_FLAG as GRID_FEE_ANNOUNCED_FLAG
from assume.strategies.portfolio_strategies import UnitOperatorStrategy

logger = logging.getLogger(__name__)

#: Power below which charge/discharge is treated as zero when checking whether a
#: relaxed solution violated the charge/discharge disjunction, in MW.
SIMULTANEITY_TOLERANCE = 1e-7

#: Penalty on unserved driving energy and on missing the terminal SOC, in
#: EUR/MWh.  Chosen far above any credible spread between market price plus
#: grid fee, so the slacks are only ever used when no feasible schedule exists.
DEFAULT_SLACK_PENALTY = 1e6


class EVPortfolioMemberStrategy(MinMaxChargeStrategy):
    """Registration and tariff-feedback hooks for operator-controlled EVs.

    Vehicles in an EV portfolio are dispatched by their operator's
    :class:`EVPortfolioStrategy`, so this strategy deliberately refuses to bid.
    It exists so the vehicles register on the markets through the normal path
    and so grid-fee clearings are marked as announced on the unit, which is what
    lets the fee reach the portfolio optimisation.
    """

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        raise RuntimeError(
            "EV portfolio members require units_operator_ev on their operator"
        )

    def calculate_reward(self, unit, marketconfig, orderbook):
        if marketconfig.product_type == "grid_fee":
            for order in orderbook:
                unit.outputs[GRID_FEE_ANNOUNCED_FLAG].loc[
                    order["start_time"] : order["end_time"] - unit.index.freq
                ] = 1


class EVPortfolioStrategy(UnitOperatorStrategy):
    """Price-taking charging/V2G optimisation for an EV operator.

    See the module docstring for the structure of the optimisation problem.

    ``perfect_foresight`` plans over the remaining simulation; ``rolling_horizon``
    plans over ``ev_look_ahead_horizon``.  Both replan on each market opening
    from the accepted dispatch so far and hold already-cleared products fixed.
    The terminal SOC defaults to the vehicle's initial SOC so the optimiser does
    not liquidate batteries at an artificial horizon boundary.

    Bids carry the EV ids and nodes so clearing and settlement stay per unit.
    This is an EV-only portfolio.  Optional connection limits apply to net
    portfolio import/export including an exogenous ``background_load`` (positive
    for consumption); the grid fee and the capacity charge apply to the same net
    import, so all three see the connection point consistently.

    Args:
        ev_horizon_mode: ``"rolling_horizon"`` or ``"perfect_foresight"``.
        ev_look_ahead_horizon: Planning window in ``rolling_horizon`` mode.
        ev_terminal_soc: Target SOC at the horizon end; defaults to each
            vehicle's ``initial_soc``.
        ev_import_limit_mw: Connection import limit, MW.  ``None`` disables it.
        ev_export_limit_mw: Connection export limit, MW.  ``None`` disables it.
        ev_background_load_mw: Constant exogenous load behind the same
            connection, MW, positive for consumption.  It is not an EV and is
            never dispatched, but it occupies the connection, so the limits, the
            grid fee and the capacity charge all see it.  Quote the connection's
            own rating in the two limits and put the other load here rather than
            netting it off by hand.
        ev_peak_price: Capacity charge on the peak net import, EUR/MW.
        ev_observed_peak_mw: Peak already billed in the current billing period.
            In :meth:`calculate_bids` the peak realised so far in the simulation
            is taken into account on top of this floor.
        ev_solver: Pyomo solver name.
        ev_solver_time_limit: Per-solve wall-clock limit in seconds, or ``None``.
        ev_mip_gap: Relative MIP gap, or ``None`` for the solver default.
        ev_force_milp: Skip the LP relaxation and always solve the MILP.
        ev_slack_penalty: Penalty on unserved driving energy and terminal-SOC
            misses, EUR/MWh.
    """

    def __init__(
        self,
        *args,
        ev_horizon_mode="rolling_horizon",
        ev_look_ahead_horizon="48h",
        ev_terminal_soc=None,
        ev_import_limit_mw=None,
        ev_export_limit_mw=None,
        ev_background_load_mw=0.0,
        ev_peak_price=0.0,
        ev_observed_peak_mw=0.0,
        ev_solver="appsi_highs",
        ev_solver_time_limit=None,
        ev_mip_gap=None,
        ev_force_milp=False,
        ev_slack_penalty=DEFAULT_SLACK_PENALTY,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Portfolio strategies are constructed with the shared
        # ``bidding_strategy_params`` of the scenario, which also carry keys for
        # other strategies.  Silently swallowing an "ev_" key would turn a typo
        # into a wrong but plausible-looking run, so reject those explicitly
        # while leaving everyone else's parameters alone.
        unknown = sorted(k for k in kwargs if k.startswith("ev_"))
        if unknown:
            raise ValueError(f"Unknown EV portfolio parameters: {', '.join(unknown)}")
        if ev_horizon_mode not in ("perfect_foresight", "rolling_horizon"):
            raise ValueError("Unknown EV horizon mode")
        self.horizon_mode = ev_horizon_mode
        self.look_ahead = pd.Timedelta(ev_look_ahead_horizon).to_pytimedelta()
        if self.look_ahead.total_seconds() <= 0:
            raise ValueError("EV look-ahead must be positive")
        self.terminal_soc = ev_terminal_soc
        self.import_limit = ev_import_limit_mw
        self.export_limit = ev_export_limit_mw
        self.background_load = float(ev_background_load_mw)
        self.peak_price = float(ev_peak_price)
        self.observed_peak = float(ev_observed_peak_mw)
        for value in (
            self.import_limit,
            self.export_limit,
            self.background_load,
            self.peak_price,
            self.observed_peak,
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise ValueError("EV limits and peak prices must be finite/nonnegative")
        self.solver = ev_solver
        self.solver_time_limit = (
            None if ev_solver_time_limit is None else float(ev_solver_time_limit)
        )
        if self.solver_time_limit is not None and self.solver_time_limit <= 0:
            raise ValueError("EV solver time limit must be positive")
        self.mip_gap = None if ev_mip_gap is None else float(ev_mip_gap)
        if self.mip_gap is not None and self.mip_gap < 0:
            raise ValueError("EV MIP gap must be nonnegative")
        self.force_milp = bool(ev_force_milp)
        self.slack_penalty = float(ev_slack_penalty)
        if not math.isfinite(self.slack_penalty) or self.slack_penalty < 0:
            raise ValueError("EV slack penalty must be finite/nonnegative")

    # ------------------------------------------------------------------
    # optimisation
    # ------------------------------------------------------------------

    def _solve(self, model, relaxed):
        """Solve *model*, optionally with the integrality of ``charging`` relaxed.

        Returns the termination condition so the caller can report it.  Raises
        only when no usable solution came back at all; a solve that stopped on a
        time limit with an incumbent in hand is used with a warning, because for
        a rolling simulation a slightly suboptimal schedule beats no schedule.
        """
        if relaxed:
            for index in model.charging:
                model.charging[index].domain = pyo.UnitInterval

        solver = pyo.SolverFactory(self.solver)
        config = getattr(solver, "config", None)
        if config is not None:
            if self.solver_time_limit is not None and "time_limit" in config:
                config.time_limit = self.solver_time_limit
            if self.mip_gap is not None and "mip_gap" in config:
                config.mip_gap = self.mip_gap
        result = solver.solve(model, load_solutions=False)
        condition = result.solver.termination_condition

        usable = condition in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
            pyo.TerminationCondition.globallyOptimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.maxIterations,
        )
        if usable:
            try:
                model.solutions.load_from(result)
            except (ValueError, AttributeError):
                usable = False
        if not usable:
            if relaxed:
                # The relaxation is a superset of the MILP: if it has no
                # solution neither does the MILP, so do not retry.
                raise RuntimeError(f"EV portfolio relaxation failed: {condition}")
            raise RuntimeError(f"EV portfolio infeasible or unsolved: {condition}")
        if condition != pyo.TerminationCondition.optimal:
            logger.warning(
                "EV portfolio solve stopped early (%s); using the incumbent schedule",
                condition,
            )
        if relaxed:
            for index in model.charging:
                model.charging[index].domain = pyo.Binary
        return condition

    def _simultaneous(self, model):
        """Vehicle/step pairs where a relaxed solution charges and discharges at once."""
        return [
            (i, k)
            for i in model.U
            for k in model.T
            if pyo.value(model.charge[i, k]) > SIMULTANEITY_TOLERANCE
            and pyo.value(model.discharge[i, k]) > SIMULTANEITY_TOLERANCE
        ]

    def optimize(
        self,
        units,
        start,
        end,
        market_id="EOM",
        *,
        grid_fees=None,
        background_load=None,
        observed_peak=None,
    ):
        """Plan the portfolio over ``[start, end)``.

        Args:
            units: The vehicles to plan for; all must be ``ElectricVehicleUnit``.
            start: First delivery step of the horizon.
            end: Exclusive end of the horizon.
            market_id: Market whose price forecast drives the energy cost.
            grid_fees: Per-timestep volumetric grid fee in EUR/MWh, levied on
                net import only.  When given, the energy price is read without
                any fee that :func:`price_plus_grid_fee` folded in, so the fee
                is charged exactly once.
            background_load: Per-timestep exogenous load at the same connection
                in MW, positive for consumption.  Counts towards the connection
                limits, the grid fee and the peak.
            observed_peak: Peak net import already billed in this billing
                period, MW.  Defaults to ``ev_observed_peak_mw``.

        Returns:
            ``{unit_id: {timestamp: signed MW}}`` - positive discharging,
            negative charging.  Dispatch is not modified.
        """
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
        floor = self.observed_peak if observed_peak is None else float(observed_peak)
        if not math.isfinite(floor) or floor < 0:
            raise ValueError("EV observed peak must be finite/nonnegative")

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
        # Slacks keep the model solvable when the fleet is simply short of
        # energy; the penalty holds them at zero whenever a schedule exists.
        model.unmet = pyo.Var(model.U, model.T, domain=pyo.NonNegativeReals)
        model.shortfall = pyo.Var(model.U, domain=pyo.NonNegativeReals)
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
            # The fee is charged on net import below, so read the price without
            # the fee that price_plus_grid_fee may have folded into it.
            prices = (
                grid_fee_free_price(unit.forecaster, market_id)
                if grid_fees is not None
                else unit.forecaster.price[market_id]
            )
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
                    + model.unmet[i, k]
                )
                # Driving can never take more than the trip needed.
                model.constraints.add(
                    model.unmet[i, k] <= unit.trip_energy_consumption.at[t]
                )
                model.constraints.add(
                    pyo.inequality(
                        unit.min_soc * unit.capacity, e, unit.max_soc * unit.capacity
                    )
                )
                if unit.outputs["energy_committed"].at[t]:
                    model.constraints.add(d - c == unit.outputs["energy"].at[t])
                price = prices.at[t]
                cost += hours * (
                    price * (c - d)
                    + unit.additional_cost_charge * c
                    + unit.additional_cost_discharge * d
                )
                cost += self.slack_penalty * model.unmet[i, k]
            model.constraints.add(
                model.energy[i, len(times) - 1] + model.shortfall[i]
                >= terminal * unit.capacity
            )
            cost += self.slack_penalty * model.shortfall[i]
        model.imports = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.peak = pyo.Var(domain=pyo.NonNegativeReals, bounds=(floor, None))
        for k in model.T:
            withdrawal = (
                sum(model.charge[i, k] - model.discharge[i, k] for i in model.U)
                + background[k]
            )
            # imports is the positive part of the net withdrawal at the
            # connection point: >= 0 and >= withdrawal, pushed down to the
            # larger of the two by the nonnegative fee and peak price.  The
            # background load belongs in it because the DSO meters and bills the
            # whole connection, not the vehicles alone.
            model.constraints.add(model.imports[k] >= withdrawal)
            model.constraints.add(model.peak >= model.imports[k])
            if self.import_limit is not None:
                model.constraints.add(withdrawal <= float(self.import_limit))
            if self.export_limit is not None:
                model.constraints.add(withdrawal >= -float(self.export_limit))
            cost += hours * fees[k] * model.imports[k]
        cost += self.peak_price * (model.peak - floor)
        model.objective = pyo.Objective(expr=cost)

        # Solve the relaxation first; it is exact unless the effective price of
        # energy goes negative somewhere, and checking the solution is a cheaper
        # and safer test than trying to predict that up front.
        if self.force_milp:
            self._solve(model, relaxed=False)
        else:
            self._solve(model, relaxed=True)
            if self._simultaneous(model):
                logger.debug(
                    "Relaxed EV schedule charges and discharges at once; solving the MILP"
                )
                self._solve(model, relaxed=False)

        for i, unit in enumerate(units):
            unserved = sum(pyo.value(model.unmet[i, k]) for k in model.T)
            missed = pyo.value(model.shortfall[i])
            if unserved > SIMULTANEITY_TOLERANCE or missed > SIMULTANEITY_TOLERANCE:
                logger.warning(
                    "EV %s cannot be kept charged over %s..%s: %.6g MWh of driving "
                    "energy unserved, %.6g MWh short of the terminal SOC",
                    unit.id,
                    start,
                    end,
                    unserved,
                    missed,
                )
            for k, t in enumerate(times):
                unit.outputs["ev_planned_unmet_energy"].at[t] = pyo.value(
                    model.unmet[i, k]
                )
        return {
            u.id: {
                t: pyo.value(model.discharge[i, k] - model.charge[i, k])
                for k, t in enumerate(times)
            }
            for i, u in enumerate(units)
        }

    # ------------------------------------------------------------------
    # horizon
    # ------------------------------------------------------------------

    def _safe_horizon_end(self, unit, index, start, nominal_end):
        """Earliest horizon end at or after *nominal_end* that is fair to *unit*.

        The terminal-SOC target is meaningless in the middle of a trip, and
        punitive right after one: the vehicle comes home empty and would be
        asked to be full again before it has had time to charge.  Walk forward
        from the vehicle's last absence inside the window until it could have
        recharged from ``min_soc`` to the target at full power, counting any
        further trips on the way, and end the horizon there.

        Returns ``None`` when the vehicle cannot get there within the index, in
        which case the caller should plan to the end of the simulation.
        """
        # index slicing by datetime is inclusive on both ends, so step back one
        # frequency to get the half-open window [start, nominal_end).
        away = [
            t
            for t in index[start : nominal_end - index.freq]
            if not unit.forecaster.availability.at[t]
        ]
        if not away or unit.max_power_charge == 0:
            return nominal_end
        hours = index.freq.total_seconds() / 3600
        target = (
            unit.initial_soc if self.terminal_soc is None else float(self.terminal_soc)
        )
        recharge = max(0, target - unit.min_soc) * unit.capacity
        if away[-1] + index.freq > index[-1]:
            return None
        for t in index[away[-1] + index.freq :]:
            if unit.forecaster.availability.at[t]:
                recharge -= -unit.max_power_charge * unit.efficiency_charge * hours
            else:
                recharge += unit.trip_energy_consumption.at[t]
            if recharge <= 0:
                return max(nominal_end, t + index.freq)
        return None

    def _horizon_end(self, units, index, start, nominal_end):
        """Horizon end that is fair to *every* vehicle in the portfolio.

        The terminal SOC is imposed on all vehicles at the same, shared end, so
        it is not enough for each vehicle to have some end that suits it: the
        end the portfolio actually uses must suit them all.  Extending to the
        per-vehicle maximum - which is what a plain ``max`` over the individual
        answers gives - can easily land inside another vehicle's trip and make
        the whole portfolio unsolvable.  Iterate instead: extend to the largest
        requested end, re-ask every vehicle about *that* end, and repeat until
        the answer stops moving.
        """
        end = nominal_end
        for _ in range(len(units) + 1):
            candidates = [self._safe_horizon_end(u, index, start, end) for u in units]
            if any(c is None for c in candidates):
                return index[-1] + index.freq
            settled = max(candidates)
            if settled == end:
                return end
            end = settled
        # Vehicles kept pushing each other outwards; plan to the end of the data.
        return index[-1] + index.freq

    # ------------------------------------------------------------------
    # bidding
    # ------------------------------------------------------------------

    def _connection_fees(self, units, times):
        """Announced grid fee per timestep, or ``None`` when there is no tariff.

        The fee is a property of the connection and is cleared ``pay_as_clear``,
        so every vehicle behind it sees the same price; read it from the first
        vehicle that participates in the tariff market.
        """
        for unit in units:
            if GRID_FEE_ANNOUNCED_FLAG not in unit.outputs:
                continue
            fee = effective_grid_fee(
                unit.outputs,
                default_fee=getattr(unit.forecaster, "default_grid_fee", 0.0),
            )
            positions = {t: p for p, t in enumerate(unit.index)}
            return {t: float(fee[positions[t]]) for t in times}
        return None

    def _realised_peak(self, units, start):
        """Largest net connection import already delivered, MW.

        The capacity charge is billed on the highest net import over a billing
        period, so a rolling plan that only ever looks at its own window would
        re-buy the same peak every round.  Feed the peak already realised back
        in as the floor.  The billing period is the simulation run; see the
        TODOs for shorter periods.
        """
        index = units[0].index
        if start <= index[0]:
            return 0.0
        withdrawal = None
        for unit in units:
            # outputs["energy"] is positive discharging, so negate to get import.
            served = -unit.outputs["energy"].loc[index[0] : start - index.freq]
            withdrawal = served if withdrawal is None else withdrawal + served
        if withdrawal is None or not len(withdrawal):
            return 0.0
        # The charge is billed on the connection, so the floor has to be the
        # connection's peak, not the fleet's.
        return float(max(0.0, withdrawal.max() + self.background_load))

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
            end = self._horizon_end(
                units, index, start, min(end, start + self.look_ahead)
            )
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
        times = list(index[start : end - index.freq])
        plan = self.optimize(
            units,
            start,
            end,
            grid_fees=self._connection_fees(units, times),
            background_load=dict.fromkeys(times, self.background_load),
            observed_peak=max(self.observed_peak, self._realised_peak(units, start)),
        )
        bids = []
        for unit in units:
            for t, power in plan[unit.id].items():
                unit.outputs["ev_planned_energy"].at[t] = power
            for k, (left, right, only_hours) in enumerate(product_tuples):
                power = plan[unit.id][left]
                if market_config.product_type == "grid_fee":
                    # The tariff market prices announced withdrawal, so always
                    # announce a consumption volume; a vehicle that plans to
                    # export announces an epsilon rather than dropping out.
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
