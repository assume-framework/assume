# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import logging
from datetime import datetime, timedelta

import numpy as np
import pyomo.environ as pyo

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration
from assume.strategies.dmas_storage import get_solver_factory
from assume.strategies.market_discovery import MarketInfo, discover_markets

logger = logging.getLogger(__name__)

# product types whose sell/buy side is an actual energy flow (affects the
# SoC) - all other product types only reserve power capacity
ENERGY_TYPES = {"energy", "heat", "energy_pos", "energy_neg"}
SELL_TYPES = ENERGY_TYPES | {"capacity_pos"}
BUY_TYPES = ENERGY_TYPES | {"capacity_neg"}


class MultiMarketStorageStrategy(MinMaxChargeStrategy):
    """
    A generic multi-market bidding strategy for storage units.

    At the beginning of the simulation, the strategy discovers all markets of
    the world where the unit is eligible to participate (see
    :func:`assume.strategies.market_discovery.discover_markets`). For each
    market, the opening windows, gate closures and product delivery windows
    are extracted from the market configuration.

    Whenever one of the markets opens, the strategy optimizes the storage
    operation over a user-defined foresight window across *all* eligible
    markets at once:

    - decision variables are the bid volumes (sell and buy) for every product
      window of every market that is not cleared yet, plus the state of
      charge,
    - the objective maximizes the expected profit over all markets using the
      price forecasts of the unit's forecaster,
    - constraints are the SoC dynamics (with charge/discharge efficiencies),
      the SoC limits, the charge/discharge power limits, and the coupling
      between the markets: the sum of all commitments delivering in a period
      must not exceed the power limits of the unit, and balancing energy
      bids are bounded by the balancing capacity commitments of the same
      direction.

    Commitments from already cleared markets are read from the unit outputs
    and enter the optimization as fixed parameters. The optimization follows
    a rolling horizon: the plan is optimized over the foresight window, but
    only used for a shorter overlap period (by default until the next market
    opening). Afterwards, the storage is re-optimized over a shifted foresight
    window. As the optimization sees beyond the end of the used plan, the
    stored energy keeps its value for the planned future dispatch instead of
    being dumped at the end of the used window. Multiple markets opening at
    the same time reuse one optimization and each market only receives its
    own slice of the shared plan.

    The bid prices are derived from the marginal storage value: the dual
    (shadow price) of the state-of-charge balance of the solved optimization.
    As the optimization already contains the price forecasts, the dual
    reflects the expected future dispatch opportunities of the stored energy
    (in EUR/MWh stored). Following the symmetric convention, the buy and sell
    bid prices are anchored symmetrically around this value with a spread of
    the round-trip efficiency: buy price = value * sqrt(efficiency), sell
    price = value / sqrt(efficiency). The additional costs of the unit (for
    charging respectively discharging) and the degradation cost enter the
    optimization objective and widen the bid prices accordingly. A deadband
    (by default the price tick of the respective market) widens the bid-ask
    band further, so that small price movements do not trigger trades.

    Attributes:
        foresight (datetime.timedelta): The optimization horizon: how far
            ahead the storage operation is optimized.
        overlap (datetime.timedelta): How long the optimization plan is used
            before it is re-optimized (rolling horizon). Defaults to zero,
            i.e. re-optimization at every market opening.
        terminal_condition (bool): Whether the optimization must end the
            horizon with at least the energy currently stored. Defaults to
            False.
        deadband (float): The symmetric widening of the bid-ask band. If
            None, the price tick of the respective market is used (0 if the
            market has no price tick).
        degradation_cost (float): Degradation cost per MWh discharged
            (grid-side). Defaults to 0.

    Args:
        foresight (str, optional): The foresight window (e.g. "24h").
            Defaults to "24h".
        overlap (str, optional): How long the optimization plan is used
            before re-optimizing (e.g. "12h": optimize over 24h foresight,
            use the plan for 12h, then re-optimize over the shifted window).
            Defaults to re-optimizing at every market opening.
        terminal_condition (bool, optional): If True, the optimization must
            not end the horizon with less energy than the storage currently
            holds (prevents net energy withdrawal over the horizon).
            Defaults to False.
        deadband (float, optional): Symmetric widening of the bid-ask band.
            Defaults to the price tick of the respective market.
        degradation_cost (float, optional): Degradation cost per MWh
            discharged (grid-side). Defaults to 0.
        solver_name (str, optional): The name of the solver to use. Defaults
            to the first available of appsi_highs, gurobi, glpk, cbc, cplex.
        capacity_price_fallback (str | float, optional): Price for capacity
            markets without a dedicated price forecast. Either the market id
            of another market (uses its forecast) or a flat price. Defaults
            to 0.
        **kwargs: Additional keyword arguments, including the bidding
            parameters of the world.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = parse_duration(kwargs.get("foresight", "24h"))
        # rolling horizon: how long the plan is used before re-optimizing.
        # Zero means re-optimization at every market opening.
        self.overlap = parse_duration(kwargs.get("overlap", "0h"))
        self.terminal_condition = bool(kwargs.get("terminal_condition", False))
        # symmetric widening of the bid-ask band; None means the price tick
        # of the respective market is used (0 if the market has none)
        self.deadband = kwargs.get("deadband", None)
        # degradation cost per MWh discharged (grid-side)
        self.degradation_cost = float(kwargs.get("degradation_cost", 0.0))
        self.solver_name = kwargs.get("solver_name", None)
        # fallback price for capacity markets without a price forecast:
        # a market id (str) to use the forecast of that market, or a flat
        # price (number). Defaults to zero.
        self.capacity_price_fallback = kwargs.get("capacity_price_fallback", 0.0)

        # discovered markets - filled on first calculate_bids call
        self.markets: dict[str, MarketInfo] | None = None
        # solver instance - created lazily to fail late if no solver exists
        self.opt = None
        # cached optimization result: solution mapping
        # (market_id, start, end) -> bid quantities
        self._cache_solution: dict[tuple, dict[str, float]] = {}
        # rolling horizon: decision time of the last optimization, the time
        # until which its plan is used, and the planned SoC trajectory
        self._plan_time: datetime | None = None
        self._plan_valid_until: datetime | None = None
        self._plan_soc: dict[datetime, float] = {}
        # marginal storage value per time step (duals of the SoC balance)
        self._marginal_value: dict[datetime, float] = {}

    def _discover_markets(self, unit: SupportsMinMaxCharge) -> None:
        """
        Discovers all markets the unit can participate in.
        Runs once before the first optimization.
        """
        self.markets = discover_markets(
            unit,
            unit.available_markets,
            unit.index[0],
            unit.index[-1],
        )
        logger.info(
            "unit %s discovered eligible markets: %s",
            unit.id,
            list(self.markets.keys()),
        )

    def _get_price_forecast(
        self,
        unit: SupportsMinMaxCharge,
        market_info: MarketInfo,
        start: datetime,
        end: datetime,
    ) -> np.ndarray:
        """
        Returns the price forecast of a market for the given time range.

        Energy-type markets fall back to the EOM price forecast if no
        dedicated forecast exists. Capacity markets without a dedicated
        forecast use the `capacity_price_fallback`: either the price forecast
        of another market (given as market id) or a flat price (given as
        number). Defaults to zero.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            market_info (MarketInfo): The market to get the forecast for.
            start (datetime.datetime): The start of the time range.
            end (datetime.datetime): The end of the time range (exclusive).

        Returns:
            numpy.ndarray: The price forecast values.
        """
        market_id = market_info.market_id

        price_series = unit.forecaster.price.get(market_id)
        if price_series is None and market_info.product_type in ENERGY_TYPES:
            # energy-type markets fall back to the EOM price forecast
            price_series = unit.forecaster.price.get("EOM")
        if price_series is None:
            # capacity markets use the configured fallback
            fallback = self.capacity_price_fallback
            if isinstance(fallback, str):
                price_series = unit.forecaster.price.get(fallback)
            elif fallback:
                return np.full(1, float(fallback))
        if price_series is None:
            return np.zeros(1)

        # the end of the range is exclusive, the last index step is included
        values = price_series.loc[start : end - unit.index.freq]
        return np.asarray(values, dtype=float)

    def _get_fixed_commitments(
        self,
        unit: SupportsMinMaxCharge,
        times: list[datetime],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads the already cleared commitments of the unit for the given times.

        Returns the fixed power reservations (limiting the power limits) and
        the fixed energy flows (affecting the SoC) for each time step.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            times (list[datetime.datetime]): The time steps of the horizon.

        Returns:
            tuple[numpy.ndarray]: fixed discharge power, fixed charge power,
                fixed discharge energy and fixed charge energy per time step.
        """
        energy = np.asarray(unit.outputs["energy"].loc[times[0] : times[-1]])
        capacity_pos = np.asarray(
            unit.outputs["capacity_pos"].loc[times[0] : times[-1]]
        )
        capacity_neg = np.asarray(
            unit.outputs["capacity_neg"].loc[times[0] : times[-1]]
        )
        energy_pos = np.asarray(unit.outputs["energy_pos"].loc[times[0] : times[-1]])
        energy_neg = np.asarray(unit.outputs["energy_neg"].loc[times[0] : times[-1]])

        # energy is a signed net flow: positive is discharge, negative is charge
        discharge_energy = np.maximum(energy, 0)
        charge_energy = np.maximum(-energy, 0)

        # activations flow energy like a normal dispatch and reserve power
        fixed_discharge_energy = discharge_energy + energy_pos
        fixed_charge_energy = charge_energy + energy_neg
        # capacity reserves only reserve power, they do not flow energy
        fixed_discharge_power = fixed_discharge_energy + capacity_pos
        fixed_charge_power = fixed_charge_energy + capacity_neg

        return (
            fixed_discharge_power,
            fixed_charge_power,
            fixed_discharge_energy,
            fixed_charge_energy,
        )

    def _needs_reoptimization(
        self,
        unit: SupportsMinMaxCharge,
        decision_time: datetime,
    ) -> bool:
        """
        Returns True if the cached plan is outdated and must be re-optimized.

        This is the case when the actual SoC deviates from the planned
        trajectory - e.g. because bids were rejected or only partially
        accepted, or because activations changed the dispatch. The planned
        trajectory already accounts for the plan's own bids, so clearing
        them as planned does not trigger a re-optimization. Deviations of
        capacity commitments (which do not flow energy) only make the plan
        conservative, never infeasible, and are handled at the next
        re-optimization.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            decision_time (datetime.datetime): The current decision time.

        Returns:
            bool: True if the plan must be re-optimized.
        """
        if not self._plan_soc:
            return True

        # if the actual SoC deviates from the planned one, the plan's future
        # volumes may no longer be feasible
        planned_soc = self._plan_soc.get(decision_time)
        if planned_soc is None:
            return True
        actual_soc = unit.outputs["soc"].at[decision_time] * unit.capacity
        return not np.isclose(actual_soc, planned_soc, atol=1e-6)

    def _optimize(
        self,
        unit: SupportsMinMaxCharge,
        decision_time: datetime,
    ) -> dict[tuple[str, datetime, datetime], dict[str, float]]:
        """
        Optimizes the storage operation over the foresight window across all
        eligible markets.

        Builds a linear optimization model with bid volume variables for all
        product windows that are not cleared yet, plus the state of charge.
        The solution is returned as a mapping from
        (market_id, window_start, window_end) to the optimal sell/buy volumes.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            decision_time (datetime.datetime): The current decision time
                (the opening time of the market that triggered the
                optimization).

        Returns:
            dict[tuple[str, datetime, datetime], dict[str, float]]: The
                optimization result per market and product window.
        """
        if self.opt is None:
            self.opt = get_solver_factory()

        freq = unit.index.freq
        dt_h = freq / timedelta(hours=1)

        # horizon: foresight window, truncated at the end of the simulation
        horizon_end = min(decision_time + self.foresight, unit.index[-1] + freq)
        n = int((horizon_end - decision_time) / freq)
        if n <= 0:
            return {}
        times = [decision_time + freq * i for i in range(n)]

        # collect all product windows within the horizon that are not cleared
        # yet, ordered by delivery start (chronological commitment order)
        windows: list[tuple[MarketInfo, datetime, datetime]] = []
        for market_info in self.markets.values():
            for window_start, window_end in market_info.deliveries:
                if window_start >= horizon_end or window_end <= decision_time:
                    continue
                if market_info.gate_closures[(window_start, window_end)] <= decision_time:
                    # already cleared - enters as fixed parameter
                    continue
                windows.append((market_info, window_start, window_end))
        windows.sort(key=lambda w: (w[1], w[2], w[0].market_id))

        fixed_dis, fixed_ch, fixed_dis_e, fixed_ch_e = self._get_fixed_commitments(
            unit, times
        )

        # index helpers: which windows are sell/buy/energy-type
        sell_indices = [
            i for i, (info, _, _) in enumerate(windows) if info.product_type in SELL_TYPES
        ]
        buy_indices = [
            i for i, (info, _, _) in enumerate(windows) if info.product_type in BUY_TYPES
        ]
        energy_sell = [
            i
            for i, (info, _, _) in enumerate(windows)
            if info.product_type in ENERGY_TYPES
        ]

        model = pyo.ConcreteModel("multi_market_storage")
        model.S = pyo.Set(initialize=sell_indices)
        model.B = pyo.Set(initialize=buy_indices)
        model.T = pyo.Set(initialize=range(n))
        model.sell = pyo.Var(model.S, domain=pyo.NonNegativeReals)
        model.buy = pyo.Var(model.B, domain=pyo.NonNegativeReals)
        model.soc = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
            bounds=(unit.min_soc * unit.capacity, unit.max_soc * unit.capacity),
        )
        # to extract the duals of the SoC constraints (marginal storage
        # value) from the solved optimization
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        model.constraints = self._build_constraints(
            model,
            unit,
            windows,
            times,
            dt_h,
            decision_time,
            fixed_dis,
            fixed_ch,
            fixed_dis_e,
            fixed_ch_e,
            sell_indices,
            buy_indices,
            energy_sell,
            with_terminal=self.terminal_condition,
        )

        # objective: maximize the expected profit over all markets, minus
        # the additional costs of the unit and the degradation cost
        obj_terms = []
        for i, (info, w_start, w_end) in enumerate(windows):
            covered_end = min(w_end, horizon_end)
            hours = (covered_end - w_start) / timedelta(hours=1)
            prices = self._get_price_forecast(unit, info, w_start, covered_end)
            price = float(np.mean(prices)) if len(prices) else 0.0
            if info.product_type in ("capacity_neg", "energy_neg"):
                # negative reserve/activation is paid to the unit
                obj_terms.append(price * hours * model.buy[i])
            elif info.product_type in ("capacity_pos",):
                # positive reserve is paid to the unit
                obj_terms.append(price * hours * model.sell[i])
            else:
                # energy sells earn the price minus the additional costs of
                # discharging and the degradation, energy buys pay the price
                # plus the additional costs of charging
                sell_price = price - unit.additional_cost_discharge - self.degradation_cost
                buy_price = price + unit.additional_cost_charge
                obj_terms.append(hours * (sell_price * model.sell[i] - buy_price * model.buy[i]))
        model.obj = pyo.Objective(expr=pyo.quicksum(obj_terms), sense=pyo.maximize)

        solution = self._solve(model, windows, sell_indices, buy_indices)
        if solution is None and self.terminal_condition:
            # retry without the terminal condition (e.g. at the end of the
            # simulation when the horizon is truncated)
            model.constraints.deactivate()
            model.constraints = self._build_constraints(
                model,
                unit,
                windows,
                times,
                dt_h,
                decision_time,
                fixed_dis,
                fixed_ch,
                fixed_dis_e,
                fixed_ch_e,
                sell_indices,
                buy_indices,
                energy_sell,
                with_terminal=False,
            )
            solution = self._solve(model, windows, sell_indices, buy_indices)

        if solution is None:
            logger.warning(
                "multi-market optimization for unit %s at %s is infeasible - no bids",
                unit.id,
                decision_time,
            )
            self._plan_soc = {}
            self._marginal_value = {}
            return {}

        # store the planned SoC trajectory (start-of-step values, matching
        # the convention of the unit's soc output) and the marginal storage
        # value (the duals of the SoC balance constraints) to derive the bid
        # prices; both detect deviations of the actual SoC from the plan
        # (e.g. after rejected or clamped bids)
        planned_soc = unit.outputs["soc"].at[decision_time] * unit.capacity
        self._plan_soc = {}
        self._marginal_value = {}
        for i in range(n):
            self._plan_soc[times[i]] = planned_soc
            planned_soc = float(pyo.value(model.soc[i]))
            self._marginal_value[times[i]] = float(model.dual[model.soc_constraints[i]])
        return solution

    def _build_constraints(
        self,
        model: pyo.ConcreteModel,
        unit: SupportsMinMaxCharge,
        windows: list[tuple[MarketInfo, datetime, datetime]],
        times: list[datetime],
        dt_h: float,
        decision_time: datetime,
        fixed_dis: np.ndarray,
        fixed_ch: np.ndarray,
        fixed_dis_e: np.ndarray,
        fixed_ch_e: np.ndarray,
        sell_indices: list[int],
        buy_indices: list[int],
        energy_indices: list[int],
        with_terminal: bool,
    ) -> pyo.ConstraintList:
        """
        Builds the constraints of the optimization model.

        Args:
            model (pyo.ConcreteModel): The optimization model.
            unit (SupportsMinMaxCharge): The storage unit.
            windows (list): The (market_info, start, end) tuples of all
                product windows in the horizon.
            times (list[datetime.datetime]): The time steps of the horizon.
            dt_h (float): The length of a time step in hours.
            decision_time (datetime.datetime): The decision time.
            fixed_dis (numpy.ndarray): Fixed discharge power per time step.
            fixed_ch (numpy.ndarray): Fixed charge power per time step.
            fixed_dis_e (numpy.ndarray): Fixed discharge energy per time step.
            fixed_ch_e (numpy.ndarray): Fixed charge energy per time step.
            sell_indices (list[int]): Indices of the sell variables.
            buy_indices (list[int]): Indices of the buy variables.
            energy_indices (list[int]): Indices of the energy-flow variables.
            with_terminal (bool): Whether to add the terminal SoC condition.

        Returns:
            pyo.ConstraintList: The constraints.
        """
        n = len(times)
        constraints = pyo.ConstraintList()
        constraints.construct()
        # the SoC balance constraints, to extract their duals (marginal
        # storage value) from the solved optimization
        soc_constraints = []

        # windows covering each time step
        covering = [[] for _ in range(n)]
        for i, (_, w_start, w_end) in enumerate(windows):
            for t in range(n):
                if w_start <= times[t] < w_end:
                    covering[t].append(i)

        def _covered(i: int, t: int) -> bool:
            return i in covering[t]

        # SoC dynamics: energy flows of energy-type markets and the fixed
        # energy flows of cleared markets change the state of charge.
        # The constraints are collected to extract their duals (the marginal
        # storage value) from the solved optimization.
        model.soc_constraints = []
        soc_prev = unit.outputs["soc"].at[decision_time] * unit.capacity
        for t in range(n):
            charge_flow = 0.0
            discharge_flow = 0.0
            for i in energy_indices:
                if _covered(i, t):
                    charge_flow += model.buy[i]
                    discharge_flow += model.sell[i]
            soc_delta = (
                (charge_flow + fixed_ch_e[t]) * unit.efficiency_charge
                - (discharge_flow + fixed_dis_e[t]) / unit.efficiency_discharge
            ) * dt_h
            soc_constraints.append(
                constraints.add(model.soc[t] == soc_prev + soc_delta)
            )
            soc_prev = model.soc[t]

        # power limits: the sum of all commitments delivering in a period plus
        # the fixed commitments must not exceed the power limits of the unit
        for t in range(n):
            total_sell = 0.0
            total_buy = 0.0
            has_vars = False
            for i in sell_indices:
                if _covered(i, t):
                    total_sell += model.sell[i]
                    has_vars = True
            for i in buy_indices:
                if _covered(i, t):
                    total_buy += model.buy[i]
                    has_vars = True
            if not has_vars:
                # constant expressions cannot become constraints - the fixed
                # commitments are already cleared and feasible by construction
                continue
            constraints.add(total_sell + fixed_dis[t] <= unit.max_power_discharge)
            constraints.add(total_buy + fixed_ch[t] <= -unit.max_power_charge)

        # balancing energy bids are bounded by the balancing capacity
        # commitments of the same direction (reserved capacity)
        capacity_pos_windows = [
            i
            for i, (info, _, _) in enumerate(windows)
            if info.product_type == "capacity_pos"
        ]
        capacity_neg_windows = [
            i
            for i, (info, _, _) in enumerate(windows)
            if info.product_type == "capacity_neg"
        ]
        for i, (info, w_start, w_end) in enumerate(windows):
            if info.product_type not in ("energy_pos", "energy_neg"):
                continue
            for t in range(n):
                if not _covered(i, t):
                    continue
                if info.product_type == "energy_pos":
                    # reserved power: capacity commitments of this direction
                    # plus the fixed capacity cleared on other markets
                    reserved = fixed_dis[t] - fixed_dis_e[t]
                    reserved += sum(
                        model.sell[j] for j in capacity_pos_windows if _covered(j, t)
                    )
                    constraints.add(model.sell[i] <= reserved)
                else:
                    reserved = fixed_ch[t] - fixed_ch_e[t]
                    reserved += sum(
                        model.buy[j] for j in capacity_neg_windows if _covered(j, t)
                    )
                    constraints.add(model.buy[i] <= reserved)

        # terminal condition (optional): do not end the horizon with less
        # energy than now (prevents myopic dumping of the stored energy)
        if with_terminal and n > 0:
            constraints.add(
                model.soc[n - 1]
                >= unit.outputs["soc"].at[decision_time] * unit.capacity
            )

        model.soc_constraints = soc_constraints
        return constraints

    def _solve(
        self,
        model: pyo.ConcreteModel,
        windows: list[tuple[MarketInfo, datetime, datetime]],
        sell_indices: list[int],
        buy_indices: list[int],
    ) -> dict[tuple[str, datetime, datetime], dict[str, float]] | None:
        """
        Solves the optimization model and extracts the solution.

        Returns None if the model is infeasible.
        """
        results = self.opt.solve(model, load_solutions=False)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            return None
        model.solutions.load_from(results)

        solution = {}
        for i, (info, w_start, w_end) in enumerate(windows):
            entry: dict[str, float] = {}
            if i in sell_indices:
                entry["sell"] = float(pyo.value(model.sell[i]))
            if i in buy_indices:
                entry["buy"] = float(pyo.value(model.buy[i]))
            solution[(info.market_id, w_start, w_end)] = entry
        return solution

    def _bid_price(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        start: datetime,
        end: datetime,
        discharge: bool,
    ) -> float:
        """
        Calculates the bid price from the marginal storage value.

        The marginal storage value is the mean of the SoC balance duals over
        the delivery window (EUR/MWh stored). Following the symmetric
        convention, the buy and sell prices are anchored symmetrically
        around this value with a spread of the round-trip efficiency:

        - sell (discharge) price: value / sqrt(efficiency) plus the
          additional costs of discharging and the degradation cost
        - buy (charge) price: value * sqrt(efficiency) minus the additional
          costs of charging

        A deadband (by default the price tick of the market) widens the
        band symmetrically, so that small price movements do not trigger
        trades.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            market_config (MarketConfig): The market to bid on.
            start (datetime.datetime): The start of the delivery window.
            end (datetime.datetime): The end of the delivery window.
            discharge (bool): Whether the bid is a sell (True) or buy
                (False) bid.

        Returns:
            float: The bid price.
        """
        freq = unit.index.freq
        values = [
            self._marginal_value[t]
            for t in self._marginal_value
            if start <= t < end
        ]
        value = float(np.mean(values)) if values else 0.0

        efficiency = unit.efficiency_charge * unit.efficiency_discharge
        if discharge:
            bid_price = value / np.sqrt(efficiency)
            bid_price += unit.additional_cost_discharge + self.degradation_cost
        else:
            bid_price = value * np.sqrt(efficiency)
            bid_price -= unit.additional_cost_charge

        # deadband: widen the band symmetrically, default is the price tick
        deadband = self.deadband
        if deadband is None:
            deadband = market_config.price_tick or 0.0
        if discharge:
            bid_price += deadband
        else:
            bid_price -= deadband

        return float(bid_price)

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates the bids for the given products of the given market.

        On the first call, the markets of the world are discovered. The
        optimization follows a rolling horizon: the plan is optimized over
        the foresight window and reused until the overlap period has passed
        (by default until the next market opening). Each market opening only
        extracts its own bids from the shared plan.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            market_config (MarketConfig): The market to bid on.
            product_tuples (list[Product]): The products to bid on.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: The bids for the given market.
        """
        if self.markets is None:
            self._discover_markets(unit)

        market_info = self.markets.get(market_config.market_id)
        if market_info is None:
            logger.warning(
                "unit %s has no eligibility for market %s - no bids",
                unit.id,
                market_config.market_id,
            )
            return []

        # the decision time is the opening time of the market: the products
        # start at opening + first_delivery (+ duration * i)
        first_delivery = min(
            product.first_delivery for product in market_config.market_products
        )
        decision_time = min(product[0] for product in product_tuples) - first_delivery

        # rolling horizon: re-optimize when the current plan is expired
        # (the overlap period has passed) or when the actual SoC deviated
        # from the planned trajectory. Markets opening at the same decision
        # time share one optimization.
        if (
            self._plan_time is None
            # a later decision time beyond the plan validity re-optimizes
            # (with overlap = 0h, every later opening re-optimizes, while
            # markets opening at the same time share one optimization)
            or (decision_time > self._plan_time and decision_time >= self._plan_valid_until)
            or self._needs_reoptimization(unit, decision_time)
        ):
            self._cache_solution = self._optimize(unit, decision_time)
            # the plan is used until the overlap period has passed
            self._plan_time = decision_time
            self._plan_valid_until = decision_time + self.overlap

        bids = []
        for product in product_tuples:
            start, end, only_hours = product
            if only_hours is not None:
                # peak/off-peak products are not supported yet
                logger.debug("only_hours products are not supported by this strategy")
                continue

            entry = self._cache_solution.get((market_config.market_id, start, end))
            if entry is None:
                continue

            # net out sell and buy volumes: the LP is indifferent between
            # simultaneous buy and sell at the same price (circular flow
            # through the storage loses energy but does not change profit),
            # so the solver may return both - the net volume is what matters
            sell = entry.get("sell", 0.0)
            buy = entry.get("buy", 0.0)
            net = sell - buy

            if net > 1e-9 or net < -1e-9:
                bid_price = self._bid_price(
                    unit, market_config, start, end, discharge=net > 0
                )
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": bid_price,
                        "volume": net,
                        "node": unit.node,
                    }
                )

        # respect the market price and volume limits
        for bid in bids:
            if market_config.maximum_bid_price is not None:
                bid["price"] = min(bid["price"], market_config.maximum_bid_price)
            if market_config.minimum_bid_price is not None:
                bid["price"] = max(bid["price"], market_config.minimum_bid_price)
            if (
                market_config.maximum_bid_volume is not None
                and abs(bid["volume"]) > market_config.maximum_bid_volume
            ):
                bid["volume"] = (
                    np.sign(bid["volume"]) * market_config.maximum_bid_volume
                )
            if "bid_type" in market_config.additional_fields:
                bid["bid_type"] = "SB"

        return self.remove_empty_bids(bids)
