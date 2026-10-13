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
    and enter the optimization as fixed parameters. The solution is cached
    per decision time, so that multiple markets opening at the same time
    reuse one optimization and each market only receives its own slice of
    the shared plan.

    Attributes:
        foresight (datetime.timedelta): The user-defined foresight window for
            the optimization.
        price_epsilon (float): The price margin added/subtracted from the
            price forecast when setting bid prices.

    Args:
        foresight (str, optional): The foresight window (e.g. "24h").
            Defaults to "24h".
        price_epsilon (float, optional): Price margin for bid prices.
            Defaults to 0.1.
        solver_name (str, optional): The name of the solver to use. Defaults
            to the first available of appsi_highs, gurobi, glpk, cbc, cplex.
        **kwargs: Additional keyword arguments, including the bidding
            parameters of the world.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = parse_duration(kwargs.get("foresight", "24h"))
        self.price_epsilon = float(kwargs.get("price_epsilon", 0.1))
        self.solver_name = kwargs.get("solver_name", None)

        # discovered markets - filled on first calculate_bids call
        self.markets: dict[str, MarketInfo] | None = None
        # solver instance - created lazily to fail late if no solver exists
        self.opt = None
        # cached optimization result: fingerprint of the last optimization
        # and the solution mapping (market_id, start, end) -> bid quantities
        self._cache_fingerprint: tuple | None = None
        self._cache_solution: dict[tuple, dict[str, float]] = {}

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

        Falls back to the EOM price forecast for energy-type markets if no
        dedicated forecast exists. Capacity markets without a forecast are
        priced at zero.

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
            # capacity markets without forecast are priced at zero
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
            with_terminal=True,
        )

        # objective: maximize the expected profit over all markets
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
                # energy sells earn the price, energy buys pay it
                obj_terms.append(price * hours * (model.sell[i] - model.buy[i]))
        model.obj = pyo.Objective(expr=pyo.quicksum(obj_terms), sense=pyo.maximize)

        solution = self._solve(model, windows, sell_indices, buy_indices)
        if solution is None:
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
            return {}

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

        # windows covering each time step
        covering = [[] for _ in range(n)]
        for i, (_, w_start, w_end) in enumerate(windows):
            for t in range(n):
                if w_start <= times[t] < w_end:
                    covering[t].append(i)

        def _covered(i: int, t: int) -> bool:
            return i in covering[t]

        # SoC dynamics: energy flows of energy-type markets and the fixed
        # energy flows of cleared markets change the state of charge
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
            constraints.add(model.soc[t] == soc_prev + soc_delta)
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

        # terminal condition: do not end the horizon with less energy than
        # now (prevents myopic dumping of the stored energy)
        if with_terminal and n > 0:
            constraints.add(
                model.soc[n - 1]
                >= unit.outputs["soc"].at[decision_time] * unit.capacity
            )

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
        optimization is run once per decision time and cached; each market
        opening only extracts its own bids from the shared plan.

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

        # fingerprint to detect whether a re-optimization is needed: the
        # decision time and the cleared commitments of the unit
        freq = unit.index.freq
        horizon_end = min(decision_time + self.foresight, unit.index[-1] + freq)
        n = int((horizon_end - decision_time) / freq)
        if n <= 0:
            return []
        times = [decision_time + freq * i for i in range(n)]
        fingerprint = (
            decision_time,
            tuple(
                float(np.sum(unit.outputs[product_type].loc[times[0] : times[-1]]))
                for product_type in ("energy", "capacity_pos", "capacity_neg")
            ),
        )

        if fingerprint != self._cache_fingerprint:
            self._cache_solution = self._optimize(unit, decision_time)
            self._cache_fingerprint = fingerprint

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

            prices = self._get_price_forecast(unit, market_info, start, end)
            price = float(np.mean(prices)) if len(prices) else 0.0

            # net out sell and buy volumes: the LP is indifferent between
            # simultaneous buy and sell at the same price (circular flow
            # through the storage loses energy but does not change profit),
            # so the solver may return both - the net volume is what matters
            sell = entry.get("sell", 0.0)
            buy = entry.get("buy", 0.0)
            net = sell - buy

            if net > 1e-9:
                if market_info.product_type in ("energy", "heat"):
                    bid_price = price - self.price_epsilon
                else:
                    bid_price = price
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
            elif net < -1e-9:
                if market_info.product_type in ("energy", "heat"):
                    bid_price = price + self.price_epsilon
                else:
                    bid_price = price
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
