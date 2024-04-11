# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from operator import itemgetter

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition, check_available_solvers

from assume.common.market_objects import MarketConfig, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["gurobi", "glpk"]
EPS = 1e-4


def market_clearing_opt(orders, market_products):
    time_dt = market_products[1][0] - market_products[0][0]
    complex_bids = {
        order["bid_id"]: order for order in orders if order["bid_type"] == "MPB"
    }

    model = pyo.ConcreteModel()

    model.T = pyo.Set(
        initialize=[market_product[0] for market_product in market_products],
        doc="timesteps",
    )
    model.sBids = pyo.Set(
        initialize=[order["bid_id"] for order in orders if order["bid_type"] == "SB"],
        doc="simple_bids",
    )
    model.bBids = pyo.Set(
        initialize=[order["bid_id"] for order in orders if order["bid_type"] == "MPB"],
        doc="block_bids",
    )

    model.xs = pyo.Var(
        model.sBids,
        domain=pyo.NonNegativeReals,
        doc="simple_bid_acceptance",
        bounds=(0, 1),
    )

    model.gen_b = pyo.Var(
        model.bBids,
        model.T,
        domain=pyo.NonNegativeReals,
        doc="block_bid_acceptance",
    )

    model.c_up = pyo.Var(
        model.bBids,
        model.T,
        domain=pyo.NonNegativeReals,
        doc="start_up_cost",
    )

    model.c_down = pyo.Var(
        model.bBids,
        model.T,
        domain=pyo.NonNegativeReals,
        doc="shut_down_cost",
    )

    model.status = pyo.Var(
        model.bBids,
        model.T,
        domain=pyo.Binary,
        doc="status",
    )

    balance_expr = {t: 0.0 for t in model.T}
    for order in orders:
        if order["bid_type"] == "SB":
            balance_expr[order["start_time"]] += (
                order["volume"] * model.xs[order["bid_id"]]
            )
        elif order["bid_type"] == "MPB":
            for start_time, price in order["price"].items():
                balance_expr[start_time] += model.gen_b[order["bid_id"], start_time]

    def energy_balance_rule(model, t):
        return balance_expr[t] == 0

    model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

    obj_expr = 0
    for order in orders:
        if order["bid_type"] == "SB":
            obj_expr += order["price"] * order["volume"] * model.xs[order["bid_id"]]
        elif order["bid_type"] == "MPB":
            for start_time, price in order["price"].items():
                obj_expr += price * model.gen_b[order["bid_id"], start_time]
                obj_expr += (
                    model.c_up[order["bid_id"], start_time]
                    + model.c_down[order["bid_id"], start_time]
                )

    model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

    def max_power_rule(model, bid_id, t):
        return (
            model.gen_b[bid_id, t]
            <= complex_bids[bid_id]["max_power"] * model.status[bid_id, t]
        )

    def min_power_rule(model, bid_id, t):
        return (
            model.gen_b[bid_id, t]
            >= complex_bids[bid_id]["min_power"] * model.status[bid_id, t]
        )

    model.min_power = pyo.Constraint(model.bBids, model.T, rule=min_power_rule)
    model.max_power = pyo.Constraint(model.bBids, model.T, rule=max_power_rule)

    def ramp_up_rule(model, bid_id, t):
        if t == market_products[0][0]:
            return (
                model.gen_b[bid_id, t]
                <= complex_bids[bid_id]["initial_output"]
                + complex_bids[bid_id]["ramp_up"]
            )
        else:
            return (
                model.gen_b[bid_id, t]
                <= model.gen_b[bid_id, t - time_dt] + complex_bids[bid_id]["ramp_up"]
            )

    def ramp_down_rule(model, bid_id, t):
        if t == market_products[0][0]:
            return (
                model.gen_b[bid_id, t]
                >= complex_bids[bid_id]["initial_output"]
                - complex_bids[bid_id]["ramp_down"]
            )
        else:
            return (
                model.gen_b[bid_id, t]
                >= model.gen_b[bid_id, t - time_dt] - complex_bids[bid_id]["ramp_down"]
            )

    model.ramp_up = pyo.Constraint(model.bBids, model.T, rule=ramp_up_rule)
    model.ramp_down = pyo.Constraint(model.bBids, model.T, rule=ramp_down_rule)

    def start_up_rule(model, bid_id, t):
        if t == market_products[0][0]:
            return (
                model.c_up[bid_id, t]
                - (model.status[bid_id, t] - complex_bids[bid_id]["initial_status"])
                * complex_bids[bid_id]["start_up_cost"]
                >= 0
            )
        else:
            return (
                model.c_up[bid_id, t]
                - (model.status[bid_id, t] - model.status[bid_id, t - time_dt])
                * complex_bids[bid_id]["start_up_cost"]
                >= 0
            )

    def shut_down_rule(model, bid_id, t):
        if t == market_products[0][0]:
            return (
                model.c_down[bid_id, t]
                - (complex_bids[bid_id]["initial_status"] - model.status[bid_id, t])
                * complex_bids[bid_id]["shut_down_cost"]
                >= 0
            )
        else:
            return (
                model.c_down[bid_id, t]
                - (model.status[bid_id, t - time_dt] - model.status[bid_id, t])
                * complex_bids[bid_id]["shut_down_cost"]
                >= 0
            )

    model.start_up = pyo.Constraint(model.bBids, model.T, rule=start_up_rule)
    model.shut_down = pyo.Constraint(model.bBids, model.T, rule=shut_down_rule)

    solvers = check_available_solvers(*SOLVERS)
    if len(solvers) < 1:
        raise Exception(f"None of {SOLVERS} are available")

    solver = SolverFactory(solvers[0])
    options = {"OptimalityTol": EPS} if solver.name == "gurobi" else {}

    # Solve the model
    instance = model.create_instance()
    solver.solve(instance, options=options)

    # make new instance with fixed u
    instance_fixed_u = model.create_instance()
    instance_fixed_u.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    for bid_id in instance.bBids:
        for t in instance.T:
            instance_fixed_u.status[bid_id, t].fix(instance.status[bid_id, t].value)

    results = solver.solve(instance_fixed_u, options=options)

    return instance_fixed_u, results


class UCClearingRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def validate_orderbook(self, orderbook: Orderbook, agent_tuple) -> None:
        super().validate_orderbook(orderbook, agent_tuple)

        for order in orderbook:
            order["bid_type"] = (
                "SB" if order.get("bid_type") is None else order["bid_type"]
            )
            assert order["bid_type"] in [
                "SB",
                "MPB",
            ], f"bid_type {order['bid_type']} not in ['SB', 'MPB']"

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        This implements pay-as-clear with more complex bid structures, including acceptance ratios, bid types, and profiled volumes.

        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]

        :return extract_results(model=model, eps=eps, orders=orders, market_products=market_products, market_clearing_prices=market_clearing_prices)
        :rtype: tuple[Orderbook, Orderbook, list[dict]]
        """

        if len(orderbook) == 0:
            return [], [], []

        market_getter = itemgetter("start_time", "end_time")
        orderbook.sort(key=market_getter)

        for order in orderbook:
            order["accepted_price"] = {}
            order["accepted_volume"] = {}

        rejected_orders: Orderbook = []

        orderbook = check_for_tensors(orderbook)

        instance, results = market_clearing_opt(
            orders=orderbook,
            market_products=market_products,
        )

        if results.solver.termination_condition == TerminationCondition.infeasible:
            raise Exception("infeasible")

        # extract dual from model.energy_balance
        market_clearing_prices = {
            t: instance.dual[instance.energy_balance[t]] for t in instance.T
        }

        self.all_orders = []

        return extract_results(
            model=instance,
            orders=orderbook,
            rejected_orders=rejected_orders,
            market_products=market_products,
            market_clearing_prices=market_clearing_prices,
        )


def extract_results(
    model,
    orders,
    rejected_orders,
    market_products,
    market_clearing_prices,
):
    accepted_orders: Orderbook = []
    meta = []

    supply_volume_dict = {t: 0.0 for t in model.T}
    demand_volume_dict = {t: 0.0 for t in model.T}

    for order in orders:
        if order["bid_type"] == "SB":
            acceptance = model.xs[order["bid_id"]].value
            acceptance = 0 if acceptance < EPS else acceptance
            order["accepted_volume"] = acceptance * order["volume"]
            order["accepted_price"] = market_clearing_prices[order["start_time"]]

            if order["accepted_volume"] > 0:
                supply_volume_dict[order["start_time"]] += order["accepted_volume"]
            else:
                demand_volume_dict[order["start_time"]] += order["accepted_volume"]

        elif order["bid_type"] == "MPB":
            acceptance = 0
            for start_time, price in order["price"].items():
                acceptance += model.gen_b[order["bid_id"], start_time].value
                order["accepted_volume"][start_time] = model.gen_b[
                    order["bid_id"], start_time
                ].value
                order["accepted_price"][start_time] = market_clearing_prices[start_time]

                if order["accepted_volume"][start_time] > 0:
                    supply_volume_dict[start_time] += order["accepted_volume"][
                        start_time
                    ]
                else:
                    demand_volume_dict[start_time] += order["accepted_volume"][
                        start_time
                    ]

        if acceptance > 0:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    for product in market_products:
        t = product[0]

        clear_price = market_clearing_prices[t]

        supply_volume = supply_volume_dict[t]
        demand_volume = demand_volume_dict[t]
        duration_hours = (product[1] - product[0]) / timedelta(hours=1)

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "demand_volume_energy": demand_volume * duration_hours,
                "supply_volume_energy": supply_volume * duration_hours,
                "price": clear_price,
                "max_price": clear_price,
                "min_price": clear_price,
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    return accepted_orders, rejected_orders, meta


def check_for_tensors(orderbook):
    """
    Checks if the data contains tensors and converts them to floats.

    :param data: The data to be checked.
    :type data: any
    """
    try:
        import torch as th

        # data is a list of dictionaries
        # check if any value in those dictionaries is a tensor
        for order in orderbook:
            if isinstance(order["price"], dict):
                for start, value in order["price"].items():
                    if isinstance(value, th.Tensor):
                        order["price"][start] = value.item()
            if isinstance(order["volume"], dict):
                for start, value in order["volume"].items():
                    if isinstance(value, th.Tensor):
                        order["volume"][start] = value.item()

    except ImportError:
        pass

    return orderbook
