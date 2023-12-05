# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import cProfile
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


def market_clearing_opt(orders, market_products, mode, with_linked_bids):
    # with cProfile.Profile() as pr:
    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    model.T = pyo.Set(
        initialize=[market_product[0] for market_product in market_products],
        doc="timesteps",
    )
    model.sBids = pyo.Set(
        initialize=[order["bid_id"] for order in orders if order["bid_type"] == "SB"],
        doc="simple_bids",
    )
    model.bBids = pyo.Set(
        initialize=[
            order["bid_id"] for order in orders if order["bid_type"] in ["BB", "LB"]
        ],
        doc="block_bids",
    )

    model.xs = pyo.Var(
        model.sBids,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
        doc="simple_bid_acceptance",
    )
    model.xb = pyo.Var(
        model.bBids,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
        doc="block_bid_acceptance",
    )

    if mode == "with_min_acceptance_ratio":
        model.Bids = pyo.Set(
            initialize=[order["bid_id"] for order in orders], doc="all_bids"
        )
        model.x = pyo.Var(
            model.Bids,
            domain=pyo.Binary,
            doc="bid_accepted",
        )

        model.mar_constr = pyo.ConstraintList()
        for order in orders:
            if order["min_acceptance_ratio"] is None:
                continue
            elif order["bid_type"] == "SB":
                model.mar_constr.add(
                    model.xs[order["bid_id"]]
                    >= order["min_acceptance_ratio"] * model.x[order["bid_id"]]
                )
                model.mar_constr.add(
                    model.xs[order["bid_id"]] <= model.x[order["bid_id"]]
                )

            elif order["bid_type"] in ["BB", "LB"]:
                model.mar_constr.add(
                    model.xb[order["bid_id"]]
                    >= order["min_acceptance_ratio"] * model.x[order["bid_id"]]
                )
                model.mar_constr.add(
                    model.xb[order["bid_id"]] <= model.x[order["bid_id"]]
                )

    balance_expr = {t: 0.0 for t in model.T}
    for order in orders:
        if order["bid_type"] == "SB":
            balance_expr[order["start_time"]] += (
                order["volume"] * model.xs[order["bid_id"]]
            )
        elif order["bid_type"] in ["BB", "LB"]:
            for start_time, volume in order["volume"].items():
                balance_expr[start_time] += volume * model.xb[order["bid_id"]]

    def energy_balance_rule(m, t):
        return balance_expr[t] == 0

    model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

    if with_linked_bids:
        model.linked_bid_constr = pyo.ConstraintList()
        for order in orders:
            if "parent_bid_id" in order.keys() and order["parent_bid_id"] is not None:
                parent_bid_id = order["parent_bid_id"]
                model.linked_bid_constr.add(
                    model.xb[order["bid_id"]] <= model.xb[parent_bid_id]
                )

    obj_expr = 0
    for order in orders:
        if order["bid_type"] == "SB":
            obj_expr += order["price"] * order["volume"] * model.xs[order["bid_id"]]
        elif order["bid_type"] in ["BB", "LB"]:
            for start_time, volume in order["volume"].items():
                obj_expr += order["price"] * volume * model.xb[order["bid_id"]]

    model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

    solvers = check_available_solvers(*SOLVERS)
    if len(solvers) < 1:
        raise Exception(f"None of {SOLVERS} are available")

    solver = SolverFactory(solvers[0])

    if solver.name == "gurobi":
        options = {"cutoff": -1.0, "eps": EPS}
    elif solver.name == "cplex":
        options = {
            "mip.tolerances.lowercutoff": -1.0,
            "mip.tolerances.absmipgap": EPS,
        }
    elif solver.name == "cbc":
        options = {"sec": 60, "ratio": 0.1}
    # elif solver.name == "glpk":
    #     options = {"tmlim": 60, "mipgap": 0.1}
    else:
        options = {}

    # Solve the model
    instance = model.create_instance()
    results = solver.solve(instance, options=options)

    # fix all model.x to the values in the solution
    if mode == "with_min_acceptance_ratio":
        for bid_id in instance.Bids:
            instance.x[bid_id].fix(instance.x[bid_id].value)

        # resolve the model
        results = solver.solve(instance, options=options)

    # pr.print_stats(sort='cumulative')

    return instance, results


class ComplexClearingRole(MarketRole):
    required_fields = ["bid_type"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def validate_orderbook(self, orderbook: Orderbook, agent_tuple) -> None:
        super().validate_orderbook(orderbook, agent_tuple)
        max_volume = self.marketconfig.maximum_bid_volume
        for order in orderbook:
            order["bid_type"] = "SB" if order["bid_type"] is None else order["bid_type"]
            assert order["bid_type"] in [
                "SB",
                "BB",
                "LB",
            ], f"bid_type {order['bid_type']} not in ['SB', 'BB', 'LB']"

            if order["bid_type"] in ["BB", "LB"]:
                assert False not in [
                    abs(volume) <= max_volume for _, volume in order["volume"].items()
                ], f"max_volume {order['volume']}"

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        This implements pay-as-clear with more complex bid structures, including acceptance ratios, bid types, and profiled volumes.

        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]

        :return extract_results(model=model, eps=eps, orders=orders, market_products=market_products, market_clearing_prices=market_clearing_prices)
        :rtype: tuple[Orderbook, Orderbook, list[dict]]
        """

        if len(orderbook) == 0:
            return [], [], []

        orderbook.sort(key=itemgetter("start_time", "end_time", "only_hours"))

        child_orders = []
        for order in orderbook:
            order["accepted_price"] = {}
            order["accepted_volume"] = {}
            # get child linked bids
            if "parent_bid_id" in order.keys() and order["parent_bid_id"] is not None:
                # check whether the parent bid is in the orderbook
                parent_bid_id = order["parent_bid_id"]
                parent_bid = next(
                    (bid for bid in orderbook if bid["bid_id"] == parent_bid_id), None
                )
                if parent_bid is None:
                    order["parent_bid_id"] = None
                    log.warning(f"Parent bid {parent_bid_id} not in orderbook")
                else:
                    child_orders.append(order)

        with_linked_bids = bool(child_orders)
        rejected_orders: Orderbook = []

        mode = "default"
        if "min_acceptance_ratio" in self.marketconfig.additional_fields:
            mode = "with_min_acceptance_ratio"

        while True:
            instance, results = market_clearing_opt(
                orders=orderbook,
                market_products=market_products,
                mode=mode,
                with_linked_bids=with_linked_bids,
            )

            if results.solver.termination_condition == TerminationCondition.infeasible:
                raise Exception("infeasible")

            # extract dual from model.energy_balance
            market_clearing_prices = {
                t: instance.dual[instance.energy_balance[t]] for t in instance.T
            }

            # check the profit of each order and remove those with negative profit
            orders_profit = []
            for order in orderbook:
                children = []
                if with_linked_bids:
                    children = [
                        child
                        for child in child_orders
                        if child["parent_bid_id"] == order["bid_id"]
                    ]

                order_profit = calculate_order_profit(
                    order, market_clearing_prices, instance, children
                )

                # correct rounding
                if order_profit != 0 and abs(order_profit) < EPS:
                    order_profit = 0

                orders_profit.append(order_profit)

                if order_profit < 0:
                    rejected_orders.append(order)
                    orderbook.remove(order)
                    rejected_orders.extend(children)
                    for child in children:
                        orderbook.remove(child)

            # check if all orders have positive profit
            if all(order_profit >= 0 for order_profit in orders_profit):
                break

        return extract_results(
            model=instance,
            orders=orderbook,
            rejected_orders=rejected_orders,
            market_products=market_products,
            market_clearing_prices=market_clearing_prices,
        )


def calculate_order_profit(order, market_clearing_prices, instance, children):
    order_profit = 0

    if order["bid_type"] == "SB":
        if (
            pyo.value(instance.xs[order["bid_id"]]) < EPS
            or abs(market_clearing_prices[order["start_time"]] - order["price"]) < EPS
        ):
            order_profit = 0
        else:
            order_profit = (
                (market_clearing_prices[order["start_time"]] - order["price"])
                * order["volume"]
                * pyo.value(instance.xs[order["bid_id"]])
            )
    elif order["bid_type"] in ["BB", "LB"]:
        bid_volume = sum(order["volume"].values())
        if pyo.value(instance.xb[order["bid_id"]]) < EPS:
            order_profit = 0
        else:
            order_profit = (
                sum(market_clearing_prices[t] * v for t, v in order["volume"].items())
                - order["price"] * bid_volume
            ) * pyo.value(instance.xb[order["bid_id"]])

        # add the child linked bids
        for child_order in children:
            child_profit = (
                sum(
                    market_clearing_prices[t] * v
                    for t, v in child_order["volume"].items()
                )
                - child_order["price"] * bid_volume
            ) * pyo.value(instance.xb[child_order["bid_id"]])
            if child_profit > 0:
                order_profit += child_profit

    # correct rounding
    if order_profit != 0 and abs(order_profit) < EPS:
        order_profit = 0

    return order_profit


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

        elif order["bid_type"] in ["BB", "LB"]:
            acceptance = model.xb[order["bid_id"]].value
            acceptance = 0 if acceptance < EPS else acceptance

            for start_time, volume in order["volume"].items():
                order["accepted_volume"][start_time] = acceptance * volume
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
                "demand_volume": -demand_volume,
                "demand_volume_energy": -demand_volume * duration_hours,
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
