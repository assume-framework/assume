# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from itertools import groupby
from operator import itemgetter

import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    Suffix,
    Var,
    maximize,
)
from pyomo.opt import SolverFactory, check_available_solvers

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]


class NodalPyomoMarketRole(MarketRole):
    required_fields = ["node_id"]

    def __init__(
        self,
        marketconfig: MarketConfig,
        nodes=[0, 1, 2],
        network={"Line_0": (0, 1, 100), "Line_1": (1, 2, 100), "Line_2": (2, 0, 100)},
    ):
        """
        Network can be for example:

        defined as connections between nodes as a tuple of (node1, node2, capacity)
        network = {"Line_0": (0, 1, 0), "Line_1": (1, 2, 0), "Line_2": (2, 0, 0)}

        or with added congestion
        network = {"Line_0": (0, 1, 100), "Line_1": (1, 2, 100), "Line_2": (2, 0, 100)}
        """
        super().__init__(marketconfig)
        self.nodes = nodes
        self.network = network

        self.incidence_matrix = pd.DataFrame(0, index=self.nodes, columns=self.network)
        for i, (node1, node2, capacity) in self.network.items():
            self.incidence_matrix.at[node1, i] = 1
            self.incidence_matrix.at[node2, i] = -1

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        Performs a nodal pricing optimization using the Pyomo library.
        It takes market orders, simulates network congestion, and computes optimal power generation and
        consumption at different nodes while considering constraints and objectives.
        The results are used to update order information and collect meta-information for reporting.

        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]
        :return: accepted_orders, [], meta
        :rtype: tuple[Orderbook, Orderbook, list[dict]]
        """
        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            product_orders = list(product_orders)
            if product[0:3] not in market_products:
                rejected_orders.extend(product_orders)
                # log.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            # volume 0 is ignored/invalid

            if "acceptance_ratio" in self.marketconfig.additional_fields:
                supply_bids = list(
                    map(
                        itemgetter(
                            "node_id", "price", "volume", "agent_id", "acceptance_ratio"
                        ),
                        supply_orders,
                    )
                )
                demand_bids = []
                for order in demand_orders:
                    demand_bids.append(
                        (
                            order["node_id"],
                            order["price"],
                            -order["volume"],
                            order["agent_id"],
                            order["acceptance_ratio"],
                        )
                    )
            else:
                supply_bids = list(
                    map(
                        itemgetter("node_id", "price", "volume", "agent_id"),
                        supply_orders,
                    )
                )
                demand_bids = []
                for order in demand_orders:
                    demand_bids.append(
                        (
                            order["node_id"],
                            order["price"],
                            -order["volume"],
                            order["agent_id"],
                        )
                    )
            # Create a model
            model = ConcreteModel()

            model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

            # Create power variable for generation and consumption
            model.p_generation = Var(range(len(supply_bids)), domain=NonNegativeReals)
            model.p_consumption = Var(range(len(demand_bids)), domain=NonNegativeReals)

            # Create a set for the lines in the network
            model.lines = Set(initialize=self.network.keys())
            # Create a variable for the flow over each line
            model.flow = Var(model.lines, domain=Reals)

            # Create a constraint that the flow over each line must be less than or equal to the capacity of the line
            model.capacity_constraint = ConstraintList()
            for i, (node1, node2, capacity) in self.network.items():
                model.capacity_constraint.add(model.flow[i] <= capacity)
                model.capacity_constraint.add(model.flow[i] >= -capacity)

            # Create a constraint that the flow over each line must be less than or equal to the capacity of the line
            model.balance_constraint = ConstraintList()
            for node in self.nodes:
                model.balance_constraint.add(
                    sum(
                        model.p_generation[i]
                        for i in range(len(supply_bids))
                        if supply_bids[i][0] == node
                    )
                    - sum(
                        self.incidence_matrix.at[node, i] * model.flow[i]
                        for i in self.network.keys()
                    )
                    - sum(
                        model.p_consumption[i]
                        for i in range(len(demand_bids))
                        if demand_bids[i][0] == node
                    )
                    == 0
                )

            if "acceptance_ratio" in self.marketconfig.additional_fields:
                # Maximum power generation constraint
                model.max_generation = ConstraintList()
                for i, (node, price, volume, bid_id, ratio) in enumerate(supply_bids):
                    model.max_generation.add(model.p_generation[i] <= volume)

                # Maximum power consumption constraint
                model.max_consumption = ConstraintList()
                for i, (node, price, volume, bid_id, ratio) in enumerate(demand_bids):
                    model.max_consumption.add(model.p_consumption[i] <= volume)

                # Minimum power generation constraint
                model.min_generation = ConstraintList()
                for i, (node, price, volume, bid_id, ratio) in enumerate(supply_bids):
                    model.max_generation.add(
                        model.p_generation[i] == 0
                        or model.p_generation[i] >= volume * ratio
                    )

                # Minimum power consumption constraint
                model.min_consumption = ConstraintList()
                for i, (node, price, volume, bid_id, ratio) in enumerate(demand_bids):
                    model.max_consumption.add(
                        model.p_consumption[i] == 0
                        or model.p_generation[i] >= volume * ratio
                    )
            else:
                # Maximum power generation constraint
                model.max_generation = ConstraintList()
                for i, (node, price, volume, bid_id) in enumerate(supply_bids):
                    model.max_generation.add(model.p_generation[i] <= volume)

                # Maximum power consumption constraint
                model.max_consumption = ConstraintList()
                for i, (node, price, volume, bid_id) in enumerate(demand_bids):
                    model.max_consumption.add(model.p_consumption[i] <= volume)

            # Obective function
            model.obj = Objective(
                expr=sum(
                    model.p_consumption[i] * demand_bids[i][1]
                    for i in range(len(demand_bids))
                )
                - sum(
                    model.p_generation[i] * supply_bids[i][1]
                    for i in range(len(supply_bids))
                ),
                sense=maximize,
            )

            # Create a solver
            solvers = check_available_solvers(*SOLVERS)
            if len(solvers) < 1:
                raise Exception(f"None of {SOLVERS} are available")
            solver = SolverFactory(solvers[0])

            # Solve the model
            result = solver.solve(model)

            if not result["Solver"][0]["Status"] == "ok":
                raise Exception("infeasible")

            # Find the dual variable for the balance constraint
            duals_dict = {str(key): -model.dual[key] for key in model.dual.keys()}

            # Find sum of generation per node
            generation = {node: 0 for node in self.nodes}
            consumption = {node: 0 for node in self.nodes}
            # add demand to accepted orders with confirmed volume
            for i in range(len(demand_orders)):
                node = demand_orders[i]["node_id"]
                opt_volume = model.p_consumption[i].value
                consumption[node] += opt_volume
                demand_orders[i]["volume"] = -opt_volume
                demand_orders[i]["price"] = duals_dict[f"balance_constraint[{node+1}]"]
                if opt_volume != 0:
                    accepted_orders.append(demand_orders[i])

            for i in range(len(supply_orders)):
                node = supply_orders[i]["node_id"]
                opt_volume = model.p_generation[i].value
                generation[node] += opt_volume
                supply_orders[i]["volume"] = opt_volume
                supply_orders[i]["price"] = duals_dict[f"balance_constraint[{node+1}]"]
                if opt_volume != 0:
                    accepted_orders.append(supply_orders[i])

            # calculate meta
            for node in self.nodes:
                # Find sum of power flowing into each node
                power_in = sum(
                    self.incidence_matrix.at[node, i] * model.flow[i]()
                    for i in self.network.keys()
                )
                meta.append(
                    {
                        "supply_volume": generation[node],
                        "demand_volume": consumption[node],
                        "uniform_price": duals_dict[f"balance_constraint[{node+1}]"],
                        "price": duals_dict[f"balance_constraint[{node+1}]"],
                        "node_id": node,
                        "flow": power_in,
                        "product_start": product[0],
                        "product_end": product[1],
                        "only_hours": product[2],
                    }
                )
        return accepted_orders, [], meta
