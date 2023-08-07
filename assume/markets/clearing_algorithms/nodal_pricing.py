import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]


def nodal_pricing_example(
    market_agent: MarketRole, market_products: list[MarketProduct]
):
    assert "node_id" in market_agent.marketconfig.additional_fields
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

    # define list of nodes
    nodes = [0, 1, 2]

    # define a dict with connections between nodes as a tuple of (node1, node2, capacity)
    network = {"Line_0": (0, 1, 0), "Line_1": (1, 2, 0), "Line_2": (2, 0, 0)}

    # add congestion
    network = {"Line_0": (0, 1, 100), "Line_1": (1, 2, 100), "Line_2": (2, 0, 100)}

    incidence_matrix = pd.DataFrame(0, index=nodes, columns=network)
    for i, (node1, node2, capacity) in network.items():
        incidence_matrix.at[node1, i] = 1
        incidence_matrix.at[node2, i] = -1

    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        product_orders = list(product_orders)
        if product[0:3] not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        demand_orders = list(filter(lambda x: x["volume"] < 0, product_orders))
        supply_orders = list(filter(lambda x: x["volume"] > 0, product_orders))
        # volume 0 is ignored/invalid

        if "acceptance_ratio" in market_agent.marketconfig.additional_fields:
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
                map(itemgetter("node_id", "price", "volume", "agent_id"), supply_orders)
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
        model.lines = Set(initialize=network.keys())
        # Create a variable for the flow over each line
        model.flow = Var(model.lines, domain=Reals)

        # Create a constraint that the flow over each line must be less than or equal to the capacity of the line
        model.capacity_constraint = ConstraintList()
        for i, (node1, node2, capacity) in network.items():
            model.capacity_constraint.add(model.flow[i] <= capacity)
            model.capacity_constraint.add(model.flow[i] >= -capacity)

        # Create a constraint that the flow over each line must be less than or equal to the capacity of the line
        model.balance_constraint = ConstraintList()
        for node in nodes:
            model.balance_constraint.add(
                sum(
                    model.p_generation[i]
                    for i in range(len(supply_bids))
                    if supply_bids[i][0] == node
                )
                - sum(
                    incidence_matrix.at[node, i] * model.flow[i] for i in network.keys()
                )
                - sum(
                    model.p_consumption[i]
                    for i in range(len(demand_bids))
                    if demand_bids[i][0] == node
                )
                == 0
            )

        if "acceptance_ratio" in market_agent.marketconfig.additional_fields:
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
        generation = {node: 0 for node in nodes}
        consumption = {node: 0 for node in nodes}
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

        # Find sum of power flowing into each node
        power_in = {
            node: sum(
                incidence_matrix.at[node, i] * model.flow[i]() for i in network.keys()
            )
            for node in nodes
        }
        for node in nodes:
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
    return accepted_orders, meta
