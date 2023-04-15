import logging
from itertools import groupby
from operator import itemgetter
from typing import Callable

from assume.common.marketclasses import MarketConfig, MarketProduct, Order, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


def cumsum(orderbook: Orderbook):
    sum_ = 0
    for order in orderbook:
        sum_ += order["volume"]
        order["cumsum"] = sum_
    return orderbook


# does not allow to have partially accepted bids
# all or nothing
def pay_as_clear_aon(market_agent: MarketRole, market_products: list[MarketProduct]):
    """
    This implements pay-as-clear where each bids volume needs an exactly matching order with the same volume.
    Partial clearing is not allowed here.
    This has the side effect, that the cleared price can be much higher if bids with different volume are accepted
    """
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    clear_price = market_agent.marketconfig.minimum_bid
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        # groupby does only group consecutive groups
        product_orders = sorted(product_orders, key=lambda x: abs(x["volume"]))
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            orders = list(orders)
            demand_orders = filter(lambda x: x["volume"] < 0, orders)
            supply_orders = filter(lambda x: x["volume"] > 0, orders)
            # volume 0 is ignored/invalid

            # generation
            sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])

            # demand
            sorted_demand_orders = sorted(
                demand_orders, key=lambda i: i["price"], reverse=True
            )

            min_len = min(len(sorted_supply_orders), len(sorted_demand_orders))
            i = 0
            for i in range(min_len):
                if sorted_supply_orders[i]["price"] <= sorted_demand_orders[i]["price"]:
                    # pay as clear - all accepted receive the highest needed/cleared price
                    if clear_price < sorted_supply_orders[i]["price"]:
                        clear_price = sorted_supply_orders[i]["price"]
                else:
                    # as we have sorted before, the other bids/supply_orders can't be matched either
                    # once we get here
                    break
            # resulting i is the cut point
            accepted_product_orders.extend(sorted_demand_orders[:i])
            accepted_product_orders.extend(sorted_supply_orders[:i])
            rejected_orders.extend(sorted_demand_orders[i:])
            rejected_orders.extend(sorted_supply_orders[i:])

        for order in accepted_product_orders:
            order["price"] = clear_price

        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        accepted_demand_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
        demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("price"), accepted_supply_orders))
        if not prices:
            prices = [market_agent.marketconfig.maximum_bid]

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "uniform_price": clear_price,
                "price": clear_price,
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )
    # remember unused orders - necessary if the same hour will be cleared again
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future
    return accepted_orders, meta


# with partial accepted bids
def pay_as_clear(market_agent: MarketRole, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    clear_price = 0
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

        # generation
        sorted_supply_orders = sorted(supply_orders, key=itemgetter("price"))
        # demand
        sorted_demand_orders = sorted(
            demand_orders, key=itemgetter("price"), reverse=True
        )
        dem_vol, gen_vol = 0, 0
        # the following algorithm is inspired by one bar for generation and one for demand
        # add generation for currents demand price, until it matches demand
        # generation above it has to be sold for the lower price (or not at all)
        for i in range(len(sorted_demand_orders)):
            demand_order: Order = sorted_demand_orders[i]
            if not sorted_supply_orders:
                # if no more generation - reject left over demand
                rejected_orders.append(demand_order)
                continue

            assert dem_vol == gen_vol
            # now add the next demand order
            dem_vol += -demand_order["volume"]
            to_commit: Orderbook = []

            # and add supply until the demand order is matched
            while sorted_supply_orders and gen_vol < dem_vol:
                supply_order = sorted_supply_orders.pop(0)
                if supply_order["price"] <= demand_order["price"]:
                    to_commit.append(supply_order)
                    gen_vol += supply_order["volume"]
                else:
                    rejected_orders.append(supply_order)
            # now we know which orders we need
            # we only need to see how to arrange it.

            diff = gen_vol - dem_vol

            if diff < 0:
                # gen < dem
                # generation is not enough - split last demand bid
                split_demand_order = demand_order.copy()
                split_demand_order["volume"] = diff
                demand_order["volume"] -= diff
                rejected_orders.append(split_demand_order)
            elif diff > 0:
                # generation left over - split last generation bid
                supply_order = to_commit[-1]
                split_supply_order = supply_order.copy()
                split_supply_order["volume"] = diff
                supply_order["volume"] -= diff
                # changed supply_order is still part of to_commit and will be added
                # only volume-diff can be sold for current price
                gen_vol -= diff

                # add left over to supply_orders again
                sorted_supply_orders.insert(0, split_supply_order)
            # else: diff == 0 perfect match

            accepted_product_orders.append(demand_order)
            accepted_product_orders.extend(to_commit)

        # set clearing price - merit order - uniform pricing
        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        if len(accepted_supply_orders) > 0:
            clear_price = max(map(itemgetter("price"), accepted_supply_orders))
        else:
            clear_price = market_agent.marketconfig.maximum_bid
        for order in accepted_product_orders:
            order["price"] = clear_price
        accepted_orders.extend(accepted_product_orders)

        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        accepted_demand_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
        demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "uniform_price": clear_price,
                "price": clear_price,
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


# does not allow to have partial accepted bids
def pay_as_bid_aon(market_agent: MarketRole, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = sorted(product_orders, key=lambda x: abs(x["volume"]))
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            orders = list(orders)
            demand_orders = filter(lambda x: x["volume"] < 0, orders)
            supply_orders = filter(lambda x: x["volume"] > 0, orders)
            # volume 0 is ignored/invalid

            # generation
            sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])

            # demand
            sorted_demand_orders = sorted(
                demand_orders, key=lambda i: i["price"], reverse=True
            )

            min_len = min(len(sorted_supply_orders), len(sorted_demand_orders))
            i = 0
            for i in range(min_len):
                if sorted_supply_orders[i]["price"] <= sorted_demand_orders[i]["price"]:
                    # pay as bid - so the generator gets payed more than he needed to operate
                    sorted_supply_orders[i]["price"] = sorted_demand_orders[i]["price"]

                else:
                    # as we have sorted before, the other bids/supply_orders can't be matched either
                    # once we get here
                    break

            accepted_product_orders.extend(sorted_demand_orders[:i])
            accepted_product_orders.extend(sorted_supply_orders[:i])
            rejected_orders.extend(sorted_demand_orders[i:])
            rejected_orders.extend(sorted_supply_orders[i:])

        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        accepted_demand_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
        demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("price"), accepted_supply_orders))
        if not prices:
            prices = [market_agent.marketconfig.maximum_bid]

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "price": sum(prices) / len(prices),
                "max_price": max(prices),
                "min_price": min(prices),
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


# with partial accepted bids
def pay_as_bid(market_agent: MarketRole, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

        # generation
        sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])
        # demand
        sorted_demand_orders = sorted(
            demand_orders, key=lambda i: i["price"], reverse=True
        )

        dem_vol, gen_vol = 0, 0
        # the following algorithm is inspired by one bar for generation and one for demand
        # add generation for currents demand price, until it matches demand
        # generation above it has to be sold for the lower price (or not at all)
        for i in range(len(sorted_demand_orders)):
            demand_order: Order = sorted_demand_orders[i]
            if not sorted_supply_orders:
                # if no more generation - reject left over demand
                rejected_orders.append(demand_order)
                continue

            dem_vol += -demand_order["volume"]
            to_commit: Orderbook = []

            while sorted_supply_orders and gen_vol < dem_vol:
                supply_order = sorted_supply_orders.pop(0)
                if supply_order["price"] <= demand_order["price"]:
                    to_commit.append(supply_order)
                    gen_vol += supply_order["volume"]
                else:
                    rejected_orders.append(supply_order)
            # now we know which orders we need
            # we only need to see how to arrange it.

            diff = gen_vol - dem_vol

            if diff < 0:
                # gen < dem
                # generation is not enough - split demand
                split_demand_order = demand_order.copy()
                split_demand_order["volume"] = diff
                demand_order["volume"] -= diff
                rejected_orders.append(split_demand_order)
            elif diff > 0:
                # generation left over - split generation
                supply_order = to_commit[-1]
                split_supply_order = supply_order.copy()
                split_supply_order["volume"] = diff
                supply_order["volume"] -= diff
                # only volume-diff can be sold for current price
                # add left over to supply_orders again
                gen_vol -= diff

                sorted_supply_orders.insert(0, split_supply_order)
            # else: diff == 0 perfect match

            accepted_orders.append(demand_order)
            # pay as bid
            for supply_order in to_commit:
                supply_order["price"] = demand_order["price"]
            accepted_product_orders.extend(to_commit)

        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        accepted_demand_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
        demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("price"), accepted_supply_orders))
        if not prices:
            prices = [market_agent.marketconfig.maximum_bid]

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "price": sum(prices) / len(prices),
                "max_price": max(prices),
                "min_price": min(prices),
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


def nodal_pricing_pypsa_unflexible_demand(
    market_agent: MarketRole, market_products: list[MarketProduct]
):
    import pypsa

    # requires node_id
    n = pypsa.Network()
    for i in range(3):
        n.add("Bus", f"{i}")
    n.add("Link", "1 - 0", bus0="1", bus1="0", p_nom=300, p_min_pu=-1)
    n.add("Link", "2 - 0", bus0="2", bus1="0", p_nom=300, p_min_pu=-1)
    n.add("Link", "0 - 1", bus0="2", bus1="1", p_nom=300, p_min_pu=-1)

    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        # don't compare node_id too
        if product[0:3] not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        network = n.copy()
        # solve_time = product['start_time']
        # network.snapshots = [solve_time]
        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid
        # generation
        sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])
        # demand
        sorted_demand_orders = sorted(
            demand_orders, key=lambda i: i["price"], reverse=True
        )

        unique_idxs = list(range(len(sorted_supply_orders)))
        names = []
        for idx in unique_idxs:
            names.append(f"{sorted_supply_orders[idx]['agent_id']}_{idx}")
        network.madd(
            "Generator",
            names,
            bus=map(itemgetter("node_id"), sorted_supply_orders),
            p_nom=map(itemgetter("volume"), sorted_supply_orders),
            marginal_cost=map(itemgetter("price"), sorted_supply_orders),
        )
        # make sure enough generation exists
        # should be magic source
        network.generators.iloc[0, 1] = 100

        unique_idxs = list(range(len(sorted_demand_orders)))
        names = []
        for idx in unique_idxs:
            names.append(f"{sorted_demand_orders[idx]['agent_id']}_{idx}")
        network.madd(
            "Load",
            names,
            bus=map(itemgetter("node_id"), sorted_demand_orders),
            p_set=map(lambda o: -o["volume"], sorted_demand_orders)
            # XXX: does not respect cost of load
            # marginal_cost=map(itemgetter("price"), sorted_demand_orders),
        )

        status, solution = network.lopf()

        if status != "ok" or solution != "optimal":
            print(f"Demand to match: {network.loads.p_set.sum()}")
            print(f"Generation to match: {network.generators.p_nom.sum()}")
            raise Exception("could not solve")

        price_dict = network.buses_t.marginal_price.to_dict()
        load_dict = network.loads_t.p.to_dict()

        checker = lambda o: o["agent_id"] == agent_id
        price = sum([o["now"] for o in price_dict.values()]) / len(price_dict)
        for key in load_dict.keys():
            agent_id = "_".join(key.split("_")[:-1])
            nr = int(key.split("_")[-1]) - 1

            def check_agent_id(o):
                return o["agent_id"] == agent_id and o["node_id"] == nr

            orders = list(filter(check_agent_id, sorted_demand_orders))
            for o in orders:
                o["price"] = price_dict[str(o["node_id"])]["now"]
            accepted_orders.extend(orders)
            # can only accept all orders

        gen_dict = network.generators_t.p.to_dict()
        for order_key, order_val in gen_dict.items():
            agent_id = "_".join(key.split("_")[:-1])
            nr = int(key.split("_")[-1]) - 1

            def check_agent_id(o):
                return o["agent_id"] == agent_id and o["node_id"] == nr

            orders = list(filter(check_agent_id, sorted_supply_orders))
            for o in orders:
                o["volume"] = order_val["now"]

                if o["volume"] > 0:
                    accepted_orders.append(o)
                else:
                    rejected_orders.append(o)

        # links are not needed as congestion can't occur: network.links_t.p0
        # link shadow prices: network.links_t.mu_lower

        # TODO
        meta.append(
            {
                "supply_volume": network.generators.p_nom.sum(),
                "demand_volume": network.loads.p_set.sum(),
                "price": price,
                "node_id": 1,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    market_agent.all_orders = rejected_orders
    return accepted_orders, meta


def nodal_pricing_pypsa(market_agent: MarketRole, market_products: list[MarketProduct]):
    raise NotImplementedError()


def nodal_pricing_pyomo(market_agent: MarketRole, market_products: list[MarketProduct]):
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
    from pyomo.opt import SolverFactory

    # define list of nodes
    nodes = [0, 1, 2]

    # define a dict with connections between nodes as a tupple of (node1, node2, capacity)
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
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid
        supply_bids = list(
            map(itemgetter("node_id", "price", "volume", "agent_id"), supply_orders)
        )
        demand_bids = []
        for order in demand_orders:
            demand_bids.append(
                (order["node_id"], order["price"], -order["volume"], order["agent_id"])
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

        # Maximum power generation constraint
        model.max_generation = ConstraintList()
        for i, (node, price, quantity, bid_id) in enumerate(supply_bids):
            model.max_generation.add(model.p_generation[i] <= quantity)

        # Maximum power consumption constraint
        model.max_consumption = ConstraintList()
        for i, (node, price, quantity, bid_id) in enumerate(demand_bids):
            model.max_consumption.add(model.p_consumption[i] <= quantity)

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
        solver = SolverFactory("glpk")  # glpk

        # Solve the model
        result = solver.solve(model)

        if not result["Solver"][0]["Status"] == "ok":
            raise Exception("infeasible")

        # Find the dual variable for the balance constraint
        duals_dict = {str(key): -model.dual[key] for key in model.dual.keys()}

        # Find sum of generation per node
        generation = {
            node: sum(
                model.p_generation[i]()
                for i in range(len(supply_bids))
                if supply_bids[i][0] == node
            )
            for node in nodes
        }

        # Find sum of generation per node
        consumption = {
            node: sum(
                model.p_consumption[i]()
                for i in range(len(demand_bids))
                if demand_bids[i][0] == node
            )
            for node in nodes
        }

        # Find sum of power flowing into each node
        power_in = {
            node: sum(
                incidence_matrix.at[node, i] * model.flow[i]() for i in network.keys()
            )
            for node in nodes
        }
        # accepted_orders.extend()
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
        # TODO add to accepted_orders..
    return accepted_orders, meta


# 1. multi-stage market -> clears locally, rejected_bids are pushed up a layer
# 2. nodal pricing -> centralized market which handles different node_ids different - can also be used for country coupling
# 3. nodal limited market -> clear by node_id, select cheapest generation orders from surrounding area up to max_capacity, clear market
# 4. one sided market? - fixed demand as special case of two sided market
# 5.


available_clearing_strategies: dict[str, Callable] = {
    "pay_as_bid_all_or_nothing": pay_as_bid_aon,
    "pay_as_bid": pay_as_bid,
    "pay_as_clear_all_or_nothing": pay_as_clear_aon,
    "pay_as_clear": pay_as_clear,
    "nodal_pricing_pypsa_unflexible_demand": nodal_pricing_pypsa_unflexible_demand,
    # "nodal_market_pypsa": nodal_pricing_pypsa,
    "nodal_market_pyomo": nodal_pricing_pyomo,
}

if __name__ == "__main__":
    from datetime import datetime, timedelta

    from dateutil import rrule as rr
    from dateutil.relativedelta import relativedelta as rd

    from assume.common.utils import get_available_products

    simple_dayahead_auction_config = MarketConfig(
        "simple_dayahead_auction",
        market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2005, 6, 1),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        amount_unit="MW",
        amount_tick=0.1,
        price_unit="€/MW",
        market_mechanism="pay_as_clear",
    )

    mr = MarketRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    print(products)
    start = products[0][0]
    end = products[0][1]
    only_hours = products[0][2]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "only_hours": None,
        },
    ]
    simple_dayahead_auction_config.market_mechanism = available_clearing_strategies[
        simple_dayahead_auction_config.market_mechanism
    ]
    mr.all_orders = orderbook
    clearing_result, meta = simple_dayahead_auction_config.market_mechanism(
        mr, products
    )
    import pandas as pd

    print(pd.DataFrame.from_dict(mr.all_orders))
    print(pd.DataFrame.from_dict(clearing_result))
    print(meta)
