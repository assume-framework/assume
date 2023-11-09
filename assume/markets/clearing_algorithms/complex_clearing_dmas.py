# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import time
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Reals,
    SolverFactory,
    Var,
    minimize,
    quicksum,
)
from pyomo.environ import value as get_real_number
from pyomo.opt import SolverFactory, check_available_solvers

from assume.common.market_objects import MarketConfig, MarketProduct, Order, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]

order_types = ["single_ask", "single_bid", "linked_ask", "exclusive_ask"]


class ComplexDmasClearingRole(MarketRole):
    required_fields = ["link", "block_id", "exclusive_id"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        This performs the process of "market clearing" for a given market agent and its orders.
        During this process, incoming orders are matched against each other and allocations are determined to adhere to market rules.
        The result are an orderbook, the rejected orders and market metadata.

        :param market_agent: The market agent
        :type market_agent: MarketRole
        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]
        :return: orderbook, rejected_orders, meta
        :rtype: tuple[Orderbook, Orderbook, list[dict]]
        """
        # INIT
        start = market_products[0][0]
        T = len(market_products)
        t_range = np.arange(T)
        # Orders have (block, hour, name) as key and (price, volume, link) as values
        orders = {type_: {} for type_ in order_types}
        # Index Orders have t as key and (block, name) as value
        index_orders = {type_: defaultdict(list) for type_ in order_types}
        model_vars = {}
        parent_blocks = {}
        start_block = []
        model = ConcreteModel("dmas_market")
        # Create a solver
        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")
        opt = SolverFactory(solvers[0])

        bid_ids = {}

        for order in orderbook:
            order_type = None
            if order["exclusive_id"] is not None:
                if order["block_id"] is None and order["link"] is None:
                    order_type = "exclusive_ask"
                else:
                    order_type = None
                    log.error(f"received invalid order: {order=}")
            elif not (order["block_id"] is None or order["link"] is None):
                if order["exclusive_id"] is None:
                    order_type = "linked_ask"
                else:
                    order_type = None
                    log.error(f"received invalid order: {order=}")
            else:
                if order["volume"] < 0:
                    order_type = "single_bid"
                elif order["volume"] > 0:
                    order_type = "single_ask"
                else:
                    order_type = None
            if order_type is not None:
                tt = (order["start_time"] - start) / timedelta(hours=1)
                # block_id, hour, name
                name = order["agent_id"]
                bid_ids[name] = order["bid_id"]
                if "exclusive" in order_type:
                    idx = (order["exclusive_id"], tt, name)
                elif "linked" in order_type:
                    idx = (order["block_id"], tt, name)
                else:
                    idx = (order["bid_id"], tt, name)

                index_orders[order_type][tt].append((idx[0], idx[2]))

                if "linked" in order_type:
                    val = order["price"], order["volume"], order["link"]
                else:
                    val = order["price"], order["volume"]

                orders[order_type][idx] = val

        ################ set parameter ################

        for key_tuple, val in orders["linked_ask"].items():
            block, _, agent = key_tuple
            _, _, parent_id = val
            child_key = (block, agent)
            parent_blocks[child_key] = parent_id
            if parent_id == -1:
                start_block.append((block, agent))

        start_block = set(start_block)

        # optimize
        model.clear()
        log.info("start building model")
        t1 = time.time()
        # Step 1 initialize binary variables for hourly ask block per agent and id
        model.use_hourly_ask = Var(
            set(
                (block, hour, agent)
                for block, hour, agent in orders["single_ask"].keys()
            ),
            within=Reals,
            bounds=(0, 1),
        )
        model_vars["single_ask"] = model.use_hourly_ask
        # Step 3 initialize binary variables for ask order in block per agent
        model.use_linked_order = Var(
            set(
                [
                    (block, hour, agent)
                    for block, hour, agent in orders["linked_ask"].keys()
                ]
            ),
            within=Reals,
            bounds=(0, 1),
        )
        model_vars["linked_ask"] = model.use_linked_order

        model.use_mother_order = Var(start_block, within=Binary)

        # Step 4 initialize binary variables for exclusive block and agent
        model.use_exclusive_block = Var(
            set([(block, agent) for block, _, agent in orders["exclusive_ask"].keys()]),
            within=Binary,
        )
        model_vars["exclusive_ask"] = model.use_exclusive_block

        model.sink = Var(t_range, within=NonNegativeReals)
        model.source = Var(t_range, within=NonNegativeReals)

        # Step 6 set constraint: If parent block of an agent is used -> enable usage of child block
        model.enable_child_block = ConstraintList()
        model.mother_bid = ConstraintList()
        orders_local = defaultdict(lambda: [])
        for block, hour, agent in orders["linked_ask"].keys():
            orders_local[(block, agent)].append(hour)

        for order, hours in orders_local.items():
            block, agent = order
            parent_id = parent_blocks[block, agent]
            if parent_id != -1:
                if (parent_id, agent) in orders_local.keys():
                    parent_hours = orders_local[(parent_id, agent)]
                    model.enable_child_block.add(
                        quicksum(model.use_linked_order[block, h, agent] for h in hours)
                        <= 2
                        * quicksum(
                            model.use_linked_order[parent_id, h, agent]
                            for h in parent_hours
                        )
                    )
                else:
                    log.warning(
                        f"Agent {agent} send invalid linked orders "
                        f"- block {block} has no parent_id {parent_id}"
                    )
                    log.warning("Block, Hour, Agent, Price, Volume, Link")
                    for key, data in orders["linked_ask"].items():
                        if key[2] == agent:
                            log.warning(
                                f"{key[0], key[1], key[2], data[0], data[1], data[2]}"
                            )
            else:
                # mother bid must exist with at least one entry
                # either the whole mother bid can be used | None
                mother_bid_counter = len(hours)
                model.mother_bid.add(
                    quicksum(model.use_linked_order[block, h, agent] for h in hours)
                    == mother_bid_counter * model.use_mother_order[(block, agent)]
                )

        # Constraints for exclusive block orders
        # ------------------------------------------------
        # Step 7 set constraint: only one scheduling can be used
        model.one_exclusive_block = ConstraintList()
        for data in set([(agent,) for _, _, agent in orders["exclusive_ask"].keys()]):
            agent = data
            model.one_exclusive_block.add(
                1 >= quicksum(model.use_exclusive_block[:, agent])
            )

        def get_volume(type_: str, hour: int):
            if type_ == "single_bid":
                return quicksum(
                    orders[type_][block, hour, name][1]
                    for block, name in index_orders[type_][hour]
                )
            elif type_ == "exclusive_ask":
                return quicksum(
                    orders[type_][block, hour, name][1] * model_vars[type_][block, name]
                    for block, name in index_orders[type_][hour]
                )
            else:
                return quicksum(
                    orders[type_][block, hour, name][1]
                    * model_vars[type_][block, hour, name]
                    for block, name in index_orders[type_][hour]
                )

        def get_cost(type_: str, hour: int):
            if type_ == "single_bid":
                return quicksum(
                    orders[type_][block, hour, name][0]
                    * orders[type_][block, hour, name][1]
                    for block, name in index_orders[type_][hour]
                )
            elif type_ == "exclusive_ask":
                return quicksum(
                    orders[type_][block, hour, name][0]
                    * orders[type_][block, hour, name][1]
                    * model_vars[type_][block, name]
                    for block, name in index_orders[type_][hour]
                    if orders[type_][block, hour, name][1] > 0
                )
            else:
                return quicksum(
                    orders[type_][block, hour, name][0]
                    * orders[type_][block, hour, name][1]
                    * model_vars[type_][block, hour, name]
                    for block, name in index_orders[type_][hour]
                )

        magic_source = [
            -1
            * quicksum(
                get_volume(type_=order_type, hour=t) for order_type in order_types
            )
            for t in t_range
        ]

        # generation must be smaller than demand
        model.gen_dem = ConstraintList()
        for t in t_range:
            if not index_orders["single_bid"][t]:
                log.error(f"no hourly_bids available at hour {t}")
            elif not (index_orders["single_ask"][t] or index_orders["linked_ask"][t]):
                # constraints with 0 <= 0 are not valid
                log.error(f"no hourly_asks available at hour {t}")
            else:
                model.gen_dem.add(magic_source[t] == model.source[t] - model.sink[t])

        # Step 9 set constraint: Cost for each hour
        generation_cost = quicksum(
            quicksum(
                get_cost(type_=order_type, hour=t)
                for order_type in order_types
                if "bid" not in order_type
            )
            + (model.source[t] + model.sink[t]) * 1e12
            for t in t_range
        )

        model.obj = Objective(expr=generation_cost, sense=minimize)
        log.info(f"built model in {time.time() - t1:.2f} seconds")
        log.info("start optimization/market clearing")
        t1 = time.time()
        try:
            if opt.name == "gurobi":
                options = {"MIPGap": 0.1, "TimeLimit": 60}
            else:
                options = {}
            r = opt.solve(model, options=options)
            log.info(r)
        except Exception as e:
            log.exception("error solving optimization problem")
            log.error(f"Model: {model}")
            log.error(f"{repr(e)}")
        log.info(f"cleared market in {time.time() - t1:.2f} seconds")

        ################ convert internal bids to orderbook ################

        # -> determine price at each hour
        prices = []
        for t in t_range:
            max_price = -1000
            for type_ in model_vars.keys():
                for block, name in index_orders[type_][t]:
                    if type_ == "exclusive_ask":
                        order_used = model_vars[type_][block, name].value
                        # -> disable price by storage if the storage is on the demand side
                        if order_used and orders[type_][block, t, name][1] > 0:
                            order_used = True
                        else:
                            order_used = False
                    else:
                        order_used = model_vars[type_][block, t, name].value
                    if order_used:
                        price = orders[type_][block, t, name][0]
                    else:
                        price = -1000
                    if price > max_price:
                        max_price = price
            prices += [max_price]
        prices = pd.DataFrame(data=dict(price=prices))
        # -> determine volume at each hour
        volumes = []
        sum_magic_source = 0
        for t in t_range:
            sum_magic_source += get_real_number(magic_source[t])
            volume = 0
            for block, name in index_orders["single_bid"][t]:
                volume += (-1) * orders["single_bid"][block, t, name][1]
            for block, name in index_orders["exclusive_ask"][t]:
                if (
                    model.use_exclusive_block[block, name].value
                    and orders["exclusive_ask"][block, t, name][1] < 0
                ):
                    volume += (-1) * orders["exclusive_ask"][block, t, name][1]
            volumes.append(volume)
        log.info(f"Got {sum_magic_source:.2f} kWh from Magic source")
        # -> determine used ask orders

        used_orders = {type_: {} for type_ in model_vars.keys()}

        orderbook = []
        for t in t_range:
            t = int(t)
            bstart = start + timedelta(hours=t)
            end = start + timedelta(hours=t + 1)
            for type_ in model_vars.keys():
                for block, name in index_orders[type_][t]:
                    if type_ in ["single_ask", "linked_ask"]:
                        if model_vars[type_][block, t, name].value:
                            f = model_vars[type_][block, t, name].value
                            link = None
                            if "linked" in type_:
                                prc, vol, link = orders[type_][block, t, name]
                                vol *= f
                                p = (prc, vol, link)
                            else:
                                prc, vol = orders[type_][block, t, name]
                                vol *= f
                                p = (prc, vol)
                            used_orders[type_][(block, t, name)] = p
                            o: Order = {
                                "start_time": bstart,
                                "end_time": end,
                                "only_hours": None,
                                "price": prc,
                                "volume": vol,
                                "block_id": block,
                                "link": link,
                                "exclusive_id": None,
                                "agent_id": name,
                                "bid_id": bid_ids[name],
                            }
                            orderbook.append(o)

                    elif type_ == "exclusive_ask":
                        if model_vars[type_][block, name].value:
                            prc, vol = orders[type_][block, t, name]
                            used_orders[type_][(block, t, name)] = (prc, vol)
                            o: Order = {
                                "start_time": bstart,
                                "end_time": end,
                                "only_hours": None,
                                "price": prc,
                                "volume": vol,
                                "block_id": None,
                                "link": None,
                                "exclusive_id": block,
                                "agent_id": name,
                                "bid_id": bid_ids[name],
                            }
                            orderbook.append(o)

        for key, val in orders["single_bid"].items():
            block, hour, name = key
            prc, vol = val
            bstart = start + timedelta(hours=hour)
            end = start + timedelta(hours=hour + 1)
            orderbook.append(
                {
                    "start_time": bstart,
                    "end_time": end,
                    "only_hours": None,
                    "price": prc,
                    "volume": vol,
                    "block_id": None,
                    "link": None,
                    "exclusive_id": None,
                    "agent_id": name,
                    "bid_id": bid_ids[name],
                }
            )

        # -> build dataframe
        for type_ in model_vars.keys():
            orders_df = pd.DataFrame.from_dict(used_orders[type_], orient="index")
            orders_df.index = pd.MultiIndex.from_tuples(
                orders_df.index, names=["block_id", "hour", "name"]
            )

            if "linked" in type_ and orders_df.empty:
                orders_df["price"] = []
                orders_df["volume"] = []
                orders_df["link"] = []
            elif orders_df.empty:
                orders_df["price"] = []
                orders_df["volume"] = []
            elif "linked" in type_:
                orders_df.columns = ["price", "volume", "link"]
            else:
                orders_df.columns = ["price", "volume"]

            used_orders[type_] = orders_df.copy()
        # -> return all bid orders
        used_bid_orders = pd.DataFrame.from_dict(orders["single_bid"], orient="index")
        used_bid_orders.index = pd.MultiIndex.from_tuples(
            used_bid_orders.index, names=["block_id", "hour", "name"]
        )
        if used_bid_orders.empty:
            used_bid_orders["price"] = []
            used_bid_orders["volume"] = []
        else:
            used_bid_orders.columns = ["price", "volume"]

        prices["volume"] = volumes
        prices["magic_source"] = [get_real_number(m) for m in magic_source]

        # -> build merit order
        merit_order = {hour: dict(price=[], volume=[], type=[]) for hour in t_range}

        def add_to_merit_order(hour, price, volume, type_):
            pass
            # merit_order[hour]["price"].append(price)
            # merit_order[hour]["volume"].append(volume)
            # merit_order[hour]["type"].append(type_)

        for index, values in orders["linked_ask"].items():
            price, volume, _ = values
            _, hour, _ = index
            add_to_merit_order(hour, price, volume, "ask")
        for index, values in orders["exclusive_ask"].items():
            price, volume = values
            _, hour, _ = index
            if volume > 0:
                add_to_merit_order(hour, price, volume, "ask")
            else:
                add_to_merit_order(hour, price, -volume, "bid")
        for index, values in orders["single_ask"].items():
            price, volume = values
            _, hour, _ = index
            add_to_merit_order(hour, price, volume, "ask")
        for index, values in orders["single_bid"].items():
            price, volume = values
            _, hour, _ = index
            add_to_merit_order(hour, price, -volume, "bid")
        rejected = []
        meta = []
        for t in t_range:
            t = int(t)
            bstart = start + timedelta(hours=t)
            end = start + timedelta(hours=t + 1)
            prc = prices["price"][t]
            meta.append(
                {
                    "supply_volume": volumes[t],
                    "demand_volume": volumes[t],
                    "demand_volume_energy": volumes[t],
                    "supply_volume_energy": volumes[t],
                    "price": prc,
                    "max_price": prc,
                    "min_price": prc,
                    "node_id": None,
                    "product_start": bstart,
                    "product_end": end,
                    "only_hours": None,
                }
            )
        return orderbook, rejected, meta
