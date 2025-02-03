# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Reals,
    Var,
    minimize,
    quicksum,
)
from pyomo.environ import value as get_real_number
from pyomo.opt import SolverFactory, check_available_solvers

from assume.common.market_objects import MarketConfig, MarketProduct, Order, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

order_types = ["single_ask", "single_bid", "linked_ask", "exclusive_ask"]


class ComplexDmasClearingRole(MarketRole):
    required_fields = ["link", "block_id", "exclusive_id"]

    def __init__(self, marketconfig: MarketConfig, verbose: bool = False):
        super().__init__(marketconfig)
        if not verbose:
            logger.setLevel(logging.WARNING)

    def clear(
        self, accepted: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        This performs the process of "market clearing" for a given market agent and its orders.
        During this process, incoming orders are matched against each other and allocations are determined to adhere to market rules.
        Linked orders are respected to be only taken if the prior block was also taken.
        Orders with the same exclusive ID from the same agent can only be taken together, in which case no other exclusive blocks can be taken.
        The result are an orderbook, the rejected orders and market metadata.

        Limitations:
            * The clearing is currently one-sided, in the way that the price of the demand is not respected
            * A per-agent-unique bid_id is required for standard bids
            * The cost of taking additional blocks is taken individually instead (like pay-as-bid) instead of applying uniform pricing during the calculation

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple[Orderbook, Orderbook, list[dict]]: accepted orderbook, rejected orderbook and clearing meta data
        """
        # assumes same duration for all given products
        start = market_products[0][0]
        duration = market_products[0][1] - start
        s = start
        for market_product in market_products[1:]:
            if market_product[0] != s + duration:
                raise ValueError("Market products of one clearing must align")
            s = market_product[0]

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
        agent_addrs = {}
        unit_ids = {}

        for order in accepted:
            order_type = None
            if order["exclusive_id"] is not None:
                if order["block_id"] is None and order["link"] is None:
                    order_type = "exclusive_ask"
                else:
                    order_type = None
                    logger.error(f"received invalid order: {order=}")
            elif not (order["block_id"] is None or order["link"] is None):
                if order["exclusive_id"] is None:
                    order_type = "linked_ask"
                else:
                    order_type = None
                    logger.error(f"received invalid order: {order=}")
            else:
                if order["volume"] < 0:
                    order_type = "single_bid"
                elif order["volume"] > 0:
                    order_type = "single_ask"
                else:
                    order_type = None
            if order_type is not None:
                tt = (order["start_time"] - start) / duration
                # block_id, hour, name
                name = f'{order["agent_addr"]} {order.get("unit_id", "")}'
                if "exclusive" in order_type:
                    idx = (order["exclusive_id"], tt, name)
                elif "linked" in order_type:
                    idx = (order["block_id"], tt, name)
                else:
                    # needs bid_id to distinguish orders in the set
                    name += str(order["bid_id"])
                    idx = (None, tt, name)
                agent_addrs[name] = order["agent_addr"]
                bid_ids[name] = order["bid_id"]
                unit_ids[name] = order.get("unit_id", "")

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
        logger.info("start building model")
        t1 = time.time()
        # Step 1 initialize binary variables for hourly ask block per agent and id
        model.use_hourly_ask = Var(
            set(
                (block, hour, agent)
                for block, hour, agent in orders["single_ask"].keys()
            ),
            within=Reals,
            bounds=(0, 1),
            initialize=0,
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
        orders_local = defaultdict(list)
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
                        <= 100  # this factor is arbitrary and means that if we took at least 0.01 from the linked block, we can use our full block
                        * quicksum(
                            model.use_linked_order[parent_id, h, agent]
                            for h in parent_hours
                        )
                    )
                else:
                    logger.warning(
                        f"Agent {agent} send invalid linked orders "
                        f"- block {block} has no parent_id {parent_id}"
                    )
                    logger.warning("Block, Hour, Agent, Price, Volume, Link")
                    for key, data in orders["linked_ask"].items():
                        if key[2] == agent:
                            logger.warning(
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
                # TODO actually for linked order in the same hour,
                # the maximum price of all its prior required blocks
                # should be used to determine the cost of the additional block
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

        # generation +- magic_source must match demand
        model.gen_dem = ConstraintList()
        for t in t_range:
            if not index_orders["single_bid"][t]:
                logger.error(f"no hourly_bids available at hour {t}")
            elif not (index_orders["single_ask"][t] or index_orders["linked_ask"][t]):
                # constraints with 0 <= 0 are not valid
                logger.error(f"no hourly_asks available at hour {t}")
            else:
                model.gen_dem.add(magic_source[t] == model.source[t] - model.sink[t])

        # Step 9 set constraint: Cost for each hour
        # add magic_cost as very expensive, to overbid bids with marketconfig.maximum_bid_price
        generation_cost = quicksum(
            quicksum(
                get_cost(type_=order_type, hour=t)
                for order_type in order_types
                if "bid" not in order_type
            )
            + (model.source[t] + model.sink[t])
            * self.marketconfig.maximum_bid_price
            * 10
            for t in t_range
        )
        # TODO currently, this does not represent a two-sided clearing, as demand has to be taken
        # and is magically filled if not

        model.obj = Objective(expr=generation_cost, sense=minimize)
        logger.info(f"built model in {time.time() - t1:.2f} seconds")
        logger.info("start optimization/market clearing")
        t1 = time.time()
        try:
            if hasattr(opt, "name") and opt.name == "gurobi":
                options = {"MIPGap": 0.1, "TimeLimit": 60}
            else:
                options = {}
            r = opt.solve(model, options=options)
            logger.info(r)
        except Exception as e:
            logger.exception("error solving optimization problem")
            logger.error(f"Model: {model}")
            logger.error(f"{repr(e)}")
        logger.info(f"cleared market in {time.time() - t1:.2f} seconds")

        ################ convert internal bids to orderbook ################

        # -> determine price at each hour
        prices = []
        for t in t_range:
            max_price = self.marketconfig.minimum_bid_price
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
                        if price > max_price:
                            max_price = price

            prices += [max_price]
        prices = pd.DataFrame(data=dict(price=prices))

        # check volume in price in
        # orders["single_ask"]
        # {k: v.value for k, v in list(model_vars["single_ask"].items())}
        # watch order

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
        logger.info(f"Got {sum_magic_source:.2f} kWh from Magic source")
        # -> determine used ask orders

        accepted = []
        rejected = []
        for t in t_range:
            t = int(t)
            bstart = start + duration * t
            end = start + duration * (t + 1)
            clear_price = prices["price"][t]
            for type_ in model_vars.keys():
                for block, name in index_orders[type_][t]:
                    if type_ in ["single_ask", "linked_ask"]:
                        # usage from 0 to 1
                        usage = model_vars[type_][block, t, name].value or 0
                        link = None
                        if "linked" in type_:
                            prc, vol, link = orders[type_][block, t, name]
                        else:
                            prc, vol = orders[type_][block, t, name]
                        o: Order = {
                            "start_time": bstart,
                            "end_time": end,
                            "only_hours": None,
                            "price": prc,
                            "volume": vol,
                            "accepted_price": clear_price,
                            "accepted_volume": vol * usage,
                            "block_id": block,
                            "link": link,
                            "exclusive_id": None,
                            "agent_addr": agent_addrs[name],
                            "bid_id": bid_ids[name],
                            "unit_id": unit_ids[name],
                        }
                        if usage > 0:
                            accepted.append(o)
                        else:
                            rejected.append(o)

                    elif type_ == "exclusive_ask":
                        # usage from 0 to 1
                        usage = model_vars[type_][block, t, name].value or 0

                        prc, vol = orders[type_][block, t, name]
                        o: Order = {
                            "start_time": bstart,
                            "end_time": end,
                            "only_hours": None,
                            "price": prc,
                            "volume": vol,
                            "accepted_price": prc,
                            "accepted_volume": vol * usage,
                            "block_id": None,
                            "link": None,
                            "exclusive_id": block,
                            "agent_addr": agent_addrs[name],
                            "bid_id": bid_ids[name],
                            "unit_id": unit_ids[name],
                        }
                        if usage > 0:
                            accepted.append(o)
                        else:
                            rejected.append(o)

        for key, val in orders["single_bid"].items():
            block, hour, name = key
            _, vol = val
            prc = prices["price"][hour]
            bstart = start + duration * hour
            end = start + duration * (hour + 1)
            accepted.append(
                {
                    "start_time": bstart,
                    "end_time": end,
                    "only_hours": None,
                    "price": prc,
                    "volume": vol,
                    "accepted_price": prc,
                    "accepted_volume": vol,
                    "block_id": None,
                    "link": None,
                    "exclusive_id": None,
                    "agent_addr": agent_addrs[name],
                    "bid_id": bid_ids[name],
                    "unit_id": unit_ids[name],
                }
            )

        prices["volume"] = volumes
        prices["magic_source"] = [get_real_number(m) for m in magic_source]

        meta = []
        for t in t_range:
            t = int(t)
            bstart = start + duration * t
            end = start + duration * (t + 1)
            prc = prices["price"][t]
            supply = volumes[t] - prices["magic_source"][t]
            meta.append(
                {
                    "supply_volume": supply,
                    "demand_volume": volumes[t],
                    "demand_volume_energy": volumes[t],
                    "supply_volume_energy": supply,
                    "price": prc,
                    "max_price": prc,
                    "min_price": prc,
                    "node": None,
                    "product_start": bstart,
                    "product_end": end,
                    "only_hours": None,
                }
            )

        # write network flows here if applicable
        flows = []

        return accepted, rejected, meta, flows
