# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
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
from pyomo.opt import SolverFactory

from assume.common.market_objects import MarketConfig, MarketProduct, Order, Orderbook
from assume.common.utils import get_supported_solver_pyomo
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)

order_types = ["single_ask", "single_bid", "linked_ask", "exclusive_ask"]


class ComplexDmasClearingRole(MarketRole):
    required_fields = ["link", "block_id", "exclusive_id"]

    def __init__(
        self,
        marketconfig: MarketConfig,
        verbose: bool = False,
        elastic_demand: bool = False,
    ):
        super().__init__(marketconfig)
        self.elastic_demand = elastic_demand
        if not verbose:
            logger.setLevel(logging.WARNING)

    @staticmethod
    def _validate_product_alignment(
        market_products: list[MarketProduct], start, duration
    ) -> None:
        s = start
        for market_product in market_products[1:]:
            if market_product[0] != s + duration:
                raise ValueError("Market products of one clearing must align")
            s = market_product[0]

    def _empty_orderbook_results(
        self, start, duration, T: int
    ) -> tuple[Orderbook, Orderbook, list[dict], list]:
        min_p = self.marketconfig.minimum_bid_price or 0.0
        meta = [
            {
                "supply_volume": 0.0,
                "demand_volume": 0.0,
                "demand_volume_energy": 0.0,
                "supply_volume_energy": 0.0,
                "price": min_p,
                "max_price": min_p,
                "min_price": min_p,
                "node": None,
                "product_start": start + duration * t,
                "product_end": start + duration * (t + 1),
                "only_hours": None,
            }
            for t in range(T)
        ]
        return [], [], meta, []

    def _parse_orders(self, accepted: Orderbook, start, duration) -> dict:
        orders = {type_: {} for type_ in order_types}
        index_orders = {type_: defaultdict(list) for type_ in order_types}
        parent_blocks = {}
        start_block = []
        bid_ids = {}
        agent_addrs = {}
        unit_ids = {}

        for order in accepted:
            order_type = None
            if order["exclusive_id"] is not None:
                if order["block_id"] is None and order["link"] is None:
                    order_type = "exclusive_ask"
                else:
                    logger.error(f"received invalid order: {order=}")
            elif not (order["block_id"] is None or order["link"] is None):
                if order["exclusive_id"] is None:
                    order_type = "linked_ask"
                else:
                    logger.error(f"received invalid order: {order=}")
            else:
                if order["volume"] < 0:
                    order_type = "single_bid"
                elif order["volume"] > 0:
                    order_type = "single_ask"

            if order_type is not None:
                tt = (order["start_time"] - start) / duration
                name = f"{order['agent_addr']} {order.get('unit_id', '')}"
                if "exclusive" in order_type:
                    idx = (order["exclusive_id"], tt, name)
                elif "linked" in order_type:
                    idx = (order["block_id"], tt, name)
                else:
                    name += str(order["bid_id"])
                    idx = (None, tt, name)

                agent_addrs[name] = order["agent_addr"]
                bid_ids[name] = order["bid_id"]
                unit_ids[name] = order.get("unit_id", "")

                index_orders[order_type][tt].append((idx[0], idx[2]))

                if "linked" in order_type:
                    val = (order["price"], order["volume"], order["link"])
                else:
                    val = (order["price"], order["volume"])

                orders[order_type][idx] = val

        for key_tuple, val in orders["linked_ask"].items():
            block, _, agent = key_tuple
            _, _, parent_id = val
            child_key = (block, agent)
            parent_blocks[child_key] = parent_id
            if parent_id == -1:
                start_block.append((block, agent))

        return {
            "orders": orders,
            "index_orders": index_orders,
            "parent_blocks": parent_blocks,
            "start_block": set(start_block),
            "bid_ids": bid_ids,
            "agent_addrs": agent_addrs,
            "unit_ids": unit_ids,
        }

    def _build_model(
        self, parsed: dict, t_range: np.ndarray
    ) -> tuple[ConcreteModel, dict, list]:
        orders = parsed["orders"]
        index_orders = parsed["index_orders"]
        parent_blocks = parsed["parent_blocks"]
        start_block = parsed["start_block"]

        model = ConcreteModel("dmas_market")
        model_vars = {}

        # 1. Decision Variables
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

        if self.elastic_demand:
            model.use_single_bid = Var(
                set(
                    (block, hour, agent)
                    for block, hour, agent in orders["single_bid"].keys()
                ),
                within=Reals,
                bounds=(0, 1),
                initialize=0,
            )
            model_vars["single_bid"] = model.use_single_bid

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

        model.use_exclusive_block = Var(
            set([(block, agent) for block, _, agent in orders["exclusive_ask"].keys()]),
            within=Binary,
        )
        model_vars["exclusive_ask"] = model.use_exclusive_block

        model.sink = Var(t_range, within=NonNegativeReals)
        model.source = Var(t_range, within=NonNegativeReals)

        # 2. Linked constraints
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
                        len(parent_hours)
                        * quicksum(
                            model.use_linked_order[block, h, agent] for h in hours
                        )
                        <= len(hours)
                        * quicksum(
                            model.use_linked_order[parent_id, h, agent]
                            for h in parent_hours
                        )
                    )
                else:
                    logger.warning(
                        f"Agent {agent} sent invalid linked orders "
                        f"- block {block} has no parent_id {parent_id}"
                    )
            else:
                mother_bid_counter = len(hours)
                model.mother_bid.add(
                    quicksum(model.use_linked_order[block, h, agent] for h in hours)
                    == mother_bid_counter * model.use_mother_order[(block, agent)]
                )

        # 3. Exclusive block constraints
        model.one_exclusive_block = ConstraintList()
        for agent in {agent for _, _, agent in orders["exclusive_ask"].keys()}:
            model.one_exclusive_block.add(
                1 >= quicksum(model.use_exclusive_block[:, agent])
            )

        # 4. Supply/demand volumes and balance
        def get_volume(type_: str, hour: int):
            if type_ == "single_bid":
                if self.elastic_demand and "single_bid" in model_vars:
                    return quicksum(
                        orders[type_][block, hour, name][1]
                        * model_vars[type_][block, hour, name]
                        for block, name in index_orders[type_][hour]
                    )
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

        model.gen_dem = ConstraintList()
        for t in t_range:
            if not index_orders["single_bid"][t]:
                logger.error(f"no hourly_bids available at hour {t}")
            elif not (
                index_orders["single_ask"][t]
                or index_orders["linked_ask"][t]
                or index_orders["exclusive_ask"][t]
            ):
                logger.error(f"no hourly_asks available at hour {t}")
            else:
                model.gen_dem.add(magic_source[t] == model.source[t] - model.sink[t])

        # 5. Objective: Social welfare / cost minimization
        if self.elastic_demand and "single_bid" in model_vars:
            generation_cost = quicksum(
                quicksum(
                    get_cost(type_=order_type, hour=t)
                    for order_type in order_types
                    if "bid" not in order_type
                )
                + quicksum(
                    orders["single_bid"][block, t, name][0]
                    * orders["single_bid"][block, t, name][1]
                    * model.use_single_bid[block, t, name]
                    for block, name in index_orders["single_bid"][t]
                )
                + (model.source[t] + model.sink[t])
                * self.marketconfig.maximum_bid_price
                * 10
                for t in t_range
            )
        else:
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

        model.obj = Objective(expr=generation_cost, sense=minimize)
        return model, model_vars, magic_source

    def _solve_and_extract_prices(
        self,
        model: ConcreteModel,
        model_vars: dict,
        parsed: dict,
        t_range: np.ndarray,
    ) -> pd.DataFrame:
        orders = parsed["orders"]
        index_orders = parsed["index_orders"]

        opt = SolverFactory(get_supported_solver_pyomo())
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

        prices = []
        for t in t_range:
            max_price = self.marketconfig.minimum_bid_price
            for type_ in model_vars.keys():
                if type_ == "single_bid":
                    continue
                for block, name in index_orders[type_][t]:
                    if type_ == "exclusive_ask":
                        order_used = model_vars[type_][block, name].value
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

            prices.append(max_price)
        return pd.DataFrame(data=dict(price=prices))

    def _reconstruct_orders(
        self,
        model: ConcreteModel,
        model_vars: dict,
        parsed: dict,
        prices: pd.DataFrame,
        magic_source: list,
        start,
        duration,
        t_range: np.ndarray,
    ) -> tuple[Orderbook, Orderbook, list[dict], list]:
        orders = parsed["orders"]
        index_orders = parsed["index_orders"]
        agent_addrs = parsed["agent_addrs"]
        bid_ids = parsed["bid_ids"]
        unit_ids = parsed["unit_ids"]

        volumes = []
        sum_magic_source = 0
        for t in t_range:
            sum_magic_source += get_real_number(magic_source[t])
            volume = 0
            for block, name in index_orders["single_bid"][t]:
                if self.elastic_demand and "single_bid" in model_vars:
                    u_bid = (
                        model.use_single_bid[block, t, name].value
                        if (block, t, name) in model.use_single_bid
                        else 0
                    ) or 0
                    volume += (-1) * orders["single_bid"][block, t, name][1] * u_bid
                else:
                    volume += (-1) * orders["single_bid"][block, t, name][1]
            for block, name in index_orders["exclusive_ask"][t]:
                if (
                    model.use_exclusive_block[block, name].value
                    and orders["exclusive_ask"][block, t, name][1] < 0
                ):
                    volume += (-1) * orders["exclusive_ask"][block, t, name][1]
            volumes.append(volume)
        logger.info(f"Got {sum_magic_source:.2f} kWh from Magic source")

        accepted = []
        rejected = []
        for t in t_range:
            t = int(t)
            bstart = start + duration * t
            end = start + duration * (t + 1)
            clear_price = prices["price"][t]
            for type_ in model_vars.keys():
                if type_ == "single_bid":
                    continue
                for block, name in index_orders[type_][t]:
                    if type_ in ["single_ask", "linked_ask"]:
                        usage = model_vars[type_][block, t, name].value or 0
                        link = None
                        if "linked" in type_:
                            prc, vol, link = orders[type_][block, t, name]
                        else:
                            prc, vol = orders[type_][block, t, name]
                        accepted_prc = (
                            clear_price
                            if self.marketconfig.market_mechanism == "pay_as_clear"
                            else prc
                        )
                        o: Order = {
                            "start_time": bstart,
                            "end_time": end,
                            "only_hours": None,
                            "price": prc,
                            "volume": vol,
                            "accepted_price": accepted_prc,
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
                        usage = model_vars[type_][block, name].value or 0
                        prc, vol = orders[type_][block, t, name]
                        accepted_prc = (
                            clear_price
                            if self.marketconfig.market_mechanism == "pay_as_clear"
                            else prc
                        )
                        o: Order = {
                            "start_time": bstart,
                            "end_time": end,
                            "only_hours": None,
                            "price": prc,
                            "volume": vol,
                            "accepted_price": accepted_prc,
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
            orig_price, vol = val
            prc = prices["price"][hour]
            bstart = start + duration * hour
            end = start + duration * (hour + 1)
            if self.elastic_demand and "single_bid" in model_vars:
                usage = (
                    model.use_single_bid[key].value
                    if key in model.use_single_bid
                    else 0
                ) or 0
            else:
                usage = 1.0
            accepted_prc = (
                prc
                if self.marketconfig.market_mechanism == "pay_as_clear"
                else orig_price
            )
            o: Order = {
                "start_time": bstart,
                "end_time": end,
                "only_hours": None,
                "price": orig_price,
                "volume": vol,
                "accepted_price": accepted_prc,
                "accepted_volume": vol * usage,
                "block_id": None,
                "link": None,
                "exclusive_id": None,
                "agent_addr": agent_addrs[name],
                "bid_id": bid_ids[name],
                "unit_id": unit_ids[name],
            }
            if usage > 0:
                accepted.append(o)
            else:
                rejected.append(o)

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

        flows = []
        return accepted, rejected, meta, flows

    def clear(
        self, accepted: Orderbook, market_products: list[MarketProduct]
    ) -> tuple[Orderbook, Orderbook, list[dict], list]:
        """Clear market orders against products adhering to DMAS rules.

        Incoming orders are matched and allocations are determined using
        a MILP formulation with linked blocks and exclusive order groups.
        """
        if not market_products:
            return [], [], [], []

        start = market_products[0][0]
        duration = market_products[0][1] - start
        self._validate_product_alignment(market_products, start, duration)

        T = len(market_products)
        t_range = np.arange(T)

        if not accepted:
            return self._empty_orderbook_results(start, duration, T)

        parsed = self._parse_orders(accepted, start, duration)
        model, model_vars, magic_source = self._build_model(parsed, t_range)
        prices = self._solve_and_extract_prices(model, model_vars, parsed, t_range)
        return self._reconstruct_orders(
            model, model_vars, parsed, prices, magic_source, start, duration, t_range
        )
