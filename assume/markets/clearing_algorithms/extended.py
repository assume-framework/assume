# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketConfig, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


class NodalPricingInflexDemandRole(MarketRole):
    required_fields = ["node_id"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> (Orderbook, Orderbook, list[dict]):
        import pypsa

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
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
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

        return accepted_orders, rejected_orders, meta
