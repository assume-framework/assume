# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from operator import itemgetter

import numpy as np
import pandas as pd
import pypsa
from mango import AgentAddress

from assume.common.grid_utils import read_pypsa_grid
from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import create_incidence_matrix
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)

# Set the log level to WARNING
logging.getLogger("linopy").setLevel(logging.WARNING)
logging.getLogger("pypsa").setLevel(logging.WARNING)


def calculate_meta(accepted_demand_orders, accepted_supply_orders, product):
    supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
    demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
    prices = list(map(itemgetter("accepted_price"), accepted_supply_orders)) or [0]

    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    avg_price = 0
    if supply_volume:
        weighted_price = [
            order["accepted_volume"] * order["accepted_price"]
            for order in accepted_supply_orders
        ]
        avg_price = sum(weighted_price) / supply_volume
    return {
        "supply_volume": supply_volume,
        "demand_volume": demand_volume,
        "demand_volume_energy": demand_volume * duration_hours,
        "supply_volume_energy": supply_volume * duration_hours,
        "price": avg_price,
        "max_price": max(prices),
        "min_price": min(prices),
        "node": None,
        "product_start": product[0],
        "product_end": product[1],
        "only_hours": product[2],
    }


class NodalClearingRole(MarketRole):
    """
    This class implements a nodal market clearing mechanism using a linear optimal power flow (OPF) approach.
    """

    required_fields = ["node", "max_power"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()

        if not self.grid_data:
            logger.error(f"Market '{marketconfig.market_id}': grid_data is missing.")
            raise ValueError("grid_data is missing.")

        # Define grid data
        self.nodes = ["node0"]
        self.incidence_matrix = None

        self.lines = self.grid_data["lines"]
        buses = self.grid_data["buses"]

        self.zones_id = self.marketconfig.param_dict.get("zones_identifier")
        self.node_to_zone = None

        # Generate the incidence matrix and set the nodes based on zones or individual buses
        if self.zones_id:
            # Zonal Case
            self.incidence_matrix = create_incidence_matrix(
                self.lines, buses, zones_id=self.zones_id
            )
            self.nodes = buses[self.zones_id].unique()
            self.node_to_zone = buses[self.zones_id].to_dict()
        else:
            # Nodal Case
            self.incidence_matrix = create_incidence_matrix(self.lines, buses)
            self.nodes = buses.index.values

        self.log_flows = self.marketconfig.param_dict.get("log_flows", False)
        self.pricing_mechanism = self.marketconfig.param_dict.get(
            "pricing_mechanism", "pay_as_clear"
        )
        if self.pricing_mechanism not in ["pay_as_bid", "pay_as_clear"]:
            logger.error(
                f"Market '{marketconfig.market_id}': Invalid payment mechanism '{self.pricing_mechanism}'."
            )
            raise ValueError("Invalid payment mechanism.")

        # if we have multiple hours (count >1), we cannot handle storage units bids yet
        # this is because the storage bids would be linked bids
        storage_units = self.grid_data.get("storage_units", pd.DataFrame())
        if not storage_units.empty:
            if self.marketconfig.market_products[0].count > 1:
                # make sure storages potentially present in the grid do not participate in this market
                if not (
                    self.grid_data["storage_units"][
                        f"bidding_{self.marketconfig.market_id}"
                    ].isin(["-", ""])
                    | self.grid_data["storage_units"][
                        f"bidding_{self.marketconfig.market_id}"
                    ].isna()
                ).all():
                    logger.error(
                        f"Market '{marketconfig.market_id}': Nodal clearing with multiple product counts does not support storage unit bids yet."
                    )
                    raise NotImplementedError(
                        "Nodal clearing with multiple product counts does not support storage unit bids yet."
                    )

        read_pypsa_grid(
            network=self.network,
            grid_dict=self.grid_data,
        )

        # add all units to the PyPSA network as generators with p_nom their absolute max power
        # generators have p_min_pu - p_max_pu 0 to 1
        self.network.add(
            "Generator",
            self.grid_data["generators"].index,
            bus=self.grid_data["generators"]["node"],
            p_nom=self.grid_data["generators"]["max_power"],
            p_min_pu=0,
            p_max_pu=1,
        )
        # demand units have p_min_pu - p_max_pu -1 to 0
        self.network.add(
            "Generator",
            self.grid_data["loads"].index,
            bus=self.grid_data["loads"]["node"],
            p_nom=self.grid_data["loads"]["max_power"],
            p_min_pu=-1,
            p_max_pu=0,
        )
        # storage units
        # also add them as generators, as we only regard bids here and are not interested in their internal state
        # we take the max of discharging and charging power as p_nom for PyPSA. Bids are later used to set p_min_pu and p_max_pu accordingly.
        if not storage_units.empty:
            self.network.add(
                "Generator",
                self.grid_data["storage_units"].index,
                bus=self.grid_data["storage_units"]["node"],
                p_nom=np.maximum(
                    self.grid_data["storage_units"]["max_power_discharge"].values,
                    self.grid_data["storage_units"]["max_power_charge"].values,
                ),
                p_min_pu=-1,
                p_max_pu=1,
            )

        self.solver = marketconfig.param_dict.get("solver", "highs")
        if self.solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif self.solver == "highs":
            self.solver_options = {"output_flag": False, "log_to_console": False}
        else:
            self.solver_options = {}

    def validate_orderbook(
        self, orderbook: Orderbook, agent_addr: AgentAddress
    ) -> None:
        """
        Checks whether the bid types are valid and whether the volumes are within the maximum bid volume.

        Args:
            orderbook (Orderbook): The orderbook to be validated.
            agent_addr (AgentAddress): The agent address of the market.

        Raises:
            ValueError: If the bid type is invalid.
        """
        market_id = self.marketconfig.market_id

        for order in orderbook:
            # if bid_type is None, set to default bid_type
            if order.get("bid_type") is None:
                order["bid_type"] = "SB"
            # Validate bid_type
            elif order["bid_type"] in ["BB", "LB"]:
                raise ValueError(
                    f"Market '{market_id}': Invalid bid_type '{order['bid_type']}' in order {order}. Nodal clearing nly supports 'SB' bid type. Use 'complex_clearing' for BB and LB bid types."
                )
        # validate prices and volumes using base market role validation
        super().validate_orderbook(orderbook, agent_addr)

        for order in orderbook:
            # Node validation
            node = order.get("node")
            if node:
                if self.zones_id:
                    node = self.node_to_zone.get(node, self.nodes[0])
                    order["node"] = node
                if node not in self.nodes:
                    logger.warning(
                        f"Market '{market_id}': Node '{node}' not in nodes list {self.nodes}. Setting to first node '{self.nodes[0]}'. Order details: {order}"
                    )
                    order["node"] = self.nodes[0]
            else:
                if self.incidence_matrix is not None:
                    logger.warning(
                        f"Market '{market_id}': Order without a node, setting node to the first node '{self.nodes[0]}'. Please check the bidding strategy if correct node is set. Order details: {order}"
                    )
                    order["node"] = self.nodes[0]
                else:
                    logger.warning(
                        f"Market '{market_id}': Order without a node and no incidence matrix, setting node to 'node0'. Order details: {order}"
                    )
                    order["node"] = "node0"

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict], dict[tuple, float]]:
        """
        Performs nodal clearing based on optimal linear power flow.
        The returned orderbook contains accepted orders with the accepted volumes and prices.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products for which clearing happens.

        Returns:
            Tuple[Orderbook, Orderbook, List[dict]]: The accepted orderbook, rejected orderbook and market metadata.
        """

        if len(orderbook) == 0:
            return super().clear(orderbook, market_products)
        orderbook_df = pd.DataFrame(orderbook)
        orderbook_df["accepted_volume"] = 0.0
        orderbook_df["accepted_price"] = 0.0

        snapshots = pd.date_range(
            start=market_products[0][0],  # start time
            end=market_products[-1][0],  # end time
            freq=self.marketconfig.market_products[0].duration,
        )

        # Now you can pivot the DataFrame
        volume_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="volume"
        )
        # volume_pivot.index = snapshots
        price_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="price"
        )
        # Copy the network
        n = self.network.copy()

        n.set_snapshots(snapshots)

        # Update p_max_pu for all units based on their bids in the actual snapshots
        # generators
        gen_idx = self.grid_data["generators"].index
        gen_idx = gen_idx.intersection(volume_pivot.columns)
        n.generators_t.p_max_pu.loc[snapshots, gen_idx] = (
            volume_pivot[gen_idx] / n.generators.loc[gen_idx, "p_nom"].values
        )
        n.generators_t.marginal_cost.loc[snapshots, gen_idx] = price_pivot[gen_idx]
        # demand
        demand_idx = self.grid_data["loads"].index
        demand_idx = demand_idx.intersection(volume_pivot.columns)
        n.generators_t.p_min_pu.loc[snapshots, demand_idx] = (
            volume_pivot[demand_idx] / n.generators.loc[demand_idx, "p_nom"].values
        )
        n.generators_t.marginal_cost.loc[snapshots, demand_idx] = price_pivot[
            demand_idx
        ]

        # storage
        if self.grid_data.get("storage_units") is not None:
            storage_idx = self.grid_data["storage_units"].index
            storage_idx = storage_idx.intersection(volume_pivot.columns)
            # discharging (positive bids)
            n.generators_t.p_max_pu.loc[snapshots, storage_idx] = (
                volume_pivot[storage_idx].clip(lower=0).fillna(0)
                / n.generators.loc[storage_idx, "p_nom"].values
            )
            # charging (negative bids)
            n.generators_t.p_min_pu.loc[snapshots, storage_idx] = (
                volume_pivot[storage_idx].clip(upper=0).fillna(0)
                / n.generators.loc[storage_idx, "p_nom"].values
            )
            # set bid price as marginal costs in the respective hours
            n.generators_t.marginal_cost.loc[snapshots, storage_idx] = price_pivot[
                storage_idx
            ].fillna(0)

        # run linear optimal powerflow
        n.optimize.fix_optimal_capacities()
        status, termination_condition = n.optimize(
            solver=self.solver,
            solver_options=self.solver_options,
            progress=False,
        )

        if status != "ok":
            logger.error(f"Solver exited with {termination_condition}")
            raise Exception("Solver in nodal clearing did not converge")

        # Find intersection of unit_ids in orderbook_df and columns in n.generators_t.p
        valid_units = orderbook_df["unit_id"].unique()
        dispatch = n.generators_t.p

        for unit in valid_units:
            if unit in dispatch.columns:
                # get accepted volume and price for each time snapshot
                accepted_volumes = dispatch[unit]
                if self.pricing_mechanism == "pay_as_clear":
                    accepted_prices = n.buses_t.marginal_price.loc[
                        :, n.generators.loc[unit, "bus"]
                    ]
                elif self.pricing_mechanism == "pay_as_bid":
                    accepted_prices = price_pivot[unit]
                else:
                    raise ValueError("Invalid pricing mechanism.")

                # update orderbook_df with accepted volumes and prices
                for t, (vol, price) in enumerate(
                    zip(accepted_volumes, accepted_prices)
                ):
                    mask = (orderbook_df["unit_id"] == unit) & (
                        orderbook_df["start_time"] == snapshots[t]
                    )
                    orderbook_df.loc[mask, "accepted_volume"] = vol
                    orderbook_df.loc[mask, "accepted_price"] = price

        # return orderbook_df back to orderbook format as list of dicts
        accepted_orders = orderbook_df[orderbook_df["accepted_volume"] != 0].to_dict(
            "records"
        )
        rejected_orders = orderbook_df[orderbook_df["accepted_volume"] == 0].to_dict(
            "records"
        )
        for order in rejected_orders:
            # set the accepted price for each rejected order to zero#
            # this is not yet done concisely across the framework
            order["accepted_price"] = 0
        market_clearing_prices = n.buses_t.marginal_price.to_dict()

        meta = []
        flows = {}

        accepted_orders, rejected_orders, meta, flows = extract_results(
            network=n,
            accepted_orders=accepted_orders,
            rejected_orders=rejected_orders,
            market_products=market_products,
            market_clearing_prices=market_clearing_prices,
            log_flows=self.log_flows,
        )

        return accepted_orders, rejected_orders, meta, flows


def extract_results(
    network: pypsa.Network,
    accepted_orders: Orderbook,
    rejected_orders: Orderbook,
    market_products: list[MarketProduct],
    market_clearing_prices: dict,
    log_flows: bool = False,
):
    """
    Extracts the results of the market clearing from the solved PyPSA model.

    Args:
        network (pypsa.Network): The PyPSA network after solving the market clearing.
        accepted_orders (Orderbook): List of the accepted orders
        rejected_orders (Orderbook): List of the rejected orders
        market_products (list[MarketProduct]): The products to be traded
        market_clearing_prices (dict): The market clearing prices
        log_flows (bool): Whether to log network flows

    Returns:
        tuple[Orderbook, Orderbook, list[dict], dict]: The accepted orders, rejected orders, meta information, and network flows

    """
    meta = []
    supply_volume_dict = {
        node: {t: 0.0 for t in network.snapshots} for node in network.buses.index
    }
    demand_volume_dict = {
        node: {t: 0.0 for t in network.snapshots} for node in network.buses.index
    }

    for order in accepted_orders:
        node = order["node"]
        t = order["start_time"]
        if order["accepted_volume"] > 0:
            supply_volume_dict[node][t] += order["accepted_volume"]
        else:
            demand_volume_dict[node][t] += order["accepted_volume"]

    # write the meta information for each hour of the clearing period
    for node in network.buses.index:
        for product in market_products:
            t = product[0]
            clear_price = market_clearing_prices[node][t]
            supply_volume = supply_volume_dict[node][t]
            demand_volume = demand_volume_dict[node][t]
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
                    "node": node,
                    "product_start": product[0],
                    "product_end": product[1],
                    "only_hours": product[2],
                }
            )

    flows = {}
    if log_flows:
        # extract flows
        flows = network.lines_t.p0.stack(future_stack=True).to_dict()

    return accepted_orders, rejected_orders, meta, flows
