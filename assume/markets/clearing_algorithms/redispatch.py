# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import pypsa

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole
from assume.markets.grid_utils import read_pypsa_grid

log = logging.getLogger(__name__)

logging.getLogger("linopy").setLevel(logging.WARNING)


class RedispatchMarketRole(MarketRole):
    """
    A market role that performs redispatch to resolve congestion in the electricity market.
    It uses PyPSA to model the electricity network and perform the redispatch.
    The redispatched is based on the price the units submit in their orders.
    This allows this to be a cost based redispatch if units submit their marginal costs as prices.
    Or it can be a price based redispatch if units submit actual bid prices.

    Parameters:
        marketconfig (MarketConfig): The market configuration.

    Args:
        marketconfig (MarketConfig): The market configuration.

    Notes:
        Users can also configure the path to the network data, the solver to be used,
        and the backup marginal cost in the param_dict of the market configuration.

    """

    required_fields = ["node"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)
        self.network = pypsa.Network()
        # set snapshots as list from the value marketconfig.producs.count converted to list
        self.network.snapshots = range(marketconfig.market_products[0].count)
        self.network_data = marketconfig.param_dict.get("grid_data")
        assert self.network_data
        # set backup marginal cost
        backup_marginal_cost = marketconfig.param_dict.get("backup_marginal_cost", 10e4)
        read_pypsa_grid(
            self.network, self.network_data, backup_marginal_cost=backup_marginal_cost
        )

        self.solver = self.marketconfig.param_dict.get("solver", "glpk")
        self.env = None

        if self.solver == "gurobi":
            try:
                from gurobipy import Env

                self.env = Env()
                self.env.setParam("LogToConsole", 0)
            except ImportError:
                log.error("gurobi not installed - using GLPK")
                self.solver = "glpk"

        # set the market clearing principle
        # as pay as bid or pay as clear
        self.market_clearing_mechanism = marketconfig.param_dict.get(
            "market_clearing_mechanism", "pay_as_bid"
        )
        assert self.market_clearing_mechanism in ["pay_as_bid", "pay_as_clear"]

    def setup(self):
        super().setup()

        # send grid topology data once
        self.context.schedule_instant_acl_message(
            {
                "context": "write_results",
                "type": "grid_topology",
                "data": self.network_data,
                "market_id": self.marketconfig.market_id,
            },
            receiver_addr=self.context.data.get("output_agent_addr"),
            receiver_id=self.context.data.get("output_agent_id"),
        )

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> Tuple[Orderbook, Orderbook, List[dict]]:
        """
        Performs redispatch to resolve congestion in the electricity market.
        It first checks for congestion in the network and if it finds any, it performs redispatch to resolve it.
        The returned orderbook contains accepted orders with the redispatched volumes and prices.
        The prices are positive for upward redispatch and negative for downward redispatch.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products for which clearing happens.

        Returns:
            Tuple[Orderbook, Orderbook, List[dict]]: The accepted orderbook, rejected orderbook and market metadata.
        """

        orderbook_df = pd.DataFrame(orderbook)
        orderbook_df["accepted_volume"] = 0.0
        orderbook_df["accepted_price"] = 0.0

        # Now you can pivot the DataFrame
        volume_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="volume"
        )
        max_power_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="max_power"
        )
        min_power_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="min_power"
        )
        price_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="price"
        )

        # Calculate p_set, p_max_pu_up, and p_max_pu_down directly using DataFrame operations
        p_set = volume_pivot

        # Calculate p_max_pu_up as difference between max_power and accepted volume
        p_max_pu_up = (max_power_pivot - volume_pivot).div(
            max_power_pivot.where(max_power_pivot != 0, np.inf)
        )

        # Calculate p_max_pu_down as difference between accepted volume and min_power
        p_max_pu_down = (volume_pivot - min_power_pivot).div(
            max_power_pivot.where(max_power_pivot != 0, np.inf)
        )
        p_max_pu_down = p_max_pu_down.clip(lower=0)  # Ensure no negative values

        # Determine the costs directly from the price pivot
        costs = price_pivot

        # Drop units with only negative volumes (if necessary)
        negative_only_units = volume_pivot.lt(0).all()
        p_max_pu_up = p_max_pu_up.drop(
            columns=negative_only_units.index[negative_only_units]
        )
        p_max_pu_down = p_max_pu_down.drop(
            columns=negative_only_units.index[negative_only_units]
        )
        costs = costs.drop(columns=negative_only_units.index[negative_only_units])

        # reset indexes for all dataframes
        p_set.reset_index(inplace=True, drop=True)
        p_max_pu_up.reset_index(inplace=True, drop=True)
        p_max_pu_down.reset_index(inplace=True, drop=True)
        costs.reset_index(inplace=True, drop=True)

        # Update the network parameters
        self.network.loads_t.p_set = p_set

        # Update p_max_pu for generators with _up and _down suffixes
        self.network.generators_t.p_max_pu.update(p_max_pu_up.add_suffix("_up"))
        self.network.generators_t.p_max_pu.update(p_max_pu_down.add_suffix("_down"))

        # Add _up and _down suffix to costs and update the network
        self.network.generators_t.marginal_cost.update(costs.add_suffix("_up"))
        self.network.generators_t.marginal_cost.update(costs.add_suffix("_down") * (-1))

        # run linear powerflow
        self.network.lpf()

        # check lines for congestion where power flow is larget than s_nom
        line_loading = self.network.lines_t.p0.abs() / self.network.lines.s_nom

        # if any line is congested, perform redispatch
        if line_loading.max().max() > 1:
            log.debug("Congestion detected")

            status, termination_condition = self.network.optimize(
                solver_name=self.solver,
                env=self.env,
            )

            if status != "ok":
                log.error(f"Solver exited with {termination_condition}")
                raise Exception("Solver in redispatch market did not converge")

            # process dispatch data
            self.process_dispatch_data(orderbook_df)

        # if no congestion is detected set accepted volume and price to 0
        else:
            log.debug("No congestion detected")

        # return orderbook_df back to orderbook format as list of dicts
        accepted_orders = orderbook_df.to_dict("records")
        rejected_orders = []
        meta = []

        # calculate meta data such as total upwared and downward redispatch, total backup dispatch
        # and total redispatch cost
        for i, product in enumerate(market_products):
            orders = orderbook_df[orderbook_df["start_time"] == product[0]]
            meta.append(self.calculate_meta_data(orders=orders, product=product, i=i))

        # remove all orders to clean up the orderbook and avoid double clearing
        self.all_orders = []

        return accepted_orders, rejected_orders, meta

    def process_dispatch_data(self, orderbook_df: pd.DataFrame):
        """
        This function processes the dispatch data to calculate the redispatch volumes and prices
        and update the orderbook with the accepted volumes and prices.

        Args:
            orderbook_df (pd.DataFrame): The orderbook to be cleared.
        """

        # Get all generators except for _backup generators
        generators_t_p = self.network.generators_t.p.filter(regex="^(?!.*_backup)")

        # Use regex in a single call to filter and rename columns simultaneously for efficiency
        upward_redispatch = generators_t_p.filter(regex="_up$")
        downward_redispatch = generators_t_p.filter(regex="_down$")

        # Find intersection of unit_ids in orderbook_df and columns in redispatch_volumes for direct mapping
        valid_units = orderbook_df["unit_id"].unique()

        for unit in valid_units:
            unit_orders = orderbook_df["unit_id"] == unit

            if f"{unit}_up" in upward_redispatch.columns:
                orderbook_df.loc[unit_orders, "accepted_volume"] += upward_redispatch[
                    f"{unit}_up"
                ].values

            if f"{unit}_down" in downward_redispatch.columns:
                orderbook_df.loc[unit_orders, "accepted_volume"] -= downward_redispatch[
                    f"{unit}_down"
                ].values

            if self.market_clearing_mechanism == "pay_as_bid":
                # set accepted price as the price bid price from the orderbook
                orderbook_df.loc[unit_orders, "accepted_price"] = np.where(
                    orderbook_df.loc[unit_orders, "accepted_volume"] > 0,
                    orderbook_df.loc[unit_orders, "price"],
                    np.where(
                        orderbook_df.loc[unit_orders, "accepted_volume"] < 0,
                        orderbook_df.loc[unit_orders, "price"],
                        0,  # This sets accepted_price to 0 when redispatch_volume is exactly 0
                    ),
                )

            elif self.market_clearing_mechanism == "pay_as_clear":
                # set accepted price as the nodal marginal price
                nodal_marginal_prices = abs(self.network.buses_t.marginal_price)
                unit_node = orderbook_df.loc[unit_orders, "node"].values[0]

                orderbook_df.loc[unit_orders, "accepted_price"] = np.where(
                    orderbook_df.loc[unit_orders, "accepted_volume"] != 0,
                    nodal_marginal_prices[unit_node],
                    0,
                )

    def calculate_meta_data(self, orders, product: MarketProduct, i: int):
        """
        This function calculates the meta data such as total upward and downward redispatch,
        total backup dispatch, and total redispatch cost.

        Args:
            product (MarketProduct): The product for which clearing happens.
            i (int): The index of the product in the market products list.

        Returns:
            dict: The meta data.
        """

        total_upward_redispatch = (
            self.network.generators_t.p.iloc[i].filter(regex="_up").sum()
        )
        total_downward_redispatch = (
            self.network.generators_t.p.iloc[i].filter(regex="_down").sum()
        )
        total_backup_dispatch = (
            self.network.generators_t.p.iloc[i].filter(regex="_backup").sum()
        )

        total_redispatch_cost = orders["accepted_volume"].dot(orders["accepted_price"])

        return {
            "total_upward_redispatch": round(total_upward_redispatch, 3),
            "total_downward_redispatch": round(total_downward_redispatch, 3),
            "total_backup_dispatch": round(total_backup_dispatch, 3),
            "total_redispatch_cost": round(total_redispatch_cost, 3),
            "product_start": product[0],
            "product_end": product[1],
            "only_hours": product[2],
        }
