# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import pandas as pd
import pypsa

from assume.common.grid_utils import (
    add_backup_generators,
    add_generators,
    add_nodal_loads,
    calculate_network_meta,
    read_pypsa_grid,
)
from assume.common.market_objects import MarketConfig, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

logging.getLogger("linopy").setLevel(logging.WARNING)
logging.getLogger("pypsa").setLevel(logging.WARNING)


class NodalMarketRole(MarketRole):
    """

    A market role that performs market clearing at each node (bus) in an electricity network.
    It uses PyPSA to model the electricity network and perform market clearing.

    Args:
        marketconfig (MarketConfig): The market configuration.

    Notes:
        Users can also configure the path to the network data, the solver to be used,
        and the backup marginal cost in the param_dict of the market configuration.

    """

    required_fields = ["node", "max_power", "min_power"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()
        # set snapshots as list from the value marketconfig.producs.count converted to list
        self.network.snapshots = range(marketconfig.market_products[0].count)
        assert self.grid_data

        read_pypsa_grid(
            network=self.network,
            grid_dict=self.grid_data,
        )
        add_generators(
            network=self.network,
            generators=self.grid_data["generators"],
        )
        add_backup_generators(
            network=self.network,
            backup_marginal_cost=marketconfig.param_dict.get(
                "backup_marginal_cost", 10e4
            ),
        )
        add_nodal_loads(
            network=self.network,
            loads=self.grid_data["loads"],
        )

        self.solver = marketconfig.param_dict.get("solver", "glpk")
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
        self.payment_mechanism = marketconfig.param_dict.get(
            "payment_mechanism", "pay_as_bid"
        )
        assert self.payment_mechanism in ["pay_as_bid", "pay_as_clear"]

    def setup(self):
        super().setup()

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Clears the market by running a linear optimal power flow (LOPF) with PyPSA.

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
        costs = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="price"
        )
        # change costs to negative where volume is negative
        costs = costs.where(volume_pivot > 0, -costs)

        # Calculate p_max_pu_up as difference between max_power and accepted volume
        p_max_pu = volume_pivot.div(max_power_pivot.where(max_power_pivot != 0, np.inf))

        # Calculate p_max_pu_down as difference between accepted volume and min_power
        p_min_pu = min_power_pivot.div(
            max_power_pivot.where(max_power_pivot != 0, np.inf)
        )
        p_min_pu = p_min_pu.clip(lower=0)  # Ensure no negative values

        # reset indexes for all dataframes
        p_max_pu.reset_index(inplace=True, drop=True)
        p_min_pu.reset_index(inplace=True, drop=True)
        costs.reset_index(inplace=True, drop=True)

        # Update the network parameters
        nodal_network = self.network.copy()

        # Update p_max_pu for generators
        nodal_network.generators_t.p_max_pu.update(p_max_pu)
        nodal_network.generators_t.p_min_pu.update(p_min_pu)

        # Update marginal costs for generators
        nodal_network.generators_t.marginal_cost.update(costs)

        status, termination_condition = nodal_network.optimize(
            solver_name=self.solver,
            env=self.env,
        )

        if status != "ok":
            log.error(f"Solver exited with {termination_condition}")
            raise Exception("Solver in redispatch market did not converge")

        # process dispatch data
        self.process_dispatch_data(network=nodal_network, orderbook_df=orderbook_df)

        # return orderbook_df back to orderbook format as list of dicts
        accepted_orders = orderbook_df.to_dict("records")
        rejected_orders = []
        meta = []

        # calculate meta data such as total upwared and downward redispatch, total backup dispatch
        # and total redispatch cost
        for i, product in enumerate(market_products):
            meta.extend(
                calculate_network_meta(network=nodal_network, product=product, i=i)
            )

        return accepted_orders, rejected_orders, meta

    def process_dispatch_data(self, network: pypsa.Network, orderbook_df: pd.DataFrame):
        """
        This function processes the dispatch data to calculate the dispatch volumes and prices
        and update the orderbook with the accepted volumes and prices.

        Args:
            orderbook_df (pd.DataFrame): The orderbook to be cleared.
        """

        # Get all generators except for _backup generators
        generators_t_p = network.generators_t.p.filter(regex="^(?!.*_backup)").copy()

        # select demand units as those with negative volume in orderbook
        demand_units = orderbook_df[orderbook_df["volume"] < 0]["unit_id"].unique()

        # change values to negative for demand units
        generators_t_p.loc[:, demand_units] *= -1

        # Find intersection of unit_ids in orderbook_df and columns in redispatch_volumes for direct mapping
        valid_units = orderbook_df["unit_id"].unique()

        for unit in valid_units:
            unit_orders = orderbook_df["unit_id"] == unit

            orderbook_df.loc[unit_orders, "accepted_volume"] += generators_t_p[
                unit
            ].values

            if self.payment_mechanism == "pay_as_bid":
                # set accepted price as the price bid price from the orderbook
                orderbook_df.loc[unit_orders, "accepted_price"] = np.where(
                    orderbook_df.loc[unit_orders, "accepted_volume"] > 0,
                    orderbook_df.loc[unit_orders, "price"],
                    np.where(
                        orderbook_df.loc[unit_orders, "accepted_volume"] < 0,
                        orderbook_df.loc[unit_orders, "price"],
                        0,  # This sets accepted_price to 0 when accepted_volume is exactly 0
                    ),
                )

            elif self.payment_mechanism == "pay_as_clear":
                # set accepted price as the nodal marginal price
                nodal_marginal_prices = -network.buses_t.marginal_price
                unit_node = orderbook_df.loc[unit_orders, "node"].values[0]

                orderbook_df.loc[unit_orders, "accepted_price"] = np.where(
                    orderbook_df.loc[unit_orders, "accepted_volume"] != 0,
                    nodal_marginal_prices[unit_node],
                    0,
                )
