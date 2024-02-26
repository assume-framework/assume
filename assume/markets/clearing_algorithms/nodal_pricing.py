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

log = logging.getLogger(__name__)

logging.getLogger("linopy").setLevel(logging.WARNING)


class NodalMarketRole(MarketRole):
    """

    A market role that performs market clearing at each node (bus) in an electricity network.
    It uses PyPSA to model the electricity network and perform market clearing.

    Parameters:
        marketconfig (MarketConfig): The market configuration.

    Args:
        marketconfig (MarketConfig): The market configuration.

    Notes:
        Users can also configure the path to the network data, the solver to be used,
        and the backup marginal cost in the param_dict of the market configuration.

    """

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()
        # set snapshots as list from the value marketconfig.producs.count converted to list
        self.network.snapshots = range(marketconfig.market_products[0].count)
        self.solver = marketconfig.param_dict.get("solver", "glpk")
        network_path = marketconfig.param_dict.get("network_path")
        self.env = None
        assert network_path

        if self.solver == "gurobi":
            try:
                from gurobipy import Env

                self.env = Env()
                self.env.setParam("LogToConsole", 0)
            except ImportError:
                log.error("gurobi not installed - using GLPK")
                self.solver = "glpk"

        # set backup marginal cost
        self.backup_marginal_cost = marketconfig.param_dict.get(
            "backup_marginal_cost", 10e4
        )

        # set the market clearing principle
        # as pay as bid or pay as clear
        self.market_clearing_mechanism = marketconfig.param_dict.get(
            "market_clearing_mechanism", "pay_as_bid"
        )
        assert self.market_clearing_mechanism in ["pay_as_bid", "pay_as_clear"]

        # setup the network
        # add buses
        self.add_buses(f"{network_path}/buses.csv")

        # add lines
        self.add_lines(f"{network_path}/lines.csv")

        # add generators
        self.add_generators(f"{network_path}/powerplant_units.csv")

        # add loads
        self.add_loads(f"{network_path}/demand_units.csv")

    def add_buses(self, filename: str):
        """
        This adds nodes in the PyPSA network to which the generators and loads are connected

        """
        buses = pd.read_csv(filename, index_col=0)

        self.network.madd(
            "Bus",
            names=buses.index,
            **buses,
        )

    def add_lines(self, filename: str):
        """
        This creates transmission network in PyPSA by connecting buses with predefined line capacities
        """
        lines = pd.read_csv(filename, index_col=0)

        self.network.madd(
            "Line",
            names=lines.index,
            **lines,
        )

    def add_generators(self, filename: str):
        """
        This adds generators in the PyPSA network with respective bus data to which they are connected.
        It creates upward and downward redispatch generators for each generator and adds backup generators at each node
        """
        generators = pd.read_csv(filename, index_col=0)

        # make dataframe for p_set as zeros for data and index as snapshots
        # and coliumns as generator names
        p_set = pd.DataFrame(
            np.zeros((len(self.network.snapshots), len(generators.index))),
            index=self.network.snapshots,
            columns=generators.index,
        )

        # add generators
        self.network.madd(
            "Generator",
            names=generators.index,
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            p_min_pu=p_set,
            p_max_pu=p_set + 1,
            marginal_cost=p_set,
            committable=True,
            **generators,
        )

        # add upward and downward backup generators at each node
        self.network.madd(
            "Generator",
            names=self.network.buses.index,
            suffix="_backup",
            bus=self.network.buses.index,  # bus to which the generator is connected to
            p_nom=10e4,
            marginal_cost=self.backup_marginal_cost,
        )

    def add_loads(self, filename: str):
        """
        This adds loads in the PyPSA network with respective bus data to which they are connected
        """
        loads = pd.read_csv(filename, index_col=0)

        p_set = pd.DataFrame(
            np.zeros((len(self.network.snapshots), len(loads.index))),
            index=self.network.snapshots,
            columns=loads.index,
        )

        # add loads as negative generators
        self.network.madd(
            "Generator",
            names=loads.index,
            bus=loads["node"],  # bus to which the generator is connected to
            p_nom=loads["max_power"],  # Nominal capacity of the powerplant/generator
            p_min_pu=p_set,
            p_max_pu=p_set + 1,
            marginal_cost=p_set,
            committable=True,
            sign=-1,
            **loads,
        )

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> Tuple[Orderbook, Orderbook, List[dict]]:
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

        # Calculate p_max_pu_up as difference between max_power and accepted volume
        p_max_pu = (max_power_pivot - volume_pivot).div(
            max_power_pivot.where(max_power_pivot != 0, np.inf)
        )

        # Calculate p_max_pu_down as difference between accepted volume and min_power
        p_min_pu = (volume_pivot - min_power_pivot).div(
            max_power_pivot.where(max_power_pivot != 0, np.inf)
        )
        p_min_pu = p_min_pu.clip(lower=0)  # Ensure no negative values

        # reset indexes for all dataframes
        p_max_pu.reset_index(inplace=True, drop=True)
        p_min_pu.reset_index(inplace=True, drop=True)
        costs.reset_index(inplace=True, drop=True)

        # Update the network parameters

        # Update p_max_pu for generators with _up and _down suffixes
        self.network.generators_t.p_max_pu.update(p_max_pu)
        self.network.generators_t.p_min_pu.update(p_min_pu)

        # Add _up and _down suffix to costs and update the network
        self.network.generators_t.marginal_cost.update(costs)

        status, termination_condition = self.network.optimize(
            solver_name=self.solver,
            env=self.env,
        )

        if status != "ok":
            log.error(f"Solver exited with {termination_condition}")
            raise Exception("Solver in redispatch market did not converge")

        # process dispatch data
        self.process_dispatch_data(orderbook_df)

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

            orderbook_df.loc[unit_orders, "accepted_volume"] += generators_t_p[
                unit
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

        total_redispatch = self.network.generators_t.p.iloc[i].sum()

        total_backup_dispatch = (
            self.network.generators_t.p.iloc[i].filter(regex="_backup").sum()
        )

        total_dispatch_cost = orders["accepted_volume"].dot(orders["accepted_price"])

        return {
            "total_dispatch": round(total_redispatch, 3),
            "total_backup_dispatch": round(total_backup_dispatch, 3),
            "total_dispatch_cost": round(total_dispatch_cost, 3),
            "product_start": product[0],
            "product_end": product[1],
            "only_hours": product[2],
        }
