# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import pandas as pd
import pypsa

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


class RedispatchMarketRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()
        # set snapshots as list from the value marketconfig.producs.count converted to list
        self.network.snapshots = range(marketconfig.market_products[0].count)
        self.solver = marketconfig.param_dict.get("solver", "glpk")
        network_path = marketconfig.param_dict.get("network_path")
        assert network_path

        # set backup marginal cost
        self.backup_marginal_cost = marketconfig.param_dict.get(
            "backup_marginal_cost", 10e4
        )

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

        Typically nodes are those substations to which the transmission lines are connected to.

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
        This adds generators in the PyPSA network with respective bus data to which they are connected
        """
        generators = pd.read_csv(filename, index_col=0)

        # make dataframe for p_set as zeros for data and index as snapshots
        # and coliumns as generator names
        p_set = pd.DataFrame(
            np.zeros((len(self.network.snapshots), len(generators.index))),
            index=self.network.snapshots,
            columns=generators.index,
        )

        # add generators and their sold capacities as load with reversed sign to have fixed feed in
        self.network.madd(
            "Load",
            names=generators.index,
            bus=generators["node"],  # bus to which the generator is connected to
            p_set=p_set,
            sign=1,
        )

        # add upward redispatch generators
        self.network.madd(
            "Generator",
            names=generators.index,
            suffix="_up",
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            p_min_pu=p_set,
            p_max_pu=p_set + 1,
            marginal_cost=p_set,
        )

        # add downward redispatch generators
        self.network.madd(
            "Generator",
            names=generators.index,
            suffix="_down",
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            p_min_pu=p_set,
            p_max_pu=p_set + 1,
            marginal_cost=p_set,
            sign=-1,
        )

        # add upward and downward backup generators at each node
        self.network.madd(
            "Generator",
            names=self.network.buses.index,
            suffix="_backup_up",
            bus=self.network.buses.index,  # bus to which the generator is connected to
            p_nom=10e4,
            marginal_cost=10e4,
        )

        self.network.madd(
            "Generator",
            names=self.network.buses.index,
            suffix="_backup_down",
            bus=self.network.buses.index,  # bus to which the generator is connected to
            p_nom=10e4,
            marginal_cost=self.backup_marginal_cost,
            sign=-1,
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

        # Iterate through time steps and add generators
        self.network.madd(
            "Load",
            names=loads.index,
            bus=loads["node"],  # bus to which the generator is connected to
            p_set=p_set,
            sign=1,
        )

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        Performs redispatch to resolve congestion in the electricity market.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products for which clearing happens.

        Returns:
            (Orderbook, Orderbook, list[dict]): The accepted orderbook, rejected orderbook and market metadata.
        """

        orderbook_df = pd.DataFrame(orderbook)

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
        self.network.generators_t.marginal_cost.update(costs.add_suffix("_down"))

        # run linear powerflow
        self.network.lpf()

        # check lines for congestion where power flow is larget than s_nom
        line_loading = self.network.lines_t.p0.abs() / self.network.lines.s_nom

        if line_loading.max().max() > 1:
            log.debug("Congestion detected")

            # lopf is deprecated
            status, termination_condition = self.network.optimize(
                solver_name=self.solver
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
            meta.append(self.calculate_meta_data(product, i))

        # remove all orders to clean up the orderbook and avoid double clearing
        self.all_orders = []

        return accepted_orders, rejected_orders, meta

    def process_dispatch_data(self, orderbook_df: pd.DataFrame):
        # Extract backup, upward, and downward redispatch
        generators_t_p = self.network.generators_t.p

        # Use regex in a single call to filter and rename columns simultaneously for efficiency
        upward_redispatch = generators_t_p.filter(regex="_up").rename(
            columns=lambda x: x.replace("_up", "")
        )
        downward_redispatch = generators_t_p.filter(regex="_down").rename(
            columns=lambda x: x.replace("_down", "")
        )

        # Calculate redispatch volumes
        redispatch_volumes = upward_redispatch.sub(downward_redispatch)

        # Initialize accepted_volume and accepted_price columns
        orderbook_df["accepted_volume"] = 0.0
        orderbook_df["accepted_price"] = 0.0

        # Find intersection of unit_ids in orderbook_df and columns in redispatch_volumes for direct mapping
        valid_units = orderbook_df["unit_id"].unique()
        valid_columns = [
            unit for unit in valid_units if unit in redispatch_volumes.columns
        ]

        # Directly apply values based on the valid_columns without looping
        for unit in valid_columns:
            unit_orders = orderbook_df["unit_id"] == unit

            # Direct mapping of volumes
            orderbook_df.loc[unit_orders, "accepted_volume"] = redispatch_volumes[
                unit
            ].values

            # Use np.where to directly apply conditional pricing based on the sign of the redispatch volume
            # and ensure that the volume is not zero to avoid assigning a value to accepted_price in such cases.
            orderbook_df.loc[unit_orders, "accepted_price"] = np.where(
                redispatch_volumes[unit].values > 0,
                orderbook_df.loc[unit_orders, "price"],
                np.where(
                    redispatch_volumes[unit].values < 0,
                    -orderbook_df.loc[unit_orders, "price"],
                    0,  # This sets accepted_price to 0 when redispatch_volume is exactly 0
                ),
            )

    def calculate_meta_data(self, product: MarketProduct, i: int):
        # Calculate meta data such as total upward and downward redispatch, total backup dispatch, and total redispatch cost
        redispatch_volumes = self.network.generators_t.p.iloc[i]
        upward_redispatch_price = self.network.generators_t.marginal_cost.iloc[
            i
        ].filter(regex="_up")
        downward_redispatch_price = self.network.generators_t.marginal_cost.iloc[
            i
        ].filter(regex="_down")

        # Calculate total redispatch cost as sum of accepted volumes times the marginal cost for upward redispatch
        # minus the accepted volumes times the marginal cost for downward redispatch
        total_redispatch_cost = (
            redispatch_volumes.filter(regex="_up") * upward_redispatch_price
        ).sum() - (
            redispatch_volumes.filter(regex="_down") * downward_redispatch_price
        ).sum()

        return {
            "total_upward_redispatch": redispatch_volumes.filter(regex="_up").sum(),
            "total_downward_redispatch": redispatch_volumes.filter(regex="_down").sum(),
            "total_backup_dispatch": redispatch_volumes.filter(regex="_backup").sum(),
            "total_redispatch_cost": total_redispatch_cost,
            "product_start": product[0],
            "product_end": product[1],
            "only_hours": product[2],
        }
