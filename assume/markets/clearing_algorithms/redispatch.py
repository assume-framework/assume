# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import Counter
from itertools import groupby
from operator import itemgetter

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
        snapshots = range(marketconfig.market_products[0].count)
        self.network.set_snapshots(snapshots)
        self.solver = marketconfig.solver

        # setup the network
        # add buses
        self.add_buses(f"{marketconfig.network_path}/buses.csv")

        # add lines
        self.add_lines(f"{marketconfig.network_path}/lines.csv")

        # add generators
        # TODO: add config parameter for price of the backup generators and backup generators themselves
        self.add_generators(f"{marketconfig.network_path}/powerplant_units.csv")

        # add loads
        self.add_loads(f"{marketconfig.network_path}/demand_units.csv")

    def add_buses(self, filename):
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

    def add_lines(self, filename):
        """
        This creates transmission network in PyPSA by connecting buses with predefined line capacities
        """
        lines = pd.read_csv(filename, index_col=0)

        self.network.madd(
            "Line",
            names=lines.index,
            **lines,
        )

    def add_generators(self, filename):
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
            marginal_cost=10e4,
            sign=-1,
        )

    def add_loads(self, filename):
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

        :param orderbook: The orderbook containing the orders to be cleared
        :param market_products: The products to be traded
        :return: accepted_orders, rejected_orders, meta
        """

        # convert orderbook into pandas dataframe
        orderbook_df = pd.DataFrame(orderbook)

        print(orderbook_df)

        # construct new p_set dataframe for generators using the volume
        all_units = orderbook_df["unit_id"].unique()

        p_set = pd.DataFrame(
            np.ones((len(self.network.snapshots), len(all_units))),
            index=self.network.snapshots,
            columns=all_units,
        )

        p_max_pu_up = p_set.copy()
        p_max_pu_down = p_set.copy()
        costs = p_set.copy()

        # update values of p_set for generators from orderbook_df for each unit
        for unit in all_units:
            # get all orders for the unit
            unit_orders = orderbook_df[orderbook_df["unit_id"] == unit].index

            accepted_volume = orderbook_df.loc[unit_orders, "volume"]
            p_set[unit] = accepted_volume.values

            if (accepted_volume < 0).all():
                # drop this unit from p_max_pu_up and p_max_pu_down
                p_max_pu_up.drop(unit, axis=1, inplace=True)
                p_max_pu_down.drop(unit, axis=1, inplace=True)
                costs.drop(unit, axis=1, inplace=True)
                continue

            max_power = orderbook_df.loc[unit_orders, "max_power"]
            min_power = orderbook_df.loc[unit_orders, "min_power"]

            # calculate p_max_pu_up for unit as difference between max_power and accepted volume
            pos_redispatch_max_pu = (max_power - accepted_volume) / max_power
            p_max_pu_up[unit] = pos_redispatch_max_pu.values

            # calculate p_max_pu_down for unit as difference between accepted volume and min_power
            neg_redispatch_pu = (accepted_volume - min_power) / max_power
            neg_redispatch_pu = neg_redispatch_pu.where(neg_redispatch_pu > 0, 0)
            p_max_pu_down[unit] = neg_redispatch_pu.values

            # calculate costs for unit
            marginal_cost = orderbook_df.loc[unit_orders, "price"]
            costs[unit] = marginal_cost.values

        # update p_set for loads
        self.network.loads_t.p_set = p_set

        # add _up suphix to p_max_pu_up
        p_max_pu_up = p_max_pu_up.add_suffix("_up")

        # update p_max_pu_up for generators with _up suffix
        self.network.generators_t.p_max_pu[p_max_pu_up.columns] = p_max_pu_up

        # add _down suphix to p_max_pu_down
        p_max_pu_down = p_max_pu_down.add_suffix("_down")

        # update p_max_pu_down for generators with _down suffix
        self.network.generators_t.p_max_pu[p_max_pu_down.columns] = p_max_pu_down

        # add _up and _down suffix to costs
        costs_up = costs.add_suffix("_up")
        costs_down = costs.add_suffix("_down")

        # update costs for generators with _up and _down suffix
        self.network.generators_t.marginal_cost[costs_up.columns] = costs_up
        self.network.generators_t.marginal_cost[costs_down.columns] = costs_down

        # run lopf
        self.network.lpf()

        # cehck lines for congestion where power flow is larget than s_nom
        line_loading = self.network.lines_t.p0.abs() / self.network.lines.s_nom

        if line_loading.max().max() > 1:
            log.debug("Congestion detected")
            results = self.network.lopf(solver_name=self.solver)

            # TODO: add code to check solver status and if not optimal, then raise exception

        # from self.network.generators_t.p add all values from columns with _up suffix to volume
        # and subtract all values from columns with _down suffix from volume for respective unit
        # and update orderbook_df

        self.all_orders = []
