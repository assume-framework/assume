# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import Counter

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
        snapshots = list(range(marketconfig.market_products[0].count))
        self.network.set_snapshots(snapshots)
        self.solver = marketconfig.solver

        # setup the network
        # add buses
        self.add_buses(f"{marketconfig.network_path}/buses.csv")

        # add lines
        self.add_lines(f"{marketconfig.network_path}/lines.csv")

        # add generators
        self.add_generators(f"{marketconfig.network_path}/powerplant_units.csv")

        # add loads
        self.add_loads(f"{marketconfig.network_path}/demand_units.csv")

    def add_buses(self, filename):
        """
        This adds nodes in the PyPSA network to which the generators and loads are connected

        Typically nodes are those substations to which the transmission lines are connected to.

        """
        buses = pd.read_csv(filename)

        self.network.madd(
            "Bus",
            names=buses["bus"],
            v_nom=buses["v_nom"],  # Nominal voltage level of the bus(substation)
            carrier=buses["carrier"],
            x=buses["x"],  # longitude
            y=buses["y"],  # lattitude
        )

    def add_lines(self, filename):
        """
        This creates transmission network in PyPSA by connecting buses with predefined line capacities
        """
        lines = pd.read_csv(filename)

        self.network.madd(
            "Line",
            names=lines["name"],
            bus0=lines["bus0"],
            bus1=lines["bus1"],
            s_nom=lines["s_nom"],  # transmission line capacity in MW
            x=1,
            s_nom_extendable=lines["s_nom_extendable"],
        )

    def add_generators(self, filename):
        """
        This adds generators in the PyPSA network with respective bus data to which they are connected
        """
        generators = pd.read_csv(filename, index_col=0, header=0)

        # Iterate through time steps and add generators
        self.network.madd(
            "Generator",
            names=generators["name"],
            bus=generators["node"],  # bus to which the generator is connected to
            p_nom=generators[
                "max_power"
            ],  # Nominal capacity of the powerplant/generator
            p_max_pu=1,
            p_min_pu=generators["min_power"] / generators["max_power"],
        )

    def add_loads(self, filename):
        """
        This adds loads in the PyPSA network with respective bus data to which they are connected
        """
        loads = pd.read_csv(filename, index_col=0, header=0)

        # Iterate through time steps and add generators
        self.network.madd(
            "Generator",
            names=loads["name"],
            bus=loads["node"],  # bus to which the generator is connected to
            p_nom=loads["max_power"],  # Nominal capacity of the powerplant/generator
            p_max_pu=1,
            p_min_pu=loads["min_power"] / loads["max_power"],
            sign=-1,
        )

    def congestion_identification(self):
        """
        This function identifies congestion in the transmission lines by running simple power flow in PyPSA network
        """

        zero_r_lines = self.network.lines.index[
            self.network.lines.r == 0
        ]  # A small non-zero value for resistance in lines with zero resistance
        self.network.lines.loc[zero_r_lines, "r"] = 1e-9
        self.congestion_info = []

        self.network.pf()

        line_index_counter = Counter()

        for (
            timestep
        ) in (
            self.network.snapshots
        ):  # for each timestep the capacity of line is compared with the actual loading in lines
            for index, (loading, s_nom_value) in enumerate(
                zip(
                    np.round(np.abs(self.network.lines_t.p0.values.flatten())),
                    self.network.lines.s_nom,
                )
            ):
                net_loading = loading - s_nom_value
                congestion_status = net_loading > 0
                self.congestion_info.append(
                    {
                        "timestamp": timestep,
                        "line_index": self.network.lines.index[index],
                        "nominal_grid_cap": s_nom_value,
                        "line_loading": loading,
                        "net_loading": net_loading,
                        "congestion_status": congestion_status,
                    }
                )

                if (
                    congestion_status
                ):  # this counter adds everytime for each line if congestion status is True
                    line_index_counter[self.network.lines.index[index]] += 1

        self.congestion_df = pd.DataFrame(self.congestion_info)
        self.congestion = (
            lambda x: "Network is congested" if x.any() else "There is no congestion"
        )(self.congestion_df["congestion_status"])
        counter_df = pd.DataFrame(
            list(line_index_counter.items()), columns=["line_index", "congestion_count"]
        )
        counter_df.to_csv("congestion_count.csv", index=False)

        print(f"Congestion count for each line_index:\n {counter_df}")

        return self.congestion_info

    def redispatch(self):
        """
        This function runs the redispatch of powerplants if the network is congested.

        It adds one extra generator at each bus with the capacity of power generation equal to the uncleared capacity in the day ahead market

        Also, adds one extra negative generator to symbolize curtailment
        """
        redispatch_info = []
        line_index_counter = Counter()

        if self.congestion_df[
            "congestion_status"
        ].any():  # This loops runs only when there is congestion
            self.redisp_network = self.network.copy()
            self.redisp_network.lines["s_nom_extendable"] = False
            self.redisp_network.generators["control"] = "PQ"

            # 1. Positive Redispatch: Adding one extra generator at every node for +ve redispatch
            generator_data_pos = self.generator_data.copy()

            # p_max_pu is fraction which is estimated to calculate maximum available capacity for positive redispatch
            generator_data_pos["p_max_pu"] = (
                generator_data_pos["p_nom"] - generator_data_pos["p_da"]
            ) / (generator_data_pos["p_nom"])
            generator_data_pos["p_min_pu"] = 0
            grouped_data_pos = (
                generator_data_pos.groupby("name")
                .agg(lambda x: x.tolist())
                .reset_index()
            )
            grouped_data_pos["p_nom"] = grouped_data_pos["p_nom"].apply(
                lambda x: x[0] if len(set(x)) == 1 else x
            )

            for index, row in grouped_data_pos.iterrows():
                self.redisp_network.add(
                    "Generator",
                    name=f"posredisp_{row['name']}",
                    bus=" ".join(set(row["bus"])),
                    marginal_cost=row["marginal_cost"],
                    p_nom_extendable=False,
                    p_nom=row["p_nom"],
                    p_max_pu=row["p_max_pu"],
                    p_min_pu=row["p_min_pu"],
                    p_set=row["p_da"],
                    carrier=" ".join(set(row["carrier"])),
                )
            print("Positive redispatch capacity is added")

            # 2. Negative Redispatch: Adding one extra generator at every node for -ve redispatch

            generator_data_neg = self.generator_data.copy()
            generator_data_neg["p_max_pu"] = (
                generator_data_neg["p_da"] - generator_data_neg["p_min"]
            ) / (generator_data_neg["p_nom"])
            generator_data_neg["p_min_pu"] = 0
            grouped_data_neg = (
                generator_data_neg.groupby("name")
                .agg(lambda x: x.tolist())
                .reset_index()
            )
            grouped_data_neg["p_nom"] = grouped_data_neg["p_nom"].apply(
                lambda x: x[0] if len(set(x)) == 1 else x
            )

            for index, row in grouped_data_neg.iterrows():
                self.redisp_network.add(
                    "Generator",
                    name=f"negredisp_{row['name']}",
                    bus=" ".join(set(row["bus"])),
                    marginal_cost=row["marginal_cost"],
                    p_nom_extendable=False,
                    sign=-1,
                    p_nom=row["p_nom"],
                    p_max_pu=row["p_max_pu"],
                    p_min_pu=row["p_min_pu"],
                    p_set=row["p_da"],
                    carrier=" ".join(set(row["carrier"])),
                )

            print("Negative redispatch capacity is added")

            # 3. Solve the network for redispatch
            self.redisp_network.lopf(solver_name=self.solver)

            # 4. Results and Data storing

            for timestep in self.redisp_network.snapshots:
                for index, (loading, s_nom_value) in enumerate(
                    zip(
                        np.round(
                            np.abs(self.redisp_network.lines_t.p0.values.flatten())
                        ),
                        self.redisp_network.lines.s_nom,
                    )
                ):
                    net_loading = loading - s_nom_value
                    congestion_status = net_loading > 0
                    redispatch_info.append(
                        {
                            "timestamp": timestep,
                            "line_index": self.redisp_network.lines.index[index],
                            "congestion_status": congestion_status,
                        }
                    )

                    if congestion_status:
                        line_index_counter[self.redisp_network.lines.index[index]] += 1

            counter_df = pd.DataFrame(
                list(line_index_counter.items()),
                columns=["line_index", "congestion_count"],
            )
            counter_0 = set(self.redisp_network.lines.index) - set(
                counter_df["line_index"]
            )
            missing_counts = pd.DataFrame(
                {"line_index": list(counter_0), "congestion_count": 0}
            )

            counter_df = pd.concat([counter_df, missing_counts], ignore_index=True)
            counter_df.to_csv("congestion_count_redisp.csv", index=False)

            redisp_results = pd.DataFrame(self.redisp_network.generators_t.p)

            # the following loop adds positive redispatch to the respective generator and/or reduce negative redispatch from original day ahead generation plan
            for col in self.network.generators_t.p_set.columns:
                redisp_results[f"total_{col}"] = (
                    redisp_results[col]
                    + redisp_results[f"posredisp_{col}"]
                    - redisp_results[f"negredisp_{col}"]
                )
                total_columns = redisp_results.filter(regex=r"^total_\w+$")
                total_columns.to_csv("final_dispatch.csv", index=True)

            return self.redisp_network

        else:
            print("No redispatch needed")

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        Performs redispatch to resolve congestion in the electricity market.

        :param orderbook: The orderbook containing the orders to be cleared
        :param market_products: The products to be traded
        :return: accepted_orders, rejected_orders, meta
        """
        # Initialize accepted and rejected orders
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []

        # Identify congestion in the network
        congestion_info = self.network_clear_instance.congestion_identification()

        # If there is congestion, perform redispatch
        if any(info["congestion_status"] for info in congestion_info):
            self.network_clear_instance.redispatch()

            # Read the final dispatch results
            final_dispatch = pd.read_csv("final_dispatch.csv", index_col=0)

            # Process the final dispatch results and update the orderbook accordingly
            for product in market_products:
                for index, row in final_dispatch.iterrows():
                    if f"total_{product.name}" in row:
                        accepted_order = {
                            "start_time": product.start_time,
                            "end_time": product.end_time,
                            "only_hours": product.only_hours,
                            "volume": row[f"total_{product.name}"],
                            "accepted_price": row[
                                "marginal_cost"
                            ],  # Assuming marginal_cost is available
                        }
                        accepted_orders.append(accepted_order)

            # Meta information could include details about the redispatch process
            meta.append({"redispatch": "completed"})

        # If no congestion, the original orderbook is accepted
        else:
            accepted_orders = orderbook
            meta.append({"redispatch": "not required"})

        return accepted_orders, rejected_orders, meta
