# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
import subprocess
import warnings
from collections import Counter

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandas as pd
import pypsa
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pypsa.pf").setLevel(logging.WARNING)

# Path to the location of input files
os.chdir("./examples/inputs/example_03a")

# Path to the location of output files to be saved
if not os.path.exists("./examples/outputs"):
    os.makedirs("./examples/outputs")
else:
    print(f"The directory already exists.")


class Network_clear:
    """
    This class creates PyPSA network to identify congestion in the network

    It adds components to the PyPSA network and solves the power flow to :identify congestion

    If there is congestion then it solves linear optimized powerflow to :redispatch the powerplants

    """

    def __init__(self):
        self.network = pypsa.Network()
        self.network.set_snapshots(
            pd.date_range("2024-01-01 00:00:00", periods=24, freq="H")
        )
        self.solver = "glpk"
        self.solver_path = (
            "C:\\Users\\par19744\\.conda\\envs\\PyPSA\\Library\\bin\\glpsol"
        )

    def add_buses(self, filename, header):
        """
        This adds nodes in the PyPSA network to which the generators and loads are connected

        Typically nodes are those substations to which the transmission lines are connected to.

        """
        try:
            bus_data = pd.read_csv(filename, header=header)
        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
            return

        for index, row in bus_data.iterrows():
            self.network.add(
                "Bus",
                name=row["bus"],
                v_nom=row["v_nom"],  # Nominal voltage level of the bus(substation)
                carrier=row["carrier"],
                x=row["x"],  # longitude
                y=row["y"],  # lattitude
            )
        return self.network.buses

    def add_generators(self, filename):
        """
        This adds generators in the PyPSA network with respective bus data to which they are connected
        """
        try:
            self.generator_data = pd.read_csv(filename, index_col=0, header=0)
        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
            return

        generators_da = self.generator_data.copy()
        generators_da["p_max_pu"] = generators_da["p_da"] / generators_da["p_nom"]
        generators_da["p_min_pu"] = generators_da["p_da"] / generators_da["p_nom"]
        grouped_da = (
            generators_da.groupby("name").agg(lambda x: x.tolist()).reset_index()
        )

        # Iterate through time steps and add generators
        for index, row in grouped_da.iterrows():
            self.network.add(
                "Generator",
                name=row["name"],
                bus=" ".join(
                    set(row["bus"])
                ),  # bus to which the generator is connected to
                p_set=row["p_da"],  # dayahead cleared capacity
                p_nom=" ".join(
                    map(str, set(row["p_nom"]))
                ),  # Nominal capacity of the powerplant/generator
                marginal_cost=row["marginal_cost"],
                p_nom_extendable=False,
                p_max_pu=row["p_max_pu"],
                p_min_pu=row["p_min_pu"],
                carrier=" ".join(
                    set(row["carrier"])
                ),  # here it is the generator fuel type/technology
            )

        return self.network.generators_t.p_set

    def add_consumers(self, filename, header):
        """
        This adds consumers in the PyPSA network with respective bus data to which they are connected
        """
        try:
            load = pd.read_csv(filename, header=header, index_col=0)
        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
            return

        for load_name in list(load.columns.unique()):
            self.network.add(
                "Load",
                f"{load_name}",
                bus=load_name,
                p_set=load[load_name].tolist(),  # fixed load in MW
            )
        return self.network.loads_t.p_set

    def add_lines(self, filename, header):
        """
        This creates transmission network in PyPSA by connecting buses with predefined line capacities
        """
        try:
            lines = pd.read_csv(filename, header=header)
        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
            return

        for index, row in lines.iterrows():
            self.network.add(
                "Line",
                "{}".format(index),
                bus0=row["bus0"],
                bus1=row["bus1"],
                s_nom=row["s_nom"],  # transmission line capacity in MW
                x=1,
                s_nom_extendable=row["s_nom_extendable"],
            )
        return self.network.lines

    def congestion_identification(self):
        """
        This function identifies congestion in the transmission lines by running simple power flow in PyPSA network
        """
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=DeprecationWarning)

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
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
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


# ---------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    network_clear_instance = Network_clear()

    network_clear_instance.add_buses("buses.csv", 0)
    network_clear_instance.add_generators("generators_t.csv")
    network_clear_instance.add_consumers("consumers_t.csv", 0)
    network_clear_instance.add_lines("lines.csv", 0)

    # Identify congestion
    network_clear_instance.congestion_identification()

    # Run Redispatch
    network_clear_instance.redispatch()
