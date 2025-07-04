# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import numpy as np
import pandas as pd
import pypsa

from assume.common.grid_utils import (
    add_fix_units,
    add_redispatch_generators,
    calculate_network_meta,
    read_pypsa_grid,
)
from assume.common.market_objects import MarketConfig, Orderbook
from assume.common.utils import suppress_output
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)

logging.getLogger("linopy").setLevel(logging.WARNING)
logging.getLogger("pypsa").setLevel(logging.WARNING)


class RedispatchMarketRole(MarketRole):
    """
    A market role that performs redispatch to resolve congestion in the electricity market.
    It uses PyPSA to model the electricity network and perform the redispatch.
    ...
    """

    required_fields = ["node", "max_power", "min_power"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()
        self.network.snapshots = range(marketconfig.market_products[0].count)

        if not self.grid_data:
            logger.error(f"Market '{marketconfig.market_id}': grid_data is missing.")
            raise ValueError("grid_data is missing.")

        read_pypsa_grid(
            network=self.network,
            grid_dict=self.grid_data,
        )
        add_redispatch_generators(
            network=self.network,
            generators=self.grid_data["generators"],
            backup_marginal_cost=marketconfig.param_dict.get(
                "backup_marginal_cost", 10e4
            ),
        )
        add_fix_units(
            network=self.network,
            units=self.grid_data["loads"],
        )

        add_fix_units(
            network=self.network,
            units=self.grid_data["storage_units"],
        )

        add_fix_units(
            network=self.network,
            units=self.grid_data["exchange_units"],
        )

        self.solver = marketconfig.param_dict.get("solver", "highs")
        if self.solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif self.solver == "appsi_highs":
            self.solver_options = {"output_flag": False, "log_to_console": False}
        else:
            self.solver_options = {}

        self.payment_mechanism = marketconfig.param_dict.get(
            "payment_mechanism", "pay_as_bid"
        )
        if self.payment_mechanism not in ["pay_as_bid", "pay_as_clear"]:
            logger.error(
                f"Market '{marketconfig.market_id}': Invalid payment mechanism '{self.payment_mechanism}'."
            )
            raise ValueError("Invalid payment mechanism.")

    def setup(self):
        super().setup()

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict], dict[tuple, float]]:
        if len(orderbook) == 0:
            return super().clear(orderbook, market_products)
        orderbook_df = pd.DataFrame(orderbook)
        orderbook_df["accepted_volume"] = 0.0
        orderbook_df["accepted_price"] = 0.0

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

        p_set = volume_pivot
        p_max_pu_up = (max_power_pivot - volume_pivot).div(
            max_power_pivot.where(max_power_pivot != 0, np.inf)
        )
        p_max_pu_down = (
            (volume_pivot - min_power_pivot)
            .div(max_power_pivot.where(max_power_pivot != 0, np.inf))
            .clip(lower=0)
        )
        costs = price_pivot

        negative_only_units = volume_pivot.lt(0).all()
        p_max_pu_up = p_max_pu_up.drop(
            columns=negative_only_units.index[negative_only_units]
        )
        p_max_pu_down = p_max_pu_down.drop(
            columns=negative_only_units.index[negative_only_units]
        )
        costs = costs.drop(columns=negative_only_units.index[negative_only_units])

        p_set.reset_index(inplace=True, drop=True)
        p_max_pu_up.reset_index(inplace=True, drop=True)
        p_max_pu_down.reset_index(inplace=True, drop=True)
        costs.reset_index(inplace=True, drop=True)

        redispatch_network = self.network.copy()
        redispatch_network.loads_t.p_set = p_set
        redispatch_network.generators_t.p_max_pu.update(p_max_pu_up.add_suffix("_up"))
        redispatch_network.generators_t.p_max_pu.update(
            p_max_pu_down.add_suffix("_down")
        )
        redispatch_network.generators_t.marginal_cost.update(costs.add_suffix("_up"))
        redispatch_network.generators_t.marginal_cost.update(
            costs.add_suffix("_down") * (-1)
        )

        redispatch_network.lpf()
        line_loading = (
            redispatch_network.lines_t.p0.abs() / redispatch_network.lines.s_nom
        )

        line_loading_df = line_loading.reset_index()
        output_file = "outputs/line_loading.csv"

        try:
            existing_df = pd.read_csv(output_file)
            existing_headers = list(existing_df.columns)
            if list(line_loading_df.columns) == existing_headers:
                line_loading_df.to_csv(
                    output_file,
                    mode="a",
                    header=False,
                    index=False,
                    float_format="%.5g",
                )
            else:
                line_loading_df.to_csv(
                    output_file,
                    mode="a",
                    header=True,
                    index=False,
                    float_format="%.5g",
                )
        except FileNotFoundError:
        # If the file doesn't exist, write it with headers
            line_loading_df.to_csv(
                output_file,
                mode="w",
                header=True,
                index=False,
                float_format="%.5g",
            )

        logger.info(f"Line loading data appended to {output_file}")

        if line_loading.max().max() > 1:
            logger.debug("Congestion detected")
            with suppress_output():
                status, termination_condition = redispatch_network.optimize(
                    solver_name=self.solver,
                    solver_options=self.solver_options,
                    progress=False,
                )
            if status != "ok":
                logger.error(f"Solver exited with {termination_condition}")
                raise Exception("Solver in redispatch market did not converge")
        else:
            logger.debug("No congestion detected")

        self.process_dispatch_data(
            network=redispatch_network, orderbook_df=orderbook_df
        )

        accepted_orders = orderbook_df[orderbook_df["accepted_volume"] != 0].to_dict(
            "records"
        )
        rejected_orders = orderbook_df[orderbook_df["accepted_volume"] == 0].to_dict(
            "records"
        )
        meta = []
        flows = []
        for i, product in enumerate(market_products):
            meta.extend(
                calculate_network_meta(network=redispatch_network, product=product, i=i)
            )
        return accepted_orders, rejected_orders, meta, flows

    def process_dispatch_data(self, network: pypsa.Network, orderbook_df: pd.DataFrame):
        generators_t_p = network.generators_t.p.filter(regex="^(?!.*_backup)")
        upward_redispatch = generators_t_p.filter(regex="_up$")
        downward_redispatch = generators_t_p.filter(regex="_down$")
        valid_units = orderbook_df["unit_id"].unique()
        for unit in valid_units:
            mask = orderbook_df["unit_id"] == unit
            if f"{unit}_up" in upward_redispatch.columns:
                orderbook_df.loc[mask, "accepted_volume"] += upward_redispatch[
                    f"{unit}_up"
                ].values
            if f"{unit}_down" in downward_redispatch.columns:
                orderbook_df.loc[mask, "accepted_volume"] -= downward_redispatch[
                    f"{unit}_down"
                ].values
            if self.payment_mechanism == "pay_as_bid":
                orderbook_df.loc[mask, "accepted_price"] = np.where(
                    orderbook_df.loc[mask, "accepted_volume"] > 0,
                    orderbook_df.loc[mask, "price"],
                    np.where(
                        orderbook_df.loc[mask, "accepted_volume"] < 0,
                        orderbook_df.loc[mask, "price"],
                        0,
                    ),
                )
            else:
                nodal_marginal_prices = -network.buses_t.marginal_price
                unit_node = orderbook_df.loc[mask, "node"].values[0]
                orderbook_df.loc[mask, "accepted_price"] = np.where(
                    orderbook_df.loc[mask, "accepted_volume"] != 0,
                    nodal_marginal_prices[unit_node],
                    0,
                )
