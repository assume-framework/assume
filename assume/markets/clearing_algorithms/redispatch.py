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
    add_redispatch_dsm,
    add_redispatch_generators,
    add_redispatch_storage_units,
    calculate_network_meta,
    read_pypsa_grid,
)
from assume.common.market_objects import MarketConfig, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)

logging.getLogger("linopy").setLevel(logging.WARNING)
logging.getLogger("pypsa").setLevel(logging.WARNING)


class RedispatchMarketRole(MarketRole):
    """
    A market role that performs redispatch to resolve congestion in the electricity market.
    It uses PyPSA to model the electricity network and perform the redispatch.

    required_fields:
        - node
        - max_power
        - min_power
    """

    required_fields = ["node", "max_power", "min_power"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()
        # number of snapshots in the redispatch horizon
        self.network.snapshots = range(marketconfig.market_products[0].count)

        if not self.grid_data:
            logger.error(f"Market '{marketconfig.market_id}': grid_data is missing.")
            raise ValueError("grid_data is missing.")

        self._rd_snapshots = []
        self._rd_generators_p = []
        self._rd_lines_p0 = []
        self._rd_buses_mp = []

        # base network (lines, buses, etc.)
        read_pypsa_grid(
            network=self.network,
            grid_dict=self.grid_data,
        )

        # redispatchable conventional generators
        add_redispatch_generators(
            network=self.network,
            generators=self.grid_data["generators"],
            backup_marginal_cost=marketconfig.param_dict.get(
                "backup_marginal_cost", 10e4
            ),
        )

        # fixed loads (non-redispatch)
        add_fix_units(
            network=self.network,
            units=self.grid_data["loads"],
        )

        # redispatchable storage units (batteries, PSPP, etc.)
        add_redispatch_storage_units(
            network=self.network,
            storage_units=self.grid_data["storage_units"],
        )

        # cross-border / exchange units as fixed loads
        add_fix_units(
            network=self.network,
            units=self.grid_data["exchange_units"],
        )

        # optional industrial DSM
        if self.grid_data.get("industrial_dsm_units", None) is not None:
            add_redispatch_dsm(
                network=self.network,
                industrial_dsm_units=self.grid_data["industrial_dsm_units"],
            )

        # Build a set of renewable unit IDs (match your labels exactly) to handle no-upward-redispatch
        res_gen = {"solar pv", "wind onshore", "wind offshore", "hydro", "biomass"}

        generators_df = self.grid_data.get("generators")
        if generators_df is None:
            self.res_unit_ids = set()
        else:
            # IMPORTANT: confirm the column names in generators_df
            tech_col = "technology" if "technology" in generators_df.columns else None
            if tech_col is None:
                self.res_unit_ids = set()
            else:
                self.res_unit_ids = set(
                    generators_df.index[generators_df[tech_col].isin(res_gen)].tolist()
                )

        # solver selection
        self.solver = marketconfig.param_dict.get("solver", "highs")
        if self.solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif self.solver == "highs":
            self.solver_options = {"output_flag": False, "log_to_console": False}
        else:
            self.solver_options = {}

        # payment mechanism
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
        """
        Clear the redispatch market:

        - Build time-series pivots from the orderbook
        - Translate them into PyPSA p_set / p_max_pu / costs
        - Run DC load flow and, if congested, an OPF
        - Map redispatch back to accepted volumes & prices
        """
        if len(orderbook) == 0:
            return super().clear(orderbook, market_products)

        orderbook_df = pd.DataFrame(orderbook)
        orderbook_df["accepted_volume"] = 0.0
        orderbook_df["accepted_price"] = 0.0

        # --- 1) build pivots -------------------------------------------------
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

        # fill NaNs: if a unit does not bid at a timestep, treat as 0 capacity / 0 volume / 0 cost
        volume_pivot = volume_pivot.fillna(0.0)
        max_power_pivot = max_power_pivot.fillna(0.0)
        min_power_pivot = min_power_pivot.fillna(0.0)
        price_pivot = price_pivot.fillna(0.0)

        # No upward redispatch for RES: force max_power == volume for RES units
        res_cols = [u for u in volume_pivot.columns if u in self.res_unit_ids]
        if res_cols:
            max_power_pivot.loc[:, res_cols] = volume_pivot.loc[:, res_cols]

        p_set = volume_pivot

        # denominator for per-unit capacities (avoid division by zero)
        denom = max_power_pivot.where(max_power_pivot != 0, np.inf)

        # Upward headroom per-unit: (max - schedule) / p_nom  ∈ [0, 1]
        p_max_pu_up = ((max_power_pivot - volume_pivot).div(denom)).clip(
            lower=0.0, upper=1.0
        )

        # Downward headroom per-unit: (schedule - min) / p_nom  ∈ [0, 1]
        p_max_pu_down = ((volume_pivot - min_power_pivot).div(denom)).clip(
            lower=0.0, upper=1.0
        )

        costs = price_pivot

        # remove units that can *only* move in negative direction (pure downward redispatch)
        # from the upward side (and vice versa, as in original logic)
        negative_only_units = volume_pivot.lt(0).all()
        p_max_pu_up = p_max_pu_up.drop(
            columns=negative_only_units.index[negative_only_units]
        )
        p_max_pu_down = p_max_pu_down.drop(
            columns=negative_only_units.index[negative_only_units]
        )
        costs = costs.drop(columns=negative_only_units.index[negative_only_units])

        # reset time index to 0..T-1 for PyPSA snapshots
        p_set.reset_index(inplace=True, drop=True)
        p_max_pu_up.reset_index(inplace=True, drop=True)
        p_max_pu_down.reset_index(inplace=True, drop=True)
        costs.reset_index(inplace=True, drop=True)

        # --- 2) build a working copy of the network --------------------------
        redispatch_network = self.network.copy()
        
        # align network snapshots to the redispatch horizon (same as p_set index)
        redispatch_network.set_snapshots(p_set.index)
        
        # Storage + loads + exchange units ALL must appear in p_set
        all_loads = redispatch_network.loads.index

        # Reindex p_set to match load names & snapshots; missing loads get 0 base dispatch
        p_set_aligned = (
            p_set
            .reindex(index=redispatch_network.snapshots, columns=all_loads, fill_value=0.0)
            .astype(float)
        )

        redispatch_network.loads_t.p_set = p_set_aligned

        #redispatch_network.loads_t.p_set = p_set

        # Update p_max_pu for generators with _up and _down suffixes
        redispatch_network.generators_t.p_max_pu.update(p_max_pu_up.add_suffix("_up"))
        redispatch_network.generators_t.p_max_pu.update(
            p_max_pu_down.add_suffix("_down")
        )

        # Add _up and _down suffix to costs and update the network
        redispatch_network.generators_t.marginal_cost.update(costs.add_suffix("_up"))
        redispatch_network.generators_t.marginal_cost.update(
            costs.add_suffix("_down") * (-1)
        )

        # run linear powerflow
        redispatch_network.lpf()

        # check lines for congestion where power flow is larger than s_nom
        line_loading = (
            redispatch_network.lines_t.p0.abs() / redispatch_network.lines.s_nom
        )

        # store line loading to CSV (append if exists)
        line_loading_df = line_loading.reset_index()
        output_file = "outputs/line_loading.csv"
        output_dir = os.path.dirname(output_file)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            existing_df = pd.read_csv(output_file)
            existing_headers = list(existing_df.columns)
            if list(line_loading_df.columns) == existing_headers:
                # same header -> append without header
                line_loading_df.to_csv(
                    output_file,
                    mode="a",
                    header=False,
                    index=False,
                    float_format="%.5g",
                )
            else:
                # columns changed -> write header again
                line_loading_df.to_csv(
                    output_file,
                    mode="a",
                    header=True,
                    index=False,
                    float_format="%.5g",
                )
        except FileNotFoundError:
            # first write with header
            line_loading_df.to_csv(
                output_file,
                mode="w",
                header=True,
                index=False,
                float_format="%.5g",
            )

        logger.info(f"Line loading data appended to {output_file}")

        # if no line exceeds 100% loading, skip optimization
        if line_loading.max().max() > 1:
            logger.debug("Congestion detected")

            # optional sanity check: no generator should have p_max_pu < p_min_pu
            if (
                redispatch_network.generators_t.p_max_pu
                < redispatch_network.generators_t.p_min_pu
            ).any().any():
                logger.error("Some generators have p_max_pu < p_min_pu; check redispatch bounds.")
                raise ValueError(
                    "Invalid generator bounds: p_max_pu < p_min_pu for some units."
                )

            status, termination_condition = redispatch_network.optimize(
                solver_name=self.solver,
                solver_options=self.solver_options,
                # do not show tqdm progress bars for large grids
                # https://github.com/PyPSA/linopy/pull/375
                progress=False,
            )

            if status != "ok":
                logger.error(f"Solver exited with {termination_condition}")
                raise Exception("Solver in redispatch market did not converge")
            
            self._rd_snapshots.append(pd.Index(redispatch_network.snapshots))
            self._rd_generators_p.append(redispatch_network.generators_t.p.copy())
            self._rd_lines_p0.append(redispatch_network.lines_t.p0.copy())
            if hasattr(redispatch_network, "buses_t") and hasattr(redispatch_network.buses_t, "marginal_price"):
                self._rd_buses_mp.append(redispatch_network.buses_t.marginal_price.copy())

        else:
            logger.debug("No congestion detected")

        # --- 4) map redispatch volumes & prices back to orderbook -----------
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
        """
        Process dispatch data to calculate the redispatch volumes and prices
        and update the orderbook with the accepted volumes and prices.
        """

        # Get all generators except for _backup generators
        generators_t_p = network.generators_t.p.filter(regex="^(?!.*_backup)")

        # Extract up/down redispatch columns
        upward_redispatch = generators_t_p.filter(regex="_up$")
        downward_redispatch = generators_t_p.filter(regex="_down$")

        valid_units = orderbook_df["unit_id"].unique()

        # Only compute nodal marginal prices if needed
        nodal_marginal_prices = None
        if self.payment_mechanism == "pay_as_clear":
            # PyPSA marginal_price is negative for costs; keep the old sign convention
            nodal_marginal_prices = -network.buses_t.marginal_price

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

            if self.payment_mechanism == "pay_as_bid":
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

            elif self.payment_mechanism == "pay_as_clear":
                # set accepted price as the nodal marginal price
                nodal_marginal_prices = -network.buses_t.marginal_price
                unit_node = orderbook_df.loc[unit_orders, "node"].values[0]

                orderbook_df.loc[unit_orders, "accepted_price"] = np.where(
                    orderbook_df.loc[unit_orders, "accepted_volume"] != 0,
                    nodal_marginal_prices[unit_node],
                    0,
                )
    async def on_stop(self):
        await super().on_stop()
        self.export_full_redispatch_netcdf()

    def export_full_redispatch_netcdf(self):
        if not self._rd_generators_p:
            return

        out_dir = self.marketconfig.param_dict.get("export_dir", "outputs/pypsa_nc")
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, f"redispatch_full_{self.marketconfig.market_id}.nc")

        n = self.network.copy()

        # concatenate along time
        gen_p = pd.concat(self._rd_generators_p, axis=0)
        line_p0 = pd.concat(self._rd_lines_p0, axis=0)

        # ensure snapshots match concatenated time length
        n.set_snapshots(range(len(gen_p.index)))

        # assign stitched results (these are outputs, not inputs)
        n.generators_t.p = gen_p.reset_index(drop=True)
        n.lines_t.p0 = line_p0.reset_index(drop=True)

        if self._rd_buses_mp:
            mp = pd.concat(self._rd_buses_mp, axis=0)
            n.buses_t.marginal_price = mp.reset_index(drop=True)
        
        logger.info(f"Redispatch NetCDF will be written to: {fn}")

        n.export_to_netcdf(fn)
        logger.info(f"Exported full redispatch network to {fn}")

