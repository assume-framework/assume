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

        if self.grid_data.get("industrial_dsm_units", None) is not None:
            add_redispatch_dsm(
                network=self.network,
                industrial_dsm_units=self.grid_data["industrial_dsm_units"],
            )

        self.solver = marketconfig.param_dict.get("solver", "highs")
        if self.solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif self.solver == "highs":
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

        # -------------------------
        # Helper functions (dtype-safe)
        # -------------------------
        def _to_float_frame(df: pd.DataFrame) -> pd.DataFrame:
            # Ensure purely numeric float64 frame, no objects
            return df.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)

        def _assign_ts(target: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
            # Align index/columns and assign numerically (avoids .update casts)
            cols = list(dict.fromkeys(list(target.columns) + list(src.columns)))
            out = target.reindex(index=src.index, columns=cols, fill_value=0.0).astype("float64")
            out.loc[:, src.columns] = src.values
            return out

        def _safe_series(df, col, default=1.0):
            return df[col] if col in df else pd.Series(default, index=df.index)

        # -------------------------
        # Build orderbook pivots
        # -------------------------
        orderbook_df = pd.DataFrame(orderbook)
        orderbook_df["accepted_volume"] = 0.0
        orderbook_df["accepted_price"] = 0.0

        volume_pivot = orderbook_df.pivot(index="start_time", columns="unit_id", values="volume")
        max_power_pivot = orderbook_df.pivot(index="start_time", columns="unit_id", values="max_power")
        min_power_pivot = orderbook_df.pivot(index="start_time", columns="unit_id", values="min_power")
        price_pivot = orderbook_df.pivot(index="start_time", columns="unit_id", values="price")

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

        # Clean dataframes: enforce float and zero-fill (no object)
        p_set = _to_float_frame(p_set)
        p_max_pu_up = _to_float_frame(p_max_pu_up)
        p_max_pu_down = _to_float_frame(p_max_pu_down)
        costs = _to_float_frame(costs)

        # Remove purely negative units (they cannot ramp up)
        negative_only_units = volume_pivot.lt(0).all()
        drop_cols = list(negative_only_units.index[negative_only_units])
        if drop_cols:
            p_max_pu_up = p_max_pu_up.drop(columns=drop_cols, errors="ignore")
            p_max_pu_down = p_max_pu_down.drop(columns=drop_cols, errors="ignore")
            costs = costs.drop(columns=drop_cols, errors="ignore")

        # Reset indices to 0..T-1 to match network.snapshots
        p_set.reset_index(inplace=True, drop=True)
        p_max_pu_up.reset_index(inplace=True, drop=True)
        p_max_pu_down.reset_index(inplace=True, drop=True)
        costs.reset_index(inplace=True, drop=True)

        # -------------------------
        # Build a working copy of the network & assign time series (dtype-safe)
        # -------------------------
        redispatch_network = self.network.copy()

        # loads p_set
        loads_pset = redispatch_network.loads_t.p_set
        loads_pset = loads_pset.reindex(index=p_set.index, columns=p_set.columns, fill_value=0.0).astype("float64")
        loads_pset.loc[:, :] = p_set.values
        redispatch_network.loads_t.p_set = loads_pset

        # generators p_max_pu up/down and marginal_cost
        g_p_max = redispatch_network.generators_t.p_max_pu.copy()
        g_mc = redispatch_network.generators_t.marginal_cost.copy()

        g_p_max = _assign_ts(g_p_max, p_max_pu_up.add_suffix("_up"))
        g_p_max = _assign_ts(g_p_max, p_max_pu_down.add_suffix("_down"))

        g_mc = _assign_ts(g_mc, costs.add_suffix("_up"))
        g_mc = _assign_ts(g_mc, (costs.add_suffix("_down") * -1.0))

        redispatch_network.generators_t.p_max_pu = g_p_max
        redispatch_network.generators_t.marginal_cost = g_mc

        # -------------------------
        # Quick screening DC load flow
        # -------------------------
        redispatch_network.lpf()


        # Small helper
        def _num(s, default=None):  
            s = pd.to_numeric(s, errors="coerce")
            if default is not None:
                s = s.fillna(default)
            return s.astype(float)

        # ----- Lines (AC) -----
        if not redispatch_network.lines.empty:
            p_lines = redispatch_network.lines_t.p0.abs()

            # capacity limit: prefer s_max_pu if present, else 1.0
            s_nom = _num(redispatch_network.lines["s_nom"], default=0.0)          
            s_max_pu = (
                _num(redispatch_network.lines["s_max_pu"], default=1.0)
                if "s_max_pu" in redispatch_network.lines
                else pd.Series(1.0, index=redispatch_network.lines.index)
            )  

            line_limit = pd.DataFrame(  # broadcast to snapshots; robust to dtype
                np.outer(np.ones(len(redispatch_network.snapshots)), (s_nom * s_max_pu).values),
                index=redispatch_network.snapshots,
                columns=redispatch_network.lines.index,
            )
            line_limit = line_limit.replace(0.0, np.nan)  # avoid div-by-zero

            line_loading = p_lines.divide(line_limit)  # aligned by index/columns

            line_df = (
                line_loading.reset_index()
                .melt(id_vars=["snapshot"], var_name="branch", value_name="loading")
                .assign(
                    type="line",
                    carrier=lambda d: d["branch"].map(redispatch_network.lines["carrier"]),
                    p=lambda d: d.apply(
                        lambda r: abs(redispatch_network.lines_t.p0.loc[r["snapshot"], r["branch"]]),
                        axis=1,
                    ),
                    limit=lambda d: d.apply(
                        lambda r: line_limit.loc[r["snapshot"], r["branch"]],
                        axis=1,
                    ),
                )
            )
        else:
            line_df = pd.DataFrame(columns=["snapshot","branch","type","carrier","p","limit","loading"])

        # ----- Links (HVDC, PST-as-link, etc.) -----
        if not redispatch_network.links.empty:
            p_links = redispatch_network.links_t.p0.abs()

            # static rating p_nom as float
            p_nom = _num(redispatch_network.links["p_nom"], default=0.0)       

            # time-varying p_max_pu(t) as float frame
            if hasattr(redispatch_network, "links_t") and "p_max_pu" in redispatch_network.links_t:
                p_max_pu_t = redispatch_network.links_t.p_max_pu.astype(float)   
            else:
                p_max_pu_t = p_links.copy()
                p_max_pu_t.loc[:, :] = 1.0

            # broadcast p_nom to snapshots via multiplication
            
            # p_links: flows (all zeros before OPF; that's fine)
            p_links = redispatch_network.links_t.p0.abs()

            logger.debug("links_t.p0 columns: %s", list(p_links.columns))
            logger.debug("network.links index: %s", list(redispatch_network.links.index))
            logger.debug("network.links p_nom: %s", redispatch_network.links["p_nom"].to_dict())
            logger.debug("p_nom_2d first row: %s", p_nom_2d.iloc[0].to_dict())
            logger.debug("link_limit first row: %s", link_limit.iloc[0].to_dict())

            # Coerce p_nom and align *exactly* to the columns of p_links
            p_nom = pd.to_numeric(redispatch_network.links["p_nom"], errors="coerce").astype(float)
            p_nom = p_nom.reindex(p_links.columns)  # <- ensure same names/order
            # Turn into a 2D frame by repeating for each snapshot (explicit broadcast)
            p_nom_2d = pd.DataFrame(
                np.tile(p_nom.values, (len(p_links.index), 1)),
                index=p_links.index,
                columns=p_links.columns,
            )

            # Time-varying p_max_pu(t) or 1.0 fallback, and align to p_links columns
            if "p_max_pu" in redispatch_network.links_t:
                p_max_pu_t = redispatch_network.links_t.p_max_pu.astype(float).reindex(columns=p_links.columns)
            else:
                p_max_pu_t = pd.DataFrame(1.0, index=p_links.index, columns=p_links.columns)

            # Final per-snapshot limits (14 if p_nom=14 and p_max_pu=1)
            link_limit = (p_nom_2d * p_max_pu_t)
            # Optional: if you want NaN loading on zero limit, keep next line; otherwise comment it out for debugging
            link_limit = link_limit.replace(0.0, np.nan)

            # Compute loading
            link_loading = p_links.divide(link_limit)

            link_df = (
                link_loading.reset_index()
                .melt(id_vars=["snapshot"], var_name="branch", value_name="loading")
                .assign(
                    type="link",
                    carrier=lambda d: d["branch"].map(
                        redispatch_network.links["carrier"]
                    ) if "carrier" in redispatch_network.links else "link",
                    p=lambda d: d.apply(
                        lambda r: abs(redispatch_network.links_t.p0.loc[r["snapshot"], r["branch"]]),
                        axis=1,
                    ),
                    limit=lambda d: d.apply(
                        lambda r: link_limit.loc[r["snapshot"], r["branch"]],
                        axis=1,
                    ),
                )
            )
        else:
            link_df = pd.DataFrame(columns=["snapshot","branch","type","carrier","p","limit","loading"])

        # -------------------------
        # Combine + flags (explicit dtypes, no object fillna)
        # -------------------------
        pieces = [df for df in (line_df, link_df) if not df.empty]
        if pieces:
            branches_df = pd.concat(pieces, ignore_index=True)
        else:
            branches_df = pd.DataFrame({
                "snapshot": pd.Series(dtype="int64"),
                "branch":   pd.Series(dtype="string"),
                "type":     pd.Series(dtype="string"),
                "carrier":  pd.Series(dtype="string"),
                "p":        pd.Series(dtype="float64"),
                "limit":    pd.Series(dtype="float64"),
                "loading":  pd.Series(dtype="float64"),
            })

        branches_df = branches_df.astype({
            "snapshot": "int64",
            "branch":   "string",
            "type":     "string",
            "carrier":  "string",
            "p":        "float64",
            "limit":    "float64",
            "loading":  "float64",
        })
        branches_df["overloaded"] = branches_df["loading"] > 1.0  # boolean directly, no fillna

        # -------------------------
        # Save/append with stable header
        # -------------------------
        output_file = "outputs/line_loading.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        write_mode = "a" if os.path.exists(output_file) else "w"
        write_header = not os.path.exists(output_file)

        if os.path.exists(output_file):
            try:
                existing_cols = list(pd.read_csv(output_file, nrows=0).columns)
                if existing_cols != list(branches_df.columns):
                    write_mode, write_header = "w", True
            except Exception:
                write_mode, write_header = "w", True

        branches_df.to_csv(
            output_file,
            mode=write_mode,
            header=write_header,
            index=False,
            float_format="%.5g",
        )
        logger.info(f"Branch loading (lines+links) appended to {output_file}")

        # -------------------------
        # Congestion trigger
        # -------------------------
        max_loading = branches_df["loading"].max() if not branches_df.empty else 0.0
        if max_loading > 1.0:
            logger.debug("Congestion detected (lines/links)")
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

        # -------------------------
        # Map redispatch back to orders
        # -------------------------
        self.process_dispatch_data(
            network=redispatch_network, orderbook_df=orderbook_df
        )

        accepted_orders = orderbook_df[orderbook_df["accepted_volume"] != 0].to_dict("records")
        rejected_orders = orderbook_df[orderbook_df["accepted_volume"] == 0].to_dict("records")
        meta = []
        flows = []
        for i, product in enumerate(market_products):
            meta.extend(calculate_network_meta(network=redispatch_network, product=product, i=i))
        return accepted_orders, rejected_orders, meta, flows

    def process_dispatch_data(self, network: pypsa.Network, orderbook_df: pd.DataFrame):
        generators_t_p = network.generators_t.p.filter(regex="^(?!.*_backup)")
        upward_redispatch = generators_t_p.filter(regex="_up$")
        downward_redispatch = generators_t_p.filter(regex="_down$")
        valid_units = orderbook_df["unit_id"].unique()
        for unit in valid_units:
            mask = orderbook_df["unit_id"] == unit
            if f"{unit}_up" in upward_redispatch.columns:
                orderbook_df.loc[mask, "accepted_volume"] += upward_redispatch[f"{unit}_up"].values
            if f"{unit}_down" in downward_redispatch.columns:
                orderbook_df.loc[mask, "accepted_volume"] -= downward_redispatch[f"{unit}_down"].values
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
