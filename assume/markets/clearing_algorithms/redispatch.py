# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import pypsa

from assume.common.grid_utils import (
    add_redispatch_generators,
    add_redispatch_loads,
    add_redispatch_storages,
    calculate_network_meta,
    get_supported_solver_linopy,
    read_pypsa_grid,
)
from assume.common.market_objects import MarketConfig, Orderbook
from assume.markets.base_market import MarketRole
from assume.units.storage import Storage

logger = logging.getLogger(__name__)

logging.getLogger("linopy").setLevel(logging.WARNING)
logging.getLogger("pypsa").setLevel(logging.WARNING)


class RedispatchMarketRole(MarketRole):
    """
    A market role that performs redispatch to resolve congestion in the electricity market.
    It uses PyPSA to model the electricity network and perform the redispatch.
    The redispatched is based on the price the units submit in their orders.
    This allows this to be a cost based redispatch if units submit their marginal costs as prices.
    Or it can be a price based redispatch if units submit actual bid prices.

    Args:
        marketconfig (MarketConfig): The market configuration.

    Note:
        Users can also configure the path to the network data, the name of the solver to be used,
        and the backup marginal cost in the param_dict of the market configuration.

    """

    required_fields = ["node", "max_power", "min_power"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.network = pypsa.Network()
        # set snapshots as list from the value marketconfig.producs.count converted to list
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
        add_redispatch_loads(
            network=self.network,
            loads=self.grid_data["loads"],
        )

        # storage units are optional; they participate in redispatch as
        # bidirectional generators (fixed baseline + up + down)
        storage_units = self.grid_data.get("storage_units")
        if storage_units is not None and not storage_units.empty:
            add_redispatch_storages(
                network=self.network,
                storages=storage_units,
            )

        self.solver_name = get_supported_solver_linopy(
            marketconfig.param_dict.get("solver_name", "highs")
        )

        self.log_flows = self.marketconfig.param_dict.get("log_flows", False)

        # set the market clearing principle
        # as pay as bid or pay as clear
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

        if len(orderbook) == 0:
            return super().clear(orderbook, market_products)
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
        p_nom_pivot = orderbook_df.pivot(
            index="start_time", columns="unit_id", values="p_nom"
        )

        # Update the network parameters
        redispatch_network = self.network.copy()

        # Final market dispatch from orderbook
        p_set = volume_pivot.copy()

        # Reset all pivot indices to match redispatch_network.snapshots = range(...)
        p_set.reset_index(inplace=True, drop=True)
        max_power_pivot.reset_index(inplace=True, drop=True)
        min_power_pivot.reset_index(inplace=True, drop=True)
        price_pivot.reset_index(inplace=True, drop=True)
        p_nom_pivot.reset_index(inplace=True, drop=True)

        # Generators, loads and storage units from the grid data
        generator_names = self.grid_data["generators"].index
        load_names = self.grid_data["loads"].index
        storage_units = self.grid_data.get("storage_units")
        storage_names = (
            storage_units.index if storage_units is not None else pd.Index([])
        )

        gen_cols = p_set.columns.intersection(generator_names)
        load_cols = p_set.columns.intersection(load_names)
        storage_cols = p_set.columns.intersection(storage_names)

        # Power plants and storage units are redispatched the same way: a fixed
        # baseline (the cleared dispatch) plus "_up" and "_down" generators. Storage
        # differs in only three points, marked "storage:" below.
        flex_cols = gen_cols.union(storage_cols)
        is_storage = pd.Series(flex_cols.isin(storage_cols), index=flex_cols)

        # 1. Fixed dispatch
        flex_p_set = p_set[flex_cols].copy()
        redispatch_network.generators_t.p_set = flex_p_set.reindex(
            index=redispatch_network.snapshots, columns=flex_cols, fill_value=0.0
        )
        p_nom = p_nom_pivot[flex_cols].where(p_nom_pivot[flex_cols] != 0, np.inf)
        # storage: the baseline may be negative (charging)
        lower = is_storage.map({True: -1.0, False: 0.0})
        p_set_pu = flex_p_set.div(p_nom).clip(lower=lower, upper=1, axis=1)
        redispatch_network.generators_t.p_min_pu.update(p_set_pu)
        redispatch_network.generators_t.p_max_pu.update(p_set_pu)

        # 2. Fixed demand/load profile
        # PyPSA p_set values to positive values since the sign of the p_set values is usually positive
        load_p_set = p_set[load_cols].copy().abs()

        redispatch_network.loads_t.p_set = load_p_set.reindex(
            index=redispatch_network.snapshots,
            columns=redispatch_network.loads.index,
            fill_value=0.0,
        )

        # 3. Energy information of storage units (sent in their bids). A storage
        # without it could be over-committed, so it gets no redispatch head-room.
        storage_bids = {
            field: orderbook_df.pivot(
                index="start_time", columns="unit_id", values=field
            )
            .reset_index(drop=True)
            .reindex(columns=storage_cols)
            if field in orderbook_df
            else pd.DataFrame(np.nan, index=p_set.index, columns=storage_cols)
            for field in [
                "energy_discharge_headroom",
                "energy_charge_headroom",
                "efficiency_charge",
                "efficiency_discharge",
            ]
        }
        incomplete = pd.concat(storage_bids.values()).isna().any()
        blocked = storage_cols[incomplete.to_numpy()]
        if len(blocked) > 0:
            logger.warning(
                "Market '%s': storage %s sent no energy head-room or efficiencies; "
                "its redispatch is blocked.",
                self.marketconfig.market_id,
                list(blocked),
            )
        max_power = max_power_pivot[flex_cols].copy()
        min_power = min_power_pivot[flex_cols].copy()
        max_power[blocked] = min_power[blocked] = flex_p_set[blocked]

        # 4. Redispatch flexibility (no ramping constraints yet)
        # up: max_power - cleared dispatch; down: cleared dispatch - min_power
        p_max_pu_up = (max_power - flex_p_set).div(p_nom).clip(lower=0, upper=1)
        p_max_pu_down = (flex_p_set - min_power).div(p_nom).clip(lower=0, upper=1)
        redispatch_network.generators_t.p_max_pu.update(p_max_pu_up.add_suffix("_up"))
        redispatch_network.generators_t.p_max_pu.update(
            p_max_pu_down.add_suffix("_down")
        )

        # Reducing a power plant's output refunds its costs (-price). storage:
        # charging more / discharging less saves nothing, so it costs 0 (a negative
        # price would let the optimiser charge storage purely for profit).
        costs = price_pivot[flex_cols]
        down_costs = -costs
        down_costs[storage_cols] = 0.0
        redispatch_network.generators_t.marginal_cost.update(costs.add_suffix("_up"))
        redispatch_network.generators_t.marginal_cost.update(
            down_costs.add_suffix("_down")
        )

        # 5. storage: the redispatch energy over all snapshots may not exceed what
        # the storage holds; the constraint is built by the Storage unit
        dt = (orderbook_df["end_time"] - orderbook_df["start_time"]).iloc[0]
        extra_functionality = Storage.redispatch_soc_callback(
            storage_ids=storage_cols.difference(blocked),
            storage_bids=storage_bids,
            duration_hours=dt.total_seconds() / 3600,
        )

        # run linear powerflow
        redispatch_network.lpf()

        # check lines for congestion where power flow is larger than s_nom * s_max_pu
        line_loading = redispatch_network.lines_t.p0.abs() / (
            redispatch_network.lines.s_nom * redispatch_network.lines.s_max_pu
        )

        # if any line is congested, perform redispatch
        if line_loading.max().max() > 1:
            logger.debug("Congestion detected")

            status, termination_condition = redispatch_network.optimize(
                solver_name=self.solver_name,
                log_to_console=False,
                # do not show tqdm progress bars for large grids
                # https://github.com/PyPSA/linopy/pull/375
                progress=False,
                extra_functionality=extra_functionality,
                # Constant objective terms do not affect dispatch or nodal prices; omit for speed/stability.
                include_objective_constant=False,
            )

            if status != "ok":
                logger.error(f"Solver exited with {termination_condition}")
                raise Exception("Solver in redispatch market did not converge")

        # if no congestion is detected set accepted volume and price to 0
        else:
            logger.debug("No congestion detected")

        # process dispatch data
        self.process_dispatch_data(
            network=redispatch_network, orderbook_df=orderbook_df
        )

        # return orderbook_df back to orderbook format as list of dicts
        accepted_orders = orderbook_df[orderbook_df["accepted_volume"] != 0].to_dict(
            "records"
        )
        rejected_orders = orderbook_df[orderbook_df["accepted_volume"] == 0].to_dict(
            "records"
        )
        meta = []

        # calculate meta data such as total upwared and downward redispatch, total backup dispatch
        # and total redispatch cost
        for i, product in enumerate(market_products):
            meta.extend(
                calculate_network_meta(network=redispatch_network, product=product, i=i)
            )

        flows = {}
        if self.log_flows:
            # extract flows
            flows = redispatch_network.lines_t.p0.stack(future_stack=True).to_dict()

        return accepted_orders, rejected_orders, meta, flows

    def process_dispatch_data(self, network: pypsa.Network, orderbook_df: pd.DataFrame):
        """
        This function processes the dispatch data to calculate the redispatch volumes and prices
        and update the orderbook with the accepted volumes and prices.

        Args:
            orderbook_df (pd.DataFrame): The orderbook to be cleared.
        """

        # Get all generators except for _backup generators
        generators_t_p = network.generators_t.p.filter(regex="^(?!.*_backup)")

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
