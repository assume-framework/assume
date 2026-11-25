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

        # storage units are optional; they participate in redispatch as PyPSA
        # StorageUnits with their own state of charge
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

        # duration of every snapshot (products may differ in length); it weights the
        # costs and the state of charge of the storage units
        snapshot_starts = volume_pivot.index
        ends = orderbook_df.groupby("start_time")["end_time"].max()
        duration_hours = [
            (ends[start] - start).total_seconds() / 3600 for start in snapshot_starts
        ]
        redispatch_network.snapshot_weightings.loc[:, "objective"] = duration_hours
        redispatch_network.snapshot_weightings.loc[:, "stores"] = duration_hours

        # 1. Fixed dispatch of the power plants: a fixed baseline (the cleared
        # dispatch) plus "_up" and "_down" generators
        flex_p_set = p_set[gen_cols].copy()
        redispatch_network.generators_t.p_set = flex_p_set.reindex(
            index=redispatch_network.snapshots, columns=gen_cols, fill_value=0.0
        )
        p_nom = p_nom_pivot[gen_cols].where(p_nom_pivot[gen_cols] != 0, np.inf)
        p_set_pu = flex_p_set.div(p_nom).clip(lower=0, upper=1)
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

        # 3. Redispatch flexibility of the power plants (no ramping constraints yet)
        # up: max_power - cleared dispatch; down: cleared dispatch - min_power
        max_power = max_power_pivot[gen_cols]
        min_power = min_power_pivot[gen_cols]
        p_max_pu_up = (max_power - flex_p_set).div(p_nom).clip(lower=0, upper=1)
        p_max_pu_down = (flex_p_set - min_power).div(p_nom).clip(lower=0, upper=1)
        redispatch_network.generators_t.p_max_pu.update(p_max_pu_up.add_suffix("_up"))
        redispatch_network.generators_t.p_max_pu.update(
            p_max_pu_down.add_suffix("_down")
        )

        # Reducing a power plant's output refunds its costs (-price)
        costs = price_pivot[gen_cols]
        redispatch_network.generators_t.marginal_cost.update(costs.add_suffix("_up"))
        redispatch_network.generators_t.marginal_cost.update(
            (-costs).add_suffix("_down")
        )

        # 4. Storage units: PyPSA StorageUnits around their schedule. p_nom,
        # max_hours (capacity * max_soc) and the efficiencies are already set from
        # the grid data (add_redispatch_storages); the bids add the schedule, the
        # power range, the SOC at the start and the prices. The bid "price" is the
        # discharge price, "price_charge" the charge price (default: "price").
        storage_callback = None
        price_charge_pivot = None
        if len(storage_cols) > 0:
            storage_bids = orderbook_df[orderbook_df["unit_id"].isin(storage_cols)]
            if "soc" not in storage_bids or storage_bids["soc"].isna().any():
                raise ValueError(
                    f"Market '{self.marketconfig.market_id}': every storage bid "
                    "needs 'soc' (see StorageEnergyHeuristicRedispatchStrategy)."
                )
            if "price_charge" not in orderbook_df:
                orderbook_df["price_charge"] = np.nan
            # a strategy that sends one price uses it for both directions
            orderbook_df["price_charge"] = orderbook_df["price_charge"].fillna(
                orderbook_df["price"]
            )
            soc_pivot = orderbook_df.pivot(
                index="start_time", columns="unit_id", values="soc"
            ).reset_index(drop=True)
            price_charge_pivot = orderbook_df.pivot(
                index="start_time", columns="unit_id", values="price_charge"
            ).reset_index(drop=True)[storage_cols]

            schedule = p_set[storage_cols].fillna(0.0)
            storage_p_nom = redispatch_network.storage_units.loc[storage_cols, "p_nom"]
            redispatch_network.storage_units_t.p_set = schedule
            redispatch_network.storage_units_t.p_max_pu = (
                max_power_pivot[storage_cols].div(storage_p_nom).clip(lower=0)
            )
            redispatch_network.storage_units_t.p_min_pu = (
                min_power_pivot[storage_cols].div(storage_p_nom).clip(upper=0)
            )

            capacity = storage_units.loc[storage_cols, "capacity"]
            min_soc = (
                storage_units.loc[storage_cols, "min_soc"].fillna(0.0)
                if "min_soc" in storage_units
                else 0.0
            )
            redispatch_network.storage_units.loc[
                storage_cols, "state_of_charge_initial"
            ] = soc_pivot[storage_cols].iloc[0] * capacity

            storage_callback = Storage.redispatch_storage_callback(
                storage_ids=storage_cols,
                min_energy=(capacity * min_soc).to_dict(),
                price_discharge=price_pivot[storage_cols],
                price_charge=price_charge_pivot,
                scheduled_dispatch=schedule.clip(lower=0),
                scheduled_store=(-schedule).clip(lower=0),
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

            # the storage schedule is only needed for the power flow check: a
            # StorageUnit p_set would fix the storage in the optimisation
            storage_schedule = redispatch_network.storage_units_t.p_set.copy()
            redispatch_network.storage_units_t.p_set = pd.DataFrame(
                index=redispatch_network.snapshots
            )

            status, termination_condition = redispatch_network.optimize(
                solver_name=self.solver_name,
                log_to_console=False,
                # do not show tqdm progress bars for large grids
                # https://github.com/PyPSA/linopy/pull/375
                progress=False,
                extra_functionality=storage_callback,
                # Constant objective terms do not affect dispatch or nodal prices; omit for speed/stability.
                include_objective_constant=False,
            )

            if status != "ok":
                logger.error(f"Solver exited with {termination_condition}")
                raise Exception("Solver in redispatch market did not converge")

            # restore the schedule: redispatch = dispatch - schedule
            redispatch_network.storage_units_t.p_set = storage_schedule

        # if no congestion is detected set accepted volume and price to 0
        else:
            logger.debug("No congestion detected")

        # process dispatch data
        self.process_dispatch_data(
            network=redispatch_network,
            orderbook_df=orderbook_df,
            storage_cols=storage_cols,
            storage_price_discharge=price_pivot[storage_cols],
            storage_price_charge=price_charge_pivot,
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

    def process_dispatch_data(
        self,
        network: pypsa.Network,
        orderbook_df: pd.DataFrame,
        storage_cols: pd.Index | None = None,
        storage_price_discharge: pd.DataFrame | None = None,
        storage_price_charge: pd.DataFrame | None = None,
    ):
        """
        This function processes the dispatch data to calculate the redispatch volumes and prices
        and update the orderbook with the accepted volumes and prices.

        Power plants: the volume comes from their ``_up`` / ``_down`` generators
        and is settled at the bid price.

        Storage units: the volume is the dispatch after the clearing minus the
        schedule (``storage_units_t.p - p_set``, 0 if no optimisation ran). With
        ``pay_as_bid`` discharging is settled at the discharge price and charging
        at the charge price::

            cashflow = price_discharge * (p_dispatch - scheduled discharge)
                       - price_charge * (p_store - scheduled charge)

        and ``accepted_price`` = cashflow / ``accepted_volume``, so that the usual
        cashflow ``accepted_price * accepted_volume`` stays correct. With
        ``pay_as_clear`` storage units get the nodal price like power plants.

        Args:
            network (pypsa.Network): The solved redispatch network.
            orderbook_df (pd.DataFrame): The orderbook to be cleared.
            storage_cols (pd.Index, optional): The storage units.
            storage_price_discharge (pd.DataFrame, optional): Discharge price per
                snapshot and storage.
            storage_price_charge (pd.DataFrame, optional): Charge price per
                snapshot and storage.
        """

        # Get all generators except for _backup generators
        generators_t_p = network.generators_t.p.filter(regex="^(?!.*_backup)")

        # Use regex in a single call to filter and rename columns simultaneously for efficiency
        upward_redispatch = generators_t_p.filter(regex="_up$")
        downward_redispatch = generators_t_p.filter(regex="_down$")

        # storage: redispatch = dispatch after the clearing - schedule
        if storage_cols is None:
            storage_cols = pd.Index([])
        if len(storage_cols) > 0:
            schedule = network.storage_units_t.p_set[storage_cols]
            storage_volume = (network.storage_units_t.p[storage_cols] - schedule).round(
                6
            )
            if network.storage_units_t.p_dispatch.empty:  # no optimisation ran
                storage_cashflow = 0.0 * storage_volume
            else:
                dispatch_change = network.storage_units_t.p_dispatch[
                    storage_cols
                ] - schedule.clip(lower=0)
                store_change = network.storage_units_t.p_store[storage_cols] - (
                    -schedule
                ).clip(lower=0)
                storage_cashflow = (
                    storage_price_discharge.set_axis(schedule.index) * dispatch_change
                    - storage_price_charge.set_axis(schedule.index) * store_change
                )
            # snapshot of every order (the snapshots are the sorted start times)
            snapshot_of = {
                t: i for i, t in enumerate(sorted(orderbook_df["start_time"].unique()))
            }

        # Find intersection of unit_ids in orderbook_df and columns in redispatch_volumes for direct mapping
        valid_units = orderbook_df["unit_id"].unique()

        for unit in valid_units:
            unit_orders = orderbook_df["unit_id"] == unit

            if unit in storage_cols:
                hours = orderbook_df.loc[unit_orders, "start_time"].map(snapshot_of)
                orderbook_df.loc[unit_orders, "accepted_volume"] = storage_volume[
                    unit
                ].to_numpy()[hours]

            if f"{unit}_up" in upward_redispatch.columns:
                orderbook_df.loc[unit_orders, "accepted_volume"] += upward_redispatch[
                    f"{unit}_up"
                ].values

            if f"{unit}_down" in downward_redispatch.columns:
                orderbook_df.loc[unit_orders, "accepted_volume"] -= downward_redispatch[
                    f"{unit}_down"
                ].values

            if self.payment_mechanism == "pay_as_bid" and unit in storage_cols:
                # each move at its own price: accepted_price = cashflow / volume
                volume = orderbook_df.loc[unit_orders, "accepted_volume"].to_numpy()
                cashflow = storage_cashflow[unit].to_numpy()[hours]
                orderbook_df.loc[unit_orders, "accepted_price"] = np.divide(
                    cashflow, volume, out=np.zeros_like(volume), where=volume != 0
                )

            elif self.payment_mechanism == "pay_as_bid":
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
