# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from assume.common.base import SupportsMinMaxCharge
from assume.common.exceptions import ValidationError
from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig, Orderbook

logger = logging.getLogger(__name__)
EPS = 1e-4


class Storage(SupportsMinMaxCharge):
    """
    A class for a storage unit.

    Args:
        id (str): The ID of the storage unit.
        technology (str): The technology of the storage unit.
        node (str): The node of the storage unit.
        max_power_charge (float): The maximum power input of the storage unit in MW (negative value).
        min_power_charge (float): The minimum power input of the storage unit in MW (negative value).
        max_power_discharge (float): The maximum power output of the storage unit in MW.
        min_power_discharge (float): The minimum power output of the storage unit in MW.
        capacity (float): The capacity of the storage unit in MWh.
        max_soc (float): The maximum state of charge of the storage unit as a fraction (typically 1).
        min_soc (float): The minimum state of charge of the storage unit as a fraction (typically 0).
        initial_soc (float): The initial state of charge of the storage unit as a fraction (between 0 and 1).
        efficiency_charge (float): The efficiency of the storage unit while charging.
        efficiency_discharge (float): The efficiency of the storage unit while discharging.
        additional_cost_charge (float, optional): Additional costs associated with power consumption, in EUR/MWh. Defaults to 0.
        additional_cost_discharge (float, optional): Additional costs associated with power generation, in EUR/MWh. Defaults to 0.
        ramp_up_charge (float): The ramp up rate of charging the storage unit in MW/15 minutes (negative value).
        ramp_down_charge (float): The ramp down rate of charging the storage unit in MW/15 minutes (negative value).
        ramp_up_discharge (float): The ramp up rate of discharging the storage unit in MW/15 minutes.
        ramp_down_discharge (float): The ramp down rate of discharging the storage unit in MW/15 minutes.
        hot_start_cost (float): The hot start cost of the storage unit in €/MW.
        warm_start_cost (float): The warm start cost of the storage unit in €/MW.
        cold_start_cost (float): The cold start cost of the storage unit in €/MW.
        downtime_hot_start (float): Definition of downtime before hot start in h.
        downtime_warm_start (float): Definition of downtime before warm start in h.
        min_operating_time (float): The minimum operating time of the storage unit in hours.
        min_down_time (float): The minimum down time of the storage unit in hours.
        is_active (bool): Defines whether or not the unit bids itself or is portfolio optimized.
        bidding_startegy (str): In case the unit is active it has to be defined which bidding strategy should be used

    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        forecaster: UnitForecaster,
        max_power_charge: float,
        max_power_discharge: float,
        capacity: float,
        min_power_charge: float = 0.0,
        min_power_discharge: float = 0.0,
        min_soc: float = 0.0,
        max_soc: float = 1.0,
        initial_soc: float | None = 0.0,
        soc_tick: float = 0.01,
        efficiency_charge: float = 1,
        efficiency_discharge: float = 1,
        additional_cost_charge: float = 0.0,
        additional_cost_discharge: float = 0.0,
        ramp_up_charge: float | None = None,
        ramp_down_charge: float | None = None,
        ramp_up_discharge: float | None = None,
        ramp_down_discharge: float | None = None,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,  # hours
        min_down_time: float = 0,  # hours
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        location: tuple[float, float] = (0, 0),
        node: str = "node0",
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )
        if capacity < 0:
            raise ValidationError(
                message=f"{capacity=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="",
            )
        if not 0 <= min_soc <= 1:
            raise ValidationError(
                message=f"{min_soc=} must be between 0 and 1 for unit {self.id}",
                id=self.id,
                field="",
            )
        if not 0 <= max_soc <= 1:
            raise ValidationError(
                message=f"{max_soc=} must be between 0 and 1 for unit {self.id}",
                id=self.id,
                field="",
            )
        if max_soc < min_soc:
            raise ValidationError(
                message=f"{max_soc=} must be >= {min_soc=} for unit {self.id}",
                id=self.id,
                field="",
            )
        self.capacity = capacity
        self.max_soc = max_soc
        self.min_soc = min_soc
        if initial_soc is None:
            initial_soc = 0.5
        if not 0 <= initial_soc <= 1:
            raise ValidationError(
                f"{initial_soc=} must be between 0 and 1 for unit {self.id}",
                id=self.id,
                field="initial_soc",
            )
        self.initial_soc = initial_soc

        if max_power_charge > 0:
            raise ValidationError(
                message=f"{max_power_charge=} must be <= 0 for unit {self.id}",
                id=self.id,
                field="max_power_charge",
            )
        if min_power_charge > 0:
            raise ValidationError(
                message=f"{min_power_charge=} must be <= 0 for unit {self.id}",
                id=self.id,
                field="min_power_charge",
            )
        if max_power_charge > min_power_charge:
            raise ValidationError(
                f"{max_power_charge=} must be <= {min_power_charge=} for unit {self.id}",
                id=self.id,
                field="max_power_charge",
            )
        if max_power_discharge < 0:
            raise ValidationError(
                message=f"{max_power_discharge=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="max_power_discharge",
            )
        if min_power_discharge < 0:
            raise ValidationError(
                message=f"{min_power_discharge=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="min_power_discharge",
            )
        if max_power_discharge < min_power_discharge:
            raise ValueError(
                f"{max_power_discharge=} must be >= {min_power_discharge=} for unit {self.id}"
            )
        self.max_power_charge = max_power_charge
        self.min_power_charge = min_power_charge
        self.max_power_discharge = max_power_discharge
        self.min_power_discharge = min_power_discharge

        self.outputs["soc"] = FastSeries(value=self.initial_soc, index=self.index)
        self.outputs["cost_stored_energy"] = FastSeries(value=0.0, index=self.index)

        if soc_tick < 0:
            raise ValidationError(
                message=f"{soc_tick=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="soc_tick",
            )
        self.soc_tick = soc_tick

        if not 0 <= efficiency_charge <= 1:
            raise ValidationError(
                f"{efficiency_charge=} must be between 0 and 1 for unit {self.id}",
                id=self.id,
                field="efficiency_charge",
            )
        if not 0 <= efficiency_discharge <= 1:
            raise ValidationError(
                f"{efficiency_discharge=} must be between 0 and 1 for unit {self.id}",
                id=self.id,
                field="efficiency_discharge",
            )
        # The efficiency of the storage unit while charging.
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge

        # The variable costs to charge/discharge the storage unit.
        self.additional_cost_charge = additional_cost_charge
        self.additional_cost_discharge = additional_cost_discharge

        # The ramp up/down rate of charging/discharging the storage unit.
        # if ramp_up_charge == 0, the ramp_up_charge is set to enable ramping between full charge and discharge power
        # else the ramp_up_charge is set to the negative value of the ramp_up_charge
        if ramp_up_charge is not None and ramp_up_charge > 0:
            raise ValidationError(
                message=f"{ramp_up_charge=} must be <= 0 for unit {self.id}",
                id=self.id,
                field="ramp_up_charge",
            )
        if ramp_down_charge is not None and ramp_down_charge > 0:
            raise ValidationError(
                message=f"{ramp_down_charge=} must be <= 0 for unit {self.id}",
                id=self.id,
                field="ramp_down_charge",
            )
        if ramp_up_discharge is not None and ramp_up_discharge < 0:
            raise ValidationError(
                message=f"{ramp_up_discharge=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="ramp_up_discharge",
            )
        if ramp_down_discharge is not None and ramp_down_discharge < 0:
            raise ValidationError(
                message=f"{ramp_down_discharge=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="ramp_down_discharge",
            )
        self.ramp_up_charge = ramp_up_charge
        self.ramp_down_charge = ramp_down_charge
        self.ramp_up_discharge = ramp_up_discharge
        self.ramp_down_discharge = ramp_down_discharge

        # How long the storage unit has to be in operation before it can be shut down.
        if min_operating_time < 0:
            raise ValidationError(
                message=f"{min_operating_time=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="min_operating_time",
            )
        self.min_operating_time = min_operating_time
        # How long the storage unit has to be shut down before it can be started.
        if min_down_time < 0:
            raise ValidationError(
                message=f"{min_down_time=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="min_down_time",
            )
        self.min_down_time = min_down_time
        # The downtime before hot start of the storage unit.
        if downtime_hot_start < 0:
            raise ValidationError(
                message=f"{downtime_hot_start=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="downtime_hot_start",
            )
        self.downtime_hot_start = downtime_hot_start
        # The downtime before warm start of the storage unit.
        if downtime_warm_start < 0:
            raise ValidationError(
                message=f"{downtime_warm_start=} must be >= 0 for unit {self.id}",
                id=self.id,
                field="downtime_warm_start",
            )
        self.downtime_warm_start = downtime_warm_start

        self.hot_start_cost = hot_start_cost * max_power_discharge
        self.warm_start_cost = warm_start_cost * max_power_discharge
        self.cold_start_cost = cold_start_cost * max_power_discharge

    def execute_current_dispatch(self, start: datetime, end: datetime) -> np.ndarray:
        """
        Executes the current dispatch of the unit based on the provided timestamps.

        The dispatch is only executed, if it is in the constraints given by the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.

        Returns:
            np.ndarray: The volume of the unit within the given time range.
        """
        start = max(start, self.index[0])
        time_delta = self.index.freq / timedelta(hours=1)

        for t in self.index[start:end]:
            current_power = self.outputs["energy"].at[t]

            # adjust power to constraints of the unit
            if current_power > self.max_power_discharge:
                current_power = self.max_power_discharge
            elif current_power < self.max_power_charge:
                current_power = self.max_power_charge
            elif (
                self.min_power_discharge > current_power > self.min_power_charge
                and current_power != 0
            ):
                current_power = 0

            # calculate the change in state of charge
            delta_soc = 0
            soc = self.outputs["soc"].at[t]

            # discharging
            if current_power > 0:
                max_soc_discharge = self.calculate_soc_max_discharge(soc)

                if current_power > max_soc_discharge:
                    current_power = max_soc_discharge

                delta_soc = (
                    -current_power * time_delta / self.efficiency_discharge
                ) / self.capacity

            # charging
            elif current_power < 0:
                max_soc_charge = self.calculate_soc_max_charge(soc)

                if current_power < max_soc_charge:
                    current_power = max_soc_charge

                delta_soc = (
                    -current_power * time_delta * self.efficiency_charge
                ) / self.capacity

            # update the values of the state of charge and the energy
            next_freq = t + self.index.freq
            if next_freq in self.index:
                self.outputs["soc"].at[next_freq] = soc + delta_soc
            self.outputs["energy"].at[t] = current_power

        return self.outputs["energy"].loc[start:end]

    @lru_cache(maxsize=256)
    def calculate_marginal_cost(
        self,
        start: datetime,
        power: float,
    ) -> float:
        """
        Calculates the marginal cost of the unit based on the provided start time and power output and returns it.
        Returns the marginal cost of the unit.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: The marginal cost of the unit.
        """

        if power > 0:
            additional_cost = self.additional_cost_discharge
        else:
            additional_cost = self.additional_cost_charge

        marginal_cost = additional_cost

        return marginal_cost

    def set_dispatch_plan(
        self, marketconfig: MarketConfig, orderbook: Orderbook
    ) -> None:
        """
        Records the committed (pre-SoC-clip) energy schedule, then delegates to the
        base implementation.

        ``SupportsMinMaxCharge.set_dispatch_plan`` first accumulates the accepted
        volumes into ``outputs["energy"]`` and then clips them down to what the state
        of charge actually allows, overwriting the committed values. Markets that
        balance against the *market-committed* dispatch need that pre-clip schedule:
        in redispatch the generators were dispatched against the unclipped storage
        volume, so if storage reported only the clipped value the network would not
        balance and the optimiser would ramp real generation to close the gap.

        The committed schedule is therefore mirrored into
        ``outputs["energy_committed"]`` here - before the clip runs - by repeating the
        same accumulation the base implementation performs. Only the energy product is
        mirrored; capacity products do not feed the redispatch baseline.

        Args:
            marketconfig (MarketConfig): The market configuration.
            orderbook (Orderbook): The orderbook.
        """
        if orderbook and marketconfig.product_type == "energy":
            start = min(order["start_time"] for order in orderbook)
            end_excl = max(order["end_time"] for order in orderbook) - self.index.freq

            # seed with the energy already booked (i.e. after the previous clip) ...
            self.outputs["energy_committed"].loc[start:end_excl] = self.outputs[
                "energy"
            ].loc[start:end_excl]

            # ... then add this clearing's accepted volumes, exactly as BaseUnit does
            for order in orderbook:
                order_end_excl = order["end_time"] - self.index.freq
                accepted_volume = order["accepted_volume"]
                if isinstance(accepted_volume, dict):
                    accepted_volume = list(accepted_volume.values())
                self.outputs["energy_committed"].loc[
                    order["start_time"] : order_end_excl
                ] += accepted_volume

        super().set_dispatch_plan(marketconfig, orderbook)

        # Feasibility guard for redispatch. The redispatch clearing limits the
        # cumulative redispatch energy with ``add_redispatch_soc_constraints``, so the
        # SOC clip in the base implementation should never have to cut an accepted
        # redispatch volume. If it does, the guarantee was broken - say so loudly.
        if (
            orderbook
            and marketconfig.product_type == "energy"
            and getattr(marketconfig, "market_mechanism", None) == "redispatch"
        ):
            committed = self.outputs["energy_committed"].loc[start:end_excl]
            executed = self.outputs["energy"].loc[start:end_excl]
            curtailed = np.abs(np.asarray(committed) - np.asarray(executed))
            if curtailed.max(initial=0.0) > EPS:
                logger.warning(
                    "Storage %s: accepted redispatch is not SOC-feasible, %.3f MWh "
                    "were curtailed between %s and %s.",
                    self.id,
                    float(curtailed.sum() * (self.index.freq / timedelta(hours=1))),
                    start,
                    end_excl,
                )

    def calculate_soc_path(self, start: datetime, end: datetime) -> dict:
        """
        Calculates the state of charge at every time step between ``start`` and
        ``end`` (both included) if the unit follows its current schedule
        ``outputs["energy"]``.

        The path starts at the stored SOC at ``start`` and is advanced with the same
        rules as :meth:`execute_current_dispatch`, i.e. the schedule is clipped to the
        SOC limits. Time steps that no market has cleared yet have a schedule of 0, so
        the storage is assumed idle there. This makes the path valid also for hours
        beyond the last day-ahead clearing (e.g. overlapping redispatch windows).

        Args:
            start (datetime.datetime): First time step of the path.
            end (datetime.datetime): Last time step of the path (included).

        Returns:
            dict: Mapping of time step to SOC (between 0 and 1).
        """
        time_delta = self.index.freq / timedelta(hours=1)
        soc = self.outputs["soc"].at[start]
        path = {start: soc}

        t = start
        while t < end:
            power = self.outputs["energy"].at[t] if t in self.index else 0.0
            if power > 0:
                power = min(power, self.calculate_soc_max_discharge(soc))
                soc -= power * time_delta / self.efficiency_discharge / self.capacity
            elif power < 0:
                power = max(power, self.calculate_soc_max_charge(soc))
                soc -= power * time_delta * self.efficiency_charge / self.capacity
            # guard against floating point drift outside the SOC limits
            soc = min(max(soc, self.min_soc), self.max_soc)
            t = t + self.index.freq
            path[t] = soc

        return path

    def calculate_redispatch_power_bounds(
        self, start: datetime, soc: float
    ) -> tuple[float, float]:
        """
        Calculates the absolute power range (MW) the storage can run at in the time
        step starting at ``start``, given the state of charge at its start.

        The SOC limits are applied to the ABSOLUTE power, not to the change relative
        to the current schedule. This matters for redispatch: a full storage that is
        scheduled to discharge 50 MW cannot charge, but it can still discharge less
        (down to 0 MW). Clipping the change instead would block that downward
        redispatch. Capacity reserved on other markets (CRM) is kept free, using the
        same convention as :meth:`calculate_min_max_charge` and
        :meth:`calculate_min_max_discharge`.

        Args:
            start (datetime.datetime): Start of the time step.
            soc (float): State of charge at ``start`` (between 0 and 1).

        Returns:
            tuple[float, float]: Lowest (charging, <= 0) and highest (discharging,
            >= 0) absolute power in MW.
        """
        capacity_pos = self.outputs["capacity_pos"].at[start]
        capacity_neg = self.outputs["capacity_neg"].at[start]

        max_power = min(
            self.max_power_discharge - capacity_pos,
            self.calculate_soc_max_discharge(soc),
        )
        min_power = max(
            self.max_power_charge - capacity_neg,
            self.calculate_soc_max_charge(soc),
        )
        return min(min_power, 0.0), max(max_power, 0.0)

    def calculate_redispatch_energy_headroom(self, soc: float) -> tuple[float, float]:
        """
        Converts a state of charge into the energy the redispatch may still take out
        of or put into the storage.

        Args:
            soc (float): The state of charge (between 0 and 1).

        Returns:
            tuple[float, float]: Energy above ``min_soc`` and energy below ``max_soc``
            in MWh, both measured inside the storage (i.e. before efficiencies).
        """
        energy_discharge_headroom = max((soc - self.min_soc) * self.capacity, 0.0)
        energy_charge_headroom = max((self.max_soc - soc) * self.capacity, 0.0)
        return energy_discharge_headroom, energy_charge_headroom

    @staticmethod
    def redispatch_soc_callback(storage_ids, storage_bids: dict, duration_hours: float):
        """
        Builds the ``extra_functionality`` hook that PyPSA calls in the redispatch
        clearing after building the model and before solving it.

        For every storage in ``storage_ids`` it adds the cumulative SOC constraints of
        :meth:`add_redispatch_soc_constraints`. It relies on the naming of the
        redispatch network (see ``grid_utils.add_redispatch_storages``): the variable
        ``Generator-p`` and the generators ``<id>_up`` / ``<id>_down``.

        Args:
            storage_ids (pandas.Index): Storage units to constrain.
            storage_bids (dict): Per field (``energy_discharge_headroom``,
                ``energy_charge_headroom``, ``efficiency_charge``,
                ``efficiency_discharge``) a DataFrame with one row per snapshot and one
                column per storage, taken from the bids.
            duration_hours (float): Duration of every snapshot in hours.

        Returns:
            Callable or None: The hook, or None if there is no storage to constrain.
        """
        if len(storage_ids) == 0:
            return None

        def add_soc_constraints(n, snapshots):
            p = n.model["Generator-p"]
            # component dimension: "Generator" in older, "name" in newer PyPSA
            (dim,) = (d for d in p.dims if d != "snapshot")
            for u in storage_ids:
                Storage.add_redispatch_soc_constraints(
                    model=n.model,
                    name=u,
                    up=p.sel({dim: f"{u}_up"}, drop=True),
                    down=p.sel({dim: f"{u}_down"}, drop=True),
                    duration_hours=[duration_hours] * len(snapshots),
                    energy_discharge_headroom=storage_bids["energy_discharge_headroom"][
                        u
                    ],
                    energy_charge_headroom=storage_bids["energy_charge_headroom"][u],
                    efficiency_charge=float(
                        storage_bids["efficiency_charge"][u].iloc[0]
                    ),
                    efficiency_discharge=float(
                        storage_bids["efficiency_discharge"][u].iloc[0]
                    ),
                )

        return add_soc_constraints

    @staticmethod
    def add_redispatch_soc_constraints(
        model,
        name: str,
        up,
        down,
        duration_hours,
        energy_discharge_headroom,
        energy_charge_headroom,
        efficiency_charge: float,
        efficiency_discharge: float,
        snapshot_dim: str = "snapshot",
    ) -> None:
        """
        Adds the cumulative SOC constraints for one storage unit to a redispatch
        optimisation model.

        Redispatch models a storage as a fixed baseline plus an upward (``up``) and a
        downward (``down``) redispatch variable per snapshot. Power bounds per
        snapshot alone allow the optimiser to request more energy over the horizon
        than the storage holds, so for every snapshot ``t``::

            sum_{k<=t} (up_k / eta_d - down_k * eta_c) * dt_k <= E_dis[t]  (not empty)
            sum_{k<=t} (down_k / eta_d - up_k * eta_c) * dt_k <= E_ch[t]   (not full)

        ``E_dis`` / ``E_ch`` are the energies above ``min_soc`` / below ``max_soc``
        at the end of snapshot ``t`` along the baseline SOC path
        (see :meth:`calculate_soc_path` and
        :meth:`calculate_redispatch_energy_headroom`).

        Upward redispatch can mean "discharge more" (drains ``up / eta_d``) or
        "charge less" (drains ``up * eta_c``); downward redispatch can mean "charge
        more" (adds ``down * eta_c``) or "discharge less" (adds ``down / eta_d``).
        Each constraint uses the worst case for its own direction. The model stays
        linear and the storage can never be over-committed; with ``eta < 1`` it can
        be slightly conservative in snapshots where redispatch reverses the baseline
        direction.

        Args:
            model (linopy.Model): The optimisation model.
            name (str): Storage id, used to name the constraints.
            up (linopy.Variable): Upward redispatch per snapshot (MW, >= 0).
            down (linopy.Variable): Downward redispatch per snapshot (MW, >= 0).
            duration_hours (array-like): Duration of every snapshot in hours.
            energy_discharge_headroom (array-like): ``E_dis`` per snapshot (MWh).
            energy_charge_headroom (array-like): ``E_ch`` per snapshot (MWh).
            efficiency_charge (float): Charge efficiency.
            efficiency_discharge (float): Discharge efficiency.
            snapshot_dim (str): Name of the time dimension of ``up`` and ``down``.
        """
        # linopy aligns pandas objects by their index name, so plain Series indexed
        # like the snapshot dimension of the variables are enough here
        snapshots = pd.Index(up.coords[snapshot_dim].values, name=snapshot_dim)
        dt = pd.Series(np.asarray(duration_hours, dtype=float), index=snapshots)
        e_dis = pd.Series(
            np.asarray(energy_discharge_headroom, dtype=float), index=snapshots
        )
        e_ch = pd.Series(
            np.asarray(energy_charge_headroom, dtype=float), index=snapshots
        )

        drained = ((up / efficiency_discharge - down * efficiency_charge) * dt).cumsum(
            snapshot_dim
        )
        added = ((down / efficiency_discharge - up * efficiency_charge) * dt).cumsum(
            snapshot_dim
        )

        model.add_constraints(drained <= e_dis, name=f"{name}-redispatch-soc-min")
        model.add_constraints(added <= e_ch, name=f"{name}-redispatch-soc-max")

    def calculate_soc_max_discharge(self, soc) -> float:
        """
        Calculates the maximum discharge power depending on the current state of charge.

        Args:
            soc (float): The current state of charge (between 0 and 1).

        Returns:
            float: The maximum discharge power in MW.
        """
        duration = self.index.freq / timedelta(hours=1)
        power = max(
            0,
            (
                (soc - self.min_soc)
                * self.capacity
                * self.efficiency_discharge
                / duration
            ),
        )
        return power

    def calculate_soc_max_charge(
        self,
        soc,
    ) -> float:
        """
        Calculates the maximum charge power depending on the current state of charge.

        Args:
            soc (float): The current state of charge (between 0 and 1).

        Returns:
            float: The maximum charge power in MW.
        """
        duration = self.index.freq / timedelta(hours=1)
        power = min(
            0,
            ((soc - self.max_soc) * self.capacity / self.efficiency_charge / duration),
        )
        return power

    def calculate_min_max_charge(
        self, start: datetime, end: datetime, soc: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the min and max charging power for the given time period.
        This is relative to the already sold output on other markets for the same period.
        It also adheres to reserved positive and negative capacities reserved for other markets.

        Args:
            start (datetime.datetime): The start of the current dispatch.
            end (datetime.datetime): The end of the current dispatch.
            soc (float): The current state-of-charge (between 0 and 1). Defaults to None, then using soc at given start time.

        Returns:
            tuple[np.ndarray, np.ndarray]: The minimum and maximum charge power levels of the storage unit in MW.
        """
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"].loc[start:end_excl]
        capacity_pos = self.outputs["capacity_pos"].loc[start:end_excl]
        capacity_neg = self.outputs["capacity_neg"].loc[start:end_excl]

        min_power_charge = self.min_power_charge - (base_load + capacity_pos)
        min_power_charge = min_power_charge.clip(max=0)

        max_power_charge = self.max_power_charge - (base_load + capacity_neg)
        max_power_charge = np.where(
            max_power_charge <= min_power_charge, max_power_charge, 0
        )
        min_power_charge = np.where(
            min_power_charge >= max_power_charge, min_power_charge, 0
        )

        # restrict charging according to max_soc
        if soc is None:
            soc = self.outputs["soc"].at[start]
        max_soc_charge = self.calculate_soc_max_charge(soc)
        max_power_charge = max_power_charge.clip(min=max_soc_charge)

        return min_power_charge, max_power_charge

    def calculate_min_max_discharge(
        self, start: datetime, end: datetime, soc: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the min and max discharging power for the given time period.
        This is relative to the already sold output on other markets for the same period.
        It also adheres to reserved positive and negative capacities.

        Args:
            start (datetime.datetime): The start of the current dispatch.
            end (datetime.datetime): The end of the current dispatch.
            soc (float): The current state-of-charge (between 0 and 1). Defaults to None, then using soc at given start time.

        Returns:
            tuple[np.ndarray, np.ndarray]: The minimum and maximum discharge power levels of the storage unit in MW.
        """
        # end includes the end of the last product, to get the last products' start time we deduct the frequency once
        end_excl = end - self.index.freq

        base_load = self.outputs["energy"].loc[start:end_excl]
        capacity_pos = self.outputs["capacity_pos"].loc[start:end_excl]
        capacity_neg = self.outputs["capacity_neg"].loc[start:end_excl]

        min_power_discharge = self.min_power_discharge - (base_load + capacity_neg)
        min_power_discharge = min_power_discharge.clip(min=0)

        max_power_discharge = self.max_power_discharge - (base_load + capacity_pos)

        # Adjust max_power_discharge using np.where
        max_power_discharge = np.where(
            max_power_discharge >= min_power_discharge, max_power_discharge, 0
        )

        # Adjust min_power_discharge using np.where
        min_power_discharge = np.where(
            min_power_discharge < max_power_discharge, min_power_discharge, 0
        )

        # restrict according to min_soc
        if soc is None:
            soc = self.outputs["soc"].at[start]
        max_soc_discharge = self.calculate_soc_max_discharge(soc)
        max_power_discharge = max_power_discharge.clip(max=max_soc_discharge)

        return min_power_discharge, max_power_discharge

    def calculate_ramp_discharge(
        self,
        soc: float,
        previous_power: float,
        power_discharge: float,
        current_power: float = 0,
        min_power_discharge: float = 0,
    ) -> float:
        """
        Adjusts the discharging power to the ramping constraints.

        Args:
            soc (float): The current state of charge (between 0 and 1).
            previous_power (float): The previous power output of the unit.
            power_discharge (float): The discharging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.
            min_power_discharge (float, optional): The minimum discharging power output of the unit. Defaults to 0.

        Returns:
            float: The discharging power adjusted to the ramping constraints.
        """
        power_discharge = super().calculate_ramp_discharge(
            previous_power,
            power_discharge,
            current_power,
        )
        # restrict according to min_SOC

        max_soc_discharge = self.calculate_soc_max_discharge(soc)
        power_discharge = min(power_discharge, max_soc_discharge)
        if power_discharge < min_power_discharge:
            power_discharge = 0

        return power_discharge

    def calculate_ramp_charge(
        self,
        soc: float,
        previous_power: float,
        power_charge: float,
        current_power: float = 0,
        min_power_charge: float = 0,
    ) -> float:
        """
        Adjusts the charging power to the ramping constraints.

        Args:
            soc (float): The current state of charge (between 0 and 1).
            previous_power (float): The previous power output of the unit.
            power_charge (float): The charging power output of the unit.
            current_power (float, optional): The current power output of the unit. Defaults to 0.
            min_power_charge (float, optional): The minimum charging power output of the unit. Defaults to 0.

        Returns:
            float: The charging power adjusted to the ramping constraints.
        """
        power_charge = super().calculate_ramp_charge(
            previous_power,
            power_charge,
            current_power,
        )

        # restrict charging according to max_SOC
        max_soc_charge = self.calculate_soc_max_charge(soc)

        power_charge = max(power_charge, max_soc_charge)
        if power_charge > min_power_charge:
            power_charge = 0

        return power_charge

    def get_starting_costs(self, op_time):
        """
        Calculates the starting costs of the unit depending on how long it was shut down

        Args:
            op_time (float): The time the unit was shut down in hours.

        Returns:
            float: The starting costs of the unit.
        """
        if op_time > 0:
            # unit is running
            return 0
        if -op_time < self.downtime_hot_start:
            return self.hot_start_cost
        elif -op_time < self.downtime_warm_start:
            return self.warm_start_cost
        else:
            return self.cold_start_cost

    def as_dict(self) -> dict:
        """
        Return the storage unit's attributes as a dictionary, including specific attributes.

        Returns:
            dict: The storage unit's attributes as a dictionary.
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "capacity": self.capacity,
                "max_soc": self.max_soc,
                "min_soc": self.min_soc,
                "max_power_charge": self.max_power_charge,
                "max_power_discharge": self.max_power_discharge,
                "min_power_charge": self.min_power_charge,
                "min_power_discharge": self.min_power_discharge,
                "efficiency_charge": self.efficiency_charge,
                "efficiency_discharge": self.efficiency_discharge,
                "unit_type": "storage",
            }
        )

        return unit_dict