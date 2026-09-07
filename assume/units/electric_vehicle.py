# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A storage-sized EV model, independent of DSM and Building optimisation."""

from assume.common.base import BaseUnit
from assume.common.fast_pandas import FastSeries
from assume.units.storage import Storage


class ElectricVehicleUnit(Storage):
    """Battery with plug availability and exogenous driving consumption.

    Uses Storage's MW/MWh conventions (charging power is negative, SOC is a
    fraction). ``trip_energy_consumption`` is battery-side MWh **per step**,
    already allocated to away periods, not a power or a daily trip total.
    Availability is binary. Set max_power_discharge=0 to disable V2G.
    """

    def __init__(self, *args, trip_energy_consumption=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.capacity <= 0 or not (
            self.efficiency_charge > 0 and self.efficiency_discharge > 0
        ):
            raise ValueError("EV capacity and efficiencies must be positive")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("Initial EV SOC must lie within SOC bounds")
        self.trip_energy_consumption = FastSeries(
            index=self.index,
            value=(
                trip_energy_consumption
                if trip_energy_consumption is not None
                else getattr(self.forecaster, "trip_energy_consumption", 0.0)
            ),
        )
        for t in self.index:
            available = self.forecaster.availability.at[t]
            trip = self.trip_energy_consumption.at[t]
            if available not in (0, 1) or not 0 <= trip < float("inf"):
                raise ValueError(
                    "EV availability must be binary and trips finite/nonnegative"
                )
            if available and trip:
                raise ValueError("Driving consumption must occur while unplugged")

    def energy_at(self, start):
        """Reconstruct battery energy from accepted dispatch, never provisional plans."""
        energy = self.initial_soc * self.capacity
        for t in self.index:
            if t >= start:
                break
            energy, _, _ = self._transition(t, energy)
        return energy

    def _transition(self, t, energy):
        hours = self.index.freq.total_seconds() / 3600
        available = self.forecaster.availability.at[t]
        power = self.outputs["energy"].at[t]
        power = min(
            max(power, self.max_power_charge * available),
            self.max_power_discharge * available,
        )
        power = min(
            power,
            max(0, energy - self.min_soc * self.capacity)
            * self.efficiency_discharge
            / hours,
        )
        power = max(
            power,
            -max(0, self.max_soc * self.capacity - energy)
            / self.efficiency_charge
            / hours,
        )
        energy += (
            max(-power, 0) * self.efficiency_charge
            - max(power, 0) / self.efficiency_discharge
        ) * hours
        trip = self.trip_energy_consumption.at[t]
        unmet = max(0, self.min_soc * self.capacity - (energy - trip))
        return max(self.min_soc * self.capacity, energy - trip), power, unmet

    def execute_current_dispatch(self, start, end):
        energy = self.energy_at(start)
        for t in self.index[max(start, self.index[0]) : end]:
            self.outputs["soc"].at[t] = energy / self.capacity
            energy, power, unmet = self._transition(t, energy)
            self.outputs["energy"].at[t] = power
            self.outputs["unmet_driving_energy"].at[t] = unmet
            if t + self.index.freq in self.index:
                self.outputs["soc"].at[t + self.index.freq] = energy / self.capacity
        return self.outputs["energy"].loc[start:end]

    def set_dispatch_plan(self, marketconfig, orderbook):
        # Storage's implementation would change SOC for non-energy tariffs too.
        BaseUnit.set_dispatch_plan(self, marketconfig, orderbook)
        if marketconfig.product_type == "energy":
            for order in orderbook:
                self.outputs["energy_committed"].loc[
                    order["start_time"] : order["end_time"] - self.index.freq
                ] = 1

    def calculate_min_max_charge(self, start, end, soc=None):
        if soc is None:
            soc = self.energy_at(start) / self.capacity
        limits = super().calculate_min_max_charge(start, end, soc)
        plug = self.forecaster.availability.loc[start : end - self.index.freq]
        return tuple(limit * plug for limit in limits)

    def calculate_min_max_discharge(self, start, end, soc=None):
        if soc is None:
            soc = self.energy_at(start) / self.capacity
        limits = super().calculate_min_max_discharge(start, end, soc)
        plug = self.forecaster.availability.loc[start : end - self.index.freq]
        return tuple(limit * plug for limit in limits)

    def as_dict(self):
        result = super().as_dict()
        result["unit_type"] = "electric_vehicle"
        result["efficiency_charge"] = self.efficiency_charge
        result["efficiency_discharge"] = self.efficiency_discharge
        return result
