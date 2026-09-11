# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A storage-sized EV model, independent of DSM and Building optimisation.

An :class:`ElectricVehicleUnit` is a :class:`~assume.units.storage.Storage` with
two additions: it is only connected to the grid part of the time, and it loses
energy to driving while it is away.  Everything else - the MW/MWh conventions,
the SOC bookkeeping, the charge/discharge efficiencies - is inherited unchanged,
which is what lets the vehicles take part in ordinary markets and be dispatched
by :class:`~assume.strategies.ev_aggregator.EVPortfolioStrategy`.

Conventions
-----------

* Power follows the Storage sign convention: **negative is charging**, positive
  is discharging (V2G).  Set ``max_power_discharge=0`` to disable V2G.
* ``availability`` is binary: 1 while plugged in, 0 while away.  Both charging
  and discharging are forced to zero while away.
* ``trip_energy_consumption`` is battery-side MWh **per step**, already spread
  over the away periods - it is neither a power nor a daily trip total, and it
  may only be nonzero while the vehicle is unplugged.

Degrading rather than failing
-----------------------------

A trip can ask for more energy than the battery holds.  Rather than treat that
as an error, :meth:`ElectricVehicleUnit._transition` floors the battery at
``min_soc`` and books the difference in ``outputs["unmet_driving_energy"]``, so
an under-charged fleet shows up as a reported service failure instead of a
crashed simulation.  ``EVPortfolioStrategy`` mirrors this: it plans with a
penalised slack rather than a hard driving constraint, and writes its own
expectation to ``outputs["ev_planned_unmet_energy"]``.

Outputs
-------

===============================  ===============================================
``soc``                          State of charge at the start of the step
``energy``                       Accepted dispatch, signed MW
``energy_committed``             1 once a market has cleared the step
``unmet_driving_energy``         Driving energy the battery could not supply
``ev_planned_energy``            Aggregator's plan for the step, signed MW
``ev_planned_unmet_energy``      Unserved driving energy the aggregator expects
``grid_fee_announced``           1 once the tariff for the step is published
===============================  ===============================================
"""

from assume.common.base import BaseUnit
from assume.common.fast_pandas import FastSeries
from assume.units.storage import Storage


class ElectricVehicleUnit(Storage):
    """Battery with plug availability and exogenous driving consumption.

    See the module docstring for conventions and outputs.

    Args:
        trip_energy_consumption: Battery-side MWh per step consumed by driving.
            Falls back to the forecaster's series when not given.
        *args, **kwargs: Forwarded to :class:`~assume.units.storage.Storage`.

    Raises:
        ValueError: If the capacity or efficiencies are not positive, the
            initial SOC is outside the SOC bounds, availability is not binary,
            driving consumption is negative, infinite or scheduled while the
            vehicle is plugged in, or a minimum-power/ramp limit is set (see
            below).
    """

    def __init__(self, *args, trip_energy_consumption=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.capacity <= 0 or not (
            self.efficiency_charge > 0 and self.efficiency_discharge > 0
        ):
            raise ValueError("EV capacity and efficiencies must be positive")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("Initial EV SOC must lie within SOC bounds")
        # Neither this unit's dispatch nor the portfolio MILP models a
        # minimum-power deadband or ramp limits - a charger has neither. Storage
        # accepts them, so reject them here rather than let a scenario set a
        # limit that is silently ignored everywhere it would matter.
        unsupported = {
            "min_power_charge": self.min_power_charge,
            "min_power_discharge": self.min_power_discharge,
            "ramp_up_charge": self.ramp_up_charge,
            "ramp_down_charge": self.ramp_down_charge,
            "ramp_up_discharge": self.ramp_up_discharge,
            "ramp_down_discharge": self.ramp_down_discharge,
        }
        set_anyway = sorted(
            name for name, value in unsupported.items() if value not in (None, 0, 0.0)
        )
        if set_anyway:
            raise ValueError(
                "EV units do not model minimum power or ramp limits; "
                f"remove {', '.join(set_anyway)}"
            )
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
        """Reconstruct battery energy at *start* from accepted dispatch, in MWh.

        Replays the whole run rather than reading ``outputs["soc"]`` so the
        answer only ever depends on dispatch the market actually accepted, never
        on a provisional plan an optimiser happened to write.
        """
        energy = self.initial_soc * self.capacity
        for t in self.index:
            if t >= start:
                break
            energy, _, _ = self._transition(t, energy)
        return energy

    def _transition(self, t, energy):
        """Advance the battery one step.

        Clips the accepted dispatch to what the plug and the SOC bounds allow,
        applies charge/discharge efficiencies, then subtracts the step's driving
        consumption - flooring at ``min_soc`` and reporting whatever the battery
        could not supply.

        Args:
            t: The step to advance over.
            energy: Battery energy at the start of the step, MWh.

        Returns:
            ``(energy, power, unmet)``: energy at the end of the step in MWh,
            the dispatch actually realised in signed MW, and the driving energy
            that could not be served in MWh.
        """
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
        """Realise the accepted dispatch over ``[start, end]`` and update the SOC.

        Idempotent: the battery is rebuilt from ``energy_at(start)`` each time,
        so re-running the same window yields the same trajectory.
        """
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
        """Book a market result, and mark energy products as committed.

        Deliberately bypasses ``Storage.set_dispatch_plan``, which would move the
        SOC for *any* product type - a grid-fee clearing carries an announced
        volume that must not land in the energy dispatch.  ``energy_committed``
        is set for accepted and rejected orders alike: once a market has cleared
        a step, that product is gone and the aggregator may not replan it.
        """
        BaseUnit.set_dispatch_plan(self, marketconfig, orderbook)
        if marketconfig.product_type == "energy":
            for order in orderbook:
                self.outputs["energy_committed"].loc[
                    order["start_time"] : order["end_time"] - self.index.freq
                ] = 1

    def calculate_min_max_charge(self, start, end, soc=None):
        """Storage charging limits, zeroed out whenever the vehicle is away."""
        if soc is None:
            soc = self.energy_at(start) / self.capacity
        limits = super().calculate_min_max_charge(start, end, soc)
        plug = self.forecaster.availability.loc[start : end - self.index.freq]
        return tuple(limit * plug for limit in limits)

    def calculate_min_max_discharge(self, start, end, soc=None):
        """Storage discharging limits, zeroed out whenever the vehicle is away."""
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
