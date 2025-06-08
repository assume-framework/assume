# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import ElectricVehicle

use_solver = "appsi_highs"


@pytest.fixture
def ev_config():
    return {
        "max_capacity": 50.0,
        "efficiency_charge": 1,
        "efficiency_discharge": 1,
        "initial_soc": 1,
        "mileage": 0.77,
        "max_power_charge": 10.0,
        "max_power_discharge": 10.0,
        "ramp_up": 10,
        "ramp_down": 10,
    }


@pytest.fixture
def ev_model_with_availability(ev_config):
    time_steps = list(range(10))
    availability_profile = pd.Series([0, 1, 0, 0, 1, 1, 1, 0, 0, 1], index=time_steps)
    external_range = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    electricity_price = [0.25] * 10

    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=time_steps)
    model.electricity_price = pyo.Param(
        model.time_steps, initialize={t: electricity_price[t] for t in time_steps}
    )
    model.external_range = pyo.Param(
        model.time_steps, initialize={t: external_range[t] for t in time_steps}
    )

    ev = ElectricVehicle(
        time_steps=model.time_steps,
        availability_profile=availability_profile,
        power_flow_directionality="unidirectional",  # "bidirectional" # if needed
        **ev_config,
    )

    model.ev = pyo.Block()
    ev.add_to_model(model, model.ev, external_range=model.external_range)

    # minimize total operating cost
    model.obj = pyo.Objective(
        expr=sum(model.ev.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model)

    return model, results, availability_profile, external_range


def test_soc_balance_and_constraints(ev_model_with_availability):
    model, _, availability_profile, ext_range = ev_model_with_availability
    soc = model.ev.soc
    charge = model.ev.charge
    discharge = model.ev.discharge
    usage = model.ev.usage

    eff_c = pyo.value(model.ev.efficiency_charge)
    eff_d = pyo.value(model.ev.efficiency_discharge)
    initial_soc = pyo.value(model.ev.initial_soc)

    ts = sorted(model.time_steps)
    for i, t in enumerate(ts):
        if i == 0:
            soc_expected = (
                initial_soc
                + eff_c * pyo.value(charge[t])
                - (1 / eff_d) * pyo.value(discharge[t])
                - pyo.value(usage[t])
            )
        else:
            soc_expected = (
                pyo.value(soc[ts[i - 1]])
                + eff_c * pyo.value(charge[t])
                - (1 / eff_d) * pyo.value(discharge[t])
                - pyo.value(usage[t])
            )

        assert abs(pyo.value(soc[t]) - soc_expected) < 1e-4

        # Ramp constraints
        if i > 0:
            assert (
                pyo.value(charge[t]) - pyo.value(charge[ts[i - 1]])
                <= pyo.value(model.ev.ramp_up) + 1e-5
            )
            assert (
                pyo.value(charge[ts[i - 1]]) - pyo.value(charge[t])
                <= pyo.value(model.ev.ramp_down) + 1e-5
            )

            assert (
                pyo.value(discharge[t]) - pyo.value(discharge[ts[i - 1]])
                <= pyo.value(model.ev.ramp_up) + 1e-5
            )
            assert (
                pyo.value(discharge[ts[i - 1]]) - pyo.value(discharge[t])
                <= pyo.value(model.ev.ramp_down) + 1e-5
            )

        # Charge/discharge availability
        max_pc = pyo.value(model.ev.max_power_charge)
        max_pd = pyo.value(model.ev.max_power_discharge)
        if availability_profile[t] == 0:
            assert pyo.value(charge[t]) <= 1e-5
            assert pyo.value(discharge[t]) <= 1e-5
        else:
            assert pyo.value(charge[t]) <= max_pc + 1e-5
            assert pyo.value(discharge[t]) <= max_pd + 1e-5

        # Usage check
        expected_usage = (1 - availability_profile[t]) * (
            ext_range[t] / pyo.value(model.ev.mileage)
        )
        assert abs(pyo.value(usage[t]) - expected_usage) < 1e-4


if __name__ == "__main__":
    pytest.main(["-s", __file__])
