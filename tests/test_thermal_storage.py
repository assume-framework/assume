# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import ThermalStorage

use_solver = "appsi_highs"


@pytest.fixture
def price_profile():
    return pd.Series([20, 22, 25, 21, 18, 19, 23, 24, 26, 22], index=range(10))


@pytest.fixture
def storage_config():
    return {
        "capacity": 200,
        "min_soc": 0,
        "max_soc": 1,
        "max_power_charge": 40,
        "max_power_discharge": 40,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.95,
        "initial_soc": 0,
        "ramp_up": 50,
        "ramp_down": 0,
        "storage_loss_rate": 0.0,
    }


@pytest.fixture
def storage_schedule():
    # 0: Only charge, 1: Only discharge
    return pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], index=range(10))


@pytest.fixture
def short_term_storage_model(storage_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    storage = ThermalStorage(
        storage_type="short-term", **storage_config, time_steps=model.time_steps
    )
    model.storage = pyo.Block()
    storage.add_to_model(model, model.storage)

    model.energy_balance = pyo.Constraint(
        expr=sum(model.storage.discharge[t] for t in model.time_steps)
        == sum(model.storage.charge[t] for t in model.time_steps)
    )

    model.obj = pyo.Objective(
        expr=sum(
            model.storage.charge[t] * model.electricity_price[t]
            - model.storage.discharge[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)
    return model, results


@pytest.fixture
def long_term_storage_model(storage_config, storage_schedule, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    storage = ThermalStorage(
        storage_type="long-term",
        storage_schedule_profile=storage_schedule,
        **storage_config,
        time_steps=model.time_steps,
    )
    model.storage = pyo.Block()
    storage.add_to_model(model, model.storage)

    model.energy_balance = pyo.Constraint(
        expr=sum(model.storage.discharge[t] for t in model.time_steps)
        == sum(model.storage.charge[t] for t in model.time_steps)
    )

    model.obj = pyo.Objective(
        expr=sum(
            model.storage.charge[t] * model.electricity_price[t]
            - model.storage.discharge[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)
    return model, results


# --- SHORT-TERM TESTS ---
def test_short_term_storage_solves(short_term_storage_model):
    model, results = short_term_storage_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    )


# --- LONG-TERM TESTS ---
def test_long_term_storage_solves(long_term_storage_model):
    model, results = long_term_storage_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    )


def test_long_term_storage_follows_schedule(
    long_term_storage_model, storage_schedule, storage_config
):
    model, _ = long_term_storage_model
    max_charge = storage_config["max_power_charge"]
    max_discharge = storage_config["max_power_discharge"]
    for t in model.time_steps:
        sch = storage_schedule[t]
        charge = pyo.value(model.storage.charge[t])
        discharge = pyo.value(model.storage.discharge[t])
        if sch == 0:
            # Only charging allowed, no discharge
            assert discharge <= 1e-6, f"Discharge at t={t} not allowed by schedule"
            assert charge <= max_charge + 1e-5, f"Charge at t={t} exceeds max"
        if sch == 1:
            # Only discharge allowed, no charge
            assert charge <= 1e-6, f"Charge at t={t} not allowed by schedule"
            assert discharge <= max_discharge + 1e-5, f"Discharge at t={t} exceeds max"


def test_long_term_storage_soc_limits(long_term_storage_model, storage_config):
    model, _ = long_term_storage_model
    max_soc = storage_config["max_soc"]
    min_soc = storage_config["min_soc"]
    for t in model.time_steps:
        soc = pyo.value(model.storage.soc[t])
        assert soc <= max_soc + 1e-5
        assert soc >= min_soc - 1e-5


def test_long_term_storage_initial_soc(long_term_storage_model, storage_config):
    model, _ = long_term_storage_model
    initial_soc = storage_config["initial_soc"]
    soc_0 = pyo.value(model.storage.soc[0])
    assert soc_0 == initial_soc


if __name__ == "__main__":
    pytest.main(["-s", __file__])
