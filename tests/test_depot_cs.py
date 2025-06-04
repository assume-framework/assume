import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import ChargingStation  

use_solver = "appsigurob_highs"

@pytest.fixture
def cs_config():
    return {
        "max_power": 10.0,
        "min_power": 2.0,
        "ramp_up": 3.0,
        "ramp_down": 2.5,
    }


@pytest.fixture
def cs_model_with_availability(cs_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=list(range(10)))

    availability_profile = pd.Series(
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1], index=model.time_steps
    )

    cs = ChargingStation(
        time_steps=model.time_steps,
        availability_profile=availability_profile,
        power_flow_directionality="unidirectional",
        **cs_config,
    )

    model.cs = pyo.Block()
    cs.add_to_model(model, model.cs)

    # maximize total discharge
    model.total_discharge = pyo.Objective(
        expr=sum(model.cs.discharge[t] for t in model.time_steps),
        sense=pyo.maximize
    )

    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=True)

    return model, results, availability_profile


def test_charging_station_availability(cs_model_with_availability):
    model, results, availability_profile = cs_model_with_availability

    for t in model.time_steps:
        val = availability_profile[t]
        discharge = pyo.value(model.cs.discharge[t])
        if val == 0:
            assert discharge == 0
        else:
            assert discharge <= model.cs.max_power
            assert discharge >= model.cs.min_power


def test_charging_station_ramp_limits(cs_model_with_availability):
    model, results, _ = cs_model_with_availability
    ts = sorted(list(model.time_steps))

    for i in range(1, len(ts)):
        prev = ts[i - 1]
        curr = ts[i]
        d_prev = pyo.value(model.cs.discharge[prev])
        d_curr = pyo.value(model.cs.discharge[curr])

        # Ramp up constraint
        assert d_curr - d_prev <= pyo.value(model.cs.ramp_up) + 1e-5
        # Ramp down constraint
        assert d_prev - d_curr <= pyo.value(model.cs.ramp_down) + 1e-5


if __name__ == "__main__":
    pytest.main(["-s", __file__])
