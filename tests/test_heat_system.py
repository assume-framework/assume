# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecaster import HeatSystemForecaster
from assume.units.heat_system import HeatSystem


USE_SOLVER = "appsi_highs"
TOL = 1e-5


def _val(value):
    return pyo.value(value)


def _series(value, index):
    """Return a constant time series over the supplied index."""
    return pd.Series([value] * len(index), index=index, dtype=float)


def _make_forecaster(
    time_index,
    *,
    heat_demand=4.0,
    electricity_price=40.0,
    natural_gas_price=50.0,
):
    """
    Create the simple HeatSystem forecast used in the 3-node validation case.

    The validation economics are intentionally transparent:

        HP heat cost     = electricity_price / COP = 40 / 2 = 20 EUR/MWh_th
        Boiler heat cost = natural_gas_price / eta = 50 / 1 = 50 EUR/MWh_th

    Therefore the heat pump must be selected in the base case.
    """
    return HeatSystemForecaster(
        index=time_index,
        fuel_prices={
            # HeatSystem maps the boiler fuel_type "natural_gas"
            # to the established fuel-price key "natural gas".
            "natural gas": _series(natural_gas_price, time_index),
        },
        market_prices={
            "EOM": _series(electricity_price, time_index),
        },
        heat_demand=_series(heat_demand, time_index),
    )


def _make_components(
    *,
    hp_max_power=2.0,
    cop=2.0,
    boiler_max_power=5.0,
    boiler_efficiency=1.0,
):
    """Components corresponding to the simplified HeatSystem validation case."""
    return {
        "heat_pump": {
            "max_power": hp_max_power,
            "min_power": 0.0,
            "cop": cop,
            "ramp_up": hp_max_power,
            "ramp_down": hp_max_power,
        },
        "boiler_gas": {
            "max_power": boiler_max_power,
            "min_power": 0.0,
            "efficiency": boiler_efficiency,
            "fuel_type": "natural_gas",
            "ramp_up": boiler_max_power,
            "ramp_down": boiler_max_power,
        },
    }


def _make_heat_system(
    forecaster,
    components,
    *,
    unit_id="hs_north",
    node="north",
):
    return HeatSystem(
        id=unit_id,
        unit_operator="test_operator",
        bidding_strategies={},
        forecaster=forecaster,
        components=components,
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=10,
        node=node,
    )


def _solve_heat_system(heat_system):
    """
    Build and solve the HeatSystem in its cost-optimal operating mode.

    This follows the same pattern already used by ASSUME's DSM/building tests:
    set up the model, create an instance, switch to the optimal-operation
    objective, and solve with appsi_highs.
    """
    heat_system.setup_model(presolve=True)
    instance = heat_system.model.create_instance()
    instance = heat_system.switch_to_opt(instance)

    solver = pyo.SolverFactory(USE_SOLVER)
    results = solver.solve(instance)

    return heat_system, instance, results


@pytest.fixture
def time_index():
    # A short constant horizon is sufficient for the static validation case.
    return pd.date_range("2023-01-01 00:00:00", periods=4, freq="h")


@pytest.fixture
def base_heat_system(time_index):
    forecaster = _make_forecaster(
        time_index,
        heat_demand=4.0,
        electricity_price=40.0,
        natural_gas_price=50.0,
    )
    components = _make_components(
        hp_max_power=2.0,
        cop=2.0,
        boiler_max_power=5.0,
        boiler_efficiency=1.0,
    )
    return _make_heat_system(forecaster, components)


@pytest.fixture
def solved_base_heat_system(base_heat_system):
    return _solve_heat_system(base_heat_system)


def test_heat_system_model_solves_successfully(solved_base_heat_system):
    """The coupled HeatSystem optimization must solve to optimality."""
    _, _, results = solved_base_heat_system

    assert results.solver.status == pyo.SolverStatus.ok
    assert (
        results.solver.termination_condition
        == pyo.TerminationCondition.optimal
    )


def test_heat_system_base_case_uses_heat_pump(solved_base_heat_system):
    """
    Base validation case:

        heat demand       = 4 MW_th
        COP               = 2
        electricity price = 40 EUR/MWh_el
        gas price         = 50 EUR/MWh_gas
        boiler efficiency = 1

    Expected:
        HP electricity = 2 MW_el
        HP heat        = 4 MW_th
        boiler heat    = 0 MW_th
    """
    _, instance, _ = solved_base_heat_system

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(4.0, abs=TOL)

        assert _val(boiler.natural_gas_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(0.0, abs=TOL)

        assert _val(instance.total_power_input[t]) == pytest.approx(
            2.0, abs=TOL
        )


def test_heat_system_heat_balance(solved_base_heat_system):
    """Heat production must equal heat demand exactly when no storage is present."""
    _, instance, _ = solved_base_heat_system

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        heat_supply = _val(hp.heat_out[t]) + _val(boiler.heat_out[t])
        heat_demand = _val(instance.heat_demand[t])

        assert heat_supply == pytest.approx(heat_demand, abs=TOL)


def test_heat_system_uses_natural_gas_price_key(solved_base_heat_system):
    """
    Guard the deliberate naming bridge:

        component fuel_type = "natural_gas"
        fuel_prices key     = "natural gas"
    """
    _, instance, _ = solved_base_heat_system

    for t in instance.time_steps:
        assert _val(instance.natural_gas_price[t]) == pytest.approx(
            50.0, abs=TOL
        )


def test_heat_system_uses_boiler_when_electricity_is_more_expensive(time_index):
    """
    At 120 EUR/MWh_el and COP=2, HP heat costs 60 EUR/MWh_th.
    Gas heat costs 50 EUR/MWh_th because eta_boiler=1.

    Therefore the gas boiler must cover the full 4 MW_th demand.
    """
    forecaster = _make_forecaster(
        time_index,
        heat_demand=4.0,
        electricity_price=120.0,
        natural_gas_price=50.0,
    )
    heat_system = _make_heat_system(
        forecaster,
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )

    _, instance, _ = _solve_heat_system(heat_system)

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(0.0, abs=TOL)

        assert _val(boiler.natural_gas_in[t]) == pytest.approx(4.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(4.0, abs=TOL)

        # A natural-gas boiler has no electrical consumption.
        assert _val(instance.total_power_input[t]) == pytest.approx(
            0.0, abs=TOL
        )


def test_heat_system_uses_boiler_when_heat_pump_capacity_is_binding(time_index):
    """
    Capacity-binding validation:

        heat demand          = 6 MW_th
        HP max electricity   = 2 MW_el
        COP                  = 2
        HP max heat          = 4 MW_th

    Electricity is cheaper than gas, so the HP should run at its maximum and
    the boiler should supply the remaining 2 MW_th.
    """
    forecaster = _make_forecaster(
        time_index,
        heat_demand=6.0,
        electricity_price=40.0,
        natural_gas_price=50.0,
    )
    heat_system = _make_heat_system(
        forecaster,
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )

    _, instance, _ = _solve_heat_system(heat_system)

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(4.0, abs=TOL)

        assert _val(boiler.natural_gas_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(2.0, abs=TOL)

        assert _val(instance.total_power_input[t]) == pytest.approx(
            2.0, abs=TOL
        )

        heat_supply = _val(hp.heat_out[t]) + _val(boiler.heat_out[t])
        assert heat_supply == pytest.approx(
            _val(instance.heat_demand[t]), abs=TOL
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
