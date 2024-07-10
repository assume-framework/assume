# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig
from assume.strategies.naive_strategies import NaiveDASteelplantStrategy
from assume.units.steel_plant import SteelPlant


@pytest.fixture
def dsm_components():
    return {
        "electrolyser": {
            "rated_power": 922,
            "min_power": 0,
            "ramp_up": 922,
            "ramp_down": 922,
            "min_operating_time": 0,
            "min_down_time": 0,
            "efficiency": 0.8,
            "fuel_type": "hydrogen",
        },
        "dri_plant": {
            "specific_hydrogen_consumption": 1.83,
            "specific_natural_gas_consumption": 0.9,
            "specific_electricity_consumption": 0.3,
            "specific_iron_ore_consumption": 1.43,
            "rated_power": 120,
            "min_power": 0,
            "fuel_type": "hydrogen",
            "ramp_up": 120,
            "ramp_down": 120,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
        "eaf": {
            "rated_power": 162,
            "min_power": 0,
            "specific_electricity_consumption": 0.44,
            "specific_dri_demand": 1.09,
            "specific_lime_demand": 0.046,
            "ramp_up": 162,
            "ramp_down": 162,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
    }


@pytest.fixture
def steel_plant(dsm_components) -> SteelPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[50] * 24,
        fuel_price_natural_gas=[30] * 24,
        co2_price=[20] * 24,
    )

    bidding_strategies = {"EOM": NaiveDASteelplantStrategy()}
    return SteelPlant(
        id="test_steel_plant",
        unit_operator="test_operator",
        bidding_strategies=bidding_strategies,
        index=index,
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
    )


def setup_model_parameters(model, steel_plant):
    steel_plant.define_sets(model)
    steel_plant.define_parameters(model)


def test_initialize_components(steel_plant, dsm_components):
    model = pyo.ConcreteModel()
    setup_model_parameters(model, steel_plant)
    steel_plant.initialize_components(dsm_components, model)
    assert "electrolyser" in steel_plant.dsm_components
    assert "dri_plant" in steel_plant.dsm_components
    assert "eaf" in steel_plant.dsm_components


def test_initialize_process_sequence(steel_plant, dsm_components):
    model = pyo.ConcreteModel()
    setup_model_parameters(model, steel_plant)
    steel_plant.initialize_components(dsm_components, model)
    steel_plant.initialize_process_sequence(model)
    assert model.direct_hydrogen_flow_constraint is not None
    assert model.direct_dri_flow_constraint is not None
    assert model.shaft_to_arc_furnace_material_flow_constraint is not None


def setup_model_variables(model, steel_plant):
    steel_plant.define_variables(model)


def test_define_constraints(steel_plant, dsm_components):
    model = pyo.ConcreteModel()
    setup_model_parameters(model, steel_plant)
    setup_model_variables(model, steel_plant)
    steel_plant.initialize_components(dsm_components, model)
    steel_plant.define_constraints(model)
    assert model.steel_output_association_constraint is not None
    assert model.total_power_input_constraint is not None
    assert model.cost_per_time_step is not None


def test_define_objectives(steel_plant):
    model = pyo.ConcreteModel()
    setup_model_parameters(model, steel_plant)
    setup_model_variables(model, steel_plant)
    steel_plant.define_objective_opt(model)
    assert model.obj_rule_opt is not None


def test_determine_optimal_operation_without_flex(steel_plant):
    steel_plant.determine_optimal_operation_without_flex()
    assert steel_plant.opt_power_requirement is not None
    assert isinstance(steel_plant.opt_power_requirement, pd.Series)


def test_set_dispatch_plan(steel_plant):
    market_config = MarketConfig(market_id="EOM", product_type="energy")
    orderbook = [
        {
            "start_time": pd.Timestamp("2023-01-01 00:00:00"),
            "end_time": pd.Timestamp("2023-01-01 01:00:00"),
            "accepted_volume": 50,
            "price": 50,
        }
    ]

    steel_plant.set_dispatch_plan(market_config, orderbook)
    assert steel_plant.outputs["energy"].iloc[0] == 50


def test_calculate_marginal_cost(steel_plant):
    # Sequence to initialize the model correctly
    steel_plant.model = pyo.ConcreteModel()
    steel_plant.define_sets(steel_plant.model)
    steel_plant.define_parameters(steel_plant.model)
    steel_plant.initialize_components(steel_plant.dsm_components, steel_plant.model)
    steel_plant.initialize_process_sequence(steel_plant.model)
    steel_plant.define_variables(steel_plant.model)
    steel_plant.define_constraints(steel_plant.model)
    steel_plant.define_objective_opt(steel_plant.model)

    # Create an instance of the model and switch to opt
    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_opt(instance)

    start = pd.Timestamp("2023-01-01 00:00:00")
    power = 100
    marginal_cost = steel_plant.calculate_marginal_cost(start, power)
    assert marginal_cost > 0


if __name__ == "__main__":
    pytest.main(["-s", __file__])
