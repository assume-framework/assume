# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import pytest

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecaster import CementForecaster, SteelplantForecaster
from assume.common.market_objects import MarketConfig, Product
from assume.strategies.industrial_hybrid import (
    IndustrialHybridCapacityNegStrategy,
    IndustrialHybridEomStrategy,
    IndustrialHybridOtcStrategy,
    capacity_neg_commitment_constraint,
    component_operations_by_step,
    electric_wtp,
)
from assume.units.cement_plant import CementPlant
from assume.units.steel_plant import SteelPlant


def make_unit():
    index = FastIndex("2023-01-01", periods=2, freq="1h")
    return SimpleNamespace(
        id="cement_1",
        index=index,
        outputs=defaultdict(lambda: FastSeries(index=index, value=0.0)),
        _rh_full_horizon_production=[0.0, 0.0],
        _rh_init_states={},
    )


def make_rolling_horizon_cement_plant(
    components, electricity_price, demand, commit_hours=2
):
    """Create a small real cement unit for the hybrid strategy integration tests."""
    index = pd.date_range("2023-01-01", periods=4, freq="h")
    forecaster = CementForecaster(
        index,
        electricity_price=electricity_price,
        fuel_prices={
            "natural_gas": [500.0] * len(index),
            "coal": [500.0] * len(index),
            "hydrogen": [500.0] * len(index),
            "co2": [20.0] * len(index),
        },
        clinker_demand=[demand / len(index)] * len(index),
    )
    strategy = IndustrialHybridEomStrategy()
    plant = CementPlant(
        id="cement_hybrid",
        unit_operator="operator",
        bidding_strategies={"EOM": strategy},
        forecaster=forecaster,
        components=components,
        demand=demand,
        dsm_optimisation_config={
            "horizon_mode": "rolling_horizon",
            "look_ahead_horizon": "4h",
            "commit_horizon": f"{commit_hours}h",
            "rolling_step": f"{commit_hours}h",
        },
    )
    plant.setup_model()
    return plant, strategy


def steel_electrolyser_buffer_components():
    return {
        "electrolyser": {
            "max_power": 5.0,
            "min_power": 0.0,
            "efficiency": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
        "hydrogen_buffer_storage": {
            "capacity": 10.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "initial_soc": 0.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
        },
        "dri_plant": {
            "max_power": 5.0,
            "min_power": 0.0,
            "fuel_type": "hydrogen",
            "specific_hydrogen_consumption": 1.0,
            "specific_natural_gas_consumption": 1.0,
            "specific_electricity_consumption": 1.0,
            "specific_iron_ore_consumption": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
        "eaf": {
            "max_power": 5.0,
            "min_power": 0.0,
            "specific_electricity_consumption": 1.0,
            "specific_dri_demand": 1.0,
            "specific_lime_demand": 1.0,
            "lime_co2_factor": 0.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
    }


def make_rolling_horizon_steel_plant(electricity_price):
    """Create a real steel plant with a bufferable electrolyser route."""
    index = pd.date_range("2023-01-01", periods=4, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=electricity_price,
        steel_demand=[1.0] * len(index),
        fuel_prices={
            "natural_gas": [50.0] * len(index),
            "hydrogen": [100.0] * len(index),
            "iron_ore": [1.0] * len(index),
            "steel": [1.0] * len(index),
            "lime": [1.0] * len(index),
            "co2": [1.0] * len(index),
        },
    )
    strategy = IndustrialHybridEomStrategy(electricity_wtp=500.0)
    plant = SteelPlant(
        id="steel_hybrid",
        unit_operator="operator",
        bidding_strategies={"EOM": strategy},
        forecaster=forecaster,
        components=steel_electrolyser_buffer_components(),
        demand=4.0,
        dsm_optimisation_config={
            "horizon_mode": "rolling_horizon",
            "look_ahead_horizon": "4h",
            "commit_horizon": "4h",
            "rolling_step": "4h",
        },
    )
    plant.setup_model()
    return plant, strategy


def next_commit_products(plant):
    return [
        Product(start, start + pd.Timedelta(hours=1)) for start in plant.index[:2]
    ]


def test_direct_hybrid_bid_price_includes_gas_and_co2_costs():
    unit = make_unit()
    unit.forecaster = SimpleNamespace(
        fuel_prices={"natural_gas": object(), "co2": object()}
    )
    unit.natural_gas_price = FastSeries(index=unit.index, value=[30.0, 30.0])
    unit.co2_price = FastSeries(index=unit.index, value=[90.0, 90.0])
    unit.forecaster.get_price = lambda fuel: getattr(unit, f"{fuel}_price")
    calciner = {
        "eta_electric": 0.95,
        "eta_fossil": 0.9,
        "ng_co2_factor": 0.2,
    }

    price = electric_wtp(unit, unit.index[0], "cement_direct", calciner, None)

    assert price == pytest.approx((30 + 90 * 0.2) / 0.9 * 0.95)


def test_direct_rule_uses_gas_for_rejected_electric_heat():
    strategy = IndustrialHybridEomStrategy()
    unit = make_unit()
    start = unit.index[0]
    market = MarketConfig(market_id="EOM")
    strategy.pending_bid_contexts[(unit.id, "EOM")] = {
        "route": "cement_direct",
        "process": {"eta_electric": 0.9, "eta_fossil": 0.75},
        "storage": None,
        "initial_states": {},
        "products": {
            start: {
                "global_t": 0,
                "power": 10.0,
                "flexible_power": 6.0,
                "production": 100.0,
                "storage_discharge": 0.0,
            }
        },
    }

    strategy.on_market_feedback(
        unit,
        market,
        [{"start_time": start, "accepted_volume": -7.0}],
    )

    # Four MW of non-replaceable load are served first; the three MW of
    # rejected direct electric heat are replaced by natural gas.
    assert unit.outputs["gas_fallback"].at[start] == pytest.approx(3.6)
    assert unit.outputs["unserved_clinker"].at[start] == pytest.approx(0.0)
    assert unit._rh_full_horizon_production[0] == pytest.approx(100.0)


def test_etes_rule_tracks_soc_and_replaces_missing_stored_heat_with_gas():
    strategy = IndustrialHybridEomStrategy()
    unit = make_unit()
    start = unit.index[0]
    market = MarketConfig(market_id="EOM")
    strategy.pending_bid_contexts[(unit.id, "EOM")] = {
        "route": "cement_etes",
        "process": {"eta_fossil": 0.9},
        "storage": {
            "capacity": 10.0,
            "initial_soc": 0.0,
            "eta_electric": 1.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "storage_loss_rate": 0.0,
        },
        "initial_states": {"thermal_storage": {"soc": 0.0}},
        "products": {
            start: {
                "global_t": 0,
                "power": 10.0,
                "flexible_power": 6.0,
                "production": 100.0,
                "storage_discharge": 5.0,
            }
        },
    }

    strategy.on_market_feedback(
        unit,
        market,
        [{"start_time": start, "accepted_volume": -7.0}],
    )

    # Three MW reach the electric heater after the four MW base load.  The
    # scheduled 5 MWh_th discharge therefore needs 2 MWh_th of gas heat.
    assert unit.outputs["gas_fallback"].at[start] == pytest.approx(2 / 0.9)
    assert unit.outputs["thermal_storage_soc"].at[start] == pytest.approx(0.0)
    assert unit._rh_init_states["thermal_storage"]["soc"] == pytest.approx(0.0)


def test_rejected_nonreplaceable_load_is_reported_as_unserved_clinker():
    strategy = IndustrialHybridEomStrategy()
    unit = make_unit()
    start = unit.index[0]
    market = MarketConfig(market_id="EOM")
    strategy.pending_bid_contexts[(unit.id, "EOM")] = {
        "route": "cement_direct",
        "process": {"eta_electric": 0.9, "eta_fossil": 0.75},
        "storage": None,
        "initial_states": {},
        "products": {
            start: {
                "global_t": 0,
                "power": 10.0,
                "flexible_power": 6.0,
                "production": 100.0,
                "storage_discharge": 0.0,
            }
        },
    }

    strategy.on_market_feedback(
        unit,
        market,
        [{"start_time": start, "accepted_volume": -2.0}],
    )

    assert unit.outputs["unserved_clinker"].at[start] == pytest.approx(50.0)
    assert unit._rh_full_horizon_production[0] == pytest.approx(50.0)


def test_direct_hybrid_uses_market_feedback_on_a_real_rolling_cement_plant():
    components = {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": 0.01,
            "fuel_type": "both",
            "eta_electric": 0.95,
            "eta_fossil": 0.90,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        }
    }
    plant, strategy = make_rolling_horizon_cement_plant(
        components, [1.0, 1.0, 300.0, 300.0], demand=4.0
    )
    market = MarketConfig(market_id="EOM")
    bids = strategy.calculate_bids(plant, market, next_commit_products(plant))

    assert bids
    bid_context = strategy.pending_bid_contexts[(plant.id, market.market_id)]
    cleared_orders = []
    for bid in bids:
        item = bid_context["products"][bid["start_time"]]
        # Serve auxiliary load but reject half of the direct electric heat.
        accepted_power = item["power"] - item["flexible_power"] / 2
        cleared_orders.append(
            {**bid, "accepted_volume": -accepted_power, "accepted_price": 0.0}
        )

    plant.set_dispatch_plan(market, cleared_orders)

    assert plant.outputs["gas_fallback"].loc[plant.index[:2]].sum() > 0
    assert plant.outputs["unserved_clinker"].loc[plant.index[:2]].sum() == pytest.approx(
        0.0
    )


def test_etes_hybrid_updates_carried_soc_on_a_real_rolling_cement_plant():
    components = {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": 0.01,
            "fuel_type": "fossil",
            "eta_fossil": 0.90,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
        "thermal_storage": {
            "capacity": 20.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 20.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "initial_soc": 0.0,
            "storage_type": "short-term_with_generator",
            "eta_electric": 0.97,
        },
    }
    plant, strategy = make_rolling_horizon_cement_plant(
        components, [1.0, 1.0, 300.0, 300.0], demand=4.0
    )
    market = MarketConfig(market_id="EOM")
    bids = strategy.calculate_bids(plant, market, next_commit_products(plant))

    assert bids
    plant.set_dispatch_plan(
        market,
        [
            {**bid, "accepted_volume": bid["volume"], "accepted_price": 0.0}
            for bid in bids
        ],
    )

    assert "soc" in plant._rh_init_states["thermal_storage"]
    assert 0.0 <= plant._rh_init_states["thermal_storage"]["soc"] <= 1.0


def fully_electric_etes_components(auxiliary_power=0.1):
    return {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": auxiliary_power,
            "fuel_type": "electricity",
            "eta_electric": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
        "thermal_storage": {
            "capacity": 20.0,
            "max_power_charge": 5.0,
            "max_power_discharge": 20.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
            "initial_soc": 0.0,
            "storage_type": "short-term_with_generator",
            "eta_electric": 1.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
        },
    }


def test_fully_electric_etes_eom_bids_at_the_electricity_forecast():
    plant, strategy = make_rolling_horizon_cement_plant(
        fully_electric_etes_components(), [10.0, 20.0, 30.0, 40.0], 4.0,
        commit_hours=4,
    )
    plant.forecaster.fuel_prices = {}
    products = [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index]

    bids = strategy.calculate_bids(plant, MarketConfig(market_id="EOM"), products)

    assert [bid["price"] for bid in bids] == pytest.approx([10.0, 20.0, 30.0, 40.0])


def test_fully_electric_etes_eom_feedback_prioritises_core_load():
    strategy = IndustrialHybridEomStrategy()
    unit = make_unit()
    start = unit.index[0]
    market = MarketConfig(market_id="EOM")
    strategy.pending_bid_contexts[(unit.id, "EOM")] = {
        "route": "cement_electric_etes",
        "process": {"specific_heat_demand": 1.0},
        "storage": {
            "capacity": 10.0,
            "initial_soc": 0.0,
            "eta_electric": 1.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "storage_loss_rate": 0.0,
        },
        "initial_states": {"thermal_storage": {"soc": 0.0}},
        "products": {
            start: {
                "global_t": 0,
                "power": 10.0,
                "flexible_power": 6.0,
                "production": 100.0,
                "storage_discharge": 5.0,
            }
        },
    }

    strategy.on_market_feedback(
        unit, market, [{"start_time": start, "accepted_volume": -7.0}]
    )

    assert unit.outputs["gas_fallback"].at[start] == pytest.approx(0.0)
    assert unit.outputs["unserved_clinker"].at[start] == pytest.approx(2.0)
    assert unit._rh_full_horizon_production[0] == pytest.approx(98.0)
    assert unit.outputs["thermal_storage_soc"].at[start] == pytest.approx(0.0)


def test_fully_electric_etes_otc_bids_auxiliary_below_forecast_eom_price():
    plant, _eom = make_rolling_horizon_cement_plant(
        fully_electric_etes_components(), [10.0, 20.0, 30.0, 40.0], 4.0,
        commit_hours=4,
    )
    strategy = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    market = MarketConfig(
        market_id="LTM_OTC", product_type="energy", price_tick=1.0
    )

    bids = strategy.calculate_bids(plant, market, four_hour_product(plant))

    assert [bid["industrial_hybrid_tranche"] for bid in bids] == ["auxiliary"]
    assert bids[0]["volume"] == pytest.approx(-0.1)
    assert bids[0]["price"] == pytest.approx(24.0)

    no_lower_price_market = MarketConfig(
        market_id="LTM_OTC_minimum",
        product_type="energy",
        minimum_bid_price=25.0,
        price_tick=1.0,
    )
    assert strategy.calculate_bids(
        plant, no_lower_price_market, four_hour_product(plant)
    ) == []


def test_fully_electric_etes_otc_reduces_eom_residual_procurement():
    plant, eom = make_rolling_horizon_cement_plant(
        fully_electric_etes_components(), [10.0, 20.0, 30.0, 40.0], 4.0,
        commit_hours=4,
    )
    otc = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    otc_market = MarketConfig(
        market_id="LTM_OTC", product_type="energy", price_tick=1.0
    )
    otc_bids = otc.calculate_bids(plant, otc_market, four_hour_product(plant))
    otc.on_market_feedback(
        plant, otc_market, [{**otc_bids[0], "accepted_volume": otc_bids[0]["volume"]}]
    )
    eom_market = MarketConfig(market_id="EOM")
    eom_bids = eom.calculate_bids(
        plant,
        eom_market,
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )
    bid_context = eom.pending_bid_contexts[(plant.id, eom_market.market_id)]

    assert all(
        plant.industrial_hybrid_otc_procurement[timestamp] == pytest.approx(0.1)
        for timestamp in plant.index
    )
    assert [bid["volume"] for bid in eom_bids] == pytest.approx(
        [
            -(bid_context["products"][bid["start_time"]]["power"] - 0.1)
            for bid in eom_bids
        ]
    )


def test_fully_electric_etes_capacity_neg_uses_preview_cost_and_reservation():
    plant, eom = make_rolling_horizon_cement_plant(
        fully_electric_etes_components(), [1.0, 1.0, 100.0, 100.0], 4.0,
        commit_hours=4,
    )
    capacity = IndustrialHybridCapacityNegStrategy()
    crm_market = MarketConfig(
        market_id="CRM_neg",
        product_type="capacity_neg",
        maximum_bid_volume=4.8,
        volume_tick=0.1,
    )

    bids = capacity.calculate_bids(plant, crm_market, four_hour_product(plant))

    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(4.8)
    item = capacity.pending_bid_contexts[(plant.id, crm_market.market_id)]["products"][
        bids[0]["start_time"]
    ]
    commitments = {
        timestamp: [{key: value for key, value in item.items()}]
        for timestamp in item["hours"]
    }
    normal_cost = plant.preview_rolling_schedule_cost(plant.index[0])
    reserved_cost = plant.preview_rolling_schedule_cost(
        plant.index[0],
        constraint_builder=capacity_neg_commitment_constraint(
            plant, item["route"], commitments
        ),
    )
    expected_price = max(
        0.0,
        (reserved_cost - normal_cost)
        / (item["volume"] * len(item["hours"])),
    )
    assert bids[0]["price"] == pytest.approx(expected_price)

    capacity.on_market_feedback(
        plant, crm_market, [{**bids[0], "accepted_volume": bids[0]["volume"]}]
    )
    eom.calculate_bids(
        plant,
        MarketConfig(market_id="EOM"),
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )

    rows = component_operations_by_step(plant)
    assert all(
        rows[hour]["thermal_storage_power_input"] <= 0.2 + 1e-9
        for hour in range(4)
    )


def test_fully_electric_etes_capacity_neg_keeps_soc_space_across_blocks():
    plant, _eom = make_rolling_horizon_cement_plant(
        fully_electric_etes_components(), [1.0, 1.0, 100.0, 100.0], 4.0,
        commit_hours=4,
    )
    capacity = IndustrialHybridCapacityNegStrategy()
    crm_market = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    products = [
        Product(plant.index[0], plant.index[0] + pd.Timedelta(hours=2)),
        Product(plant.index[2], plant.index[2] + pd.Timedelta(hours=2)),
    ]

    bids = capacity.calculate_bids(plant, crm_market, products)

    assert [bid["volume"] for bid in bids] == pytest.approx([5.0, 5.0])


def test_fully_electric_etes_skips_an_infeasible_capacity_product():
    components = fully_electric_etes_components()
    components["calciner"]["max_heat_out"] = 1.0
    plant, _eom = make_rolling_horizon_cement_plant(
        components, [1.0, 1.0, 100.0, 100.0], 6.0, commit_hours=4
    )
    plant.forecaster.clinker_demand = FastSeries(
        index=plant.index, value=[1.0, 1.0, 2.0, 2.0]
    )
    capacity = IndustrialHybridCapacityNegStrategy()

    bids = capacity.calculate_bids(
        plant,
        MarketConfig(market_id="CRM_neg", product_type="capacity_neg"),
        four_hour_product(plant),
    )

    assert bids == []


def four_hour_product(plant):
    return [Product(plant.index[0], plant.index[0] + pd.Timedelta(hours=4))]


def test_direct_capacity_neg_uses_block_minimum_and_opportunity_cost():
    components = {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": 0.01,
            "fuel_type": "both",
            "eta_electric": 1.0,
            "eta_fossil": 1.0,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        }
    }
    plant, strategy = make_rolling_horizon_cement_plant(
        components, [10.0, 20.0, 30.0, 40.0], demand=4.0, commit_hours=4
    )
    capacity = IndustrialHybridCapacityNegStrategy()
    market = MarketConfig(
        market_id="CRM_neg",
        product_type="capacity_neg",
        maximum_bid_volume=0.8,
        volume_tick=0.1,
        price_tick=0.1,
    )

    bids = capacity.calculate_bids(plant, market, four_hour_product(plant))

    assert len(bids) == 1
    # Four hourly clinker outputs require 1 MW electric heat each; the product
    # is one firm 4-hour bid and is capped by the configured market maximum.
    assert bids[0]["volume"] == pytest.approx(0.8)
    # WTP is gas plus CO2 (504.04 EUR/MWh); price is its four-hour mean margin.
    assert bids[0]["price"] == pytest.approx(479.0)


def test_etes_capacity_neg_chains_soc_across_consecutive_blocks():
    components = {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": 0.01,
            "fuel_type": "fossil",
            "eta_fossil": 0.9,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
        "thermal_storage": {
            "capacity": 8.0,
            "max_power_charge": 4.0,
            "max_power_discharge": 20.0,
            "ramp_up": 4.0,
            "ramp_down": 4.0,
            "initial_soc": 0.0,
            "storage_type": "short-term_with_generator",
            "eta_electric": 1.0,
        },
    }
    plant, eom = make_rolling_horizon_cement_plant(
        components, [1.0, 1.0, 1.0, 1.0], demand=4.0, commit_hours=4
    )
    capacity = IndustrialHybridCapacityNegStrategy()
    market = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    products = [
        Product(plant.index[0], plant.index[0] + pd.Timedelta(hours=2)),
        Product(plant.index[2], plant.index[2] + pd.Timedelta(hours=2)),
    ]

    bids = capacity.calculate_bids(plant, market, products)

    assert [bid["volume"] for bid in bids] == pytest.approx([4.0])
    bid_context = capacity.pending_bid_contexts[(plant.id, market.market_id)]
    assert bid_context["products"][products[1].start]["volume"] == pytest.approx(0.0)
    capacity.on_market_feedback(
        plant, market, [{**bids[0], "accepted_volume": bids[0]["volume"]}]
    )
    eom.calculate_bids(
        plant,
        MarketConfig(market_id="EOM"),
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )
    rows = component_operations_by_step(plant)
    assert all(rows[hour]["thermal_storage_power_input"] == pytest.approx(0.0) for hour in range(2))


def test_accepted_capacity_forces_reservation_aware_eom_plan():
    components = {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": 0.01,
            "fuel_type": "both",
            "eta_electric": 1.0,
            "eta_fossil": 1.0,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        }
    }
    plant, eom = make_rolling_horizon_cement_plant(
        components, [1.0, 1.0, 1.0, 1.0], demand=4.0, commit_hours=4
    )
    capacity = IndustrialHybridCapacityNegStrategy()
    crm = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    capacity_bids = capacity.calculate_bids(plant, crm, four_hour_product(plant))
    capacity.on_market_feedback(
        plant, crm, [{**capacity_bids[0], "accepted_volume": capacity_bids[0]["volume"]}]
    )
    eom_market = MarketConfig(market_id="EOM")
    eom_products = [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index]

    eom.calculate_bids(plant, eom_market, eom_products)

    rows = component_operations_by_step(plant)
    assert all(
        rows[global_t]["calciner_power_input"] <= 0.0
        for global_t in range(4)
    )


def test_rejected_capacity_leaves_the_normal_eom_schedule_unreserved():
    components = {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": 0.01,
            "fuel_type": "both",
            "eta_electric": 1.0,
            "eta_fossil": 1.0,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        }
    }
    plant, eom = make_rolling_horizon_cement_plant(
        components, [1.0, 1.0, 1.0, 1.0], demand=4.0, commit_hours=4
    )
    capacity = IndustrialHybridCapacityNegStrategy()
    crm = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    capacity_bids = capacity.calculate_bids(plant, crm, four_hour_product(plant))
    capacity.on_market_feedback(
        plant, crm, [{**capacity_bids[0], "accepted_volume": 0.0}]
    )

    eom_market = MarketConfig(market_id="EOM")
    eom_products = [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index]
    eom.calculate_bids(plant, eom_market, eom_products)
    rows = component_operations_by_step(plant)

    assert any(rows[global_t]["calciner_power_input"] > 0.0 for global_t in range(4))


def test_capacity_neg_rejects_unsupported_routes_and_out_of_window_products():
    unit = make_unit()
    unit.technology = "steel_plant"
    unit.horizon_mode = "rolling_horizon"
    strategy = IndustrialHybridCapacityNegStrategy()
    market = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    with pytest.raises(ValueError, match="compatible cement hybrid route"):
        strategy.calculate_bids(unit, market, [])

    rolling_unit = SimpleNamespace(
        index=FastIndex("2023-01-01", periods=4, freq="1h"),
        _rh_commit="2h",
        _parse_duration_to_steps=lambda _duration: 2,
    )
    with pytest.raises(ValueError, match="entirely inside"):
        strategy.validate_capacity_products(
            rolling_unit,
            [Product(rolling_unit.index[0], rolling_unit.index[0] + pd.Timedelta(hours=3))],
        )


def direct_hybrid_components(auxiliary_power=0.1):
    return {
        "calciner": {
            "max_heat_out": 5.0,
            "specific_heat_demand": 1.0,
            "specific_electricity_aux": auxiliary_power,
            "fuel_type": "both",
            "eta_electric": 1.0,
            "eta_fossil": 1.0,
            "fossil_ng_share": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        }
    }


def test_direct_otc_bids_firm_auxiliary_and_process_tranches():
    plant, _eom = make_rolling_horizon_cement_plant(
        direct_hybrid_components(), [10.0, 20.0, 700.0, 800.0], 4.0, commit_hours=4
    )
    strategy = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    market = MarketConfig(
        market_id="LTM_OTC",
        product_type="energy",
        maximum_bid_volume=0.8,
        maximum_bid_price=600.0,
        volume_tick=0.1,
        price_tick=0.1,
    )

    bids = strategy.calculate_bids(plant, market, four_hour_product(plant))

    by_tranche = {bid["industrial_hybrid_tranche"]: bid for bid in bids}
    assert by_tranche["auxiliary"]["volume"] == pytest.approx(-0.1)
    assert by_tranche["auxiliary"]["price"] == pytest.approx(600.0)
    # The direct process tranche is limited by the configured 0.8 MW bid cap.
    assert by_tranche["direct_process"]["volume"] == pytest.approx(-0.8)
    # Its ceiling averages min(EOM forecast, electric WTP) over the full block.
    assert by_tranche["direct_process"]["price"] == pytest.approx(259.5)


def test_etes_otc_bids_auxiliary_tranche_only():
    components = {
        **direct_hybrid_components(),
        "thermal_storage": {
            "capacity": 20.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 20.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "initial_soc": 0.0,
            "storage_type": "short-term_with_generator",
            "eta_electric": 1.0,
        },
    }
    components["calciner"]["fuel_type"] = "fossil"
    plant, _eom = make_rolling_horizon_cement_plant(
        components, [10.0, 20.0, 30.0, 40.0], 4.0, commit_hours=4
    )
    strategy = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    market = MarketConfig(market_id="LTM_OTC", product_type="energy")

    bids = strategy.calculate_bids(plant, market, four_hour_product(plant))

    assert [bid["industrial_hybrid_tranche"] for bid in bids] == ["auxiliary"]
    assert bids[0]["volume"] == pytest.approx(-0.1)


def test_accepted_otc_procurement_forces_eom_and_reduces_residual_bids():
    plant, eom = make_rolling_horizon_cement_plant(
        direct_hybrid_components(), [1.0, 1.0, 1.0, 1.0], 4.0, commit_hours=4
    )
    otc = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    otc_market = MarketConfig(market_id="LTM_OTC", product_type="energy")
    otc_bids = otc.calculate_bids(plant, otc_market, four_hour_product(plant))
    process_bid = next(
        bid for bid in otc_bids if bid["industrial_hybrid_tranche"] == "direct_process"
    )
    otc.on_market_feedback(
        plant,
        otc_market,
        [{**process_bid, "accepted_volume": process_bid["volume"] / 2}],
    )

    eom_bids = eom.calculate_bids(
        plant,
        MarketConfig(market_id="EOM"),
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )

    assert all(
        plant.industrial_hybrid_otc_procurement[timestamp] == pytest.approx(0.5)
        for timestamp in plant.index
    )
    # The re-solved total demand is 1.1 MW, so the 0.5 MW OTC contract leaves
    # exactly 0.6 MW to procure on the EOM in each hour.
    assert [bid["volume"] for bid in eom_bids] == pytest.approx([-0.6] * 4)


def test_rejected_otc_leaves_eom_schedule_unchanged_and_missing_prices_skip_bid():
    plant, eom = make_rolling_horizon_cement_plant(
        direct_hybrid_components(), [1.0, 1.0, 1.0, 1.0], 4.0, commit_hours=4
    )
    otc = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    otc_market = MarketConfig(market_id="LTM_OTC", product_type="energy")
    otc_bids = otc.calculate_bids(plant, otc_market, four_hour_product(plant))
    otc.on_market_feedback(
        plant,
        otc_market,
        [{**bid, "accepted_volume": 0.0} for bid in otc_bids],
    )
    eom_bids = eom.calculate_bids(
        plant,
        MarketConfig(market_id="EOM"),
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )

    assert [bid["volume"] for bid in eom_bids] == pytest.approx([-1.1] * 4)
    outside_forecast = [Product(plant.index[0], plant.index[-1] + pd.Timedelta(hours=2))]
    assert otc.calculate_bids(plant, otc_market, outside_forecast) == []

    plant.forecaster.electricity_price.at[plant.index[2]] = float("nan")
    assert otc.calculate_bids(plant, otc_market, four_hour_product(plant)) == []


def test_otc_and_crm_constraints_are_combined_for_eom_bidding():
    plant, eom = make_rolling_horizon_cement_plant(
        direct_hybrid_components(), [1.0, 1.0, 1.0, 1.0], 4.0, commit_hours=4
    )
    otc = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    otc_market = MarketConfig(market_id="LTM_OTC", product_type="energy")
    otc_bids = otc.calculate_bids(plant, otc_market, four_hour_product(plant))
    process_bid = next(
        bid for bid in otc_bids if bid["industrial_hybrid_tranche"] == "direct_process"
    )
    otc.on_market_feedback(
        plant,
        otc_market,
        [{**process_bid, "accepted_volume": process_bid["volume"] / 2}],
    )

    capacity = IndustrialHybridCapacityNegStrategy()
    crm_market = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    capacity_bids = capacity.calculate_bids(plant, crm_market, four_hour_product(plant))
    assert capacity_bids[0]["volume"] == pytest.approx(0.5)
    capacity.on_market_feedback(
        plant,
        crm_market,
        [{**capacity_bids[0], "accepted_volume": capacity_bids[0]["volume"]}],
    )

    eom.calculate_bids(
        plant,
        MarketConfig(market_id="EOM"),
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )
    rows = component_operations_by_step(plant)
    assert all(rows[hour]["calciner_power_input"] <= 0.5 + 1e-9 for hour in range(4))


def test_steel_electrolyser_buffer_capacity_neg_and_eom_reservation():
    plant, eom = make_rolling_horizon_steel_plant([1.0] * 4)
    products = four_hour_product(plant)
    capacity = IndustrialHybridCapacityNegStrategy()
    crm = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")

    capacity_bids = capacity.calculate_bids(plant, crm, products)

    assert len(capacity_bids) == 1
    # The 10 MWh hydrogen buffer has room for 2.5 MW for all four hours;
    # electrolyser headroom alone would allow 4 MW.
    assert capacity_bids[0]["volume"] == pytest.approx(2.5)
    # Stored hydrogen is worth the avoided external hydrogen procurement.
    assert capacity_bids[0]["price"] == pytest.approx(99.0)

    capacity.on_market_feedback(
        plant,
        crm,
        [{**capacity_bids[0], "accepted_volume": capacity_bids[0]["volume"]}],
    )
    eom_bids = eom.calculate_bids(
        plant,
        MarketConfig(market_id="EOM"),
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )

    rows = component_operations_by_step(plant)
    assert all(rows[hour]["electrolyser_power"] <= 2.5 + 1e-9 for hour in range(4))
    assert all(bid["price"] == pytest.approx(500.0) for bid in eom_bids)


def test_steel_eom_requires_wtp_and_reports_unserved_steel_when_rejected():
    plant, _eom = make_rolling_horizon_steel_plant([1.0] * 4)
    products = [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index]

    with pytest.raises(ValueError, match="electricity_wtp"):
        IndustrialHybridEomStrategy().calculate_bids(
            plant, MarketConfig(market_id="EOM"), products
        )

    strategy = IndustrialHybridEomStrategy(electricity_wtp=500.0)
    bids = strategy.calculate_bids(plant, MarketConfig(market_id="EOM"), products)
    strategy.on_market_feedback(
        plant,
        MarketConfig(market_id="EOM"),
        [{**bid, "accepted_volume": 0.0} for bid in bids],
    )

    assert sum(float(plant.outputs["unserved_steel"].at[t]) for t in plant.index) > 0


def test_cement_otc_crm_eom_sequence_uses_assume_market_feedback():
    plant, eom = make_rolling_horizon_cement_plant(
        direct_hybrid_components(), [1.0] * 4, 4.0, commit_hours=4
    )
    otc = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    capacity = IndustrialHybridCapacityNegStrategy()
    plant.bidding_strategies.update(
        {"LTM_OTC": otc, "CRM_neg": capacity, "EOM": eom}
    )
    otc_market = MarketConfig(market_id="LTM_OTC", product_type="energy")
    crm_market = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    eom_market = MarketConfig(market_id="EOM")

    otc_bids = otc.calculate_bids(plant, otc_market, four_hour_product(plant))
    process_bid = next(
        bid for bid in otc_bids if bid["industrial_hybrid_tranche"] == "direct_process"
    )
    plant.set_dispatch_plan(
        otc_market,
        [{**process_bid, "accepted_volume": process_bid["volume"] / 2, "accepted_price": 1.0}],
    )

    capacity_bids = capacity.calculate_bids(
        plant, crm_market, four_hour_product(plant)
    )
    plant.set_dispatch_plan(
        crm_market,
        [
            {
                **capacity_bids[0],
                "accepted_volume": capacity_bids[0]["volume"],
                "accepted_price": 1.0,
            }
        ],
    )

    eom_bids = eom.calculate_bids(
        plant,
        eom_market,
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )
    plant.set_dispatch_plan(
        eom_market,
        [
            {**bid, "accepted_volume": bid["volume"], "accepted_price": 1.0}
            for bid in eom_bids
        ],
    )

    assert all(
        plant.industrial_hybrid_otc_procurement[timestamp] == pytest.approx(0.5)
        for timestamp in plant.index
    )
    assert plant.industrial_hybrid_capacity_neg_commitments
    assert sum(
        float(plant.outputs["unserved_clinker"].at[timestamp])
        for timestamp in plant.index.get_date_list()
    ) == pytest.approx(0.0)


def test_steel_otc_crm_eom_sequence_uses_assume_market_feedback():
    plant, eom = make_rolling_horizon_steel_plant([1.0] * 4)
    otc = IndustrialHybridOtcStrategy(auxiliary_security_value=800.0)
    capacity = IndustrialHybridCapacityNegStrategy()
    plant.bidding_strategies.update(
        {"LTM_OTC": otc, "CRM_neg": capacity, "EOM": eom}
    )
    otc_market = MarketConfig(market_id="LTM_OTC", product_type="energy")
    crm_market = MarketConfig(market_id="CRM_neg", product_type="capacity_neg")
    eom_market = MarketConfig(market_id="EOM")

    otc_bids = otc.calculate_bids(plant, otc_market, four_hour_product(plant))
    assert len(otc_bids) == 1
    assert otc_bids[0]["industrial_hybrid_tranche"] == "auxiliary"
    assert otc_bids[0]["volume"] == pytest.approx(-1.0)
    plant.set_dispatch_plan(
        otc_market,
        [
            {
                **otc_bids[0],
                "accepted_volume": otc_bids[0]["volume"],
                "accepted_price": 1.0,
            }
        ],
    )
    capacity_bids = capacity.calculate_bids(
        plant, crm_market, four_hour_product(plant)
    )
    plant.set_dispatch_plan(
        crm_market,
        [
            {
                **capacity_bids[0],
                "accepted_volume": capacity_bids[0]["volume"],
                "accepted_price": 1.0,
            }
        ],
    )
    eom_bids = eom.calculate_bids(
        plant,
        eom_market,
        [Product(start, start + pd.Timedelta(hours=1)) for start in plant.index],
    )
    plant.set_dispatch_plan(
        eom_market,
        [
            {**bid, "accepted_volume": bid["volume"], "accepted_price": 1.0}
            for bid in eom_bids
        ],
    )

    rows = component_operations_by_step(plant)
    assert all(rows[hour]["electrolyser_power"] <= 2.5 + 1e-9 for hour in range(4))
    assert plant.industrial_hybrid_capacity_neg_commitments
    assert all(
        plant.industrial_hybrid_otc_procurement[timestamp] == pytest.approx(1.0)
        for timestamp in plant.index
    )
    assert sum(
        float(plant.outputs["unserved_steel"].at[timestamp])
        for timestamp in plant.index.get_date_list()
    ) == pytest.approx(0.0)


def test_otc_requires_auxiliary_security_value_and_energy_market():
    strategy = IndustrialHybridOtcStrategy()
    unit = make_unit()
    unit.technology = "cement_plant"
    unit.horizon_mode = "rolling_horizon"
    unit.components = {"calciner": {"fuel_type": "both", "fossil_ng_share": 1.0}}
    with pytest.raises(ValueError, match="auxiliary_security_value"):
        strategy.calculate_bids(unit, MarketConfig(product_type="energy"), [])

    strategy = IndustrialHybridOtcStrategy(auxiliary_security_value=1.0)
    with pytest.raises(ValueError, match="energy-market"):
        strategy.calculate_bids(unit, MarketConfig(product_type="capacity_neg"), [])
