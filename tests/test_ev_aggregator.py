# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig
from assume.common.utils import datetime2timestamp
from assume.strategies.ev_aggregator import (
    EVPortfolioMemberStrategy,
    EVPortfolioStrategy,
)
from assume.units.electric_vehicle import ElectricVehicleUnit


def vehicle(
    prices=(10, 100, 50, 50),
    availability=(1, 1, 0, 1),
    trips=(0, 0, 0.2, 0),
    freq="1h",
    **kwargs,
):
    index = pd.date_range("2020-01-01", periods=len(prices), freq=freq)
    return ElectricVehicleUnit(
        id="ev",
        unit_operator="aggregator",
        technology="electric_vehicle",
        bidding_strategies={},
        forecaster=UnitForecaster(
            index, market_prices={"EOM": list(prices)}, availability=list(availability)
        ),
        capacity=1,
        max_power_charge=-1,
        max_power_discharge=1,
        initial_soc=0.5,
        trip_energy_consumption=list(trips),
        **kwargs,
    )


def test_arbitrage_driving_and_terminal_energy():
    unit = vehicle()
    plan = EVPortfolioStrategy().optimize(
        [unit], unit.index[0], unit.index[-1] + unit.index.freq
    )[unit.id]
    assert plan[unit.index[0]] == pytest.approx(-0.5)
    assert plan[unit.index[1]] == pytest.approx(0.8)
    assert plan[unit.index[2]] == 0
    for t, power in plan.items():
        unit.outputs["energy"].at[t] = power
    assert unit.energy_at(unit.index[-1] + unit.index.freq) == pytest.approx(0.5)
    assert sum(plan.values()) == pytest.approx(-0.2)


def test_shared_connection_limits_include_background():
    first = vehicle(availability=(1, 1, 1, 1), trips=(0, 0, 0, 0))
    second = vehicle(availability=(1, 1, 1, 1), trips=(0, 0, 0, 0))
    second.id = "second"
    times = list(first.index)
    plan = EVPortfolioStrategy(ev_import_limit_mw=0.5, ev_export_limit_mw=0.2).optimize(
        [first, second],
        times[0],
        times[-1] + first.index.freq,
        background_load=dict.fromkeys(times, 0.1),
    )
    net = np.array([-sum(plan[u.id][t] for u in (first, second)) + 0.1 for t in times])
    assert net.max() <= 0.5 + 1e-7
    assert net.min() >= -0.2 - 1e-7
    assert net.max() == pytest.approx(0.5)


def test_peak_charge_flattens_charging_and_import_fee_has_no_export_credit():
    unit = vehicle(
        prices=(10, 20, 30, 40), availability=(1, 1, 0, 1), trips=(0, 0, 0.4, 0)
    )
    unit.max_power_discharge = 0
    start, end = unit.index[0], unit.index[-1] + unit.index.freq
    plain = EVPortfolioStrategy().optimize([unit], start, end)[unit.id]
    peak = EVPortfolioStrategy(ev_peak_price=1000).optimize([unit], start, end)[unit.id]
    assert max(-p for p in peak.values()) < max(-p for p in plain.values())
    assert sum(peak.values()) == pytest.approx(-0.4)
    unit = vehicle(prices=(10, 100, 50, 50))
    tariff = EVPortfolioStrategy().optimize(
        [unit],
        start,
        end,
        grid_fees=dict.fromkeys(unit.index, 1000),
    )[unit.id]
    # Import-only fees eliminate arbitrage, but cannot pay for exports.
    assert sum(max(p, 0) for p in tariff.values()) == pytest.approx(0)
    assert -sum(tariff.values()) == pytest.approx(0.2)


def test_independent_and_joint_energy_objectives_match():
    units = [vehicle(), vehicle(availability=(0, 1, 1, 1), trips=(0.1, 0, 0, 0))]
    units[1].id = "other"
    start, end = units[0].index[0], units[0].index[-1] + units[0].index.freq
    strategy = EVPortfolioStrategy()
    joint = strategy.optimize(units, start, end)
    separate = {u.id: strategy.optimize([u], start, end)[u.id] for u in units}

    def cost(plan):
        return sum(
            -power * u.forecaster.price["EOM"].at[t]
            for u in units
            for t, power in plan[u.id].items()
        )

    assert cost(joint) == pytest.approx(cost(separate))


def test_contracted_capacity_allows_cheap_charging_up_to_threshold():
    unit = vehicle(
        prices=(10, 20, 30, 40), availability=(1, 1, 0, 1), trips=(0, 0, 0.4, 0)
    )
    unit.max_power_discharge = 0
    plan = EVPortfolioStrategy(ev_peak_price=1000, ev_observed_peak_mw=0.2).optimize(
        [unit],
        unit.index[0],
        unit.index[-1] + unit.index.freq,
    )[unit.id]
    assert -plan[unit.index[0]] == pytest.approx(0.2)
    assert -plan[unit.index[1]] == pytest.approx(0.2)
    assert max(-p for p in plan.values()) <= 0.2 + 1e-7


@pytest.mark.parametrize(
    "params",
    [
        {"ev_import_limit_mw": -1},
        {"ev_peak_price": float("nan")},
        {"ev_export_limit_mw": float("inf")},
    ],
)
def test_invalid_portfolio_limits(params):
    with pytest.raises(ValueError, match="finite/nonnegative"):
        EVPortfolioStrategy(**params)


def test_half_hour_efficiency_and_idempotent_dispatch():
    unit = vehicle(
        prices=(1, 2, 3, 4),
        availability=(1, 0, 1, 1),
        trips=(0, 0.1, 0, 0),
        freq="30min",
        efficiency_charge=0.8,
    )
    unit.outputs["energy"].at[unit.index[0]] = -0.5
    unit.outputs["energy"].at[unit.index[1]] = -1  # unplugged dispatch is clipped
    unit.execute_current_dispatch(unit.index[0], unit.index[-1])
    first = unit.outputs["soc"].copy()
    unit.execute_current_dispatch(unit.index[0], unit.index[-1])
    assert np.allclose(first, unit.outputs["soc"])
    assert unit.energy_at(unit.index[2]) == pytest.approx(0.6)


def test_storage_interface_respects_plug_availability():
    unit = vehicle()
    start, end = unit.index[2], unit.index[3]
    assert all(
        np.allclose(limit, 0) for limit in unit.calculate_min_max_charge(start, end)
    )
    assert all(
        np.allclose(limit, 0) for limit in unit.calculate_min_max_discharge(start, end)
    )


def test_tariff_does_not_commit_energy_and_rejected_bid_does():
    unit = vehicle()
    order = {
        "start_time": unit.index[0],
        "end_time": unit.index[1],
        "accepted_volume": -0.4,
        "accepted_price": 50,
    }
    unit.set_dispatch_plan(MarketConfig(product_type="grid_fee"), [order])
    assert unit.energy_at(unit.index[1]) == 0.5
    assert unit.outputs["energy_committed"].at[unit.index[0]] == 0
    order["accepted_volume"] = 0
    unit.set_dispatch_plan(MarketConfig(product_type="energy"), [order])
    plan = EVPortfolioStrategy().optimize(
        [unit], unit.index[0], unit.index[-1] + unit.index.freq
    )
    assert plan[unit.id][unit.index[0]] == pytest.approx(0)


def test_negative_prices_cannot_create_simultaneous_losses():
    unit = vehicle(
        prices=(-100, -100, -100, -100),
        availability=(1, 1, 1, 1),
        trips=(0, 0, 0, 0),
        efficiency_charge=0.8,
        efficiency_discharge=0.8,
    )
    plan = EVPortfolioStrategy().optimize(
        [unit], unit.index[0], unit.index[-1] + unit.index.freq
    )[unit.id]
    for t, power in plan.items():
        unit.outputs["energy"].at[t] = power
    unit.execute_current_dispatch(unit.index[0], unit.index[-1])
    assert unit.energy_at(unit.index[-1] + unit.index.freq) >= 0.5 - 1e-7
    assert np.allclose(list(plan.values()), unit.outputs["energy"])


def test_infeasible_trip_is_reported():
    unit = vehicle(trips=(0, 0, 2, 0))
    with pytest.raises(RuntimeError, match="infeasible"):
        EVPortfolioStrategy().optimize(
            [unit], unit.index[0], unit.index[-1] + unit.index.freq
        )


@pytest.mark.parametrize("mode", ["rolling_horizon", "perfect_foresight"])
def test_operator_bids_preserve_unit_and_node(mode):
    unit = vehicle()
    operator = SimpleNamespace(
        units={unit.id: unit},
        context=SimpleNamespace(
            current_timestamp=datetime2timestamp(unit.index[0] - unit.index.freq),
            addr="operator",
        ),
    )
    strategy = EVPortfolioStrategy(ev_horizon_mode=mode)
    products = [(unit.index[0], unit.index[1], None)]
    bids = strategy.calculate_bids(
        operator, MarketConfig(market_id="EOM", product_type="energy"), products
    )
    assert bids[0]["unit_id"] == unit.id
    assert bids[0]["node"] == unit.node
    assert bids[0]["volume"] < 0
    assert unit.outputs["energy"].at[unit.index[0]] == 0


def test_invalid_availability_and_plugged_driving():
    with pytest.raises(ValueError, match="binary"):
        vehicle(availability=(0.5, 1, 0, 1))
    with pytest.raises(ValueError, match="unplugged"):
        vehicle(trips=(0.1, 0, 0.2, 0))


def test_tariff_feedback_updates_price_and_changes_plan():
    from assume.common.forecast_algorithms import price_plus_grid_fee

    unit = vehicle()
    strategy = EVPortfolioStrategy()
    before = strategy.optimize([unit], unit.index[0], unit.index[-1] + unit.index.freq)
    market = MarketConfig(product_type="grid_fee")
    order = {
        "start_time": unit.index[0],
        "end_time": unit.index[1],
        "accepted_volume": 0,
        "accepted_price": 200,
    }
    unit.set_dispatch_plan(market, [order])
    EVPortfolioMemberStrategy().calculate_reward(unit, market, [order])
    # Explicitly publish zero for subsequent hours, rather than persist 200.
    unit.outputs["grid_fee_announced"].loc[unit.index[1] : unit.index[-1]] = 1
    unit.forecaster.price = price_plus_grid_fee(unit.forecaster.price, {}, unit=unit)
    after = strategy.optimize([unit], unit.index[0], unit.index[-1] + unit.index.freq)
    assert after[unit.id][unit.index[0]] > before[unit.id][unit.index[0]]
    assert unit.outputs["energy"].at[unit.index[0]] == 0


def test_csv_example_loads_independent_evs():
    from assume.scenario.loader_csv import (
        load_config_and_create_forecaster,
        setup_world,
    )
    from assume.world import World

    world = World()
    world.scenario_data = load_config_and_create_forecaster(
        inputs_path="examples/inputs",
        scenario="example_lv_tariff",
        study_case="no_tariff",
    )
    setup_world(world)
    world._validate_setup()
    operator = world.unit_operators["aggregator_operator"]
    assert len(operator.units) == 6
    assert all(isinstance(u, ElectricVehicleUnit) for u in operator.units.values())
    assert isinstance(operator.portfolio_strategies["EOM"], EVPortfolioStrategy)


def test_short_horizon_differs_from_perfect_foresight():
    unit = vehicle(
        prices=(50, 50, 1, 100), availability=(1, 1, 1, 1), trips=(0, 0, 0, 0)
    )
    operator = SimpleNamespace(
        units={unit.id: unit},
        context=SimpleNamespace(
            current_timestamp=datetime2timestamp(unit.index[0] - unit.index.freq),
            addr="operator",
        ),
    )
    product = [(unit.index[0], unit.index[1], None)]
    market = MarketConfig(market_id="EOM", product_type="energy")
    rolling = EVPortfolioStrategy(ev_look_ahead_horizon="1h").calculate_bids(
        operator, market, product
    )
    perfect = EVPortfolioStrategy(ev_horizon_mode="perfect_foresight").calculate_bids(
        operator, market, product
    )
    assert rolling[0]["volume"] == pytest.approx(0)
    assert perfect[0]["volume"] > 0


def test_rolling_window_extends_through_return_and_recharge():
    unit = vehicle(
        prices=(10, 10, 10, 10), availability=(1, 0, 0, 1), trips=(0, 0.4, 0.4, 0)
    )
    operator = SimpleNamespace(
        units={unit.id: unit},
        context=SimpleNamespace(
            current_timestamp=datetime2timestamp(unit.index[0] - unit.index.freq),
            addr="operator",
        ),
    )
    bids = EVPortfolioStrategy(ev_look_ahead_horizon="3h").calculate_bids(
        operator,
        MarketConfig(market_id="EOM", product_type="energy"),
        [(unit.index[0], unit.index[1], None)],
    )
    assert bids[0]["volume"] <= -0.3 + 1e-7
    assert unit.outputs["ev_planned_energy"].at[unit.index[3]] < 0


def test_portfolio_keeps_vehicle_availability_separate():
    first = vehicle()
    second = vehicle(availability=(0, 1, 1, 1), trips=(0.1, 0, 0, 0))
    second.id = "second"
    plan = EVPortfolioStrategy().optimize(
        [first, second], first.index[0], first.index[-1] + first.index.freq
    )
    assert plan[first.id][first.index[0]] < 0
    assert plan[second.id][second.index[0]] == 0


def test_published_products_beyond_simulation_are_ignored():
    unit = vehicle()
    operator = SimpleNamespace(
        units={unit.id: unit},
        context=SimpleNamespace(
            current_timestamp=datetime2timestamp(unit.index[-2]), addr="operator"
        ),
    )
    bids = EVPortfolioStrategy().calculate_bids(
        operator,
        MarketConfig(product_type="grid_fee"),
        [
            (
                unit.index[-1] + unit.index.freq,
                unit.index[-1] + 2 * unit.index.freq,
                None,
            )
        ],
    )
    assert bids == []
