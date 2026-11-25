# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.strategies import (
    StorageCapacityHeuristicBalancingNegStrategy,
    StorageCapacityHeuristicBalancingPosStrategy,
    StorageEnergyHeuristicFlexableStrategy,
)
from assume.strategies.flexable_storage import (
    StorageEnergyHeuristicRedispatchStrategy,
    StorageEnergyOpportunityCostRedispatchStrategy,
)
from assume.units import Storage


def assert_allclose_with_indices(actual, expected, atol):
    assert len(actual) == len(expected), (
        f"Expected {len(expected)} values, got {len(actual)}"
    )

    mismatches = [
        (index, expected_value, actual_value)
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected))
        if not np.isclose(actual_value, expected_value, atol=atol)
    ]
    assert not mismatches, "Values not close: " + ", ".join(
        f"index {index}: expected {expected_value}, actual {actual_value}"
        for index, expected_value, actual_value in mismatches
    )


# run all tests with two versions of handling simultaneous charge and discharge: True and False
# False is the new default (10/2026) ramping behavior, True is the previous default behavior. The tests should pass for both versions.
@pytest.fixture(
    params=[False, True],
    ids=[
        "no-simultaneous-charge-discharge_ramping",
        "simultaneous-charge-discharge_ramping",
    ],
)
def storage(request: pytest.FixtureRequest) -> Storage:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=48, freq="h")
    forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=1000,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        additional_cost_charge=3,
        additional_cost_discharge=4,
        additional_cost=1,
        simultaneous_charge_discharge_ramping=request.param,
    )


def test_redispatch_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyOpportunityCostRedispatchStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]

    # constant EOM price of 50
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)

    assert len(bids) == 1
    bid = bids[0]

    # cost-neutral prices ("sell" valuation): the stored energy is worth what it
    # would be sold for later, 50 * eta_discharge per MWh inside the storage
    value = 50 * storage.efficiency_discharge
    expected_discharge = (
        value / storage.efficiency_discharge + storage.additional_cost_discharge
    )
    expected_charge = value * storage.efficiency_charge - storage.additional_cost_charge
    assert bid["stored_energy_value"] == pytest.approx(value)
    assert bid["price_discharge"] == pytest.approx(expected_discharge)
    assert bid["price_charge"] == pytest.approx(expected_charge)
    assert bid["price"] == pytest.approx(expected_discharge)
    # full nameplate power span as p_nom (charge is negative, so this adds)
    assert bid["p_nom"] == pytest.approx(
        storage.max_power_discharge - storage.max_power_charge
    )
    # idle storage: zero baseline and the full power range in both directions
    assert bid["volume"] == 0
    assert bid["max_power"] == pytest.approx(storage.max_power_discharge)
    assert bid["min_power"] == pytest.approx(storage.max_power_charge)
    assert bid["soc"] == pytest.approx(storage.initial_soc)
    # capacity, SOC limits and efficiencies come from the grid data
    assert "capacity" not in bid
    assert "node" in bid

    # near-empty storage: the power range stays, the SOC goes to the clearing
    storage.outputs["soc"].at[start] = 0.05
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert bids[0]["max_power"] == pytest.approx(storage.max_power_discharge)
    assert bids[0]["soc"] == pytest.approx(0.05)


def test_redispatch_storage_refill_valuation(mock_market_config, storage):
    """With "refill" the stored energy is worth what it costs to buy it back."""
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    strategy = StorageEnergyOpportunityCostRedispatchStrategy(
        stored_energy_valuation="refill"
    )
    bid = strategy.calculate_bids(
        storage,
        mock_market_config,
        product_tuples=[(start, start + timedelta(hours=1), None)],
    )[0]

    value = 50 / storage.efficiency_charge
    assert bid["stored_energy_value"] == pytest.approx(value)
    assert bid["price_discharge"] == pytest.approx(
        value / storage.efficiency_discharge + storage.additional_cost_discharge
    )
    assert bid["price_charge"] == pytest.approx(50 - storage.additional_cost_charge)


def test_redispatch_storage_invalid_valuation():
    with pytest.raises(ValueError, match="stored_energy_valuation"):
        StorageEnergyOpportunityCostRedispatchStrategy(stored_energy_valuation="avg")


def test_redispatch_storage_baseline_is_runnable_power(mock_market_config, storage):
    """
    The redispatch baseline (grid injection) is what the storage can actually run of
    its DA plan, not the committed plan. Otherwise the grid model would compute line
    flows with an injection that never happens.
    """
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyOpportunityCostRedispatchStrategy()
    product_tuples = [(start, end, None)]

    # market committed 60 MW of charging, but the storage only has room for 40 MW
    storage.set_soc(
        start, storage.max_soc - 40 * storage.efficiency_charge / storage.capacity
    )
    storage.outputs["energy"][start] = -60
    _, runnable = storage.calculate_soc_path(start, end)
    assert runnable[start] == pytest.approx(-40)

    bids = strategy.calculate_bids(
        storage, mock_market_config, product_tuples=product_tuples
    )

    # the grid baseline is the runnable -40, NOT the committed -60
    assert bids[0]["volume"] == pytest.approx(-40)
    # the clearing starts from the SOC at the start of the hour
    assert bids[0]["soc"] == pytest.approx(
        storage.max_soc - 40 * storage.efficiency_charge / storage.capacity
    )


def test_storage_keeps_committed_plan_and_runs_what_soc_allows(
    mock_market_config, storage
):
    """
    outputs["energy"] keeps the market-committed plan (the redispatch baseline,
    because the generators were dispatched against it), while the SOC path only
    runs what the state of charge can back.
    """
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    mc = mock_market_config
    mc.product_type = "energy"

    # nearly empty battery: it can physically deliver only
    # soc * capacity * efficiency_discharge = 0.01 * 1000 * 0.95 = 9.5 MW
    storage.set_soc(start, 0.01)

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "accepted_volume": 100,  # market committed 100 MW of discharge
            "accepted_price": 50,
        }
    ]
    storage.set_dispatch_plan(mc, orderbook)

    # the plan keeps the committed 100 ...
    assert storage.outputs["energy"].at[start] == pytest.approx(100)
    # ... while only what the SOC allows can actually run
    soc_path, runnable = storage.calculate_soc_path(start, end)
    assert runnable[start] == pytest.approx(
        0.01 * storage.capacity * storage.efficiency_discharge
    )
    assert soc_path[end] == pytest.approx(storage.min_soc)


def test_redispatch_storage_excludes_reserved_crm_capacity(mock_market_config, storage):
    """
    Capacity already sold on the CRM/aFRR markets must not be offered again to the
    redispatch market. Reserving upward capacity has to shrink the upward (discharge)
    head-room by the same amount, otherwise the same MW are committed twice.
    """
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyOpportunityCostRedispatchStrategy()
    product_tuples = [(start, end, None)]

    baseline = strategy.calculate_bids(
        storage, mock_market_config, product_tuples=product_tuples
    )[0]

    # reserve 30 MW of upward capacity on the CRM market
    storage.outputs["capacity_pos"][start] = 30
    reserved = strategy.calculate_bids(
        storage, mock_market_config, product_tuples=product_tuples
    )[0]

    assert reserved["max_power"] == pytest.approx(baseline["max_power"] - 30)


def test_flexable_eom_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyHeuristicFlexableStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]

    # constant price of 50
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    # no change in price forecast -> no bidding
    assert len(bids) == 1
    assert bids[0]["price"] == 50 / storage.efficiency_discharge
    assert bids[0]["volume"] == 60

    # increase the current price forecast -> discharging
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": [60, 50, 50, 50]})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 52.5 / storage.efficiency_discharge
    assert bids[0]["volume"] == 60

    # decrease current price forecast -> charging
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": [40, 50, 50, 50]})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 47.5 * storage.efficiency_charge
    assert bids[0]["volume"] == -60

    # change to dam bidding
    day = pd.date_range(start, start + timedelta(hours=23), freq="h")
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    product_tuples = [(start, start + timedelta(hours=1), None) for start in day]
    storage.foresight = pd.Timedelta(hours=4)
    forecast = [
        20,
        50,
        50,
        50,
        80,
        50,
        50,
        50,
        80,
        50,
        50,
        50,
        80,
        50,
        50,
        50,
        20,
        50,
        50,
        50,
        20,
        50,
        50,
        50,
    ]
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": forecast})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    if storage.simultaneous_charge_discharge_ramping:
        # if simultaneously ramping charge and discharge is allowed, a storage can switch faster between charging and discharging, ramp constraints are less restricitve
        assert_allclose_with_indices(
            [b["volume"] for b in bids],
            [
                -60,
                -100,
                -100,
                -100,
                -50,
                -100,
                -45.555,
                # 0,
                60,
                100,
                100,
                100,
                100,
                50,
                -60,
                -100,
                -100,
                -50,
                60,
                100,
                50,
                100,
                100,
                100,
            ],
            atol=0.01,
        )
        assert_allclose_with_indices(
            [b["price"] for b in bids],
            [
                np.mean(forecast[0:13]) * storage.efficiency_charge,
                np.mean(forecast[0:14]) * storage.efficiency_charge,
                np.mean(forecast[0:15]) * storage.efficiency_charge,
                np.mean(forecast[0:16]) * storage.efficiency_charge,
                np.mean(forecast[0:17]) * storage.efficiency_charge,
                np.mean(forecast[0:18]) * storage.efficiency_charge,
                np.mean(forecast[0:19]) * storage.efficiency_charge,
                # 0,
                np.mean(forecast[0:21]) / storage.efficiency_discharge,
                np.mean(forecast[0:22]) / storage.efficiency_discharge,
                np.mean(forecast[0:23]) / storage.efficiency_discharge,
                np.mean(forecast[0:23]) / storage.efficiency_discharge,
                np.mean(forecast[0:24]) / storage.efficiency_discharge,
                np.mean(forecast[1:24]) / storage.efficiency_discharge,
                np.mean(forecast[2:24]) * storage.efficiency_charge,
                np.mean(forecast[3:24]) * storage.efficiency_charge,
                np.mean(forecast[4:24]) * storage.efficiency_charge,
                np.mean(forecast[5:24]) * storage.efficiency_charge,
                np.mean(forecast[6:24]) / storage.efficiency_discharge,
                np.mean(forecast[7:24]) / storage.efficiency_discharge,
                np.mean(forecast[8:24]) / storage.efficiency_discharge,
                np.mean(forecast[9:24]) / storage.efficiency_discharge,
                np.mean(forecast[10:24]) / storage.efficiency_discharge,
                np.mean(forecast[11:24]) / storage.efficiency_discharge,
            ],
            atol=0.01,
        )

    elif not storage.simultaneous_charge_discharge_ramping:
        # we have 24 timesteps, from 0 to 23, first 4 we want to charge [-60, -100, -100, -100]
        # the price in timestep 4 is high, so we would like to discharge, but ramp_down_charge=-50 restricts this, the strategy issues a -50 charging bid (could also be handled differently)
        # in timesteps 5, 6, 7 prices favor charging again, but the SoC is near 1, so [-100, -45.55, 0]. Timestep 6 with volume -45.55 violates ramping (-100 - -50) = -50, but SoC is enforced.
        # Timestep 7 is 0 because SoC is 1.
        # timesteps 8 to 12 favor discharge, we get [60, 100, 100, 100, 100]
        # timesteps 13 to 16 favor charging, starting from SoC 0.5158. Timestep 12 was discharging 100, so in step 13 we should still discharge 50 because of the ramp limit. timestep 14 0, so [50, 0, -60, -100]
        # timesteps 17, 18, 19 favor discharge, but we first need to ramp back, so we get [-50, 0, 60]
        # timestep 20 favors charging but we have previous_power = 60, so we ramp back and get [10]
        # timesteps 21, 22, 23 discharge, so ramp_up_discharge from previous_power = 10 and we get [70, 100, 100]
        # 0 volume (empty) bids in 3 timesteps results in 21 bids
        assert len(bids) == 21
        assert_allclose_with_indices(
            [b["volume"] for b in bids],
            [
                -60,
                -100,
                -100,
                -100,
                -50,
                -100,
                -41 / 0.9,
                # 0,
                60,
                100,
                100,
                100,
                100,
                50,
                # 0,
                -60,
                -100,
                -50,
                # 0,
                60,
                10,
                70,
                100,
                100,
            ],
            atol=0.01,
        )
        assert_allclose_with_indices(
            [b["price"] for b in bids],
            [
                np.mean(forecast[0:13]) * storage.efficiency_charge,
                np.mean(forecast[0:14]) * storage.efficiency_charge,
                np.mean(forecast[0:15]) * storage.efficiency_charge,
                np.mean(forecast[0:16]) * storage.efficiency_charge,
                np.mean(forecast[0:17]) * storage.efficiency_charge,
                np.mean(forecast[0:18]) * storage.efficiency_charge,
                np.mean(forecast[0:19]) * storage.efficiency_charge,
                # 0,
                np.mean(forecast[0:21]) / storage.efficiency_discharge,
                np.mean(forecast[0:22]) / storage.efficiency_discharge,
                np.mean(forecast[0:23]) / storage.efficiency_discharge,
                np.mean(forecast[0:23]) / storage.efficiency_discharge,
                np.mean(forecast[0:24]) / storage.efficiency_discharge,
                np.mean(forecast[1:24]) / storage.efficiency_discharge,
                # 0,
                np.mean(forecast[3:24]) * storage.efficiency_charge,
                np.mean(forecast[4:24]) * storage.efficiency_charge,
                np.mean(forecast[5:24]) * storage.efficiency_charge,
                # 0,
                np.mean(forecast[7:24]) / storage.efficiency_discharge,
                np.mean(forecast[8:24]) / storage.efficiency_discharge,
                np.mean(forecast[9:24]) / storage.efficiency_discharge,
                np.mean(forecast[10:24]) / storage.efficiency_discharge,
                np.mean(forecast[11:24]) / storage.efficiency_discharge,
            ],
            atol=0.01,
        )


def test_flexable_pos_crm_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = StorageCapacityHeuristicBalancingPosStrategy()
    mc = mock_market_config
    mc.product_type = "energy_pos"
    product_tuples = [(start, end, None)]

    # constant price of 50, marginal cost of 4, volume of 360, theoretic soc change (without efficiency) of 0.36, capacity of 1000
    specific_revenue = (50 - 4) * 360 / (0.36 * 1000)

    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], specific_revenue)
    assert bids[0]["volume"] == 60

    # assert capacity_pos
    mc.product_type = "capacity_pos"
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], specific_revenue)
    assert bids[0]["volume"] == 60

    # specific revenue < 0
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 3})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 60

    # was charging before and ramping constraint makes it impossible to discharge in the next step
    storage.outputs["energy"][start] = -60
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 0

    # two steps after the charging step, ramping constraint allows to discharge again
    storage.outputs["energy"][start] = -60
    product_tuples = [
        (start + pd.Timedelta(hours=2), end + pd.Timedelta(hours=2), None)
    ]
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1


def test_flexable_neg_crm_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = StorageCapacityHeuristicBalancingNegStrategy()
    mc = mock_market_config
    # Calculations for negative energy
    mc.product_type = "energy_neg"
    product_tuples = [(start, end, None)]

    # constant price of 50
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], 0)
    assert bids[0]["volume"] == 60

    # assert capacity_pos
    mc.product_type = "capacity_neg"
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], 0)
    assert bids[0]["volume"] == 60

    # was charging before and ramping constraint makes it impossible to discharge in the next step
    storage.outputs["energy"][start] = 60
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 0

    # two steps after the charging step, ramping constraint allows to discharge again
    storage.outputs["energy"][start] = 60
    product_tuples = [
        (start + pd.Timedelta(hours=2), end + pd.Timedelta(hours=2), None)
    ]
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])


@pytest.mark.parametrize(
    "market_prices, expected_price",
    [
        # no EOM forecast: use the redispatch market's own forecast
        ({"redispatch": 60.0}, 60.0),
        # neither EOM nor the own market: use any available forecast
        ({"other_market": 70.0}, 70.0),
    ],
)
def test_redispatch_storage_price_forecast_fallback(market_prices, expected_price):
    from assume.common.market_objects import MarketConfig, MarketProduct

    index = pd.date_range("2023-07-01", periods=4, freq="h")
    unit = Storage(
        id="battery",
        unit_operator="op",
        technology="battery",
        bidding_strategies={},
        forecaster=UnitForecaster(index, availability=1, market_prices=market_prices),
        max_power_charge=-10,
        max_power_discharge=10,
        capacity=40,
        efficiency_charge=0.8,
        efficiency_discharge=0.5,
        initial_soc=0.5,
    )
    config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        market_mechanism="redispatch",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )
    bids = StorageEnergyOpportunityCostRedispatchStrategy().calculate_bids(
        unit, config, product_tuples=[(index[0], index[1], None)]
    )
    # flat forecast, "sell" valuation: a MWh discharged is worth the forecast, a
    # MWh charged the forecast after both efficiencies
    assert bids[0]["price_discharge"] == pytest.approx(expected_price)
    assert bids[0]["price_charge"] == pytest.approx(expected_price * 0.8 * 0.5)


@pytest.mark.parametrize(
    "move, grid_mwh, stored_mwh",
    [
        # one MWh at the grid changes the stored energy by 1/eta_d or eta_c
        ("discharge_more", 1.0, -1 / 0.95),
        ("discharge_less", -1.0, 1 / 0.95),
        ("charge_less", 1.0, -0.9),
        ("charge_more", -1.0, 0.9),
    ],
)
def test_opportunity_cost_prices_are_cost_neutral(
    mock_market_config, storage, move, grid_mwh, stored_mwh
):
    """
    Settled at its price, every move leaves the storage neither better nor worse
    off: the payment equals the value of the stored energy it gains or loses plus
    or minus the variable cost of running (storage: eta 0.9 / 0.95, costs 3 / 4,
    flat EOM forecast 50).
    """
    prices = (
        StorageEnergyOpportunityCostRedispatchStrategy().calculate_redispatch_prices(
            storage, mock_market_config, datetime(2023, 7, 1)
        )
    )
    value = prices["stored_energy_value"]
    if move.startswith("discharge"):
        payment = prices["price_discharge"] * grid_mwh
        running_cost = 4 * grid_mwh  # more discharge costs more
    else:
        payment = prices["price_charge"] * grid_mwh
        running_cost = -3 * grid_mwh  # charging less saves the charge cost
    assert payment + stored_mwh * value - running_cost == pytest.approx(0.0)


def test_heuristic_redispatch_bids_one_price(mock_market_config, storage):
    """
    The heuristic strategy sends the same bid as the opportunity-cost strategy,
    with one price for both directions: the reference price over a full cycle
    plus both variable costs (flat EOM forecast 50, eta 0.9 / 0.95, costs 3 / 4).
    """
    start = datetime(2023, 7, 1)
    products = [(start, start + timedelta(hours=1), None)]
    heuristic = StorageEnergyHeuristicRedispatchStrategy().calculate_bids(
        storage, mock_market_config, product_tuples=products
    )[0]
    opportunity = StorageEnergyOpportunityCostRedispatchStrategy().calculate_bids(
        storage, mock_market_config, product_tuples=products
    )[0]

    expected = 50 / (0.9 * 0.95) + 3 + 4
    assert heuristic["price_discharge"] == pytest.approx(expected)
    assert heuristic["price_charge"] == pytest.approx(expected)
    assert heuristic["price"] == heuristic["price_discharge"]
    assert "stored_energy_value" not in heuristic
    for field in ["volume", "max_power", "min_power", "p_nom", "soc", "node"]:
        assert heuristic[field] == opportunity[field]
