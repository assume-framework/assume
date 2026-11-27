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
from assume.strategies.naive_strategies import (
    StorageEnergyHeuristicRedispatchStrategy,
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
def set_eom_price(unit: Storage, price: float) -> None:
    """Price the storage got on the EOM in every time step (cost-based reference)."""
    unit.outputs["energy_accepted_price"].loc[unit.index[0] : unit.index[-1]] = price


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
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyHeuristicRedispatchStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]

    # accepted EOM price of 50
    set_eom_price(storage, 50)
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)

    assert len(bids) == 1
    bid = bids[0]

    # flat EOM price 50 all day: the best later price is 50 as well
    assert bid["reference_price"] == pytest.approx(50)
    assert bid["price_discharge"] == pytest.approx(50)
    assert bid["price_discharge_less"] == pytest.approx(50)
    assert bid["price_charge"] == pytest.approx(0.9 * 0.95 * 50)
    assert bid["price"] == bid["price_discharge"]
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


def test_redispatch_storage_baseline_is_runnable_power(mock_market_config, storage):
    """
    The redispatch baseline (grid injection) is what the storage can actually run of
    its DA plan, not the committed plan. Otherwise the grid model would compute line
    flows with an injection that never happens.
    """
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyHeuristicRedispatchStrategy()
    product_tuples = [(start, end, None)]

    # market committed 60 MW of charging, but the storage only has room for 40 MW
    storage.set_soc(
        start, storage.max_soc - 40 * storage.efficiency_charge / storage.capacity
    )
    storage.outputs["energy"][start] = -60

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
    runnable = storage.calculate_feasible_power(
        storage.outputs["energy"].at[start], storage.get_soc(start)
    )
    assert runnable == pytest.approx(
        0.01 * storage.capacity * storage.efficiency_discharge
    )
    assert storage.get_soc(end) == pytest.approx(storage.min_soc)


def test_redispatch_storage_excludes_reserved_crm_capacity(mock_market_config, storage):
    """
    Capacity already sold on the CRM/aFRR markets must not be offered again to the
    redispatch market. Reserving upward capacity has to shrink the upward (discharge)
    head-room by the same amount, otherwise the same MW are committed twice.
    """
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyHeuristicRedispatchStrategy()
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


def day_storage(accepted_prices):
    """Battery (eta 0.8 / 0.5) with the given accepted EOM prices, one per hour."""
    index = pd.date_range("2023-07-01", periods=len(accepted_prices), freq="h")
    unit = Storage(
        id="battery",
        unit_operator="op",
        technology="battery",
        bidding_strategies={},
        forecaster=UnitForecaster(index, availability=1, market_prices={"EOM": 50}),
        max_power_charge=-10,
        max_power_discharge=10,
        capacity=40,
        efficiency_charge=0.8,
        efficiency_discharge=0.5,
        initial_soc=0.5,
    )
    for t, price in zip(index, accepted_prices):
        unit.outputs["energy_accepted_price"].at[t] = price
    return unit, index


@pytest.mark.parametrize(
    "hour, expected",
    [
        # P = 60, best later price F = 70: discharge more max(P, F), less F
        (0, {"reference_price": 60, "price_discharge": 70, "price_discharge_less": 70}),
        # P = 70 is the peak, F = 40: discharging less only sells later at 40
        (1, {"reference_price": 70, "price_discharge": 70, "price_discharge_less": 40}),
        # last hour of the day: no later price, F = P
        (3, {"reference_price": 30, "price_discharge": 30, "price_discharge_less": 30}),
    ],
)
def test_redispatch_prices_follow_the_rest_of_the_day(hour, expected):
    """
    Prices from the accepted EOM price of the hour (not the forecast of 50) and
    the best accepted price later that day; charge price = eta_c * eta_d * F.
    """
    unit, index = day_storage([60.0, 70.0, 40.0, 30.0])
    prices = StorageEnergyHeuristicRedispatchStrategy().calculate_redispatch_prices(
        unit, index[hour]
    )
    for field, value in expected.items():
        assert prices[field] == pytest.approx(value)
    assert prices["price_charge"] == pytest.approx(
        0.8 * 0.5 * expected["price_discharge_less"]
    )


def test_redispatch_reference_falls_back_to_the_forecast(mock_market_config, storage):
    """
    Without an EOM result in that time step (no accepted price, no dispatch), the
    EOM price forecast of that time step is used (flat forecast 50 in the fixture).
    """
    start = datetime(2023, 7, 1)
    strategy = StorageEnergyHeuristicRedispatchStrategy()
    assert strategy.reference_price(storage, start) == pytest.approx(50)

    storage.outputs["energy_accepted_price"].at[start] = 30
    assert strategy.reference_price(storage, start) == pytest.approx(30)
