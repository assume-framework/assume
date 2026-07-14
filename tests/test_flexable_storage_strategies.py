# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    StorageEnergyHeuristicRedispatchStrategy,
)
from assume.units import Storage


@pytest.fixture
def storage() -> Storage:
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
    )


def test_redispatch_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 1)
    strategy = StorageEnergyHeuristicRedispatchStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]

    # constant EOM price of 50
    storage.forecaster = UnitForecaster(index, market_prices={"EOM": 50})
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)

    assert len(bids) == 1
    bid = bids[0]

    # cost-basis price: round-trip adjusted market price plus variable costs
    expected_price = (
        50 / (storage.efficiency_charge * storage.efficiency_discharge)
        + storage.additional_cost_charge
        + storage.additional_cost_discharge
    )
    assert bid["price"] == pytest.approx(expected_price)
    # full nameplate power span as p_nom (charge is negative, so this adds)
    assert bid["p_nom"] == pytest.approx(
        storage.max_power_discharge - storage.max_power_charge
    )
    # idle storage: zero baseline, but bidirectional SoC-aware flex bounds
    assert bid["volume"] == 0
    assert bid["max_power"] == pytest.approx(storage.max_power_discharge)
    assert bid["min_power"] == pytest.approx(storage.max_power_charge)
    assert "node" in bid

    # near-empty storage offers less additional discharge (SoC-aware upper bound)
    storage.outputs["soc"].at[start] = 0.05
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert bids[0]["max_power"] == pytest.approx(
        0.05 * storage.capacity * storage.efficiency_discharge
    )
    assert bids[0]["max_power"] < storage.max_power_discharge

    # near-full storage offers less additional charge (SoC-aware lower bound)
    storage.outputs["soc"].at[start] = 0.98
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert bids[0]["min_power"] == pytest.approx(
        (0.98 - storage.max_soc) * storage.capacity / storage.efficiency_charge
    )
    assert bids[0]["min_power"] > storage.max_power_charge


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
    assert len(bids) == 24
    assert math.isclose(
        bids[0]["price"],
        np.mean(forecast[0:13]) * storage.efficiency_charge,
        abs_tol=0.01,
    )
    assert bids[0]["volume"] == -60
    assert math.isclose(
        bids[1]["price"],
        np.mean(forecast[0:14]) * storage.efficiency_charge,
        abs_tol=0.01,
    )
    assert bids[1]["volume"] == -100
    assert math.isclose(
        bids[2]["price"],
        np.mean(forecast[0:15]) * storage.efficiency_charge,
        abs_tol=0.01,
    )
    assert bids[2]["volume"] == -100
    assert math.isclose(
        bids[3]["price"],
        np.mean(forecast[0:16]) * storage.efficiency_charge,
        abs_tol=0.01,
    )
    assert bids[3]["volume"] == -100
    assert math.isclose(
        bids[4]["price"],
        np.mean(forecast[0:17]) / storage.efficiency_discharge,
        abs_tol=0.01,
    )
    assert math.isclose(bids[4]["volume"], 60, abs_tol=0.01)
    assert math.isclose(
        bids[14]["price"],
        np.mean(forecast[2:]) * storage.efficiency_charge,
        abs_tol=0.01,
    )
    assert bids[14]["volume"] == -100
    assert math.isclose(
        bids[20]["price"],
        np.mean(forecast[6:]) * storage.efficiency_charge,
        abs_tol=0.01,
    )
    assert bids[20]["volume"] == -60


def test_flexable_pos_crm_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="h")
    start = datetime(2023, 7, 1)
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = StorageCapacityHeuristicBalancingPosStrategy()
    mc = mock_market_config
    mc.product_type = "energy_pos"
    product_tuples = [(start, end, None)]

    # constant price of 50
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

    # was charging before
    storage.outputs["energy"][start] = -60
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
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

    # was charging before
    storage.outputs["energy"][start] = 60
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
