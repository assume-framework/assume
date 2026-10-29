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


def test_flexable_eom_storage_no_simultaneous_charge_and_discharge(mock_market_config, storage):
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
    assert np.allclose(
        [b["volume"] for b in bids], [
        -60,
        -100,
        -100,
        -100,
        -50,
        -100,
        -41/0.9,
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
