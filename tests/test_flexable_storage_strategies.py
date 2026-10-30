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


def _eom_storage(prices, capacity, initial_soc):
    strategy = StorageEnergyHeuristicFlexableStrategy(eom_foresight="4h")
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": strategy},
        forecaster=UnitForecaster(
            prices.index, availability=1, market_prices={"EOM": prices}
        ),
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=capacity,
        efficiency_charge=1,
        efficiency_discharge=1,
        initial_soc=initial_soc,
    )


def _accept(storage, market_config, orders):
    storage.set_dispatch_plan(
        market_config,
        [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "accepted_volume": volume,
                "accepted_price": 80,
            }
            for start, end, volume in orders
        ],
    )


def test_flexable_eom_storage_bids_on_top_of_a_committed_plan(mock_market_config):
    """
    The SoC estimate of a later market's bids follows what the storage can
    deliver: of the 100 MW committed at t1, 20 MWh only deliver 20 MW. The
    estimate is empty at t2 instead of 80 MWh below empty, so after charging
    at t2 the storage can discharge fully at t3.
    """
    index = pd.date_range("2022-01-01", periods=8, freq="h")
    prices = pd.Series([50, 50, 80, 80, 80, 80, 80, 80], index=index)
    storage = _eom_storage(prices, capacity=100, initial_soc=0.2)
    _accept(
        storage,
        mock_market_config,
        [(index[1], index[2], 100), (index[2], index[3], -50)],
    )
    # the storage is empty after delivering 20 MW of the 100 MW
    assert storage.get_soc(index[2]) == 0

    strategy = storage.bidding_strategies["EOM"]
    product_tuples = [(index[i], index[i + 1], None) for i in (1, 2, 3)]
    bids = strategy.calculate_bids(storage, mock_market_config, product_tuples)

    # charge fully at t2 and discharge it again at t3
    assert [(bid["start_time"], bid["volume"]) for bid in bids] == [
        (index[2], pytest.approx(-50)),
        (index[3], pytest.approx(100)),
    ]

    # once accepted, everything committed can be delivered - except the 80 MW
    # at t1 the storage cannot deliver
    _accept(
        storage,
        mock_market_config,
        [(bid["start_time"], bid["end_time"], bid["volume"]) for bid in bids],
    )
    committed = [storage.outputs["energy"].at[index[i]] for i in (1, 2, 3)]
    assert committed == pytest.approx([100, -100, 100])
    delivered = storage.execute_current_dispatch(index[1], index[3])
    assert list(delivered) == pytest.approx([20, -100, 100])
