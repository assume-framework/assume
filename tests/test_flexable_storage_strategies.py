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
    by_hour = {bid["start_time"].hour: bid for bid in bids}

    # the ramp down limits allow no change of direction within one hour, so the
    # unit keeps charging at the price peak at 4 and cannot charge at 13 after
    # discharging - and it is full at 6, 7 and 20
    assert sorted(set(range(24)) - set(by_hour)) == [6, 7, 13, 20]
    assert [round(by_hour[hour]["volume"], 1) for hour in sorted(by_hour)] == [
        *[-60, -100, -100, -100, -100, -95.6],  # 0 - 5
        *[60, 100, 100, 100, 100],  # 8 - 12
        *[-60, -100, -100, -100, -100, -78],  # 14 - 19
        *[60, 100, 100],  # 21 - 23
    ]
    # the price is the average over the foresight, adjusted by the efficiency
    for hour, bid in by_hour.items():
        average_price = np.mean(forecast[max(0, hour - 12) : hour + 13])
        if bid["volume"] > 0:
            expected_price = average_price / storage.efficiency_discharge
        else:
            expected_price = average_price * storage.efficiency_charge
        assert math.isclose(bid["price"], expected_price, abs_tol=0.01)


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

    # was charging before - at -40 MW it can ramp down to 0 and discharge, at
    # -60 MW only down to -10 MW
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]
    storage.outputs["energy"][start] = -40
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    storage.outputs["energy"][start] = -60
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 0


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

    # was discharging before - at 40 MW it can ramp down to 0 and charge, at
    # 60 MW only down to 10 MW
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]
    storage.outputs["energy"][start] = 40
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    storage.outputs["energy"][start] = 60
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 0


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])


def test_flexable_eom_storage_bids_on_top_of_an_infeasible_plan(mock_market_config):
    """
    A later energy market bids on hours an earlier one already committed. The
    SoC estimate of the strategy has to build on what the unit can actually run
    at there - 20 MWh are stored, so a committed 100 MW discharge only delivers
    20 MW - rather than on the plan, which would drive it far below empty.
    """
    index = pd.date_range("2022-01-01", periods=8, freq="h")
    prices = pd.Series([50, 50, 80, 80, 80, 80, 80, 80], index=index)
    strategy = StorageEnergyHeuristicFlexableStrategy(eom_foresight="4h")
    storage = Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": strategy},
        forecaster=UnitForecaster(index, availability=1, market_prices={"EOM": prices}),
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=100,
        efficiency_charge=1,
        efficiency_discharge=1,
        initial_soc=0.2,
    )
    storage.set_dispatch_plan(
        mock_market_config,
        [
            {
                "start_time": index[1],
                "end_time": index[2],
                "only_hours": None,
                "accepted_volume": 100,
                "accepted_price": 80,
            }
        ],
    )
    assert storage.get_feasible_energy(index[1], index[1])[0] == 20

    product_tuples = [(index[i], index[i + 1], None) for i in (1, 2, 3)]
    bids = strategy.calculate_bids(storage, mock_market_config, product_tuples)

    # empty after t1, the unit can charge 80 MW on top of its 20 MW discharge,
    # discharge them again at t2 and then charge fully at t3
    assert [(bid["start_time"], bid["volume"]) for bid in bids] == [
        (index[1], pytest.approx(-80)),
        (index[2], pytest.approx(80)),
        (index[3], pytest.approx(-100)),
    ]
