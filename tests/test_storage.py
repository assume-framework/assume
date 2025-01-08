# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies.flexable_storage import flexableEOMStorage
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from assume.units import Storage


@pytest.fixture
def storage_unit() -> Storage:
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = NaiveForecast(index, availability=1, price_forecast=50)
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": NaiveSingleBidStrategy()},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        max_soc=1000,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        additional_cost_charge=3,
        additional_cost_discharge=4,
        additional_cost=1,
        initial_soc=500,
    )


def test_init_function(storage_unit):
    assert storage_unit.id == "Test_Storage"
    assert storage_unit.unit_operator == "TestOperator"
    assert storage_unit.technology == "TestTechnology"
    assert storage_unit.max_power_charge == -100
    assert storage_unit.max_power_discharge == 100
    assert storage_unit.efficiency_charge == 0.9
    assert storage_unit.efficiency_discharge == 0.95
    assert storage_unit.ramp_down_charge == -50
    assert storage_unit.ramp_down_discharge == 50
    assert storage_unit.ramp_up_charge == -60
    assert storage_unit.ramp_up_discharge == 60
    assert storage_unit.initial_soc == 500


def test_reset_function(storage_unit):
    # check if total_power_output is reset
    assert (
        storage_unit.outputs["energy"]
        == pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="h"))
    ).all()

    # the same for pos and neg capacity reserve
    assert (
        storage_unit.outputs["pos_capacity"]
        == pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="h"))
    ).all()
    assert (
        storage_unit.outputs["neg_capacity"]
        == pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="h"))
    ).all()

    # check if state of charge (soc) is reset correctly
    assert (
        storage_unit.outputs["soc"]
        == pd.Series(500.0, index=pd.date_range("2022-01-01", periods=4, freq="h"))
    ).all()


def test_calculate_operational_window(storage_unit):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )
    cost_discharge = storage_unit.calculate_marginal_cost(start, max_power_discharge[0])

    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100
    assert cost_discharge == 4 / 0.95

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end, product_type="energy"
    )
    cost_charge = storage_unit.calculate_marginal_cost(start, max_power_charge[0])

    assert min_power_charge[0] == 0
    assert max_power_charge[0] == -100
    assert math.isclose(cost_charge, 3 / 0.9)

    assert storage_unit.outputs["energy"].at[start] == 0

    storage_unit.outputs["energy"][start] = 10
    storage_unit.outputs["capacity_neg"][start] = -50
    storage_unit.outputs["capacity_pos"][start] = 30

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )
    assert min_power_charge[0] == -40
    assert max_power_charge[0] == -60

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    assert max_power_discharge[0] == 60

    start = start + timedelta(hours=1)


def test_soc_constraint(storage_unit):
    # start should not be the first hour of index to manipulate soc
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    storage_unit.outputs["energy"][start] = 10
    storage_unit.outputs["capacity_neg"][start] = -50
    storage_unit.outputs["capacity_pos"][start] = 30

    storage_unit.outputs["soc"][start - timedelta(hours=1)] = (
        0.05 * storage_unit.max_soc
    )
    assert storage_unit.get_soc_before(start) == 0.05 * storage_unit.max_soc
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    assert math.isclose(
        max_power_discharge[0], (50 * storage_unit.efficiency_discharge)
    )

    storage_unit.outputs["soc"][start] = 0.95 * storage_unit.max_soc
    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )
    assert min_power_charge[0] == -40
    assert math.isclose(
        max_power_charge[0], -50 / storage_unit.efficiency_charge, abs_tol=0.1
    )


def test_storage_feedback(storage_unit, mock_market_config):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end, product_type="energy"
    )

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )
    cost_discharge = storage_unit.calculate_marginal_cost(start, max_power_discharge[0])

    assert min_power_charge[0] == 0
    assert max_power_charge[0] == -100

    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100
    assert storage_unit.outputs["energy"][start] == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": cost_discharge,
            "accepted_price": cost_discharge,
            "accepted_volume": max_power_discharge[0] / 2,
        }
    ]
    # max_power_charge gets accepted
    mc = mock_market_config
    storage_unit.set_dispatch_plan(mc, orderbook)

    # second market request for same interval
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )

    # we do not need additional min_power, as our runtime requirement is fulfilled
    assert min_power_discharge[0] == 0
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power_discharge[0] == 50

    storage_unit.execute_current_dispatch(start, end)
    # second market request for next interval
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )

    # now we can bid max_power and need min_power again
    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100


def test_storage_ramping(storage_unit):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end, product_type="energy"
    )

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )

    assert min_power_charge[0] == 0
    assert max_power_charge[0] == -100

    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        500, 0, max_power_discharge[0]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(500, 0, max_power_charge[0])

    assert max_ramp_discharge == 60
    assert max_ramp_charge == -60

    # discharge power gets accepted
    storage_unit.outputs["energy"][start] += 60

    # next hour
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        500, 60, max_power_discharge[0]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(500, 60, max_power_charge[0])

    assert max_ramp_discharge == 100
    assert max_ramp_charge == -60

    # charging scenario
    storage_unit.outputs["energy"][start] = -60

    # next hour
    start = datetime(2022, 1, 1, 2)
    end = datetime(2022, 1, 1, 3)

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        500, -60, max_power_discharge[0]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(500, -60, max_power_charge[0])

    assert max_ramp_discharge == 60
    assert max_ramp_charge == -100


def test_execute_dispatch(storage_unit):
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.5 * storage_unit.max_soc

    # dispatch full discharge
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == 100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        500 - 100 / storage_unit.efficiency_discharge,
    )

    # dispatch full charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.5 * storage_unit.max_soc
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == -100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        500 + 100 * storage_unit.efficiency_charge,
    )
    # adjust dispatch to soc limit for discharge
    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.05 * storage_unit.max_soc
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert math.isclose(
        dispatched_energy[0], 50 * storage_unit.efficiency_discharge, abs_tol=0.1
    )
    # adjust dispatch to soc limit for charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.95 * storage_unit.max_soc
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert math.isclose(
        dispatched_energy[0], -50 / storage_unit.efficiency_charge, abs_tol=0.1
    )
    assert math.isclose(
        storage_unit.outputs["soc"][end], storage_unit.max_soc, abs_tol=0.001
    )

    # step into the next hour
    start = start + storage_unit.index.freq
    end = end + storage_unit.index.freq
    storage_unit.outputs["energy"][start] = -100
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == 0
    assert math.isclose(
        storage_unit.outputs["soc"][end], storage_unit.max_soc, abs_tol=0.001
    )


def test_set_dispatch_plan(mock_market_config, storage_unit):
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    mc = mock_market_config

    strategy = flexableEOMStorage()
    product_tuples = [(start, end, None)]

    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.5 * storage_unit.max_soc

    bids = strategy.calculate_bids(storage_unit, mc, product_tuples=product_tuples)
    assert len(bids) == 0

    # dispatch full discharge
    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert storage_unit.outputs["energy"][start] == 100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        500 - 100 / storage_unit.efficiency_discharge,
    )
    # dispatch full charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.5 * storage_unit.max_soc

    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert storage_unit.outputs["energy"][start] == -100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        500 + 100 * storage_unit.efficiency_charge,
    )
    # adjust dispatch to soc limit for discharge
    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.05 * storage_unit.max_soc

    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert math.isclose(
        storage_unit.outputs["energy"][start],
        50 * storage_unit.efficiency_discharge,
        abs_tol=0.1,
    )
    # adjust dispatch to soc limit for charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.95 * storage_unit.max_soc

    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert math.isclose(
        storage_unit.outputs["energy"][start],
        -50 / storage_unit.efficiency_charge,
        abs_tol=0.1,
    )
    assert math.isclose(
        storage_unit.outputs["soc"][end], storage_unit.max_soc, abs_tol=0.001
    )

    # step into the next hour
    start = start + storage_unit.index.freq
    end = end + storage_unit.index.freq
    product_tuples = [(start, end, None)]

    bids = strategy.calculate_bids(storage_unit, mc, product_tuples=product_tuples)
    assert len(bids) == 1


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
