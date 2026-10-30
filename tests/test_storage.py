# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.strategies.flexable_storage import StorageEnergyHeuristicFlexableStrategy
from assume.units import Storage


@pytest.fixture
def storage_unit() -> Storage:
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = UnitForecaster(index, availability=1, market_prices={"EOM": 50})
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": StorageEnergyHeuristicFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=1000,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        additional_cost_charge=3,
        additional_cost_discharge=4,
        additional_cost=1,
        initial_soc=None,
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
    assert storage_unit.initial_soc == 0.5


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
        == pd.Series(0.5, index=pd.date_range("2022-01-01", periods=4, freq="h"))
    ).all()


def test_calculate_operational_window(storage_unit):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    cost_discharge = storage_unit.calculate_marginal_cost(start, max_power_discharge[0])

    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100
    assert cost_discharge == 4

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )
    cost_charge = storage_unit.calculate_marginal_cost(start, max_power_charge[0])

    assert min_power_charge[0] == 0
    assert max_power_charge[0] == -100
    assert math.isclose(cost_charge, 3)

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

    storage_unit.outputs["soc"][start - timedelta(hours=1)] = 0.05
    assert storage_unit.outputs["soc"][start - storage_unit.index.freq] == 0.05
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    # the SoC is propagated to start, so it limits the discharge to 47.5 MW
    assert math.isclose(
        max_power_discharge[0],
        0.05 * storage_unit.capacity * storage_unit.efficiency_discharge,
    )

    storage_unit.outputs["soc"][start] = 0.95
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
        start, end
    )

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
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
        start, end
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
        start, end
    )

    # now we can bid max_power and need min_power again
    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100


def test_storage_ramping(storage_unit):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )

    assert min_power_charge[0] == 0
    assert max_power_charge[0] == -100

    assert min_power_discharge[0] == 0
    assert max_power_discharge[0] == 100

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        0.5, 0, max_power_discharge[0]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(0.5, 0, max_power_charge[0])

    assert max_ramp_discharge == 60
    assert max_ramp_charge == -60

    # discharge power gets accepted
    storage_unit.outputs["energy"][start] += 60

    # next hour
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        0.5, 60, max_power_discharge[0]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(0.5, 60, max_power_charge[0])

    assert max_ramp_discharge == 100
    assert max_ramp_charge == -60

    # charging scenario
    storage_unit.outputs["energy"][start] = -60

    # next hour
    start = datetime(2022, 1, 1, 2)
    end = datetime(2022, 1, 1, 3)

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        0.5, -60, max_power_discharge[0]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(0.5, -60, max_power_charge[0])

    assert max_ramp_discharge == 60
    assert max_ramp_charge == -100


def test_execute_dispatch(storage_unit):
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.5

    # dispatch full discharge
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == 100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        (500 - 100 / storage_unit.efficiency_discharge) / storage_unit.capacity,
    )

    # dispatch full charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.5
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == -100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        (500 + 100 * storage_unit.efficiency_charge) / storage_unit.capacity,
    )
    # adjust dispatch to soc limit for discharge
    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.05
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert math.isclose(
        dispatched_energy[0], 50 * storage_unit.efficiency_discharge, abs_tol=0.1
    )
    # adjust dispatch to soc limit for charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.95
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

    strategy = StorageEnergyHeuristicFlexableStrategy()
    product_tuples = [(start, end, None)]

    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.5

    bids = strategy.calculate_bids(storage_unit, mc, product_tuples=product_tuples)
    assert len(bids) == 0

    # dispatch full discharge
    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert storage_unit.outputs["energy"][start] == 100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        (500 - 100 / storage_unit.efficiency_discharge) / storage_unit.capacity,
    )
    # dispatch full charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.5

    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert storage_unit.outputs["energy"][start] == -100
    assert math.isclose(
        storage_unit.outputs["soc"][end],
        (500 + 100 * storage_unit.efficiency_charge) / storage_unit.capacity,
    )
    # adjust dispatch to soc limit for discharge
    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.05

    storage_unit.set_dispatch_plan(mc, bids)
    storage_unit.execute_current_dispatch(start, end)

    assert math.isclose(
        storage_unit.outputs["energy"][start],
        50 * storage_unit.efficiency_discharge,
        abs_tol=0.1,
    )
    # adjust dispatch to soc limit for charging
    storage_unit.outputs["energy"][start] = -100
    storage_unit.outputs["soc"][start] = 0.95

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


def test_set_dispatch_plan_multi_hours(mock_market_config, storage_unit):
    """
    This test checks that the dispatch of a storage does set the SoC output correctly.
    It also makes sure to work with multiple consecutive hours
    """
    product_tuples = []
    start = datetime(2022, 1, 1, 0)
    for i in range(3):
        s = datetime(2022, 1, 1, i)
        end = datetime(2022, 1, 1, i + 1)
        product_tuples.append((s, end, None))

    mc = mock_market_config
    strategy = StorageEnergyHeuristicFlexableStrategy()

    storage_unit.outputs["energy"][start] = 100
    storage_unit.outputs["soc"][start] = 0.5

    bids = strategy.calculate_bids(storage_unit, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    # continue discharging 100 MW in second and third hour
    assert bids[0]["start_time"] == datetime(2022, 1, 1, 1)
    assert bids[0]["volume"] == 100
    bids[0]["accepted_volume"] = 100
    bids[0]["accepted_price"] = 45
    assert bids[1]["start_time"] == datetime(2022, 1, 1, 2)
    assert bids[1]["volume"] == 100
    bids[1]["accepted_volume"] = 100
    bids[1]["accepted_price"] = 45

    # now dispatch full discharge
    storage_unit.set_dispatch_plan(mc, bids)

    # is the dispatch plan set correctly
    for i in range(1, 3):
        s = datetime(2022, 1, 1, i)
        s_next = datetime(2022, 1, 1, i + 1)
        delta_soc_set_dispatch = storage_unit.get_soc(s) - storage_unit.get_soc(s_next)

        if delta_soc_set_dispatch <= 0:
            delta_set_dispatch = (
                storage_unit.outputs["energy"][s] * storage_unit.efficiency_charge
            )
        else:
            delta_set_dispatch = (
                storage_unit.outputs["energy"][s] / storage_unit.efficiency_discharge
            )
        assert math.isclose(
            delta_set_dispatch / storage_unit.capacity, delta_soc_set_dispatch
        )

    # test if it is executed correctly, which should be the same with the mock market config only covering one market
    storage_unit.execute_current_dispatch(start, end)

    for i in range(1, 3):
        s = datetime(2022, 1, 1, i)
        s_next = datetime(2022, 1, 1, i + 1)
        delta_soc = storage_unit.outputs["soc"][s] - storage_unit.outputs["soc"][s_next]

        if delta_soc <= 0:
            delta = storage_unit.outputs["energy"][s] * storage_unit.efficiency_charge
        else:
            delta = (
                storage_unit.outputs["energy"][s] / storage_unit.efficiency_discharge
            )
        assert math.isclose(delta / storage_unit.capacity, delta_soc)

    # check that deltas are the same, which again must be due to only one considered market
    assert math.isclose(delta_soc_set_dispatch, delta_soc)


def test_initialising_invalid_storages():
    index = pd.date_range(
        start=datetime(2023, 7, 1),
        end=datetime(2023, 7, 2),
        freq="1h",
    )
    param_dict = {
        "id": "id",
        "unit_operator": "operator",
        "technology": "technology",
        "bidding_strategies": {},
        "forecaster": UnitForecaster(index=index),
        "max_power_charge": 0.0,
        "max_power_discharge": 0.0,
        "max_soc": 0.0,
        "capacity": 0.0,
    }
    with pytest.raises(
        ValueError, match="max_power_charge=10 must be <= 0 for unit id"
    ):
        d = param_dict.copy()
        d["max_power_charge"] = 10
        Storage(**d)
    with pytest.raises(
        ValueError, match="min_power_charge=10 must be <= 0 for unit id"
    ):
        d = param_dict.copy()
        d["min_power_charge"] = 10
        Storage(**d)
    with pytest.raises(
        ValueError,
        match="max_power_charge=-10 must be <= min_power_charge=-20 for unit id",
    ):
        d = param_dict.copy()
        d["max_power_charge"] = -10
        d["min_power_charge"] = -20
        Storage(**d)
    with pytest.raises(
        ValueError, match="max_power_discharge=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["max_power_discharge"] = -10
        Storage(**d)
    with pytest.raises(
        ValueError, match="min_power_discharge=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["min_power_discharge"] = -10
        Storage(**d)
    with pytest.raises(
        ValueError,
        match="max_power_discharge=10 must be >= min_power_discharge=20 for unit id",
    ):
        d = param_dict.copy()
        d["max_power_discharge"] = 10
        d["min_power_discharge"] = 20
        Storage(**d)
    with pytest.raises(
        ValueError, match="efficiency_charge=1.1 must be between 0 and 1 for unit id"
    ):
        d = param_dict.copy()
        d["efficiency_charge"] = 1.1
        Storage(**d)
    with pytest.raises(
        ValueError, match="efficiency_discharge=1.1 must be between 0 and 1 for unit id"
    ):
        d = param_dict.copy()
        d["efficiency_discharge"] = 1.1
        Storage(**d)
    with pytest.raises(ValueError, match="ramp_up_charge=10 must be <= 0 for unit id"):
        d = param_dict.copy()
        d["ramp_up_charge"] = 10
        Storage(**d)
    with pytest.raises(
        ValueError, match="ramp_down_charge=10 must be <= 0 for unit id"
    ):
        d = param_dict.copy()
        d["ramp_down_charge"] = 10
        Storage(**d)
    with pytest.raises(
        ValueError, match="ramp_up_discharge=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["ramp_up_discharge"] = -10
        Storage(**d)
    with pytest.raises(
        ValueError, match="ramp_down_discharge=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["ramp_down_discharge"] = -10
        Storage(**d)
    with pytest.raises(
        ValueError, match="min_operating_time=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["min_operating_time"] = -10
        Storage(**d)
    with pytest.raises(ValueError, match="min_down_time=-10 must be >= 0 for unit id"):
        d = param_dict.copy()
        d["min_down_time"] = -10
        Storage(**d)
    with pytest.raises(
        ValueError, match="downtime_hot_start=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["downtime_hot_start"] = -10
        Storage(**d)
    with pytest.raises(
        ValueError, match="downtime_warm_start=-10 must be >= 0 for unit id"
    ):
        d = param_dict.copy()
        d["downtime_warm_start"] = -10
        Storage(**d)


class BalancingCapacityMarketConfig:
    """A balancing capacity market - it awards capacity, not energy."""

    market_id = "CRM_pos"
    maximum_bid_price = 3000.0
    minimum_bid_price = -500.0
    product_type = "capacity_pos"
    additional_fields = []


def _order(start, end, volume, price=45):
    return {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "accepted_volume": volume,
        "accepted_price": price,
    }


def _charged_soc(unit, power):
    """The SoC after charging ``power`` MW for one hour from the initial SoC."""
    return unit.initial_soc + power * unit.efficiency_charge / unit.capacity


def test_soc_is_propagated_over_idle_time_steps(mock_market_config, storage_unit):
    """
    Time steps without orders - e.g. after ``remove_empty_bids`` - carry the
    SoC forward instead of keeping the initial SoC (#562, #884).
    """
    t0, t1, t2, t3 = storage_unit.index[:4]
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, -100)])
    storage_unit.set_dispatch_plan(mock_market_config, [])

    charged = _charged_soc(storage_unit, 100)
    assert storage_unit.get_soc(t1) == pytest.approx(charged)
    assert storage_unit.get_soc(t2) == pytest.approx(charged)
    assert storage_unit.get_soc(t3) == pytest.approx(charged)


def test_soc_is_invalidated_by_a_later_commitment(mock_market_config, storage_unit):
    """A commitment within the already propagated range updates the SoC after it."""
    t0, t1, t3 = storage_unit.index[0], storage_unit.index[1], storage_unit.index[3]
    assert storage_unit.get_soc(t3) == storage_unit.initial_soc

    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, -100)])

    assert storage_unit.get_soc(t3) == pytest.approx(_charged_soc(storage_unit, 100))


def test_balancing_capacity_products_do_not_move_the_soc(storage_unit):
    """Only energy products move the SoC, capacity products do not."""
    t0, t1, t3 = storage_unit.index[0], storage_unit.index[1], storage_unit.index[3]
    storage_unit.set_dispatch_plan(
        BalancingCapacityMarketConfig(), [_order(t0, t1, 100, price=20)]
    )

    assert storage_unit.outputs["capacity_pos"].at[t0] == 100
    assert storage_unit.outputs["energy"].at[t0] == 0
    assert storage_unit.get_soc(t3) == storage_unit.initial_soc


def test_projected_soc_matches_executed_soc(mock_market_config, storage_unit):
    """
    The SoC read before delivery equals the SoC after it, also if more is
    committed than the SoC backs: 20 MWh deliver 19 MW instead of 100 MW.
    """
    t0, t1, t2 = storage_unit.index[:3]
    storage_unit.set_soc(t0, 0.02)
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, 100)])
    projected_soc = storage_unit.get_soc(t1)

    storage_unit.execute_current_dispatch(t0, t2)

    assert projected_soc == pytest.approx(storage_unit.min_soc)
    assert storage_unit.get_soc(t1) == pytest.approx(projected_soc)
    assert storage_unit.outputs["energy"].at[t0] == pytest.approx(19)


def test_soc_follows_the_power_limits(mock_market_config, storage_unit):
    """
    Two markets commit 120 MW on a 100 MW unit. The SoC moves by the 100 MW
    the unit can run at, the 120 MW stay the plan until delivery.
    """
    t0, t1 = storage_unit.index[:2]
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, 60)])
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, 60)])

    assert storage_unit.get_soc(t1) == pytest.approx(
        storage_unit.initial_soc
        - 100 / storage_unit.efficiency_discharge / storage_unit.capacity
    )
    assert storage_unit.outputs["energy"].at[t0] == 120
    assert storage_unit.execute_current_dispatch(t0, t0)[0] == 100


@pytest.mark.parametrize("read_soc_in_between", [False, True])
def test_dispatch_does_not_depend_on_when_the_soc_is_read(
    mock_market_config, storage_unit, read_soc_in_between
):
    """
    Reading the SoC between two clearings leaves the committed plan as it is:
    -100 MW and +60 MW add up to -40 MW, also on a nearly full storage.
    """
    t0, t1, t2 = storage_unit.index[:3]
    storage_unit.set_soc(t0, 0.98)
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, -100)])
    if read_soc_in_between:
        storage_unit.get_soc(t2)
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, 60)])

    assert storage_unit.outputs["energy"].at[t0] == -40


def test_planning_builds_on_the_committed_plan(mock_market_config, storage_unit):
    """
    100 MW are sold, but 50 MWh only deliver 47.5 MW. Further bids build on the
    100 MW sold: nothing is left to sell, and all of it can be bought back.
    The SoC builds on the 47.5 MW delivered, so the storage ends empty.
    """
    t0, t1 = storage_unit.index[:2]
    storage_unit.set_soc(t0, 0.05)
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t0, t1, 100)])

    assert storage_unit.get_soc(t1) == pytest.approx(storage_unit.min_soc)
    assert storage_unit.get_output_before(t1) == 100
    _, max_power_discharge = storage_unit.calculate_min_max_discharge(t0, t1)
    _, max_power_charge = storage_unit.calculate_min_max_charge(t0, t1)
    assert max_power_discharge[0] == 0
    assert max_power_charge[0] == storage_unit.max_power_charge - 100
    # without a further market, only 47.5 MW are delivered
    assert storage_unit.execute_current_dispatch(t0, t0)[0] == pytest.approx(47.5)


def test_set_soc_invalidates_everything_after_it(mock_market_config, storage_unit):
    """The SoC after a set SoC is propagated from it again."""
    t1, t2, t3 = storage_unit.index[1], storage_unit.index[2], storage_unit.index[3]
    storage_unit.set_dispatch_plan(mock_market_config, [_order(t1, t2, -100)])
    assert storage_unit.get_soc(t3) != storage_unit.initial_soc

    storage_unit.set_soc(t1, 0.3)

    expected = 0.3 + 100 * storage_unit.efficiency_charge / storage_unit.capacity
    assert storage_unit.get_soc(t2) == pytest.approx(expected)
    assert storage_unit.get_soc(t3) == pytest.approx(expected)


def test_soc_frontier_stays_on_the_index_grid(storage_unit):
    """
    Times between two index steps - the units operator executes from one
    second past the last executed step - are rounded up to the next step.
    """
    t1, t2, t3 = storage_unit.index[1], storage_unit.index[2], storage_unit.index[3]
    storage_unit.outputs["energy"].at[t1] = -100

    storage_unit.execute_current_dispatch(t1 + timedelta(seconds=1), t2)
    assert storage_unit._soc_valid_until in storage_unit.index

    storage_unit.ensure_soc(t3 - timedelta(minutes=30))
    assert storage_unit._soc_valid_until in storage_unit.index
    assert storage_unit.get_soc(t2) == pytest.approx(_charged_soc(storage_unit, 100))


def test_execute_current_dispatch_unaligned_end_does_not_advance_past_executed(
    storage_unit,
):
    """An end between two steps marks the SoC valid only up to the executed steps."""
    t0, t1 = storage_unit.index[:2]
    storage_unit.outputs["energy"].at[t0] = -100
    storage_unit.outputs["energy"].at[t1] = -100

    storage_unit.execute_current_dispatch(t0, t0 + timedelta(minutes=15))

    assert storage_unit._soc_valid_until == t1


def test_execute_current_dispatch_after_the_horizon(mock_market_config, storage_unit):
    """A window after the last time step executes nothing and keeps the plan."""
    t0, t1 = storage_unit.index[:2]
    last, freq = storage_unit.index[-1], storage_unit.index.freq
    storage_unit.set_dispatch_plan(
        mock_market_config,
        [_order(t0, t1, -100), _order(last, last + freq, 250)],
    )

    dispatch = storage_unit.execute_current_dispatch(last + freq, last + 2 * freq)

    assert len(dispatch) == 0
    assert storage_unit.outputs["energy"].at[last] == 250
    assert storage_unit._soc_valid_until in storage_unit.index
    assert storage_unit.get_soc(t1) == pytest.approx(_charged_soc(storage_unit, 100))


def test_ensure_soc_stops_at_the_end_of_the_horizon(storage_unit):
    """Propagating past the last time step stops at the last time step."""
    t3 = storage_unit.index[-1]

    storage_unit.ensure_soc(t3 + 10 * storage_unit.index.freq)

    assert storage_unit._soc_valid_until == t3
    assert storage_unit.get_soc(t3) == storage_unit.initial_soc


def test_ensure_soc_clips_the_last_time_step(mock_market_config, storage_unit):
    """The last time step moves no later SoC, but its dispatch is still clipped."""
    t2, t3 = storage_unit.index[2], storage_unit.index[-1]
    storage_unit.set_dispatch_plan(
        mock_market_config, [_order(t3, t3 + storage_unit.index.freq, 250)]
    )
    storage_unit.execute_current_dispatch(t2, t2)

    assert storage_unit.execute_current_dispatch(t3, t3)[0] == (
        storage_unit.max_power_discharge
    )


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
