# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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

    storage_unit.set_soc(start - timedelta(hours=1), 0.05)
    assert storage_unit.get_soc(start - storage_unit.index.freq) == 0.05
    # the SoC at start is propagated from the one just set - the unit is nearly
    # empty, so the discharge limit binds on the SoC rather than on the power
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    assert math.isclose(
        max_power_discharge[0],
        0.05 * storage_unit.capacity * storage_unit.efficiency_discharge,
    )

    storage_unit.set_soc(start, 0.95)
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


class CapacityMarketConfig:
    """A reserve market - its product is a capacity reservation, not energy."""

    market_id = "CRM_pos"
    maximum_bid_price = 3000.0
    minimum_bid_price = -500.0
    product_type = "capacity_pos"
    additional_fields = []


def test_soc_is_propagated_over_idle_time_steps(mock_market_config, storage_unit):
    """
    Regression test for https://github.com/assume-framework/assume/issues/562.

    ``outputs["soc"]`` is pre-filled with ``initial_soc``, so *not* writing a
    time step is not the same as carrying the SoC forward. Reading through
    ``get_soc`` propagates over the idle steps instead of returning the filler.
    """
    mc = mock_market_config
    t0, t1, t2, t3 = storage_unit.index[:4]

    # charge 100 MW during the first hour only
    order = {
        "start_time": t0,
        "end_time": t1,
        "only_hours": None,
        "accepted_volume": -100,
        "accepted_price": 10,
    }
    storage_unit.set_dispatch_plan(mc, [order])

    expected_soc = storage_unit.initial_soc + (
        100 * storage_unit.efficiency_charge / storage_unit.capacity
    )
    assert math.isclose(storage_unit.get_soc(t1), expected_soc)
    # the idle hours must keep the charged SoC, not fall back to initial_soc
    assert math.isclose(storage_unit.get_soc(t2), expected_soc)
    assert math.isclose(storage_unit.get_soc(t3), expected_soc)


def test_soc_without_market_participation(mock_market_config, storage_unit):
    """
    Regression test for https://github.com/assume-framework/assume/issues/837.

    A unit which submitted no bids - or whose bids ``remove_empty_bids`` dropped
    - gets an empty orderbook. That writes no volume and so must not disturb the
    SoC, which stays readable for the whole horizon.
    """
    mc = mock_market_config
    t0, t1, t3 = storage_unit.index[0], storage_unit.index[1], storage_unit.index[3]

    order = {
        "start_time": t0,
        "end_time": t1,
        "only_hours": None,
        "accepted_volume": -100,
        "accepted_price": 10,
    }
    storage_unit.set_dispatch_plan(mc, [order])
    soc_after_charge = storage_unit.get_soc(t1)
    assert soc_after_charge != storage_unit.initial_soc

    # next delivery period: this unit is not in the orderbook at all
    storage_unit.set_dispatch_plan(mc, [])

    assert math.isclose(storage_unit.get_soc(t3), soc_after_charge)


def test_soc_is_invalidated_by_a_later_commitment(mock_market_config, storage_unit):
    """
    A read propagates the SoC to the point it was asked for. A commitment made
    *afterwards*, inside the already propagated range, has to invalidate it -
    otherwise the stale trajectory would be handed out unchanged.
    """
    mc = mock_market_config
    t0, t1, t2, t3 = storage_unit.index[:4]

    # nothing committed yet - the SoC is flat over the whole horizon
    assert storage_unit.get_soc(t3) == storage_unit.initial_soc

    # now a market clears energy in the first hour, inside the propagated range
    order = {
        "start_time": t0,
        "end_time": t1,
        "only_hours": None,
        "accepted_volume": -100,
        "accepted_price": 10,
    }
    storage_unit.set_dispatch_plan(mc, [order])

    expected_soc = storage_unit.initial_soc + (
        100 * storage_unit.efficiency_charge / storage_unit.capacity
    )
    assert math.isclose(storage_unit.get_soc(t2), expected_soc)
    assert math.isclose(storage_unit.get_soc(t3), expected_soc)


def test_capacity_products_do_not_move_the_soc(storage_unit):
    """
    Only energy moves the SoC. A reserve market awards *capacity*, which is
    written to ``outputs["capacity_pos"]`` and leaves ``outputs["energy"]``
    untouched, so it must neither drain nor fill the storage.
    """
    mc = CapacityMarketConfig()
    t0, t1, t3 = storage_unit.index[0], storage_unit.index[1], storage_unit.index[3]
    order = {
        "start_time": t0,
        "end_time": t1,
        "only_hours": None,
        "accepted_volume": 100,
        "accepted_price": 20,
    }

    storage_unit.set_dispatch_plan(mc, [order])

    assert storage_unit.outputs["capacity_pos"].at[t0] == 100
    assert storage_unit.outputs["energy"].at[t0] == 0
    assert storage_unit.get_soc(t3) == storage_unit.initial_soc


def test_projected_soc_matches_executed_soc(mock_market_config, storage_unit):
    """
    The SoC path a bidding strategy reads must be the one which is later
    actually executed - otherwise strategies plan against a trajectory
    ``execute_current_dispatch`` will not reproduce. Here the accepted discharge
    volume exceeds what the SoC allows, so both have to clip it the same way.
    """
    mc = mock_market_config
    t0, t1, t2 = storage_unit.index[:3]
    storage_unit.set_soc(t0, 0.02)  # 20 MWh left

    order = {
        "start_time": t0,
        "end_time": t1,
        "only_hours": None,
        "accepted_volume": 100,  # more than the SoC supports
        "accepted_price": 45,
    }
    storage_unit.set_dispatch_plan(mc, [order])

    projected_soc = storage_unit.get_soc(t1)
    projected_energy = storage_unit.get_feasible_energy(t0, t0)[0]
    assert projected_soc >= storage_unit.min_soc
    assert projected_energy < 100

    storage_unit.execute_current_dispatch(t0, t2)

    assert math.isclose(storage_unit.get_soc(t1), projected_soc)
    assert math.isclose(storage_unit.outputs["energy"].at[t0], projected_energy)


def _storage(**overrides) -> Storage:
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = UnitForecaster(index, availability=1, market_prices={"EOM": 50})
    kwargs = dict(
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
        initial_soc=0.5,
    )
    kwargs.update(overrides)
    return Storage(**kwargs)


def _energy_order(start, end, volume, price=45):
    return {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "accepted_volume": volume,
        "accepted_price": price,
    }


def test_ensure_soc_clips_to_the_power_limits(mock_market_config, storage_unit):
    """
    A market can accept more than the unit can physically run at. The SoC must
    be derived from the power the unit can actually deliver, not from the
    accepted volume, and the committed energy has to be corrected along with it.
    """
    mc = mock_market_config
    t0, t1 = storage_unit.index[0], storage_unit.index[1]

    # 250 MW is well within what the SoC supports (475 MW), but way above the
    # 100 MW the unit can discharge at
    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, 250)])

    assert storage_unit.get_soc(t1) == pytest.approx(
        storage_unit.initial_soc
        - (storage_unit.max_power_discharge / storage_unit.efficiency_discharge)
        / storage_unit.capacity
    )
    assert storage_unit.get_feasible_energy(t0, t0)[0] == (
        storage_unit.max_power_discharge
    )
    # ... while the plan itself is left alone until it is executed
    assert storage_unit.outputs["energy"].at[t0] == 250


def test_ensure_soc_clips_volumes_stacked_by_several_markets(
    mock_market_config, storage_unit
):
    """
    Every single clearing can be feasible on its own while their sum is not -
    two markets awarding 60 MW each add up to 120 MW on a 100 MW unit. Clipping
    during propagation catches that, clipping at commit time cannot.
    """
    mc = mock_market_config
    t0, t1 = storage_unit.index[0], storage_unit.index[1]

    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, 60)])
    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, 60)])
    assert storage_unit.outputs["energy"].at[t0] == 120

    assert storage_unit.get_soc(t1) == pytest.approx(
        storage_unit.initial_soc
        - (storage_unit.max_power_discharge / storage_unit.efficiency_discharge)
        / storage_unit.capacity
    )
    assert storage_unit.get_feasible_energy(t0, t0)[0] == (
        storage_unit.max_power_discharge
    )


def test_ensure_soc_applies_both_the_power_and_the_soc_limit(
    mock_market_config, storage_unit
):
    """
    The two limits are independent: the power limit caps 250 MW at 100 MW, and
    an almost empty storage caps it further at the 19 MW its 20 MWh can deliver.
    """
    mc = mock_market_config
    t0, t1 = storage_unit.index[0], storage_unit.index[1]
    storage_unit.set_soc(t0, 0.02)  # 20 MWh left

    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, 250)])

    soc_max_discharge = storage_unit.calculate_soc_max_discharge(0.02)
    assert soc_max_discharge < storage_unit.max_power_discharge
    assert storage_unit.get_soc(t1) == pytest.approx(storage_unit.min_soc)
    assert storage_unit.get_feasible_energy(t0, t0)[0] == pytest.approx(
        soc_max_discharge
    )


def test_ensure_soc_clips_the_last_time_step(mock_market_config, storage_unit):
    """
    The energy of the last time step moves no following SoC, so it is easily
    left out of the propagation walk - but it is still dispatched and written to
    the database, and therefore has to be feasible like every other time step.
    """
    mc = mock_market_config
    t2, t3 = storage_unit.index[2], storage_unit.index[-1]
    storage_unit.set_dispatch_plan(
        mc, [_energy_order(t3, t3 + storage_unit.index.freq, 250)]
    )

    # the last step used to be skipped by the walk, and once the frontier had
    # reached it nothing could pull it back to clip it either
    storage_unit.execute_current_dispatch(t2, t2)

    assert storage_unit.execute_current_dispatch(t3, t3)[0] == (
        storage_unit.max_power_discharge
    )
    assert storage_unit.outputs["energy"].at[t3] == storage_unit.max_power_discharge


def test_ensure_soc_drops_power_below_the_minimum():
    """
    A unit cannot run between zero and its minimum power, so such a volume
    becomes zero and leaves the SoC untouched.
    """
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = UnitForecaster(index, availability=1, market_prices={"EOM": 50})
    unit = Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"EOM": StorageEnergyHeuristicFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        min_power_charge=-20,
        min_power_discharge=20,
        capacity=1000,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        initial_soc=0.5,
    )
    t0, t1, t2 = unit.index[:3]
    unit.outputs["energy"].at[t0] = 5  # below min_power_discharge
    unit.outputs["energy"].at[t1] = -5  # above min_power_charge

    assert unit.get_soc(t2) == unit.initial_soc
    assert unit.get_feasible_energy(t0, t1) == pytest.approx([0, 0])


def test_ensure_soc_is_a_no_op_once_covered(mock_market_config, storage_unit):
    """
    Propagation must be idempotent - a second call over an already covered range
    may neither move the SoC again nor clip the energy a second time.
    """
    mc = mock_market_config
    t0, t1, t2 = storage_unit.index[:3]
    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, 250)])

    storage_unit.ensure_soc(t2)
    soc = dict(
        zip(storage_unit.index[:3], [storage_unit.get_soc(t) for t in (t0, t1, t2)])
    )
    energy = storage_unit.outputs["energy"].at[t0]

    storage_unit.ensure_soc(t2)
    storage_unit.ensure_soc(t1)

    assert storage_unit._soc_valid_until == t2
    assert storage_unit.outputs["energy"].at[t0] == energy
    for t, expected in soc.items():
        assert storage_unit.get_soc(t) == expected


def test_ensure_soc_stops_at_the_end_of_the_horizon(storage_unit):
    """
    There is no SoC past the last time step, so asking for one clamps to the
    horizon instead of running off the index.
    """
    t3 = storage_unit.index[-1]

    storage_unit.ensure_soc(t3 + 10 * storage_unit.index.freq)

    assert storage_unit._soc_valid_until == t3
    assert storage_unit.get_soc(t3) == storage_unit.initial_soc


def test_set_soc_does_not_validate_the_range_before_it(
    mock_market_config, storage_unit
):
    """
    Moving the frontier forward to ``t`` would mark everything up to ``t`` as
    valid - including time steps which still hold the pre-filled ``initial_soc``.
    ``set_soc`` therefore propagates up to ``t`` before overwriting it.
    """
    mc = mock_market_config
    t0, t1, t2 = storage_unit.index[:3]

    # charge in the first hour, so t1 differs from the pre-filled initial_soc
    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, -100)])

    expected_t1 = storage_unit.initial_soc + (
        100 * storage_unit.efficiency_charge / storage_unit.capacity
    )
    # the frontier is still at t0, and a known SoC is set past it
    assert storage_unit._soc_valid_until == t0
    storage_unit.set_soc(t2, 0.7)

    assert storage_unit.get_soc(t1) == pytest.approx(expected_t1)
    assert storage_unit.get_soc(t2) == 0.7


def test_set_soc_invalidates_everything_after_it(mock_market_config, storage_unit):
    """A known SoC replaces the derived one, so the path after it is rebuilt."""
    mc = mock_market_config
    t1, t2, t3 = storage_unit.index[1], storage_unit.index[2], storage_unit.index[3]
    storage_unit.set_dispatch_plan(mc, [_energy_order(t1, t2, -100)])
    assert storage_unit.get_soc(t3) != storage_unit.initial_soc

    storage_unit.set_soc(t1, 0.3)

    expected = 0.3 + (100 * storage_unit.efficiency_charge / storage_unit.capacity)
    assert storage_unit.get_soc(t2) == pytest.approx(expected)
    assert storage_unit.get_soc(t3) == pytest.approx(expected)


def test_soc_frontier_stays_on_the_index_grid(storage_unit):
    """
    Slicing the index rounds a start up, so a frontier between two time steps
    would skip the step it falls inside and that step's energy would never move
    the SoC. ``execute_current_dispatch`` is called with exactly such an
    unaligned start - one second past the last executed step - so the frontier
    is aligned before it is stored. Aligning *up* keeps the exclusion of the
    already executed step intact. No current call order makes the unaligned
    frontier produce a wrong SoC, so this guards the invariant itself.
    """
    t0, t1, t2, t3 = storage_unit.index[:4]
    storage_unit.outputs["energy"].at[t1] = -100

    # one second past t1, the way the units operator excludes the executed step
    storage_unit.execute_current_dispatch(t1 + timedelta(seconds=1), t2)
    assert storage_unit._soc_valid_until in storage_unit.index

    storage_unit.ensure_soc(t3 - timedelta(minutes=30))
    assert storage_unit._soc_valid_until in storage_unit.index

    # an unaligned `until` covers the step it falls into rather than half of it
    charged = 100 * storage_unit.efficiency_charge / storage_unit.capacity
    assert storage_unit.get_soc(t2) == pytest.approx(storage_unit.initial_soc + charged)


def test_execute_current_dispatch_unaligned_end_does_not_advance_past_executed(
    storage_unit,
):
    """
    An off-grid end must not round up and advance the SoC validity frontier past
    the time steps that were actually executed.
    """
    t0, t1, t2 = storage_unit.index[:3]
    storage_unit.outputs["energy"].at[t0] = -100
    storage_unit.outputs["energy"].at[t1] = -100

    # Call with off-grid end falling between t0 and t1; only t0 is executed
    off_grid_end = t0 + timedelta(minutes=15)
    storage_unit.execute_current_dispatch(t0, off_grid_end)

    # Only t0 was executed, updating SoC at t1. The frontier must be t1 (not t2).
    assert storage_unit._soc_valid_until == t1


def test_dispatch_does_not_depend_on_when_the_soc_is_read(mock_market_config):
    """
    Clipping the committed volume in place would make the recorded dispatch
    depend on whether a strategy happened to read the SoC between two market
    clearings - and ``get_soc`` is called on every bidding round. On a nearly
    full storage that flips the unit from charging to discharging.
    """
    mc = mock_market_config

    def run(peek: bool) -> float:
        unit = _storage(initial_soc=0.98)
        t0, t2 = unit.index[0], unit.index[2]
        unit.set_dispatch_plan(mc, [_energy_order(t0, t0 + unit.index.freq, -100)])
        if peek:
            unit.get_soc(t2)  # what a bidding strategy does between clearings
        unit.set_dispatch_plan(mc, [_energy_order(t0, t0 + unit.index.freq, 60)])
        unit.get_soc(t2)
        return unit.outputs["energy"].at[t0]

    assert run(peek=False) == run(peek=True) == -40


def test_planned_energy_survives_until_it_is_executed(mock_market_config, storage_unit):
    """
    outputs["energy"] is what markets committed - a plan. It only becomes what
    the unit runs at once execute_current_dispatch delivers it.
    """
    mc = mock_market_config
    t0, t1 = storage_unit.index[0], storage_unit.index[1]
    storage_unit.set_dispatch_plan(mc, [_energy_order(t0, t1, 250)])

    storage_unit.get_soc(t1)
    assert storage_unit.outputs["energy"].at[t0] == 250

    storage_unit.execute_current_dispatch(t0, t0)
    assert storage_unit.outputs["energy"].at[t0] == storage_unit.max_power_discharge


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


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
