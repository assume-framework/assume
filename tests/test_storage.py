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
    assert max_power_discharge[0] == 60

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
        delta_soc_set_dispatch = (
            storage_unit.outputs["soc"][s] - storage_unit.outputs["soc"][s_next]
        )

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


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])


@pytest.fixture
def small_storage() -> Storage:
    """20 MWh / 20 MW storage, lossless, starting half full."""
    index = pd.date_range("2023-01-02", periods=6, freq="h")
    forecaster = UnitForecaster(index, availability=1, market_prices={"EOM": 50})
    return Storage(
        id="small_storage",
        unit_operator="op",
        technology="battery",
        bidding_strategies={},
        forecaster=forecaster,
        max_power_charge=-20,
        max_power_discharge=20,
        capacity=20,
        initial_soc=0.5,
    )


def test_calculate_soc_path_follows_schedule_and_clips(small_storage):
    s = small_storage
    t = s.index
    # charge 8 MW, then 8 MW again (only 2 MWh of room left), then discharge 5 MW
    s.outputs["energy"].loc[t[0] : t[2]] = [-8, -8, 5]

    path = s.calculate_soc_path(t[0], t[4])

    assert [round(path[x] * s.capacity, 6) for x in t[:5]] == [10, 18, 20, 15, 15]


def test_calculate_soc_path_beyond_index_treats_storage_as_idle(small_storage):
    s = small_storage
    t = s.index
    s.outputs["energy"].at[t[5]] = 4  # last time step in the index

    path = s.calculate_soc_path(t[5], t[5] + 2 * s.index.freq)

    assert path[t[5] + s.index.freq] * s.capacity == pytest.approx(6)
    assert path[t[5] + 2 * s.index.freq] * s.capacity == pytest.approx(6)


def test_calculate_redispatch_energy_headroom(small_storage):
    assert small_storage.calculate_redispatch_energy_headroom(0.25) == (5.0, 15.0)
    assert small_storage.calculate_redispatch_energy_headroom(1.0) == (20.0, 0.0)


def test_add_redispatch_soc_constraints_limits_cumulative_energy():
    """4 snapshots, 20 MW of upward redispatch would be cheap, but only 10 MWh exist."""
    linopy = pytest.importorskip("linopy")

    m = linopy.Model()
    snapshots = pd.RangeIndex(4, name="snapshot")
    up = m.add_variables(lower=0, upper=20, coords=[snapshots], name="up")
    down = m.add_variables(lower=0, upper=20, coords=[snapshots], name="down")
    m.add_objective(-up.sum() + 0.1 * down.sum())  # maximise upward redispatch

    Storage.add_redispatch_soc_constraints(
        model=m,
        name="s",
        up=up,
        down=down,
        duration_hours=[1, 1, 1, 1],
        energy_discharge_headroom=[10, 10, 10, 10],
        energy_charge_headroom=[10, 10, 10, 10],
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
    )
    status, _ = m.solve(solver_name="highs")

    assert status == "ok"
    assert float(up.solution.sum() - down.solution.sum()) == pytest.approx(10)


def test_set_dispatch_plan_warns_if_redispatch_is_not_soc_feasible(
    small_storage, caplog
):
    from assume.common.market_objects import MarketConfig, MarketProduct

    s = small_storage
    t = s.index
    config = MarketConfig(
        market_id="redispatch",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        product_type="energy",
        market_mechanism="redispatch",
    )
    # 20 MW for 1 h needs 20 MWh, but only 10 MWh are stored: 10 MWh are curtailed
    order = {
        "start_time": t[0],
        "end_time": t[1],
        "accepted_volume": 20,
        "accepted_price": 50,
    }
    with caplog.at_level("WARNING"):
        s.set_dispatch_plan(config, [order])

    assert "not SOC-feasible" in caplog.text
    assert s.outputs["energy"].at[t[0]] == pytest.approx(10)


def test_as_dict_keeps_efficiencies_in_place(small_storage):
    small_storage.efficiency_charge = 0.9
    small_storage.efficiency_discharge = 0.8
    d = small_storage.as_dict()
    assert d["efficiency_charge"] == 0.9
    assert d["efficiency_discharge"] == 0.8


@pytest.mark.parametrize(
    "initial_soc, da_volume, expected_up, expected_down",
    [
        (1.0, 50.0, 50.0, 50.0),  # full, discharging: can still discharge less
        (0.0, -50.0, 50.0, 50.0),  # empty, charging: can still charge less
        (0.5, 0.0, 50.0, 50.0),  # half full, idle: 50 MWh either way
    ],
)
def test_redispatch_bid_power_range_around_da_schedule(
    initial_soc, da_volume, expected_up, expected_down
):
    """100 MWh / 100 MW lossless battery: redispatch range around the DA schedule."""
    from assume.common.market_objects import MarketConfig, MarketProduct
    from assume.strategies.flexable_storage import (
        StorageEnergyHeuristicRedispatchStrategy,
    )

    index = pd.date_range("2023-01-02", periods=4, freq="h")
    t0, t1, t2 = index[:3]
    storage = Storage(
        id="battery",
        unit_operator="op",
        technology="battery",
        bidding_strategies={},
        forecaster=UnitForecaster(index, availability=1, market_prices={"EOM": 50}),
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=100,
        initial_soc=initial_soc,
    )

    def config(market_id):
        return MarketConfig(
            market_id=market_id,
            product_type="energy",
            market_mechanism="redispatch"
            if market_id == "redispatch"
            else "pay_as_clear",
            market_products=[MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))],
        )

    def order(t, volume):
        return {
            "start_time": t,
            "end_time": t + timedelta(hours=1),
            "accepted_volume": volume,
            "accepted_price": 50,
        }

    storage.set_dispatch_plan(config("EOM"), [order(t0, da_volume), order(t1, 0)])
    bid = StorageEnergyHeuristicRedispatchStrategy().calculate_bids(
        storage, config("redispatch"), [(t0, t1, None), (t1, t2, None)]
    )[0]

    assert bid["volume"] == pytest.approx(da_volume)
    assert bid["max_power"] - bid["volume"] == pytest.approx(expected_up)
    assert bid["volume"] - bid["min_power"] == pytest.approx(expected_down)


def test_soc_follows_da_plus_redispatch():
    """Full 100 MWh battery: DA accepts 50 of a 70 MW bid, redispatch adds +20."""
    from assume.common.market_objects import MarketConfig, MarketProduct

    index = pd.date_range("2023-01-02", periods=4, freq="h")
    t0, t1 = index[:2]
    storage = Storage(
        id="battery",
        unit_operator="op",
        technology="battery",
        bidding_strategies={},
        forecaster=UnitForecaster(index, availability=1, market_prices={"EOM": 50}),
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=100,
        initial_soc=1.0,
    )
    eom = MarketConfig(
        market_id="EOM",
        product_type="energy",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )
    redispatch = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        market_mechanism="redispatch",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    def order(volume):
        return {
            "start_time": t0,
            "end_time": t1,
            "accepted_volume": volume,
            "accepted_price": 50,
        }

    storage.set_dispatch_plan(eom, [order(50)])  # only the accepted volume counts
    assert storage.outputs["soc"].at[t1] * 100 == pytest.approx(50)
    storage.set_dispatch_plan(redispatch, [order(20)])
    storage.execute_current_dispatch(t0, t0)

    assert storage.outputs["energy"].at[t0] == pytest.approx(70)
    assert storage.outputs["soc"].at[t1] * 100 == pytest.approx(30)