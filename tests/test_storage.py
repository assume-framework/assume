import pandas as pd
import pytest

from assume.strategies.naive_strategies import NaiveStrategy
from assume.units import Storage


@pytest.fixture
def storage_unit() -> Storage:
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={"energy": NaiveStrategy()},
        max_power_charge=-100,
        max_power_discharge=100,
        max_volume=1000,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        index=pd.date_range("2022-01-01", periods=4, freq="H"),
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        variable_cost_charge=3,
        variable_cost_discharge=4,
        fixed_cost=1,
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


def test_reset_function(storage_unit):
    storage_unit.current_status = 0
    assert storage_unit.current_status == 0

    storage_unit.reset()
    assert storage_unit.current_status == 1

    # check if total_power_output is reset
    assert storage_unit.outputs["energy"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )
    # the same for pos and neg capacity reserve
    assert storage_unit.outputs["pos_capacity"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )
    assert storage_unit.outputs["neg_capacity"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )


def test_calculate_operational_window(storage_unit):
    storage_unit.reset()

    product_tuple = (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        None,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )
    cost_discharge = storage_unit.calculate_marginal_cost(
        start, max_power_discharge[start]
    )

    assert min_power_discharge[start] == 0
    assert max_power_discharge[start] == 100
    assert cost_discharge == 4 / 0.95 + 1

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end, product_type="energy"
    )
    cost_charge = storage_unit.calculate_marginal_cost(start, max_power_charge[start])

    assert min_power_charge[start] == 0
    assert max_power_charge[start] == -100
    assert cost_charge == 3 / 0.9 + 1

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

    storage_unit.current_SOC = 0.05
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    assert max_power_discharge[0] == round(50 * storage_unit.efficiency_discharge, 3)

    storage_unit.current_SOC = 0.95
    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )
    assert min_power_charge[0] == -40
    assert max_power_charge[0] == round(-50 / storage_unit.efficiency_charge, 3)


def test_storage_feedback(storage_unit, mock_market_config):
    storage_unit.reset()

    product_tuple = (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        None,
    )
    product_type = "energy"

    start = product_tuple[0]
    end = product_tuple[1]
    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end, product_type="energy"
    )
    cost_charge = storage_unit.calculate_marginal_cost(start, max_power_charge[start])

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )
    cost_discharge = storage_unit.calculate_marginal_cost(
        start, max_power_discharge[start]
    )

    assert min_power_charge[start] == 0
    assert max_power_charge[start] == -100

    assert min_power_discharge[start] == 0
    assert max_power_discharge[start] == 100
    assert storage_unit.outputs["energy"][start] == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": cost_discharge,
            "accepted_volume": max_power_discharge[start] / 2,
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
    assert min_power_discharge[start] == 0
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power_discharge[start] == 50

    # second market request for next interval
    start = pd.Timestamp("2022-01-01 01:00:00")
    end = pd.Timestamp("2022-01-01 02:00:00")
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )

    # now we can bid max_power and need min_power again
    assert min_power_discharge[start] == 0
    assert max_power_discharge[start] == 100


def test_storage_ramping(storage_unit):
    storage_unit.reset()

    product_tuple = (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 04:00:00"),
        None,
    )

    start = product_tuple[0]
    end = product_tuple[1]

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end, product_type="energy"
    )
    cost_charge = storage_unit.calculate_marginal_cost(start, max_power_charge[start])

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end, product_type="energy"
    )
    cost_discharge = storage_unit.calculate_marginal_cost(
        start, max_power_discharge[start]
    )

    assert min_power_charge[start] == 0
    assert max_power_charge[start] == -100

    assert min_power_discharge[start] == 0
    assert max_power_discharge[start] == 100

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        0, max_power_discharge[start]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(0, max_power_charge[start])

    assert max_ramp_discharge == 60
    assert max_ramp_charge == -60

    # discharge power gets accepted
    storage_unit.outputs["energy"][start] += 60

    # next hour
    product_tuple = (
        pd.Timestamp("2022-01-01 01:00:00"),
        pd.Timestamp("2022-01-01 02:00:00"),
        None,
    )

    start = product_tuple[0]
    end = product_tuple[1]

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        60, max_power_discharge[start]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(60, max_power_charge[start])

    assert max_ramp_discharge == 100
    assert max_ramp_charge == 0

    # chargin scenario
    storage_unit.outputs["energy"][start] = -60

    # next hour
    product_tuple = (
        pd.Timestamp("2022-01-01 02:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        None,
    )

    start = product_tuple[0]
    end = product_tuple[1]

    max_ramp_discharge = storage_unit.calculate_ramp_discharge(
        -60, max_power_discharge[start]
    )
    max_ramp_charge = storage_unit.calculate_ramp_charge(-60, max_power_charge[start])

    assert max_ramp_discharge == 0
    assert max_ramp_charge == -100


def test_execute_dispatch(storage_unit):
    storage_unit.reset()

    product_tuple = (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        None,
    )
    start = product_tuple[0]
    end = product_tuple[1]

    storage_unit.outputs["energy"][start] = 100
    storage_unit.current_SOC = 0.5
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == 100
    assert storage_unit.current_SOC == round(
        0.5 - 100 / storage_unit.efficiency_discharge / storage_unit.max_volume, 3
    )
    assert storage_unit.current_status == 1
    assert storage_unit.current_down_time == 0
    assert storage_unit.market_success_list == [1]

    storage_unit.outputs["energy"][start] = -100
    storage_unit.current_SOC = 0.5
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == -100
    assert storage_unit.current_SOC == round(
        0.5 + 100 * storage_unit.efficiency_charge / storage_unit.max_volume, 3
    )
    assert storage_unit.current_status == 1
    assert storage_unit.current_down_time == 0
    assert storage_unit.market_success_list == [2]

    storage_unit.outputs["energy"][start] = 100
    storage_unit.current_SOC = 0.05
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == round(50 * storage_unit.efficiency_discharge, 1)
    assert storage_unit.current_SOC == 0
    assert storage_unit.current_status == 1
    assert storage_unit.current_down_time == 0
    assert storage_unit.market_success_list == [3]

    storage_unit.outputs["energy"][start] = -100
    storage_unit.current_SOC = 0.95
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == round(-50 / storage_unit.efficiency_charge, 1)
    assert storage_unit.current_SOC == 1
    assert storage_unit.current_status == 1
    assert storage_unit.current_down_time == 0
    assert storage_unit.market_success_list == [4]

    storage_unit.outputs["energy"][start] = -100
    dispatched_energy = storage_unit.execute_current_dispatch(start, end)
    assert dispatched_energy[0] == 0
    assert storage_unit.current_SOC == 1
    assert storage_unit.current_status == 0
    assert storage_unit.current_down_time == 1
    assert storage_unit.market_success_list == [4, 0]


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
