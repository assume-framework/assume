# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from assume.units import PowerPlant


@pytest.fixture
def power_plant_1() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = NaiveForecast(
        index, availability=1, fuel_price=[10, 11, 12, 13], co2_price=[10, 20, 30, 30]
    )
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        bidding_strategies={"EOM": NaiveSingleBidStrategy()},
        index=forecaster.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
    )


@pytest.fixture
def power_plant_2() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        bidding_strategies={"EOM": NaiveSingleBidStrategy()},
        index=forecaster.index,
        max_power=1000,
        min_power=0,
        efficiency=0.5,
        additional_cost=10,
        fuel_type="lignite",
        forecaster=forecaster,
        emission_factor=0.5,
    )


@pytest.fixture
def power_plant_3() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    forecaster = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        bidding_strategies={"EOM": NaiveSingleBidStrategy()},
        index=forecaster.index,
        max_power=1000,
        min_power=0,
        efficiency=0.5,
        additional_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
        partial_load_eff=True,
    )


def test_init_function(power_plant_1, power_plant_2, power_plant_3):
    assert power_plant_1.id == "test_pp"
    assert power_plant_1.unit_operator == "test_operator"
    assert power_plant_1.technology == "hard coal"
    assert power_plant_1.max_power == 1000
    assert power_plant_1.min_power == 200
    assert power_plant_1.efficiency == 0.5
    assert power_plant_1.additional_cost == 10
    assert power_plant_1.fuel_type == "lignite"
    assert power_plant_1.emission_factor == 0.5
    assert power_plant_1.ramp_up == 1000
    assert power_plant_1.ramp_down == 1000

    index = pd.date_range("2022-01-01", periods=4, freq="h")
    assert (
        power_plant_1.marginal_cost == pd.Series([40.0, 52.0, 64.0, 66.0], index)
    ).all()

    assert (power_plant_2.marginal_cost == pd.Series(40, index)).all()
    assert (power_plant_3.marginal_cost == pd.Series(40, index)).all()


def test_reset_function(power_plant_1):
    # Expected series with zero values
    expected_series = pd.Series(
        0.0, index=pd.date_range("2022-01-01", periods=4, freq="h")
    )

    # Check if total_power_output is reset
    assert (power_plant_1.outputs["energy"].data == expected_series.values).all()

    # The same for pos and neg capacity reserve
    assert (power_plant_1.outputs["pos_capacity"].data == expected_series.values).all()
    assert (power_plant_1.outputs["neg_capacity"].data == expected_series.values).all()

    # The same for total_heat_output and power_loss_chp
    assert (power_plant_1.outputs["heat"].data == expected_series.values).all()
    assert (power_plant_1.outputs["power_loss"].data == expected_series.values).all()


def test_calculate_operational_window(power_plant_1):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    min_cost = power_plant_1.calculate_marginal_cost(start, min_power[0])
    max_cost = power_plant_1.calculate_marginal_cost(start, max_power[0])

    assert min_power[0] == 200
    assert min_cost == 40.0

    assert max_power[0] == 1000
    assert max_cost == 40

    assert power_plant_1.outputs["energy"].at[start] == 0


def test_powerplant_feedback(power_plant_1, mock_market_config):
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    min_cost = power_plant_1.calculate_marginal_cost(start, min_power[0])
    max_cost = power_plant_1.calculate_marginal_cost(start, max_power[0])

    assert min_power[0] == 200
    assert min_cost == 40.0

    assert max_power[0] == 1000
    assert max_cost == 40
    assert power_plant_1.outputs["energy"].at[start] == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": min_cost,
            "accepted_price": min_cost,
            "accepted_volume": min_power[0],
        }
    ]

    # min_power gets accepted
    mc = mock_market_config
    power_plant_1.set_dispatch_plan(mc, orderbook)

    # second market request for same interval
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    # we do not need additional min_power, as our runtime requirement is fulfilled
    assert min_power[0] == 0
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power[0] == 800

    # second market request for next interval
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    # now we can bid max_power and need min_power again
    assert min_power[0] == 200
    assert max_power[0] == 1000


def test_powerplant_ramping(power_plant_1):
    power_plant_1.ramp_down = 100
    power_plant_1.ramp_up = 200
    power_plant_1.min_operating_time = 3
    power_plant_1.min_down_time = 2
    power_plant_1.min_power = 50

    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    end_excl = end - power_plant_1.index.freq
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    assert min_power[0] == 50
    assert max_power[0] == 1000

    op_time = power_plant_1.get_operation_time(start)
    assert op_time == 3

    min_cost = power_plant_1.calculate_marginal_cost(start, min_power[0])
    max_cost = power_plant_1.calculate_marginal_cost(start, max_power[0])
    max_ramp = power_plant_1.calculate_ramp(op_time, 100, max_power[0])
    min_ramp = power_plant_1.calculate_ramp(op_time, 100, min_power[0])

    assert min_ramp == 50
    assert min_cost == 40.0

    assert max_ramp == 300
    assert max_cost == 40

    # min_power gets accepted

    power_plant_1.outputs["energy"].loc[start:end_excl] += 300

    # next hour
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)
    end_excl = end - power_plant_1.index.freq

    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    assert min_power[0] == 50
    assert max_power[0] == 1000

    op_time = power_plant_1.get_operation_time(start)
    assert op_time == 1

    min_ramp = power_plant_1.calculate_ramp(op_time, 300, min_power[0])
    max_ramp = power_plant_1.calculate_ramp(op_time, 300, max_power[0])

    assert min_ramp == 200
    assert max_ramp == 500

    # accept max_power
    power_plant_1.outputs["energy"].loc[start:end_excl] += 500

    # next hour
    start = datetime(2022, 1, 1, 2)
    end = datetime(2022, 1, 1, 3)
    end_excl = end - power_plant_1.index.freq

    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    op_time = power_plant_1.get_operation_time(start)
    assert op_time == 2

    min_ramp = power_plant_1.calculate_ramp(op_time, 500, min_power[0])
    max_ramp = power_plant_1.calculate_ramp(op_time, 500, max_power[0])

    assert min_ramp == 400
    assert max_ramp == 700

    # ramp_up if min_down_time is not reached
    power_plant_1.outputs["energy"].loc[start - power_plant_1.index.freq] = 0

    op_time = power_plant_1.get_operation_time(start)
    assert op_time == -1

    min_ramp = power_plant_1.calculate_ramp(op_time, 0, 0)
    max_ramp = power_plant_1.calculate_ramp(op_time, 0, 100)

    assert min_ramp == 0
    assert max_ramp == 0

    # ramp_down if min_operating_time is not reached
    power_plant_1.outputs["energy"].loc[start - power_plant_1.index.freq * 2] = 0
    power_plant_1.outputs["energy"].loc[start - power_plant_1.index.freq] = 100

    op_time = power_plant_1.get_operation_time(start)
    assert op_time == 1

    min_ramp = power_plant_1.calculate_ramp(op_time, 100, 0)
    max_ramp = power_plant_1.calculate_ramp(op_time, 100, 1000)

    assert min_ramp == 50
    assert max_ramp == 300


def test_powerplant_availability(power_plant_1):
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    ff = NaiveForecast(
        index,
        availability=[0.5, 0.01, 1, 1],
        fuel_price=[10, 11, 12, 13],
        co2_price=[10, 20, 30, 30],
    )
    # set availability
    power_plant_1.forecaster = ff
    power_plant_1.max_power = 1000
    power_plant_1.min_power = 200
    power_plant_1.ramp_down = 1000
    power_plant_1.ramp_up = 1000

    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    ### HOUR 0
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    op_time = power_plant_1.get_operation_time(start)
    max_ramp = power_plant_1.calculate_ramp(op_time, 0, max_power[0])
    assert max_ramp == power_plant_1.max_power / 2

    ### HOUR 1
    start += timedelta(hours=1)
    end += timedelta(hours=1)
    _, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    op_time = power_plant_1.get_operation_time(start)
    # run min_power if 0 < power <= min_power is needed
    max_ramp = power_plant_1.calculate_ramp(op_time, 0, max_power[0])
    assert max_ramp == power_plant_1.min_power

    ### HOUR 2
    start += timedelta(hours=1)
    end += timedelta(hours=1)
    _, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    op_time = power_plant_1.get_operation_time(start)
    max_ramp = power_plant_1.calculate_ramp(op_time, 0, max_power[0])
    assert max_ramp == power_plant_1.max_power


def test_powerplant_execute_dispatch():
    index = pd.date_range("2022-01-01", periods=24, freq="h")
    forecaster = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    power_plant = PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"EOM": NaiveSingleBidStrategy()},
        index=forecaster.index,
        max_power=700,
        min_power=50,
        efficiency=0.5,
        fuel_type="lignite",
        ramp_down=100,
        ramp_up=200,
        min_operating_time=3,
        min_down_time=2,
        forecaster=forecaster,
    )
    # was running before
    assert power_plant.execute_current_dispatch(index[0], index[0])[0] == 0

    power_plant.outputs["energy"].loc[index] = [
        0,
        0,
        0,
        200,
        200,
        100,
        0,
        0,  # correct dispatch
        100,
        100,
        0,
        0,  # breaking min_operating_time
        100,
        0,
        200,
        200,  # breaking min_down_time
        200,
        500,
        600,
        700,  # breaking ramp_up constraint
        700,
        400,
        300,
        200,  # breaking ramp_down constraint
    ]
    assert (
        len(power_plant.execute_current_dispatch(start=index[0], end=index[-1])) == 24
    )
    assert all(
        power_plant.outputs["energy"].loc[index[0] : index[7]]
        == [0, 0, 0, 200, 200, 100, 0, 0]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[8] : index[11]] == [100, 100, 50, 0]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[12] : index[15]] == [0, 0, 200, 200]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[16] : index[19]] == [200, 400, 600, 700]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[20] : index[23]] == [700, 600, 500, 400]
    )

    # check combinations of constraints
    power_plant.outputs["energy"].loc[index] = [
        0,
        0,
        200,
        0,  # breaking min_operation_time and ramp_down
        0,
        0,
        500,
        750,  # breaking min_down_time and ramp_up
        400,
        20,
        320,
        200,  # ramp_down
        120,
        20,
        20,
        220,  # breaking min_power and ramp_down
        420,
        720,
        620,
        520,
        420,
        720,
        800,
        700,  # breaking max_power and ramp_up
    ]
    power_plant.execute_current_dispatch(start=index[0], end=index[-1])
    assert all(
        power_plant.outputs["energy"].loc[index[0] : index[3]] == [0, 0, 200, 100]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[4] : index[7]] == [50, 0, 0, 200]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[8] : index[11]] == [400, 300, 320, 220]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[12] : index[15]] == [120, 50, 50, 220]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[16] : index[19]] == [420, 620, 620, 520]
    )
    assert all(
        power_plant.outputs["energy"].loc[index[20] : index[23]] == [420, 620, 700, 700]
    )


def test_powerplant_min_feedback(power_plant_1, mock_market_config):
    """
    Test that powerplant works fine for multi market bidding.
    Has two bids which add up to be above the minimum power.
    Make sure that ramping is not enforced to early.
    """
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    product_type = "energy"

    # start bidding by calculating min and max power
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )
    assert min_power[0] == 200
    assert max_power[0] == 1000
    assert power_plant_1.outputs[product_type].at[start] == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 40,
            "accepted_price": 40,
            "accepted_volume": 100,
            # half of min_power
        }
    ]

    # min_power gets accepted by fictional market
    power_plant_1.set_dispatch_plan(mock_market_config, orderbook)

    # second market request for same interval
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )

    # we still need 100kw as a runtime requirement
    assert min_power[0] == 100
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power[0] == 900

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 40,
            "accepted_price": 40,
            "accepted_volume": 200,
            # half of min_power
        }
    ]

    # min_power gets accepted
    power_plant_1.set_dispatch_plan(mock_market_config, orderbook)

    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )

    # we do not need additional min_power, as our runtime requirement is fulfilled
    assert min_power[0] == 0
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power[0] == 700

    # this should not do anything here, as we are in our constraints
    power_plant_1.execute_current_dispatch(start, end)

    # second market request for next interval
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )

    # now we can bid max_power and need min_power again
    assert min_power[0] == 200
    assert max_power[0] == 1000


def test_powerplant_ramp_feedback(power_plant_1, mock_market_config):
    """
    Make sure that ramping is enforced when a accepted volume at the prior time
    is below the minimum power.
    """
    product_type = "energy"
    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)

    # start bidding by calculating min and max power
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )
    assert min_power[0] == 200
    assert max_power[0] == 1000
    assert power_plant_1.outputs[product_type].at[start] == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 40,
            "accepted_price": 40,
            "accepted_volume": 100,
            # half of min_power
        }
    ]

    # min_power gets accepted by fictional market
    power_plant_1.set_dispatch_plan(mock_market_config, orderbook)

    # second market request for same interval
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )

    # we still need 100kw as a runtime requirement
    assert min_power[0] == 100
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power[0] == 900

    power_plant_1.execute_current_dispatch(start, end)

    # second market request for next interval
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type=product_type
    )

    # now we can bid max_power and need min_power again
    assert min_power[0] == 200
    assert max_power[0] == 1000


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
