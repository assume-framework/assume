from datetime import datetime, timedelta

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import NaiveStrategy
from assume.units import PowerPlant


@pytest.fixture
def power_plant_1() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq="H")
    ff = NaiveForecast(
        index, availability=1, fuel_price=[10, 11, 12, 13], co2_price=[10, 20, 30, 30]
    )
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"energy": NaiveStrategy()},
        index=index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        fixed_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
    )


@pytest.fixture
def power_plant_2() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq="H")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"energy": NaiveStrategy()},
        index=index,
        max_power=1000,
        min_power=0,
        efficiency=0.5,
        fixed_cost=10,
        fuel_type="lignite",
        forecaster=ff,
        emission_factor=0.5,
    )


@pytest.fixture
def power_plant_3() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq="H")
    ff = NaiveForecast(index, availability=1, fuel_price=10, co2_price=10)
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"energy": NaiveStrategy()},
        index=index,
        max_power=1000,
        min_power=0,
        efficiency=0.5,
        fixed_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=ff,
        partial_load_eff=True,
    )


def test_init_function(power_plant_1, power_plant_2, power_plant_3):
    assert power_plant_1.id == "test_pp"
    assert power_plant_1.unit_operator == "test_operator"
    assert power_plant_1.technology == "coal"
    assert power_plant_1.max_power == 1000
    assert power_plant_1.min_power == 200
    assert power_plant_1.efficiency == 0.5
    assert power_plant_1.fixed_cost == 10
    assert power_plant_1.fuel_type == "lignite"
    assert power_plant_1.emission_factor == 0.5
    assert power_plant_1.ramp_up == 1000
    assert power_plant_1.ramp_down == 1000
    index = pd.date_range("2022-01-01", periods=4, freq="H")
    assert (
        power_plant_1.marginal_cost.to_dict()
        == pd.Series(
            [40.0, 52.0, 64.0, 66.0],
            index,
        ).to_dict()
    )

    assert power_plant_2.marginal_cost.to_dict() == pd.Series(40, index).to_dict()
    assert power_plant_3.marginal_cost.to_dict() == pd.Series(40, index).to_dict()


def test_reset_function(power_plant_1):
    power_plant_1.current_status = 0
    assert power_plant_1.current_status == 0

    power_plant_1.reset()
    assert power_plant_1.current_status == 1

    # check if total_power_output is reset
    assert power_plant_1.outputs["energy"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )
    # the same for pos and neg capacity reserve
    assert power_plant_1.outputs["pos_capacity"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )
    assert power_plant_1.outputs["neg_capacity"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )

    # the same for total_heat_output and power_loss_chp
    assert power_plant_1.outputs["heat"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )
    assert power_plant_1.outputs["power_loss"].equals(
        pd.Series(0.0, index=pd.date_range("2022-01-01", periods=4, freq="H"))
    )


def test_calculate_operational_window(power_plant_1):
    power_plant_1.reset()

    product_tuple = (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        None,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    min_cost = power_plant_1.calculate_marginal_cost(start, min_power[start])
    max_cost = power_plant_1.calculate_marginal_cost(start, max_power[start])

    assert min_power[start] == 200
    assert min_cost == 40.0

    assert max_power[start] == 1000
    assert max_cost == 40

    assert power_plant_1.outputs["energy"].at[start] == 0


def test_powerplant_feedback(power_plant_1, mock_market_config):
    power_plant_1.reset()

    product_tuple = (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-01-01 01:00:00"),
        None,
    )
    product_type = "energy"
    start = product_tuple[0]
    end = product_tuple[1]
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    min_cost = power_plant_1.calculate_marginal_cost(start, min_power[start])
    max_cost = power_plant_1.calculate_marginal_cost(start, max_power[start])

    assert min_power[start] == 200
    assert min_cost == 40.0

    assert max_power[start] == 1000
    assert max_cost == 40
    assert power_plant_1.outputs["energy"].at[start] == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": min_cost,
            "volume": min_power[start],
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
    assert min_power[start] == 0
    # we can not bid the maximum anymore, because we already provide energy on the other market
    assert max_power[start] == 800

    # second market request for next interval
    start = pd.Timestamp("2022-01-01 01:00:00")
    end = pd.Timestamp("2022-01-01 02:00:00")
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    # now we can bid max_power and need min_power again
    assert min_power[start] == 200
    assert max_power[start] == 1000


def test_powerplant_ramping(power_plant_1):
    power_plant_1.ramp_down = 100
    power_plant_1.ramp_up = 200
    power_plant_1.reset()

    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    min_cost = power_plant_1.calculate_marginal_cost(start, min_power[start])
    max_cost = power_plant_1.calculate_marginal_cost(start, max_power[start])
    max_ramp = power_plant_1.calculate_ramp(0, max_power[start])

    assert min_power[start] == 200
    assert min_cost == 40.0

    assert max_ramp == 200
    assert max_cost == 40

    # min_power gets accepted
    end_excl = end - power_plant_1.index.freq
    power_plant_1.outputs["energy"].loc[start:end_excl] += 200

    # next hour
    start = datetime(2022, 1, 1, 1)
    end = datetime(2022, 1, 1, 2)

    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    min_ramp = power_plant_1.calculate_ramp(200, min_power[start])
    max_ramp = power_plant_1.calculate_ramp(200, max_power[start])

    assert min_ramp == 200
    assert max_ramp == 400

    # accept max_power
    power_plant_1.outputs["energy"].loc[start:end_excl] += 400

    # next hour
    start = datetime(2022, 1, 1, 2)
    end = datetime(2022, 1, 1, 3)

    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    min_ramp = power_plant_1.calculate_ramp(400, min_power[start])
    max_ramp = power_plant_1.calculate_ramp(400, max_power[start])

    assert min_ramp == 300
    assert max_ramp == 600


def test_powerplant_availability(power_plant_1):
    index = pd.date_range("2022-01-01", periods=4, freq="H")
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
    power_plant_1.reset()

    start = datetime(2022, 1, 1, 0)
    end = datetime(2022, 1, 1, 1)
    ### HOUR 0
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )

    max_ramp = power_plant_1.calculate_ramp(0, max_power[start])
    assert max_ramp == power_plant_1.max_power / 2

    ### HOUR 1
    start += timedelta(hours=1)
    end += timedelta(hours=1)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    # run min_power if 0 < power <= min_power is needed
    max_ramp = power_plant_1.calculate_ramp(0, max_power[start])
    assert max_ramp == power_plant_1.min_power

    ### HOUR 2
    start += timedelta(hours=1)
    end += timedelta(hours=1)
    min_power, max_power = power_plant_1.calculate_min_max_power(
        start, end, product_type="energy"
    )
    max_ramp = power_plant_1.calculate_ramp(0, max_power[start])
    assert max_ramp == power_plant_1.max_power


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
