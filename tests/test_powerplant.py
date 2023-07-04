from unittest.mock import MagicMock

import pandas as pd
import pytest

from assume.units import PowerPlant


@pytest.fixture
def power_plant_1():
    # Create a PowerPlant instance with some example parameters
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"energy": "naive"},
        index=pd.date_range("2022-01-01", periods=4, freq="H"),
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        fixed_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        fuel_price=pd.Series(
            [10, 11, 12, 13], index=pd.date_range("2022-01-01", periods=4, freq="H")
        ),
        co2_price=pd.Series(
            [10, 20, 30, 30], index=pd.date_range("2022-01-01", periods=4, freq="H")
        ),
    )


@pytest.fixture
def power_plant_2():
    # Create a PowerPlant instance with some example parameters
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"energy": "naive"},
        index=pd.date_range("2022-01-01", periods=4, freq="H"),
        max_power=1000,
        min_power=0,
        efficiency=0.5,
        fixed_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        fuel_price=10,
        co2_price=10,
    )


@pytest.fixture
def power_plant_3():
    # Create a PowerPlant instance with some example parameters
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="coal",
        bidding_strategies={"energy": "naive"},
        index=pd.date_range("2022-01-01", periods=4, freq="H"),
        max_power=1000,
        min_power=0,
        efficiency=0.5,
        fixed_cost=10,
        fuel_type="lignite",
        emission_factor=0.5,
        fuel_price=10,
        co2_price=10,
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

    assert (
        power_plant_1.marginal_cost
        == pd.Series(
            [40.0, 52.0, 64.0, 66.0],
            index=pd.date_range("2022-01-01", periods=4, freq="H"),
        ).to_dict()
    )

    assert power_plant_2.marginal_cost == 40.0

    assert power_plant_3.marginal_cost is None


def test_reset_function(power_plant_1):
    power_plant_1.current_status = 0
    assert power_plant_1.current_status == 0

    power_plant_1.reset()
    assert power_plant_1.current_status == 1

    # check if total_power_output is reset
    assert power_plant_1.outputs["power"].equals(
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
    operational_window = power_plant_1.calculate_operational_window(
        product_tuple=product_tuple, product_type="energy"
    )

    assert operational_window["min_power"]["power"] == 200
    assert operational_window["min_power"]["marginal_cost"] == 40.0

    assert operational_window["max_power"]["power"] == 1000
    assert operational_window["max_power"]["marginal_cost"] == 40

    assert operational_window["current_power"]["power"] == 0


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
