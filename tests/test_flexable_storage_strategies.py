import math
from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies import (
    flexableEOMStorage,
    flexableNegCRMStorage,
    flexablePosCRMStorage,
)
from assume.units import Storage

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def storage() -> Storage:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=4, freq="H")
    # constant price of 50
    ff = NaiveForecast(index, availability=1, price_forecast=50)
    return Storage(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        max_power_charge=-100,
        max_power_discharge=100,
        max_volume=1000,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        index=index,
        ramp_down_charge=-50,
        ramp_down_discharge=50,
        ramp_up_charge=-60,
        ramp_up_discharge=60,
        variable_cost_charge=3,
        variable_cost_discharge=4,
        fixed_cost=1,
    )


def test_flexable_eom_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="H")
    end = datetime(2023, 7, 1, 1, 0, 0)
    strategy = flexableEOMStorage()
    mc = mock_market_config
    product_tuples = [(start, end, None)]

    # constant price of 50
    storage.forecaster = NaiveForecast(index, availability=1, price_forecast=50)
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    # no change in price forecast -> no bidding
    assert bids == []

    # increase the current price forecast -> discharging
    storage.forecaster = NaiveForecast(
        index, availability=1, price_forecast=[60, 50, 50, 50]
    )
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 52.5
    assert bids[0]["volume"] == 60

    # decrease current price forecast -> charging
    storage.forecaster = NaiveForecast(
        index, availability=1, price_forecast=[40, 50, 50, 50]
    )
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 47.5
    assert bids[0]["volume"] == -60


def test_flexable_pos_crm_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="H")
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = flexablePosCRMStorage()
    mc = mock_market_config
    mc.product_type = "energy_pos"
    product_tuples = [(start, end, None)]

    # constant price of 50
    specific_revenue = (50 - (4 / 0.95 + 1)) * 360 / (0.36 * 1000)

    storage.forecaster = NaiveForecast(index, availability=1, price_forecast=50)
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], specific_revenue / (0.5 * 1000))
    assert bids[0]["volume"] == 60

    # assert capacity_pos
    mc.product_type = "capacity_pos"
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], specific_revenue)
    assert bids[0]["volume"] == 60

    # specific revenue < 0
    storage.forecaster = NaiveForecast(
        index, availability=1, price_forecast=[5, 5, 5, 5]
    )
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
    assert bids == []


def test_flexable_neg_crm_storage(mock_market_config, storage):
    index = pd.date_range("2023-07-01", periods=4, freq="H")
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = flexableNegCRMStorage()
    mc = mock_market_config
    # Calculations for negative energy
    mc.product_type = "energy_neg"
    product_tuples = [(start, end, None)]

    # constant price of 50
    storage.forecaster = NaiveForecast(index, availability=1, price_forecast=50)
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], 0)
    assert bids[0]["volume"] == -60

    # assert capacity_pos
    mc.product_type = "capacity_neg"
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], 0)
    assert bids[0]["volume"] == -60

    # was charging before
    storage.outputs["energy"][start] = 60
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]
    bids = strategy.calculate_bids(storage, mc, product_tuples=product_tuples)
    assert bids == []


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
