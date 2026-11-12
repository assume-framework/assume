# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import math
from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecaster import PowerplantForecaster
from assume.strategies import (
    CapacityHeuristicBalancingNegStrategy,
    CapacityHeuristicBalancingPosStrategy,
    EnergyHeuristicFlexableStrategy,
)
from assume.strategies.flexable import (
    calculate_EOM_price_if_off,
    calculate_EOM_price_if_on,
)
from assume.units import PowerPlant

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def power_plant() -> PowerPlant:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2023-07-01", periods=48, freq="h")
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
    )
    return PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        index=forecaster.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
    )


def test_flexable_eom(mock_market_config, power_plant):
    end = datetime(2023, 7, 1, 1)
    strategy = EnergyHeuristicFlexableStrategy()
    mc = mock_market_config
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 200
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 800

    # start-up situation with ramping restriction
    power_plant.ramp_up = 400
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 200
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 200

    # CHP with ramping restrictions, still a start-up (hour 0 is assumed off),
    # so the heat discount of calculate_EOM_price_if_on does not apply
    power_plant.outputs["heat"][start] = 300
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 2
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 300
    assert bids[1]["price"] == 40
    assert bids[1]["volume"] == 100


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_flexable_start_up_markup_independent_of_resolution(freq):
    """The start-up markup (unit off) and the restart price reduction (unit on) are in
    €/MWh, so they spread the start-up cost over the minimum times in hours. They must
    not depend on the resolution, although the minimum times are stored in time steps."""
    index = pd.date_range("2023-07-01", periods=48 * 4, freq=freq)
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        market_prices={"EOM": 100},
    )
    pp = PowerPlant(
        id="test_pp",
        unit_operator="test_operator",
        technology="hard coal",
        index=forecaster.index,
        max_power=1000,
        min_power=200,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
        cold_start_cost=10,  # €/MW -> 10,000 € per start, every start is cold
        min_operating_time=4,  # hours
        min_down_time=2,  # hours
    )
    steps_per_hour = {"h": 1, "15min": 4}[freq]
    assert pp.min_operating_time == 4 * steps_per_hour
    assert pp.min_down_time == 2 * steps_per_hour

    marginal_cost = 40
    bid_quantity = 200

    # off: 10,000 € / 4 h / 200 MW = 12.5 €/MWh on top of the marginal cost
    price_off = calculate_EOM_price_if_off(
        pp, marginal_cost, bid_quantity, op_time=-pp.get_max_lookback_op_time()
    )
    assert math.isclose(price_off, marginal_cost + 12.5)

    # on: 10,000 € / 2 h / 200 MW = 25 €/MWh below the marginal cost
    price_on = calculate_EOM_price_if_on(
        pp,
        "EOM",
        index[0],
        marginal_cost,
        bid_quantity,
        foresight=pd.Timedelta(hours=12),
    )
    assert math.isclose(price_on, marginal_cost - 25)


def test_flexable_pos_reserve(mock_market_config, power_plant):
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = CapacityHeuristicBalancingPosStrategy()
    mc = mock_market_config
    mc.product_type = "energy_pos"
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 40
    assert bids[0]["volume"] == 1000

    mc.product_type = "capacity_pos"
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 50
    assert bids[0]["volume"] == 1000

    # increase the marginal cost to ensure specific_revenue < 0
    power_plant.marginal_cost[start] = 60
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 10
    assert bids[0]["volume"] == 1000


def test_flexable_neg_reserve(mock_market_config, power_plant):
    end = datetime(2023, 7, 1, 4, 0, 0)
    strategy = CapacityHeuristicBalancingNegStrategy()
    mc = mock_market_config
    # Calculations for negative energy
    mc.product_type = "energy_neg"
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert bids == []

    # change previous power to max_power/2 to enable ramp down
    power_plant.outputs["energy"][start] = power_plant.max_power / 2
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == -40
    assert bids[0]["volume"] == 300

    # Calculations for negative capacity
    mc.product_type = "capacity_neg"
    product_tuples = [(start, end, None)]
    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert bids == []

    # change previous power to max_power/2 to enable ramp down
    power_plant.outputs["energy"][start] = power_plant.max_power / 2
    product_tuples = [
        (start + pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), None)
    ]

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert bids[0]["price"] == 0
    assert bids[0]["volume"] == 300

    # increase the marginal cost to ensure specific_revenue < 0
    power_plant.marginal_cost[start + pd.Timedelta(hours=1)] = 60

    bids = strategy.calculate_bids(power_plant, mc, product_tuples=product_tuples)
    assert len(bids) == 1
    assert math.isclose(bids[0]["price"], 50 / 3)
    assert bids[0]["volume"] == 300


if __name__ == "__main__":
    # run pytest and enable prints
    import pytest

    pytest.main(["-s", __file__])
