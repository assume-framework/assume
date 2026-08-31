# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecast_algorithms import (
    effective_grid_fee,
    price_plus_grid_fee,
)
from assume.common.forecaster import DemandForecaster
from assume.strategies.grid_tariff import (
    ANNOUNCED_FLAG,
    GridFeeAnnouncementStrategy,
    GridFeeDsoStrategy,
)
from tests.conftest import MockMinMaxUnit

START = datetime(2023, 7, 1)
END = datetime(2023, 7, 2)


class GridFeeMarketConfig:
    market_id = "GridTariff"
    product_type = "grid_fee"
    maximum_bid_price = 100000.0
    minimum_bid_price = 0.0
    maximum_bid_volume = 100000.0
    additional_fields = []


@pytest.fixture
def market_config():
    return GridFeeMarketConfig()


@pytest.fixture
def unit():
    index = pd.date_range(start=START, end=END, freq="1h")
    return MockMinMaxUnit(forecaster=DemandForecaster(index, demand=-150))


def hourly_products(n=3, first_hour=0):
    return [
        (
            START + timedelta(hours=first_hour + i),
            START + timedelta(hours=first_hour + i + 1),
            None,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# DSO supply curve
# --------------------------------------------------------------------------


def test_dso_bids_a_single_unlimited_block_without_headroom(unit, market_config):
    strategy = GridFeeDsoStrategy(grid_fee_base=30.0)
    bids = strategy.calculate_bids(unit, market_config, hourly_products(2))

    assert len(bids) == 2
    for bid in bids:
        assert bid["price"] == 30.0
        assert bid["volume"] == market_config.maximum_bid_volume


def test_dso_headroom_produces_a_stepwise_curve(unit, market_config):
    strategy = GridFeeDsoStrategy(
        grid_fee_base=30.0, grid_fee_scarcity=400.0, grid_fee_headroom=0.5
    )
    bids = strategy.calculate_bids(unit, market_config, hourly_products(1))

    assert [(b["volume"], b["price"]) for b in bids] == [
        (0.5, 30.0),
        (market_config.maximum_bid_volume - 0.5, 430.0),
    ]


def test_dso_peak_hours_add_to_every_block(unit, market_config):
    strategy = GridFeeDsoStrategy(
        grid_fee_base=30.0,
        grid_fee_scarcity=400.0,
        grid_fee_headroom=0.5,
        grid_fee_peak_hours="1",
        grid_fee_peak_adder=250.0,
    )
    bids = strategy.calculate_bids(unit, market_config, hourly_products(2))

    off_peak = [b for b in bids if b["start_time"].hour == 0]
    peak = [b for b in bids if b["start_time"].hour == 1]
    assert sorted(b["price"] for b in off_peak) == [30.0, 430.0]
    assert sorted(b["price"] for b in peak) == [280.0, 680.0]


def test_dso_peak_hours_accepts_a_list(unit, market_config):
    strategy = GridFeeDsoStrategy(grid_fee_peak_hours=[1, 2])
    assert strategy.grid_fee_peak_hours == {1, 2}


# --------------------------------------------------------------------------
# Aggregator announcement
# --------------------------------------------------------------------------


def test_announcement_falls_back_to_a_token_volume(unit, market_config):
    strategy = GridFeeAnnouncementStrategy(token_volume=1e-3)
    bids = strategy.calculate_bids(unit, market_config, hourly_products(3))

    # every hour must carry a non-zero bid, otherwise the clearing drops it and
    # the unit silently keeps the previous hour's fee
    assert len(bids) == 3
    for bid in bids:
        assert bid["volume"] == -1e-3
        assert bid["price"] == market_config.maximum_bid_price


def test_announcement_uses_the_provisional_plan(unit, market_config):
    unit.planned_power_requirement = FastSeries(index=unit.index, value=0.0)
    unit.planned_power_requirement.at[START + timedelta(hours=1)] = 0.4

    bids = GridFeeAnnouncementStrategy().calculate_bids(
        unit, market_config, hourly_products(3)
    )

    assert bids[0]["volume"] == pytest.approx(-1e-3)
    assert bids[1]["volume"] == pytest.approx(-0.4)
    assert bids[2]["volume"] == pytest.approx(-1e-3)


def test_announcement_prefers_the_plan_over_the_committed_schedule(unit, market_config):
    unit.opt_power_requirement = FastSeries(index=unit.index, value=0.1)
    unit.planned_power_requirement = FastSeries(index=unit.index, value=0.7)

    bids = GridFeeAnnouncementStrategy().calculate_bids(
        unit, market_config, hourly_products(1)
    )
    assert bids[0]["volume"] == pytest.approx(-0.7)


def test_announcement_marks_the_hours_it_received_a_fee_for(unit, market_config):
    strategy = GridFeeAnnouncementStrategy()
    orderbook = [
        {
            "start_time": START + timedelta(hours=1),
            "end_time": START + timedelta(hours=2),
            "accepted_price": 30.0,
            "accepted_volume": -0.4,
        }
    ]
    strategy.calculate_reward(unit, market_config, orderbook)

    flag = unit.outputs[ANNOUNCED_FLAG]
    assert flag.at[START] == 0.0
    assert flag.at[START + timedelta(hours=1)] == 1.0
    assert flag.at[START + timedelta(hours=2)] == 0.0


# --------------------------------------------------------------------------
# Folding the fee into the price forecast
# --------------------------------------------------------------------------


def test_effective_grid_fee_forward_fills_the_last_announcement(unit):
    unit.outputs[ANNOUNCED_FLAG].iat[1] = 1.0
    unit.outputs[ANNOUNCED_FLAG].iat[2] = 1.0
    unit.outputs["grid_fee_accepted_price"].iat[1] = 30.0
    unit.outputs["grid_fee_accepted_price"].iat[2] = 80.0

    fee = effective_grid_fee(unit.outputs, default_fee=5.0)

    # before the first announcement: the default
    assert fee[0] == 5.0
    # announced hours: as published
    assert fee[1] == 30.0
    assert fee[2] == 80.0
    # beyond the announcement: persistence of the last published value
    assert np.all(fee[3:] == 80.0)


def test_effective_grid_fee_without_any_announcement(unit):
    fee = effective_grid_fee(unit.outputs, default_fee=7.0)
    assert np.all(fee == 7.0)


def test_price_plus_grid_fee_is_a_noop_without_a_grid_fee_market(unit):
    before = unit.forecaster.price["EOM"].data.copy()
    result = price_plus_grid_fee(unit.forecaster.price, None, unit=unit)
    assert np.allclose(result["EOM"].data, before)


def test_price_plus_grid_fee_is_idempotent(unit):
    forecaster = unit.forecaster
    base = forecaster.price["EOM"].data.copy()

    unit.outputs[ANNOUNCED_FLAG] += 1.0
    unit.outputs["grid_fee_accepted_price"] += 30.0

    forecaster.price = price_plus_grid_fee(forecaster.price, None, unit=unit)
    once = forecaster.price["EOM"].data.copy()
    forecaster.price = price_plus_grid_fee(forecaster.price, None, unit=unit)
    twice = forecaster.price["EOM"].data

    assert np.allclose(once, base + 30.0)
    assert np.allclose(twice, once), "the fee must not accumulate across updates"


def test_price_plus_grid_fee_follows_a_changing_fee(unit):
    forecaster = unit.forecaster
    base = forecaster.price["EOM"].data.copy()

    unit.outputs[ANNOUNCED_FLAG] += 1.0
    unit.outputs["grid_fee_accepted_price"] += 30.0
    forecaster.price = price_plus_grid_fee(forecaster.price, None, unit=unit)

    unit.outputs["grid_fee_accepted_price"].iat[2] = 280.0
    forecaster.price = price_plus_grid_fee(forecaster.price, None, unit=unit)

    assert forecaster.price["EOM"].data[2] == pytest.approx(base[2] + 280.0)
    assert forecaster.price["EOM"].data[1] == pytest.approx(base[1] + 30.0)
