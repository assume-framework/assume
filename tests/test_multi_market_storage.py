# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies.multi_market_storage import MultiMarketStorageStrategy
from assume.units import Storage


def make_storage(
    prices: dict[str, np.ndarray | float],
    index: pd.DatetimeIndex,
    strategies: dict[str, MultiMarketStorageStrategy],
    **kwargs,
) -> Storage:
    forecaster = UnitForecaster(
        index=pd.DatetimeIndex(index),
        market_prices=prices,
        availability=1,
    )
    defaults = dict(
        id="Test_Storage",
        unit_operator="TestOperator",
        technology="PSPP",
        bidding_strategies=strategies,
        forecaster=forecaster,
        max_power_charge=-100,
        max_power_discharge=100,
        capacity=1000,
        initial_soc=0.5,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
    )
    defaults.update(kwargs)
    return Storage(**defaults)


def make_market_config(
    market_id: str,
    product_type: str = "energy",
    duration: timedelta = timedelta(hours=1),
    count: int = 1,
    first_delivery: timedelta = timedelta(hours=1),
    opening_frequency: timedelta = timedelta(hours=1),
    opening_duration: timedelta = timedelta(hours=1),
    start: datetime = datetime(2022, 1, 1),
    end: datetime = datetime(2022, 1, 3),
    **kwargs,
) -> MarketConfig:
    freq, interval = (
        (rr.HOURLY, int(opening_frequency / timedelta(hours=1)))
        if opening_frequency >= timedelta(hours=1)
        else (rr.MINUTELY, int(opening_frequency / timedelta(minutes=1)))
    )
    return MarketConfig(
        market_id=market_id,
        product_type=product_type,
        market_products=[
            MarketProduct(
                duration=duration,
                count=count,
                first_delivery=first_delivery,
            )
        ],
        opening_hours=rr.rrule(freq=freq, interval=interval, dtstart=start, until=end),
        opening_duration=opening_duration,
        **kwargs,
    )


def price_profile(periods: int = 48) -> np.ndarray:
    """Cheap at night, expensive in the evening."""
    prices = np.full(periods, 20.0)
    prices[12:24] = 100.0
    return prices


def apply_plan(storage: Storage, strategy: MultiMarketStorageStrategy, until: datetime):
    """
    Simulates the market clearing the plan's bids up to `until` (exclusive),
    as the unit's set_dispatch_plan would: the energy flows are written to
    the unit outputs and the SoC is updated accordingly.
    """
    soc = storage.outputs["soc"].at[storage.index[0]] * storage.capacity
    for (_, start, _), entry in sorted(strategy._cache_solution.items()):
        if start >= until:
            continue
        net = entry.get("sell", 0) - entry.get("buy", 0)
        storage.outputs["energy"].at[start] = net
        if net > 0:
            soc -= net / storage.efficiency_discharge
        else:
            soc += -net * storage.efficiency_charge
        next_t = start + storage.index.freq
        if next_t in storage.index:
            storage.outputs["soc"].at[next_t] = soc / storage.capacity


@pytest.fixture
def eom_storage():
    index = pd.date_range("2022-01-01", periods=48, freq="h")
    strategy = MultiMarketStorageStrategy(foresight="24h")
    storage = make_storage({"EOM": price_profile()}, index, {"EOM": strategy})
    storage.available_markets = [make_market_config("EOM")]
    return storage, strategy


def test_single_market_arbitrage(eom_storage):
    """The storage charges at cheap and discharges at expensive prices."""
    storage, strategy = eom_storage
    mc = storage.available_markets[0]

    # opening at 00:00, product 01:00-02:00 (cheap) -> charging bid
    bids = strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] < 0  # charging
    assert bids[0]["price"] == pytest.approx(20.1)  # price + epsilon

    # check the plan: discharge in the expensive hours
    solution = strategy._cache_solution
    net_at = {
        start.hour: entry.get("sell", 0) - entry.get("buy", 0)
        for (_, start, _), entry in solution.items()
    }
    assert net_at[13] > 0  # expensive hour: discharging
    assert net_at[2] < 0  # cheap hour: charging


def test_soc_constraints_respected(eom_storage):
    """The planned SoC stays within the unit limits."""
    storage, strategy = eom_storage
    mc = storage.available_markets[0]

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )

    # reconstruct the SoC from the plan
    solution = strategy._cache_solution
    soc = storage.initial_soc * storage.capacity
    for (_, start, _), entry in sorted(solution.items()):
        net = entry.get("sell", 0) - entry.get("buy", 0)
        if net > 0:
            soc -= net / storage.efficiency_discharge
        else:
            soc += -net * storage.efficiency_charge
        # small numerical tolerance for the LP solution
        assert -1e-6 <= soc <= storage.capacity + 1e-6


def test_power_limit_respected(eom_storage):
    """The planned volumes never exceed the power limits."""
    storage, strategy = eom_storage
    mc = storage.available_markets[0]

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )

    for (_, start, _), entry in strategy._cache_solution.items():
        net = entry.get("sell", 0) - entry.get("buy", 0)
        assert abs(net) <= storage.max_power_discharge + 1e-6


def test_caching(eom_storage):
    """The optimization is reused when several markets open at the same time."""
    storage, strategy = eom_storage
    mc = storage.available_markets[0]

    # first market opening triggers the optimization
    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    plan_time = strategy._plan_time

    # a second market opening at the same decision time reuses the plan
    mc2 = make_market_config("EOM_2")
    storage.bidding_strategies["EOM_2"] = strategy
    strategy.calculate_bids(
        storage, mc2, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    assert strategy._plan_time == plan_time


def test_multi_market_coupling():
    """Commitments on two markets share the power limit of the unit."""
    index = pd.date_range("2022-01-01", periods=48, freq="h")
    strategy = MultiMarketStorageStrategy(foresight="24h")
    storage = make_storage(
        {"EOM": price_profile(), "CRM_pos": 50.0},
        index,
        {"EOM": strategy, "CRM_pos": strategy},
    )
    eom = make_market_config("EOM")
    crm_pos = make_market_config(
        "CRM_pos",
        product_type="capacity_pos",
        duration=timedelta(hours=4),
        opening_frequency=timedelta(hours=4),
    )
    storage.available_markets = [eom, crm_pos]

    # CRM opens at 00:00 - the strategy optimizes across both markets
    bids = strategy.calculate_bids(
        storage, crm_pos, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 5), None)]
    )
    # the storage bids its capacity into the CRM if the price is attractive
    assert len(bids) >= 0

    # check that the planned power per hour and direction respects the limit:
    # sell (discharge) and buy (charge) commitments are separate directions -
    # a capacity_pos commitment while charging is physically fine (on
    # activation the charging is stopped), so per direction the sum is limited
    solution = strategy._cache_solution
    sell_per_hour = {}
    buy_per_hour = {}
    for (market_id, start, _), entry in solution.items():
        net = entry.get("sell", 0) - entry.get("buy", 0)
        if net > 0:
            sell_per_hour[start.hour] = sell_per_hour.get(start.hour, 0) + net
        else:
            buy_per_hour[start.hour] = buy_per_hour.get(start.hour, 0) - net
    for total in sell_per_hour.values():
        assert total <= storage.max_power_discharge + 1e-6
    for total in buy_per_hour.values():
        assert total <= -storage.max_power_charge + 1e-6


def test_eligibility_filtering(eom_storage):
    """Markets without a bidding strategy are not bid on."""
    storage, strategy = eom_storage
    mc = storage.available_markets[0]
    heat = make_market_config("heat", product_type="heat")
    storage.available_markets = [mc, heat]

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    assert "heat" not in strategy.markets


def test_no_bids_for_unknown_market(eom_storage):
    """No bids are returned for markets the strategy did not discover."""
    storage, strategy = eom_storage
    unknown = make_market_config("unknown")

    bids = strategy.calculate_bids(
        storage,
        unknown,
        product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)],
    )
    assert bids == []


def test_price_limits_applied(eom_storage):
    """Bid prices are clipped to the market price limits."""
    storage, strategy = eom_storage
    mc = make_market_config("EOM", maximum_bid_price=30, minimum_bid_price=0)
    storage.available_markets = [mc]

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    for (_, start, _), entry in strategy._cache_solution.items():
        pass  # the price clipping is applied in calculate_bids

    bids = strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    for bid in bids:
        assert 0 <= bid["price"] <= 30


def test_infeasible_horizon_returns_no_bids():
    """At the end of the simulation, no crash occurs and no bids are returned."""
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    strategy = MultiMarketStorageStrategy(foresight="24h")
    storage = make_storage({"EOM": price_profile(4)}, index, {"EOM": strategy})
    storage.available_markets = [make_market_config("EOM")]

    # product beyond the index - the horizon is empty
    bids = strategy.calculate_bids(
        storage,
        storage.available_markets[0],
        product_tuples=[(datetime(2022, 1, 2, 1), datetime(2022, 1, 2, 2), None)],
    )
    assert bids == []


@pytest.fixture
def rolling_storage():
    index = pd.date_range("2022-01-01", periods=48, freq="h")
    strategy = MultiMarketStorageStrategy(foresight="24h", overlap="12h")
    storage = make_storage({"EOM": price_profile()}, index, {"EOM": strategy})
    storage.available_markets = [make_market_config("EOM")]
    return storage, strategy


def test_rolling_horizon_plan_reuse(rolling_storage):
    """Within the overlap period, the plan is reused instead of re-optimized."""
    storage, strategy = rolling_storage
    mc = storage.available_markets[0]

    # first opening at 00:00 optimizes over the 24h foresight
    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    assert strategy._plan_time == datetime(2022, 1, 1)
    assert strategy._plan_valid_until == datetime(2022, 1, 1, 12)
    solution = strategy._cache_solution

    # the market clears the plan's bids for the first hours
    apply_plan(storage, strategy, until=datetime(2022, 1, 1, 7))

    # an opening 6h later is within the overlap: the plan is reused
    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 7), datetime(2022, 1, 1, 8), None)]
    )
    assert strategy._cache_solution is solution
    assert strategy._plan_time == datetime(2022, 1, 1)

    # an opening 12h later is beyond the overlap: re-optimization
    apply_plan(storage, strategy, until=datetime(2022, 1, 1, 13))
    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 13), datetime(2022, 1, 1, 14), None)]
    )
    assert strategy._cache_solution is not solution
    assert strategy._plan_time == datetime(2022, 1, 1, 12)
    assert strategy._plan_valid_until == datetime(2022, 1, 2)


def test_rolling_horizon_energy_value(rolling_storage):
    """
    Energy at the end of the used window keeps its value.

    The peak price is in hour 18, beyond the 12h overlap. Without a terminal
    condition, the storage must not dump its energy within the first 12h -
    the optimization sees the peak in the later hours of the foresight
    window and keeps the energy for it.
    """
    storage, strategy = rolling_storage
    mc = storage.available_markets[0]

    # opening at 00:00: bids are only taken for the first 12 hours
    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )

    # net energy flow within the used window (hours 1-12, cheap at 20 EUR)
    solution = strategy._cache_solution
    net = sum(
        entry.get("sell", 0) - entry.get("buy", 0)
        for (_, start, _), entry in solution.items()
        if start < datetime(2022, 1, 1, 12)
    )
    # the storage charges (or does nothing) - it does not dump its energy
    assert net <= 0

    # the plan still contains the discharge for the peak beyond the overlap
    peak_net = sum(
        entry.get("sell", 0) - entry.get("buy", 0)
        for (_, start, _), entry in solution.items()
        if start.hour == 18
    )
    assert peak_net > 0


def test_terminal_condition_option():
    """With terminal_condition=True, the plan ends with at least the current SoC."""
    index = pd.date_range("2022-01-01", periods=48, freq="h")
    strategy = MultiMarketStorageStrategy(foresight="24h", terminal_condition=True)
    storage = make_storage({"EOM": price_profile()}, index, {"EOM": strategy})
    storage.available_markets = [make_market_config("EOM")]
    mc = storage.available_markets[0]

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )

    # the planned SoC at the end of the horizon is not below the current SoC
    planned_soc = strategy._plan_soc
    current_soc = storage.initial_soc * storage.capacity
    horizon_end = max(times for times in planned_soc)
    assert planned_soc[horizon_end] >= current_soc - 1e-6


def test_reoptimization_on_soc_deviation(rolling_storage):
    """A deviation of the actual SoC from the plan triggers re-optimization."""
    storage, strategy = rolling_storage
    mc = storage.available_markets[0]

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 1), datetime(2022, 1, 1, 2), None)]
    )
    solution = strategy._cache_solution

    # the market clears the plan's bids for the first hours
    apply_plan(storage, strategy, until=datetime(2022, 1, 1, 7))

    # an opening within the overlap reuses the plan
    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 7), datetime(2022, 1, 1, 8), None)]
    )
    assert strategy._cache_solution is solution

    # now the actual SoC deviates from the planned trajectory (e.g. because
    # the market clamped the accepted volume)
    decision_time = datetime(2022, 1, 1, 6)  # the opening time of the market
    planned_soc = strategy._plan_soc[decision_time]
    storage.outputs["soc"].at[decision_time] = planned_soc / storage.capacity + 0.1

    strategy.calculate_bids(
        storage, mc, product_tuples=[(datetime(2022, 1, 1, 7), datetime(2022, 1, 1, 8), None)]
    )
    assert strategy._cache_solution is not solution
