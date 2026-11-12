# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Focused rolling-horizon tests for SectorCoupledSystem.

These tests deliberately avoid World.run() and market clearing.
They validate:

1. the physical rolling-horizon mechanics in isolation, and
2. the rolling lifecycle through the dedicated sector-coupled EOM/RD strategies.

"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from assume.common.forecaster import CustomUnitForecaster
from assume.strategies.naive_strategies import (
    SectorCoupledEnergyNaiveStrategy,
    SectorCoupledRedispatchNaiveStrategy,
)
from assume.units.sector_coupled_system import SectorCoupledSystem

RH_CONFIG = {
    "horizon_mode": "rolling_horizon",
    "look_ahead_horizon": "4h",
    "commit_horizon": "2h",
    "rolling_step": "2h",
}


@pytest.fixture
def sector_heat_rh():
    """6-hour heat-pump + TES case designed to expose limited foresight."""
    index = pd.date_range("2023-01-01 00:00", periods=6, freq="h")

    prices = pd.Series(
        [10.0, 10.0, 200.0, 200.0, 300.0, 10.0],
        index=index,
    )
    heat_demand = pd.Series(
        [15.0, 15.0, 20.0, 20.0, 40.0, 15.0],
        index=index,
    )

    forecaster = CustomUnitForecaster(
        index=index,
        electricity_price=prices,
        heat_demand=heat_demand,
    )

    components = {
        "heat_pump": {
            "max_power": 20.0,
            "min_power": 0.0,
            "cop": 3.0,
            "ramp_up": 20.0,
            "ramp_down": 20.0,
        },
        "thermal_storage": {
            "capacity": 90.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "initial_soc": 0.0,
            "max_power_charge": 45.0,
            "max_power_discharge": 45.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "storage_loss_rate": 0.0,
            "ramp_up": 45.0,
            "ramp_down": 45.0,
            "storage_type": "short-term",
        },
    }

    unit = SectorCoupledSystem(
        id="test_sector_heat_rh",
        unit_operator="test_operator",
        bidding_strategies={
            "EOM": SectorCoupledEnergyNaiveStrategy(),
            "redispatch": SectorCoupledRedispatchNaiveStrategy(),
        },
        forecaster=forecaster,
        components=components,
        system_type="heat",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=10,
        node="node0",
        market_representation="aggregate",
        enforce_terminal_soc=True,
        dsm_optimisation_config=RH_CONFIG,
    )

    unit.setup_model(presolve=True)
    return unit


def _product(start):
    """Create one hourly ASSUME product tuple."""
    return (
        start,
        start + pd.Timedelta(hours=1),
        None,
    )


def test_sector_coupled_system_stores_rolling_config(sector_heat_rh):
    unit = sector_heat_rh

    assert unit.horizon_mode == "rolling_horizon"
    assert unit._rh_look_ahead == "4h"
    assert unit._rh_commit == "2h"
    assert unit._rh_step == "2h"


def test_first_window_commits_two_hours(sector_heat_rh):
    unit = sector_heat_rh
    t0 = unit.index[0]

    assert unit._rh_optimized_until_step == 0

    triggered = unit._check_and_reoptimize_rolling_window(t0)

    assert triggered is True
    assert unit._rh_optimized_until_step == 2
    assert unit._rh_init_states is not None
    assert "thermal_storage" in unit._rh_init_states
    assert "soc" in unit._rh_init_states["thermal_storage"]


def test_thermal_storage_soc_is_carried_into_second_window(sector_heat_rh):
    """The SOC committed by Window 0 must become Window 1 initial_soc."""
    unit = sector_heat_rh

    original_prepare = unit._prepare_window_components
    observed = []

    def observed_prepare(
        window_start,
        window_end,
        full_len,
        init_states,
        remaining_demand=None,
    ):
        prepared = original_prepare(
            window_start,
            window_end,
            full_len,
            init_states,
            remaining_demand,
        )
        observed.append(
            {
                "window_start": window_start,
                "incoming_states": copy.deepcopy(init_states),
                "storage_initial_soc": prepared["thermal_storage"]["initial_soc"],
            }
        )
        return prepared

    unit._prepare_window_components = observed_prepare

    # Window 0: h0-h3, commits h0-h1.
    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True

    carried_soc = float(unit._rh_init_states["thermal_storage"]["soc"])

    # Window 1: h2-h5. Capture the initial_soc inserted into its component dict.
    assert unit._check_and_reoptimize_rolling_window(unit.index[2]) is True

    assert len(observed) >= 2
    second_window_initial_soc = float(observed[1]["storage_initial_soc"])

    assert second_window_initial_soc == pytest.approx(carried_soc, abs=1e-8)


def test_full_reference_and_limited_foresight_can_diverge(sector_heat_rh):
    """Demonstrate why a cached full-horizon reference can become stale.

    The full-horizon reference sees expensive h4 from the beginning.
    Window 0 sees only h0-h3. Therefore the full-horizon solve has an incentive
    to hold more energy by the end of h1 than the first rolling window.
    """
    unit = sector_heat_rh

    # Store the current full-horizon cost-optimal reference.
    unit.prepare_market_state(force=True)
    full_reference_soc_after_h1 = float(
        unit.da_component_state["thermal_storage"]["soc"].iloc[1]
    )

    # Now execute only the first rolling window: h0-h3, commit h0-h1.
    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True
    rolling_committed_soc_after_h1 = float(
        unit._rh_init_states["thermal_storage"]["soc"]
    )

    # In this deliberately constructed price/demand case the two optimisation
    # horizons should produce different storage preparation at the same boundary.
    assert full_reference_soc_after_h1 > (rolling_committed_soc_after_h1 + 1e-6), (
        "The diagnostic case did not create the intended divergence. "
        f"full_ref={full_reference_soc_after_h1:.6f}, "
        f"rolling={rolling_committed_soc_after_h1:.6f}"
    )


def test_rolling_window_updates_committed_market_reference(sector_heat_rh):
    """Committed DA reference must come from the same solved rolling window."""
    unit = sector_heat_rh

    # Seed an intentionally different full-horizon reference first. The rolling
    # hook must overwrite committed hours and invalidate uncommitted future hours.
    unit.prepare_market_state(force=True)

    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True

    # Parent DA power for committed h0-h1 equals the rolling schedule that ASSUME
    # also wrote into opt_power_requirement.
    for i in range(2):
        ts = unit.index[i]
        assert unit.da_power.at[ts] == pytest.approx(
            float(unit.opt_power_requirement.at[ts]),
            abs=1e-8,
        )

    # Storage state at the end of the committed block is the same state carried
    # into the next rolling window.
    da_soc_h1 = float(unit.da_component_state["thermal_storage"]["soc"].iloc[1])
    carried_soc_h1 = float(unit._rh_init_states["thermal_storage"]["soc"])
    assert da_soc_h1 == pytest.approx(carried_soc_h1, abs=1e-8)

    # h2 onward is not committed yet and must not retain stale full-horizon values.
    assert unit.da_power.iloc[2:].isna().all()
    assert unit.da_component_state["thermal_storage"]["soc"].iloc[2:].isna().all()

    # The complete look-ahead solution is retained separately for later RD use.
    ref = unit._rh_market_reference
    assert ref["window_start"] == 0
    assert ref["window_end"] == 4
    assert ref["commit_end"] == 2
    assert list(ref["committed_index"]) == list(unit.index[:2])
    assert list(ref["index"]) == list(unit.index[:4])


def test_second_window_extends_reference_without_changing_first_commit(sector_heat_rh):
    """Window 1 adds h2-h3 while preserving already committed h0-h1."""
    unit = sector_heat_rh

    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True

    first_commit_power = unit.da_power.iloc[:2].copy()
    first_commit_soc = unit.da_component_state["thermal_storage"]["soc"].iloc[:2].copy()

    assert unit._check_and_reoptimize_rolling_window(unit.index[2]) is True

    pd.testing.assert_series_equal(
        unit.da_power.iloc[:2],
        first_commit_power,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        unit.da_component_state["thermal_storage"]["soc"].iloc[:2],
        first_commit_soc,
        check_names=False,
    )

    # h2-h3 are now committed from Window 1.
    for i in (2, 3):
        ts = unit.index[i]
        assert unit.da_power.at[ts] == pytest.approx(
            float(unit.opt_power_requirement.at[ts]),
            abs=1e-8,
        )

    # h4-h5 are still provisional.
    assert unit.da_power.iloc[4:].isna().all()
    assert unit.da_component_state["thermal_storage"]["soc"].iloc[4:].isna().all()

    ref = unit._rh_market_reference
    assert ref["window_start"] == 2
    assert ref["window_end"] == 6
    assert ref["commit_end"] == 4


def test_rolling_redispatch_maps_global_time_to_local_window(sector_heat_rh):
    """RD at h3 in Window 1 must target local timestep 1, not global timestep 3."""
    unit = sector_heat_rh

    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True
    assert unit._check_and_reoptimize_rolling_window(unit.index[2]) is True

    reference, target_local = unit._rolling_reference_for_start(unit.index[3])

    assert reference["window_start"] == 2
    assert reference["window_end"] == 6
    assert reference["commit_end"] == 4
    assert target_local == 1


def test_rolling_redispatch_fixes_only_local_past(sector_heat_rh):
    """The local past is fixed to the rolling DA trajectory before RD re-optimisation."""
    unit = sector_heat_rh

    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True
    assert unit._check_and_reoptimize_rolling_window(unit.index[2]) is True

    reference, target_local = unit._rolling_reference_for_start(unit.index[3])
    instance = reference["instance"].clone()

    unit._fix_rolling_past_to_reference(
        instance,
        target_local,
        reference,
    )

    assert target_local == 1

    assert instance.total_power_input[0].fixed
    assert float(instance.total_power_input[0].value) == pytest.approx(
        float(reference["power"].iloc[0]),
        abs=1e-8,
    )

    storage = instance.dsm_blocks["thermal_storage"]
    assert storage.soc[0].fixed
    assert float(storage.soc[0].value) == pytest.approx(
        float(reference["component_state"]["thermal_storage"]["soc"].iloc[0]),
        abs=1e-8,
    )

    # The RD target itself remains free.
    assert not instance.total_power_input[target_local].fixed


def test_rolling_redispatch_bounds_contain_da_baseline(sector_heat_rh):
    """The committed rolling DA point must lie inside its own feasible RD envelope."""
    unit = sector_heat_rh

    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True
    assert unit._check_and_reoptimize_rolling_window(unit.index[2]) is True

    start = unit.index[3]
    base = float(unit.da_power.at[start])

    min_consumption, max_consumption = unit.get_redispatch_power_bounds(start)

    assert min_consumption <= base + 1e-8
    assert base <= max_consumption + 1e-8


def test_rolling_redispatch_rejects_uncommitted_lookahead(sector_heat_rh):
    """Provisional look-ahead hours must not be used as an RD baseline."""
    unit = sector_heat_rh

    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True

    # h2 belongs to Window 0's look-ahead but commit_end=2, so it is provisional.
    with pytest.raises(RuntimeError, match="not committed"):
        unit._rolling_reference_for_start(unit.index[2])


def test_redispatch_can_use_older_committed_window(sector_heat_rh):
    unit = sector_heat_rh

    # Window 0: [0:4], commit [0:2]
    assert unit._check_and_reoptimize_rolling_window(unit.index[0]) is True

    # Window 1: [2:6], commit [2:4]
    assert unit._check_and_reoptimize_rolling_window(unit.index[2]) is True

    # h1 belongs to Window 0, although Window 1 is now the latest one.
    reference, target_local = unit._rolling_reference_for_start(unit.index[1])

    assert reference["window_start"] == 0
    assert reference["commit_end"] == 2
    assert target_local == 1

    min_consumption, max_consumption = unit.get_redispatch_power_bounds(unit.index[1])

    base = float(unit.da_power.at[unit.index[1]])

    assert min_consumption <= base + 1e-8
    assert base <= max_consumption + 1e-8


# ---------------------------------------------------------------------------
# Strategy-level rolling-horizon integration
# ---------------------------------------------------------------------------


def test_sector_coupled_eom_strategy_triggers_rolling_window(sector_heat_rh):
    """
    The dedicated sector-coupled EOM strategy must trigger Window 0 itself.

    This validates the actual strategy -> rolling-horizon integration rather
    than calling _check_and_reoptimize_rolling_window() manually.
    """
    unit = sector_heat_rh
    strategy = SectorCoupledEnergyNaiveStrategy()

    assert unit._rh_optimized_until_step == 0
    assert unit._rh_market_references == {}

    bids = strategy.calculate_bids(
        unit=unit,
        market_config=None,
        product_tuples=[_product(unit.index[0])],
    )

    # Window 0 = [0:4), committed part = [0:2).
    assert unit._rh_optimized_until_step == 2
    assert set(unit._rh_market_references) == {0}

    reference = unit._rh_market_references[0]
    assert reference["window_start"] == 0
    assert reference["window_end"] == 4
    assert reference["commit_end"] == 2

    # The EOM bid must use the committed rolling DA reference.
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(
        -float(unit.da_power.at[unit.index[0]]),
        abs=1e-8,
    )


def test_sector_coupled_redispatch_does_not_duplicate_eom_window(
    sector_heat_rh,
):
    """
    EOM at h0 solves Window 0.

    Redispatch at the same market time also reaches the rolling trigger, but
    must not solve/replace Window 0 again. It must calculate RD around the DA
    reference already committed by EOM.
    """
    unit = sector_heat_rh
    eom_strategy = SectorCoupledEnergyNaiveStrategy()
    rd_strategy = SectorCoupledRedispatchNaiveStrategy()

    eom_strategy.calculate_bids(
        unit=unit,
        market_config=None,
        product_tuples=[_product(unit.index[0])],
    )

    optimized_until_before = unit._rh_optimized_until_step
    reference_keys_before = set(unit._rh_market_references)
    reference_instance_before = unit._rh_market_references[0]["instance"]

    rd_bids = rd_strategy.calculate_bids(
        unit=unit,
        market_config=None,
        product_tuples=[_product(unit.index[0])],
    )

    # The RD strategy must not create another rolling DA solution.
    assert unit._rh_optimized_until_step == optimized_until_before
    assert set(unit._rh_market_references) == reference_keys_before
    assert unit._rh_market_references[0]["instance"] is reference_instance_before

    assert len(rd_bids) == 1

    bid = rd_bids[0]
    base_market_volume = -float(unit.da_power.at[unit.index[0]])

    assert bid["volume"] == pytest.approx(
        base_market_volume,
        abs=1e-8,
    )
    assert bid["min_power"] <= bid["volume"] + 1e-8
    assert bid["volume"] <= bid["max_power"] + 1e-8


def test_sector_coupled_redispatch_strategy_uses_historical_committed_window(
    sector_heat_rh,
):
    """
    After Window 1 becomes active, RD for h1 must still be evaluated against
    historical Window 0 and must not alter the rolling-horizon state.
    """
    unit = sector_heat_rh
    eom_strategy = SectorCoupledEnergyNaiveStrategy()
    rd_strategy = SectorCoupledRedispatchNaiveStrategy()

    # EOM creates Window 0: [0:4), commit [0:2).
    eom_strategy.calculate_bids(
        unit=unit,
        market_config=None,
        product_tuples=[_product(unit.index[0])],
    )

    window_0_instance = unit._rh_market_references[0]["instance"]

    # EOM advances to Window 1: [2:6), commit [2:4).
    eom_strategy.calculate_bids(
        unit=unit,
        market_config=None,
        product_tuples=[_product(unit.index[2])],
    )

    assert unit._rh_optimized_until_step == 4
    assert set(unit._rh_market_references) == {0, 2}

    # Confirm h1 still maps to the historical Window 0 reference.
    reference, target_local = unit._rolling_reference_for_start(unit.index[1])

    assert reference["window_start"] == 0
    assert reference["commit_end"] == 2
    assert target_local == 1
    assert reference["instance"] is window_0_instance

    optimized_until_before = unit._rh_optimized_until_step
    reference_keys_before = set(unit._rh_market_references)

    rd_bids = rd_strategy.calculate_bids(
        unit=unit,
        market_config=None,
        product_tuples=[_product(unit.index[1])],
    )

    # RD for an older committed hour must not create/replace a DA window.
    assert unit._rh_optimized_until_step == optimized_until_before
    assert set(unit._rh_market_references) == reference_keys_before
    assert unit._rh_market_references[0]["instance"] is window_0_instance

    assert len(rd_bids) == 1

    bid = rd_bids[0]
    base_market_volume = -float(unit.da_power.at[unit.index[1]])

    assert bid["volume"] == pytest.approx(
        base_market_volume,
        abs=1e-8,
    )
    assert bid["min_power"] <= bid["volume"] + 1e-8
    assert bid["volume"] <= bid["max_power"] + 1e-8
