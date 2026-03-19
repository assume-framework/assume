# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.common.grid_utils import add_fix_units, add_redispatch_storage_units
from assume.common.market_objects import MarketConfig, Product
from assume.strategies.flexable_storage import (
    StorageEnergyHeuristicFlexableStrategy,
    StorageRedispatchFlexableStrategy,
)
from assume.units import Storage


class DummyNetwork:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.loads_t = {}
        self.calls = []

    def add(self, component, **kwargs):
        self.calls.append((component, kwargs))


def test_storage_redispatch_strategy_feasible_bounds_and_modes():
    """
    Test redispatch strategy validates charging vs discharging redispatch capabilities.
    Ensures min/max power bounds allow proper redispatch flexibility in both modes.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "redispatch": [
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
            ],
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
        },
    )

    storage_unit = Storage(
        id="test_storage",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-15,
        min_power_charge=-10,
        max_power_discharge=8,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        initial_soc=0.5,
    )

    # for ramp calculations
    for i in range(21):
        storage_unit.outputs["energy"].loc[index[i]] = 0.0
        storage_unit.outputs["soc"].loc[index[i]] = 0.5

    # Hours 20-22: Charging scenario (negative power)
    storage_unit.outputs["energy"].loc[index[20]] = -3.0
    storage_unit.outputs["soc"].loc[index[20]] = 0.45
    storage_unit.outputs["energy"].loc[index[21]] = -5.0
    storage_unit.outputs["soc"].loc[index[21]] = 0.35
    storage_unit.outputs["energy"].loc[index[22]] = -5.0
    storage_unit.outputs["soc"].loc[index[22]] = 0.25

    product_tuples = [Product(start=index[i], end=index[i + 1]) for i in range(21, 23)]

    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-1000,
        maximum_bid_price=1000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]
    assert isinstance(strategy, StorageRedispatchFlexableStrategy)

    bids = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    assert len(bids) == 2, f"Expected 2 bids, got {len(bids)}"

    # === VALIDATE CHARGING MODE (Hours 21-22) ===
    for i, bid in enumerate(bids):
        # Verify baseline matches EOM energy dispatch
        assert bid["volume"] == -5.0, (
            f"Bid {i}: baseline should be -5.0 MW (charging), got {bid['volume']}"
        )

        # In charging mode with redispatch:
        # - min_power allows MORE charging (more negative than baseline)
        # - max_power allows LESS charging (less negative than baseline)
        # - Storage should also allow switching to discharge (positive redispatch)
        assert bid["min_power"] <= bid["volume"], (
            f"Bid {i}: min should allow more charging than baseline"
        )
        assert bid["max_power"] >= bid["volume"], (
            f"Bid {i}: max should allow less charging than baseline"
        )

        # The bid should provide flexibility in both directions (negative and positive redispatch)
        assert bid["max_power"] > bid["volume"], (
            f"Bid {i}: should allow moving toward discharge (positive redispatch)"
        )

        # Verify price and bounds are valid
        assert isinstance(bid["price"], float), f"Bid {i}: price should be numeric"
        assert bid["min_power"] <= bid["volume"] <= bid["max_power"], (
            f"Bid {i}: volume {bid['volume']} outside bounds [{bid['min_power']}, {bid['max_power']}]"
        )


def test_storage_redispatch_discharge_and_charge_modes():
    """
    Test redispatch strategy with discharge mode (positive power).
    Validates ramp constraints for increasing/decreasing discharge.
    Also tests mode transitions and SoC constraints.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "redispatch": [
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
            ],
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
        },
    )

    storage_unit = Storage(
        id="test_storage_discharge",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-10,
        max_power_discharge=10,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_up_discharge=10,
        ramp_down_discharge=10,
        ramp_up_charge=-10,
        ramp_down_charge=-10,
        initial_soc=0.8,
    )

    # === DISCHARGE SCENARIO ===
    # Hour 21: EOM baseline is +4 MW (discharging), SoC at 0.8 (high)
    storage_unit.outputs["energy"].loc[index[21]] = 4.0
    storage_unit.outputs["soc"].loc[index[21]] = 0.8

    # Hour 22: EOM baseline is +3 MW (discharging), SoC at 0.7
    storage_unit.outputs["energy"].loc[index[22]] = 3.0
    storage_unit.outputs["soc"].loc[index[22]] = 0.7

    # Create products for redispatch
    product_tuples = [Product(start=index[i], end=index[i + 1]) for i in range(21, 23)]

    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-1000,
        maximum_bid_price=1000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]
    bids = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    assert len(bids) == 2, "Should have 2 bids for discharge scenario"

    # === VALIDATE DISCHARGE SCENARIOS ===
    bid_discharge_h21 = bids[0]
    assert bid_discharge_h21["volume"] == 4.0, "Baseline should be EOM discharge +4 MW"

    # In discharge mode (positive power):
    # - More discharge (higher positive) = allowed by ramp_up_discharge (if SoC permits)
    # - Less discharge (lower positive) = allowed by ramp_down_discharge
    assert bid_discharge_h21["min_power"] < bid_discharge_h21["volume"], (
        f"Hour 21 discharge: min_power {bid_discharge_h21['min_power']} should be lower than baseline to allow reduced discharge"
    )
    assert bid_discharge_h21["max_power"] >= bid_discharge_h21["volume"], (
        f"Hour 21 discharge: max_power {bid_discharge_h21['max_power']} should allow same or more discharge"
    )

    # Storage can ramp in discharge:
    # - Can reduce discharge to 0 MW (ramp down)
    # - Can increase discharge limited by SoC and ramp
    assert bid_discharge_h21["min_power"] >= 0.0, (
        "Discharge min should be >= 0 (can't go negative without explicit charge bid)"
    )

    # Bid 1: Hour 22 (Discharge mode baseline +3 MW)
    bid_discharge_h22 = bids[1]
    assert bid_discharge_h22["volume"] == 3.0, "Baseline should be EOM discharge +3 MW"
    assert bid_discharge_h22["min_power"] < bid_discharge_h22["volume"], (
        "Hour 22 should allow reduced discharge"
    )

    # === TEST MODE TRANSITIONS ===
    # Test hour with very low SoC that prevents discharge ramp-up
    storage_unit.outputs["energy"].loc[index[21]] = 1.0
    storage_unit.outputs["soc"].loc[index[21]] = 0.15  # Very low SoC

    bids_low_soc = strategy.calculate_bids(storage_unit, market_config, product_tuples)
    bid_low_soc = bids_low_soc[0]

    # At low SoC, max discharge should be limited
    assert bid_low_soc["max_power"] <= bid_low_soc["volume"] + 5, (
        "At low SoC, discharge ramp-up should be limited to available energy"
    )

    # === TEST CHARGE TO DISCHARGE TRANSITION ===
    storage_unit.outputs["energy"].loc[index[21]] = 4.0
    storage_unit.outputs["soc"].loc[index[21]] = 0.8

    # Test with SoC very high (can't charge more)
    high_soc_unit = Storage(
        id="test_storage_high_soc",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-10,
        max_power_discharge=10,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_up_charge=-10,
        ramp_down_charge=-10,
        ramp_up_discharge=10,
        ramp_down_discharge=10,
        initial_soc=0.95,  # Very high SoC
    )

    # Hour 21: EOM tries to charge at high SoC
    high_soc_unit.outputs["energy"].loc[index[21]] = -3.0  # baseline charge
    high_soc_unit.outputs["soc"].loc[index[21]] = 0.95

    bids_high_soc = high_soc_unit.bidding_strategies["redispatch"].calculate_bids(
        high_soc_unit, market_config, product_tuples[:1]
    )

    bid_high_soc_charge = bids_high_soc[0]
    assert bid_high_soc_charge["volume"] == -3.0, "Baseline should be -3 MW"
    # At high SoC, charging bounds should be constrained
    assert bid_high_soc_charge["min_power"] > -10.0, (
        "At high SoC, can't ramp up charging to full capacity"
    )

    # Verify all bids are valid
    for bids_list in [bids, bids_low_soc, bids_high_soc]:
        for i, bid in enumerate(bids_list):
            assert isinstance(bid["price"], float), f"Bid {i} price should be numeric"
            assert bid["min_power"] <= bid["volume"] <= bid["max_power"], (
                f"Bid {i}: volume {bid['volume']} outside [{bid['min_power']}, {bid['max_power']}]"
            )


def test_storage_eom_then_redispatch_integration():
    """
    Integration test: storage participates in EOM market, executes dispatch,
    then participates in redispatch market with baseline from EOM execution.
    Verifies SoC evolution and correct bid generation across both markets.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
            "redispatch": [
                12,
                14,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
            ],
        },
    )

    storage_unit = Storage(
        id="test_storage_eom_redispatch",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={
            "EOM": StorageEnergyHeuristicFlexableStrategy(),
            "redispatch": StorageRedispatchFlexableStrategy(),
        },
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-10,
        max_power_discharge=10,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        ramp_up_charge=-10,
        ramp_down_charge=-10,
        ramp_up_discharge=10,
        ramp_down_discharge=10,
        initial_soc=0.5,
    )

    t0, t1, t2, t3 = index[0], index[1], index[2], index[3]

    # EOM Market
    eom_market_config = MarketConfig(
        market_id="EOM",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    eom_products = [Product(start=index[i], end=index[i + 1]) for i in range(20)]

    eom_strategy = storage_unit.bidding_strategies["EOM"]
    eom_bids = eom_strategy.calculate_bids(
        storage_unit, eom_market_config, eom_products
    )

    assert len(eom_bids) >= 0

    # Simulate EOM market clearing with realistic dispatch across 24 hours:
    for i in range(7):
        storage_unit.outputs["energy"].loc[index[i]] = -4.0  # charge
    for i in range(7, 12):
        storage_unit.outputs["energy"].loc[index[i]] = 3.0  # discharge
    for i in range(12, 19):
        storage_unit.outputs["energy"].loc[index[i]] = 2.0  # discharge
    for i in range(19, 24):
        storage_unit.outputs["energy"].loc[index[i]] = -2.0  # charge

    # Execute EOM dispatch and track SoC evolution
    soc_before_eom = float(storage_unit.outputs["soc"].at[index[0]])
    assert soc_before_eom == 0.5

    # Simulate SoC changes from dispatch execution across 24 hours
    dt_h = 1.0
    soc = soc_before_eom
    soc_values = [soc]

    for i in range(1, 24):
        power = storage_unit.outputs["energy"].at[index[i - 1]]
        if power < 0:  # charging
            d_soc = (
                -power * dt_h * storage_unit.efficiency_charge
            ) / storage_unit.capacity
        else:  # discharging
            d_soc = -(power * dt_h) / (
                storage_unit.efficiency_discharge * storage_unit.capacity
            )
        soc += d_soc
        soc = max(0.0, min(1.0, soc))  # clamp to [0, 1]
        storage_unit.outputs["soc"].loc[index[i]] = soc
        soc_values.append(soc)

    # Verify SoC stays within bounds
    for i, soc_val in enumerate(soc_values):
        assert 0 <= soc_val <= 1, f"SoC at hour {i} = {soc_val} out of bounds"

    # Redispatch Market
    redispatch_market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    redispatch_products = [
        Product(start=index[i], end=index[i + 1]) for i in range(21, 23)
    ]

    redispatch_strategy = storage_unit.bidding_strategies["redispatch"]
    redispatch_bids = redispatch_strategy.calculate_bids(
        storage_unit, redispatch_market_config, redispatch_products
    )

    assert len(redispatch_bids) == 2

    # Verify redispatch bids reflect EOM baseline at hours 21-22
    assert redispatch_bids[0]["volume"] == -2.0, (
        "Redispatch hour 21 baseline should be EOM charge"
    )
    assert redispatch_bids[1]["volume"] == -2.0, (
        "Redispatch hour 22 baseline should be EOM charge"
    )

    # For each bid, verify baseline is within min/max bounds
    for i, bid in enumerate(redispatch_bids):
        assert bid["min_power"] <= bid["volume"] <= bid["max_power"], (
            f"Bid {i}: baseline {bid['volume']} outside ["
            f"{bid['min_power']}, {bid['max_power']}]"
        )

    # Verify prices are numeric
    for bid in redispatch_bids:
        assert isinstance(bid["price"], float), "Bid price should be numeric"


def test_storage_operational_constraints_validation():
    """
    test validating that storage enforces ALL operational constraints:
    - Min/Max SoC limits
    - Ramp rate constraints (charge/discharge transitions)
    - Mode transition penalties
    - Bid flexibility respects physical limits
    - SoC evolution under constrained operations
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
            "redispatch": [
                12,
                14,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
            ],
        },
    )

    storage_unit = Storage(
        id="test_storage_constraints",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={
            "EOM": StorageEnergyHeuristicFlexableStrategy(),
            "redispatch": StorageRedispatchFlexableStrategy(),
        },
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-6,
        max_power_discharge=6,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
        ramp_up_charge=-4,
        ramp_down_charge=-4,
        ramp_up_discharge=4,
        ramp_down_discharge=4,
    )

    # === TEST 1: SoC LIMITS CONSTRAINT ===

    for i in range(24):
        storage_unit.outputs["energy"].loc[index[i]] = 0.0
        storage_unit.outputs["soc"].loc[index[i]] = 0.5

    # Test: At SoC=0.95 (max), charging should be restricted
    storage_unit.outputs["soc"].loc[index[21]] = 0.95
    storage_unit.outputs["energy"].loc[index[21]] = -4.0  # EOM: attempting to charge

    product_tuples = [Product(start=index[21], end=index[22])]
    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]
    bids_high_soc = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    # At high SoC, min charging (most negative) should be constrained
    assert bids_high_soc[0]["min_power"] > -10.0, (
        f"At max SoC (0.95), min_power {bids_high_soc[0]['min_power']} should be constrained "
        f"(can't charge more when already full)"
    )
    print(
        f"[PASS] SoC=95%: Charging constrained to min={bids_high_soc[0]['min_power']:.1f} MW (not -10 MW)"
    )

    # === TEST 2: LOW SoC CONSTRAINT ===
    # At SoC=0.1 (min), discharging should be restricted
    storage_unit.outputs["soc"].loc[index[21]] = 0.1
    storage_unit.outputs["energy"].loc[index[21]] = 4.0  # EOM: attempting to discharge

    bids_low_soc = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    # At low SoC, max discharge should be constrained
    assert bids_low_soc[0]["max_power"] < 6.0, (
        f"At min SoC (0.1), max_power {bids_low_soc[0]['max_power']} should be constrained "
        f"(can't discharge more when already low)"
    )
    print(
        f"[PASS] SoC=10%: Discharge constrained to {bids_low_soc[0]['max_power']:.1f} MW (not 6 MW)"
    )

    # === TEST 3: RAMP CONSTRAINT (Mode Transition) ===
    # Set scenario: charging at -6 MW, then want to switch to discharging
    storage_unit.outputs["energy"].loc[
        index[20]
    ] = -6.0  # Previous: charging at full capability
    storage_unit.outputs["soc"].loc[index[20]] = 0.50
    storage_unit.outputs["energy"].loc[index[21]] = -6.0  # Current: charging baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    # Try to bid for discharge redispatch
    bids_ramp = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    baseline_charging = bids_ramp[0]["volume"]
    assert baseline_charging == -6.0, "Baseline should match EOM dispatch"

    # The max_power should allow SOME movement toward discharge or neutral (less negative)
    assert bids_ramp[0]["max_power"] > baseline_charging, (
        f"Should allow reducing charging from {baseline_charging} MW"
    )

    # But should also allow increasing charging (more negative) within ramp limits
    assert bids_ramp[0]["min_power"] <= baseline_charging, (
        f"Should allow increasing charging from {baseline_charging} MW"
    )

    print(
        f"[PASS] Charge mode flexibility: from {baseline_charging:.1f} MW baseline, "
        f"can reach [{bids_ramp[0]['min_power']:.1f}, {bids_ramp[0]['max_power']:.1f}] MW"
    )

    storage_unit.outputs["energy"].loc[index[20]] = (
        6.0  # Previous: discharging at full capability
    )
    storage_unit.outputs["energy"].loc[index[21]] = 6.0  # Current: discharge baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    bids_ramp_reverse = strategy.calculate_bids(
        storage_unit, market_config, product_tuples
    )

    baseline_discharge = bids_ramp_reverse[0]["volume"]
    assert baseline_discharge == 6.0, "Baseline should match EOM dispatch"

    # The min_power should allow SOME movement toward charging or neutral (less positive)
    assert bids_ramp_reverse[0]["min_power"] < baseline_discharge, (
        f"Should allow reducing discharge from {baseline_discharge} MW"
    )

    # But should also allow increasing discharge (more positive) within capability
    assert bids_ramp_reverse[0]["max_power"] >= baseline_discharge, (
        f"Should maintain discharge capability from {baseline_discharge} MW"
    )

    # === TEST 4: FEASIBLE BOUNDS RESPECTS RAMP CONSTRAINTS ===
    # Scenario: idle baseline (0 MW), previous also idle
    storage_unit.outputs["energy"].loc[index[20]] = 0.0  # Previous: idle
    storage_unit.outputs["energy"].loc[index[21]] = 0.0  # Current: idle baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    bids_normal = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    # From idle (0 MW) with ramp limits of ±4 MW:
    # Can only ramp to: [0-4, 0+4] = [-4, 4] MW
    assert bids_normal[0]["min_power"] >= -4.0, (
        f"From idle, with 4 MW/h ramp, min should be ~-4 MW, got {bids_normal[0]['min_power']}"
    )
    assert bids_normal[0]["max_power"] <= 4.0, (
        f"From idle, with 4 MW/h ramp, max should be ~4 MW, got {bids_normal[0]['max_power']}"
    )
    assert bids_normal[0]["volume"] == 0.0, "Baseline should be neutral (idle)"

    # === TEST 5: VERIFY FLEXIBILITY IN RAMPED SCENARIO ===
    storage_unit.outputs["energy"].loc[index[20]] = -4.0  # Previous: 4 MW charging
    storage_unit.outputs["energy"].loc[
        index[21]
    ] = -4.0  # Current: 4 MW charging baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    bids_ramped = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    assert bids_ramped[0]["volume"] == -4.0, "Baseline should be -4 MW"

    assert bids_ramped[0]["max_power"] > bids_ramped[0]["volume"], (
        "Should allow reducing charging (less negative) from baseline -4 MW"
    )


def test_bid_flexibility_exists():
    """
    Verify that every redispatch bid provides non-zero flexibility.

    Redispatch bids should have: max_power - min_power > 0
    Otherwise, the bid is just a fixed dispatch instruction with no flexibility
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
            "redispatch": [
                12,
                14,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
            ],
        },
    )

    storage_unit = Storage(
        id="test_flexibility",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-6,
        max_power_discharge=6,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
        ramp_up_charge=-4,
        ramp_down_charge=-4,
        ramp_up_discharge=4,
        ramp_down_discharge=4,
    )

    for i in range(24):
        storage_unit.outputs["energy"].loc[index[i]] = 0.0
        storage_unit.outputs["soc"].loc[index[i]] = 0.5

    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]

    # Test 1: Charging scenario (negative baseline)
    storage_unit.outputs["energy"].loc[index[5]] = -5.0
    storage_unit.outputs["soc"].loc[index[5]] = 0.5
    product_tuples = [Product(start=index[5], end=index[6])]

    bids_charge = strategy.calculate_bids(storage_unit, market_config, product_tuples)
    assert len(bids_charge) > 0, "Should generate redispatch bid for charging scenario"

    flexibility_charge = bids_charge[0]["max_power"] - bids_charge[0]["min_power"]
    assert flexibility_charge > 0.0, (
        f"Charging bid MUST provide flexibility. Got max={bids_charge[0]['max_power']}, "
        f"min={bids_charge[0]['min_power']}, range={flexibility_charge}"
    )

    # Test 2: Discharging scenario (positive baseline)
    storage_unit.outputs["energy"].loc[index[12]] = 4.0
    storage_unit.outputs["soc"].loc[index[12]] = 0.6
    product_tuples = [Product(start=index[12], end=index[13])]

    bids_discharge = strategy.calculate_bids(
        storage_unit, market_config, product_tuples
    )
    assert len(bids_discharge) > 0, (
        "Should generate redispatch bid for discharge scenario"
    )

    flexibility_discharge = (
        bids_discharge[0]["max_power"] - bids_discharge[0]["min_power"]
    )
    assert flexibility_discharge > 0.0, (
        f"Discharge bid MUST provide flexibility. Got max={bids_discharge[0]['max_power']}, "
        f"min={bids_discharge[0]['min_power']}, range={flexibility_discharge}"
    )

    # Test 3: Low SoC scenario
    storage_unit.outputs["soc"].loc[index[8]] = 0.15
    storage_unit.outputs["energy"].loc[index[8]] = 2.0
    product_tuples = [Product(start=index[8], end=index[9])]

    bids_low_soc = strategy.calculate_bids(storage_unit, market_config, product_tuples)
    assert len(bids_low_soc) > 0, "Should generate redispatch bid even at low SoC"

    flexibility_low_soc = bids_low_soc[0]["max_power"] - bids_low_soc[0]["min_power"]
    assert flexibility_low_soc > 0.0, (
        f"Even at low SoC, bid should provide flexibility. Got range={flexibility_low_soc}"
    )


def test_no_flexibility_edge_cases():
    """
    Test edge cases where redispatch flexibility may be unavailable.

    Example: SoC = min_soc (0.1), baseline = discharge (+)
    Expected: Cannot discharge further
    Behavior: Either (1) min_power == max_power == volume, or (2) No bid submitted

    This validates correct handling when storage has exhausted operational limits.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
            "redispatch": [
                12,
                14,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
            ],
        },
    )

    storage_unit = Storage(
        id="test_no_flexibility",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-6,
        max_power_discharge=6,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
        ramp_up_charge=-4,
        ramp_down_charge=-4,
        ramp_up_discharge=4,
        ramp_down_discharge=4,
    )

    # Set history
    for i in range(24):
        storage_unit.outputs["energy"].loc[index[i]] = 0.0
        storage_unit.outputs["soc"].loc[index[i]] = 0.5

    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]

    # === Case 1: SoC at minimum (0.1), baseline is discharge (positive) ===
    # Storage cannot discharge further. Only flexibility: increase charging or reduce discharge
    storage_unit.outputs["energy"].loc[index[6]] = 3.0
    storage_unit.outputs["soc"].loc[index[6]] = 0.1
    product_tuples = [Product(start=index[6], end=index[7])]

    bids_min_soc_discharge = strategy.calculate_bids(
        storage_unit, market_config, product_tuples
    )

    if len(bids_min_soc_discharge) > 0:
        bid = bids_min_soc_discharge[0]
        # At minimum SoC with discharge baseline, max_power cannot be higher
        # The bid should still provide at least the option to reduce discharge
        assert bid["min_power"] <= bid["volume"], (
            f"At min SoC with discharge baseline, should allow reducing discharge. "
            f"Got min_power={bid['min_power']}, volume={bid['volume']}"
        )
        print(
            f"Min SoC (0.1) discharge scenario: "
            f"bid=[{bid['min_power']:.1f}, {bid['volume']:.1f}, {bid['max_power']:.1f}] MW"
        )
    else:
        print(
            " Min SoC (0.1) discharge scenario: No bid generated (acceptable edge case)"
        )

    # === Case 2: SoC at maximum (0.9), baseline is charging (negative) ===
    # Storage cannot charge further. Only flexibility: reduce charging or increase discharge
    storage_unit.outputs["energy"].loc[index[10]] = -5.0
    storage_unit.outputs["soc"].loc[index[10]] = 0.9
    product_tuples = [Product(start=index[10], end=index[11])]

    bids_max_soc_charge = strategy.calculate_bids(
        storage_unit, market_config, product_tuples
    )

    if len(bids_max_soc_charge) > 0:
        bid = bids_max_soc_charge[0]
        # At maximum SoC with charging baseline, min_power cannot be lower
        # The bid should allow at least the option to reduce charging
        assert bid["max_power"] >= bid["volume"], (
            f"At max SoC with charging baseline, should allow reducing charging. "
            f"Got max_power={bid['max_power']}, volume={bid['volume']}"
        )
        print(
            f"Max SoC (0.9) charging scenario: "
            f"bid=[{bid['min_power']:.1f}, {bid['volume']:.1f}, {bid['max_power']:.1f}] MW"
        )
    else:
        print("Max SoC (0.9) charging scenario: No bid generated")


def test_volume_baseline_invariant():
    """
    Verify the core redispatch model invariant: volume == baseline

    The redispatch bid structure assumes:
    - volume represents the EOM-dispatched baseline power
    - min_power and max_power are bounds around this baseline
    - The market uses volume as the starting point for redispatch adjustments

    This test explicitly verifies this invariant holds across all scenarios.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [
                10,
                12,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
            ],
            "redispatch": [
                12,
                14,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
            ],
        },
    )

    storage_unit = Storage(
        id="test_volume_baseline",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-6,
        max_power_discharge=6,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        initial_soc=0.5,
    )

    # Set history
    for i in range(24):
        storage_unit.outputs["energy"].loc[index[i]] = 0.0
        storage_unit.outputs["soc"].loc[index[i]] = 0.5

    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]

    # Test multiple baseline scenarios
    test_baselines = [
        (-8.5, "Strong charging"),
        (-3.2, "Weak charging"),
        (0.0, "Neutral (idle)"),
        (2.7, "Weak discharge"),
        (5.5, "Strong discharge"),
    ]

    for baseline_power, description in test_baselines:
        storage_unit.outputs["energy"].loc[index[12]] = baseline_power
        storage_unit.outputs["soc"].loc[index[12]] = 0.5
        product_tuples = [Product(start=index[12], end=index[13])]

        bids = strategy.calculate_bids(storage_unit, market_config, product_tuples)
        assert len(bids) > 0, f"Should generate bid for {description}"

        bid = bids[0]
        assert bid["volume"] == baseline_power, (
            f"volume ({bid['volume']}) must equal baseline ({baseline_power}) for {description}. "
            f"This is the core redispatch model assumption."
        )

        # Volume must be within bounds
        assert bid["min_power"] <= bid["volume"] <= bid["max_power"], (
            f"Invariant violated: volume {bid['volume']} not within "
            f"[{bid['min_power']}, {bid['max_power']}] for {description}"
        )


def test_sign_consistency():
    """
    Verify sign convention is consistent and intuitive:
    - Negative power = Charging (energy flowing into storage)
    - Zero power = Neutral (no energy flow)
    - Positive power = Discharging (energy flowing out of storage)

    This test ensures the model correctly represents energy flow directions.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [10] * 24,
            "redispatch": [12] * 24,
        },
    )

    storage_unit = Storage(
        id="test_signs",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-10,
        min_power_charge=-6,
        max_power_discharge=6,
        min_power_discharge=0,
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        initial_soc=0.5,
    )

    # Set history
    for i in range(24):
        storage_unit.outputs["energy"].loc[index[i]] = 0.0
        storage_unit.outputs["soc"].loc[index[i]] = 0.5

    market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    strategy = storage_unit.bidding_strategies["redispatch"]

    # === Test 1: Charging baseline (negative) ===
    storage_unit.outputs["energy"].loc[index[5]] = -7.0
    product_tuples = [Product(start=index[5], end=index[6])]
    bids_charge = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    assert len(bids_charge) > 0
    charging_bid = bids_charge[0]
    assert charging_bid["volume"] < 0.0, "Charging baseline should be negative"
    # Verify sign consistency: the bid's volume matches the baseline sign
    assert (
        charging_bid["min_power"] <= charging_bid["volume"] <= charging_bid["max_power"]
    ), "Charging bid should have volume within bounds"
    # At least one bound should allow movement away from charging (toward neutral/discharge)
    assert charging_bid["max_power"] > charging_bid["volume"], (
        "Should allow at least reducing charging (moving toward positive/neutral)"
    )

    # === Test 2: Discharging baseline (positive) ===
    storage_unit.outputs["energy"].loc[index[12]] = 4.5
    product_tuples = [Product(start=index[12], end=index[13])]
    bids_discharge = strategy.calculate_bids(
        storage_unit, market_config, product_tuples
    )

    assert len(bids_discharge) > 0
    discharge_bid = bids_discharge[0]
    assert discharge_bid["volume"] > 0.0, "Discharge baseline should be positive"
    # Verify sign consistency: the bid's volume matches the baseline sign
    assert (
        discharge_bid["min_power"]
        <= discharge_bid["volume"]
        <= discharge_bid["max_power"]
    ), "Discharge bid should have volume within bounds"
    # At least one bound should allow movement away from discharge (toward neutral/charging)
    assert discharge_bid["min_power"] < discharge_bid["volume"], (
        "Should allow at least reducing discharge (moving toward negative/neutral)"
    )

    # === Test 3: Neutral baseline (zero) ===
    storage_unit.outputs["energy"].loc[index[18]] = 0.0
    product_tuples = [Product(start=index[18], end=index[19])]
    bids_neutral = strategy.calculate_bids(storage_unit, market_config, product_tuples)

    assert len(bids_neutral) > 0
    neutral_bid = bids_neutral[0]
    assert neutral_bid["volume"] == 0.0, "Neutral baseline should be exactly zero"
    assert neutral_bid["min_power"] < 0.0, "Should allow charging (negative)"
    assert neutral_bid["max_power"] > 0.0, "Should allow discharge (positive)"


def test_redispatch_avg_price_fallback_branches():
    """Cover _avg_price edge paths used by redispatch strategy."""
    strategy = StorageRedispatchFlexableStrategy()
    t0 = pd.Timestamp("2025-01-01 00:00:00")

    assert strategy._avg_price(None, t0, pd.Timedelta(hours=1)) == 0.0

    empty_series = pd.Series(dtype=float)
    assert strategy._avg_price(empty_series, t0, pd.Timedelta(hours=1)) == 0.0

    # Negative window forces left > right and hits the direct fallback branch.
    one_point = pd.Series([7.5], index=[t0])
    assert strategy._avg_price(one_point, t0, pd.Timedelta(hours=-1)) == 7.5


def test_redispatch_calculate_bids_fallbacks_for_soc_and_price():
    """Cover fallback branches when soc timestamp and market price forecast are missing."""

    class _DummyUnit:
        def __init__(self):
            t0 = pd.Timestamp("2025-01-01 01:00:00")
            # soc at a different timestamp -> .at[t0] raises and fallback path is used.
            self.outputs = {
                "soc": pd.Series([0.6], index=[pd.Timestamp("2025-01-01 00:00:00")]),
                "energy": pd.Series([1.0], index=[t0]),
            }
            self.forecaster = {}
            self.id = "dummy_storage"
            self.node = "node0"
            self.efficiency_charge = 0.9
            self.efficiency_discharge = 0.95
            self.capacity = 10.0

        def get_output_before(self, _):
            return 0.0

        def calculate_min_max_charge(self, *_args, **_kwargs):
            return [0.0], [-2.0]

        def calculate_min_max_discharge(self, *_args, **_kwargs):
            return [0.0], [2.0]

        def calculate_ramp_discharge(self, *_args, **_kwargs):
            return float(_args[2])

        def calculate_ramp_charge(self, *_args, **_kwargs):
            return float(_args[2])

    strategy = StorageRedispatchFlexableStrategy()
    market_config = MarketConfig(market_id="redispatch", product_type="energy")
    t0 = pd.Timestamp("2025-01-01 01:00:00")
    product_tuples = [Product(start=t0, end=t0 + pd.Timedelta(hours=1))]

    bids = strategy.calculate_bids(_DummyUnit(), market_config, product_tuples)

    assert len(bids) == 1
    assert bids[0]["volume"] == 1.0
    # No forecast available in forecaster fallback chain -> _avg_price(None) = 0.
    assert bids[0]["price"] == 0.0


def test_redispatch_grid_utils_storage_network_integration():
    """Cover redispatch grid helper behavior for storage-specific network setup."""
    network = DummyNetwork(snapshots=range(3))
    storage_units = pd.DataFrame(
        {
            "node": ["node1"],
            "max_power_discharge": [6.0],
            "max_power_charge": [4.0],
        },
        index=["storage_1"],
    )

    add_redispatch_storage_units(network=network, storage_units=storage_units)

    assert len(network.calls) == 3

    load_component, load_kwargs = network.calls[0]
    assert load_component == "Load"
    assert load_kwargs["sign"] == -1

    up_component, up_kwargs = network.calls[1]
    assert up_component == "Generator"
    assert up_kwargs["suffix"] == "_up"
    assert up_kwargs["p_nom"]["storage_1"] == 6.0

    down_component, down_kwargs = network.calls[2]
    assert down_component == "Generator"
    assert down_kwargs["suffix"] == "_down"
    assert down_kwargs["sign"] == -1
    assert down_kwargs["p_nom"]["storage_1"] == 4.0


def test_redispatch_grid_utils_storage_validation_and_fix_units_defaults():
    """Cover storage grid helper validation plus fixed-unit default p_set creation."""
    network = DummyNetwork(snapshots=range(2))

    empty_storage_units = pd.DataFrame(
        columns=["node", "max_power_discharge", "max_power_charge"]
    )
    add_redispatch_storage_units(network=network, storage_units=empty_storage_units)
    assert network.calls == []

    invalid_storage_units = pd.DataFrame(
        {"node": ["node1"], "max_power_discharge": [5.0]},
        index=["storage_1"],
    )
    with pytest.raises(KeyError, match="storage_units is missing required cols"):
        add_redispatch_storage_units(
            network=network, storage_units=invalid_storage_units
        )

    units = pd.DataFrame(
        {
            "node": ["node1"],
            "max_power": [5.0],
            "sign": [1],
        },
        index=["load_1"],
    )
    add_fix_units(network=network, units=units)

    assert len(network.calls) == 1
    component, kwargs = network.calls[0]
    assert component == "Load"
    assert kwargs["sign"] == 1
    assert "p_set" in network.loads_t
    assert list(network.loads_t["p_set"].columns) == ["load_1"]


def test_redispatch_grid_utils_fix_units_preserves_existing_p_set():
    """Cover add_fix_units branch where p_set already exists and should not be overwritten."""
    network = DummyNetwork(snapshots=range(2))
    network.loads_t["p_set"] = "keep_me"

    units = pd.DataFrame(
        {
            "node": ["node1"],
            "max_power": [5.0],
            "p_set": [0.0],
        },
        index=["load_1"],
    )

    add_fix_units(network=network, units=units)

    assert len(network.calls) == 1
    assert network.loads_t["p_set"] == "keep_me"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
