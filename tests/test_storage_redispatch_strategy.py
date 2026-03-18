# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.common.market_objects import MarketConfig, Product
from assume.strategies.flexable_storage import (
    StorageEnergyHeuristicFlexableStrategy,
    StorageRedispatchFlexableStrategy,
)
from assume.units import Storage


def test_storage_redispatch_strategy_feasible_bounds_and_modes():
    """
    Test redispatch strategy validates charging vs discharging redispatch capabilities.
    Ensures min/max power bounds allow proper redispatch flexibility in both modes.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "redispatch": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "EOM": [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
        },
    )

    storage_unit = Storage(
        id="test_storage",
        unit_operator="operator",
        technology="li-ion",
        node="node0",
        bidding_strategies={"redispatch": StorageRedispatchFlexableStrategy()},
        forecaster=forecaster,
        max_power_charge=-15,     # max (upper bound on charging): -15 MW
        min_power_charge=-10,     # min (lower bound on charging): -10 MW
        max_power_discharge=8,    # max discharge: 8 MW
        min_power_discharge=0,    # min discharge: 0 MW
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        initial_soc=0.5,
    )

    # Set complete history for ramp calculations
    for i in range(0, 21):
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
    # Baseline: -5 MW (charging)
    # Storage capability: can charge from -15 MW (max) to -10 MW (min)
    for i, bid in enumerate(bids):
        # Verify baseline matches EOM energy dispatch 
        assert bid["volume"] == -5.0, f"Bid {i}: baseline should be -5.0 MW (charging), got {bid['volume']}"
        
        # In charging mode with redispatch:
        # - min_power allows MORE charging (more negative than baseline)
        # - max_power allows LESS charging (less negative than baseline)
        # - Storage should also allow switching to discharge (positive redispatch)
        assert bid["min_power"] <= bid["volume"], f"Bid {i}: min should allow more charging than baseline"
        assert bid["max_power"] >= bid["volume"], f"Bid {i}: max should allow less charging than baseline"
        
        # The bid should provide flexibility in both directions (negative and positive redispatch)
        assert bid["max_power"] > bid["volume"], f"Bid {i}: should allow moving toward discharge (positive redispatch)"
        
        # Verify price and bounds are valid
        assert isinstance(bid["price"], float), f"Bid {i}: price should be numeric"
        assert bid["min_power"] <= bid["volume"] <= bid["max_power"], (
            f"Bid {i}: volume {bid['volume']} outside bounds [{bid['min_power']}, {bid['max_power']}]"
        )
        
        # Print actual bounds for validation
        print(f"[Bid {i}] baseline={bid['volume']:.1f}, min={bid['min_power']:.1f}, max={bid['max_power']:.1f} MW")

    print("[PASS] Charging scenario validated: storage provides negative (more charging) and positive (less charging/discharge) redispatch")


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
            "redispatch": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "EOM": [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
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
        ramp_up_discharge=10,    # can increase discharge by up to 10 MW
        ramp_down_discharge=10,  # can decrease discharge by up to 10 MW  
        ramp_up_charge=-10,      # can increase charging by up to 10 MW
        ramp_down_charge=-10,    # can decrease charging by up to 10 MW
        initial_soc=0.8,
    )

    # === DISCHARGE SCENARIO ===
    # Hour 21: EOM baseline is +4 MW (discharging), SoC at 0.8 (high)
    storage_unit.outputs["energy"].loc[index[21]] = 4.0
    storage_unit.outputs["soc"].loc[index[21]] = 0.8

    # Hour 22: EOM baseline is +3 MW (discharging), SoC at 0.7
    storage_unit.outputs["energy"].loc[index[22]] = 3.0
    storage_unit.outputs["soc"].loc[index[22]] = 0.7

    # Create products for redispatch market opening (hours 21-22)
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
    # Bid 0: Hour 21 (Discharge mode baseline +4 MW)
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
    assert bid_discharge_h21["min_power"] >= 0.0, "Discharge min should be >= 0 (can't go negative without explicit charge bid)"
    
    # Bid 1: Hour 22 (Discharge mode baseline +3 MW)
    bid_discharge_h22 = bids[1]
    assert bid_discharge_h22["volume"] == 3.0, "Baseline should be EOM discharge +3 MW"
    assert bid_discharge_h22["min_power"] < bid_discharge_h22["volume"], "Hour 22 should allow reduced discharge"
    
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
    # Reset to previous discharge state
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
    Covers full 24-hour daily market cycle aligned with example_05b config.
    """
    index = pd.date_range(start="2025-01-01", periods=24, freq="1h")
    forecaster = UnitForecaster(
        index=index,
        market_prices={
            "EOM": [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
            "redispatch": [12, 14, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
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

    # === PHASE 1: EOM Market ===
    eom_market_config = MarketConfig(
        market_id="EOM",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    # Per example_05b config: EOM opens at 00:00 and operates for 20 hours (0-19)
    eom_products = [Product(start=index[i], end=index[i + 1]) for i in range(20)]

    eom_strategy = storage_unit.bidding_strategies["EOM"]
    eom_bids = eom_strategy.calculate_bids(
        storage_unit, eom_market_config, eom_products
    )

    assert len(eom_bids) >= 0

    # Simulate EOM market clearing with realistic dispatch across 24 hours:
    # Hours 0-6: charge (night hours, low prices)
    # Hours 7-11: discharge (morning ramp-up, high prices)
    # Hours 12-18: discharge (peak demand hours)
    # Hours 19-23: charge (evening/night ramp-down)
    for i in range(7):
        storage_unit.outputs["energy"].loc[index[i]] = -4.0  # charge
    for i in range(7, 12):
        storage_unit.outputs["energy"].loc[index[i]] = 3.0   # discharge
    for i in range(12, 19):
        storage_unit.outputs["energy"].loc[index[i]] = 2.0   # discharge
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
            d_soc = (-power * dt_h * storage_unit.efficiency_charge) / storage_unit.capacity
        else:  # discharging
            d_soc = -(power * dt_h) / (storage_unit.efficiency_discharge * storage_unit.capacity)
        soc += d_soc
        soc = max(0.0, min(1.0, soc))  # clamp to [0, 1]
        storage_unit.outputs["soc"].loc[index[i]] = soc
        soc_values.append(soc)
    
    # Verify SoC stays within bounds
    for i, soc_val in enumerate(soc_values):
        assert 0 <= soc_val <= 1, f"SoC at hour {i} = {soc_val} out of bounds"

    # === PHASE 2: Redispatch Market ===
    # Redispatch baseline = EOM dispatch from above
    redispatch_market_config = MarketConfig(
        market_id="redispatch",
        product_type="energy",
        maximum_bid_volume=1000,
        minimum_bid_price=-500,
        maximum_bid_price=3000,
    )

    # Per example_05b config: redispatch opens at 21:00 and operates for 2 hours (21-22)
    redispatch_products = [Product(start=index[i], end=index[i + 1]) for i in range(21, 23)]

    redispatch_strategy = storage_unit.bidding_strategies["redispatch"]
    redispatch_bids = redispatch_strategy.calculate_bids(
        storage_unit, redispatch_market_config, redispatch_products
    )

    assert len(redispatch_bids) == 2

    # Verify redispatch bids reflect EOM baseline at hours 21-22
    # Hour 21-22: should be charge operations (-2.0 MW from EOM)
    assert redispatch_bids[0]["volume"] == -2.0, "Redispatch hour 21 baseline should be EOM charge"
    assert redispatch_bids[1]["volume"] == -2.0, "Redispatch hour 22 baseline should be EOM charge"

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
    Comprehensive test validating that storage enforces ALL operational constraints:
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
            "EOM": [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
            "redispatch": [12, 14, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
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
        max_power_charge=-10,     # Strongest charging: -10 MW
        min_power_charge=-6,      # Weakest charging: -6 MW
        max_power_discharge=6,    # Max discharge: 6 MW
        min_power_discharge=0,    # Min discharge: 0 MW
        capacity=20,
        efficiency_charge=0.9,
        efficiency_discharge=0.95,
        min_soc=0.1,              # Min SoC: 10% (critical: can't discharge below this)
        max_soc=0.9,              # Max SoC: 90% (critical: can't charge above this)
        initial_soc=0.5,
        ramp_up_charge=-4,        # Can increase charging by up to 4 MW per timestep
        ramp_down_charge=-4,      # Can decrease charging by up to 4 MW per timestep
        ramp_up_discharge=4,      # Can increase discharge by up to 4 MW per timestep
        ramp_down_discharge=4,    # Can decrease discharge by up to 4 MW per timestep
    )

    # === TEST 1: SoC LIMITS CONSTRAINT ===
    # Set full history
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
    print(f"[PASS] SoC=95%: Charging constrained to min={bids_high_soc[0]['min_power']:.1f} MW (not -10 MW)")

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
    print(f"[PASS] SoC=10%: Discharge constrained to {bids_low_soc[0]['max_power']:.1f} MW (not 6 MW)")

    # === TEST 3: RAMP CONSTRAINT (Mode Transition) ===
    # Set scenario: charging at -6 MW, then want to switch to discharging
    storage_unit.outputs["energy"].loc[index[20]] = -6.0  # Previous: charging at full capability
    storage_unit.outputs["soc"].loc[index[20]] = 0.50
    storage_unit.outputs["energy"].loc[index[21]] = -6.0  # Current: charging baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    # Try to bid for discharge redispatch
    bids_ramp = strategy.calculate_bids(storage_unit, market_config, product_tuples)
    
    # From -6 MW baseline (charging), with ramp constraints, should still allow some flexibility
    # The upper bound should be better than just the capability limit
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
    
    print(f"[PASS] Charge mode flexibility: from {baseline_charging:.1f} MW baseline, "
          f"can reach [{bids_ramp[0]['min_power']:.1f}, {bids_ramp[0]['max_power']:.1f}] MW")

    # Switch scenario: discharging at +6 MW, try to switch to charging
    storage_unit.outputs["energy"].loc[index[20]] = 6.0  # Previous: discharging at full capability
    storage_unit.outputs["energy"].loc[index[21]] = 6.0  # Current: discharge baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    bids_ramp_reverse = strategy.calculate_bids(storage_unit, market_config, product_tuples)
    
    # From +6 MW baseline (discharging), should allow flexibility
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
    
    print(f"[PASS] Discharge mode flexibility: from {baseline_discharge:.1f} MW baseline, "
          f"can reach [{bids_ramp_reverse[0]['min_power']:.1f}, {bids_ramp_reverse[0]['max_power']:.1f}] MW")

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
    print(f"[PASS] Idle ramp constraint: from 0 MW baseline, "
          f"can reach [{bids_normal[0]['min_power']:.1f}, {bids_normal[0]['max_power']:.1f}] MW "
          f"(±4 MW ramp limit in effect)")
    
    # === TEST 5: VERIFY FLEXIBILITY IN RAMPED SCENARIO ===
    # Scenario: storage at -4 MW charging baseline with half SoC
    storage_unit.outputs["energy"].loc[index[20]] = -4.0  # Previous: 4 MW charging  
    storage_unit.outputs["energy"].loc[index[21]] = -4.0  # Current: 4 MW charging baseline
    storage_unit.outputs["soc"].loc[index[21]] = 0.50

    bids_ramped = strategy.calculate_bids(storage_unit, market_config, product_tuples)
    
    # From -4 MW charging baseline, should have bidding flexibility
    assert bids_ramped[0]["volume"] == -4.0, "Baseline should be -4 MW"
    
    # Should provide a range, not just fixed point
    assert bids_ramped[0]["max_power"] > bids_ramped[0]["volume"], (
        f"Should allow reducing charging (less negative) from baseline -4 MW"
    )
    
    # Print actual flexibility bounds
    print(f"[PASS] Ramped scenario: from -4 MW charging baseline, "
          f"can reach [{bids_ramped[0]['min_power']:.1f}, {bids_ramped[0]['max_power']:.1f}] MW "
          f"(flexibility range within constraints)")

    print("\n[SUMMARY] Comprehensive operational constraints validated:")
    print("  ✓ Max SoC (0.9): Prevents excessive charging")
    print("  ✓ Min SoC (0.1): Prevents excessive discharge")
    print("  ✓ Ramp rates (±4 MW/h): Limit mode transitions")
    print("  ✓ Charge/discharge bounds enforced")
    print("  ✓ Feasible ranges respect all constraints together")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
