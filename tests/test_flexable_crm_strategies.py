# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for the power-plant CRM strategies (PowerPlantCrmPosStrategy /
PowerPlantCrmNegStrategy). Each class handles its direction's capacity *and*
energy product, switching on product_type.

Conventions (so the arithmetic is transparent):
    fuel_price = 10, efficiency = 0.5, co2 = 10, emission_factor = 0.5,
    additional_cost = 10
    => marginal_cost = 10/0.5 + 10*0.5/0.5 + 10 = 20 + 10 + 10 = 40 EUR/MWh
    eom_foresight = 12h (default) -> 12 hourly steps in the capacity-price sum.
"""

from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecaster import PowerplantForecaster
from assume.strategies.flexable import EnergyHeuristicFlexableStrategy
from assume.strategies.flexable_crm import (
    PowerPlantCrmNegStrategy,
    PowerPlantCrmPosStrategy,
)
from assume.units import PowerPlant

MC = 40.0  # marginal_cost EUR/MWh given the fixture below
MAX_P = 1000.0
MIN_P = 200.0
FORESIGHT_STEPS = 12  # default eom_foresight "12h" on an hourly index

BLOCK_START = datetime(2023, 7, 1, 0)
BLOCK_END = datetime(2023, 7, 1, 4)  # 4h block
SECOND_BLOCK_END = datetime(2023, 7, 1, 8)

HOUR_START = BLOCK_START
HOUR_END = datetime(2023, 7, 1, 1)
NEXT_HOUR_END = datetime(2023, 7, 1, 2)


def _make_powerplant(
    da_price=60.0, crm_en_pos=50.0, crm_en_neg=30.0, include_crm_signals=True
):
    """PowerPlant fixture with controllable cross-market price forecasts."""
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    market_prices = {"EOM": da_price}
    if include_crm_signals:
        market_prices["CRM_energy_pos"] = crm_en_pos
        market_prices["CRM_energy_neg"] = crm_en_neg
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        market_prices=market_prices,
    )
    return PowerPlant(
        id="test_pp",
        unit_operator="test_op",
        technology="hard coal",
        index=forecaster.index,
        max_power=MAX_P,
        min_power=MIN_P,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
    )


class _CfgStub:
    maximum_bid_price = 9999.0
    minimum_bid_price = 0.0
    additional_fields = []


def _cap_cfg(direction="pos"):
    cfg = _CfgStub()
    cfg.product_type = f"capacity_{direction}"
    cfg.market_id = f"CRM_capacity_{direction}"
    cfg.minimum_bid_price = 0.0
    return cfg


def _en_cfg(direction="pos"):
    cfg = _CfgStub()
    cfg.product_type = f"energy_{direction}"
    cfg.market_id = f"CRM_energy_{direction}"
    cfg.minimum_bid_price = -9999.0
    return cfg


# ---------------------------------------------------------------------------
# Positive capacity
# ---------------------------------------------------------------------------


def test_pos_cap_basic_block_net_opportunity_cost():
    """
    da=60, MC=40, crm_en_pos=50.
    per-hour net opp = max(0, 60-40) - max(0, 50-40) = 20 - 10 = 10
    price = 10 * 12 (foresight steps) = 120 EUR/MW
    volume = cross-direction guard = MAX_P - MIN_P = 800.
    """
    pp = _make_powerplant(da_price=60.0, crm_en_pos=50.0)
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P - MIN_P)
    assert bids[0]["price"] == pytest.approx(10.0 * FORESIGHT_STEPS)
    assert bids[0]["start_time"] == BLOCK_START
    assert bids[0]["end_time"] == BLOCK_END


def test_pos_cap_energy_beats_eom_clips_to_floor():
    """
    crm_en_pos (100) > EOM (60): net opp per hour = 20 - 60 = -40 -> negative.
    Price clips to the market floor (0); the unit reserves eagerly.
    """
    pp = _make_powerplant(da_price=60.0, crm_en_pos=100.0)
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P - MIN_P)
    assert bids[0]["price"] == pytest.approx(0.0)


def test_pos_cap_respects_existing_cap_neg_commitment():
    """cap_neg=200 already committed -> pos volume <= 800 - 200 = 600."""
    pp = _make_powerplant(da_price=60.0)
    for h in range(4):
        pp.outputs["capacity_neg"][BLOCK_START + pd.Timedelta(hours=h)] = 200.0
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P - MIN_P - 200.0)


def test_pos_cap_volume_takes_min_across_block():
    """Drop availability in the 3rd block hour to 0.5 -> volume = 500."""
    pp = _make_powerplant(da_price=60.0)
    pp.forecaster.availability[BLOCK_START + pd.Timedelta(hours=2)] = 0.5
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(500.0)
    # MC is constant -> price unchanged at 120
    assert bids[0]["price"] == pytest.approx(10.0 * FORESIGHT_STEPS)


def test_pos_cap_one_bid_per_block():
    pp = _make_powerplant(da_price=60.0)
    strategy = PowerPlantCrmPosStrategy()
    products = [(BLOCK_START, BLOCK_END, None), (BLOCK_END, SECOND_BLOCK_END, None)]
    bids = strategy.calculate_bids(pp, _cap_cfg("pos"), products)
    assert len(bids) == 2
    assert bids[0]["end_time"] == BLOCK_END
    assert bids[1]["start_time"] == BLOCK_END
    assert bids[0]["price"] == pytest.approx(bids[1]["price"])


def test_pos_cap_price_clipped_to_market_max():
    pp = _make_powerplant(da_price=20_000.0)
    strategy = PowerPlantCrmPosStrategy()
    cfg = _cap_cfg("pos")
    cfg.maximum_bid_price = 500.0
    bids = strategy.calculate_bids(pp, cfg, [(BLOCK_START, BLOCK_END, None)])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Negative capacity
# ---------------------------------------------------------------------------


def test_neg_cap_no_bid_without_dispatch():
    """base_power=0 -> no downward room -> no neg capacity bid."""
    pp = _make_powerplant(da_price=60.0)
    strategy = PowerPlantCrmNegStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert bids == []


def test_neg_cap_uses_committed_dispatch():
    """
    base=800 committed -> ramp-down room = 800 - min_power(200) = 600.
    da=60 > MC=40 so the must-run opp cost is 0; with crm_en_neg=30 the energy
    margin makes net opp negative -> price floors at 0.
    """
    pp = _make_powerplant(da_price=60.0, crm_en_neg=30.0)
    for h in range(4):
        pp.outputs["energy"][BLOCK_START + pd.Timedelta(hours=h)] = 800.0
    strategy = PowerPlantCrmNegStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(600.0)
    assert bids[0]["price"] == pytest.approx(0.0)


def test_neg_cap_must_run_cost_when_da_below_mc():
    """
    da=30 < MC=40 -> must-run opp cost = max(0, 40-30) = 10/h.
    crm_en_neg=40 -> energy margin max(0, 40-40)=0.
    price = 10 * 12 = 120; volume = 600 (base=800 committed).
    """
    pp = _make_powerplant(da_price=30.0, crm_en_neg=40.0)
    for h in range(4):
        pp.outputs["energy"][BLOCK_START + pd.Timedelta(hours=h)] = 800.0
    strategy = PowerPlantCrmNegStrategy()
    bids = strategy.calculate_bids(
        pp, _cap_cfg("neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(600.0)
    assert bids[0]["price"] == pytest.approx(10.0 * FORESIGHT_STEPS)


# ---------------------------------------------------------------------------
# Positive energy
# ---------------------------------------------------------------------------


def test_en_pos_must_offer_only_priced_at_signal():
    """base=700, cap_pos=300, max=1000 -> voluntary 0; single must bid at signal."""
    pp = _make_powerplant(crm_en_pos=50.0)
    pp.outputs["energy"][HOUR_START] = 700.0
    pp.outputs["capacity_pos"][HOUR_START] = 300.0
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(300.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_en_pos_must_plus_voluntary_priced_at_signal():
    """base=500, cap_pos=200 -> must 200 + voluntary 300, both at the signal (50)."""
    pp = _make_powerplant(crm_en_pos=50.0)
    pp.outputs["energy"][HOUR_START] = 500.0
    pp.outputs["capacity_pos"][HOUR_START] = 200.0
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(200.0)
    assert vol["volume"] == pytest.approx(300.0)
    assert must["price"] == pytest.approx(50.0)
    assert vol["price"] == pytest.approx(50.0)


def test_en_pos_voluntary_only_when_no_cap():
    pp = _make_powerplant(crm_en_pos=50.0)
    pp.outputs["energy"][HOUR_START] = 300.0
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P - 300.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_en_pos_voluntary_skipped_when_offline():
    """base=0 (offline), no cap -> cannot deliver pos voluntary -> no bids."""
    pp = _make_powerplant(crm_en_pos=50.0)
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert bids == []


def test_en_pos_must_offer_emitted_even_when_offline():
    """base=0 but cap_pos=100 -> must-offer is still emitted (binding)."""
    pp = _make_powerplant(crm_en_pos=50.0)
    pp.outputs["capacity_pos"][HOUR_START] = 100.0
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_en_pos_marginal_cost_fallback_without_signal():
    """No CRM energy signal -> price falls back to marginal cost (40)."""
    pp = _make_powerplant(include_crm_signals=False)
    pp.outputs["energy"][HOUR_START] = 300.0
    strategy = PowerPlantCrmPosStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(MC)


# ---------------------------------------------------------------------------
# Negative energy
# ---------------------------------------------------------------------------


def test_en_neg_must_offer_only():
    """base=300, cap_neg=100, min=200 -> total room 100 = must; voluntary 0."""
    pp = _make_powerplant(crm_en_neg=30.0)
    pp.outputs["energy"][HOUR_START] = 300.0
    pp.outputs["capacity_neg"][HOUR_START] = 100.0
    strategy = PowerPlantCrmNegStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("neg"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(30.0)


def test_en_neg_must_plus_voluntary():
    """base=800, cap_neg=100, min=200 -> room 600; must 100 + voluntary 500 at signal."""
    pp = _make_powerplant(crm_en_neg=30.0)
    pp.outputs["energy"][HOUR_START] = 800.0
    pp.outputs["capacity_neg"][HOUR_START] = 100.0
    strategy = PowerPlantCrmNegStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("neg"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(100.0)
    assert vol["volume"] == pytest.approx(500.0)
    assert must["price"] == pytest.approx(30.0)
    assert vol["price"] == pytest.approx(30.0)


def test_en_neg_no_bids_at_min_power():
    pp = _make_powerplant(crm_en_neg=30.0)
    pp.outputs["energy"][HOUR_START] = MIN_P
    strategy = PowerPlantCrmNegStrategy()
    bids = strategy.calculate_bids(pp, _en_cfg("neg"), [(HOUR_START, HOUR_END, None)])
    assert bids == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_pos_strategy_rejects_neg_product():
    pp = _make_powerplant()
    strategy = PowerPlantCrmPosStrategy()
    bad = _cap_cfg("pos")
    bad.product_type = "capacity_neg"
    with pytest.raises(ValueError, match="capacity_pos/energy_pos"):
        strategy.calculate_bids(pp, bad, [(BLOCK_START, BLOCK_END, None)])


def test_neg_strategy_rejects_pos_product():
    pp = _make_powerplant()
    strategy = PowerPlantCrmNegStrategy()
    bad = _en_cfg("neg")
    bad.product_type = "energy_pos"
    with pytest.raises(ValueError, match="capacity_neg/energy_neg"):
        strategy.calculate_bids(pp, bad, [(HOUR_START, HOUR_END, None)])


# ===========================================================================
# EOM strategy CRM-awareness patch (shared code in flexable.py)
# ===========================================================================


def _make_pp_with_startup_cost():
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        market_prices={"EOM": 60.0},
    )
    return PowerPlant(
        id="committed_pp",
        unit_operator="op",
        technology="hard coal",
        index=forecaster.index,
        max_power=MAX_P,
        min_power=MIN_P,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
        hot_start_cost=100,
        downtime_hot_start=4,
        min_operating_time=1,
        min_down_time=1,
    )


def _eom_cfg():
    cfg = _CfgStub()
    cfg.product_type = "energy"
    cfg.market_id = "EOM"
    cfg.maximum_bid_price = 3000.0
    cfg.minimum_bid_price = -500.0
    return cfg


def test_eom_applies_price_reduction_without_commitment():
    pp = _make_pp_with_startup_cost()
    start = datetime(2023, 7, 1, 1)
    end = datetime(2023, 7, 1, 2)
    pp.outputs["energy"][datetime(2023, 7, 1, 0)] = 500.0  # was on -> op_time > 0
    strategy = EnergyHeuristicFlexableStrategy()
    bids = strategy.calculate_bids(pp, _eom_cfg(), product_tuples=[(start, end, None)])
    assert len(bids) == 2
    assert bids[0]["volume"] == pytest.approx(MIN_P)
    assert bids[0]["price"] == pytest.approx(-460.0)


def test_eom_skips_price_reduction_when_capacity_neg_committed():
    pp = _make_pp_with_startup_cost()
    start = datetime(2023, 7, 1, 1)
    end = datetime(2023, 7, 1, 2)
    pp.outputs["energy"][datetime(2023, 7, 1, 0)] = 500.0
    pp.outputs["capacity_neg"][start] = 50.0  # CRM commitment
    strategy = EnergyHeuristicFlexableStrategy()
    bids = strategy.calculate_bids(pp, _eom_cfg(), product_tuples=[(start, end, None)])
    assert len(bids) == 2
    assert bids[0]["volume"] == pytest.approx(MIN_P + 50.0)
    assert bids[0]["price"] == pytest.approx(MC)


def test_eom_skips_price_reduction_when_capacity_pos_committed():
    pp = _make_pp_with_startup_cost()
    start = datetime(2023, 7, 1, 1)
    end = datetime(2023, 7, 1, 2)
    pp.outputs["energy"][datetime(2023, 7, 1, 0)] = 500.0
    pp.outputs["capacity_pos"][start] = 100.0
    strategy = EnergyHeuristicFlexableStrategy()
    bids = strategy.calculate_bids(pp, _eom_cfg(), product_tuples=[(start, end, None)])
    assert len(bids) == 2
    assert bids[0]["price"] == pytest.approx(MC)


# ===========================================================================
# calculate_ramp capacity-commitment bypass (shared code in base.py)
# ===========================================================================


def _ramp_pp():
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    forecaster = PowerplantForecaster(index, fuel_prices={"lignite": 10, "co2": 10})
    return PowerPlant(
        id="ramp_pp",
        unit_operator="op",
        technology="hard coal",
        index=index,
        max_power=MAX_P,
        min_power=MIN_P,
        efficiency=0.5,
        additional_cost=10,
        bidding_strategies={},
        fuel_type="lignite",
        emission_factor=0.5,
        forecaster=forecaster,
        ramp_up=200,
        ramp_down=200,
        min_operating_time=1,
        min_down_time=2,
    )


def test_calculate_ramp_blocks_startup_without_commitment():
    pp = _ramp_pp()
    result = pp.calculate_ramp(op_time=-1, previous_power=0, power=300, current_power=0)
    assert result == 0


def test_calculate_ramp_allows_startup_with_capacity_pos_commitment():
    pp = _ramp_pp()
    t = pp.index[0]
    pp.outputs["capacity_pos"][t] = 100.0
    result = pp.calculate_ramp(
        op_time=-1, previous_power=0, power=300, current_power=0, start=t
    )
    assert result == 200.0


def test_calculate_ramp_allows_startup_with_capacity_neg_commitment():
    pp = _ramp_pp()
    t = pp.index[0]
    pp.outputs["capacity_neg"][t] = 50.0
    result = pp.calculate_ramp(
        op_time=-1, previous_power=0, power=300, current_power=0, start=t
    )
    assert result == 200.0


def test_calculate_ramp_no_commitment_with_start_arg_unchanged():
    pp = _ramp_pp()
    t = pp.index[0]
    result = pp.calculate_ramp(
        op_time=-1, previous_power=0, power=300, current_power=0, start=t
    )
    assert result == 0
