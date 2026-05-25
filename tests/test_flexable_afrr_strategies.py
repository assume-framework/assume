# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for PowerPlantAfrrCapBlockStrategy (4h block aFRR capacity bids).

Conventions (so the arithmetic is transparent):
    fuel_price = 10, efficiency = 0.5, co2 = 10, emission_factor = 0.5,
    additional_cost = 10
    => marginal_cost = 10/0.5 + 10*0.5/0.5 + 10 = 20 + 10 + 10 = 40 EUR/MWh
"""

from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecaster import PowerplantForecaster
from assume.strategies.flexable import EnergyHeuristicFlexableStrategy
from assume.strategies.flexable_afrr import (
    PowerPlantAfrrCapBlockStrategy,
    PowerPlantAfrrEnergyStrategy,
)
from assume.units import PowerPlant

MC = 40.0  # marginal_cost EUR/MWh given the fixture below
MAX_P = 1000.0
MIN_P = 200.0

BLOCK_START = datetime(2023, 7, 1, 0)
BLOCK_END = datetime(2023, 7, 1, 4)  # 4h block
SECOND_BLOCK_END = datetime(2023, 7, 1, 8)


def _make_powerplant(da_price=60.0, afrr_en_pos=50.0, afrr_en_neg=30.0):
    """PowerPlant fixture with controllable cross-market price forecasts."""
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        market_prices={
            "EOM": da_price,
            "aFRR_en_pos": afrr_en_pos,
            "aFRR_en_neg": afrr_en_neg,
        },
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


def _cap_market_config(product_type="capacity_pos"):
    cfg = _CfgStub()
    cfg.product_type = product_type
    cfg.market_id = "aFRR_cap_pos" if product_type == "capacity_pos" else "aFRR_cap_neg"
    return cfg


def _energy_market_config(product_type="energy_pos"):
    cfg = _CfgStub()
    cfg.product_type = product_type
    cfg.market_id = "aFRR_en_pos" if product_type == "energy_pos" else "aFRR_en_neg"
    return cfg


HOUR_START = BLOCK_START
HOUR_END = datetime(2023, 7, 1, 1)
NEXT_HOUR_END = datetime(2023, 7, 1, 2)


# ---------------------------------------------------------------------------
# Positive capacity tests
# ---------------------------------------------------------------------------


def test_pos_cap_basic_block_no_activation():
    """
    DA = 60, MC = 40 -> per-hour opp cost = 20*1000 = 20_000 EUR
    over 4h block: opp_cost_total = 80_000 EUR
    activation_probability = 0 -> no energy revenue offset
    expected block_price = 80_000 / 1000 = 80 EUR/MW, volume = 1000 MW
    """
    pp = _make_powerplant(da_price=60.0)
    strat = PowerPlantAfrrCapBlockStrategy(activation_probability=0.0)
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P)
    assert bids[0]["price"] == pytest.approx(80.0)
    assert bids[0]["start_time"] == BLOCK_START
    assert bids[0]["end_time"] == BLOCK_END


def test_pos_cap_activation_reduces_price():
    """
    DA = 60, aFRR_en_pos = 50, MC = 40, p_activation = 0.5
    opp_cost     = (60-40)*1000*4 = 80_000
    en_value     = (50-40)*1000*4 = 40_000
    net_value    = 80_000 - 0.5 * 40_000 = 60_000
    block_price  = 60_000 / 1000 = 60 EUR/MW
    """
    pp = _make_powerplant(da_price=60.0, afrr_en_pos=50.0)
    strat = PowerPlantAfrrCapBlockStrategy(activation_probability=0.5)
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(60.0)


def test_pos_cap_da_below_mc_yields_floor():
    """
    DA = 20 < MC = 40 -> no EOM opportunity cost.
    aFRR_en activation revenue is positive but only enters via -p_act*revenue,
    so net_value is non-positive. Result clipped to capacity_price_floor.
    """
    pp = _make_powerplant(da_price=20.0, afrr_en_pos=50.0)
    strat = PowerPlantAfrrCapBlockStrategy(
        activation_probability=0.5, capacity_price_floor=0.0
    )
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(0.0)


def test_pos_cap_volume_takes_min_across_block():
    """
    Cap availability of the 3rd hour to 0.5. Block bid volume must equal the
    most-constrained hour: 0.5 * max_power = 500 MW.
    """
    pp = _make_powerplant(da_price=60.0)
    # availability series is full at 1.0, drop hour 2 (third hour of block)
    pp.forecaster.availability[BLOCK_START + pd.Timedelta(hours=2)] = 0.5

    strat = PowerPlantAfrrCapBlockStrategy(activation_probability=0.0)
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(500.0)
    # block price = opp_cost / volume:
    # opp_cost per hour = 20 * vol; total = 20*4*500 = 40_000; price = 40_000/500 = 80
    assert bids[0]["price"] == pytest.approx(80.0)


def test_pos_cap_emits_one_bid_per_block():
    """Two consecutive 4h blocks should produce two bids."""
    pp = _make_powerplant(da_price=60.0)
    strat = PowerPlantAfrrCapBlockStrategy(activation_probability=0.0)
    products = [
        (BLOCK_START, BLOCK_END, None),
        (BLOCK_END, SECOND_BLOCK_END, None),
    ]
    bids = strat.calculate_bids(pp, _cap_market_config("capacity_pos"), products)
    assert len(bids) == 2
    assert bids[0]["start_time"] == BLOCK_START
    assert bids[0]["end_time"] == BLOCK_END
    assert bids[1]["start_time"] == BLOCK_END
    assert bids[1]["end_time"] == SECOND_BLOCK_END
    # both blocks see the same flat DA price -> same bid
    assert bids[0]["price"] == pytest.approx(bids[1]["price"])
    assert bids[0]["volume"] == pytest.approx(bids[1]["volume"])


def test_pos_cap_price_clipped_to_market_max():
    """
    Extremely high DA price -> opp_cost very large -> price would exceed the
    market's maximum_bid_price. Verify clipping.
    """
    pp = _make_powerplant(da_price=20_000.0)
    strat = PowerPlantAfrrCapBlockStrategy(activation_probability=0.0)
    cfg = _cap_market_config("capacity_pos")
    cfg.maximum_bid_price = 500.0
    bids = strat.calculate_bids(pp, cfg, [(BLOCK_START, BLOCK_END, None)])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Negative capacity tests
# ---------------------------------------------------------------------------


def test_neg_cap_with_proxy_yields_bid():
    """
    With forecast_eom_dispatch=True and DA > MC, the proxy assumes the unit
    will run at max_power in EOM, so it has (max_power - min_power) = 800 MW
    of ramp-down room per hour.
    Opp cost for neg = max(0, MC - DA) * V = 0 since DA(60) > MC(40).
    en_revenue (neg) = (MC - aFRR_en_neg) * V = (40 - 30) * 800 = 8_000 per hour
                     * 4h = 32_000
    net_value = 0 - p_act * en_revenue = -p_act * 32_000  (negative)
    price clipped to floor = 0.
    """
    pp = _make_powerplant(da_price=60.0, afrr_en_neg=30.0)
    strat = PowerPlantAfrrCapBlockStrategy(
        activation_probability=0.5,
        capacity_price_floor=0.0,
        forecast_eom_dispatch=True,
    )
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P - MIN_P)
    assert bids[0]["price"] == pytest.approx(0.0)


def test_neg_cap_must_run_cost_when_da_below_mc():
    """
    With proxy ON but DA = 30 < MC = 40, the proxy treats the unit as NOT
    running in EOM (DA < MC). With base_power = 0 we have no ramp-down room.
    Verify no bid emitted.
    """
    pp = _make_powerplant(da_price=30.0)
    strat = PowerPlantAfrrCapBlockStrategy(
        activation_probability=0.0, forecast_eom_dispatch=True
    )
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert bids == []


def test_neg_cap_without_proxy_yields_no_bid():
    """
    forecast_eom_dispatch=False, base_power=0 -> no ramp-down headroom even if
    DA > MC. Demonstrates the pre-clearing limitation when the proxy is off.
    """
    pp = _make_powerplant(da_price=60.0)
    strat = PowerPlantAfrrCapBlockStrategy(
        activation_probability=0.0, forecast_eom_dispatch=False
    )
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert bids == []


def test_neg_cap_uses_committed_dispatch_when_present():
    """
    When base_power is non-zero (e.g. EOM has cleared in earlier scenario step),
    the strategy should NOT rely on the proxy -- it should use the actual
    committed dispatch. Verify by pre-loading outputs["energy"] to 800 MW for
    the block hours, and disabling the proxy.
    """
    pp = _make_powerplant(da_price=60.0, afrr_en_neg=30.0)
    for h in range(4):
        pp.outputs["energy"][BLOCK_START + pd.Timedelta(hours=h)] = 800.0

    strat = PowerPlantAfrrCapBlockStrategy(
        activation_probability=0.0, forecast_eom_dispatch=False
    )
    bids = strat.calculate_bids(
        pp, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    # ramp-down room = 800 - min_power(200) = 600 MW per hour
    assert bids[0]["volume"] == pytest.approx(600.0)
    # opp_cost = 0 (DA > MC), so price clipped to floor
    assert bids[0]["price"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unsupported_product_type_raises():
    pp = _make_powerplant()
    strat = PowerPlantAfrrCapBlockStrategy()
    bad_cfg = _cap_market_config("capacity_pos")
    bad_cfg.product_type = "energy"  # not a capacity product
    with pytest.raises(ValueError, match="capacity_pos/capacity_neg"):
        strat.calculate_bids(pp, bad_cfg, [(BLOCK_START, BLOCK_END, None)])


# ===========================================================================
# PowerPlantAfrrEnergyStrategy tests
# ===========================================================================


def test_en_pos_must_offer_only_when_no_headroom():
    """
    base=700, cap_pos=300, max=1000  => voluntary headroom = 0
    => single must-offer bid at MC=40 for V=300.
    """
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = 700.0
    pp.outputs["capacity_pos"][HOUR_START] = 300.0

    strat = PowerPlantAfrrEnergyStrategy()
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(300.0)
    assert bids[0]["price"] == pytest.approx(MC)


def test_en_pos_must_plus_voluntary():
    """
    base=500, cap_pos=200, max=1000 => voluntary headroom = 300
    must-offer: 200 MW at MC=40
    voluntary:  300 MW at MC + markup = 45
    """
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = 500.0
    pp.outputs["capacity_pos"][HOUR_START] = 200.0

    strat = PowerPlantAfrrEnergyStrategy(voluntary_markup=5.0)
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(200.0)
    assert must["price"] == pytest.approx(MC)
    assert vol["volume"] == pytest.approx(300.0)
    assert vol["price"] == pytest.approx(MC + 5.0)


def test_en_pos_voluntary_only_when_no_cap_reserved():
    """
    base=0, cap_pos=0, max=1000 => no must-offer; voluntary covers full max.
    """
    pp = _make_powerplant()
    strat = PowerPlantAfrrEnergyStrategy(voluntary_markup=0.0)
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_P)
    assert bids[0]["price"] == pytest.approx(MC)


def test_en_pos_no_bids_when_unit_fully_committed():
    """base=max_power, cap_pos=0 => no headroom in either direction."""
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = MAX_P

    strat = PowerPlantAfrrEnergyStrategy()
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert bids == []


def test_en_neg_must_offer_only_when_at_floor_plus_cap():
    """
    base=300, cap_neg=100, min_power=200 => total ramp-down room = 100,
    consumed entirely by must-offer; no voluntary.
    """
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = 300.0
    pp.outputs["capacity_neg"][HOUR_START] = 100.0

    strat = PowerPlantAfrrEnergyStrategy()
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(MC)


def test_en_neg_must_plus_voluntary():
    """
    base=800, cap_neg=100, min_power=200 => total ramp-down = 600
    must-offer:  100 MW at MC=40 (saved fuel break-even)
    voluntary:   500 MW at MC - discount = 35
    """
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = 800.0
    pp.outputs["capacity_neg"][HOUR_START] = 100.0

    strat = PowerPlantAfrrEnergyStrategy(voluntary_discount=5.0)
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(100.0)
    assert must["price"] == pytest.approx(MC)
    assert vol["volume"] == pytest.approx(500.0)
    assert vol["price"] == pytest.approx(MC - 5.0)


def test_en_neg_voluntary_only_when_no_cap():
    """base=800, cap_neg=0 => entire 800-200=600 MW is voluntary."""
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = 800.0

    strat = PowerPlantAfrrEnergyStrategy(voluntary_discount=0.0)
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(600.0)
    assert bids[0]["price"] == pytest.approx(MC)


def test_en_neg_no_bids_when_at_min_power():
    """base=min_power, cap_neg=0 => zero ramp-down room."""
    pp = _make_powerplant()
    pp.outputs["energy"][HOUR_START] = MIN_P

    strat = PowerPlantAfrrEnergyStrategy()
    bids = strat.calculate_bids(
        pp, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert bids == []


def test_en_pos_emits_two_bids_per_hourly_product():
    """Two hourly products with the same state => 4 bids (2 each)."""
    pp = _make_powerplant()
    for h in range(2):
        t = HOUR_START + pd.Timedelta(hours=h)
        pp.outputs["energy"][t] = 500.0
        pp.outputs["capacity_pos"][t] = 200.0

    strat = PowerPlantAfrrEnergyStrategy(voluntary_markup=5.0)
    bids = strat.calculate_bids(
        pp,
        _energy_market_config("energy_pos"),
        [(HOUR_START, HOUR_END, None), (HOUR_END, NEXT_HOUR_END, None)],
    )
    assert len(bids) == 4
    # bid order: (must_h0, vol_h0, must_h1, vol_h1)
    assert all(b["volume"] in (200.0, 300.0) for b in bids)
    assert {b["start_time"] for b in bids} == {HOUR_START, HOUR_END}


def test_en_strategy_rejects_capacity_product_type():
    pp = _make_powerplant()
    strat = PowerPlantAfrrEnergyStrategy()
    bad_cfg = _energy_market_config("energy_pos")
    bad_cfg.product_type = "capacity_pos"
    with pytest.raises(ValueError, match="energy_pos/energy_neg"):
        strat.calculate_bids(pp, bad_cfg, [(HOUR_START, HOUR_END, None)])


# ===========================================================================
# EOM strategy aFRR-awareness patch
# ===========================================================================


def _make_pp_with_startup_cost():
    """
    PP with non-zero hot_start_cost so that calculate_EOM_price_if_on produces a
    visible price_reduction_restart. ``downtime_hot_start=4`` ensures
    get_starting_costs(-min_down_time=-1) returns the hot start cost (the
    `-op_time < downtime_hot_start` branch).
    """
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        market_prices={"EOM": 60.0},  # > MC=40 so unit is profitable
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
        hot_start_cost=100,  # scaled by max_power in __init__ -> 100_000 EUR
        downtime_hot_start=4,
        min_operating_time=1,
        min_down_time=1,
    )


def test_eom_strategy_applies_price_reduction_without_capacity_commitment():
    """
    Baseline (control) for the aFRR-awareness patch.

    Setup:
        - hot_start_cost (scaled) = 100 * 1000 = 100_000 EUR
        - min_down_time = 1, bid_quantity_inflex = 200, MC = 40
        - unit was producing at the prior hour -> op_time > 0 -> "if_on" path
    Without a capacity commitment, the price reduction is:
        price_reduction = 100_000 / 1 / 200 = 500 EUR/MWh
    Inflex bid price = max(-500 + 40, -499) = -460.
    """
    pp = _make_pp_with_startup_cost()
    start = datetime(2023, 7, 1, 1)
    end = datetime(2023, 7, 1, 2)
    pp.outputs["energy"][datetime(2023, 7, 1, 0)] = 500.0  # was on -> op_time > 0

    strategy = EnergyHeuristicFlexableStrategy()
    cfg = _CfgStub()
    cfg.product_type = "energy"
    cfg.market_id = "EOM"
    cfg.maximum_bid_price = 3000.0
    cfg.minimum_bid_price = -500.0

    bids = strategy.calculate_bids(pp, cfg, product_tuples=[(start, end, None)])
    assert len(bids) == 2  # inflex + flex
    assert bids[0]["volume"] == pytest.approx(MIN_P)
    assert bids[0]["price"] == pytest.approx(-460.0)


def test_eom_strategy_skips_price_reduction_when_capacity_neg_committed():
    """
    With aFRR-cap_neg=50 committed at the product hour:
      - calculate_min_max_power raises the EOM min_power by capacity_neg,
        so the inflex bid volume becomes MIN_P + 50 = 250 MW.
      - The unit is contractually online -> price_reduction_restart is skipped.
        bid_price = max(0 + MC, -499) = MC = 40 EUR/MWh.
    """
    pp = _make_pp_with_startup_cost()
    start = datetime(2023, 7, 1, 1)
    end = datetime(2023, 7, 1, 2)
    pp.outputs["energy"][datetime(2023, 7, 1, 0)] = 500.0
    pp.outputs["capacity_neg"][start] = 50.0  # aFRR commitment

    strategy = EnergyHeuristicFlexableStrategy()
    cfg = _CfgStub()
    cfg.product_type = "energy"
    cfg.market_id = "EOM"
    cfg.maximum_bid_price = 3000.0
    cfg.minimum_bid_price = -500.0

    bids = strategy.calculate_bids(pp, cfg, product_tuples=[(start, end, None)])
    assert len(bids) == 2
    assert bids[0]["volume"] == pytest.approx(MIN_P + 50.0)  # raised by cap_neg
    assert bids[0]["price"] == pytest.approx(MC)  # no reduction applied


def test_eom_strategy_skips_price_reduction_when_capacity_pos_committed():
    """Same behavior should hold for capacity_pos commitment (also implies on-state)."""
    pp = _make_pp_with_startup_cost()
    start = datetime(2023, 7, 1, 1)
    end = datetime(2023, 7, 1, 2)
    pp.outputs["energy"][datetime(2023, 7, 1, 0)] = 500.0
    pp.outputs["capacity_pos"][start] = 100.0

    strategy = EnergyHeuristicFlexableStrategy()
    cfg = _CfgStub()
    cfg.product_type = "energy"
    cfg.market_id = "EOM"
    cfg.maximum_bid_price = 3000.0
    cfg.minimum_bid_price = -500.0

    bids = strategy.calculate_bids(pp, cfg, product_tuples=[(start, end, None)])
    assert len(bids) == 2
    assert bids[0]["price"] == pytest.approx(MC)
