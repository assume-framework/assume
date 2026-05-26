# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for StorageAfrrCapBlockStrategy (4h block aFRR capacity bids).

Conventions (so the arithmetic is transparent):
    capacity              = 400 MWh
    max_power_discharge   = 100 MW
    max_power_charge      = -100 MW
    efficiency_charge     = 1.0
    efficiency_discharge  = 1.0
    additional_cost_dis   = 5 EUR/MWh    -> MC_discharge = 5
    additional_cost_chrg  = 2 EUR/MWh    -> MC_charge    = 2
    initial_soc           = 0.5          -> 200 MWh available, 200 MWh headroom

With delivery_duration_hours = 1h:
    SOC-derived pos limit = 0.5 * 400 * 1.0 / 1 = 200 MW  -> not binding
    Power-derived pos limit = 100 MW                       -> binding
    => block volume = 100 MW
"""

from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.strategies.flexable_afrr_storage import (
    StorageAfrrCapBlockStrategy,
    StorageAfrrEnergyStrategy,
)
from assume.units import Storage

CAPACITY = 400.0
MAX_PD = 100.0
MAX_PC = -100.0
MC_DIS = 5.0
MC_CHG = 2.0

BLOCK_START = datetime(2023, 7, 1, 0)
BLOCK_END = datetime(2023, 7, 1, 4)
SECOND_BLOCK_END = datetime(2023, 7, 1, 8)


def _make_storage(
    initial_soc=0.5,
    da_price=60.0,
    afrr_en_pos=50.0,
    afrr_en_neg=1.0,
):
    """Storage fixture with controllable cross-market price forecasts."""
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    forecaster = UnitForecaster(
        index,
        market_prices={
            "EOM": da_price,
            "aFRR_en_pos": afrr_en_pos,
            "aFRR_en_neg": afrr_en_neg,
        },
    )
    return Storage(
        id="test_storage",
        unit_operator="test_op",
        technology="battery",
        bidding_strategies={},
        forecaster=forecaster,
        max_power_charge=MAX_PC,
        max_power_discharge=MAX_PD,
        capacity=CAPACITY,
        initial_soc=initial_soc,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        additional_cost_charge=MC_CHG,
        additional_cost_discharge=MC_DIS,
    )


class _CfgStub:
    maximum_bid_price = 9999.0
    minimum_bid_price = -500.0
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


def test_pos_cap_volume_power_limited():
    """
    SOC=0.5 -> SOC-derived limit = 200 MW, power-limited at 100 MW.
    Flat DA price -> opp_cost=0 -> price clipped to floor (0).
    """
    s = _make_storage()
    strategy = StorageAfrrCapBlockStrategy(
        activation_probability=0.0, delivery_duration_hours=1.0
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_PD)
    assert bids[0]["price"] == pytest.approx(0.0)


def test_pos_cap_volume_soc_limited():
    """
    delivery_duration=10h, SOC=0.5: SOC limit = 200/10 = 20 MW < 100 MW.
    => volume = 20 MW
    """
    s = _make_storage()
    strategy = StorageAfrrCapBlockStrategy(
        activation_probability=0.0, delivery_duration_hours=10.0
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(20.0)


def test_pos_cap_zero_when_soc_empty():
    """SOC=0 -> no SOC to discharge -> no bid."""
    s = _make_storage(initial_soc=0.0)
    strategy = StorageAfrrCapBlockStrategy(activation_probability=0.0)
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert bids == []


def test_pos_cap_price_reflects_da_spread():
    """
    eom_foresight=4h, block_start=0:00 -> window covers [0:00, 4:00] = 5 steps.
    DA = [100, 100, 100, 100, 60, 60, ..., 60]
       window values = [100, 100, 100, 100, 60]
       DA_avg = 92
    spread in each block hour = 100 - 92 = 8
    opp_cost = 8 * 100 * 4 = 3200
    p_activation=0 -> net = 3200 -> block_price = 32 EUR/MW
    """
    s = _make_storage(da_price=[100.0] * 4 + [60.0] * 20)
    strategy = StorageAfrrCapBlockStrategy(
        eom_foresight="4h",
        activation_probability=0.0,
        delivery_duration_hours=1.0,
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_PD)
    assert bids[0]["price"] == pytest.approx(32.0)


def test_pos_cap_activation_reduces_price():
    """
    Same DA setup as above (opp_cost=3200) but with activation_probability=0.5
    and aFRR_en_pos=10:
        en_value = (10 - 5) * 100 * 4 = 2000
        net      = 3200 - 0.5 * 2000 = 2200
        price    = 2200 / 100 = 22 EUR/MW
    """
    s = _make_storage(
        da_price=[100.0] * 4 + [60.0] * 20,
        afrr_en_pos=10.0,
    )
    strategy = StorageAfrrCapBlockStrategy(
        eom_foresight="4h",
        activation_probability=0.5,
        delivery_duration_hours=1.0,
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# Negative capacity tests
# ---------------------------------------------------------------------------


def test_neg_cap_volume_power_limited():
    """
    SOC=0.5 -> headroom 200 MWh -> SOC-derived limit = 200 MW for d=1h.
    Power-limited at |max_power_charge| = 100 MW.
    """
    s = _make_storage()
    strategy = StorageAfrrCapBlockStrategy(
        activation_probability=0.0, delivery_duration_hours=1.0
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(abs(MAX_PC))
    assert bids[0]["price"] == pytest.approx(0.0)


def test_neg_cap_zero_when_soc_full():
    """SOC=1 -> no headroom to absorb -> no bid."""
    s = _make_storage(initial_soc=1.0)
    strategy = StorageAfrrCapBlockStrategy(activation_probability=0.0)
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert bids == []


def test_neg_cap_price_reflects_da_average_minus_da():
    """
    DA = [40, 40, 40, 40, 80, 80, ..., 80]
        window [0:00, 4:00] values = [40, 40, 40, 40, 80] -> DA_avg = 48
    spread per block hour (neg) = max(0, 48 - 40) = 8
    opp_cost = 8 * 100 * 4 = 3200
    p_activation=0 -> price = 32 EUR/MW
    """
    s = _make_storage(da_price=[40.0] * 4 + [80.0] * 20)
    strategy = StorageAfrrCapBlockStrategy(
        eom_foresight="4h",
        activation_probability=0.0,
        delivery_duration_hours=1.0,
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(32.0)


def test_neg_cap_volume_soc_limited():
    """
    delivery_duration=10h, SOC=0.5: headroom limit = 200/10 = 20 MW < 100 MW.
    """
    s = _make_storage()
    strategy = StorageAfrrCapBlockStrategy(
        activation_probability=0.0, delivery_duration_hours=10.0
    )
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Existing-commitment + multi-block tests
# ---------------------------------------------------------------------------


def test_pos_cap_subtracts_existing_capacity_pos():
    """
    If 30 MW of capacity_pos is already reserved at block start, the strategy
    must not double-commit -> remaining bid = 100 - 30 = 70 MW.
    """
    s = _make_storage()
    s.outputs["capacity_pos"][BLOCK_START] = 30.0

    strategy = StorageAfrrCapBlockStrategy(activation_probability=0.0)
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(70.0)


def test_emits_one_bid_per_block():
    """Two consecutive 4h blocks -> two bids."""
    s = _make_storage()
    strategy = StorageAfrrCapBlockStrategy(activation_probability=0.0)
    bids = strategy.calculate_bids(
        s,
        _cap_market_config("capacity_pos"),
        [(BLOCK_START, BLOCK_END, None), (BLOCK_END, SECOND_BLOCK_END, None)],
    )
    assert len(bids) == 2
    assert bids[0]["volume"] == pytest.approx(MAX_PD)
    assert bids[1]["volume"] == pytest.approx(MAX_PD)


def test_unsupported_product_type_raises():
    s = _make_storage()
    strategy = StorageAfrrCapBlockStrategy()
    bad_cfg = _cap_market_config("capacity_pos")
    bad_cfg.product_type = "energy"
    with pytest.raises(ValueError, match="capacity_pos/capacity_neg"):
        strategy.calculate_bids(s, bad_cfg, [(BLOCK_START, BLOCK_END, None)])


# ===========================================================================
# StorageAfrrEnergyStrategy tests
# ===========================================================================


def test_en_pos_must_offer_only_when_no_headroom():
    """cap_pos=100, base=0 => no voluntary headroom; one must-offer bid at MC_dis=5."""
    s = _make_storage()
    s.outputs["capacity_pos"][HOUR_START] = 100.0

    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(MC_DIS)


def test_en_pos_must_plus_voluntary():
    """cap_pos=30, base=0 => must=30 at 5, voluntary=70 at 5+markup."""
    s = _make_storage()
    s.outputs["capacity_pos"][HOUR_START] = 30.0

    strategy = StorageAfrrEnergyStrategy(voluntary_markup=2.0)
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(30.0)
    assert must["price"] == pytest.approx(MC_DIS)
    assert vol["volume"] == pytest.approx(70.0)
    assert vol["price"] == pytest.approx(MC_DIS + 2.0)


def test_en_pos_voluntary_only_when_no_cap():
    """No cap reserved => single voluntary bid at MC."""
    s = _make_storage()
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_PD)
    assert bids[0]["price"] == pytest.approx(MC_DIS)


def test_en_pos_soc_limited_voluntary():
    """
    SOC=0.1 => soc_room = 0.1*400*1/1 = 40 MW < power_room=100. Voluntary=40.
    """
    s = _make_storage(initial_soc=0.1)
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(40.0)


def test_en_pos_no_bid_when_soc_empty():
    """SOC=0 => no discharge possible."""
    s = _make_storage(initial_soc=0.0)
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert bids == []


def test_en_pos_accounts_for_base_load_discharge():
    """
    base_load=+30 (EOM discharging 30) => power_room = 100 - 30 - cap = 70 - cap.
    cap=0, so voluntary = min(70, soc_room=170) = 70.
    """
    s = _make_storage()
    s.outputs["energy"][HOUR_START] = 30.0
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_pos"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(70.0)


def test_en_neg_must_offer_only_when_no_headroom():
    """cap_neg=100, base=0 => no voluntary; one must-offer at MC_chg=2."""
    s = _make_storage()
    s.outputs["capacity_neg"][HOUR_START] = 100.0

    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(MC_CHG)


def test_en_neg_must_plus_voluntary():
    """cap_neg=30, base=0 => must=30 at 2, voluntary=70 at 2-discount."""
    s = _make_storage()
    s.outputs["capacity_neg"][HOUR_START] = 30.0

    strategy = StorageAfrrEnergyStrategy(voluntary_discount=1.0)
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(30.0)
    assert must["price"] == pytest.approx(MC_CHG)
    assert vol["volume"] == pytest.approx(70.0)
    assert vol["price"] == pytest.approx(MC_CHG - 1.0)


def test_en_neg_voluntary_only_when_no_cap():
    """No cap reserved => single voluntary bid."""
    s = _make_storage()
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(abs(MAX_PC))
    assert bids[0]["price"] == pytest.approx(MC_CHG)


def test_en_neg_soc_limited_voluntary():
    """
    SOC=0.9 => soc_headroom = 0.1*400/1/1 = 40 MWh -> 40 MW voluntary cap.
    """
    s = _make_storage(initial_soc=0.9)
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(40.0)


def test_en_neg_no_bid_when_soc_full():
    """SOC=1.0 => no charge headroom."""
    s = _make_storage(initial_soc=1.0)
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert bids == []


def test_en_neg_accounts_for_base_load_charge():
    """
    base_load=-50 (already charging 50) => power_room = 100 - 50 - cap = 50 - cap.
    cap=0 => voluntary = min(50, soc_room=150) = 50.
    """
    s = _make_storage()
    s.outputs["energy"][HOUR_START] = -50.0
    strategy = StorageAfrrEnergyStrategy()
    bids = strategy.calculate_bids(
        s, _energy_market_config("energy_neg"), [(HOUR_START, HOUR_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(50.0)


def test_en_multiple_products_emit_per_hour_pairs():
    """2 hourly products with must+voluntary => 4 bids."""
    s = _make_storage()
    for h in range(2):
        s.outputs["capacity_pos"][HOUR_START + pd.Timedelta(hours=h)] = 30.0
    strategy = StorageAfrrEnergyStrategy(voluntary_markup=2.0)
    bids = strategy.calculate_bids(
        s,
        _energy_market_config("energy_pos"),
        [(HOUR_START, HOUR_END, None), (HOUR_END, NEXT_HOUR_END, None)],
    )
    assert len(bids) == 4
    assert {b["start_time"] for b in bids} == {HOUR_START, HOUR_END}
    volumes = sorted(b["volume"] for b in bids)
    assert volumes == [30.0, 30.0, 70.0, 70.0]


def test_en_strategy_rejects_capacity_product_type():
    s = _make_storage()
    strategy = StorageAfrrEnergyStrategy()
    bad_cfg = _energy_market_config("energy_pos")
    bad_cfg.product_type = "capacity_pos"
    with pytest.raises(ValueError, match="energy_pos/energy_neg"):
        strategy.calculate_bids(s, bad_cfg, [(HOUR_START, HOUR_END, None)])


# ===========================================================================
# Forecasted SOC walk tests
# ===========================================================================


def test_walk_off_default_matches_flat_behavior():
    """forecasted_soc_walk defaults to False -> same result as before."""
    s = _make_storage(da_price=[100.0] * 4 + [60.0] * 20)
    strategy = StorageAfrrCapBlockStrategy(
        eom_foresight="4h", activation_probability=0.0, delivery_duration_hours=1.0
    )  # walk omitted -> defaults to False
    bids = strategy.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(MAX_PD)


def test_walk_no_da_forecast_falls_back_to_flat():
    """
    No DA forecast available -> walk falls back to a flat trajectory ->
    walk-mode bid matches flat-mode bid.

    Use a forecaster without the EOM key (drop it post-construction so the
    walk's `da_price` lookup returns None).
    """
    s = _make_storage()
    s.forecaster.price.pop("EOM", None)

    strategy_flat = StorageAfrrCapBlockStrategy(
        activation_probability=0.0, delivery_duration_hours=1.0
    )
    strategy_walk = StorageAfrrCapBlockStrategy(
        activation_probability=0.0,
        delivery_duration_hours=1.0,
        forecasted_soc_walk=True,
    )
    bids_flat = strategy_flat.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    bids_walk = strategy_walk.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids_flat) == len(bids_walk) == 1
    assert bids_flat[0]["volume"] == pytest.approx(bids_walk[0]["volume"])


def test_walk_pos_depleted_soc_blocks_bid():
    """
    capacity=400, SOC=0.5 (=> 200 MWh available).
    DA = [100, 100, 60, 60, 40, 40, ...] with eom_foresight=4h.

    Hour 0: DA(100) >= avg(72)    -> discharge 100 MW. SOC: 0.5 -> 0.25.
    Hour 1: DA(100) >= avg(66.67) -> discharge 100 MW. SOC: 0.25 -> 0.0.
    Hour 2: SOC=0, no more discharge. Stays at 0.
    Hour 3: SOC=0. Stays.

    Flat-SOC bid: V = 100 (SOC ample), price = opp_cost / V > 0.
    Walk bid:    per-hour V = [100, 100, 0, 0] -> min = 0 -> no bid.
    """
    s = _make_storage(da_price=[100.0, 100.0, 60.0, 60.0] + [40.0] * 20)
    strategy_flat = StorageAfrrCapBlockStrategy(
        eom_foresight="4h", activation_probability=0.0, delivery_duration_hours=1.0
    )
    strategy_walk = StorageAfrrCapBlockStrategy(
        eom_foresight="4h",
        activation_probability=0.0,
        delivery_duration_hours=1.0,
        forecasted_soc_walk=True,
    )
    bids_flat = strategy_flat.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    bids_walk = strategy_walk.calculate_bids(
        s, _cap_market_config("capacity_pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids_flat) == 1
    assert bids_flat[0]["volume"] == pytest.approx(MAX_PD)
    assert bids_walk == []  # walk blocks the bid since later hours have no SOC


def test_walk_neg_filled_soc_blocks_bid():
    """
    Mirror for neg direction: low DA in block hours -> walk forecasts charging
    -> SOC rises to 1.0 -> no charge headroom in later hours -> no neg bid.

    Hour 0: DA(20) < avg(28) -> charge 100 MW. SOC: 0.5 -> 0.75.
    Hour 1: DA(20) < avg(33.33) -> charge max(100, (1-0.75)*400=100). SOC: 0.75 -> 1.0.
    Hour 2,3: SOC=1.0, no more charging.

    Flat-SOC bid: V = 100. Walk: min per-hour = 0 -> no bid.
    """
    s = _make_storage(da_price=[20.0] * 4 + [60.0] * 20)
    strategy_flat = StorageAfrrCapBlockStrategy(
        eom_foresight="4h", activation_probability=0.0, delivery_duration_hours=1.0
    )
    strategy_walk = StorageAfrrCapBlockStrategy(
        eom_foresight="4h",
        activation_probability=0.0,
        delivery_duration_hours=1.0,
        forecasted_soc_walk=True,
    )
    bids_flat = strategy_flat.calculate_bids(
        s, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    bids_walk = strategy_walk.calculate_bids(
        s, _cap_market_config("capacity_neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids_flat) == 1
    assert bids_flat[0]["volume"] == pytest.approx(abs(MAX_PC))
    assert bids_walk == []
