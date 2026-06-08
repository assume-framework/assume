# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for the storage CRM strategies (StorageCrmPosStrategy /
StorageCrmNegStrategy). Each class handles its direction's capacity *and*
energy product, switching on product_type.

Fixture conventions:
    capacity              = 400 MWh
    max_power_discharge   = 100 MW
    max_power_charge      = -100 MW
    efficiency_charge     = efficiency_discharge = 1.0
    additional_cost_dis   = 5  -> MC_discharge = 5
    additional_cost_chrg  = 2  -> MC_charge    = 2
    initial_soc           = 0.5 -> 200 MWh available, 200 MWh headroom
    step (sustainability horizon) = 1h

Capacity price uses the EOM *arbitrage spread* (price minus its foresight-window
average), so a flat EOM forecast yields a zero opportunity cost (price floors at 0).
"""

from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecaster import UnitForecaster
from assume.strategies.flexable_crm_storage import (
    StorageCrmNegStrategy,
    StorageCrmPosStrategy,
)
from assume.units import Storage

CAPACITY = 400.0
MAX_PD = 100.0
MAX_PC = -100.0
MC_DIS = 5.0
MC_CHG = 2.0

BLOCK_START = datetime(2023, 7, 1, 0)
BLOCK_END = datetime(2023, 7, 1, 4)

HOUR_START = BLOCK_START
HOUR_END = datetime(2023, 7, 1, 1)


def _make_storage(
    initial_soc=0.5,
    da_price=60.0,
    crm_en_pos=50.0,
    crm_en_neg=8.0,
    include_crm_signals=True,
):
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    market_prices = {"EOM": da_price}
    if include_crm_signals:
        market_prices["CRM_energy_pos"] = crm_en_pos
        market_prices["CRM_energy_neg"] = crm_en_neg
    forecaster = UnitForecaster(index, market_prices=market_prices)
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
# Capacity volume (SOC- and power-feasibility)
# ---------------------------------------------------------------------------


def test_pos_cap_volume_power_bound():
    """soc=0.5 -> SOC limit 200 MW, power limit 100 MW -> volume 100."""
    st = _make_storage(initial_soc=0.5)
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(
        st, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)


def test_pos_cap_volume_soc_bound():
    """soc=0.1 -> SOC limit 40 MW binds below the 100 MW power limit."""
    st = _make_storage(initial_soc=0.1)
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(
        st, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(40.0)


def test_pos_cap_volume_nets_existing_reservation():
    """capacity_pos=30 already committed -> power limit 100 - 30 = 70."""
    st = _make_storage(initial_soc=0.5)
    for h in range(4):
        st.outputs["capacity_pos"][BLOCK_START + pd.Timedelta(hours=h)] = 30.0
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(
        st, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(70.0)


def test_neg_cap_volume_power_bound():
    """soc=0.5 -> headroom 200 MWh -> SOC limit 200, power limit 100 -> 100."""
    st = _make_storage(initial_soc=0.5)
    strategy = StorageCrmNegStrategy()
    bids = strategy.calculate_bids(
        st, _cap_cfg("neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)


def test_neg_cap_volume_soc_bound():
    """soc=0.9 -> headroom 40 MWh binds below the 100 MW power limit."""
    st = _make_storage(initial_soc=0.9)
    strategy = StorageCrmNegStrategy()
    bids = strategy.calculate_bids(
        st, _cap_cfg("neg"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Capacity price (arbitrage-spread net opportunity cost)
# ---------------------------------------------------------------------------


def test_pos_cap_flat_price_floors_to_zero():
    """Flat EOM forecast -> zero arbitrage spread -> price floors at 0."""
    st = _make_storage(da_price=60.0, crm_en_pos=50.0)
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(
        st, _cap_cfg("pos"), [(BLOCK_START, BLOCK_END, None)]
    )
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(0.0)


def test_pos_cap_price_from_arbitrage_spread():
    """
    eom_foresight=2h, block 12:00-14:00. EOM = 50 flat except 90 at 12:00 & 13:00.
    EOM_avg over [10:00,14:00] = (50+50+90+90+50)/5 = 66.
    price window {12:00,13:00}: opp = max(0, 90-66)*2 = 48; energy margin
    (crm_en_pos=50 < 66) = 0 -> net opp = 48.
    """
    st = _make_storage(da_price=50.0, crm_en_pos=50.0)
    st.forecaster.price["EOM"].at[datetime(2023, 7, 1, 12)] = 90.0
    st.forecaster.price["EOM"].at[datetime(2023, 7, 1, 13)] = 90.0
    strategy = StorageCrmPosStrategy(eom_foresight="2h")
    block = (datetime(2023, 7, 1, 12), datetime(2023, 7, 1, 14), None)
    bids = strategy.calculate_bids(st, _cap_cfg("pos"), [block])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(48.0)


def test_pos_cap_energy_signal_beats_spread_floors_to_zero():
    """Same spread (opp 48) but crm_en_pos=100 > EOM_avg(66) -> energy margin
    34*2 = 68 > 48 -> net opp negative -> price floors at 0."""
    st = _make_storage(da_price=50.0, crm_en_pos=100.0)
    st.forecaster.price["EOM"].at[datetime(2023, 7, 1, 12)] = 90.0
    st.forecaster.price["EOM"].at[datetime(2023, 7, 1, 13)] = 90.0
    strategy = StorageCrmPosStrategy(eom_foresight="2h")
    block = (datetime(2023, 7, 1, 12), datetime(2023, 7, 1, 14), None)
    bids = strategy.calculate_bids(st, _cap_cfg("pos"), [block])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Energy legs
# ---------------------------------------------------------------------------


def test_en_pos_voluntary_only_priced_at_signal():
    """No reservation, soc=0.5 -> voluntary = min(power 100, soc 200) = 100 @ signal."""
    st = _make_storage(crm_en_pos=50.0)
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(st, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_en_pos_must_plus_voluntary():
    """cap_pos=30 -> must 30 + voluntary min(70, 170)=70, both at signal."""
    st = _make_storage(crm_en_pos=50.0)
    st.outputs["capacity_pos"][HOUR_START] = 30.0
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(st, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(30.0)
    assert vol["volume"] == pytest.approx(70.0)
    assert must["price"] == pytest.approx(50.0)
    assert vol["price"] == pytest.approx(50.0)


def test_en_pos_marginal_cost_fallback():
    """No CRM energy signal -> price falls back to discharge MC (5)."""
    st = _make_storage(include_crm_signals=False)
    strategy = StorageCrmPosStrategy()
    bids = strategy.calculate_bids(st, _en_cfg("pos"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(MC_DIS)


def test_en_neg_voluntary_only_priced_at_signal():
    """No reservation, soc=0.5 -> voluntary = min(charge 100, headroom 200) = 100 @ signal."""
    st = _make_storage(crm_en_neg=8.0)
    strategy = StorageCrmNegStrategy()
    bids = strategy.calculate_bids(st, _en_cfg("neg"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(8.0)


def test_en_neg_must_plus_voluntary():
    """cap_neg=30 -> must 30 + voluntary min(70, 170)=70, both at signal."""
    st = _make_storage(crm_en_neg=8.0)
    st.outputs["capacity_neg"][HOUR_START] = 30.0
    strategy = StorageCrmNegStrategy()
    bids = strategy.calculate_bids(st, _en_cfg("neg"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 2
    must, vol = bids
    assert must["volume"] == pytest.approx(30.0)
    assert vol["volume"] == pytest.approx(70.0)
    assert must["price"] == pytest.approx(8.0)
    assert vol["price"] == pytest.approx(8.0)


def test_en_neg_marginal_cost_fallback():
    """No CRM energy signal -> price falls back to charge MC (2)."""
    st = _make_storage(include_crm_signals=False)
    strategy = StorageCrmNegStrategy()
    bids = strategy.calculate_bids(st, _en_cfg("neg"), [(HOUR_START, HOUR_END, None)])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(MC_CHG)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_pos_strategy_rejects_neg_product():
    st = _make_storage()
    strategy = StorageCrmPosStrategy()
    bad = _cap_cfg("pos")
    bad.product_type = "capacity_neg"
    with pytest.raises(ValueError, match="capacity_pos/energy_pos"):
        strategy.calculate_bids(st, bad, [(BLOCK_START, BLOCK_END, None)])


def test_neg_strategy_rejects_pos_product():
    st = _make_storage()
    strategy = StorageCrmNegStrategy()
    bad = _en_cfg("neg")
    bad.product_type = "energy_pos"
    with pytest.raises(ValueError, match="capacity_neg/energy_neg"):
        strategy.calculate_bids(st, bad, [(HOUR_START, HOUR_END, None)])
