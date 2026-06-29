# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for the intraday-continuous (IDC) position-adjustment strategies.

The strategies bid the delta between the desired position (given the updated
intraday forecast / IDC price) and the committed day-ahead position, priced at
the IDC price signal: positive delta = supply (sell more), negative = demand
(buy back).

Thermal fixture marginal cost: fuel 10 / eff 0.5 + co2 10 * emission 0.5 / 0.5 +
additional 10 = 40 EUR/MWh.
"""

from datetime import datetime

import pandas as pd
import pytest

from assume.common.forecaster import PowerplantForecaster, UnitForecaster
from assume.strategies.flexable_intraday import (
    EnergyIntradayAdjustmentStrategy,
    StorageEnergyIntradayAdjustmentStrategy,
)
from assume.units import PowerPlant, Storage

MC = 40.0
MAX_P = 1000.0
MIN_P = 200.0
RAMP = 500.0

T0 = datetime(2023, 7, 1, 0)
T1 = datetime(2023, 7, 1, 1)


class _CfgStub:
    maximum_bid_price = 9999.0
    minimum_bid_price = -9999.0
    additional_fields = []


def _idc_cfg():
    cfg = _CfgStub()
    cfg.product_type = "energy"
    cfg.market_id = "IDC"
    return cfg


def _make_thermal(price_idc=50.0, availability_intraday=1.0, with_idc=True, ramp=RAMP):
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    market_prices = {"EOM": 45.0}
    if with_idc:
        market_prices["IDC"] = price_idc
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={"lignite": 10, "co2": 10},
        market_prices=market_prices,
        availability_intraday=availability_intraday,
    )
    return PowerPlant(
        id="pp",
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
        ramp_up=ramp,
        ramp_down=ramp,
    )


def _make_renewable(price_idc=50.0, availability_intraday=0.5):
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    forecaster = PowerplantForecaster(
        index,
        fuel_prices={},  # no fuel price -> marginal cost 0
        market_prices={"EOM": 45.0, "IDC": price_idc},
        availability_intraday=availability_intraday,
    )
    return PowerPlant(
        id="wind",
        unit_operator="op",
        technology="wind",
        index=forecaster.index,
        max_power=MAX_P,
        min_power=0,
        efficiency=1.0,
        additional_cost=0,
        bidding_strategies={},
        fuel_type="renewable",
        emission_factor=0,
        forecaster=forecaster,
    )


# ---------------------------------------------------------------------------
# Thermal (price arbitrage)
# ---------------------------------------------------------------------------


def test_thermal_sell_more_when_idc_above_mc():
    """committed=300, IDC=50>MC=40, ramp_up=500 -> desired=800 -> supply +500 @ 50."""
    pp = _make_thermal(price_idc=50.0)
    pp.outputs["energy"][T0] = 300.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        pp, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(500.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_thermal_buy_back_when_idc_below_mc():
    """committed=300, IDC=30<MC=40 -> target 0, ramp_down=500 -> desired=0 -> demand -300 @ 30."""
    pp = _make_thermal(price_idc=30.0)
    pp.outputs["energy"][T0] = 300.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        pp, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(-300.0)
    assert bids[0]["price"] == pytest.approx(30.0)


def test_thermal_no_trade_when_at_target():
    """committed=max_avail=1000, IDC>=MC -> desired=1000 -> delta 0 -> no bid."""
    pp = _make_thermal(price_idc=50.0)
    pp.outputs["energy"][T0] = MAX_P
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        pp, _idc_cfg(), [(T0, T1, None)]
    )
    assert bids == []


def test_thermal_price_clipped_to_market_max():
    pp = _make_thermal(price_idc=20_000.0)
    pp.outputs["energy"][T0] = 300.0
    cfg = _idc_cfg()
    cfg.maximum_bid_price = 3000.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(pp, cfg, [(T0, T1, None)])
    assert len(bids) == 1
    assert bids[0]["price"] == pytest.approx(3000.0)


def test_no_idc_signal_no_bids():
    pp = _make_thermal(with_idc=False)
    pp.outputs["energy"][T0] = 300.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        pp, _idc_cfg(), [(T0, T1, None)]
    )
    assert bids == []


# ---------------------------------------------------------------------------
# Renewable (forecast-error driven)
# ---------------------------------------------------------------------------


def test_renewable_sells_extra_infeed():
    """committed=200, actual infeed=0.5*1000=500, IDC=50>=0 -> supply +300 @ 50."""
    wind = _make_renewable(price_idc=50.0, availability_intraday=0.5)
    wind.outputs["energy"][T0] = 200.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        wind, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(300.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_renewable_buys_back_on_overforecast():
    """committed=200, actual infeed=0.1*1000=100 -> demand -100 @ 50."""
    wind = _make_renewable(price_idc=50.0, availability_intraday=0.1)
    wind.outputs["energy"][T0] = 200.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        wind, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(-100.0)
    assert bids[0]["price"] == pytest.approx(50.0)


def test_renewable_curtails_at_negative_price():
    """IDC=-5 < MC=0 -> target 0 -> demand -200 @ -5 (within band)."""
    wind = _make_renewable(price_idc=-5.0, availability_intraday=0.5)
    wind.outputs["energy"][T0] = 200.0
    bids = EnergyIntradayAdjustmentStrategy().calculate_bids(
        wind, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(-200.0)
    assert bids[0]["price"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Storage (IDC vs day-ahead reference)
# ---------------------------------------------------------------------------

CAPACITY = 400.0
MAX_PD = 100.0
MAX_PC = -100.0


def _make_storage(price_idc=60.0, eom=50.0, initial_soc=0.5, with_eom=True):
    index = pd.date_range("2023-07-01", periods=24, freq="h")
    market_prices = {"IDC": price_idc}
    if with_eom:
        market_prices["EOM"] = eom
    forecaster = UnitForecaster(index, market_prices=market_prices)
    return Storage(
        id="batt",
        unit_operator="op",
        technology="battery",
        bidding_strategies={},
        forecaster=forecaster,
        max_power_charge=MAX_PC,
        max_power_discharge=MAX_PD,
        capacity=CAPACITY,
        initial_soc=initial_soc,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        additional_cost_charge=2,
        additional_cost_discharge=5,
    )


def test_storage_discharges_more_when_idc_above_reference():
    """IDC=60 >= EOM=50, soc=0.5 -> desired=+100 (max discharge), committed 0 -> +100 @ 60."""
    st = _make_storage(price_idc=60.0, eom=50.0)
    bids = StorageEnergyIntradayAdjustmentStrategy().calculate_bids(
        st, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)
    assert bids[0]["price"] == pytest.approx(60.0)


def test_storage_charges_when_idc_below_reference():
    """IDC=40 < EOM=50 -> desired=-100 (max charge), committed 0 -> demand -100 @ 40."""
    st = _make_storage(price_idc=40.0, eom=50.0)
    bids = StorageEnergyIntradayAdjustmentStrategy().calculate_bids(
        st, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(-100.0)
    assert bids[0]["price"] == pytest.approx(40.0)


def test_storage_flips_from_committed_discharge():
    """committed=+50 discharge, IDC=40<EOM=50 -> desired=-100 -> delta=-150 @ 40."""
    st = _make_storage(price_idc=40.0, eom=50.0)
    st.outputs["energy"][T0] = 50.0
    bids = StorageEnergyIntradayAdjustmentStrategy().calculate_bids(
        st, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(-150.0)


def test_storage_soc_bounds_discharge():
    """soc=0.1 -> SOC-limited discharge 40 (< 100 power) -> +40 @ 60."""
    st = _make_storage(price_idc=60.0, eom=50.0, initial_soc=0.1)
    bids = StorageEnergyIntradayAdjustmentStrategy().calculate_bids(
        st, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(40.0)


def test_storage_no_eom_reference_uses_idc():
    """No EOM ref -> reference = IDC -> price>=ref -> discharge +100."""
    st = _make_storage(price_idc=60.0, with_eom=False)
    bids = StorageEnergyIntradayAdjustmentStrategy().calculate_bids(
        st, _idc_cfg(), [(T0, T1, None)]
    )
    assert len(bids) == 1
    assert bids[0]["volume"] == pytest.approx(100.0)


def test_storage_no_idc_signal_no_bids():
    st = _make_storage(price_idc=60.0)
    # remove the IDC signal
    del st.forecaster.price["IDC"]
    bids = StorageEnergyIntradayAdjustmentStrategy().calculate_bids(
        st, _idc_cfg(), [(T0, T1, None)]
    )
    assert bids == []
