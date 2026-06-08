# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Interdependent CRM bidding strategies for storage units.

See ``flexable_crm.py`` for the cross-market design overview. As there, the
strategies are split by *direction*: ``StorageCrmPosStrategy`` handles
``capacity_pos`` and ``energy_pos``; ``StorageCrmNegStrategy`` handles
``capacity_neg`` and ``energy_neg``.

Storage specifics:
    - Capacity volume is bounded by both power and a *flat-SOC* sustainability
      check over one dispatch interval (the index timestep): the SOC at block
      start is assumed to hold across the block (conservative, easy to reason
      about), net of any reservation already committed in either direction.
    - Storage marginal cost is ~0, so the EOM opportunity cost uses the
      *arbitrage spread* (price relative to its foresight-window average) rather
      than price-minus-marginal-cost.
"""

from datetime import timedelta

import numpy as np

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class _StorageCrmStrategy(MinMaxChargeStrategy):
    """
    Shared implementation for the directional storage CRM strategies.

    Subclasses fix the direction via ``sign`` (+1 pos / -1 neg), ``direction``,
    ``cap_product``, ``energy_product`` and set ``crm_energy_market_id``.

    Args:
        eom_foresight (str): window for the EOM arbitrage-spread valuation.
            Default "12h".
        eom_market_id (str): forecaster key for the day-ahead price. Default "EOM".
        crm_energy_market_id_pos / _neg (str): forecaster key for the CRM energy
            signal of this direction. Defaults "CRM_energy_pos" / "CRM_energy_neg".
    """

    # set by subclasses
    sign: float = 1.0
    direction: str = "pos"
    cap_product: str = "capacity_pos"
    energy_product: str = "energy_pos"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foresight = parse_duration(kwargs.get("eom_foresight", "12h"))
        self.eom_market_id = kwargs.get("eom_market_id", "EOM")
        # concrete value provided by each subclass
        self.crm_energy_market_id = None

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """Dispatch to the capacity or the energy leg based on product_type."""
        product_type = market_config.product_type
        if product_type == self.cap_product:
            return self._capacity_bids(unit, market_config, product_tuples)
        if product_type == self.energy_product:
            return self._energy_bids(unit, market_config, product_tuples)
        raise ValueError(
            f"{type(self).__name__} supports {self.cap_product}/{self.energy_product}, "
            f"got product_type={product_type!r}"
        )

    @staticmethod
    def _step_hours(unit: SupportsMinMaxCharge) -> float:
        """Length of one dispatch interval in hours (the sustainability horizon)."""
        return unit.index.freq / timedelta(hours=1)

    # ----------------------------------------------------------- capacity leg

    def _capacity_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """One capacity bid per product (= per block)."""
        eom_price = unit.forecaster.price.get(self.eom_market_id)
        energy_price = unit.forecaster.price.get(self.crm_energy_market_id)

        bids: list[dict] = []
        for product in product_tuples:
            block_start, block_end = product[0], product[1]
            block_steps = list(unit.index[block_start : block_end - unit.index.freq])
            if not block_steps:
                continue

            block_volume = self._block_sustainable_volume(unit, block_steps)
            if block_volume <= 0:
                continue

            price = self._capacity_price(unit, block_start, eom_price, energy_price)
            if price >= 0:
                price = min(price, market_config.maximum_bid_price)
            else:
                price = max(price, market_config.minimum_bid_price)

            bids.append(
                {
                    "start_time": block_start,
                    "end_time": block_end,
                    "only_hours": None,
                    "price": price,
                    "volume": block_volume,
                    "node": unit.node,
                }
            )

        return self.remove_empty_bids(bids)

    def _block_sustainable_volume(
        self,
        unit: SupportsMinMaxCharge,
        block_steps: list,
    ) -> float:
        """
        Power- and SOC-feasible volume across the block (flat-SOC approximation),
        net of any reservation already committed in this direction. ``d`` is one
        dispatch interval.
        """
        block_start = block_steps[0]
        d = self._step_hours(unit)
        capacity = unit.capacity
        soc = float(unit.outputs["soc"].at[block_start])

        hourly_max: list[float] = []
        for t in block_steps:
            if self.direction == "pos":
                existing = float(unit.outputs["capacity_pos"].at[t])
                soc_room_mwh = max(0.0, soc - unit.min_soc) * capacity
                soc_limit = soc_room_mwh * unit.efficiency_discharge / d
                power_limit = max(0.0, unit.max_power_discharge - existing)
            else:
                existing = float(unit.outputs["capacity_neg"].at[t])
                soc_headroom_mwh = max(0.0, unit.max_soc - soc) * capacity
                eff_ch = unit.efficiency_charge if unit.efficiency_charge > 0 else 1.0
                soc_limit = soc_headroom_mwh / eff_ch / d
                power_limit = max(0.0, abs(unit.max_power_charge) - existing)
            hourly_max.append(max(0.0, min(soc_limit, power_limit)))

        return float(min(hourly_max)) if hourly_max else 0.0

    def _capacity_price(
        self,
        unit: SupportsMinMaxCharge,
        block_start,
        eom_price,
        energy_price,
    ) -> float:
        """
        Net opportunity cost per MW over the eom_foresight window, using the
        arbitrage spread (storage MC ~ 0):

            Σ_h [ max(0, sign·(EOM_signal[h] − EOM_avg))       # forgone arbitrage
                − max(0, sign·(energy_signal[h] − EOM_avg)) ]   # CRM energy benefit
        """
        if eom_price is None:
            return 0.0

        eom_avg = self._foresight_average(eom_price, block_start)
        # exactly eom_foresight worth of steps (exclusive of the window end)
        window = list(
            unit.index[block_start : block_start + self.foresight - unit.index.freq]
        )
        net = 0.0
        for t in window:
            eom_margin = max(0.0, self.sign * (float(eom_price.at[t]) - eom_avg))
            energy_margin = 0.0
            if energy_price is not None:
                energy_margin = max(
                    0.0, self.sign * (float(energy_price.at[t]) - eom_avg)
                )
            net += eom_margin - energy_margin

        return net

    def _foresight_average(self, price_forecast, current_time) -> float:
        """Average price over [current_time - foresight, current_time + foresight]."""
        start = max(current_time - self.foresight, price_forecast.index[0])
        end = min(current_time + self.foresight, price_forecast.index[-1])
        window = price_forecast.loc[start:end]
        return float(np.mean(window))

    # ------------------------------------------------------------- energy leg

    def _energy_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """Up to two bids per product: must-offer (won capacity) + voluntary."""
        energy_price = unit.forecaster.price.get(market_config.market_id)
        d = self._step_hours(unit)

        bids: list[dict] = []
        for product in product_tuples:
            t_start, t_end = product[0], product[1]
            steps = list(unit.index[t_start : t_end - unit.index.freq])
            if not steps:
                continue
            t = steps[0]

            base_load = float(unit.outputs["energy"].at[t])
            soc = float(unit.outputs["soc"].at[t])
            must_volume = float(unit.outputs[self.cap_product].at[t])
            price = self._energy_price(unit, t, energy_price, market_config)

            # ---- must-offer leg ----
            if must_volume > 0:
                bids.append(
                    {
                        "start_time": t_start,
                        "end_time": t_end,
                        "only_hours": None,
                        "price": price,
                        "volume": must_volume,
                        "node": unit.node,
                    }
                )

            # ---- voluntary leg ----
            vol_volume = self._voluntary_headroom(unit, base_load, soc, must_volume, d)
            if vol_volume > 0:
                bids.append(
                    {
                        "start_time": t_start,
                        "end_time": t_end,
                        "only_hours": None,
                        "price": price,
                        "volume": vol_volume,
                        "node": unit.node,
                    }
                )

        return self.remove_empty_bids(bids)

    def _voluntary_headroom(
        self,
        unit: SupportsMinMaxCharge,
        base_load: float,
        soc: float,
        must_volume: float,
        d: float,
    ) -> float:
        """Power- and SOC-feasible voluntary headroom beyond the must-offer."""
        if self.direction == "pos":
            power_room = unit.max_power_discharge - max(0.0, base_load) - must_volume
            soc_supply = (
                (soc - unit.min_soc) * unit.capacity * unit.efficiency_discharge / d
            )
            soc_room = soc_supply - max(0.0, base_load) - must_volume
        else:
            eff_ch = unit.efficiency_charge if unit.efficiency_charge > 0 else 1.0
            power_room = abs(unit.max_power_charge) - max(0.0, -base_load) - must_volume
            soc_supply = (unit.max_soc - soc) * unit.capacity / eff_ch / d
            soc_room = soc_supply - max(0.0, -base_load) - must_volume

        return max(0.0, min(power_room, soc_room))

    def _energy_price(
        self,
        unit: SupportsMinMaxCharge,
        t,
        energy_price,
        market_config: MarketConfig,
    ) -> float:
        """CRM energy signal at t, clipped to the band; marginal-cost fallback."""
        if energy_price is not None:
            price = float(energy_price.at[t])
        else:
            mc_sign = 1.0 if self.direction == "pos" else -1.0
            price = float(unit.calculate_marginal_cost(t, mc_sign))

        if price >= 0:
            return min(price, market_config.maximum_bid_price)
        return max(price, market_config.minimum_bid_price)


class StorageCrmPosStrategy(_StorageCrmStrategy):
    """Positive-direction storage CRM strategy: ``capacity_pos`` and ``energy_pos``."""

    sign = 1.0
    direction = "pos"
    cap_product = "capacity_pos"
    energy_product = "energy_pos"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crm_energy_market_id = kwargs.get(
            "crm_energy_market_id_pos", "CRM_energy_pos"
        )


class StorageCrmNegStrategy(_StorageCrmStrategy):
    """Negative-direction storage CRM strategy: ``capacity_neg`` and ``energy_neg``."""

    sign = -1.0
    direction = "neg"
    cap_product = "capacity_neg"
    energy_product = "energy_neg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crm_energy_market_id = kwargs.get(
            "crm_energy_market_id_neg", "CRM_energy_neg"
        )
