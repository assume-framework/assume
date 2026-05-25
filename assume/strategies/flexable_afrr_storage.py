# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Interdependent bidding strategies for the German aFRR markets (storage units).

See ``flexable_afrr.py`` for the cross-market design overview.

Storage-specific subtleties:
    - aFRR capacity for storage must be sustainable for some delivery duration
      under SOC limits. We use a configurable ``delivery_duration_hours`` knob
      (default 1h, matching typical aFRR prequalification requirements).
    - Storage marginal cost is just ``additional_cost_charge/discharge``; the
      real economic signal is the arbitrage spread, so we use
      ``(DA - DA_avg)`` over a foresight window as the opportunity-cost signal,
      mirroring StorageEnergyHeuristicFlexableStrategy's charge/discharge rule.
    - For v1 we use a *flat-SOC* approximation: the SOC at the block start is
      assumed to hold through the block. This is conservative and easy to
      reason about; a future version can walk a forecasted EOM dispatch.
"""

from datetime import datetime

import numpy as np

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class StorageAfrrCapBlockStrategy(MinMaxChargeStrategy):
    """
    Bids aFRR capacity as a single (price, volume) per 4h block, SOC-aware.

    Volume = min over the block of:
        - pos: min( max_power_discharge,  (SOC - min_soc) * capacity * eff_dis / d )
        - neg: min(|max_power_charge|,    (max_soc - SOC) * capacity / eff_ch  / d )
      where d = delivery_duration_hours and SOC is the value at block start
      (flat-SOC v1 approximation).

    Price (per MW) = max(
        (opp_cost - p_activation * en_value) / volume,
        capacity_price_floor,
    )
      opp_cost = sum_h max(0,  spread[h]) * V        # pos: DA - DA_avg; neg: DA_avg - DA
      en_value = sum_h signed activation P&L * V     # see _expected_afrr_energy_revenue

    Args:
        eom_foresight: window for DA price averaging (used as opp cost reference).
            Default "12h".
        activation_probability: expected fraction of capacity called as energy.
            Default 0.05.
        capacity_price_floor: lower bound on the bid price. Default 0.
        delivery_duration_hours: sustained delivery requirement. Default 1.0.
        eom_market_id: forecaster key for DA price. Default "EOM".
        afrr_energy_market_id_pos / _neg: forecaster keys for aFRR energy clearing
            prices. Defaults "aFRR_en_pos" / "aFRR_en_neg".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eom_foresight = parse_duration(kwargs.get("eom_foresight", "12h"))
        self.activation_probability = float(kwargs.get("activation_probability", 0.05))
        self.capacity_price_floor = float(kwargs.get("capacity_price_floor", 0.0))
        self.delivery_duration_hours = float(
            kwargs.get("delivery_duration_hours", 1.0)
        )
        self.eom_market_id = kwargs.get("eom_market_id", "EOM")
        self.afrr_energy_market_id_pos = kwargs.get(
            "afrr_energy_market_id_pos", "aFRR_en_pos"
        )
        self.afrr_energy_market_id_neg = kwargs.get(
            "afrr_energy_market_id_neg", "aFRR_en_neg"
        )
        # When True, walk an EOM-dispatch-driven SOC trajectory across the block
        # and check capacity feasibility hour-by-hour. When False (default,
        # backward-compatible), use a flat-SOC approximation.
        self.forecasted_soc_walk = bool(kwargs.get("forecasted_soc_walk", False))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        One bid per product (= per 4h block).
        """
        direction = self._direction_from_product_type(market_config.product_type)
        afrr_en_market_id = (
            self.afrr_energy_market_id_pos
            if direction == "pos"
            else self.afrr_energy_market_id_neg
        )

        da_price = unit.forecaster.price.get(self.eom_market_id)
        afrr_en_price = unit.forecaster.price.get(afrr_en_market_id)

        bids: list[dict] = []

        for product in product_tuples:
            block_start, block_end = product[0], product[1]
            block_steps = list(unit.index[block_start : block_end - unit.index.freq])
            if not block_steps:
                continue

            block_volume = self._block_sustainable_volume(
                unit=unit,
                block_steps=block_steps,
                direction=direction,
                da_price=da_price,
            )
            if block_volume <= 0:
                continue

            opp_cost = self._eom_opportunity_cost(
                unit=unit,
                block_steps=block_steps,
                volume=block_volume,
                da_price=da_price,
                direction=direction,
            )
            en_value = self._expected_afrr_energy_revenue(
                unit=unit,
                block_steps=block_steps,
                volume=block_volume,
                afrr_en_price=afrr_en_price,
                direction=direction,
            )

            net_value = opp_cost - self.activation_probability * en_value
            block_price = max(net_value / block_volume, self.capacity_price_floor)
            block_price = self._clip(block_price, market_config)

            bids.append(
                {
                    "start_time": block_start,
                    "end_time": block_end,
                    "only_hours": None,
                    "price": block_price,
                    "volume": block_volume,
                    "node": unit.node,
                }
            )

        return self.remove_empty_bids(bids)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _direction_from_product_type(product_type: str) -> str:
        if product_type == "capacity_pos":
            return "pos"
        if product_type == "capacity_neg":
            return "neg"
        raise ValueError(
            f"StorageAfrrCapBlockStrategy supports capacity_pos/capacity_neg, "
            f"got product_type={product_type!r}"
        )

    @staticmethod
    def _clip(price: float, market_config: MarketConfig) -> float:
        if price >= 0:
            return min(price, market_config.maximum_bid_price)
        return max(price, market_config.minimum_bid_price)

    def _block_sustainable_volume(
        self,
        unit: SupportsMinMaxCharge,
        block_steps: list,
        direction: str,
        da_price,
    ) -> float:
        """
        Sustainable volume across the 4h block for ``delivery_duration_hours``.

        Two modes (see ``forecasted_soc_walk``):
          - Flat-SOC v1: assumes SOC at block start holds through the block.
            Conservative + easy to reason about.
          - Forecasted SOC walk: forecasts EOM dispatch hour-by-hour using
            the same DA-vs-rolling-average rule as the storage EOM strategy,
            and tracks SOC accordingly. Per-hour capacity is checked at each
            forecasted SOC; the *min* across the block is the bid volume.

        Already-committed ``capacity_pos|neg`` from other markets is netted out
        of the power limit (taken from block start; same value used per hour
        since those reservations apply over the whole block).
        """
        block_start = block_steps[0]
        d = self.delivery_duration_hours
        capacity = unit.capacity
        existing_pos = float(unit.outputs["capacity_pos"].at[block_start])
        existing_neg = float(unit.outputs["capacity_neg"].at[block_start])

        if self.forecasted_soc_walk:
            soc_traj = self._walk_soc_trajectory(unit, block_steps, da_price)
        else:
            soc_start = float(unit.outputs["soc"].at[block_start])
            soc_traj = [soc_start] * len(block_steps)

        hourly_max = []
        for soc in soc_traj:
            if direction == "pos":
                soc_room_mwh = max(0.0, (soc - unit.min_soc)) * capacity
                soc_limit = soc_room_mwh * unit.efficiency_discharge / d
                power_limit = max(0.0, unit.max_power_discharge - existing_pos)
            else:
                soc_headroom_mwh = max(0.0, (unit.max_soc - soc)) * capacity
                eff_ch = unit.efficiency_charge if unit.efficiency_charge > 0 else 1.0
                soc_limit = soc_headroom_mwh / eff_ch / d
                power_limit = max(0.0, abs(unit.max_power_charge) - existing_neg)
            hourly_max.append(max(0.0, min(soc_limit, power_limit)))

        return float(min(hourly_max)) if hourly_max else 0.0

    def _walk_soc_trajectory(
        self,
        unit: SupportsMinMaxCharge,
        block_steps: list,
        da_price,
    ) -> list[float]:
        """
        Forecast SOC at the start of each hour in the block.

        Each hour, compare DA[h] to the rolling DA average over ``±eom_foresight``
        (matching the heuristic in ``StorageEnergyHeuristicFlexableStrategy``):
            DA[h] >= avg  -> storage discharges at max (SOC- and power-bounded)
            DA[h] <  avg  -> storage charges at max
        Update SOC for the next hour accordingly.

        Returns SOC values at the *start* of each hour (length == len(block_steps)).
        If no DA forecast is available, returns a flat trajectory.
        """
        soc = float(unit.outputs["soc"].at[block_steps[0]])
        if da_price is None:
            return [soc] * len(block_steps)

        capacity = unit.capacity
        eff_dis = unit.efficiency_discharge
        eff_ch = unit.efficiency_charge if unit.efficiency_charge > 0 else 1.0

        trajectory: list[float] = []
        for t in block_steps:
            trajectory.append(soc)
            da_h = float(da_price.at[t])
            da_avg = self._foresight_average(da_price, t)

            if da_h >= da_avg:
                # discharge: bounded by power and by SOC at start of this hour
                max_p = min(
                    unit.max_power_discharge,
                    max(0.0, (soc - unit.min_soc)) * capacity * eff_dis,
                )
                soc -= max_p / eff_dis / capacity if max_p > 0 else 0.0
            else:
                # charge
                max_p = min(
                    abs(unit.max_power_charge),
                    max(0.0, (unit.max_soc - soc)) * capacity / eff_ch,
                )
                soc += max_p * eff_ch / capacity if max_p > 0 else 0.0

        return trajectory

    def _eom_opportunity_cost(
        self,
        unit: SupportsMinMaxCharge,
        block_steps,
        volume: float,
        da_price,
        direction: str,
    ) -> float:
        """
        Storage opportunity cost: foregone arbitrage spread relative to DA average.

            pos: max(0,  DA[h] - DA_avg) * V       # storage would discharge when high
            neg: max(0,  DA_avg - DA[h]) * V       # storage would charge when low

        DA_avg is taken over the foresight window centered on the block.
        """
        if da_price is None or volume <= 0:
            return 0.0

        da_avg = self._foresight_average(da_price, block_steps[0])

        opp_cost = 0.0
        for t in block_steps:
            da_h = float(da_price.at[t])
            if direction == "pos":
                opp_cost += max(0.0, da_h - da_avg) * volume
            else:
                opp_cost += max(0.0, da_avg - da_h) * volume
        return opp_cost

    def _expected_afrr_energy_revenue(
        self,
        unit: SupportsMinMaxCharge,
        block_steps,
        volume: float,
        afrr_en_price,
        direction: str,
    ) -> float:
        """
        Expected aFRR energy P&L per fully-activated block (signed).

            pos: (aFRR_en_pos[h] - MC_discharge) * V     # discharge revenue
            neg: (MC_charge - aFRR_en_neg[h]) * V        # symmetric to PP neg

        Storage MC is small (additional_cost_charge/discharge only).
        """
        if afrr_en_price is None or volume <= 0:
            return 0.0

        revenue = 0.0
        for t in block_steps:
            if direction == "pos":
                mc = float(unit.calculate_marginal_cost(t, volume))
                revenue += (float(afrr_en_price.at[t]) - mc) * volume
            else:
                mc = float(unit.calculate_marginal_cost(t, -volume))
                revenue += (mc - float(afrr_en_price.at[t])) * volume
        return revenue

    def _foresight_average(self, price_forecast, current_time) -> float:
        """Average of price_forecast over [current_time - foresight, current_time + foresight]."""
        start = max(current_time - self.eom_foresight, price_forecast.index[0])
        end = min(current_time + self.eom_foresight, price_forecast.index[-1])
        window = price_forecast.loc[start:end]
        return float(np.mean(window))


class StorageAfrrEnergyStrategy(MinMaxChargeStrategy):
    """
    Bids aFRR energy for storage with must-offer + voluntary legs.

    For each hourly product:
        a) MUST-OFFER: volume = capacity_{pos|neg} already reserved by aFRR-cap.
           Price = MC_discharge (pos) or MC_charge (neg) -- break-even on the
           additional operational cost. For v1 we do NOT add an
           opportunity-cost-of-SOC term to the must-offer price; that is a
           future enhancement.
        b) VOLUNTARY: remaining bid on power- and SOC-feasible headroom beyond
           the must-offer. Priced at MC + voluntary_markup (pos) or
           MC - voluntary_discount (neg).

    By aFRR-energy clearing time, both aFRR-cap and EOM have already cleared,
    so ``outputs["energy"]``, ``outputs["capacity_pos|neg"]`` and
    ``outputs["soc"]`` are real values.

    Voluntary headroom is computed manually (not via
    ``calculate_min_max_charge/discharge``) because those helpers were designed
    for the "first-to-write-capacity" case and over-state headroom when
    capacity_neg/pos is already populated.

    Args:
        voluntary_markup: EUR/MWh added to MC for voluntary pos bids. Default 0.
        voluntary_discount: EUR/MWh subtracted from MC for voluntary neg bids.
            Default 0.
        delivery_duration_hours: how long voluntary activation must be sustainable
            from current SOC. Default 1.0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voluntary_markup = float(kwargs.get("voluntary_markup", 0.0))
        self.voluntary_discount = float(kwargs.get("voluntary_discount", 0.0))
        self.delivery_duration_hours = float(
            kwargs.get("delivery_duration_hours", 1.0)
        )

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Up to two bids per hourly product (must-offer + voluntary).
        product_type in {"energy_pos", "energy_neg"}.
        """
        direction = self._direction_from_product_type(market_config.product_type)
        cap_key = f"capacity_{direction}"
        d = self.delivery_duration_hours

        bids: list[dict] = []

        for product in product_tuples:
            t_start, t_end = product[0], product[1]
            steps = list(unit.index[t_start : t_end - unit.index.freq])
            if not steps:
                continue
            t = steps[0]

            base_load = float(unit.outputs["energy"].at[t])
            soc = float(unit.outputs["soc"].at[t])
            must_volume = float(unit.outputs[cap_key].at[t])

            # ---- must-offer leg ----
            if must_volume > 0:
                must_price = self._marginal(unit, t, direction)
                must_price = self._clip(must_price, market_config)
                bids.append(
                    {
                        "start_time": t_start,
                        "end_time": t_end,
                        "only_hours": None,
                        "price": must_price,
                        "volume": must_volume,
                        "node": unit.node,
                    }
                )

            # ---- voluntary leg ----
            vol_volume = self._voluntary_headroom(
                unit=unit,
                direction=direction,
                base_load=base_load,
                soc=soc,
                must_volume=must_volume,
                d=d,
            )
            if vol_volume > 0:
                vol_mc = self._marginal(unit, t, direction)
                vol_price = (
                    vol_mc + self.voluntary_markup
                    if direction == "pos"
                    else vol_mc - self.voluntary_discount
                )
                vol_price = self._clip(vol_price, market_config)
                bids.append(
                    {
                        "start_time": t_start,
                        "end_time": t_end,
                        "only_hours": None,
                        "price": vol_price,
                        "volume": vol_volume,
                        "node": unit.node,
                    }
                )

        return self.remove_empty_bids(bids)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _direction_from_product_type(product_type: str) -> str:
        if product_type == "energy_pos":
            return "pos"
        if product_type == "energy_neg":
            return "neg"
        raise ValueError(
            f"StorageAfrrEnergyStrategy supports energy_pos/energy_neg, "
            f"got product_type={product_type!r}"
        )

    @staticmethod
    def _clip(price: float, market_config: MarketConfig) -> float:
        if price >= 0:
            return min(price, market_config.maximum_bid_price)
        return max(price, market_config.minimum_bid_price)

    @staticmethod
    def _marginal(unit: SupportsMinMaxCharge, t, direction: str) -> float:
        """Storage MC is additional_cost_discharge (pos) or additional_cost_charge (neg)."""
        sign = 1.0 if direction == "pos" else -1.0
        return float(unit.calculate_marginal_cost(t, sign))

    def _voluntary_headroom(
        self,
        unit: SupportsMinMaxCharge,
        direction: str,
        base_load: float,
        soc: float,
        must_volume: float,
        d: float,
    ) -> float:
        """
        Voluntary headroom = additional V beyond must_volume that is both
        power-feasible (won't exceed physical limits given EOM dispatch) and
        SOC-sustainable for ``delivery_duration_hours``.

        pos (discharge):
            power_room = max_pd - max(0, base) - must
            soc_room   = (SOC - min_soc) * capacity * eff_dis / d - max(0, base) - must

        neg (charge, magnitudes throughout):
            power_room = |max_pc| + min(0, base) - must     # equiv to |max_pc| - max(0, -base) - must
            soc_room   = (max_soc - SOC) * capacity / eff_ch / d - max(0, -base) - must
        """
        if direction == "pos":
            power_room = unit.max_power_discharge - max(0.0, base_load) - must_volume
            soc_supply = (soc - unit.min_soc) * unit.capacity * unit.efficiency_discharge / d
            soc_room = soc_supply - max(0.0, base_load) - must_volume
        else:
            eff_ch = unit.efficiency_charge if unit.efficiency_charge > 0 else 1.0
            power_room = abs(unit.max_power_charge) - max(0.0, -base_load) - must_volume
            soc_supply = (unit.max_soc - soc) * unit.capacity / eff_ch / d
            soc_room = soc_supply - max(0.0, -base_load) - must_volume

        return max(0.0, min(power_room, soc_room))
