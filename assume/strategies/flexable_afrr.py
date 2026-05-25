# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Interdependent bidding strategies for the German aFRR markets (power plants).

Sequence of markets (German design, clearing chronologically):
    1. aFRR capacity (4h block, opens day-ahead before EOM)  -> PowerPlantAfrrCapBlockStrategy
    2. EOM / Day-ahead (hourly)                              -> existing EnergyHeuristicFlexableStrategy
    3. aFRR energy (hourly, must-offer for cap winners)      -> PowerPlantAfrrEnergyStrategy

Interdependence is realized via three mechanisms:
    - aFRR cap awards are written to ``unit.outputs["capacity_pos"|"capacity_neg"]``
      by the unit operator after market clearing; ``PowerPlant.calculate_min_max_power``
      then honors that reservation automatically when the EOM strategy runs next.
    - The aFRR energy strategy reads the same ``capacity_pos|neg`` series to
      compute the *must-offer* portion of its bid.
    - Each strategy uses cross-market forecasts (``forecaster.price[market_id]``)
      to value opportunity cost across the three markets.
"""

from datetime import datetime

import numpy as np

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class PowerPlantAfrrCapBlockStrategy(MinMaxStrategy):
    """
    Bids aFRR capacity as a single (price, volume) per 4h block.

    Pricing (heuristic, per block b of length H hours):
        opp_cost_b  = sum_{h in b} max(0, sign * (DA_forecast[h] - MC[h])) * V
        en_value_b  = sum_{h in b} sign * (aFRR_en_forecast[h] - MC[h]) * V
        capacity_price_b = max( (opp_cost_b - p_activation * en_value_b) / V, floor )

    where:
        sign = +1 for capacity_pos (we lose upward EOM profit when reserving headroom)
        sign = -1 for capacity_neg (we incur must-run cost when MC > DA)
        V    = minimum hour-by-hour deliverable volume across the block

    Args:
        eom_foresight (str): foresight for the existing flexable utilities; kept for
            compatibility. Default "12h".
        activation_probability (float): expected fraction of capacity called as energy.
            Default 0.05.
        capacity_price_floor (float): lower bound for the capacity bid price. Default 0.
        eom_market_id (str): forecaster key for day-ahead price. Default "EOM".
        afrr_energy_market_id_pos (str): forecaster key for aFRR positive energy clearing
            price. Default "aFRR_en_pos".
        afrr_energy_market_id_neg (str): forecaster key for aFRR negative energy clearing
            price. Default "aFRR_en_neg".
        forecast_eom_dispatch (bool): for capacity_neg only. When True, treat the unit as
            running at max_power whenever the DA forecast exceeds marginal cost. This is
            a proxy for the EOM dispatch decision that has not happened yet when aFRR cap
            clears first. Default True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eom_foresight = parse_duration(kwargs.get("eom_foresight", "12h"))
        self.activation_probability = float(kwargs.get("activation_probability", 0.05))
        self.capacity_price_floor = float(kwargs.get("capacity_price_floor", 0.0))
        self.eom_market_id = kwargs.get("eom_market_id", "EOM")
        self.afrr_energy_market_id_pos = kwargs.get(
            "afrr_energy_market_id_pos", "aFRR_en_pos"
        )
        self.afrr_energy_market_id_neg = kwargs.get(
            "afrr_energy_market_id_neg", "aFRR_en_neg"
        )
        self.forecast_eom_dispatch = bool(kwargs.get("forecast_eom_dispatch", True))

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        One bid per product (= per 4h block).

        Returns an empty orderbook if no block can be served.
        """
        direction = self._direction_from_product_type(market_config.product_type)
        afrr_en_market_id = (
            self.afrr_energy_market_id_pos
            if direction == "pos"
            else self.afrr_energy_market_id_neg
        )

        da_price = unit.forecaster.price.get(self.eom_market_id)
        afrr_en_price = unit.forecaster.price.get(afrr_en_market_id)

        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        min_p_full, max_p_full = unit.calculate_min_max_power(start, end)

        previous_power = unit.get_output_before(start)
        idx_offset = 0
        bids: list[dict] = []

        for product in product_tuples:
            block_start, block_end = product[0], product[1]
            block_steps = list(unit.index[block_start : block_end - unit.index.freq])
            n = len(block_steps)
            if n == 0:
                continue

            block_min_p = min_p_full[idx_offset : idx_offset + n]
            block_max_p = max_p_full[idx_offset : idx_offset + n]
            idx_offset += n

            block_volume, previous_power = self._block_feasible_volume(
                unit=unit,
                block_steps=block_steps,
                block_min_p=block_min_p,
                block_max_p=block_max_p,
                direction=direction,
                previous_power=previous_power,
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

            # clip to market price band
            if block_price >= 0:
                block_price = min(block_price, market_config.maximum_bid_price)
            else:
                block_price = max(block_price, market_config.minimum_bid_price)

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
            f"PowerPlantAfrrCapBlockStrategy supports capacity_pos/capacity_neg, "
            f"got product_type={product_type!r}"
        )

    def _eom_dispatch_proxy(self, unit, t, da_price) -> float:
        """
        Heuristic forecast of EOM dispatch at time t when aFRR cap clears before EOM.

        Used only for capacity_neg, where we need *some* assumed upper bound on the
        unit's headroom above min_power. If the DA price forecast exceeds the
        unit's marginal cost at max_power, we assume the unit will clear EOM at
        availability * max_power; otherwise zero.
        """
        if da_price is None or not self.forecast_eom_dispatch:
            return float(unit.outputs["energy"].at[t])
        try:
            availability = float(unit.forecaster.availability.at[t])
        except (AttributeError, KeyError):
            availability = 1.0
        max_p = availability * unit.max_power
        if max_p <= 0:
            return 0.0
        mc = unit.calculate_marginal_cost(t, max_p)
        return max_p if float(da_price.at[t]) >= mc else 0.0

    def _block_feasible_volume(
        self,
        unit,
        block_steps,
        block_min_p,
        block_max_p,
        direction: str,
        previous_power: float,
        da_price,
    ) -> tuple[float, float]:
        """
        Smallest hour-by-hour deliverable volume across the block (a block bid must
        clear in every hour). Returns (block_volume, previous_power_after_block).

        - pos: ramp- and availability-feasible upward headroom relative to the
          unit's currently planned dispatch at each hour.
        - neg: ramp-feasible downward room from the unit's assumed operating point
          at each hour, down to the technical minimum. When the EOM dispatch is not
          yet known (aFRR cap clears first), an EOM dispatch proxy is used.
        """
        hourly_volumes: list[float] = []
        local_prev = float(previous_power)

        for i, t in enumerate(block_steps):
            op_time = unit.get_operation_time(t)
            max_p_h = float(block_max_p[i])
            min_p_h = float(block_min_p[i])

            if direction == "pos":
                current_power = float(unit.outputs["energy"].at[t])
                feas = unit.calculate_ramp(
                    op_time, local_prev, max_p_h, current_power
                )
                # capacity reservation does not change actual dispatch
                next_prev = current_power
            else:
                # neg: need an operating point we can ramp down FROM. The unit
                # must stay above its absolute minimum at all times: the technical
                # min plus any neg capacity already committed elsewhere, and at
                # least heat demand if it's a CHP. We do NOT use min_p_h from
                # calculate_min_max_power, which is the "additional min above
                # base_load" and clips to 0 when base_load already covers it.
                current_power = self._eom_dispatch_proxy(unit, t, da_price)
                absolute_min = max(
                    unit.min_power + float(unit.outputs["capacity_neg"].at[t]),
                    float(unit.outputs["heat"].at[t]),
                )
                min_ramped = unit.calculate_ramp(
                    op_time, local_prev, absolute_min, current_power
                )
                feas = current_power - min_ramped
                next_prev = current_power

            hourly_volumes.append(max(0.0, float(feas)))
            local_prev = next_prev

        block_volume = float(min(hourly_volumes)) if hourly_volumes else 0.0
        return block_volume, local_prev

    def _eom_opportunity_cost(
        self,
        unit,
        block_steps,
        volume: float,
        da_price,
        direction: str,
    ) -> float:
        """
        Forgone EOM profit if `volume` MW of capacity is reserved for aFRR across
        the block.

            pos: unit gives up V MW of upward EOM volume when DA > MC
            neg: unit must run V MW above min_power; pays (MC - DA) per MW when MC > DA
        """
        if da_price is None or volume <= 0:
            return 0.0

        opp_cost = 0.0
        for t in block_steps:
            base_power = float(unit.outputs["energy"].at[t])
            if direction == "pos":
                mc_point = base_power + volume
            else:
                # for neg: MC evaluated at the operating point we'd ramp down from
                # (proxy for the assumed EOM dispatch)
                mc_point = max(volume, base_power)
            mc = float(unit.calculate_marginal_cost(t, mc_point))
            da_p = float(da_price.at[t])

            if direction == "pos":
                opp_cost += max(0.0, da_p - mc) * volume
            else:
                opp_cost += max(0.0, mc - da_p) * volume

        return opp_cost

    def _expected_afrr_energy_revenue(
        self,
        unit,
        block_steps,
        volume: float,
        afrr_en_price,
        direction: str,
    ) -> float:
        """
        Expected aFRR energy P&L *per fully-activated block* (signed value).

            pos: (aFRR_en_clearing - MC) * V        -- positive when activation is profitable
            neg: (MC - aFRR_en_clearing) * V        -- positive when MC > aFRR_en bid

        The activation probability is applied by the caller, not here.
        """
        if afrr_en_price is None or volume <= 0:
            return 0.0

        revenue = 0.0
        for t in block_steps:
            base_power = float(unit.outputs["energy"].at[t])
            mc_point = (
                base_power + volume if direction == "pos" else max(volume, base_power)
            )
            mc = float(unit.calculate_marginal_cost(t, mc_point))
            p = float(afrr_en_price.at[t])

            if direction == "pos":
                revenue += (p - mc) * volume
            else:
                revenue += (mc - p) * volume

        return revenue


class PowerPlantAfrrEnergyStrategy(MinMaxStrategy):
    """
    Bids aFRR energy with two components per hourly product:

        a) MUST-OFFER: volume already reserved via aFRR cap (read from
           ``unit.outputs["capacity_pos"|"capacity_neg"]``). Priced at the
           break-even marginal cost. The unit is contractually obliged to
           make this offer; pricing reflects no premium.

        b) VOLUNTARY: additional energy bid on the ramp-feasible headroom that
           remains after the must-offer. Priced at marginal_cost + voluntary_markup
           (pos) or marginal_cost - voluntary_discount (neg).

    A unit that did NOT participate in aFRR capacity can still bid voluntary
    energy; in that case there is no must-offer leg.

    By aFRR-energy clearing time, both aFRR cap and EOM have already cleared,
    so ``outputs["energy"]`` and ``outputs["capacity_pos|neg"]`` are real values
    (not forecasts).

    Args:
        voluntary_markup: EUR/MWh added to MC for voluntary pos bids. Default 0.
        voluntary_discount: EUR/MWh subtracted from MC for voluntary neg bids.
            Bidding below MC means the unit pays more than its saved fuel cost
            to be more competitive in the auction. Default 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voluntary_markup = float(kwargs.get("voluntary_markup", 0.0))
        self.voluntary_discount = float(kwargs.get("voluntary_discount", 0.0))

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        For each hourly product, emit up to two bids:
            - must-offer bid (volume = capacity_pos|neg, price = MC)
            - voluntary bid (volume = remaining ramp-feasible headroom, price = MC ± markup)

        product_type is either "energy_pos" or "energy_neg".
        """
        direction = self._direction_from_product_type(market_config.product_type)
        cap_key = f"capacity_{direction}"

        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        min_p_full, max_p_full = unit.calculate_min_max_power(start, end)

        previous_power = unit.get_output_before(start)
        bids: list[dict] = []
        idx_offset = 0

        for product in product_tuples:
            t_start, t_end = product[0], product[1]
            steps = list(unit.index[t_start : t_end - unit.index.freq])
            n = len(steps)
            if n == 0:
                continue

            # we treat the product as 1 timestep wide (standard hourly aFRR energy);
            # if the product spans multiple steps the must-offer + voluntary use the
            # first step's state as the reference point
            t = steps[0]
            base_power = float(unit.outputs["energy"].at[t])
            op_time = unit.get_operation_time(t)
            max_p_h = float(max_p_full[idx_offset])
            idx_offset += n

            must_volume = float(unit.outputs[cap_key].at[t])

            # ---- must-offer leg ----
            if must_volume > 0:
                must_price = self._marginal_at(
                    unit, t, base_power, must_volume, direction
                )
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
                t=t,
                direction=direction,
                base_power=base_power,
                must_volume=must_volume,
                max_p_h=max_p_h,
                op_time=op_time,
                previous_power=previous_power,
            )
            if vol_volume > 0:
                vol_mc = self._marginal_at(
                    unit, t, base_power, must_volume + vol_volume, direction
                )
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

            previous_power = base_power

        return self.remove_empty_bids(bids)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _direction_from_product_type(product_type: str) -> str:
        if product_type == "energy_pos":
            return "pos"
        if product_type == "energy_neg":
            return "neg"
        raise ValueError(
            f"PowerPlantAfrrEnergyStrategy supports energy_pos/energy_neg, "
            f"got product_type={product_type!r}"
        )

    @staticmethod
    def _clip(price: float, market_config: MarketConfig) -> float:
        if price >= 0:
            return min(price, market_config.maximum_bid_price)
        return max(price, market_config.minimum_bid_price)

    @staticmethod
    def _marginal_at(unit, t, base_power, delta, direction: str) -> float:
        """
        MC at the post-activation operating point: base+delta for pos,
        base-delta for neg (clipped at 0).
        """
        if direction == "pos":
            point = base_power + delta
        else:
            point = max(0.0, base_power - delta)
        return float(unit.calculate_marginal_cost(t, point))

    def _voluntary_headroom(
        self,
        unit,
        t,
        direction: str,
        base_power: float,
        must_volume: float,
        max_p_h: float,
        op_time: int,
        previous_power: float,
    ) -> float:
        """
        Ramp-feasible voluntary headroom *beyond* the must-offer.

        - pos: max_p_h from calculate_min_max_power already subtracts capacity_pos
          and base_load, so it directly represents the additional voluntary
          headroom above the must-offer. We just apply the ramp constraint.
        - neg: total ramp-down room = base_power - absolute_floor, where
          absolute_floor = max(unit.min_power, heat_demand). Voluntary part is
          that minus the must-offer volume.
        """
        if direction == "pos":
            ramp_feas = unit.calculate_ramp(op_time, previous_power, max_p_h, base_power)
            return max(0.0, float(ramp_feas))

        # neg
        absolute_floor = max(
            unit.min_power, float(unit.outputs["heat"].at[t])
        )
        ramp_floor = unit.calculate_ramp(
            op_time, previous_power, absolute_floor, base_power
        )
        total_room = max(0.0, float(base_power) - float(ramp_floor))
        return max(0.0, total_room - must_volume)


# ---------------------------------------------------------------------------
# Notes on the linked EOM strategy
# ---------------------------------------------------------------------------
# The existing `EnergyHeuristicFlexableStrategy` (flexable.py) already reads
# `capacity_pos|neg` via `calculate_min_max_power`, so EOM headroom is correct
# once aFRR cap has cleared.
#
# In addition, `calculate_EOM_price_if_on` in flexable.py now skips its
# "avoid shutdown" price reduction whenever the unit has a capacity_pos or
# capacity_neg commitment at the bidding hour. The reduction's rationale
# (cheap-bid to avoid restart) does not apply when the unit is contractually
# required to stay online via an aFRR/CRM commitment.
