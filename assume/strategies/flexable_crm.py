# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Interdependent bidding strategies for the German Control Reserve Market (CRM),
power plants. The CRM trades a *capacity* product and an *energy* product, each
in a *positive* and a *negative* direction (the German aFRR design).

Market sequence (clearing chronologically):
    1. CRM capacity (4h block, opens day-ahead before EOM)
    2. EOM / Day-ahead (hourly, energy only)
    3. CRM energy (hourly; must-offer for capacity winners + voluntary)

A unit that wins CRM capacity is obliged to offer the won volume on the matching
CRM energy market (must-offer); a unit may also offer voluntary CRM energy
without holding capacity.

Architecture (mirrors the stock ``CapacityHeuristicBalancingPos/NegStrategy``):
one class per *direction*. ``PowerPlantCrmPosStrategy`` handles ``capacity_pos``
and ``energy_pos``; ``PowerPlantCrmNegStrategy`` handles ``capacity_neg`` and
``energy_neg``. The direction (sign, which ``outputs["capacity_pos"|"capacity_neg"]``
to read, which energy price signal to use) is fixed by the class; the class
switches on ``market_config.product_type`` to pick the capacity or the energy leg.

Interdependence is realized via ``unit.outputs["capacity_pos"|"capacity_neg"]``,
written by the unit operator after capacity clearing. ``calculate_min_max_power``
then nets the reservation out of EOM headroom, ``calculate_ramp`` keeps a
committed unit online, and the energy leg reads the same series for the
must-offer volume.

Pricing:
    - Capacity = NET opportunity cost: forgone EOM margin minus the CRM energy
      activation margin the unit would also earn, both read from forecaster price
      signals over ``eom_foresight``. The unit reserves capacity only when that
      beats EOM; partial reservation and the EOM fallback happen automatically via
      market clearing against the demand units plus the capacity netting above.
    - Energy = must-offer + voluntary, both priced at the forecasted CRM energy
      signal (with a marginal-cost fallback if no signal is provided).
"""

from assume.common.base import MinMaxStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class _PowerPlantCrmStrategy(MinMaxStrategy):
    """
    Shared implementation for the directional power-plant CRM strategies.

    Subclasses fix the direction via the class attributes ``sign`` (+1 pos / -1
    neg), ``direction``, ``cap_product``, ``energy_product`` and set
    ``crm_energy_market_id`` in ``__init__``.

    Args:
        eom_foresight (str): window for the EOM opportunity-cost valuation.
            Default "12h".
        eom_market_id (str): forecaster key for the day-ahead price. Default "EOM".
        crm_energy_market_id_pos / _neg (str): forecaster key for the CRM energy
            clearing-price signal of this direction. Defaults "CRM_energy_pos" /
            "CRM_energy_neg".
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
        unit: SupportsMinMax,
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

    # ----------------------------------------------------------- capacity leg

    def _capacity_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """One capacity bid per product (= per block)."""
        eom_price = unit.forecaster.price.get(self.eom_market_id)
        energy_price = unit.forecaster.price.get(self.crm_energy_market_id)

        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        _, max_p_full = unit.calculate_min_max_power(start, end)

        previous_power = unit.get_output_before(start)
        idx_offset = 0
        bids: list[dict] = []

        for product in product_tuples:
            block_start, block_end = product[0], product[1]
            block_steps = list(unit.index[block_start : block_end - unit.index.freq])
            n = len(block_steps)
            if n == 0:
                continue

            block_max_p = max_p_full[idx_offset : idx_offset + n]
            idx_offset += n

            block_volume, previous_power = self._block_feasible_volume(
                unit, block_steps, block_max_p, previous_power
            )
            if block_volume <= 0:
                continue

            price = self._capacity_price(
                unit, block_start, block_volume, eom_price, energy_price
            )

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

    def _block_feasible_volume(
        self,
        unit: SupportsMinMax,
        block_steps: list,
        block_max_p,
        previous_power: float,
    ) -> tuple[float, float]:
        """
        Smallest hour-by-hour deliverable volume across the block (a block bid
        must clear in every hour), net of any reservation already committed in the
        other direction. Returns (block_volume, previous_power_after_block).

        Uses the actual ``outputs["energy"]`` operating point (no EOM-dispatch
        proxy): like the stock CRM strategies the unit offers technical headroom.
        """
        hourly_volumes: list[float] = []
        local_prev = float(previous_power)

        for i, t in enumerate(block_steps):
            op_time = unit.get_operation_time(t)
            current_power = float(unit.outputs["energy"].at[t])

            if self.direction == "pos":
                max_p_h = float(block_max_p[i])
                feas = unit.calculate_ramp(
                    op_time, local_prev, max_p_h, current_power, start=t
                )
            else:
                # downward room from the current operating point to the absolute
                # minimum (technical min + neg reservation, at least heat demand)
                absolute_min = max(
                    unit.min_power + float(unit.outputs["capacity_neg"].at[t]),
                    float(unit.outputs["heat"].at[t]),
                )
                min_ramped = unit.calculate_ramp(
                    op_time, local_prev, absolute_min, current_power, start=t
                )
                feas = current_power - min_ramped

            # combined reservation cannot exceed the technical flexibility range
            cap_pos = float(unit.outputs["capacity_pos"].at[t])
            cap_neg = float(unit.outputs["capacity_neg"].at[t])
            cross_dir_room = max(
                0.0, (unit.max_power - unit.min_power) - cap_pos - cap_neg
            )
            feas = min(float(feas), cross_dir_room)

            hourly_volumes.append(max(0.0, float(feas)))
            local_prev = current_power

        block_volume = float(min(hourly_volumes)) if hourly_volumes else 0.0
        return block_volume, local_prev

    def _capacity_price(
        self,
        unit: SupportsMinMax,
        block_start,
        volume: float,
        eom_price,
        energy_price,
    ) -> float:
        """
        Net opportunity cost per MW over the eom_foresight window:

            Σ_h [ max(0, sign·(EOM_signal[h] − MC[h]))      # forgone EOM margin
                − max(0, sign·(energy_signal[h] − MC[h])) ]  # CRM energy benefit

        sign = +1 (pos) / −1 (neg). The energy term replaces a hardcoded activation
        probability: activation is assumed when the energy signal beats cost.
        """
        if eom_price is None:
            return 0.0

        # exactly eom_foresight worth of steps (exclusive of the window end)
        window = list(
            unit.index[block_start : block_start + self.foresight - unit.index.freq]
        )
        net = 0.0
        for t in window:
            base_power = float(unit.outputs["energy"].at[t])
            mc_point = (
                base_power + volume
                if self.direction == "pos"
                else max(volume, base_power)
            )
            mc = float(unit.calculate_marginal_cost(t, mc_point))

            eom_margin = max(0.0, self.sign * (float(eom_price.at[t]) - mc))
            energy_margin = 0.0
            if energy_price is not None:
                energy_margin = max(0.0, self.sign * (float(energy_price.at[t]) - mc))
            net += eom_margin - energy_margin

        return net

    # ------------------------------------------------------------- energy leg

    def _energy_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """Up to two bids per product: must-offer (won capacity) + voluntary."""
        energy_price = unit.forecaster.price.get(market_config.market_id)

        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        _, max_p_full = unit.calculate_min_max_power(start, end)

        previous_power = unit.get_output_before(start)
        bids: list[dict] = []
        idx_offset = 0

        for product in product_tuples:
            t_start, t_end = product[0], product[1]
            steps = list(unit.index[t_start : t_end - unit.index.freq])
            n = len(steps)
            if n == 0:
                continue

            t = steps[0]
            base_power = float(unit.outputs["energy"].at[t])
            op_time = unit.get_operation_time(t)
            max_p_h = float(max_p_full[idx_offset])
            idx_offset += n

            price = self._energy_price(unit, t, base_power, energy_price, market_config)
            must_volume = float(unit.outputs[self.cap_product].at[t])

            # ---- must-offer leg (binding obligation from a capacity win) ----
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
            vol_volume = self._voluntary_headroom(
                unit, t, base_power, must_volume, max_p_h, op_time, previous_power
            )
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

            previous_power = base_power

        return self.remove_empty_bids(bids)

    def _voluntary_headroom(
        self,
        unit: SupportsMinMax,
        t,
        base_power: float,
        must_volume: float,
        max_p_h: float,
        op_time: int,
        previous_power: float,
    ) -> float:
        """Ramp-feasible voluntary headroom beyond the must-offer."""
        if self.direction == "pos":
            if base_power <= 0:
                # offline at this hour -> cannot deliver pos voluntary in-window
                return 0.0
            ramp_feas = unit.calculate_ramp(
                op_time, previous_power, max_p_h, base_power, start=t
            )
            return max(0.0, float(ramp_feas))

        absolute_floor = max(unit.min_power, float(unit.outputs["heat"].at[t]))
        ramp_floor = unit.calculate_ramp(
            op_time, previous_power, absolute_floor, base_power, start=t
        )
        total_room = max(0.0, float(base_power) - float(ramp_floor))
        return max(0.0, total_room - must_volume)

    def _energy_price(
        self,
        unit: SupportsMinMax,
        t,
        base_power: float,
        energy_price,
        market_config: MarketConfig,
    ) -> float:
        """CRM energy signal at t, clipped to the band; marginal-cost fallback."""
        if energy_price is not None:
            price = float(energy_price.at[t])
        else:
            price = float(unit.calculate_marginal_cost(t, base_power))

        if price >= 0:
            return min(price, market_config.maximum_bid_price)
        return max(price, market_config.minimum_bid_price)


class PowerPlantCrmPosStrategy(_PowerPlantCrmStrategy):
    """Positive-direction CRM strategy: handles ``capacity_pos`` and ``energy_pos``."""

    sign = 1.0
    direction = "pos"
    cap_product = "capacity_pos"
    energy_product = "energy_pos"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crm_energy_market_id = kwargs.get(
            "crm_energy_market_id_pos", "CRM_energy_pos"
        )


class PowerPlantCrmNegStrategy(_PowerPlantCrmStrategy):
    """Negative-direction CRM strategy: handles ``capacity_neg`` and ``energy_neg``."""

    sign = -1.0
    direction = "neg"
    cap_product = "capacity_neg"
    energy_product = "energy_neg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crm_energy_market_id = kwargs.get(
            "crm_energy_market_id_neg", "CRM_energy_neg"
        )
