# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np

from assume.common.base import MinMaxChargeStrategy, SupportsMinMaxCharge
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import parse_duration


class StorageEnergyHeuristicFlexableStrategy(MinMaxChargeStrategy):
    """
    The strategy is based on the storage strategy in flexABLE.

    If the current price forecast is higher than the average price forecast for a given foresight,
    the unit will discharge.
    The price is then set as the average price divided by the discharge efficiency of the unit.
    Otherwise, the unit will charge with the price defined as the average price multiplied by the charge efficiency of the unit.

    Attributes:
        foresight (datetime.timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = parse_duration(kwargs.get("eom_foresight", "12h"))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: Bids containing start_time, end_time, only_hours, price, volume.

        Note:
            The strategy is based on flexABLE
        """

        # =============================================================================
        # Calculate bids
        # =============================================================================
        # save a theoretic SOC to calculate the ramping
        start = product_tuples[0][0]
        theoretic_SOC = unit.get_soc(start)
        previous_power = unit.get_output_before(start)

        bids = []
        for product in product_tuples:
            start, end = product[0], product[1]

            current_power = unit.outputs["energy"].at[start]
            # ensures that potential additional commitment in this market is only in direction of already committed power in other market (=current_power)
            # does not use full potential of ramping in multi-market bidding, is a simplification of this heuristic strategy in accordance with flexABLE methodology
            current_power_discharge = max(current_power, 0)
            current_power_charge = min(current_power, 0)

            # calculate min and max power for charging and discharging
            min_power_charge, max_power_charge = unit.calculate_min_max_charge(
                start, end, soc=theoretic_SOC
            )
            min_power_discharge, max_power_discharge = unit.calculate_min_max_discharge(
                start, end, soc=theoretic_SOC
            )

            min_power_charge, max_power_charge = (
                min_power_charge[0],
                max_power_charge[0],
            )
            min_power_discharge, max_power_discharge = (
                min_power_discharge[0],
                max_power_discharge[0],
            )

            # Calculate ramping constraints using helper function
            max_power_discharge = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                max_power_discharge,
                current_power_discharge,
                min_power_discharge,
            )
            min_power_discharge = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                min_power_discharge,
                current_power_discharge,
                min_power_discharge,
            )
            max_power_charge = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                max_power_charge,
                current_power_charge,
                min_power_charge,
            )
            min_power_charge = unit.calculate_ramp_charge(
                theoretic_SOC,
                previous_power,
                min_power_charge,
                current_power_charge,
                min_power_charge,
            )
            price_forecast = unit.forecaster.price[market_config.market_id]

            # calculate average price
            average_price = calculate_price_average(
                current_time=start,
                foresight=self.foresight,
                price_forecast=price_forecast,
            )

            # if price is higher than average price, discharge
            # if price is lower than average price, charge
            if price_forecast[start] >= average_price:
                # favor discharge:
                if max_power_discharge != 0:
                    price = average_price / unit.efficiency_discharge
                    bid_quantity = max_power_discharge
                else:
                    # might be that we must charge, due to technical constraints
                    price = average_price * unit.efficiency_charge
                    bid_quantity = min_power_charge
            else:
                # favor charge
                if max_power_charge != 0:
                    price = average_price * unit.efficiency_charge
                    bid_quantity = max_power_charge
                else:
                    # might be that we must discharge, due to technical constraints
                    price = average_price / unit.efficiency_discharge
                    bid_quantity = min_power_discharge

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                    "node": unit.node,
                }
            )

            # calculate theoretic SOC
            time_delta = (end - start) / timedelta(hours=1)
            if bid_quantity + current_power > 0:
                delta_soc = (
                    -(
                        (bid_quantity + current_power)
                        * time_delta
                        / unit.efficiency_discharge
                    )
                    / unit.capacity
                )
            elif bid_quantity + current_power < 0:
                delta_soc = (
                    -(
                        (bid_quantity + current_power)
                        * time_delta
                        * unit.efficiency_charge
                    )
                    / unit.capacity
                )
            else:
                delta_soc = 0

            theoretic_SOC += delta_soc
            previous_power = bid_quantity + current_power

        bids = self.remove_empty_bids(bids)

        return bids


class StorageEnergyHeuristicRedispatchStrategy(MinMaxChargeStrategy):
    """
    Heuristic redispatch strategy for storage units: one price for every move,
    from a rule of thumb.

    A storage can be pushed both ways around its cleared position: discharge more
    or charge less (upward), charge more or discharge less (downward). For every
    product the bid carries what the clearing needs:

    - ``volume``: the schedule around which redispatch is applied, i.e. what the
      storage can actually run of its cleared dispatch (``+`` discharge, ``-``
      charge). If the cleared dispatch is not SOC-feasible, the grid model sees
      the real injection and other units cover the shortfall.
    - ``max_power`` / ``min_power``: the absolute discharge (``>= 0``) and charge
      (``<= 0``) limits without the capacity reserved on other markets. The state
      of charge is modelled by the clearing, not by these limits.
    - ``p_nom``: the full power span ``max_power_discharge - max_power_charge``.
    - ``soc``: the state of charge (fraction) at the start of the product along
      the schedule; the clearing starts the first product from it. Capacity, SOC
      limits and efficiencies come from the grid data.
    - ``node``: where the storage is connected.
    - ``price_discharge``: per MWh discharged more (the TSO pays) or less (the
      storage pays),
    - ``price_charge``: per MWh charged less (the TSO pays) or more (the storage
      pays),
    - ``price``: repeats ``price_discharge`` for mechanisms that read one price.

    The prices come from :meth:`calculate_redispatch_prices`. Here both are the
    reference price (average EOM price forecast over ``eom_foresight`` before and
    after the product, default 12 h, see :meth:`reference_price`) corrected for a
    full charge-discharge cycle, plus both variable costs::

        price = reference_price / (efficiency_charge * efficiency_discharge)
                + additional_cost_charge + additional_cost_discharge

    It is simple, but not cost-neutral: depending on the direction of the move
    the storage ends up better or worse off (see
    :class:`StorageEnergyOpportunityCostRedispatchStrategy`, which only changes
    the prices).

    Attributes:
        foresight (datetime.timedelta): Window of the price average, each side.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foresight = parse_duration(kwargs.get("eom_foresight", "12h"))

    def reference_price(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        start: datetime,
    ) -> float:
        """
        Average price forecast over ``foresight`` before and after ``start``.

        Redispatch clears after the EOM, so the EOM forecast is used. Without one,
        the forecast of the redispatch market itself, or else any forecast.
        """
        if "EOM" in unit.forecaster.price:
            price_forecast = unit.forecaster.price["EOM"]
        elif market_config.market_id in unit.forecaster.price:
            price_forecast = unit.forecaster.price[market_config.market_id]
        else:
            price_forecast = next(iter(unit.forecaster.price.values()))
        return calculate_price_average(
            current_time=start,
            foresight=self.foresight,
            price_forecast=price_forecast,
        )

    def calculate_redispatch_prices(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        start: datetime,
    ) -> dict:
        """
        Prices of the storage's redispatch moves in the product starting at
        ``start``: one price for both directions.

        Args:
            unit (SupportsMinMaxCharge): The storage unit.
            market_config (MarketConfig): The redispatch market.
            start (datetime.datetime): Start of the product.

        Returns:
            dict: ``price_discharge`` and ``price_charge`` (EUR/MWh).
        """
        price = (
            self.reference_price(unit, market_config, start)
            / (unit.efficiency_charge * unit.efficiency_discharge)
            + unit.additional_cost_charge
            + unit.additional_cost_discharge
        )
        return {"price_discharge": price, "price_charge": price}

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a storage unit and defines how it participates
        in the redispatch market.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: Bids containing start_time, end_time, only_hours, volume,
            max_power, min_power, p_nom, soc, node, price, price_discharge and
            price_charge.
        """
        # SOC and runnable power along the current schedule for the whole window.
        # It starts at unit.get_soc (i.e. ensure_soc), respects the SOC limits and
        # treats hours without a cleared schedule as idle.
        soc_path, runnable = unit.calculate_soc_path(
            product_tuples[0][0], product_tuples[-1][1]
        )
        p_nom = unit.max_power_discharge - unit.max_power_charge

        bids = []
        for start, end, only_hours in product_tuples:
            min_power, max_power = unit.calculate_redispatch_power_limits(start)
            prices = self.calculate_redispatch_prices(unit, market_config, start)
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": only_hours,
                    "price": prices["price_discharge"],
                    **prices,
                    "volume": runnable[start],
                    "max_power": max_power,
                    "min_power": min_power,
                    "p_nom": p_nom,
                    "soc": soc_path[start],
                    "node": unit.node,
                }
            )

        # keep zero-volume bids: an idle storage is a valid redispatch resource
        return bids


class StorageEnergyOpportunityCostRedispatchStrategy(
    StorageEnergyHeuristicRedispatchStrategy
):
    """
    Redispatch strategy for storage units that prices every move at its
    opportunity cost, so a call leaves the storage economically neither better
    nor worse off (cost-based redispatch, Section 13a EnWG).

    What a move costs the storage is the value of the stored energy it gains or
    loses, plus or minus the variable cost of running. With ``value`` = value of
    one MWh inside the storage:

    - one MWh discharged more or less at the grid changes the store by
      ``1 / efficiency_discharge`` MWh::

          price_discharge = value / efficiency_discharge + additional_cost_discharge

    - one MWh charged more or less at the grid changes the store by
      ``efficiency_charge`` MWh::

          price_charge = value * efficiency_charge - additional_cost_charge

    The value of the stored energy comes from the reference price (average EOM
    price forecast, see :meth:`StorageEnergyHeuristicRedispatchStrategy.reference_price`):

    - ``stored_energy_valuation="sell"`` (default): the energy would otherwise be
      sold later, ``value = reference_price * efficiency_discharge``;
    - ``stored_energy_valuation="refill"``: the energy has to be bought back,
      ``value = reference_price / efficiency_charge``.

    The bid also carries ``stored_energy_value``. Everything else in the bid is
    the same as in :class:`StorageEnergyHeuristicRedispatchStrategy`.

    Attributes:
        valuation (str): How the stored energy is valued.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valuation = kwargs.get("stored_energy_valuation", "sell")
        if self.valuation not in ("sell", "refill"):
            raise ValueError(
                f"Unknown stored_energy_valuation '{self.valuation}', "
                "use 'sell' or 'refill'."
            )

    def calculate_redispatch_prices(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        start: datetime,
    ) -> dict:
        reference_price = self.reference_price(unit, market_config, start)
        if self.valuation == "sell":
            value = reference_price * unit.efficiency_discharge
        else:
            value = reference_price / unit.efficiency_charge

        return {
            "price_discharge": value / unit.efficiency_discharge
            + unit.additional_cost_discharge,
            "price_charge": value * unit.efficiency_charge
            - unit.additional_cost_charge,
            "stored_energy_value": value,
        }


class StorageCapacityHeuristicBalancingPosStrategy(MinMaxChargeStrategy):
    """
    The strategy is based on the storage strategy in flexABLE.

    The strategy bids the energy_price for the energy_pos product if the specific revenue is positive.
    Otherwise, the strategy bids the capacity_price for the capacity_pos product.

    Attributes:
        foresight (datetime.timedelta): Foresight for the average price calculation.
        reference_market (str): The market whose price forecast is used to calculate the specific revenue.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = parse_duration(kwargs.get("crm_foresight", "4h"))
        # market used as reference for the specific revenue calculation
        self.reference_market = kwargs.get("reference_market", "EOM")

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids for the positive CRM market.

        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market and returns bids
        containing start_time, end_time, only_hours, price, volume.

        Args:
            unit (SupportsMinMaxCharge): The unit that is dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): List of product tuples.
            **kwargs: Additional keyword arguments.

        Returns:
            Orderbook: A list of bids.
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        theoretic_SOC = unit.get_soc(start)

        _, max_power_discharge_values = unit.calculate_min_max_discharge(
            start, end, soc=theoretic_SOC
        )
        bids = []

        for product, max_power_discharge in zip(
            product_tuples, max_power_discharge_values
        ):
            start = product[0]
            current_power = unit.outputs["energy"].at[start]

            # calculate ramping constraints for discharge
            bid_quantity = unit.calculate_ramp_discharge(
                theoretic_SOC,
                previous_power,
                max_power_discharge,
                current_power,
            )

            if bid_quantity == 0:
                continue

            marginal_cost = unit.calculate_marginal_cost(
                start=start, power=bid_quantity
            )

            specific_revenue = get_specific_revenue(
                unit=unit,
                marginal_cost=marginal_cost,
                t=start,
                foresight=self.foresight,
                price_forecast=unit.forecaster.price[self.reference_market],
            )

            # if specific revenue is positive, bid specific_revenue
            if specific_revenue >= 0:
                capacity_price = specific_revenue
            else:
                capacity_price = (
                    abs(specific_revenue) * unit.min_power_discharge / bid_quantity
                )

            energy_price = capacity_price

            if market_config.product_type == "capacity_pos":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": capacity_price,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                previous_power = current_power

            elif market_config.product_type == "energy_pos":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": energy_price,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                # calculate theoretic SOC
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = (
                    -(
                        (bid_quantity + current_power)
                        * time_delta
                        / unit.efficiency_discharge
                    )
                    / unit.capacity
                )
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
            else:
                previous_power = current_power
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )

        bids = self.remove_empty_bids(bids)

        return bids


class StorageCapacityHeuristicBalancingNegStrategy(MinMaxChargeStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the negative CRM(reserve market).

    Attributes:
        foresight (datetime.timedelta): Foresight for the average price calculation.

    Args:
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = parse_duration(kwargs.get("crm_foresight", "4h"))

    def calculate_bids(
        self,
        unit: SupportsMinMaxCharge,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        Args:
            unit (SupportsMinMax): A unit that the unit operator manages.
            market_config (MarketConfig): A market configuration.
            product_tuples (list[Product]): A list of tuples containing the start and end time of each product.
            kwargs (dict): Additional arguments.

        Returns:
            Orderbook: A list of bids.
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)

        theoretic_SOC = unit.get_soc(start)

        _, max_power_charge_values = unit.calculate_min_max_charge(start, end)

        bids = []
        for product, max_power_charge in zip(product_tuples, max_power_charge_values):
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            bid_quantity = abs(
                unit.calculate_ramp_charge(
                    theoretic_SOC,
                    previous_power,
                    max_power_charge,
                    current_power,
                )
            )

            # if bid_quantity >= min_bid_volume  --> not checked here
            if bid_quantity == 0:
                previous_power = current_power
                continue

            if market_config.product_type == "capacity_neg":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": 0,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                previous_power = current_power

            elif market_config.product_type == "energy_neg":
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": None,
                        "price": 0,
                        "volume": bid_quantity,
                        "node": unit.node,
                    }
                )
                # calculate theoretic SOC
                time_delta = (end - start) / timedelta(hours=1)
                delta_soc = (
                    (bid_quantity + current_power) * time_delta * unit.efficiency_charge
                ) / unit.capacity
                theoretic_SOC += delta_soc
                previous_power = bid_quantity + current_power
            else:
                previous_power = current_power
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )

        bids = self.remove_empty_bids(bids)

        return bids


def calculate_price_average(current_time, foresight, price_forecast):
    """
    Calculates the average price for a given foresight and returns the average price.

    Args:
        current_time (datetime.datetime): The current time.
        foresight (datetime.timedelta): The foresight.
        price_forecast (FastSeries): The price forecast.

    Returns:
        float: The average price.
    """
    start = max(current_time - foresight, price_forecast.index[0])
    end = min(current_time + foresight, price_forecast.index[-1])

    average_price = np.mean(price_forecast.loc[start:end])

    return average_price


def get_specific_revenue(unit, marginal_cost, t, foresight, price_forecast):
    """
    Calculates the specific revenue as difference between price forecast
    and marginal costs for the time defined by the foresight.

    Args:
        unit (SupportsMinMaxCharge): The unit that is dispatched.
        marginal_cost (float): The marginal cost.
        t (datetime.datetime): The start time of the product.
        foresight (datetime.timedelta): The foresight.
        price_forecast (FastSeries): The price forecast.

    Returns:
        float: The specific revenue.
    """

    possible_revenue = 0
    soc = unit.get_soc(t)
    theoretic_SOC = soc

    if t + foresight > price_forecast.index[-1]:
        _, max_power_discharge_values = unit.calculate_min_max_discharge(
            start=t, end=price_forecast.index[-1] + unit.index.freq, soc=theoretic_SOC
        )
        price_forecast = price_forecast.loc[t:]
    else:
        _, max_power_discharge_values = unit.calculate_min_max_discharge(
            start=t, end=t + foresight + unit.index.freq, soc=theoretic_SOC
        )
        price_forecast = price_forecast.loc[t : t + foresight]

    previous_power = unit.get_output_before(t)

    for market_price, max_power_discharge in zip(
        price_forecast, max_power_discharge_values
    ):
        theoretic_power_discharge = unit.calculate_ramp_discharge(
            theoretic_SOC,
            previous_power=previous_power,
            requested_power_discharge=max_power_discharge,
        )
        possible_revenue += (market_price - marginal_cost) * theoretic_power_discharge
        theoretic_SOC -= (
            theoretic_power_discharge
            * unit.index.freq
            / timedelta(hours=1)
            / unit.efficiency_discharge
        ) / unit.capacity
        previous_power = theoretic_power_discharge

    if soc != theoretic_SOC:
        possible_revenue = possible_revenue / (
            (soc - theoretic_SOC) * unit.efficiency_discharge * unit.capacity
        )
    else:
        possible_revenue = possible_revenue / unit.capacity

    return possible_revenue
