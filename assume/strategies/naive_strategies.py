# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product
from assume.common.utils import parse_duration


class NaiveSingleBidStrategy(BaseStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product
        previous_power = unit.get_output_before(
            start
        )  # power output of the unit before the start time of the first product
        op_time = unit.get_operation_time(start)
        min_power_values, max_power_values = unit.calculate_min_max_power(
            start, end_all
        )  # minimum and maximum power output of the unit between the start time of the first product and the end time of the last product

        bids = []
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]
            current_power = unit.outputs["energy"].at[
                start
            ]  # power output of the unit at the start time of the current product
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs
            volume = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": volume,
                    "node": unit.node,
                }
            )

            if "node" in market_config.additional_fields:
                bids[-1]["max_power"] = unit.max_power if volume > 0 else unit.min_power
                bids[-1]["min_power"] = min_power if volume > 0 else unit.max_power

            previous_power = volume + current_power
            if previous_power > 0:
                op_time = max(op_time, 0) + 1
            else:
                op_time = min(op_time, 0) - 1

        if "node" in market_config.additional_fields:
            return bids
        else:
            return self.remove_empty_bids(bids)


class NaiveProfileStrategy(BaseStrategy):
    """
    A naive strategy that bids the marginal cost of the unit as block bids over 24 hours on the day ahead market.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        op_time = unit.get_operation_time(start)
        _, max_power = unit.calculate_min_max_power(start, end_all)

        current_power = unit.outputs["energy"].at[start]
        marginal_cost = unit.calculate_marginal_cost(start, previous_power)

        # calculate the ramp up volume using the initial maximum power
        volume = unit.calculate_ramp(
            op_time, previous_power, max_power[0], current_power
        )

        profile = {product[0]: volume for product in product_tuples}
        order: Order = {
            "start_time": start,
            "end_time": product_tuples[-1][1],
            "only_hours": product_tuples[0][2],
            "price": marginal_cost,
            "volume": profile,
            "bid_type": "BB",
            "node": unit.node,
        }

        bids = [order]

        bids = self.remove_empty_bids(bids)
        return bids


class NaiveDADSMStrategy(BaseStrategy):
    """
    A naive strategy of a Demand Side Management (DSM) unit. The bid volume is the optimal power requirement of
    the unit at the start time of the product. The bid price is the marginal cost of the unit at the start time of the product.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        # check if unit has opt_power_requirement attribute
        if unit.optimisation_counter == 0:
            unit.determine_optimal_operation_without_flex()
            # self.plot_power_requirements(unit)
            unit.optimisation_counter = 1

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]

            volume = unit.opt_power_requirement.at[start]

            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": 3000,
                    "volume": -volume,
                }
            )

        return bids

    def plot_power_requirements(self, unit: SupportsMinMax):
        """
        Plots the optimal power requirement and flexibility power requirement for comparison.

        Args:
            unit (SupportsMinMax): The unit containing power requirements.
        """
        # Retrieve power requirements data
        opt_power_requirement = unit.opt_power_requirement
        flex_power_requirement = unit.flex_power_requirement

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(
            opt_power_requirement.index,
            opt_power_requirement,
            label="Reference profile",
            color="blue",
        )
        plt.plot(
            flex_power_requirement.index,
            flex_power_requirement,
            label="Flex Profile",
            color="orange",
            linestyle="--",
        )

        # Labels and title
        plt.xlabel("Time")
        plt.ylabel("Power (MW)")
        plt.title("Comparison of Reference and Flex Power Requirements")
        plt.legend()
        plt.grid(True)
        plt.show()

    def export_power_requirements_to_csv(self, unit: SupportsMinMax, file_path: str):
        """
        Exports the optimal and flexible power requirements time series to a CSV file.

        Args:
            unit (SupportsMinMax): The unit containing power requirements.
            file_path (str): The path to save the CSV file.
        """
        # Combine the two series into a DataFrame for parallel export
        df = pd.DataFrame(
            {
                "Optimal Power Requirement (kW)": unit.opt_power_requirement,
                "Flex Power Requirement (kW)": unit.flex_power_requirement,
            }
        )
        # Save to CSV
        df.to_csv(file_path)


class DSM_PosCRM_Strategy(BaseStrategy):
    """
    Strategy for Positive CRM Reserve (Demand Side, i.e., up & down, symmetric).
    """

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        bids = []
        max_power = unit.max_plant_capacity
        min_power = unit.min_plant_capacity

        for product in product_tuples:
            start, end, only_hours = product
            block_times = [dt for dt in unit.index.get_date_list() if start <= dt < end]

            # For all time steps in block, calculate possible symmetric bid
            up_caps = []
            down_caps = []
            for t in block_times:
                flex = unit.flex_power_requirement.at[t]
                up_caps.append(max_power - flex)
                down_caps.append(flex - min_power)
            # The symmetric bid is the minimum capacity that is possible in *all* timesteps in the block
            symmetric_capacity = min(min(up_caps), min(down_caps))
            if symmetric_capacity > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": 0,  # or unit.calculate_marginal_cost(...)
                        "volume": symmetric_capacity,
                        "unit_id": unit.id,
                        "market_id": "CRM_pos",
                    }
                )
        return self.remove_empty_bids(bids)


class DSM_NegCRM_Strategy(BaseStrategy):
    """
    Strategy for Negative CRM Reserve (Demand Side, i.e., up & down, symmetric).
    """

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        # IDENTICAL LOGIC as POS, since symmetric in Germany (volume is symmetric)
        # If you ever want to do *only* neg or pos (asymmetric), just change which cap you use!
        bids = []
        max_power = unit.max_plant_capacity
        min_power = unit.min_plant_capacity

        for product in product_tuples:
            start, end, only_hours = product
            block_times = [dt for dt in unit.index.get_date_list() if start <= dt < end]

            up_caps = []
            down_caps = []
            for t in block_times:
                flex = unit.flex_power_requirement.at[t]
                up_caps.append(max_power - flex)
                down_caps.append(flex - min_power)
            symmetric_capacity = min(min(up_caps), min(down_caps))
            if symmetric_capacity > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": 0,
                        "volume": symmetric_capacity,
                        "unit_id": unit.id,
                        "market_id": "CRM_neg",
                    }
                )
        return self.remove_empty_bids(bids)


class NaiveRedispatchDSMStrategy(BaseStrategy):
    """
    A naive strategy of a Demand Side Management (DSM) unit that bids the available flexibility of the unit on the redispatch market.
    The bid volume is the flexible power requirement of the unit at the start time of the product. The bid price is the marginal cost of the unit at the start time of the product.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        # calculate the optimal operation of the unit according to the objective function
        unit.determine_optimal_operation_with_flex()

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]
            volume = unit.flex_power_requirement.at[start]
            marginal_price = unit.calculate_marginal_cost(start, volume)
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_price,
                    "volume": -volume,
                }
            )

        return bids


class NaiveRedispatchStrategy(BaseStrategy):
    """
    A naive strategy that simply submits all information about the unit and
    currently dispatched power for the following hours to the redispatch market.
    Information includes the marginal cost, the ramp up and down values, and the dispatch.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        :param unit: the unit to be dispatched
        :type unit: SupportsMinMax
        :param market_config: the market configuration
        :type market_config: MarketConfig
        :param product_tuples: list of all products the unit can offer
        :type product_tuples: list[Product]
        :return: the bids consisting of the start time, end time, only hours, price and volume.
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        # end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.min_power, unit.max_power

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs

            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": current_power,
                    "max_power": max_power,
                    "min_power": min_power,
                    "node": unit.node,
                }
            )

        return bids


class FixedDispatchStrategy(BaseStrategy):
    """
    A naive strategy that simply submits all information about the unit and
    currently dispatched power for the following hours to the redispatch market.
    Information includes the marginal cost, the ramp up and down values, and the dispatch.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        :param unit: the unit to be dispatched
        :type unit: SupportsMinMax
        :param market_config: the market configuration
        :type market_config: MarketConfig
        :param product_tuples: list of all products the unit can offer
        :type product_tuples: list[Product]
        :return: the bids consisting of the start time, end time, only hours, price and volume.
        :rtype: Orderbook
        """

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]

            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": 0,
                    "volume": current_power,
                    "max_power": current_power,
                    "min_power": current_power,
                    "node": unit.node,
                }
            )

        return bids

class FlexableRedispatchDSM(BaseStrategy):
    """
    Redispatch strategy for an industrial DSM unit (cement plant).
    Expects the following per-unit series in forecaster (aligned to unit.index):
      - f"{unit.id}_baseline_power"  [MW]
      - f"{unit.id}_max_up"          [MW]
      - f"{unit.id}_max_down"        [MW]
      - f"{unit.id}_price_up"        [€/MWh]
      - f"{unit.id}_price_down"      [€/MWh]
    Submits exactly one bid (up or down) per product/hour.
    """

    # ---------- helpers ---------------------------------------------------------

    def _ensure_pandas_index(self, idx_like) -> pd.Index:
        """Convert custom FastIndex → pandas.DatetimeIndex (or generic Index)."""
        if isinstance(idx_like, pd.Index):
            return idx_like
        # try to iterate
        try:
            return pd.DatetimeIndex(list(idx_like))
        except Exception:
            pass
        # try start/end/freq on objects like FastIndex
        start = getattr(idx_like, "start", None)
        end   = getattr(idx_like, "end",   None)
        freq  = getattr(idx_like, "freq",  None)
        if start is not None and end is not None and freq is not None:
            return pd.date_range(start=start, end=end, freq=freq)
        raise TypeError(f"Cannot convert index of type {type(idx_like)} to pandas Index")

    def _need_keys(self, forecaster, keys: list[str]) -> None:
        missing = []
        for k in keys:
            try:
                _ = forecaster[k]
            except KeyError:
                missing.append(k)
        if missing:
            raise KeyError(f"Forecaster is missing keys: {missing}")

    def _as_clean_series(self, series_like, key: str, index: pd.Index) -> pd.Series:
        """
        Ensure we have a pandas Series aligned to `index`.
        - If Series with different index → reindex.
        - If array-like/list → wrap with provided index.
        - If dict-like → make Series then reindex.
        """
        if isinstance(series_like, pd.Series):
            if not series_like.index.equals(index):
                return series_like.reindex(index).rename(key)
            return series_like.rename(key)
        if hasattr(series_like, "keys") and hasattr(series_like, "__getitem__"):
            # dict-like
            return pd.Series(series_like).reindex(index).rename(key)
        # array-like / scalar
        return pd.Series(series_like, index=index, name=key)

    def _get_series(self, forecaster, key: str, index: pd.Index) -> pd.Series:
        s = forecaster[key]  # KeyError bubbles up if missing
        return self._as_clean_series(s, key, index)

    # ---------- main ------------------------------------------------------------

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        """
        Redispatch bidding for an industrial DSM (cement plant) consumer.

        Data expected in unit.forecaster (all Series aligned to unit.index):
        f"{unit.id}_baseline_power"  [MW]   (not strictly needed for bidding)
        f"{unit.id}_max_up"          [MW]   (feasible increase of consumption)
        f"{unit.id}_max_down"        [MW]   (feasible decrease of consumption)
        f"{unit.id}_price_up"        [€/MWh] (activation cost to consume more)
        f"{unit.id}_price_down"      [€/MWh] (opportunity cost to consume less)

        Sign convention for consumer bids:
        - UP (consume more):  volume NEGATIVE (−MW)
        - DOWN (consume less): volume POSITIVE (+MW)
        """

        f = unit.forecaster
        idx = unit.index

        # Dynamic keys per-plant
        k_power      = f"{unit.id}_baseline_power"
        k_up_cap     = f"{unit.id}_max_up"
        k_down_cap   = f"{unit.id}_max_down"
        k_up_price   = f"{unit.id}_price_up"
        k_down_price = f"{unit.id}_price_down"

        # Pull series (must be pandas-like and aligned to idx)
        try:
            s_power      = f[k_power]
            s_cap_up     = f[k_up_cap]
            s_cap_down   = f[k_down_cap]
            s_price_up   = f[k_up_price]
            s_price_down = f[k_down_price]
        except KeyError as e:
            raise KeyError(f"Forecaster missing required key: {e}") from e

        # Light type sanity
        for key, s in [(k_power, s_power), (k_up_cap, s_cap_up), (k_down_cap, s_cap_down),
                    (k_up_price, s_price_up), (k_down_price, s_price_down)]:
            if not hasattr(s, "at"):
                raise TypeError(f"Forecaster[{key}] must be a pandas Series aligned to unit.index")

        bids = []

        for product in product_tuples:
            t0, t1 = product[0], product[1]
            only_hours = product[2] if len(product) > 2 else None

            # Fetch values
            cap_up   = float(max(0.0, s_cap_up.at[t0]   if t0 in s_cap_up.index   else 0.0))
            cap_down = float(max(0.0, s_cap_down.at[t0] if t0 in s_cap_down.index else 0.0))
            p_up     = float(max(0.0, s_price_up.at[t0]   if t0 in s_price_up.index   else 0.0))
            p_down   = float(max(0.0, s_price_down.at[t0] if t0 in s_price_down.index else 0.0))

            # If no headroom either way, skip
            if cap_up <= 0.0 and cap_down <= 0.0:
                continue

            # Choose side: if both possible, pick the cheaper €/MWh;
            # tie-break by larger available MW.
            choose_up = False
            choose_down = False

            if cap_up > 0.0 and cap_down <= 0.0:
                choose_up = True
            elif cap_down > 0.0 and cap_up <= 0.0:
                choose_down = True
            else:
                # both > 0
                if p_up < p_down - 1e-9:
                    choose_up = True
                elif p_down < p_up - 1e-9:
                    choose_down = True
                else:
                    # equal prices -> pick the side with more volume
                    if cap_up >= cap_down:
                        choose_up = True
                    else:
                        choose_down = True

            if choose_up:
                # UP = consume more => NEGATIVE volume
                bids.append({
                    "start_time": t0,
                    "end_time": t1,
                    "only_hours": only_hours,
                    "price": p_up,                     # €/MWh for 1h block → €/MW
                    "volume": -cap_up,                 # NEGATIVE MW
                    "max_power": cap_up,               # optional metadata
                    "min_power": 0.0,                  # optional metadata
                    "node": unit.node,
                    "direction": "up",
                    "unit_id": unit.id,
                })

            if choose_down:
                # DOWN = consume less => POSITIVE volume
                bids.append({
                    "start_time": t0,
                    "end_time": t1,
                    "only_hours": only_hours,
                    "price": p_down,                   # €/MWh for 1h block → €/MW
                    "volume": cap_down,                # POSITIVE MW
                    "max_power": cap_down,             # optional metadata
                    "min_power": 0.0,                  # optional metadata
                    "node": unit.node,
                    "direction": "down",
                    "unit_id": unit.id,
                })

        return self.remove_empty_bids(bids)

class NaiveRedispatchStrategyDSM(BaseStrategy):
    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        # calculate the optimal operation of the unit according to the objective function
        unit.calculate_optimal_operation_if_needed()

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": 30,
                    "volume": -5,
                    "max_power": 5,
                    "min_power": 0,
                    "node": "west",  # unit.node,
                }
            )
        return bids


class NaiveExchangeStrategy(BaseStrategy):
    """
    A naive strategy for an exchange unit that bids the defined import and export prices on the market.
    It submits two bids, one for import and one for export, with the respective prices and volumes.
    Export bids have negative volumes and are treated as demand on the market.
    Import bids have positive volumes and are treated as supply on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        bids = []
        for product in product_tuples:
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]

            # append import bid
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.price_import,
                    "volume": unit.volume_import.at[start],
                    "node": unit.node,
                }
            )

            # append export bid
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.price_export,
                    "volume": unit.volume_export.at[start],
                    "node": unit.node,
                }
            )

        # clean up empty bids
        bids = self.remove_empty_bids(bids)

        return bids


class ElasticDemandStrategy(BaseStrategy):
    """
    A bidding strategy for a demand unit that submits multiple bids to approximate
    a marginal utility curve, based on linear or isoelastic demand theory.
    P = Price, Q = Quantity, E = Elasticity.

    - Linear model: P = P_max + slope * Q (slope is only defined by P_max and Q_max, negative value)
    - Isoelastic model: P = (Q/Q_max) ** (1/E) (E is negative)
      (derived from log-log price elasticity of demand)

    See:
    - https://en.wikipedia.org/wiki/Price_elasticity_of_demand
    - Arnold, Fabian. 2023. https://hdl.handle.net/10419/286380
    - Hirth, Lion et al. 2024. https://doi.org/10.1016/j.eneco.2024.107652.

    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        bids = []

        for product in product_tuples:
            start, end, only_hours = product
            max_abs_power = max(abs(unit.min_power), abs(unit.max_power))

            if unit.elasticity_model == "isoelastic":
                # ISOELASTIC model (constant elasticity of demand):
                # P = (Q / Q_max)^(1 / E)
                # Derived from:
                #   E = (dQ/dP) * (P / Q(P)), E == constant
                #   E = Q'(P) * (P / Q(P))
                #   Q'(P) = E * 1/P * Q(P)
                #   dQ/dP = E * (1 / P) * Q
                #   integrate:
                #   \int 1/Q dQ = E * \int 1/P dP
                #   ln(Q) = E * ln(P) + C
                #   exp(ln(Q)) = exp(E * ln(p) + C)
                #   Q(p) = (exp(ln(p))^E) * exp(C)
                #   Q(p) = P^E * exp(C)
                #   possibly C = 0, C = 1 or C >= 1. We assume C >= 1.
                #   C shifts the demand curve (demand vs. price) up / down and
                #   can be interpreted as the demand that will always be served
                #   we set C = ln(Q_max) and derive the demand curve from there:
                #   P = Q^(1/E) * exp(-C/E) and C = ln(Q_max)
                #   => P = Q^(1/E) * exp(-ln(Q_max)/E) and because exp(-ln(Q_max)/E) = Q_max^(-1/E)
                #   => P = Q^(1/E) * Q_max^(-1/E)
                #   finally
                #   => P = (Q / Q_max)^(1/E)
                # This yields decreasing price as volume increases

                # calculate first bid in isoelastic model (the volume that is bid at max price)
                first_bid_volume = self.find_first_block_bid(
                    unit.elasticity, unit.max_price, max_abs_power
                )
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": unit.max_price,
                        "volume": -first_bid_volume,
                        "node": unit.node,
                    }
                )

                remaining_volume = max_abs_power - first_bid_volume
                if remaining_volume < 0:
                    raise ValueError(
                        "Negative remaining volume for bidding. Check max_power and elasticity."
                    )

                bid_volume = remaining_volume / (unit.num_bids - 1)
                # calculate the remaining bids in isoelastic model
                # P = (Q/Q_max) ** (1/E)
                for i in range(1, unit.num_bids):
                    ratio = (first_bid_volume + i * bid_volume) / max_abs_power
                    if ratio <= 0:
                        continue
                    bid_price = ratio ** (1 / unit.elasticity)

                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": bid_price,
                            "volume": -bid_volume,
                            "node": unit.node,
                        }
                    )

            elif unit.elasticity_model == "linear":
                # LINEAR model: P = P_max - slope * Q
                # where slope = -(P_max / Q_max)
                # This ensures price drops linearly with volume
                bid_volume = max_abs_power / unit.num_bids
                slope = -(unit.max_price / max_abs_power)
                for i in range(unit.num_bids):
                    bid_price = unit.max_price + (slope * i * bid_volume)

                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": bid_price,
                            "volume": -bid_volume,
                            "node": unit.node,
                        }
                    )

        return self.remove_empty_bids(bids)

    def find_first_block_bid(
        self, elasticity: float, max_price: float, max_power: float
    ) -> float:
        """
        Calculate the first block bid volume at max_price. P = Price, Q = Quantity, E = Elasticity.
        Assumes isoelastic demand:

        .. math::
            Q = Q_{max} * P^E

        The first block bid is the volume that is always bid at maximum price, because the
        willingness to pay for it is higher than the markets maximal price.
        The first block bid volume is calculated by finding the intersection of the isoelastic demand
        curve and the maximum price in the marginal utility plot. All demand left of the intersection
        is always bought at maximum price and is called $Q_{first}$.

        .. math::
            Q_{first} = Q
            Q_{first} = Q_{max} * P^E

        Therefore:

        .. math::
            Q_{first} = power_{max} * (price_{max} ^ E)

        Returns:
            float: Volume > 0, demand that is always bought at max willingness to pay
        """
        volume = max_power * max_price**elasticity

        if abs(volume) > abs(max_power):
            raise ValueError(
                f"Calculated first block bid volume ({volume}) exceeds max power ({max_power})."
            )
        return volume


class DSM_PosCRM_Strategy(BaseStrategy):
    """
    Strategy for Positive CRM Reserve (Demand Side, i.e., up & down, symmetric).
    """

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        bids = []
        max_power = unit.max_plant_capacity
        min_power = unit.min_plant_capacity

        for product in product_tuples:
            start, end, only_hours = product
            block_times = [dt for dt in unit.index.get_date_list() if start <= dt < end]

            # For all time steps in block, calculate possible symmetric bid
            up_caps = []
            down_caps = []
            for t in block_times:
                flex = unit.flex_power_requirement.at[t]
                up_caps.append(max_power - flex)
                down_caps.append(flex - min_power)
            # The symmetric bid is the minimum capacity that is possible in *all* timesteps in the block
            symmetric_capacity = min(min(up_caps), min(down_caps))
            if symmetric_capacity > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": 0,  # or unit.calculate_marginal_cost(...)
                        "volume": symmetric_capacity,
                        "unit_id": unit.id,
                        "market_id": "CRM_pos",
                    }
                )
        return self.remove_empty_bids(bids)


class DSM_NegCRM_Strategy(BaseStrategy):
    """
    Strategy for Negative CRM Reserve (Demand Side, i.e., up & down, symmetric).
    """

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        # IDENTICAL LOGIC as POS, since symmetric in Germany (volume is symmetric)
        # If you ever want to do *only* neg or pos (asymmetric), just change which cap you use!
        bids = []
        max_power = unit.max_plant_capacity
        min_power = unit.min_plant_capacity

        for product in product_tuples:
            start, end, only_hours = product
            block_times = [dt for dt in unit.index.get_date_list() if start <= dt < end]

            up_caps = []
            down_caps = []
            for t in block_times:
                flex = unit.flex_power_requirement.at[t]
                up_caps.append(max_power - flex)
                down_caps.append(flex - min_power)
            symmetric_capacity = min(min(up_caps), min(down_caps))
            if symmetric_capacity > 0:
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": 0,
                        "volume": symmetric_capacity,
                        "unit_id": unit.id,
                        "market_id": "CRM_neg",
                    }
                )
        return self.remove_empty_bids(bids)
