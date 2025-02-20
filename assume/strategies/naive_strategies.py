# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product


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


# class NaiveDADSMStrategy(BaseStrategy):
#     """
#     A naive strategy of a Demand Side Management (DSM) unit for the Day-Ahead Market.
#     The bid volume is the optimal power requirement above the baseline threshold.
#     """

#     def calculate_bids(
#         self,
#         unit: SupportsMinMax,
#         market_config: MarketConfig,
#         product_tuples: list[Product],
#         **kwargs,
#     ) -> Orderbook:
#         """
#         Formulate bids for the Day-Ahead Market.

#         Args:
#             unit: The demand-side agent (e.g., cement plant) for which bids are being formulated.
#             market_config: Market configuration containing product details and constraints.
#             product_tuples: List of products for the Day-Ahead Market (duration and time steps).
#             **kwargs: Additional arguments for bid calculation.

#         Returns:
#             Orderbook: Contains bids for the Day-Ahead Market.
#         """

#         # Calculate the baseline threshold (80% of maximum power requirement)
#         max_power = max(unit.opt_power_requirement)
#         baseline_threshold = 0.5 * max_power

#         bids = []
#         for product in product_tuples:
#             """
#             for each product, calculate the marginal cost of the unit at the start time of the product
#             and the volume of the product. Dispatch the order to the market.
#             """
#             start = product[0]
#             end = product[1]

#             # Calculate bid volume (energy above the threshold)
#             opt_power = unit.opt_power_requirement.at[start]
#             if opt_power > baseline_threshold:
#                 bid_volume = opt_power - baseline_threshold
#             else:
#                 bid_volume = 0

#             # Skip if bid volume is 0
#             if bid_volume <= 0:
#                 continue

#             # Calculate the marginal cost for the bid volume
#             marginal_price = unit.calculate_marginal_cost(start=start, power=bid_volume)

#             bids.append(
#                 {
#                     "start_time": start,
#                     "end_time": end,
#                     "only_hours": product[2],
#                     "price": 3000,
#                     "volume": -bid_volume,
#                 }
#             )

#         return bids

#     def plot_power_requirements(self, unit: SupportsMinMax):
#         """
#         Plots the optimal power requirement and flexibility power requirement for comparison.

#         Args:
#             unit (SupportsMinMax): The unit containing power requirements.
#         """
#         # Retrieve power requirements data
#         opt_power_requirement = unit.opt_power_requirement
#         flex_power_requirement = unit.flex_power_requirement

#         # Plotting
#         plt.figure(figsize=(10, 6))
#         plt.plot(
#             opt_power_requirement.index,
#             opt_power_requirement,
#             label="Optimal Power Requirement",
#             color="blue",
#         )
#         plt.plot(
#             flex_power_requirement.index,
#             flex_power_requirement,
#             label="Flex Power Requirement",
#             color="orange",
#             linestyle="--",
#         )

#         # Labels and title
#         plt.xlabel("Time")
#         plt.ylabel("Power Requirement (kW)")
#         plt.title("Comparison of Optimal and Flexible Power Requirements")
#         plt.legend()
#         plt.grid(True)
#         plt.show()


# class OTC_DSM_Strategy(BaseStrategy):
#     """
#     Strategy for Long-Term Market (LTM) bids based on baseline power threshold.
#     """

#     def calculate_bids(
#         self,
#         unit: SupportsMinMax,
#         market_config: MarketConfig,
#         product_tuples: list[Product],
#         **kwargs,
#     ) -> Orderbook:
#         """
#         Formulate bids for the LTM market.

#         Args:
#             unit: The demand-side agent (e.g., cement plant) for which bids are being formulated.
#             market_config: Market configuration containing product details and constraints.
#             product_tuples: List of products for the LTM market (duration and time steps).
#             **kwargs: Additional arguments for bid calculation.

#         Returns:
#             Orderbook: Contains bids for the LTM market.
#         """
#         if unit.optimisation_counter == 0:
#             unit.determine_optimal_operation_with_flex()
#             # self.plot_power_requirements(unit)
#             unit.optimisation_counter = 1

#         # Calculate the baseline threshold (70% of maximum power requirement)
#         max_power = max(unit.opt_power_requirement)
#         baseline_threshold = 0.7 * max_power

#         bids = []

#         for product in product_tuples:
#             start = product[0]
#             end = product[1]

#             # Calculate total energy below the threshold
#             current_time = start
#             energy_below_threshold = 0

#             while current_time < end:
#                 opt_power = unit.opt_power_requirement.at[current_time]
#                 power_below_threshold = min(opt_power, baseline_threshold)
#                 energy_below_threshold += power_below_threshold
#                 current_time += unit.index.freq

#             # Skip if no valid energy for the product
#             if energy_below_threshold <= 0:
#                 continue

#             # Calculate the marginal cost for the bid volume (total energy below threshold)
#             marginal_price = unit.calculate_marginal_cost(
#                 start=start, power=energy_below_threshold
#             )

#             # Add the bid to the list
#             bids.append(
#                 {
#                     "start_time": start,
#                     "end_time": end,
#                     "only_hours": None,
#                     "price": 3000,
#                     "volume": -energy_below_threshold,
#                     "node": unit.node,
#                 }
#             )

#         # Clean up empty bids
#         bids = self.remove_empty_bids(bids)

#         return bids

# # Flex and Inflex
# class OTC_DSM_Strategy(BaseStrategy):
#     """
#     Strategy for Long-Term Market (LTM) bids based on aggregated power requirements.
#     """
    
#     def calculate_bids(
#         self,
#         unit: SupportsMinMax,
#         market_config: MarketConfig,
#         product_tuples: list[Product],
#         **kwargs,
#     ) -> Orderbook:
#         """
#         Formulate bids for the LTM market using aggregated power over 24-hour periods. (Inflex & Flex)
#         """
#         bids = []
        
#         for product in product_tuples:
#             start, end, only_hours = product
            
#             # Aggregate power demand for the 24-hour period
#             total_power = unit.forecaster['flex_power'].loc[start:end].sum()
#             # avg_price = unit.LTM_price.loc[start:end].mean()
            
#             # Skip bid if no power demand
#             if total_power <= 0:
#                 continue
            
#             # Add bid to the list
#             bids.append(
#                 {
#                     "start_time": start,
#                     "end_time": end,
#                     "only_hours": None,
#                     "price": 3000,
#                     "volume": -total_power,
#                     "node": unit.node,
#                 }
#             )
        
#         return self.remove_empty_bids(bids)
    
    # def plot_power_requirements(self, unit: SupportsMinMax):
    #     """
    #     Plots the optimal power requirement and flexibility power requirement for comparison.

    #     Args:
    #         unit (SupportsMinMax): The unit containing power requirements.
    #     """
    #     # Retrieve power requirements data
    #     opt_power_requirement = unit.opt_power_requirement
    #     flex_power_requirement = unit.flex_power_requirement

    #     # Plotting
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(
    #         opt_power_requirement.index,
    #         opt_power_requirement,
    #         label="Optimal Power Requirement",
    #         color="blue",
    #     )
    #     plt.plot(
    #         flex_power_requirement.index,
    #         flex_power_requirement,
    #         label="Flex Power Requirement",
    #         color="orange",
    #         linestyle="--",
    #     )

    #     # Labels and title
    #     plt.xlabel("Time")
    #     plt.ylabel("Power Requirement (kW)")
    #     plt.title("Comparison of Optimal and Flexible Power Requirements")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

# # LTM_50/80_EOM
# class OTC_DSM_Strategy(BaseStrategy):
#     """
#     Bidding strategy for the Long-Term Market (LTM) with a 50% threshold.
#     """

#     def calculate_bids(
#         self,
#         unit: SupportsMinMax,
#         market_config: MarketConfig,
#         product_tuples: list[Product],
#         **kwargs,
#     ) -> Orderbook:

#         max_power = max(unit.forecaster['flex_power'])
#         baseline_threshold = 0.8 * max_power  

#         bids = []
#         for product in product_tuples:
#             start, end, only_hours = product
#             # Compute total energy below threshold for the 24-hour block
#             energy_below_threshold = 0
#             current_time = start
#             while current_time < end:
#                 power_value = unit.forecaster['flex_power'].at[current_time]
#                 energy_below_threshold += min(power_value, baseline_threshold)
#                 current_time += unit.index.freq

#             if energy_below_threshold > 0:
#                 # avg_price = unit.LTM_price.loc[start:end].mean()  # Average LTM price over the block
#                 bids.append({
#                     "start_time": start,
#                     "end_time": end,
#                     "price": 3000,
#                     "volume": -energy_below_threshold,
#                     "node": unit.node,
#                 })
#         return self.remove_empty_bids(bids)

# # # LTM_50/80_EOM

# class NaiveDADSMStrategy(BaseStrategy):
#     """
#     Bidding strategy for the Day-Ahead Market (EOM) for excess demand.
#     """

#     def calculate_bids(
#         self,
#         unit: SupportsMinMax,
#         market_config: MarketConfig,
#         product_tuples: list[Product],
#         **kwargs,
#     ) -> Orderbook:
#         max_power = max(unit.forecaster['flex_power'])
#         baseline_threshold = 0.8 * max_power  

#         bids = []
#         for product in product_tuples:
#             start = product[0]
#             end = product[1]
#             current_time = start
#             while current_time < end:
#                 flex_power = unit.forecaster['flex_power'].at[current_time]
#                 if flex_power > baseline_threshold:
#                     bid_volume = flex_power - baseline_threshold
#                     # bid_price = unit.EOM_price.at[current_time]
#                     bids.append({
#                         "start_time": current_time,
#                         "end_time": current_time + unit.index.freq,
#                         "price": 3000,
#                         "volume": -bid_volume,
#                     })
#                 current_time += unit.index.freq

#         return self.remove_empty_bids(bids)

# Reserve LTM and EOM
    
class DSM_PosCRM_Strategy(BaseStrategy):
    """
    Strategy for trading positive reserve (CRM_pos) based on available reserve power.
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
            start = product[0]
            end = product[1]
            # Initialize min reserve power for 4-hour block
            bid_volume = float("inf")

            # Determine the lowest reserve power across the 4-hour block
            current_time = start
            while current_time < end:
                reserve_power = (
                    unit.forecaster["flex_power"][current_time] - unit.forecaster["reserve_power"][current_time]
                )
                bid_volume = min(bid_volume, reserve_power)  # Take min to ensure constant power bid
                current_time += unit.index.freq  # Move to next hour

            # Ensure we only bid if reserve power is available
            if bid_volume > 0:
                # Calculate opportunity cost as the average EOM price in the bidding period
                price_total = 0
                count = 0
                current_time = start
                while current_time < end:
                    price_total += unit.forecaster["price_EOM_flex"][current_time]
                    count += 1
                    current_time += unit.index.freq

                bid_price = price_total / count if count > 0 else 0  # Avoid division by zero

                bids.append({
                    "start_time": start,
                    "end_time": end,
                    "price": bid_price,
                    "volume": bid_volume,
                    "node": unit.node,
                })

        return self.remove_empty_bids(bids)

class OTC_DSM_Strategy(BaseStrategy):
    """
    Strategy for trading in the Long-Term Market (LTM) with 80% allocation.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        max_power = max(unit.forecaster['flex_power'])
        baseline_threshold = 0.8 * max_power  # 80% threshold

        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]

            # Accumulate the bid volume for the 24-hour period
            energy_below_threshold = 0
            current_time = start
            while current_time < end:
                flex_power = unit.forecaster["flex_power"][current_time]
                reserve_power = unit.forecaster["reserve_power"][current_time]
                available_power = flex_power - reserve_power  # Adjust for reserve trade

                # Bid up to the baseline threshold
                energy_below_threshold += min(available_power, baseline_threshold)
                current_time += unit.index.freq

            if energy_below_threshold > 0:
                avg_price = 3000  # Assuming static price for now
                bids.append({
                    "start_time": start,
                    "end_time": end,
                    "price": avg_price,
                    "volume": -energy_below_threshold,
                    "node": unit.node,
                })

        return self.remove_empty_bids(bids)
    
class NaiveDADSMStrategy(BaseStrategy):
    """
    Strategy for trading in the Day-Ahead Market (EOM) with the remaining 20% power.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        max_power = max(unit.forecaster['flex_power'])
        baseline_threshold = 0.8 * max_power  # 80% to LTM, remaining 20% in EOM

        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]
            # Accumulate EOM bid volume
            eom_bid_volume = 0
            current_time = start
            while current_time < end:
                flex_power = unit.forecaster["flex_power"][current_time]
                reserve_power = unit.forecaster["reserve_power"][current_time]
                available_power = flex_power - reserve_power

                # Allocate remaining 20% to EOM
                eom_bid_volume += max(0, available_power - baseline_threshold)
                current_time += unit.index.freq

            if eom_bid_volume > 0:
                avg_price = 3000  # Assuming static price for now
                bids.append({
                    "start_time": start,
                    "end_time": end,
                    "price": avg_price,
                    "volume": -eom_bid_volume,
                    "node": unit.node,
                })

        return self.remove_empty_bids(bids)

  ##### Optim case  
# class DSM_PosCRM_Strategy(BaseStrategy):
#     """
#     Strategy for Positive CRM Reserve (Demand Side).
#     """

#     def calculate_bids(
#         self,
#         unit: SupportsMinMax,
#         market_config: MarketConfig,
#         product_tuples: list[Product],
#         **kwargs,
#     ) -> Orderbook:
#         """
#         Calculate bids for the Positive CRM market for each 4-hour block.
#         """
#         # Determine the optimal operation of the unit
#         # unit.determine_optimal_operation_with_flex()

#         # Calculate the baseline power threshold (80% of max optimal power requirement)
#         max_power = max(unit.forecaster['flex_power'])
#         baseline_threshold = 0.8 * max_power

#         bids = []

#         for product in product_tuples:
#             start = product[0]
#             end = product[1]

#             # Find the maximum flex_upward in the 4-hour block
#             max_flex_upward = 0
#             current_time = start
#             while current_time < end:
#                 flex_upward = max(
#                     0,
#                     unit.opt_power_requirement.at[current_time]
#                     - unit.flex_power_requirement.at[current_time],
#                 )
#                 max_flex_upward = max(max_flex_upward, flex_upward)
#                 current_time += unit.index.freq  # Increment time by 1 hour

#             # Skip the block if no upward flexibility is available
#             if max_flex_upward <= 0:
#                 continue

#             # Calculate the capacity price
#             capacity_price = unit.calculate_marginal_cost(
#                 start=start, power=max_flex_upward
#             )

#             # Add the bid to the list
#             bids.append(
#                 {
#                     "start_time": start,
#                     "end_time": end,
#                     "only_hours": product[2],
#                     "price": 0,
#                     "volume": max_flex_upward,
#                     "node": unit.node,
#                 }
#             )
#         bids = self.remove_empty_bids(bids)

#         return bids
    
class DSM_NegCRM_Strategy(BaseStrategy):
    """
    Strategy for Negative CRM Reserve (Demand Side).
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculate bids for the Negative CRM market for each 4-hour block.
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        bids = []
        for product in product_tuples:
            start, end, only_hours = product

            # Find the maximum flex_downward in the 4-hour block
            max_flex_downward = 0
            current_time = start
            while current_time < end:
                flex_downward = max(
                    0,
                    unit.flex_power_requirement.at[current_time]
                    - unit.opt_power_requirement.at[current_time],
                )
                max_flex_downward = max(max_flex_downward, flex_downward)
                current_time += unit.index.freq  # Increment time by 1 hour

            # Skip the block if no downward flexibility is available
            if max_flex_downward <= 0:
                continue

            # Calculate the capacity price
            capacity_price = unit.calculate_marginal_cost(
                start=start, power=max_flex_downward
            )

            # Add the bid to the list
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": 0,
                    "volume": max_flex_downward,  # Negative for CRM_Neg
                    "node": unit.node,
                }
            )

        # Clean up empty bids
        bids = self.remove_empty_bids(bids)

        return bids

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
        # unit.determine_optimal_operation_with_flex()

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
