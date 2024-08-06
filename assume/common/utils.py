# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
import inspect
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import wraps
from itertools import groupby
from operator import itemgetter

import dateutil.rrule as rr
import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, LearningStrategy
from assume.common.market_objects import MarketProduct, Orderbook

logger = logging.getLogger(__name__)

freq_map = {
    "h": rr.HOURLY,
    "m": rr.MINUTELY,
    "d": rr.DAILY,
    "w": rr.WEEKLY,
}


def initializer(func):
    """
    Automatically assigns the parameters.

    Args:
        func (callable): The function to be initialized.

    Returns:
        callable: The wrapper function.

    Examples:
        >>> class process:
        ...     @initializer
        ...     def __init__(self, cmd, reachable=False, user='root'):
        ...         pass
        >>> p = process('halt', True)
        >>> p.cmd, p.reachable, p.user
        ('halt', True, 'root')
    """
    names, varargs, keywords, defaults, *_ = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def get_available_products(market_products: list[MarketProduct], startdate: datetime):
    """
    Get all available products for a given startdate.

    Args:
        market_products (list[MarketProduct]): List of market products.
        startdate (datetime.datetime): The startdate.

    Returns:
        list[MarketProduct]: List of available products.
    """
    options = []
    for product in market_products:
        start = startdate + product.first_delivery
        if isinstance(product.duration, rr.rrule):
            starts = list(product.duration.xafter(start, product.count + 1), inc=True)
            for i in range(product.count):
                period_start = starts[i]
                period_end = starts[i + 1]
                options.append((period_start, period_end, product.only_hours))
        else:
            for i in range(product.count):
                period_start = start + product.duration * i
                period_end = start + product.duration * (i + 1)
                options.append((period_start, period_end, product.only_hours))
    return options


def plot_orderbook(orderbook: Orderbook, results: list[dict]):
    """
    Plot the merit order of bids for each node in a separate subplot.

    Args:
        orderbook (Orderbook): The orderbook.
        results (list[dict]): The results of the clearing.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and axes of the plot.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bids = defaultdict(list)
    orderbook.sort(key=itemgetter("node"))
    for node_id, orders in groupby(orderbook, itemgetter("node")):
        bids[node_id].extend(list(map(itemgetter("price", "volume"), orders)))
    number_of_nodes = len(bids.keys()) or 1

    fig, ax = plt.subplots(1, number_of_nodes, sharey=True)
    if number_of_nodes == 1:
        ax = [ax]

    # split the bids into buy and sell bids for each node separately
    for i in range(number_of_nodes):
        # split the bids into buy and sell bids in lists of tuples
        supply_bids = [(price, quantity) for price, quantity in bids[i] if quantity > 0]
        demand_bids = [
            (price, -quantity) for price, quantity in bids[i] if quantity < 0
        ]

        # sort the bids by price
        supply_bids.sort(key=lambda x: x[0])
        demand_bids.sort(key=lambda x: x[0], reverse=True)

        # find the cumulative sum of the quantity of the bids
        cum_supply_bids = 0
        # find the cumulative sum of the quantity of the bids
        cum_demand_bids = 0

        # iterate through supply bids and plot them
        for n, bid in enumerate(supply_bids):
            price, quantity = bid
            ax[i].plot(
                [cum_supply_bids, cum_supply_bids + quantity], [price, price], "b-"
            )
            cum_supply_bids += quantity
            if n < len(supply_bids) - 1:
                ax[i].plot(
                    [cum_supply_bids, cum_supply_bids],
                    [price, supply_bids[n + 1][0]],
                    "b-",
                )
        # iterate through demand bids and plot them
        for n, bid in enumerate(demand_bids):
            price, quantity = bid
            ax[i].plot(
                [cum_demand_bids, cum_demand_bids + quantity], [price, price], "r-"
            )
            cum_demand_bids += quantity
            if n < len(demand_bids) - 1:
                ax[i].plot(
                    [cum_demand_bids, cum_demand_bids],
                    [price, demand_bids[n + 1][0]],
                    "r-",
                )
        # plot the market clearing price and quantity
        if len(results) == i:
            price = 0
            contracted_supply = 0
            contracted_demand = 0
        else:
            price = results[i]["price"]
            contracted_supply = results[i]["supply_volume"]
            contracted_demand = results[i]["demand_volume"]
        inflow = contracted_supply - contracted_demand
        ax[i].plot([contracted_supply, contracted_supply], [0, price], "k--")
        ax[i].plot([0, contracted_supply], [price, price], "k--")
        ax[i].plot(contracted_supply, price, "ko")

        # add text under the plot to show the market clearing price and quantity
        ax[i].text(0.05, -0.3, "Results:", transform=ax[i].transAxes)
        ax[i].text(0.05, -0.375, f"Price: {price:.1f}", transform=ax[i].transAxes)
        ax[i].text(
            0.05,
            -0.45,
            f"Accepted supply: {contracted_supply:.1f}",
            transform=ax[i].transAxes,
        )
        ax[i].text(
            0.05,
            -0.525,
            f"Accepted demand: {contracted_demand:.1f}",
            transform=ax[i].transAxes,
        )
        ax[i].text(0.05, -0.6, f"Total Export: {inflow:.1f}", transform=ax[i].transAxes)
        ax[i].set_title(f"Node {str(i)}")
        ax[i].set_xlabel("Quantity")
        ax[i].set_ylabel("Price")

        # plot legend outside the plot and only for last subplot
        if i == number_of_nodes - 1:
            ax[i].legend(
                handles=[
                    Line2D([0], [0], linewidth=1, color="b", label="Supply"),
                    Line2D([0], [0], linewidth=1, color="r", label="Demand"),
                ],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )

        # set x limits to 0 and max of supply or demand
        ax[i].set_xlim(0, max(cum_supply_bids, cum_demand_bids))
        ax[i].set_ylim(bottom=0)
    plt.subplots_adjust(wspace=0.3)

    return fig, ax


def visualize_orderbook(order_book: Orderbook):
    """
    Visualize the orderbook.

    Args:
        order_book (Orderbook): The orderbook.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    tab20_cmap = plt.get_cmap("tab20c")
    max_block_count = max([o.get("block_id", 1) for o in order_book])

    order_book.sort(key=itemgetter("block_id", "link"))
    start_times = sorted(set(o["start_time"] for o in order_book))
    y_past = pd.Series(0.0, index=start_times)
    for i, bids_grouped in groupby(order_book, itemgetter("block_id")):
        my_cmap_raw = np.array(tab20_cmap.colors) * i / max_block_count
        my_cmap = ListedColormap(my_cmap_raw)

        for j, o in groupby(bids_grouped, itemgetter("link")):
            s = pd.Series(0.0, index=start_times)
            o = list(o)
            for order in o:
                s[order["start_time"]] += order["volume"]
            if (s > 0).any():
                plt.bar(
                    s.index, s.values, bottom=y_past, color=my_cmap.colors[(j + 1) % 20]
                )
                y_past += s
    plt.title("Orderbook")
    plt.xlabel("hour")
    plt.xticks(rotation=80)
    plt.ylabel("MW")
    plt.show()


def aggregate_step_amount(orderbook: Orderbook, begin=None, end=None, groupby=None):
    """
    Step function with bought volume, allows setting timeframe through begin and end, and group by columns in groupby.

    Args:
        orderbook (Orderbook): The orderbook.
        begin (datetime, optional): The begin time. Defaults to None.
        end (datetime, optional): The end time. Defaults to None.
        groupby (list[str], optional): The columns to group by. Defaults to None.

    Returns:
        list[tuple[datetime, float, str, str]]: The aggregated orderbook timeseries.

    Examples:
        If called without groupby, this returns the aggregated orderbook timeseries
    """

    if groupby is None:
        groupby = []
    deltas = []

    # first we are creating a list of tuples with the following form:
    # start, delta_volume, bid_id, market_id
    for bid in orderbook:
        add = ()
        for field in groupby:
            add += (bid[field],)
        if bid["only_hours"] is None and not isinstance(bid["accepted_volume"], dict):
            deltas.append((bid["start_time"], bid["accepted_volume"]) + add)
            deltas.append((bid["end_time"], -bid["accepted_volume"]) + add)
        elif isinstance(bid["accepted_volume"], dict):
            start_hour = bid["start_time"]
            end_hour = bid["end_time"]
            duration = (start_hour - end_hour) / len(bid["accepted_volume"])
            for key in bid["accepted_volume"].keys():
                deltas.append((key, bid["accepted_volume"][key]) + add)
                deltas.append((key + duration, -bid["accepted_volume"][key]) + add)
        else:
            # only_hours allows to have peak or off-peak bids
            start_hour, end_hour = bid["only_hours"]
            duration_hours = end_hour - start_hour
            if duration_hours <= 0:
                duration_hours += 24

            starts = rr.rrule(
                rr.DAILY,
                dtstart=bid["start_time"],
                byhour=start_hour,
                until=bid["end_time"],
            )
            for date in starts:
                start = date
                end = date + timedelta(hours=duration_hours)
                deltas.append((start, bid["volume"]) + add)
                deltas.append((end, -bid["volume"]) + add)
    aggregation = defaultdict(list)
    # current_power is separated by group
    current_power = defaultdict(lambda: 0)
    for d_tuple in sorted(deltas, key=lambda i: i[0]):
        time, delta, *groupdata = d_tuple
        groupdata_str = "_".join(groupdata)
        current_power[groupdata_str] += delta
        # we don't know what the power will be at "end" yet
        # as a new order with this start point might be added
        # afterwards - so the end is excluded here
        # this also makes sure that each timestamp is only written
        # once when iteratively calling this function
        if (not begin or time >= begin) and (not end or time < end):
            if aggregation[groupdata_str] and aggregation[groupdata_str][-1][0] == time:
                aggregation[groupdata_str][-1][1] = current_power[groupdata_str]
            else:
                d_list = list(d_tuple)
                d_list[1] = current_power[groupdata_str]
                aggregation[groupdata_str].append(d_list)

    return [j for sub in list(aggregation.values()) for j in sub]


def get_test_demand_orders(power: np.ndarray):
    """
    Get test demand orders.

    Args:
        power (numpy.ndarray): Power array.

    Returns:
        pandas.DataFrame: DataFrame of demand orders.

    Examples:
        >>> power = np.array([100, 200, 150])
        >>> get_test_demand_orders(power)
    """

    order_book = {}
    for t in range(len(power)):
        order_book[t] = dict(
            type="demand", hour=t, block_id=t, name="DEM", price=3, volume=-power[t]
        )
    demand_order = pd.DataFrame.from_dict(order_book, orient="index")
    demand_order = demand_order.set_index(["block_id", "hour", "name"])
    return demand_order


def separate_orders(orderbook: Orderbook):
    """
    Separate orders with several hours into single hour orders.

    Args:
        orderbook (Orderbook): The orderbook.

    Returns:
        list: The updated orderbook.

    Notes:
        This function separates orders with several hours into single hour orders and modifies the orderbook in place.
    """

    # separate orders with several hours into single hour orders
    delete_orders = []
    for order in orderbook:
        if any([isinstance(value, dict) for value in order.values()]):
            start_hour = order["start_time"]
            end_hour = order["end_time"]
            order_len = max(
                len(value) for value in order.values() if isinstance(value, dict)
            )
            duration = (end_hour - start_hour) / order_len

            for start in pd.date_range(start_hour, end_hour - duration, freq=duration):
                single_order = order.copy()
                for key in order.keys():
                    if isinstance(order[key], dict):
                        single_order.update({key: order[key][start]})
                if single_order != order:
                    single_order.update(
                        {
                            "start_time": start,
                            "end_time": start + duration,
                        }
                    )

                orderbook.append(single_order)

            delete_orders.append(order)

    for order in delete_orders:
        orderbook.remove(order)

    return orderbook


def get_products_index(orderbook: Orderbook) -> pd.DatetimeIndex:
    """
    Creates an index containing all start times of orders in orderbook and all inbetween.

    Args:
        orderbook (Orderbook): The orderbook.

    Returns:
        pd.DatetimeIndex: The index containing all start times of orders in orderbook and all inbetween.
    """
    if orderbook == []:
        return []

    # get the minimum and maximum "start_time" for all orders in orderbook
    start_time = orderbook[0]["start_time"]
    end_time = orderbook[0]["start_time"]
    duration = orderbook[0]["end_time"] - orderbook[0]["start_time"]

    for order in orderbook:
        if order["start_time"] < start_time:
            start_time = order["start_time"]
        if order["end_time"] > end_time:
            end_time = order["end_time"]
        if order["end_time"] - order["start_time"] < duration:
            duration = order["end_time"] - order["start_time"]

    index_products = pd.date_range(
        start_time,
        end_time - duration,
        freq=duration,
    )

    return index_products


def timestamp2datetime(timestamp: float):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)


def datetime2timestamp(datetime: datetime):
    return calendar.timegm(datetime.utctimetuple())


def create_rrule(start, end, freq):
    freq, interval = convert_to_rrule_freq(freq)

    recurrency_rule = rr.rrule(
        freq=freq,
        interval=interval,
        dtstart=start,
        until=end,
        cache=True,
    )

    return recurrency_rule


def convert_to_rrule_freq(string: str) -> tuple[int, int]:
    """
    Convert a string to a rrule frequency and interval.

    Args:
        string (str): The string to be converted. Should be in the format of "1h" or "1d" or "1w".

    Returns:
        tuple[int, int]: The rrule frequency and interval.
    """
    freq = freq_map[string[-1]]
    interval = int(string[:-1])
    return freq, interval


def adjust_unit_operator_for_learning(
    bidding_strategies: dict[str, str],
    world_bidding_strategies: dict[str, BaseStrategy],
    unit_operator_id: str,
):
    """
    Check if any of the bidding strategies are learning strategies.
    And change the unit operator to RL if learning strategies are found.

    Args:
        bidding_strategies (dict[str, str]): The bidding strategies for the unit.
        world_bidding_strategies (dict[str, BaseStrategy]): The bidding strategies of the World
        unit_operator_id (str): The identifier of the unit operator.

    Returns:
        str: The corrected unit operator identifier.

    """
    if unit_operator_id == "Operator-RL":
        return unit_operator_id
    for strategy in bidding_strategies.values():
        if issubclass(world_bidding_strategies[strategy], LearningStrategy):
            unit_operator_id = "Operator-RL"
            logger.debug(
                "Your chosen unit operator %s for the learning unit %s was overwritten with 'Operator-RL', "
                "since all learning units need to be handeled by one unit operator.",
                unit_operator_id,
                id,
            )

    return unit_operator_id


def create_zonal_incidence_matrix(lines, buses, zones_id):
    nodes = buses[zones_id].unique()
    node_to_zone = buses[zones_id].to_dict()

    incidence_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    for _, line in lines.iterrows():
        zone0, zone1 = node_to_zone[line["bus0"]], node_to_zone[line["bus1"]]
        if zone0 != zone1:
            incidence_matrix.loc[zone0, zone1] += line["s_nom"]
            incidence_matrix.loc[zone1, zone0] += line["s_nom"]

    # Convert values below the diagonal to negative
    mask = np.tril(np.ones(incidence_matrix.shape), -1).astype(bool)
    incidence_matrix.values[mask] = -incidence_matrix.values[mask]

    return incidence_matrix


def create_nodal_incidence_matrix(lines, buses):
    nodes = buses.index.values
    node_index = {node: idx for idx, node in enumerate(nodes)}

    incidence_matrix = np.zeros((len(nodes), len(nodes)))

    for _, line in lines.iterrows():
        i, j = node_index[line["bus0"]], node_index[line["bus1"]]
        incidence_matrix[i, j] = line["s_nom"]
        incidence_matrix[j, i] = -line["s_nom"]

    return pd.DataFrame(incidence_matrix, index=nodes, columns=nodes)


def normalize_availability(powerplants_df, availability_df):
    # Create a copy of the availability dataframe to avoid modifying the original
    normalized_df = availability_df.copy()

    # Iterate through each column in the availability dataframe
    for column in normalized_df.columns:
        # Check if any value in the column is greater than 1
        if (normalized_df[column] > 1).any():
            try:
                # Get the max_power for the current unit from the powerplants dataframe
                max_power = powerplants_df.loc[column, "max_power"]

                # Normalize the entire column
                normalized_df[column] = normalized_df[column] / max_power

                # Ensure all values are between 0 and 1
                normalized_df[column] = normalized_df[column].clip(0, 1)
            except KeyError:
                logger.warning(
                    f"Unit '{column}' not found in powerplants dataframe. Skipping normalization for this unit."
                )

    return normalized_df
