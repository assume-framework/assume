import inspect
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from itertools import groupby
from operator import itemgetter

import dateutil.rrule as rr
import pandas as pd

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook

logger = logging.getLogger(__name__)


def initializer(func):
    """
    Automatically assigns the parameters.
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
    Plot the merit order of bids for each node in a separate subplot
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bids = defaultdict(list)
    orderbook = sorted(orderbook, key=itemgetter("node_id"))
    for node_id, orders in groupby(orderbook, itemgetter("node_id")):
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
                    Line2D([0], [0], 1, color="b", label="Supply"),
                    Line2D([0], [0], 1, color="r", label="Demand"),
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


def aggregate_step_amount(orderbook: Orderbook, begin=None, end=None, groupby=None):
    """
    step function with bought volume
    allows setting timeframe through begin and end
    and group by columns in groupby.
    This allows to have separate time series per market and bid_id/unit_id.
    The orderbook must contain all relevant orders.
    E.g. to calculate the current volume from 01.06 to 02.06, a yearly base
    order from 01.01-31.12 must also be given, to be considered.

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
        if bid["only_hours"] is None:
            deltas.append((bid["start_time"], bid["volume"]) + add)
            deltas.append((bid["end_time"], -bid["volume"]) + add)
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
    aggregation = defaultdict(lambda: [])
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
        # once when iterativley calling this function
        if (not begin or time >= begin) and (not end or time < end):
            if aggregation[groupdata_str] and aggregation[groupdata_str][-1][0] == time:
                aggregation[groupdata_str][-1][1] = current_power[groupdata_str]
            else:
                d_list = list(d_tuple)
                d_list[1] = current_power[groupdata_str]
                aggregation[groupdata_str].append(d_list)

    return [j for sub in list(aggregation.values()) for j in sub]
