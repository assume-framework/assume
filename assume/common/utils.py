import inspect
from collections import defaultdict
from datetime import datetime
from functools import wraps
from itertools import groupby
from math import isclose, log10
from operator import itemgetter

from dateutil import rrule

from .marketclasses import MarketProduct


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
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):

        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def is_mod_close(a, mod_b):
    """
    due to floating point, a mod b can be very close to 0 or very close to mod_b
    """
    abs_tol = 1e-10
    # abs_tol needed for comparison near zero
    return isclose(a % mod_b, 0, abs_tol=abs_tol) or isclose(
        a % mod_b, mod_b, abs_tol=abs_tol
    )


def round_digits(n, tick_size):
    """
    rounds n to the power of 10 of the needed tick_size
    Example_
    >>> round_digits(1.100001, 0.1)
    1.1
    >>> round_digits(400.1, 20)
    400
    """
    return round(n, 1 - int(log10(tick_size)))


def get_available_products(market_products: list[MarketProduct], startdate: datetime):
    options = []
    for product in market_products:
        start = startdate + product.first_delivery_after_start
        if isinstance(product.duration, rrule.rrule):
            starts = list(product.duration.xafter(start, product.count + 1))
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


def plot_orderbook(orderbook, results):
    """
    Plot the merit order of bids for each node in a separate subplot
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bids = defaultdict(list)
    orderbook = sorted(orderbook, key=itemgetter("node_id"))
    for node_id, orders in groupby(orderbook, itemgetter("node_id")):
        bids[node_id].extend(list(map(itemgetter("price", "volume"), orders)))
    number_of_nodes = len(bids.keys())

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
