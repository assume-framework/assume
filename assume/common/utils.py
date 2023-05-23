import inspect
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from itertools import groupby
from operator import itemgetter

import dateutil.rrule as rr
import pandas as pd

from assume.common import MarketConfig, MarketProduct, Orderbook

logger = logging.getLogger(__name__)

freq_map = {"h": rr.HOURLY, "m": rr.MINUTELY}


def load_file(
    path: str,
    config: dict,
    file_name: str,
    index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    df = None

    if file_name in config:
        file_path = f"{path}/{config[file_name]}"
    else:
        file_path = f"{path}/{file_name}.csv"

    try:
        df = pd.read_csv(file_path, index_col=0, encoding="Latin-1")
        if index is not None:
            if len(df.index) != len(index):
                logger.warning(
                    f"Length of index does not match length of {file_name} dataframe. Attempting to resample."
                )
                df = attempt_resample(df, index)
            else:
                df.index = index

        for col in df:
            # check if the column is of dtype int
            if df[col].dtype == "int":
                # convert the column to float
                df[col] = df[col].astype(float)

        return df

    except FileNotFoundError:
        logger.warning(f"File {file_name} not found. Returning None")


def attempt_resample(
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(df.index) > len(index):
        temp_index = pd.date_range(start=index[0], end=index[-1], periods=len(df))
        df.index = temp_index
        df = df.resample(index.freq).mean()
        logger.info("Resampling successful.")
    elif len(df.index) < index.freq:
        raise ValueError("Index length mismatch. Upsampling not supported.")

    return df


def convert_to_rrule_freq(string):
    freq = freq_map[string[-1]]
    interval = int(string[:-1])
    return freq, interval


def make_market_config(
    id,
    market_params,
    world_start,
    world_end,
):
    freq, interval = convert_to_rrule_freq(market_params["opening_frequency"])
    start = market_params.get("start_date")
    end = market_params.get("end_date")
    if start:
        start = pd.Timestamp(start)
    if end:
        end = pd.Timestamp(end)
    start = start or world_start
    end = end or world_end

    market_products = []
    for product in market_params["products"]:
        market_products.append(
            MarketProduct(
                duration=pd.Timedelta(product["duration"]),
                count=product["count"],
                first_delivery=pd.Timedelta(product["first_delivery"]),
            )
        )

    market_config = MarketConfig(
        name=id,
        market_products=market_products,
        opening_hours=rr.rrule(
            freq=freq,
            interval=interval,
            dtstart=start,
            until=end,
            cache=True,
        ),
        additional_fields=market_params.get("additional_fields", []),
        product_type=market_params.get("product_type", "energy"),
        opening_duration=pd.Timedelta(market_params["opening_duration"]),
        maximum_gradient=market_params.get("max_gradient"),
        volume_unit=market_params.get("volume_unit"),
        volume_tick=market_params.get("volume_tick"),
        maximum_volume=market_params["maximum_volume"],
        price_tick=market_params.get("price_tick"),
        price_unit=market_params["price_unit"],
        market_mechanism=market_params["market_mechanism"],
        supports_get_unmatched=market_params.get("supports_get_unmatched", False),
    )

    return market_config


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


def get_available_products(market_products: list[MarketProduct], startdate: datetime):
    options = []
    for product in market_products:
        start = startdate + product.first_delivery
        if isinstance(product.duration, rr.rrule):
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


def plot_orderbook(orderbook: Orderbook, results):
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
