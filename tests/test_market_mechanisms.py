from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import get_available_products
from assume.markets import MarketRole, clearing_mechanisms

from .utils import create_orderbook

simple_dayahead_auction_config = MarketConfig(
    "simple_dayahead_auction",
    market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
    additional_fields=["node_id"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def test_market():
    mr = MarketRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    print(products)
    start = products[0][0]
    end = products[0][1]
    only_hours = products[0][2]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
        },
    ]
    simple_dayahead_auction_config.market_mechanism = clearing_mechanisms[
        simple_dayahead_auction_config.market_mechanism
    ]
    mr.all_orders = orderbook
    accepted, rejected, meta = simple_dayahead_auction_config.market_mechanism(
        mr, products
    )
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    import pandas as pd

    print(pd.DataFrame(mr.all_orders))
    print(pd.DataFrame(accepted))
    print(meta)


def test_market_mechanism():
    import copy

    for name, mechanism in clearing_mechanisms.items():
        print(name)
        market_config = copy.copy(simple_dayahead_auction_config)
        market_config.market_mechanism = mechanism
        mr = MarketRole(market_config)
        next_opening = market_config.opening_hours.after(datetime.now())
        products = get_available_products(market_config.market_products, next_opening)
        assert len(products) == 1
        order = {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": products[0][2],
        }

        orderbook = create_orderbook(order, node_ids=[0, 1, 2])
        mr.all_orders = orderbook
        accepted, rejected, meta = market_config.market_mechanism(mr, products)
        assert meta[0]["supply_volume"] > 0
        assert meta[0]["price"] > 0
        # import pandas as pd
        # print(pd.DataFrame(mr.all_orders))
        # print(pd.DataFrame(clearing_result))
        # print(meta)

    # return mr.all_orders, meta


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
