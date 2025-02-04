# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
from dateutil import rrule as rr
from mango import addr

from assume.markets.clearing_algorithms.contracts import (
    available_contracts,
    market_premium,
)
from assume.strategies.extended import is_co2emissionless


def test_contract_functions():
    start = datetime(2019, 1, 1)
    end = datetime(2019, 2, 1)

    index = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )

    contract = {
        "start_time": start,
        "end_time": end,
        "only_hours": None,
        "price": 10,
        "volume": 1000,
        "sender_id": "nuclear1",
        "eligible_lambda": is_co2emissionless,
        "evaluation_frequency": rr.WEEKLY,
        "agent_addr": addr("world", "my_operator"),
        "bid_id": "nuclear1_1",
        "unit_id": "nuclear1",
        "accepted_volume": 1000,
        "accepted_price": 0,
        "contract_price": 4.5,
        "contractor_unit_id": "demand1",
        "contractor_addr": addr("world", "brd"),
        "market_id": "Support",
    }

    market_idx = pd.Series(3, index)
    gen_series = pd.Series(1000, index)

    for c_function in available_contracts.values():
        result = c_function(contract, market_idx, gen_series, start, end)
        assert result

    result = market_premium(contract, market_idx, gen_series, start, end)
    assert result
