# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from assume.common.base import SupportsMinMax
from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import NaiveForecast


class MockMarketConfig:
    market_id = "EOM"
    maximum_bid_price = 3000.0
    minimum_bid_price = -500.0
    product_type = "energy"
    additional_fields = []


class MockMinMaxUnit(SupportsMinMax):
    def __init__(self, forecaster, **kwargs):
        super().__init__("", "", "", {}, forecaster, None, **kwargs)
        self.max_power = 1000
        self.min_power = 0
        self.ramp_down = 200
        self.ramp_up = 400

    def calculate_min_max_power(
        self, start: datetime, end: datetime, product_type="energy"
    ) -> tuple[np.array, np.array]:
        min = FastSeries(value=100, index=self.index).loc[start:end]
        max = FastSeries(value=400, index=self.index).loc[start:end]
        return min, max

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        return 3


@pytest.fixture
def mock_market_config():
    return MockMarketConfig()


@pytest.fixture
def mock_supports_minmax():
    index = pd.date_range(
        start=datetime(2023, 7, 1), end=datetime(2023, 7, 2), freq="1h"
    )
    forecaster = NaiveForecast(index, demand=150)
    return MockMinMaxUnit(forecaster=forecaster)
