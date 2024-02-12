# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd
import pytest

from assume.common.base import SupportsMinMax


class MockMarketConfig:
    market_id = "EOM"
    product_type = "energy"
    additional_fields = []


class MockMinMaxUnit(SupportsMinMax):
    def __init__(self, index, **kwargs):
        super().__init__("", "", "", {}, index, None, **kwargs)
        self.max_power = 1000
        self.min_power = 0
        self.ramp_down = 200
        self.ramp_up = 400

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        min = pd.Series(100, self.index).loc[start:start]
        max = pd.Series(400, self.index).loc[start:start]
        return min, max

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        return 3


@pytest.fixture
def mock_market_config():
    return MockMarketConfig()


@pytest.fixture
def mock_supports_minmax():
    index = pd.date_range(
        start=datetime(2023, 7, 1),
        end=datetime(2023, 7, 2),
        freq="1h",
    )
    return MockMinMaxUnit(index)
