# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta

import pandas as pd
import pytest

from assume.common.base import BaseStrategy, BaseUnit
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, Orderbook, Product


class BasicStrategy(BaseStrategy):
    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        bids = []
        for product in product_tuples:
            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": 10,
                    "volume": 20,
                }
            )
        return bids


@pytest.fixture(params=["1h", "15min"])
def base_unit(request) -> BaseUnit:
    # Create a PowerPlant instance with some example parameters
    index = pd.date_range("2022-01-01", periods=4, freq=request.param)
    forecaster = NaiveForecast(
        index, availability=1, fuel_price=[10, 11, 12, 13], co2_price=[10, 20, 30, 30]
    )
    return BaseUnit(
        id="test_pp",
        unit_operator="test_operator",
        technology="base",
        bidding_strategies={"EOM": BasicStrategy()},
        forecaster=forecaster,
        index=forecaster.index,
    )


def test_init_function(base_unit, mock_market_config):
    assert base_unit.id == "test_pp"
    assert base_unit.unit_operator == "test_operator"
    assert base_unit.technology == "base"
    assert base_unit.as_dict() == {
        "id": "test_pp",
        "technology": "base",
        "unit_operator": "test_operator",
        "node": "node0",
        "unit_type": "base_unit",
    }


def test_output_before(base_unit, mock_market_config):
    index = base_unit.index
    assert base_unit.get_output_before(index[0], "energy") == 0
    assert base_unit.get_output_before(index[0], "fictional_product_type") == 0

    base_unit.outputs["energy"][index[0]] = 100
    assert base_unit.get_output_before(index[1], "energy") == 100
    assert base_unit.get_output_before(index[0], "energy") == 0

    base_unit.outputs["fictional_product_type"][index[0]] = 200
    assert base_unit.get_output_before(index[1], "fictional_product_type") == 200
    assert base_unit.get_output_before(index[0], "fictional_product_type") == 0


def test_calculate_bids(base_unit, mock_market_config):
    index = base_unit.index
    start = index[0]
    end = index[1]
    product_tuples = [(start, end, None)]
    bids = base_unit.calculate_bids(mock_market_config, product_tuples)
    assert bids == [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 10,
            "volume": 20,
        }
    ]

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 10,
            "volume": 10,
            "accepted_price": 11,
            "accepted_volume": 10,
        }
    ]
    # mock calculate_marginal_cost
    base_unit.calculate_marginal_cost = lambda *x: 10
    base_unit.set_dispatch_plan(mock_market_config, orderbook)
    base_unit.calculate_generation_cost(index[0], index[1], "energy")

    # we apply the dispatch plan of 10 MW
    assert base_unit.outputs["energy"][start] == 10
    assert base_unit.outputs["energy_marginal_costs"][start] == 10 * 10
    # we received more, as accepted_price is higher
    assert base_unit.outputs["energy_cashflow"][start] == 10 * 11

    # we somehow sold an additional 10 MW
    base_unit.set_dispatch_plan(mock_market_config, orderbook)
    base_unit.calculate_generation_cost(index[0], index[1], "energy")

    # the final output should be 10+10
    assert base_unit.outputs["energy"][start] == 20
    # the marginal cost for this volume should be twice as much too
    assert base_unit.outputs["energy_marginal_costs"][start] == 200
    assert base_unit.outputs["energy_cashflow"][start] == 20 * 11


def test_calculate_multi_bids(base_unit, mock_market_config):
    index = base_unit.index

    # when we set
    orderbook = [
        {
            "start_time": index[0],
            "end_time": index[1],
            "only_hours": None,
            "price": 10,
            "volume": 10,
            "accepted_price": 11,
            "accepted_volume": 10,
        },
        {
            "start_time": index[1],
            "end_time": index[2],
            "only_hours": None,
            "price": 10,
            "volume": 10,
            "accepted_price": 11,
            "accepted_volume": 10,
        },
    ]
    # mock calculate_marginal_cost
    base_unit.calculate_marginal_cost = lambda *x: 10
    base_unit.set_dispatch_plan(mock_market_config, orderbook)
    base_unit.calculate_generation_cost(index[0], index[1], "energy")

    assert base_unit.outputs["energy"][index[0]] == 10
    assert base_unit.outputs["energy_marginal_costs"][index[0]] == 100
    assert base_unit.outputs["energy_cashflow"][index[0]] == 110
    assert base_unit.outputs["energy"][index[1]] == 10
    assert base_unit.outputs["energy_marginal_costs"][index[1]] == 100
    assert base_unit.outputs["energy_cashflow"][index[1]] == 110

    base_unit.set_dispatch_plan(mock_market_config, orderbook)
    base_unit.calculate_generation_cost(index[0], index[1], "energy")

    # should be correctly applied for the sum, even if different hours are applied
    assert base_unit.outputs["energy"][index[0]] == 20
    assert base_unit.outputs["energy_marginal_costs"][index[0]] == 200
    assert base_unit.outputs["energy_cashflow"][index[0]] == 220
    assert base_unit.outputs["energy"][index[1]] == 20
    assert base_unit.outputs["energy_marginal_costs"][index[1]] == 200
    assert base_unit.outputs["energy_cashflow"][index[1]] == 220

    # in base_unit - this should not do anything but get return the energy dispatch
    assert (
        base_unit.execute_current_dispatch(index[0], index[2])
        == base_unit.outputs["energy"][index[0] : index[2]]
    ).all()


def test_clear_empty_bids(base_unit, mock_market_config):
    # Test empty bids
    bids = []
    index = base_unit.index
    for start in index:
        bids.append(
            {
                "start_time": start,
                "end_time": start + timedelta(hours=1),
                "only_hours": None,
                "price": 10,
                "volume": 0,
            }
        )
    assert (
        base_unit.bidding_strategies[mock_market_config.market_id].remove_empty_bids(
            bids
        )
        == []
    )

    # Test non-empty bids
    non_empty_bids = []
    for start in index:
        non_empty_bids.append(
            {
                "start_time": start,
                "end_time": start + timedelta(hours=1),
                "only_hours": None,
                "price": 10,
                "volume": 10,
            }
        )
    non_empty_bids_result = base_unit.bidding_strategies[
        mock_market_config.market_id
    ].remove_empty_bids(non_empty_bids)
    assert non_empty_bids_result == non_empty_bids

    # Test mixed empty and non-empty bids
    mixed_bids = []
    for start in index:
        mixed_bids.append(
            {
                "start_time": start,
                "end_time": start + timedelta(hours=1),
                "only_hours": None,
                "price": 10,
                "volume": 0 if start.hour % 2 == 0 else 10,
            }
        )
    mixed_bids_result = base_unit.bidding_strategies[
        mock_market_config.market_id
    ].remove_empty_bids(mixed_bids)
    assert mixed_bids_result == [bid for bid in mixed_bids if bid["volume"] > 0]


if __name__ == "__main__":
    pytest.main(["-s", __file__])
