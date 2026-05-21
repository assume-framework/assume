# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta
from dateutil import rrule as rr
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from assume.common.fast_pandas import FastIndex
from assume.common.forecast_algorithms import (
    calculate_naive_congestion_signal,
    calculate_naive_price,
    calculate_naive_price_inelastic,
    calculate_naive_renewable_utilisation,
    calculate_naive_residual_load,
    get_forecast_registries,
)
from assume.common.forecaster import (
    DemandForecaster,
    PowerplantForecaster,
)
from assume.common.forecast_algorithms import calculate_locational_marginal_price
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import EnergyHeuristicElasticStrategy, EnergyNaiveStrategy
from assume.units import Demand, PowerPlant

path = Path("./tests/fixtures/forecast_init")

parse_date = {"index_col": "datetime", "parse_dates": ["datetime"]}

@pytest.fixture
def index():
    return pd.DatetimeIndex(
        pd.date_range("2019-01-01 08:00", periods=7, freq="h"),
    )


@pytest.fixture
def shared_FastIndex(index):
    return FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))


@pytest.fixture
def simple_nodal_market_config():
    return MarketConfig(
        market_id="simple_nodal",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        additional_fields=["node"],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2005, 6, 1),
            until=datetime(2005, 6, 2),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MW",
        volume_tick=0.1,
        maximum_bid_volume=None,
        price_unit="€/MW",
        market_mechanism="nodal_clearing",
    )


eps = 1e-4

nodes = pd.DataFrame(
    {
        "name": ["N", "S"],
        "v_nom": [380.0, 380.0],
        "zones": ["zone1", "zone1"],
    }
).set_index("name")
lines = pd.DataFrame(
    {
        "name": ["line_N_S"],
        "bus0": ["N"],
        "bus1": ["S"],
        "s_nom": [500.0],
        "x": [0.01],
        "r": [0.001],
    }
).set_index("name")
generators = pd.DataFrame(
    {
        "name": ["coal_N", "coal_S", "gas_S"],
        "node": ["N", "S", "S"],
        "marginal_cost": [10.0, 50.0, 100.0],
        "max_power": [1000.0, 1000.0, 1000.0],
    }
).set_index("name")

demand = pd.DataFrame(
    {
        "name": ["dem_S"],
        "node": ["S"],
        "max_power": [-1000.0],
    }
).set_index("name")

@pytest.fixture
def grid_data_dict_2_nodes():
    return {
        "buses": nodes,
        "lines": lines,
        "generators": generators,
        "loads": demand,
    }

@pytest.fixture
def units(shared_FastIndex):
    return {
        "coal_N": PowerPlant(
            id="coal_N",
            node="N",
            marginal_cost=10.0,
            max_power=1000.0,
            unit_operator="UO1",
            technology="coal",
            bidding_strategies={"EOM": EnergyNaiveStrategy()},
            forecaster=PowerplantForecaster(index=shared_FastIndex),
        ),
        "coal_S": PowerPlant(
            id="coal_S",
            node="S",
            marginal_cost=50.0,
            max_power=1000.0,
            unit_operator="UO2",
            technology="coal",
            bidding_strategies={"EOM": EnergyNaiveStrategy()},
            forecaster=PowerplantForecaster(index=shared_FastIndex),
        ),
        "gas_S": PowerPlant(
            id="gas_S",
            node="S",
            marginal_cost=100.0,
            max_power=1000.0,
            unit_operator="UO3",
            technology="gas",
            bidding_strategies={"EOM": EnergyNaiveStrategy()},
            forecaster=PowerplantForecaster(index=shared_FastIndex),
        ),
        "dem_S": Demand(
            id="dem_S",
            node="S",
            max_power=-1000.0,
            min_power=0.0,
            unit_operator="demand_UO",
            technology="demand",
            bidding_strategies={"EOM": EnergyNaiveStrategy()},
            forecaster=DemandForecaster(index=shared_FastIndex, demand=-1000),
        ),
    }

def test_calculate_lmp_no_grid(shared_FastIndex, units, simple_nodal_market_config):
    """Test LMP calculation with no grid data."""
    # should raise ValueError
    with pytest.raises(
        ValueError,
        match="Grid data (buses and lines) must be provided in market config for LMP calculation.",
    ):
        calculate_locational_marginal_price(
            index=shared_FastIndex,
            units=units,
            config=simple_nodal_market_config,
        )

def test_calculate_lmp_infeasible(shared_FastIndex, units, simple_nodal_market_config):
    """Test LMP calculation with infeasible network."""
    simple_nodal_market_config.param_dict["grid_data"] = grid_data_dict_2_nodes
    # should raise exception because demand cannot be met
    units.loc[:, "max_power"] = 0.0
    with pytest.raises(Exception, match="Solver in nodal clearing forecast did not converge"):
        calculate_locational_marginal_price(
            index=shared_FastIndex,
            units=units,
            config=simple_nodal_market_config,
        )

def test_calculate_lmp_2_nodes_1_zone(shared_FastIndex, units, simple_nodal_market_config, grid_data_dict_2_nodes):
    """Test LMP calculation with 2 nodes that have the same zone identifier."""
    # should return same price for both nodes, even if congested
    simple_nodal_market_config.param_dict["grid_data"] = grid_data_dict_2_nodes
    simple_nodal_market_config.param_dict["zones_identifier"] = "zones"
    lmp = calculate_locational_marginal_price(
        index=shared_FastIndex,
        units=units,
        config=simple_nodal_market_config,
    )
    assert_series_equal(lmp["N"], lmp["S"], check_names=False)

def test_calculate_lmp_no_congestion(shared_FastIndex, units, simple_nodal_market_config, grid_data_dict_2_nodes):
    """Test LMP calculation with no congestion."""
    # should return same price for all nodes
    simple_nodal_market_config.param_dict["grid_data"] = grid_data_dict_2_nodes
    # adjust line s_nom to very high value to ensure no congestion
    simple_nodal_market_config.param_dict["grid_data"]["lines"].loc["line_N_S", "s_nom"] = 1e6
    lmp = calculate_locational_marginal_price(
        index=shared_FastIndex,
        units=units,
        config=simple_nodal_market_config,
    )
    assert_series_equal(lmp["N"], lmp["S"], check_names=False)

def test_calculate_lmp_congestion(shared_FastIndex, units, simple_nodal_market_config, grid_data_dict_2_nodes):
    """Test LMP calculation with congestion."""
    # should return different prices for nodes on different sides of congestion
    simple_nodal_market_config.param_dict["grid_data"] = grid_data_dict_2_nodes
    # adjust line s_nom to low value to ensure congestion
    simple_nodal_market_config.param_dict["grid_data"]["lines"].loc["line_N_S", "s_nom"] = 50.0
    lmp = calculate_locational_marginal_price(
        index=shared_FastIndex,
        units=units,
        config=simple_nodal_market_config,
    )
    assert not lmp["N"].equals(lmp["S"])
