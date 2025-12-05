# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

import datetime

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from assume.units.powerplant import PowerPlant

from assume.common.forecaster import DemandForecaster, PowerplantForecaster
from assume.units.demand import Demand
from tests.utils import index, setup_simple_world

import dateutil.rrule as rr

import warnings


world = setup_simple_world()


@pytest.fixture
def demand():
    return Demand(
        id="test_unit",
        unit_operator="test_operator",
        min_power=0,
        max_power=-1000,
        technology="demand",
        bidding_strategies={},
        forecaster=DemandForecaster(index, demand=-100),
    )


@pytest.fixture
def power_plant():
    params_dict = {
        "bidding_strategies": {"EOM": NaiveSingleBidStrategy()},
        "technology": "energy",
        "unit_operator": "test_operator",
        "max_power": 10,
        "min_power": 0,
        "forecaster": PowerplantForecaster(index),
    }
    return PowerPlant("testdemand", **params_dict)


@pytest.fixture
def grid_data():
    """ A simple mock grid. """

    bus_data = {"name": ["node1"], "v_nom": [1.0]}
    buses = pd.DataFrame(bus_data).set_index("name")
    line_data = {}
    lines = pd.DataFrame(line_data)
    generator_data = {"name": "gen1", "node": ["node1"], "max_power": [5.0]}
    generators = pd.DataFrame(generator_data).set_index("name")
    load_data = {"name": "load1", "node": ["node1"], "max_power": [5.0]}
    loads = pd.DataFrame(load_data).set_index("name")

    return {"buses": buses,
            "lines": lines,
            "generators": generators,
            "loads": loads}

def test_warning_no_generation(demand):
    """ Running a Wworld with demand but no generation raises a Warning. """
    world.add_unit_operator("test_operator")
    world.add_unit_instance(operator_id="test_operator", unit=demand)
    with pytest.warns(UserWarning) as record:
        world.run()
    assert len(record) > 0


def test_warning_no_demand(power_plant):
    """ Running a World with generation but no demand raises a Warning. """
    world.add_unit_operator("test_operator")
    world.add_unit_instance(operator_id="test_operator", unit=power_plant)
    with pytest.warns(UserWarning) as record:
        world.run()
    assert len(record) > 0


def test_market_too_early():
    """ A market that opens before simulation time, raises an Error. """
    world.add_market_operator("test_operator")
    market_start = world.start - datetime.timedelta(hours=1)
    market_end = world.end
    market_opening = rr.rrule(rr.HOURLY, dtstart=market_start, until=market_end)
    market_products= [MarketProduct(datetime.timedelta(hours=1),
                                     1,
                                     datetime.timedelta(hours=1))]
    market_config = MarketConfig(
        "test_EOM",
        opening_hours=market_opening,
        market_products=market_products)
    
    world.add_market(
        market_operator_id="test_operator",
        market_config=market_config)
    
    with pytest.raises(ValueError):
        world.run()


def test_market_too_late():
    """ A market that closes after simulation time, raises an Error. """
    world.add_market_operator("test_operator")
    market_start = world.start 
    market_end = world.end + datetime.timedelta(hours=1)
    market_opening = rr.rrule(rr.HOURLY, dtstart=market_start, until=market_end)
    market_products= [MarketProduct(
        duration=datetime.timedelta(hours=1),
        count=1,
        first_delivery=datetime.timedelta(hours=1))]
    market_config = MarketConfig(
        "test_EOM",
        opening_hours=market_opening,
        market_products=market_products)
    
    world.add_market(
        market_operator_id="test_operator",
        market_config=market_config)
    
    with pytest.raises(ValueError):
        world.run()


def test_refering_non_existant_market():
    """A UnitOperator referencing a non-existant Market, raises an Error. """
    
    strategies = {"none_existant_market": "naive_eom"}
    world.add_unit_operator("unit_operator", strategies)

    with pytest.raises(ValueError):
        world.run()
    

def test_non_referred_market():
    """ A Market referred by no UnitOperator raises a Warning. """
    world.add_market_operator("market_operator")
    world.add_unit_operator("unit_operator")
    market_opening = rr.rrule(rr.HOURLY, dtstart=world.start, until=world.end)
    market_products= [MarketProduct(
        duration=datetime.timedelta(hours=1),
        count=1,
        first_delivery=datetime.timedelta(hours=1))]
    market_config = MarketConfig(
        "test_EOM",
        opening_hours=market_opening,
        market_products=market_products)
    
    world.add_market(
        market_operator_id="market_operator",
        market_config=market_config)

    with pytest.warns(UserWarning) as record:
        world.run()
    assert len(record) > 0
    

def test_redispatch_no_dispatch(grid_data):
    """ A redispatch without a dispatch market raises an Error. """

    world.add_market_operator("market_operator")
    world.add_unit_operator("unit_operator")
    market_opening = rr.rrule(rr.HOURLY, dtstart=world.start, until=world.end)
    market_products= [MarketProduct(
        duration=datetime.timedelta(hours=1),
        count=1,
        first_delivery=datetime.timedelta(hours=1))]
        
    redispatch_config = MarketConfig(
        "redispatch",
        market_mechanism="redispatch",
        opening_hours=market_opening,
        market_products=market_products,
        param_dict={"grid_data": grid_data},
        additional_fields=["node", "max_power", "min_power"])

    world.add_market(
        market_operator_id="market_operator",
        market_config=redispatch_config)

    with pytest.raises(ValueError):
        world.run()


def test_redispatch_too_early(grid_data):
    """ A redispatch market opening before dispatch closure raises an Error. """

    world.add_market_operator("market_operator")
    world.add_unit_operator("unit_operator")
    opening_dispatch = rr.rrule(rr.HOURLY, dtstart=world.start, until=world.end)
    opening_redispatch = opening_dispatch
    
    market_products= [MarketProduct(
        duration=datetime.timedelta(hours=1),
        count=1,
        first_delivery=datetime.timedelta(hours=1))]
    
    redispatch_config = MarketConfig(
        "dispatch",
        opening_hours=opening_dispatch,
        market_products=market_products)
    
    dispatch_config = MarketConfig(
        "redispatch",
        market_mechanism="redispatch",
        opening_hours=opening_redispatch,
        market_products=market_products,
        param_dict={"grid_data": grid_data},
        additional_fields=["node", "max_power", "min_power"])

    world.add_market(
        market_operator_id="market_operator",
        market_config=dispatch_config)

    world.add_market(
        market_operator_id="market_operator",
        market_config=redispatch_config)

    with pytest.raises(ValueError):
        world.run()
    