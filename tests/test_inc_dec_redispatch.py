# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Replication of the two-node inc-dec example of Hirth & Schlecht (2019),
"Market-Based Redispatch in Zonal Electricity Markets".

The system (their Table 2) is an export-constrained North and an import-constrained
South joined by a single 30 GW line:

===============  ===================================================
North            20 GW wind @ 1 EUR/MWh, 20 x 1 GW coal @ 21..40,
                 5 x 1 GW diesel @ 66..70, no load
South            25 x 1 GW gas @ 41..65, 50 GW inelastic load
===============  ===================================================

Nodal prices are 30 EUR/MWh (North) and 60 EUR/MWh (South), their Table 3. When every
generator anticipates those prices in a zonal spot market, the North underbids and the
South overbids, the spot clears at 60 EUR/MWh, the line is overloaded by 15 GW and the
redispatch market clears at 30/60 EUR/MWh -- their Tables 7 and 8.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.fast_pandas import FastIndex
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products

pytest.importorskip("pypsa")

from assume.common.forecast_algorithms import (  # noqa: E402
    calculate_nodal_lmp_forecast,
    get_forecast_registries,
)
from assume.common.forecaster import (  # noqa: E402
    DemandForecaster,
    PowerplantForecaster,
)
from assume.markets.clearing_algorithms import (  # noqa: E402
    PayAsClearRole,
    RedispatchMarketRole,
)
from assume.strategies import (  # noqa: E402
    EnergyHeuristicRedispatchStrategy,
    EnergyNaiveRedispatchStrategy,
    EnergyNaiveStrategy,
)
from assume.units import Demand, PowerPlant  # noqa: E402

# results from the paper, in EUR/MWh and MW
NODAL_PRICE_NORTH = 30
NODAL_PRICE_SOUTH = 60
SPOT_PRICE_ANTICIPATED = 60
LINE_CAPACITY = 30_000
REDISPATCH_VOLUME = 15_000

# two delivery hours, so both markets clear each product twice
N_PRODUCTS = 2

buses = pd.DataFrame(
    {"name": ["north", "south"], "v_nom": [380.0, 380.0], "zone_id": ["DE", "DE"]}
).set_index("name")

lines = pd.DataFrame(
    {
        "name": ["line_north_south"],
        "bus0": ["north"],
        "bus1": ["south"],
        "s_nom": [float(LINE_CAPACITY)],
        "s_max_pu": [1.0],
        "x": [0.01],
        "r": [0.001],
    }
).set_index("name")


def _generator_table() -> pd.DataFrame:
    """The paper's merit order: one row per 1 GW unit (wind is a single 20 GW unit)."""
    rows = [("wind", "north", 20_000.0, 1.0)]
    rows += [(f"coal_{i}", "north", 1000.0, float(21 + i)) for i in range(20)]
    rows += [(f"diesel_{i}", "north", 1000.0, float(66 + i)) for i in range(5)]
    rows += [(f"gas_{i}", "south", 1000.0, float(41 + i)) for i in range(25)]
    return pd.DataFrame(
        rows, columns=["name", "node", "max_power", "marginal_cost"]
    ).set_index("name")


generators = _generator_table()

loads = pd.DataFrame(
    {"name": ["demand_south"], "node": ["south"], "max_power": [-50_000.0]}
).set_index("name")


@pytest.fixture
def index() -> FastIndex:
    return FastIndex(
        start=pd.Timestamp("2019-01-01 00:00"),
        end=pd.Timestamp("2019-01-01 05:00"),
        freq="h",
    )


@pytest.fixture
def eom_config() -> MarketConfig:
    """Zonal spot market. It carries the grid data so that the LMP forecast and the
    inc-dec strategy can see the network, but it clears as a single uniform-price zone."""
    return MarketConfig(
        market_id="EOM",
        market_products=[
            MarketProduct(timedelta(hours=1), N_PRODUCTS, timedelta(hours=1))
        ],
        opening_hours=rr.rrule(
            rr.HOURLY, dtstart=datetime(2019, 1, 1), until=datetime(2019, 1, 2)
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MWh",
        price_unit="EUR/MWh",
        maximum_bid_price=3000,
        minimum_bid_price=-500,
        market_mechanism="pay_as_clear",
        param_dict={"grid_data": {"buses": buses, "lines": lines}},
    )


@pytest.fixture
def redispatch_config() -> MarketConfig:
    return MarketConfig(
        market_id="Redispatch",
        market_products=[
            MarketProduct(timedelta(hours=1), N_PRODUCTS, timedelta(hours=1))
        ],
        additional_fields=["node", "min_power", "max_power"],
        opening_hours=rr.rrule(
            rr.HOURLY, dtstart=datetime(2019, 1, 1), until=datetime(2019, 1, 2)
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MWh",
        price_unit="EUR/MWh",
        maximum_bid_price=3000,
        minimum_bid_price=-500,
        market_mechanism="redispatch",
        param_dict={
            "grid_data": {
                "buses": buses,
                "lines": lines,
                "generators": generators,
                "loads": loads,
            },
            "payment_mechanism": "pay_as_bid",
        },
    )


def _build_units(index: FastIndex, powerplant_strategies: dict) -> list:
    """Build the paper's fleet. Availability is fixed at 1, fuel and CO2 are free, so a
    unit's marginal cost is exactly its ``additional_cost``."""
    registries = get_forecast_registries()
    fuel_prices = pd.DataFrame(
        0.0,
        index=index.as_datetimeindex(),
        columns=["renewable", "lignite", "diesel", "natural gas", "co2"],
    )

    units = []
    for name, gen in generators.iterrows():
        forecaster = PowerplantForecaster(
            index=index,
            availability=1.0,
            fuel_prices=fuel_prices,
            forecast_algorithms={"lmp": "lmp_nodal_forecast"},
            forecast_registries=registries,
        )
        units.append(
            PowerPlant(
                id=name,
                unit_operator="operator",
                technology=name.split("_")[0],
                bidding_strategies=powerplant_strategies,
                forecaster=forecaster,
                max_power=gen.max_power,
                min_power=0.0,
                efficiency=1.0,
                additional_cost=gen.marginal_cost,
                fuel_type="renewable",
                emission_factor=0.0,
                node=gen.node,
            )
        )

    for name, load in loads.iterrows():
        forecaster = DemandForecaster(
            index=index,
            availability=1.0,
            demand=pd.Series(load.max_power, index=index.as_datetimeindex()),
            forecast_registries=registries,
        )
        units.append(
            Demand(
                id=name,
                unit_operator="demand_operator",
                technology="inflex_demand",
                bidding_strategies={
                    "EOM": EnergyNaiveStrategy(),
                    "Redispatch": EnergyNaiveRedispatchStrategy(),
                },
                forecaster=forecaster,
                max_power=load.max_power,
                min_power=0.0,
                node=load.node,
            )
        )

    return units


@pytest.fixture
def units(index, eom_config) -> list:
    calculate_nodal_lmp_forecast.cache_clear()
    strategies = {
        "EOM": EnergyHeuristicRedispatchStrategy(),
        "Redispatch": EnergyHeuristicRedispatchStrategy(),
    }
    units = _build_units(index, strategies)
    for unit in units:
        unit.forecaster.initialize(tuple(units), (eom_config,), None, unit)
    return units


def _collect_bids(units, market_config, products) -> list:
    """Run every unit's strategy and tag the bids like the units operator would."""
    orderbook = []
    for unit in units:
        for i, bid in enumerate(
            unit.bidding_strategies[market_config.market_id].calculate_bids(
                unit, market_config, products
            )
        ):
            bid["unit_id"] = unit.id
            bid["bid_id"] = f"{unit.id}_{i}"
            orderbook.append(bid)
    return orderbook


# ---------------------------------------------------------------------------
# LMP forecast
# ---------------------------------------------------------------------------


@pytest.mark.require_network
def test_lmp_forecast_matches_paper_nodal_prices(units, index):
    """The LMP forecast must reproduce the paper's nodal pricing benchmark (Table 3):
    30 EUR/MWh in the export-constrained North, 60 EUR/MWh in the import-constrained
    South, for every hour."""
    lmp = units[0].forecaster.lmp

    assert set(lmp) == {"north_lmp", "south_lmp"}
    assert list(lmp["north_lmp"]) == pytest.approx([NODAL_PRICE_NORTH] * len(index))
    assert list(lmp["south_lmp"]) == pytest.approx([NODAL_PRICE_SOUTH] * len(index))


@pytest.mark.require_network
def test_lmp_forecast_is_shared_by_all_plants(units):
    """Every plant that opts into the LMP forecast sees the same nodal prices for both
    nodes, regardless of where it sits. The demand unit does not opt in, so it keeps an
    empty LMP forecast."""
    plants = [unit for unit in units if isinstance(unit, PowerPlant)]
    assert len(plants) == len(generators)

    reference = plants[0].forecaster.lmp
    for plant in plants:
        for node in ("north", "south"):
            assert list(plant.forecaster.lmp[f"{node}_lmp"]) == pytest.approx(
                list(reference[f"{node}_lmp"])
            )

    demand = next(unit for unit in units if isinstance(unit, Demand))
    assert demand.forecaster.lmp == {}


@pytest.mark.require_network
def test_lmp_forecast_zonal_collapses_to_single_price(index, eom_config):
    """With a ``zones_identifier`` the nodal LMPs collapse to one price per zone. Both
    buses are in zone DE, so both get the mean of 30 and 60."""
    calculate_nodal_lmp_forecast.cache_clear()
    eom_config.param_dict["zones_identifier"] = "zone_id"

    units = _build_units(index, {"EOM": EnergyNaiveStrategy()})
    for unit in units:
        unit.forecaster.initialize(tuple(units), (eom_config,), None, unit)

    lmp = units[0].forecaster.lmp
    expected = (NODAL_PRICE_NORTH + NODAL_PRICE_SOUTH) / 2
    assert list(lmp["north_lmp"]) == pytest.approx([expected] * len(index))
    assert list(lmp["south_lmp"]) == pytest.approx([expected] * len(index))


def test_lmp_forecast_without_grid_data_is_empty(index, eom_config):
    """Without grid data there are no nodes to price, so the forecast stays empty."""
    calculate_nodal_lmp_forecast.cache_clear()
    eom_config.param_dict["grid_data"] = {}

    units = _build_units(index, {"EOM": EnergyNaiveStrategy()})
    units[0].forecaster.initialize(tuple(units), (eom_config,), None, units[0])

    assert units[0].forecaster.lmp == {}


# ---------------------------------------------------------------------------
# EnergyHeuristicRedispatchStrategy
# ---------------------------------------------------------------------------


@pytest.mark.require_network
def test_strategy_bids_reproduce_inc_dec_spot_equilibrium(units, eom_config):
    """Spot market with anticipation (paper Table 7).

    Northern units above the northern LMP underbid down to 30 EUR/MWh, southern units
    below the southern LMP overbid up to 60 EUR/MWh, and the zonal market clears at
    60 EUR/MWh with the whole North (45 GW) plus 5 GW of southern gas dispatched --
    a 45 GW flow on a 30 GW line.
    """
    products = get_available_products(
        eom_config.market_products, eom_config.opening_hours.after(datetime(2019, 1, 1))
    )
    assert len(products) == N_PRODUCTS

    orderbook = _collect_bids(units, eom_config, products)
    bid_price = {(o["unit_id"], o["start_time"]): o["price"] for o in orderbook}

    for start, _, _ in products:
        # North: cheap wind and coal bid their cost, everything above the northern LMP
        # underbids to exactly that LMP.
        assert bid_price[("wind", start)] == pytest.approx(1)
        assert bid_price[("coal_0", start)] == pytest.approx(21)
        assert bid_price[("coal_19", start)] == pytest.approx(NODAL_PRICE_NORTH)
        assert bid_price[("diesel_4", start)] == pytest.approx(NODAL_PRICE_NORTH)
        # South: gas below the southern LMP overbids to it, above it bids its cost.
        assert bid_price[("gas_0", start)] == pytest.approx(NODAL_PRICE_SOUTH)
        assert bid_price[("gas_19", start)] == pytest.approx(NODAL_PRICE_SOUTH)
        assert bid_price[("gas_24", start)] == pytest.approx(65)

    accepted, _, meta, _ = PayAsClearRole(eom_config).clear(orderbook, products)

    assert [m["price"] for m in meta] == pytest.approx(
        [SPOT_PRICE_ANTICIPATED] * N_PRODUCTS
    )

    node_of = generators["node"].to_dict()
    for start, _, _ in products:
        dispatch = {"north": 0.0, "south": 0.0}
        for order in accepted:
            if order["unit_id"] in node_of and order["start_time"] == start:
                dispatch[node_of[order["unit_id"]]] += order["accepted_volume"]
        # all 45 GW of the North is scheduled, the remaining 5 GW comes from the South
        assert dispatch["north"] == pytest.approx(45_000)
        assert dispatch["south"] == pytest.approx(5_000)
        # the resulting flow exceeds the line by exactly the redispatch volume
        assert dispatch["north"] - LINE_CAPACITY == pytest.approx(REDISPATCH_VOLUME)


@pytest.mark.require_network
def test_strategy_redispatch_clears_at_paper_prices(
    units, eom_config, redispatch_config
):
    """Redispatch market with anticipation (paper Table 8).

    After the gamed spot market the TSO must buy 15 GW: the North is redispatched down
    at 30 EUR/MWh and the South up at 60 EUR/MWh, for a net cost of 450,000 EUR per
    hour -- the paper's Table 11.
    """
    products = get_available_products(
        eom_config.market_products, eom_config.opening_hours.after(datetime(2019, 1, 1))
    )

    # 1. clear the spot market and hand the dispatch back to the units
    eom_orderbook = _collect_bids(units, eom_config, products)
    accepted, rejected, _, _ = PayAsClearRole(eom_config).clear(eom_orderbook, products)
    by_unit = {unit.id: unit for unit in units}
    for order in accepted + rejected:
        by_unit[order["unit_id"]].set_dispatch_plan(eom_config, [order])

    # 2. the redispatch bid repeats the spot bid, so the anticipated LMP is offered again
    rd_orderbook = _collect_bids(units, redispatch_config, products)
    accepted_rd, _, meta, _ = RedispatchMarketRole(redispatch_config).clear(
        rd_orderbook, products
    )

    node_of = generators["node"].to_dict()
    for start, _, _ in products:
        up = [
            o
            for o in accepted_rd
            if o["start_time"] == start and o["accepted_volume"] > 0
        ]
        down = [
            o
            for o in accepted_rd
            if o["start_time"] == start and o["accepted_volume"] < 0
        ]

        assert sum(o["accepted_volume"] for o in up) == pytest.approx(REDISPATCH_VOLUME)
        assert sum(o["accepted_volume"] for o in down) == pytest.approx(
            -REDISPATCH_VOLUME
        )

        # upward redispatch happens in the South only, downward in the North only
        assert {node_of[o["unit_id"]] for o in up} == {"south"}
        assert {node_of[o["unit_id"]] for o in down} == {"north"}

        # pay-as-bid on bids that all sit at the anticipated nodal price
        assert [o["accepted_price"] for o in up] == pytest.approx(
            [NODAL_PRICE_SOUTH] * len(up)
        )
        assert [o["accepted_price"] for o in down] == pytest.approx(
            [NODAL_PRICE_NORTH] * len(down)
        )

        # net redispatch cost, paid by consumers through grid fees
        cost = sum(o["accepted_price"] * abs(o["accepted_volume"]) for o in up) - sum(
            o["accepted_price"] * abs(o["accepted_volume"]) for o in down
        )
        assert cost == pytest.approx(450_000)

    assert len(meta) == N_PRODUCTS * len(buses)
