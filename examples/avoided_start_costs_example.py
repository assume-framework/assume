# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DMAS Powerplant Strategy Avoided Startup Costs Demonstration.

This example validates and demonstrates:
1. Economic trade-off between shutting down during a short price dip vs. keeping
   the plant running at minimum power to avoid expensive startup costs later.
2. The 48-hour rolling lookahead in PowerplantDmasStrategy.
3. Generation of complex DMAS orders (single asks and linked orders) reflecting the
   avoided startup cost savings.
4. Market clearing using ComplexDmasClearingRole.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import rrule as rr

from assume.common.forecaster import PowerplantForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms.complex_clearing_dmas import (
    ComplexDmasClearingRole,
)
from assume.strategies.dmas_powerplant import (
    PowerplantDmasStrategy,
)
from assume.units import PowerPlant


def run_avoided_start_costs_demonstration():
    print("=" * 75)
    print("DMAS Powerplant Strategy: Avoided Startup Costs & Market Clearing")
    print("=" * 75)

    # 1. Setup 48-hour time horizon
    start_time = datetime(2022, 1, 1, 0, 0)
    index = pd.date_range(start_time, periods=48, freq="h")

    # 2. Build 48h electricity price scenario
    # - Fuel cost: €20/MWh_th, CO2: €30/t, eff: 0.5, emission factor: 0.35 -> marginal cost = ~€61/MWh
    # - Day 1 daytime (0-20): High prices (€80/MWh) -> profitable operation
    # - Day 1 night (21-23): Price dips to €45/MWh (< marginal cost €61) -> naive dispatch would shut down!
    # - Day 2 (24-47): Prices surge again (€85/MWh) -> plant wants to restart!
    day1_prices = [80.0] * 21 + [45.0, 45.0, 45.0]  # hours 0..23
    day2_prices = [85.0] * 24  # hours 24..47
    prices_48h = pd.Series(day1_prices + day2_prices, index=index)

    fuel_prices = pd.Series(20.0, index=index)  # €20 / MWh_th
    co2_prices = pd.Series(30.0, index=index)  # €30 / t_CO2

    forecaster = PowerplantForecaster(
        index,
        availability=1.0,
        fuel_prices={"gas": fuel_prices, "co2": co2_prices},
        market_prices={"dayahead": prices_48h},
    )

    # 3. Create a 500 MW Combined-Cycle Gas Turbine with €50,000 cold startup cost
    strategy = PowerplantDmasStrategy()
    plant = PowerPlant(
        id="CCGT_500MW",
        unit_operator="PowerGenCorp",
        technology="CCGT",
        fuel_type="gas",
        efficiency=0.5,
        emission_factor=0.35,  # 0.35 t_CO2 / MWh_th
        cold_start_cost=100.0,  # €100/MW * 500 MW = €50,000 total startup cost
        min_operating_runtime=4,
        min_operating_offtime=2,
        bidding_strategies={"dayahead": strategy},
        max_power=500.0,
        min_power=150.0,
        ramp_up=500.0,
        ramp_down=500.0,
        initial_state=1,  # initially online
        initial_runtime=10,
        forecaster=forecaster,
    )

    marginal_cost = (20.0 / 0.5) + (30.0 * 0.35 / 0.5)  # 40 + 21 = €61/MWh
    print("\nPlant Technical & Economic Specifications:")
    print(f"  - Capacity: {plant.max_power} MW (min power: {plant.min_power} MW)")
    print(f"  - Marginal Generation Cost: ~€{marginal_cost:.2f}/MWh")
    print(f"  - Cold Start Cost: €{plant.cold_start_cost:,.2f}")
    print(
        f"  - Hours 21-23 Electricity Price: €{day1_prices[21]:.2f}/MWh (< €{marginal_cost:.2f})"
    )
    print(f"  - Day 2 Electricity Price: €{day2_prices[0]:.2f}/MWh")

    # 4. Run optimization with rolling 48h horizon
    print("\nRunning DMAS Powerplant Optimization (with 48h horizon analysis)...")
    strategy.optimize(
        unit=plant,
        start=start_time,
        hour_count=24,
        prices=prices_48h,
    )

    prevent_info = strategy.prevented_start
    print("\nAvoided Start Cost Analysis Results:")
    print(f"  - Avoided Start Condition Triggered: {prevent_info['prevent']}")
    print(
        f"  - Identified Prevented Off Hours: {np.where(prevent_info['hours'] == 1)[0].tolist()}"
    )
    print(f"  - Net Economic Saving from Continuous Run: €{prevent_info['delta']:,.2f}")

    assert prevent_info["prevent"] is True, (
        "Strategy should detect avoided startup benefit!"
    )
    assert prevent_info["delta"] > 0, (
        "Avoided startup saving should be strictly positive!"
    )

    # 5. Formulate Day-Ahead Market Config & Calculate DMAS Bids
    mc = MarketConfig(
        market_id="dayahead",
        market_products=[MarketProduct(timedelta(hours=1), 24, timedelta(hours=0))],
        additional_fields=["exclusive_id", "link", "block_id"],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=start_time,
            until=start_time + timedelta(days=2),
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MW",
        price_unit="€/MW",
        market_mechanism="pay_as_clear",
    )
    products = get_available_products(mc.market_products, start_time)

    orderbook = strategy.calculate_bids(
        unit=plant,
        market_config=mc,
        product_tuples=products,
    )

    print("\nGenerated Orderbook:")
    print(f"  - Total Orders Generated: {len(orderbook)}")
    linked_orders = [o for o in orderbook if o.get("link") is not None]
    print(f"  - Linked / Conditional Orders: {len(linked_orders)}")

    # Show orders for the dip hours 20, 21, 22, 23
    print("\nSample Bids Around The Evening Price Dip (Hours 20-23):")
    for o in orderbook:
        h = int((o["start_time"] - start_time).total_seconds() / 3600)
        if 20 <= h <= 23:
            print(
                f"  Hour {h:02d}: Vol={o['volume']:6.1f} MW, Bid Price=€{o['price']:5.2f}/MWh, Block={o.get('block_id')}, Link={o.get('link')}"
            )

    # 6. Simulate Market Clearing with Demand
    print("\nClearing Orders in Complex DMAS Market...")
    for o in orderbook:
        o["agent_addr"] = "ccgt_agent"
        o["exclusive_id"] = None

    # Demand orders matching high daytime demand and moderate night demand
    demand_orders = []
    for h in range(24):
        p_start = start_time + timedelta(hours=h)
        p_end = p_start + timedelta(hours=1)
        # In hours 21-23 demand is willing to pay €45/MWh for 500 MW
        dem_price = day1_prices[h]
        demand_orders.append(
            {
                "start_time": p_start,
                "end_time": p_end,
                "volume": -500.0,
                "price": dem_price + 5.0,  # buyers will pay market price + margin
                "block_id": None,
                "link": None,
                "exclusive_id": None,
                "agent_addr": "market_demand",
                "bid_id": f"dem_{h}",
                "only_hours": None,
            }
        )

    mr = ComplexDmasClearingRole(mc)
    all_bids = orderbook + demand_orders
    accepted, rejected, meta, flows = mr.clear(all_bids, products)

    accepted_ccgt = [o for o in accepted if o.get("agent_addr") == "ccgt_agent"]
    print(f"  - Total Orders Accepted: {len(accepted)}")
    print(f"  - CCGT Orders Cleared: {len(accepted_ccgt)} of {len(orderbook)}")

    # Verify that the plant successfully stayed committed across the dip
    cleared_hours = {
        int((o["start_time"] - start_time).total_seconds() / 3600)
        for o in accepted_ccgt
    }
    dip_cleared = {21, 22, 23}.issubset(cleared_hours)
    print(f"  - Plant Remained Online in Dip Hours (21, 22, 23): {dip_cleared}")
    assert dip_cleared, (
        "Plant should remain online during dip hours to avoid cold restart!"
    )

    print("\n" + "=" * 75)
    print("SUCCESS: Avoided start costs strategy validated and demonstrated.")
    print("=" * 75)


if __name__ == "__main__":
    run_avoided_start_costs_demonstration()
