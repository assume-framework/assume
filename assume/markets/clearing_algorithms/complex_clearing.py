import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]


# TODO: clearing for BB not as expected - debugging necessary
def pay_as_clear_opt(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, check_available_solvers

    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []

    market_getter = itemgetter("start_time", "end_time", "only_hours")
    market_agent.all_orders.sort(key=market_getter)

    orders = market_agent.all_orders

    if len(market_agent.all_orders) == 0:
        return accepted_orders, [], meta

    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    model.T = pyo.Set(
        initialize=[market_product[0] for market_product in market_products],
        doc="timesteps",
    )
    model.sBids = pyo.Set(
        initialize=[order["bid_id"] for order in orders if order["bid_type"] == "SB"],
        doc="simple_bids",
    )
    model.bBids = pyo.Set(
        initialize=[order["bid_id"] for order in orders if order["bid_type"] == "BB"],
        doc="block_bids",
    )

    model.xs = pyo.Var(
        model.sBids,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
        doc="simple_bid_acceptance",
    )
    model.xb = pyo.Var(
        model.bBids,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
        doc="block_bid_acceptance",
    )

    balanceExpr = {t: 0.0 for t in model.T}
    for order in orders:
        if order["bid_type"] == "SB":
            balanceExpr[order["start_time"]] += (
                order["volume"] * model.xs[order["bid_id"]]
            )
        elif order["bid_type"] == "BB":
            for start_time, volume in order["profile"].items():
                balanceExpr[start_time] += volume * model.xb[order["bid_id"]]

    def energy_balance_rule(m, t):
        return balanceExpr[t] == 0

    model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

    obj_exp = 0
    for order in orders:
        if order["bid_type"] == "SB":
            obj_exp += order["price"] * order["volume"] * model.xs[order["bid_id"]]
        elif order["bid_type"] == "BB":
            for start_time, volume in order["profile"].items():
                obj_exp += order["price"] * volume * model.xb[order["bid_id"]]

    model.objective = pyo.Objective(expr=obj_exp, sense=pyo.minimize)

    solvers = check_available_solvers(*SOLVERS)
    if len(solvers) < 1:
        raise Exception(f"None of {SOLVERS} are available")

    solver = SolverFactory(solvers[0])

    # Solve the model
    result = solver.solve(model)

    if result["Solver"][0]["Status"] != "ok":
        raise Exception("infeasible")

    # Find the dual variable for the balance constraint
    market_clearing_prices = {t: model.dual[model.energy_balance[t]] for t in model.T}

    block_orders = [order for order in orders if order["bid_type"] == "BB"]
    simple_orders = [order for order in orders if order["bid_type"] == "SB"]

    for product, product_orders in groupby(simple_orders, market_getter):
        clear_price = market_clearing_prices[product[0]]

        supply_volume = 0
        demand_volume = 0

        for order in list(product_orders):
            order["volume"] = model.xs[order["bid_id"]].value * order["volume"]
            opt_volume = order["volume"]

            if opt_volume > 0:
                supply_volume += opt_volume
            else:
                demand_volume += opt_volume

            order["price"] = clear_price

            if opt_volume != 0:
                accepted_orders.append(order)
            else:
                rejected_orders.append(order)

        for order in block_orders:
            start_time = product[0]
            if start_time in order["profile"]:
                order["profile"][start_time] = (
                    model.xb[order["bid_id"]].value * order["profile"][start_time]
                )
                opt_volume = order["profile"][start_time]

                if opt_volume > 0:
                    supply_volume += opt_volume
                else:
                    demand_volume += opt_volume

        duration_hours = (product[1] - product[0]).total_seconds() / 60 / 60

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "demand_volume_energy": demand_volume * duration_hours,
                "supply_volume_energy": supply_volume * duration_hours,
                "price": clear_price,
                "max_price": clear_price,
                "min_price": clear_price,
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    for order in block_orders:
        order["price"] = clear_price

        if model.xb[order["bid_id"]].value != 0:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    market_agent.all_orders = []

    return accepted_orders, rejected_orders, meta
