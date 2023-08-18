import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]


# TODO: remove paradoxically accepted orders from solution
def pay_as_clear_opt(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, check_available_solvers

    eps = 1e-4

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

    return extract_results(
        model=model,
        eps=eps,
        orders=orders,
        market_products=market_products,
        market_clearing_prices=market_clearing_prices,
    )


def pay_as_clear_complex_opt(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, check_available_solvers

    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []

    price_cap = (
        market_agent.marketconfig.minimum_bid_price,
        market_agent.marketconfig.maximum_bid_price,
    )
    eps = 1e-4

    market_getter = itemgetter("start_time", "end_time", "only_hours")
    market_agent.all_orders.sort(key=market_getter)

    orders = market_agent.all_orders

    if len(market_agent.all_orders) == 0:
        return accepted_orders, [], meta

    model = pyo.ConcreteModel()

    model.T = pyo.Set(
        initialize=[market_product[0] for market_product in market_products],
        doc="timesteps",
    )
    model.Bids = pyo.Set(
        initialize=[order["bid_id"] for order in orders], doc="all_bids"
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
    model.xb = pyo.Var(model.bBids, domain=pyo.Binary, doc="block_bid_acceptance")

    model.prices = pyo.Var(model.T, domain=pyo.Reals, doc="prices", bounds=price_cap)
    model.surplus = pyo.Var(model.Bids, domain=pyo.NonNegativeReals, doc="surplus")

    balance_expr = {t: 0.0 for t in model.T}
    for order in orders:
        if order["bid_type"] == "SB":
            balance_expr[order["start_time"]] += (
                order["volume"] * model.xs[order["bid_id"]]
            )
        elif order["bid_type"] == "BB":
            for start_time, volume in order["profile"].items():
                balance_expr[start_time] += volume * model.xb[order["bid_id"]]

    def energy_balance_rule(m, t):
        return balance_expr[t] == 0

    model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_rule)

    surplus_expr = {bid_id: 0.0 for bid_id in model.Bids}
    for order in orders:
        if order["bid_type"] == "SB":
            surplus_expr[order["bid_id"]] = (
                model.prices[order["start_time"]] - order["price"]
            ) * order["volume"]

        elif order["bid_type"] == "BB":
            bid_volume = sum(order["profile"].values())
            big_M = (price_cap[1] - price_cap[0]) * bid_volume
            surplus_expr[order["bid_id"]] = (
                sum(model.prices[t] * v for t, v in order["profile"].items())
                - order["price"] * bid_volume
            ) - big_M * (1 - model.xb[order["bid_id"]])

    def surplus_rule(m, bid_id):
        return m.surplus[bid_id] >= surplus_expr[bid_id]

    model.surplus_rule = pyo.Constraint(model.Bids, rule=surplus_rule)

    primal_obj_expr = 0
    for order in orders:
        if order["bid_type"] == "SB":
            primal_obj_expr += (
                order["price"] * order["volume"] * model.xs[order["bid_id"]]
            )
        elif order["bid_type"] == "BB":
            for start_time, volume in order["profile"].items():
                primal_obj_expr += order["price"] * volume * model.xb[order["bid_id"]]

    primal_obj_expr *= -1
    model.objective = pyo.Objective(expr=primal_obj_expr, sense=pyo.maximize)

    def dual_obj_rule(m):
        return sum(m.surplus[bid_id] for bid_id in m.Bids)

    # model.objective_dual = pyo.Objective(rule=dual_obj_rule, sense=pyo.minimize)

    def primal_eql_dual(m):
        return primal_obj_expr >= dual_obj_rule(m)

    model.primal_eql_dual = pyo.Constraint(rule=primal_eql_dual)

    solvers = check_available_solvers(*SOLVERS)
    if "gurobi" not in solvers:
        raise Exception(
            "Gurobi solver is not available. You cannot use this clearing algorithm."
        )

    solver = SolverFactory("gurobi")
    solver.options["cutoff"] = -1.0
    solver.options["eps"] = eps

    # Solve the model
    result = solver.solve(model)

    if result["Solver"][0]["Status"] != "ok":
        raise Exception("infeasible")

    # Find the dual variable for the balance constraint
    market_clearing_prices = {t: model.prices[t].value for t in model.T}

    return extract_results(
        model=model,
        eps=eps,
        orders=orders,
        market_products=market_products,
        market_clearing_prices=market_clearing_prices,
    )


def extract_results(
    model,
    eps,
    orders,
    market_products,
    market_clearing_prices,
):
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []

    supply_volume_dict = {t: 0.0 for t in model.T}
    demand_volume_dict = {t: 0.0 for t in model.T}

    for order in orders:
        if order["bid_type"] == "SB":
            acceptance = model.xs[order["bid_id"]].value
            acceptance = 0 if acceptance < eps else acceptance
            order["accepted_volume"] = acceptance * order["volume"]
            order["price"] = market_clearing_prices[order["start_time"]]

            if order["accepted_volume"] > 0:
                supply_volume_dict[order["start_time"]] += order["accepted_volume"]
            else:
                demand_volume_dict[order["start_time"]] += order["accepted_volume"]

        elif order["bid_type"] == "BB":
            acceptance = model.xb[order["bid_id"]].value
            acceptance = 0 if acceptance < eps else acceptance

            for start_time, volume in order["profile"].items():
                order["accepted_profile"][start_time] = acceptance * volume
                order["price"] = market_clearing_prices[start_time]

                if order["accepted_profile"][start_time] > 0:
                    supply_volume_dict[start_time] += order["accepted_profile"][
                        start_time
                    ]
                else:
                    demand_volume_dict[start_time] += order["accepted_profile"][
                        start_time
                    ]

        if acceptance > 0:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    for product in market_products:
        t = product[0]

        clear_price = market_clearing_prices[t]

        supply_volume = supply_volume_dict[t]
        demand_volume = demand_volume_dict[t]
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

    return accepted_orders, rejected_orders, meta
