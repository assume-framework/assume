import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Order, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]

def pay_as_clear_opt(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, check_available_solvers

    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []

    meta = []

    market_agent.all_orders.sort(key=market_getter)
    # find maximal length of market_agent.all_orders dicts
    number_of_timesteps = len(market_products)

    #find maximal number of bids per timestep
    number_of_bids_per_timestep = max(len(list(x)) for _, x in groupby(market_agent.all_orders, market_getter))

    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    model.T = pyo.RangeSet(0, number_of_timesteps-1)
    model.K = pyo.RangeSet(0, number_of_bids_per_timestep-1)

    model.P = pyo.Var(model.T, model.K, domain=pyo.Reals, initialize=0)

    model.energy_balance = pyo.ConstraintList()
    model.max_power = pyo.ConstraintList()
    model.min_power = pyo.ConstraintList()

    for t, (product, product_orders) in enumerate(
        groupby(market_agent.all_orders, market_getter)
    ):
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue


        product_orders = list(product_orders)

        expr = sum(model.P[t, k] for k in range(len(product_orders))) == 0
        model.energy_balance.add(expr)

        for k, order in enumerate(
            product_orders
        ):
            if order["volume"] > 0:
                expr = model.P[t, k] <= order["volume"]
            else:
                expr = model.P[t, k] >= order["volume"]
            model.max_power.add(expr)

        if "acceptance_ratio" in market_agent.marketconfig.additional_fields:
            for k, order in enumerate(
                product_orders
            ):
                if order["volume"] > 0:
                    expr = model.P[t, k] >= order["acceptance_ratio"] * order["volume"]
                else:
                    expr = model.P[t, k] <= order["acceptance_ratio"] * order["volume"]
                model.min_power.add(expr)
        else:
            for k, order in enumerate(
                product_orders
            ):
                expr = model.P[t, k] >= 0 if order["volume"] > 0 else model.P[t, k] <= 0
                model.min_power.add(expr)

    def objective_rule(model):
        expr = sum(
            model.P[t, k] * order["price"]
            for t, (_, product_orders) in enumerate(
                groupby(market_agent.all_orders, market_getter)
            )
            for k, order in enumerate(
                product_orders
            )
        )

        return expr

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    solvers = check_available_solvers(*SOLVERS)
    if len(solvers) < 1:
        raise Exception(f"None of {SOLVERS} are available")

    solver = SolverFactory("glpk")

    # Solve the model
    result = solver.solve(model)

    if result["Solver"][0]["Status"] != "ok":
        raise Exception("infeasible")

    # Find the dual variable for the balance constraint
    market_clearing_prices = [model.dual[model.energy_balance[i+1]] for i in range(len(model.energy_balance))]
