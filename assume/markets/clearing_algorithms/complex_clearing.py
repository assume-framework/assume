import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)

SOLVERS = ["glpk", "cbc", "gurobi", "cplex"]


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

    if len(market_agent.all_orders) == 0:
        return accepted_orders, meta

    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    model.T = pyo.Set(
        initialize=[market_product[0] for market_product in market_products],
        doc="timesteps",
    )
    model.K = pyo.Set(
        initialize=[order["bid_id"] for order in market_agent.all_orders], doc="bids"
    )

    model.P = pyo.Var(model.K, domain=pyo.Reals)

    model.energy_balance = pyo.ConstraintList()
    model.min_max_power = pyo.ConstraintList()

    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)

        expr = sum(model.P[order["bid_id"]] for order in product_orders) == 0
        model.energy_balance.add(expr)

        for order in product_orders:
            if "acceptance_ratio" in market_agent.marketconfig.additional_fields:
                if order["volume"] > 0:
                    min_max_power_expr = (
                        order["acceptance_ratio"] * order["volume"],
                        model.P[order["bid_id"]],
                        order["volume"],
                    )
                else:
                    min_max_power_expr = (
                        order["volume"],
                        model.P[order["bid_id"]],
                        order["acceptance_ratio"] * order["volume"],
                    )

            else:
                if order["volume"] > 0:
                    min_max_power_expr = (0, model.P[order["bid_id"]], order["volume"])
                else:
                    min_max_power_expr = (order["volume"], model.P[order["bid_id"]], 0)

            model.min_max_power.add(min_max_power_expr)

    def objective_rule(model):
        expr = sum(
            model.P[order["bid_id"]] * order["price"]
            for order in market_agent.all_orders
        )

        return expr

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    solvers = check_available_solvers(*SOLVERS)
    if len(solvers) < 1:
        raise Exception(f"None of {SOLVERS} are available")

    solver = SolverFactory("glpk")

    # Solve the model
    instance = model.create_instance()
    result = solver.solve(instance)

    if result["Solver"][0]["Status"] != "ok":
        raise Exception("infeasible")

    # Find the dual variable for the balance constraint
    market_clearing_prices = {
        t: instance.dual[instance.energy_balance[i + 1]]
        for i, t in enumerate(instance.T)
    }

    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        clear_price = market_clearing_prices[product[0]]

        supply_volume = 0
        demand_volume = 0

        for order in product_orders:
            opt_volume = instance.P[order["bid_id"]].value

            if opt_volume > 0:
                supply_volume += opt_volume
            else:
                demand_volume += opt_volume

            order["volume"] = opt_volume
            order["price"] = clear_price

            if opt_volume != 0:
                accepted_orders.append(order)

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

    return accepted_orders, meta
