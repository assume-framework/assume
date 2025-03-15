# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from operator import itemgetter

import pandas as pd
import pyomo.environ as pyo
from mango import AgentAddress
from pyomo.opt import SolverFactory, TerminationCondition, check_available_solvers

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import create_incidence_matrix
from assume.markets.base_market import MarketRole

# Set the log level to WARNING
logging.getLogger("pyomo").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]
EPS = 1e-4


def market_clearing_opt(
    orders: Orderbook,
    market_products: list[MarketProduct],
    mode: str,
    with_linked_bids: bool,
    incidence_matrix: pd.DataFrame = None,
    lines: pd.DataFrame = None,
    solver: str = "appsi_highs",
    solver_options: dict = {},
):
    """
    Sets up and solves the market clearing optimization problem.

    Args:
        orders (Orderbook): The list of the orders.
        market_products (list[MarketProduct]): The products to be traded.
        mode (str): The mode of the market clearing determining whether the minimum acceptance ratio is considered.
        with_linked_bids (bool): Whether the market clearing should include linked bids.
        incidence_matrix (pd.DataFrame): The incidence matrix of the network. (Shows the connections between nodes.)
        lines (pd.DataFrame): The lines and their capacities of the network.

    Returns:
        tuple[pyomo.core.base.PyomoModel.ConcreteModel, pyomo.opt.results.SolverResults]: The solved pyomo model and the solver results

    Notes:
        The problem is formulated as a mixed-integer linear program (MILP) and solved using the pyomo package.
        The objective function is to maximize the social welfare and defined as the sum of the product of the price, volume, and acceptance ratio of each order.
        The decision variables are given by the acceptance ratio of each order bounded by 0 and 1 and the acceptance as a binary variable.

        The energy balance constraint ensures that the sum of the accepted volumes of all orders is zero.
        The acceptance of each order is bounded by 0 and 1.

        If the mode is 'with_min_acceptance_ratio', the minimum acceptance ratio is considered.
        The minimum acceptance ratio is defined as the ratio of the minimum volume to accept to the total volume of the order.

        If linked bids are considered, the acceptance of a child bid is bounded by the acceptance of its parent bid.

        The market clearing is solved using pyomo with the specified solver (HIGHS is used by default).
        If the specified solver is not available, the model is solved using available solver.
        If none of the solvers are available, an exception is raised.

        After solving the model, the acceptance of each order is fixed to the value in the solution and the model is solved again.
        This removes all binary variables from the model and allows to extract the market clearing prices from the dual variables of the energy balance constraint.

    """
    # Set nodes and lines based on the incidence matrix and lines DataFrame
    if incidence_matrix is not None:
        nodes = list(incidence_matrix.index)
        line_ids = list(incidence_matrix.columns)
    else:
        nodes = ["node0"]
        line_ids = ["line0"]

    model = pyo.ConcreteModel()

    # add dual suffix to the model (we need this to extract the market clearing prices later)
    # if mode is not 'with_min_acceptance_ratio', otherwise the dual suffix is added later
    if mode != "with_min_acceptance_ratio":
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
        initialize=[
            order["bid_id"] for order in orders if order["bid_type"] in ["BB", "LB"]
        ],
        doc="block_bids",
    )

    model.nodes = pyo.Set(initialize=nodes, doc="nodes")
    model.lines = pyo.Set(initialize=line_ids, doc="lines")

    # decision variables for the acceptance ratio of simple and block bids (including linked bids)
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

    # decision variables that define flows between nodes
    # assuming the orders contain the node and are collected in nodes
    if incidence_matrix is not None:
        # Decision variables for flows on each line at each timestep
        model.flows = pyo.Var(model.T, model.lines, domain=pyo.Reals, doc="power_flows")

    if mode == "with_min_acceptance_ratio":
        model.Bids = pyo.Set(
            initialize=[order["bid_id"] for order in orders], doc="all_bids"
        )
        # decision variables for the acceptance as binary variable
        model.x = pyo.Var(
            model.Bids,
            domain=pyo.Binary,
            doc="bid_accepted",
        )

        # add minimum acceptance ratio constraints
        model.mar_constr = pyo.ConstraintList()
        for order in orders:
            if order["min_acceptance_ratio"] is None:
                continue
            elif order["bid_type"] == "SB":
                model.mar_constr.add(
                    model.xs[order["bid_id"]]
                    >= order["min_acceptance_ratio"] * model.x[order["bid_id"]]
                )
                model.mar_constr.add(
                    model.xs[order["bid_id"]] <= model.x[order["bid_id"]]
                )

            elif order["bid_type"] in ["BB", "LB"]:
                model.mar_constr.add(
                    model.xb[order["bid_id"]]
                    >= order["min_acceptance_ratio"] * model.x[order["bid_id"]]
                )
                model.mar_constr.add(
                    model.xb[order["bid_id"]] <= model.x[order["bid_id"]]
                )

    # limit the acceptance of child bids by the acceptance of their parent bid
    if with_linked_bids:
        model.linked_bid_constr = pyo.ConstraintList()
        for order in orders:
            if "parent_bid_id" in order.keys() and order["parent_bid_id"] is not None:
                parent_bid_id = order["parent_bid_id"]
                model.linked_bid_constr.add(
                    model.xb[order["bid_id"]] <= model.xb[parent_bid_id]
                )

    # Function to calculate the balance for each node and time
    def energy_balance_rule(model, node, t):
        """
        Calculate the energy balance for a given node and time.

        This function calculates the energy balance for a specific node and time in a complex clearing algorithm. It iterates over the orders and adjusts the balance expression based on the bid type. It also adjusts the flow subtraction to account for actual connections if an incidence matrix is provided.

        Args:
            model: The complex clearing model.
            node: The node for which to calculate the energy balance.
            t: The time for which to calculate the energy balance.

        Returns:
            bool: True if the energy balance is zero, False otherwise.
        """
        balance_expr = 0.0  # Initialize the balance expression
        # Iterate over orders to adjust the balance expression based on bid type
        for order in orders:
            if (
                order["bid_type"] == "SB"
                and order["node"] == node
                and order["start_time"] == t
            ):
                balance_expr += order["volume"] * model.xs[order["bid_id"]]
            elif order["bid_type"] in ["BB", "LB"] and order["node"] == node:
                for start_time, volume in order["volume"].items():
                    if start_time == t:
                        balance_expr += volume * model.xb[order["bid_id"]]

        # Add contributions from line flows based on the incidence matrix
        if incidence_matrix is not None:
            for line in model.lines:
                incidence_value = incidence_matrix.loc[node, line]
                if incidence_value != 0:
                    balance_expr += incidence_value * model.flows[t, line]

        return balance_expr == 0

    # Add the energy balance constraints for each node and time period using the rule
    # Define the energy balance constraint using two indices (node and time)
    model.energy_balance = pyo.Constraint(
        model.nodes, model.T, rule=energy_balance_rule
    )

    if incidence_matrix is not None:
        model.transmission_constr = pyo.ConstraintList()
        for t in model.T:
            for line in model.lines:
                capacity = lines.at[line, "s_nom"]
                # Limit the flow on each line
                model.transmission_constr.add(model.flows[t, line] <= capacity)
                model.transmission_constr.add(model.flows[t, line] >= -capacity)

    # define the objective function as cost minimization
    obj_expr = 0
    for order in orders:
        if order["bid_type"] == "SB":
            obj_expr += order["price"] * order["volume"] * model.xs[order["bid_id"]]
        elif order["bid_type"] in ["BB", "LB"]:
            for start_time, volume in order["volume"].items():
                obj_expr += order["price"] * volume * model.xb[order["bid_id"]]

    model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

    solver = SolverFactory(solver)
    # Solve the model
    instance = model.create_instance()
    results = solver.solve(instance, options=solver_options)

    # Fix all model.x to the values in the solution
    if mode == "with_min_acceptance_ratio":
        # Add dual suffix to the model (needed to extract duals later)
        instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        for bid_id in instance.Bids:
            # Fix the binary variable to its value
            value = instance.x[bid_id].value
            if value is not None:
                value = 1 if value >= 0.99 else 0
            instance.x[bid_id].fix(value)
            # Change the domain to Reals (or appropriate continuous domain)
            instance.x[bid_id].domain = pyo.Reals

        # Resolve the model
        results = solver.solve(instance, options=solver_options)

    return instance, results


class ComplexClearingRole(MarketRole):
    """
    This class defines an optimization-based market clearing algorithm with support for complex bid types,
    including block bids, linked bids, minimum acceptance ratios, and profiled volumes. It supports network
    representations with either zonal or nodal configurations, enabling the modeling of complex markets with
    multiple zones and power flow constraints.

    The market clearing algorithm accepts additional arguments via the ``param_dict`` in the market configuration.

    Args:
        marketconfig (MarketConfig): The market configuration object containing all parameters for the market clearing process.

    Attributes:
        marketconfig (MarketConfig): The market configuration.
        incidence_matrix (pd.DataFrame): The incidence matrix representing the power network connections.
        nodes (list): List of nodes or zones in the network, depending on the selected representation.

    Supported Parameters in ``param_dict``:
        - ``solver`` (str): Specifies the solver to be used for the optimization problem. Default is `'appsi_highs'`.
        - ``log_flows`` (bool): Indicates whether to log the power flows on the lines. Default is `False`.
        - ``pricing_mechanism`` (str): Defines the pricing mechanism to be used. Default is `'pay_as_clear'`, with an alternative option of `'pay_as_bid'`.
        - ``zones_identifier`` (str): The key in the bus data that identifies the zone each bus belongs to. Used for zonal representation.

    Example market configuration:

    .. code-block:: yaml

        market_mechanism: complex_clearing
        param_dict:
            solver: apps_highs
            log_flows: true
            pricing_mechanism: pay_as_clear
            zones_identifier: zone_id

    Network Representations:
        - **Zonal Representation**: The network is divided into zones, and the incidence matrix represents the connections between these zones.
            - If a ``zones_identifier`` is provided, buses are grouped into zones based on this identifier.
            - The incidence matrix is constructed to represent the power connections between these zones.
            - The total transfer capacity between zones is determined by the sum of the capacities of the lines connecting the zones.

        - **Nodal Representation**: If no ``zones_identifier`` is provided, each bus is treated as a separate node, and the incidence matrix represents the connections between these nodes.
    """

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

        self.define_solver(solver=marketconfig.param_dict.get("solver", "appsi_highs"))

        # Define grid data
        self.nodes = ["node0"]
        self.zones_id = None
        self.incidence_matrix = None
        self.lines = None

        if self.grid_data:
            self.lines = self.grid_data["lines"]
            buses = self.grid_data["buses"]

            self.zones_id = self.marketconfig.param_dict.get("zones_identifier")
            self.node_to_zone = None

            # Generate the incidence matrix and set the nodes based on zones or individual buses
            if self.zones_id:
                # Zonal Case
                self.incidence_matrix = create_incidence_matrix(
                    self.lines, buses, zones_id=self.zones_id
                )
                self.nodes = buses[self.zones_id].unique()
                self.node_to_zone = buses[self.zones_id].to_dict()
            else:
                # Nodal Case
                self.incidence_matrix = create_incidence_matrix(self.lines, buses)
                self.nodes = buses.index.values

        self.log_flows = self.marketconfig.param_dict.get("log_flows", False)
        self.pricing_mechanism = self.marketconfig.param_dict.get(
            "pricing_mechanism", "pay_as_clear"
        )

    def define_solver(self, solver: str):
        # Get the solver from the market configuration
        if solver == "highs":
            solver = "appsi_highs"

        # Check if the solver is available and define solver options
        solvers = check_available_solvers(*SOLVERS)
        if len(solvers) < 1:
            raise Exception(f"None of {SOLVERS} are available")

        if solver == "gurobi":
            solver_options = {"cutoff": -1.0, "MIPGap": EPS, "LogToConsole": 0}
        elif solver == "appsi_highs":
            solver_options = {"output_flag": False, "log_to_console": False}
        else:
            solver_options = {}

        if solver not in solvers:
            logger.warning(f"Solver {solver} not available, using {solvers[0]}")
            solver = solvers[0]

        self.solver = solver
        self.solver_options = solver_options

    def validate_orderbook(
        self, orderbook: Orderbook, agent_addr: AgentAddress
    ) -> None:
        """
        Checks whether the bid types are valid and whether the volumes are within the maximum bid volume.

        Args:
            orderbook (Orderbook): The orderbook to be validated.
            agent_addr (AgentAddress): The agent address of the market.

        Raises:
            ValueError: If the bid type is invalid.
        """
        market_id = self.marketconfig.market_id
        max_volume = self.marketconfig.maximum_bid_volume

        for order in orderbook:
            # if bid_type is None, set to default bid_type
            if order.get("bid_type") is None:
                order["bid_type"] = "SB"
            # Validate bid_type
            elif order["bid_type"] not in ["SB", "BB", "LB"]:
                logger.warning(
                    f"Market '{market_id}': Invalid bid_type '{order['bid_type']}' in order {order}. Setting to 'SB'."
                )
                order["bid_type"] = "SB"  # Set to default bid_type

        super().validate_orderbook(orderbook, agent_addr)

        for order in orderbook:
            # Validate volumes
            if order["bid_type"] in ["BB", "LB"]:
                for key, volume in order.get("volume", {}).items():
                    if abs(volume) > max_volume:
                        logger.warning(
                            f"Market '{market_id}': Volume '{volume}' for key '{key}' exceeds max_volume {max_volume} in order {order}. Setting to max_volume."
                        )
                        order["volume"][key] = max_volume if volume > 0 else -max_volume

            # Node validation
            node = order.get("node")
            if node:
                if self.zones_id:
                    node = self.node_to_zone.get(node, self.nodes[0])
                    order["node"] = node
                if node not in self.nodes:
                    logger.warning(
                        f"Market '{market_id}': Node '{node}' not in nodes list {self.nodes}. Setting to first node '{self.nodes[0]}'. Order details: {order}"
                    )
                    order["node"] = self.nodes[0]
            else:
                if self.incidence_matrix is not None:
                    logger.warning(
                        f"Market '{market_id}': Order without a node, setting node to the first node '{self.nodes[0]}'. Please check the bidding strategy if correct node is set. Order details: {order}"
                    )
                    order["node"] = self.nodes[0]
                else:
                    logger.warning(
                        f"Market '{market_id}': Order without a node and no incidence matrix, setting node to 'node0'. Order details: {order}"
                    )
                    order["node"] = "node0"

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Implements pay-as-clear with more complex bid structures, including acceptance ratios, bid types, and profiled volumes.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products to be traded.

        Raises:
            Exception: If the problem is infeasible.

        Returns:
            accepted_orders (Orderbook): The accepted orders.
            rejected_orders (Orderbook): The rejected orders.
            meta (list[dict]): The market clearing results.
            flows (dict): The power flows on the lines.

        Notes:
            First the market clearing is solved using the cost minimization with the pyomo model market_clearing_opt.
            Then the market clearing prices are extracted from the solved model as dual variables of the energy balance constraint.
            Next the surplus of each order and its children is calculated and orders with negative surplus are removed from the orderbook.
            This is repeated until all orders remaining in the orderbook have positive surplus.
            Optional additional fields are: min_acceptance_ratio, parent_bid_id, node
        """

        if len(orderbook) == 0:
            return [], [], []

        orderbook.sort(key=itemgetter("start_time", "end_time", "only_hours"))

        # create a list of all orders linked as child to a bid
        child_orders = []
        for order in orderbook:
            order["accepted_price"] = {}
            order["accepted_volume"] = {}
            # get child linked bids
            if "parent_bid_id" in order.keys() and order["parent_bid_id"] is not None:
                # check whether the parent bid is in the orderbook
                parent_bid_id = order["parent_bid_id"]
                parent_bid = next(
                    (bid for bid in orderbook if bid["bid_id"] == parent_bid_id), None
                )
                if parent_bid is None:
                    order["parent_bid_id"] = None
                    logger.warning(f"Parent bid {parent_bid_id} not in orderbook")
                else:
                    child_orders.append(order)

        with_linked_bids = bool(child_orders)

        rejected_orders: Orderbook = []

        mode = "default"
        if "min_acceptance_ratio" in self.marketconfig.additional_fields:
            mode = "with_min_acceptance_ratio"

        # solve the market clearing problem
        while True:
            # solve the optimization with the current orderbook
            instance, results = market_clearing_opt(
                orders=orderbook,
                market_products=market_products,
                mode=mode,
                with_linked_bids=with_linked_bids,
                incidence_matrix=self.incidence_matrix,
                lines=self.lines,
                solver=self.solver,
                solver_options=self.solver_options,
            )

            if results.solver.termination_condition == TerminationCondition.infeasible:
                raise Exception("infeasible")

            # extract dual from model.energy_balance
            market_clearing_prices = {}
            for node in self.nodes:
                market_clearing_prices[node] = {
                    t: instance.dual[instance.energy_balance[node, t]]
                    for t in instance.T
                }

            # check the surplus of each order and remove those with negative surplus
            orders_surplus = []
            for order in orderbook:
                children = []
                if with_linked_bids:
                    children = [
                        child
                        for child in child_orders
                        if child["parent_bid_id"] == order["bid_id"]
                    ]

                order_surplus = calculate_order_surplus(
                    order, market_clearing_prices, instance, children
                )

                # correct rounding
                if order_surplus != 0 and abs(order_surplus) < EPS:
                    order_surplus = 0

                orders_surplus.append(order_surplus)

                # remove orders with negative profit
                if order_surplus < 0:
                    rejected_orders.append(order)
                    orderbook.remove(order)
                    rejected_orders.extend(children)
                    for child in children:
                        orderbook.remove(child)

            # check if all orders have positive surplus
            if all(order_surplus >= 0 for order_surplus in orders_surplus):
                break

        accepted_orders, rejected_orders, meta, flows = extract_results(
            model=instance,
            orders=orderbook,
            rejected_orders=rejected_orders,
            market_products=market_products,
            market_clearing_prices=market_clearing_prices,
            pricing_mechanism=self.pricing_mechanism,
            log_flows=self.log_flows,
        )

        self.all_orders = []

        return accepted_orders, rejected_orders, meta, flows


def calculate_order_surplus(
    order: dict,
    market_clearing_prices: dict,
    instance: pyo.ConcreteModel,
    children: list[dict],
):
    """
    Calculates the surplus of an order given the market clearing prices and results of the market clearing.

    Args:
        order (dict): The order
        market_clearing_prices (dict): The market clearing prices.
        instance (pyomo.core.base.PyomoModel.ConcreteModel): The solved pyomo model containing the results of the market clearing.
        children (list[dict]): The linked child bids of the given order.

    Returns:
        float: The surplus of the order as (market_clearing_price - order_price) * order_volume * order_acceptance

    Note:
        The surplus of children linked to the given order is added if it is positive to account for the rule that children can 'save' their parent bid.
    """

    order_surplus = 0

    # calculate the surplus of simple bids
    if order["bid_type"] == "SB":
        if (
            pyo.value(instance.xs[order["bid_id"]]) < EPS
            or abs(
                market_clearing_prices[order["node"]][order["start_time"]]
                - order["price"]
            )
            < EPS
        ):
            order_surplus = 0
        else:
            order_surplus = (
                (
                    market_clearing_prices[order["node"]][order["start_time"]]
                    - order["price"]
                )
                * order["volume"]
                * pyo.value(instance.xs[order["bid_id"]])
            )
    # calculate the surplus of block bidx
    elif order["bid_type"] in ["BB", "LB"]:
        bid_volume = sum(order["volume"].values())
        if pyo.value(instance.xb[order["bid_id"]]) < EPS:
            order_surplus = 0
        else:
            order_surplus = (
                sum(
                    market_clearing_prices[order["node"]][t] * v
                    for t, v in order["volume"].items()
                )
                - order["price"] * bid_volume
            ) * pyo.value(instance.xb[order["bid_id"]])

        # add the surplus of child linked bids if it is positive
        for child_order in children:
            child_surplus = (
                sum(
                    market_clearing_prices[child_order["node"]][t] * v
                    for t, v in child_order["volume"].items()
                )
                - child_order["price"] * bid_volume
            ) * pyo.value(instance.xb[child_order["bid_id"]])
            if child_surplus > 0:
                order_surplus += child_surplus

    # correct rounding
    if order_surplus != 0 and abs(order_surplus) < EPS:
        order_surplus = 0

    return order_surplus


def extract_results(
    model: pyo.ConcreteModel,
    orders: Orderbook,
    rejected_orders: Orderbook,
    market_products: list[MarketProduct],
    market_clearing_prices: dict,
    pricing_mechanism: str = "pay_as_clear",
    log_flows: bool = False,
):
    """
    Extracts the results of the market clearing from the solved pyomo model.

    Args:
        model (pyomo.core.base.PyomoModel.ConcreteModel): The solved pyomo model containing the results of the market clearing
        orders (Orderbook): List of the orders
        rejected_orders (Orderbook): List of the rejected orders
        market_products (list[MarketProduct]): The products to be traded
        market_clearing_prices (dict): The market clearing prices

    Returns:
        tuple[Orderbook, Orderbook, list[dict]]: The accepted orders, rejected orders, and meta information

    """
    if pricing_mechanism not in ["pay_as_clear", "pay_as_bid"]:
        raise ValueError(f"Invalid pricing mechanism {pricing_mechanism}")

    accepted_orders: Orderbook = []
    meta = []

    supply_volume_dict = {node: {t: 0.0 for t in model.T} for node in model.nodes}
    demand_volume_dict = {node: {t: 0.0 for t in model.T} for node in model.nodes}

    for order in orders:
        if order["bid_type"] == "SB":
            acceptance = model.xs[order["bid_id"]].value
            acceptance = 0 if acceptance < EPS else acceptance

            # set the accepted volume and price for each simple bid
            order["accepted_volume"] = acceptance * order["volume"]
            if pricing_mechanism == "pay_as_clear":
                order["accepted_price"] = market_clearing_prices[order["node"]][
                    order["start_time"]
                ]
            elif pricing_mechanism == "pay_as_bid":
                order["accepted_price"] = order["price"]

            # calculate the total cleared supply and demand volume
            if order["accepted_volume"] > 0:
                supply_volume_dict[order["node"]][order["start_time"]] += order[
                    "accepted_volume"
                ]
            else:
                demand_volume_dict[order["node"]][order["start_time"]] += order[
                    "accepted_volume"
                ]

        elif order["bid_type"] in ["BB", "LB"]:
            acceptance = model.xb[order["bid_id"]].value
            acceptance = 0 if acceptance < EPS else acceptance

            # set the accepted volume and price for each block bid
            for start_time, volume in order["volume"].items():
                order["accepted_volume"][start_time] = acceptance * volume
                if pricing_mechanism == "pay_as_clear":
                    order["accepted_price"][start_time] = market_clearing_prices[
                        order["node"]
                    ][start_time]
                elif pricing_mechanism == "pay_as_bid":
                    order["accepted_price"][start_time] = order["price"]

                # calculate the total cleared supply and demand volume
                if order["accepted_volume"][start_time] > 0:
                    supply_volume_dict[order["node"]][start_time] += order[
                        "accepted_volume"
                    ][start_time]
                else:
                    demand_volume_dict[order["node"]][start_time] += order[
                        "accepted_volume"
                    ][start_time]

        if acceptance > 0:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    for order in rejected_orders:
        # set the accepted volume and price for each rejected order to zero
        if order["bid_type"] == "SB":
            order["accepted_volume"] = 0
            order["accepted_price"] = 0

        elif order["bid_type"] in ["BB", "LB"]:
            order["accepted_volume"] = {t: 0 for t in order["volume"].keys()}
            order["accepted_price"] = {t: 0 for t in order["volume"].keys()}

    # write the meta information for each hour of the clearing period
    for node in market_clearing_prices.keys():
        for product in market_products:
            t = product[0]

            clear_price = market_clearing_prices[node][t]

            supply_volume = supply_volume_dict[node][t]
            demand_volume = demand_volume_dict[node][t]
            duration_hours = (product[1] - product[0]) / timedelta(hours=1)

            meta.append(
                {
                    "supply_volume": supply_volume,
                    "demand_volume": -demand_volume,
                    "demand_volume_energy": -demand_volume * duration_hours,
                    "supply_volume_energy": supply_volume * duration_hours,
                    "price": clear_price,
                    "max_price": clear_price,
                    "min_price": clear_price,
                    "node": node,
                    "product_start": product[0],
                    "product_end": product[1],
                    "only_hours": product[2],
                }
            )

        flows_filtered = {}

        if log_flows:
            # extract flows

            # Check if the model has the 'flows' attribute
            if hasattr(model, "flows"):
                flows = model.flows

                # filter flows and only use positive flows to half the size of the dict
                flows_filtered = {
                    index: flow.value for index, flow in flows.items() if not flow.stale
                }

    return accepted_orders, rejected_orders, meta, flows_filtered
