# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import pyomo.environ as pyo
from assume.common.utils import str_to_bool
from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

logger = logging.getLogger(__name__)

class BusDepot(DSMFlex, SupportsMinMax):
    """
    Represents a bus depot managing multiple electric vehicle (EV) buses and charging stations.
    This agent optimizes bus charging schedules based on electricity price, grid demand, and available flexibility.

    The `BusDepot` class utilizes the Pyomo optimization library to determine the optimal
    charging strategy that minimizes costs, grid impact, or maximizes renewable utilization.

    Args:
        id (str): Unique identifier for the bus depot.
        unit_operator (str): Operator managing the bus depot.
        bidding_strategies (dict): Strategies used for energy bidding in the market.
        forecaster (Forecaster): A forecaster used to get key variables such as electricity prices.
        charging_stations (dict[str, dict]): Charging stations available in the depot.
        electric_vehicles (dict[str, dict]): Electric buses assigned to the depot.
        objective (str, optional): Optimization objective ("min_cost", "min_grid_load", "max_RE_util"). Default is "min_cost".
        flexibility_measure (str, optional): Metric used to assess the depot's flexibility. Default is "cost_based_load_shift".
        max_power_capacity (float, optional): Maximum allowable power capacity. Default is 1000 kW.
        node (str, optional): Network node where the depot is connected. Default is "bus_node".
        location (tuple[float, float], optional): Geographic coordinates. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments for custom configurations.

    Attributes:
        required_technologies (list): Required technologies for the depot.
        optional_technologies (list): Optional technologies available.
    """

    required_technologies = []
    optional_technologies = ["charging_station", "electric_vehicle"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict],
        technology: str = "bus_depot",
        is_prosumer: str = "No",
        objective: str = "min_cost",
        flexibility_measure: str = "cost_based_load_shift",
        max_power_capacity: float = 1000,
        node: str = "bus_node",
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            components=components,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        self.objective = objective
        self.is_prosumer = str_to_bool(is_prosumer)
        self.flexibility_measure = flexibility_measure
        self.max_power_capacity = max_power_capacity
        self.electricity_price = self.forecaster["price_EOM"]
        self.bus_availability = self.forecaster["bus_availability"]
        self.bus_soc_profile = self.forecaster["bus_soc_profile"]
        self.renewable_availability = self.forecaster["renewable_availability"]

        self.setup_model(presolve=True)
        self.initialize_process_sequence()

    def define_parameters(self):
        """Defines parameters including electricity price, power capacity, and bus availability."""
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.max_power_capacity = pyo.Param(initialize=self.max_power_capacity)
        self.model.renewable_availability = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.renewable_availability)},
        )
    
    def define_variables(self):
        """Defines the charging and discharging power variables for EVs."""
        self.model.charging_power = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power_capacity)
        )
        self.model.ev_discharge = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power_capacity)
        )
        self.model.grid_load = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
    
    def initialize_process_sequence(self):
        """Manages EV charging and discharging based on station availability and bus schedules."""
        @self.model.Constraint(self.model.time_steps)
        def ev_charging_balance(m, t):
            total_ev_charge = sum(
                m.dsm_blocks[ev].charge[t] for ev in self.electric_vehicles.keys()
            )
            total_ev_discharge = sum(
                m.dsm_blocks[ev].discharge[t] for ev in self.electric_vehicles.keys()
            )
            return m.charging_power[t] - total_ev_discharge == total_ev_charge

        @self.model.Constraint(self.model.time_steps)
        def charging_station_availability(m, t):
            """Ensures buses connect only to available chargers."""
            return sum(
                m.dsm_blocks[cs].is_available[t] for cs in self.charging_stations.keys()
            ) >= sum(
                m.dsm_blocks[ev].is_charging[t] for ev in self.electric_vehicles.keys()
            )
    
    def define_constraints(self):
        """Ensures charging does not exceed grid capacity and optimizes based on availability."""
        @self.model.Constraint(self.model.time_steps)
        def max_power_constraint(m, t):
            return m.charging_power[t] - m.ev_discharge[t] <= m.max_power_capacity

        @self.model.Constraint(self.model.time_steps)
        def grid_load_constraint(m, t):
            """Limits grid load by optimizing charging schedules."""
            return m.grid_load[t] == m.charging_power[t] - m.ev_discharge[t]
    
    def define_objective(self):
        """Defines the optimization objective based on user selection."""
        if self.objective == "min_cost":
            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                return pyo.quicksum(m.grid_load[t] * m.electricity_price[t] for t in m.time_steps)
        elif self.objective == "min_grid_load":
            @self.model.Objective(sense=pyo.minimize)
            def obj_rule(m):
                return pyo.quicksum(m.grid_load[t] for t in m.time_steps)
        elif self.objective == "max_RE_util":
            @self.model.Objective(sense=pyo.maximize)
            def obj_rule(m):
                return pyo.quicksum(m.charging_power[t] * m.renewable_availability[t] for t in m.time_steps)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")