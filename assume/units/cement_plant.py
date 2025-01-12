# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class CementPlant(DSMFlex, SupportsMinMax):
    """
    The CementPlant class represents a cement plant unit in the energy system.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        technology (str): The technology of the unit.
        node (str): The node of the unit.
        location (tuple[float, float]): The location of the unit.
        components (dict[str, dict]): The components of the unit such as raw_material_mill, clinker_system, cement_mill, and electrolyser.
        objective (str): The objective of the unit, e.g. minimize variable cost ("min_variable_cost").
        flexibility_measure (str): The flexibility measure of the unit, e.g. maximum load shift ("max_load_shift").
        demand (float): The demand of the unit - the amount of cement to be produced.
        cost_tolerance (float): The cost tolerance of the unit - the maximum cost that can be tolerated when shifting the load.
    """

    required_technologies = []
    optional_technologies = [
        "raw_material_mill",
        "clinker_system",
        "cement_mill",
        "electrolyser",
        "ccs_system",
        "hydrogen_buffer_storage",
        "clinker_inventory",
        "cement_inventory",
    ]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        technology: str = "cement_plant",
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        components: dict[str, dict] = None,
        objective: str = None,
        flexibility_measure: str = "",
        demand: float = 0,
        cost_tolerance: float = 10,
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

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the cement plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the cement plant unit."
                )

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.hydrogen_price = self.forecaster["price_hydrogen"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.grinder_availability_profile = self.forecaster[
            "grinder_availability_profile"
        ]
        self.clinker_availability_profile = self.forecaster[
            "clinker_availability_profile"
        ]
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")
        self.cement_demand = demand
        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance

        # Initialize component flags
        self.has_raw_mill = "raw_material_mill" in self.components.keys()
        self.has_clinker_system = "clinker_system" in self.components.keys()
        self.has_cement_mill = "cement_mill" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()
        self.has_ccs_system = "ccs_system" in self.components.keys()
        self.has_hydrogen_buffer_storage = (
            "hydrogen_buffer_storage" in self.components.keys()
        )
        self.has_clinker_inventory = "clinker_inventory" in self.components.keys()
        self.has_cement_inventory = "cement_inventory" in self.components.keys()

        self.opt_power_requirement = None
        self.flex_power_requirement = None

        # Main Model part
        self.model = pyo.ConcreteModel()
        self.define_sets()
        self.define_parameters()
        self.define_variables()

        self.initialize_components()
        self.initialize_process_sequence()

        self.define_constraints()
        self.define_objective_opt()

        self.determine_optimal_operation_without_flex(switch_flex_off=False)

        # Apply the flexibility function based on flexibility measure
        if self.flexibility_measure in DSMFlex.flexibility_map:
            DSMFlex.flexibility_map[self.flexibility_measure](self, self.model)
        else:
            raise ValueError(f"Unknown flexibility measure: {self.flexibility_measure}")

        self.define_objective_flex()

    def define_sets(self) -> None:
        self.model.time_steps = pyo.Set(
            initialize=[idx for idx, _ in enumerate(self.index)]
        )

    def define_parameters(self):
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )
        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.hydrogen_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.hydrogen_price)},
        )
        self.model.lime_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.lime_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )
        self.model.grinder_availability_profile = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.grinder_availability_profile)
            },
        )
        self.model.clinker_availability_profile = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value for t, value in enumerate(self.clinker_availability_profile)
            },
        )
        self.model.cement_demand = pyo.Param(initialize=self.cement_demand)

    def define_variables(self):
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self):
        """
        Initializes the process sequence for the cement plant based on available components.
        """
        if self.has_raw_mill and not (
            self.has_clinker_system
            or self.has_cement_mill
            or self.has_electrolyser
            or self.has_ccs_system
            or self.has_hydrogen_buffer_storage
            or self.has_clinker_inventory
            or self.has_cement_inventory
        ):

            @self.model.Constraint(self.model.time_steps)
            def raw_milling_material_flow_constraint(m, t):
                """
                Ensures the raw material milling output contributes to cement demand.
                """
                return (
                    self.model.dsm_blocks["raw_material_mill"].material_output[t] >= 0
                )  # Ensuring non-negative output

        if self.has_cement_mill and not (
            self.has_clinker_system
            or self.has_raw_mill
            or self.has_electrolyser
            or self.has_ccs_system
            or self.has_hydrogen_buffer_storage
            or self.has_clinker_inventory
            or self.has_cement_inventory
        ):

            @self.model.Constraint(self.model.time_steps)
            def cement_grinding_material_flow_constraint(m, t):
                """
                Ensures the cement milling output contributes to cement demand.
                """
                return (
                    self.model.dsm_blocks["cement_mill"].material_output[t] >= 0
                )  # Ensuring non-negative output

        if (
            self.has_clinker_system
            and self.has_cement_mill
            and not (
                self.has_raw_mill
                or self.has_electrolyser
                or self.has_ccs_system
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_cement_material_flow_constraint_1(m, t):
                """
                Links the output of the clinker system to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

        if (
            self.has_raw_mill
            and self.has_clinker_system
            and self.has_cement_mill
            and not (
                self.has_electrolyser
                or self.has_ccs_system
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def raw_to_clinker_material_flow_constraint(m, t):
                """
                Links the output of the raw material mill to the input of the clinker system.
                """
                return (
                    self.model.dsm_blocks["raw_material_mill"].material_output[t]
                    == self.model.dsm_blocks["clinker_system"].raw_meal[t]
                )

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_cement_material_flow_constraint_2(m, t):
                """
                Links the output of the clinker system to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

        if (
            self.has_clinker_system
            and self.has_ccs_system
            and self.has_cement_mill
            and not (
                self.has_raw_mill
                or self.has_electrolyser
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_cement_material_flow_constraint_3(m, t):
                """
                Links the output of the clinker system to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

        if (
            self.has_clinker_system
            and self.has_cement_mill
            and self.has_electrolyser
            and not (
                self.has_ccs_system
                or self.has_raw_mill
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def electrolyser_to_clinker_hydrogen_flow_constraint_1(m, t):
                """
                Links the hydrogen production of the electrolyser to the hydrogen input of the clinker system.
                """
                return (
                    self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                    == self.model.dsm_blocks["clinker_system"].thermal_in[t]
                )

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_cement_material_flow_constraint_4(m, t):
                """
                Links the output of the clinker system to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

        if (
            self.has_clinker_system
            and self.has_cement_mill
            and self.has_electrolyser
            and self.has_ccs_system
            and not (
                self.has_raw_mill
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_ccs_material_flow_constraint_2(m, t):
                """
                Links the gross emissions from the clinker system to the CCS system.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].calcination_emissions[t]
                    == self.model.dsm_blocks["ccs_system"].gross_emission[t]
                )

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_cement_material_flow_constraint_5(m, t):
                """
                Links the output of the clinker system to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

        if (
            self.has_clinker_system
            and self.has_electrolyser
            and self.has_hydrogen_buffer_storage
            and self.has_ccs_system
            and self.has_cement_mill
            and (
                not self.has_raw_mill
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def electrolyser_to_hydrogen_buffer_flow_constraint_1(m, t):
                """
                Ensures hydrogen production of the electrolyser is directed to hydrogen storage and clinker system.
                """
                return (
                    self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                    + self.model.dsm_blocks["hydrogen_buffer_storage"].discharge[t]
                    == self.model.dsm_blocks["clinker_system"].thermal_in[t]
                    + self.model.dsm_blocks["hydrogen_buffer_storage"].charge[t]
                )

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_ccs_material_flow_constraint_3(m, t):
                """
                Links the gross emissions from the clinker system to the CCS system.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].calcination_emissions[t]
                    == self.model.dsm_blocks["ccs_system"].gross_emission[t]
                )

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_cement_material_flow_constraint_6(m, t):
                """
                Links the output of the clinker system to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

        if (
            self.has_clinker_system
            and self.has_electrolyser
            and self.has_hydrogen_buffer_storage
            and self.has_ccs_system
            and self.has_cement_mill
            and self.has_clinker_inventory
            and not (self.has_cement_inventory and self.has_raw_mill)
        ):

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_ccs_material_flow_constraint_4(m, t):
                """
                Links the gross emissions from the clinker system to the CCS system.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].calcination_emissions[t]
                    == self.model.dsm_blocks["ccs_system"].gross_emission[t]
                )

            # 1. Electrolyser to Hydrogen Buffer Storage and Clinker System
            @self.model.Constraint(self.model.time_steps)
            def electrolyser_to_hydrogen_buffer_flow_constraint_2(m, t):
                """
                Ensures hydrogen production of the electrolyser is directed to hydrogen storage and clinker system.
                """
                return (
                    self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                    + self.model.dsm_blocks["hydrogen_buffer_storage"].discharge[t]
                    == self.model.dsm_blocks["clinker_system"].thermal_in[t]
                    + self.model.dsm_blocks["hydrogen_buffer_storage"].charge[t]
                )

            # 2. Clinker System to Clinker Inventory
            @self.model.Constraint(self.model.time_steps)
            def clinker_to_inventory_flow_constraint(m, t):
                """
                Links the clinker system output to the clinker inventory.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].cement_production[t]
                    + self.model.dsm_blocks["clinker_inventory"].discharge[t]
                    == self.model.dsm_blocks["clinker_inventory"].charge[t]
                )

            # 3. Clinker Inventory to Cement Mill
            @self.model.Constraint(self.model.time_steps)
            def clinker_inventory_to_cement_mill_flow_constraint(m, t):
                """
                Links the clinker inventory discharge to the input of the cement mill.
                """
                return (
                    self.model.dsm_blocks["clinker_inventory"].discharge[t]
                    == self.model.dsm_blocks["cement_mill"].material_input[t]
                )

    def define_constraints(self):
        if self.has_raw_mill and not (
            self.has_clinker_system
            or self.has_cement_mill
            or self.has_electrolyser
            or self.has_ccs_system
            or self.has_hydrogen_buffer_storage
        ):

            @self.model.Constraint()
            def raw_milling_demand_constraint(m):
                """
                Ensures the total output from the raw material milling meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["raw_material_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if self.has_cement_mill and not (
            self.has_clinker_system
            or self.has_raw_mill
            or self.has_electrolyser
            or self.has_ccs_system
            or self.has_hydrogen_buffer_storage
            or self.has_clinker_inventory
            or self.has_cement_inventory
        ):

            @self.model.Constraint()
            def cement_milling_demand_constraint(m):
                """
                Ensures the total output from the cement milling meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_clinker_system
            and self.has_cement_mill
            and not (
                self.has_raw_mill
                or self.has_electrolyser
                or self.has_ccs_system
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint()
            def clinker_to_cement_demand_constraint_1(m):
                """
                Ensures the output from the clinker system, when processed by the cement mill, meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_raw_mill
            and self.has_clinker_system
            and self.has_cement_mill
            and not (
                self.has_electrolyser
                or self.has_ccs_system
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint()
            def raw_to_cement_demand_constraint(m):
                """
                Ensures the output from the raw material mill, after processing through the clinker system and cement mill, meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_clinker_system
            and self.has_ccs_system
            and self.has_cement_mill
            and not (
                self.has_raw_mill
                or self.has_electrolyser
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def clinker_to_ccs_material_flow_constraint_1(m, t):
                """
                Links the gross emissions from the clinker system to the CCS system.
                """
                return (
                    self.model.dsm_blocks["clinker_system"].calcination_emissions[t]
                    == self.model.dsm_blocks["ccs_system"].gross_emission[t]
                )

            @self.model.Constraint()
            def raw_to_cement_demand_constraint(m):
                """
                Ensures the output from the raw material mill, after processing through the clinker system and cement mill, meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_clinker_system
            and self.has_cement_mill
            and self.has_electrolyser
            and not (
                self.has_ccs_system
                or self.has_raw_mill
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint()
            def clinker_to_cement_demand_constraint_2(m):
                """
                Ensures the output from the clinker system, when processed by the cement mill, meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_clinker_system
            and self.has_cement_mill
            and self.has_electrolyser
            and self.has_ccs_system
            and not (
                self.has_raw_mill
                or self.has_hydrogen_buffer_storage
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint(self.model.time_steps)
            def electrolyser_to_clinker_hydrogen_flow_constraint_2(m, t):
                """
                Links the hydrogen production of the electrolyser to the hydrogen input of the clinker system.
                """
                return (
                    self.model.dsm_blocks["electrolyser"].hydrogen_out[t]
                    == self.model.dsm_blocks["clinker_system"].thermal_in[t]
                )

            @self.model.Constraint()
            def clinker_to_cement_demand_constraint_3(m):
                """
                Ensures the output from the clinker system, when processed by the cement mill, meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_clinker_system
            and self.has_electrolyser
            and self.has_hydrogen_buffer_storage
            and self.has_ccs_system
            and self.has_cement_mill
            and not (
                self.has_raw_mill
                or self.has_clinker_inventory
                or self.has_cement_inventory
            )
        ):

            @self.model.Constraint()
            def clinker_to_cement_demand_constraint_4(m):
                """
                Ensures the output from the clinker system, when processed by the cement mill, meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        if (
            self.has_clinker_system
            and self.has_electrolyser
            and self.has_hydrogen_buffer_storage
            and self.has_ccs_system
            and self.has_cement_mill
            and self.has_clinker_inventory
            and not (self.has_cement_inventory and self.has_raw_mill)
        ):
            # Demand constraint: Cement Inventory meets Cement Demand
            @self.model.Constraint()
            def cement_inventory_demand_constraint_5(m):
                """
                Ensures the total output from the cement inventory meets the cement demand.
                """
                return (
                    sum(
                        self.model.dsm_blocks["cement_mill"].material_output[t]
                        for t in self.model.time_steps
                    )
                    == self.model.cement_demand
                )

        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """
            variable_cost = 0
            if self.has_raw_mill:
                variable_cost += self.model.dsm_blocks[
                    "raw_material_mill"
                ].operating_cost[t]
            if self.has_cement_mill:
                variable_cost += self.model.dsm_blocks["cement_mill"].operating_cost[t]
            if self.has_clinker_system:
                variable_cost += self.model.dsm_blocks["clinker_system"].operating_cost[
                    t
                ]
            if self.has_ccs_system:
                variable_cost += self.model.dsm_blocks["ccs_system"].operating_cost[t]
            if self.has_electrolyser:
                variable_cost += self.model.dsm_blocks["electrolyser"].operating_cost[t]
            return self.model.variable_cost[t] == variable_cost

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            power_input = 0
            if self.has_raw_mill:
                power_input += self.model.dsm_blocks["raw_material_mill"].power_in[t]
            if self.has_cement_mill:
                power_input += self.model.dsm_blocks["cement_mill"].power_in[t]
            if self.has_clinker_system:
                power_input += self.model.dsm_blocks["clinker_system"].power_in[t]
            if self.has_ccs_system:
                power_input += self.model.dsm_blocks["ccs_system"].power_in[t]
            if self.has_electrolyser:
                power_input += self.model.dsm_blocks["electrolyser"].power_in[t]
            return m.total_power_input[t] == power_input

    def define_objective_opt(self):
        """
        Defines the objective for the optimization model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        if self.objective == "min_variable_cost":

            @self.model.Objective(sense=pyo.minimize)
            def obj_rule_opt(m):
                """
                Minimizes the total variable cost over all time steps.
                """
                total_variable_cost = sum(
                    self.model.variable_cost[t] for t in self.model.time_steps
                )

                return total_variable_cost

        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def define_objective_flex(self):
        """
        Defines the flexibility objective for the optimization model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model.
        """
        if self.flexibility_measure == "cost_based_load_shift":

            @self.model.Objective(sense=pyo.maximize)
            def obj_rule_flex(m):
                """
                Maximizes the load shift over all time steps.
                """

                maximise_load_shift = pyo.quicksum(
                    m.load_shift_pos[t] for t in m.time_steps
                )

                return maximise_load_shift

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculate the marginal cost of the unit based on the provided time and power.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        # Initialize marginal cost
        marginal_cost = 0

        if self.opt_power_requirement.at[start] > 0:
            marginal_cost = (
                self.variable_cost_series.at[start]
                / self.opt_power_requirement.at[start]
            )

        return marginal_cost

    def as_dict(self) -> dict:
        """
        Returns the attributes of the unit as a dictionary, including specific attributes.

        Returns:
            dict: The attributes of the unit as a dictionary.
        """
        # Assuming unit_dict is a dictionary that you want to save to the database
        components_list = [component for component in self.model.dsm_blocks.keys()]

        # Convert the list to a delimited string
        components_string = ",".join(components_list)

        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "unit_type": "demand",
                "components": components_string,
            }
        )

        return unit_dict
