# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster
from assume.units.dsm_load_shift import DSMFlex

SOLVERS = ["appsi_highs", "gurobi", "glpk", "cbc", "cplex"]

logger = logging.getLogger(__name__)

# Set the log level to ERROR
logging.getLogger("pyomo").setLevel(logging.WARNING)


class SteelPlant(DSMFlex, SupportsMinMax):
    """
    The SteelPlant class represents a steel plant unit within an energy system, which can
    include various components like Direct Reduced Iron (DRI) plants,
    Electric Arc Furnaces (EAF), and other supporting technologies. The class models a unit
    that consumes energy for steel production and may also implement flexibility strategies
    like cost-based load shifting.

    Args:
        id (str): A unique identifier for the steel plant unit.
        unit_operator (str): The operator responsible for the steel plant.
        bidding_strategies (dict): A dictionary of bidding strategies, which define how the unit participates in energy markets.
        forecaster (Forecaster): A forecaster used to get key variables such as fuel or electricity prices.
        technology (str, optional): The technology of the steel plant. Default is "steel_plant".
        components (dict, optional): A dictionary describing the components of the steel plant, such as Electrolyser, DRI Plant, DRI Storage, and Electric Arc Furnace. Default is an empty dictionary.
        objective (str, optional): The objective function for the steel plant, typically focused on minimizing variable costs. Default is "min_variable_cost".
        flexibility_measure (str, optional): The flexibility measure for the steel plant, such as "cost_based_load_shift". Default is "cost_based_load_shift".
        demand (float, optional): The steel production demand, representing the amount of steel that needs to be produced. Default is 0.
        cost_tolerance (float, optional): The maximum allowable cost variation when shifting the load, used in flexibility measures. Default is 10.
        congestion_threshold (float, optional): The threshold for congestion management in the plant’s energy system. Default is 0.
        peak_load_cap (float, optional): The peak load capacity of the steel plant. Default is 0.
        node (str, optional): The network node where the steel plant is located in the energy system. Default is "node0".
        location (tuple[float, float], optional): A tuple representing the geographical coordinates (latitude, longitude) of the steel plant. Default is (0.0, 0.0).
        **kwargs: Additional keyword arguments that may be passed to support more specific configurations.

    Attributes:
        required_technologies (list): A list of required technologies for the plant to function, such as DRI plant and EAF.
        optional_technologies (list): A list of optional technologies that could enhance the plant, such as electrolyser or storage systems.
    """

    # Required and optional technologies for the steel plant
    required_technologies = ["dri_plant", "eaf"]
    optional_technologies = ["electrolyser", "hydrogen_buffer_storage", "dri_storage"]

    def __init__(
        self,
        id: str,
        unit_operator: str,
        bidding_strategies: dict,
        forecaster: Forecaster,
        components: dict[str, dict] = None,
        technology: str = "steel_plant",
        objective: str = "min_variable_cost",
        flexibility_measure: str = "cost_based_load_shift",
        demand: float = 0,
        cost_tolerance: float = 10,
        congestion_threshold: float = 0,
        peak_load_cap: float = 0,
        node: str = "node0",
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

        # check if the required components are present in the components dictionary
        for component in self.required_technologies:
            if component not in components.keys():
                raise ValueError(
                    f"Component {component} is required for the steel plant unit."
                )

        # check if the provided components are valid and do not contain any unknown components
        for component in components.keys():
            if (
                component not in self.required_technologies
                and component not in self.optional_technologies
            ):
                raise ValueError(
                    f"Components {component} is not a valid component for the steel plant unit."
                )

        self.natural_gas_price = self.forecaster["fuel_price_natural_gas"]
        self.hydrogen_price = self.forecaster["price_hydrogen"]
        self.electricity_price = self.forecaster["price_EOM"]
        self.iron_ore_price = self.forecaster.get_price("iron_ore")
        self.steel_demand = demand
        self.steel_price = self.forecaster.get_price("steel")
        self.lime_price = self.forecaster.get_price("lime")
        self.co2_price = self.forecaster.get_price("co2")

        # Calculate congestion forecast and set it as a forecast column in the forecaster
        self.congestion_signal = self.forecaster[f"{node}_congestion_severity"]
        self.renewable_utilisation_signal = self.forecaster[
            f"{node}_renewable_utilisation"
        ]

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.congestion_threshold = congestion_threshold
        self.peak_load_cap = peak_load_cap

        # Check for the presence of components
        self.has_h2storage = "hydrogen_buffer_storage" in self.components.keys()
        self.has_dristorage = "dri_storage" in self.components.keys()
        self.has_electrolyser = "electrolyser" in self.components.keys()

        # Initialize the model
        self.setup_model()

    def define_parameters(self):
        """
        Defines the parameters for the Pyomo model.
        """
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.electricity_price)},
        )

        if self.components["dri_plant"]["fuel_type"] in ["natural_gas", "both"]:
            if self.has_electrolyser:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={t: 0 for t in self.model.time_steps},
                )

            else:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value for t, value in enumerate(self.hydrogen_price)
                    },
                )

        elif self.components["dri_plant"]["fuel_type"] in ["hydrogen", "both"]:
            if self.has_electrolyser:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={t: 0 for t in self.model.time_steps},
                )
            else:
                self.model.hydrogen_price = pyo.Param(
                    self.model.time_steps,
                    initialize={
                        t: value for t, value in enumerate(self.hydrogen_price)
                    },
                )

        self.model.natural_gas_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.natural_gas_price)},
        )
        self.model.steel_demand = pyo.Param(initialize=self.steel_demand)
        self.model.steel_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.steel_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.co2_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.co2_price)},
        )
        self.model.lime_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.lime_price)},
            within=pyo.NonNegativeReals,
        )
        self.model.iron_ore_price = pyo.Param(
            self.model.time_steps,
            initialize={t: value for t, value in enumerate(self.iron_ore_price)},
            within=pyo.NonNegativeReals,
        )

    def define_variables(self):
        """
        Defines the variables for the Pyomo model.
        """
        self.model.total_power_input = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )
        self.model.variable_cost = pyo.Var(
            self.model.time_steps, within=pyo.NonNegativeReals
        )

    def initialize_process_sequence(self):
        """
        Establishes the process sequence and constraints for the steel plant, ensuring that
        different components and technologies interact correctly to support steel production.

        This function defines three key constraints:

        1. **Direct Hydrogen Flow Constraint**:
        - Ensures that hydrogen produced by the electrolyzer is either supplied directly
            to the DRI (Direct Reduced Iron) plant or stored in hydrogen buffer storage.
        - If storage is available, the constraint balances hydrogen inflow and outflow
            between the electrolyzer, hydrogen storage, and the DRI plant.
        - If no electrolyzer exists, it ensures that the DRI plant has an alternative hydrogen
            source (i.e., imported hydrogen).

        2. **Direct DRI Flow Constraint**:
        - Regulates the flow of directly reduced iron (DRI) from the DRI plant to the Electric Arc
            Furnace (EAF) or DRI storage.
        - If DRI storage is present, it ensures that part of the produced DRI can be stored
            for later use, maintaining balance between production, storage, and consumption in the EAF.
        - If no storage is available, all produced DRI must go directly to the EAF.

        3. **Material Flow Constraint from DRI Plant to Electric Arc Furnace**:
        - Ensures that all DRI produced by the DRI plant is consumed by the Electric Arc Furnace.
        - This constraint enforces that the total material output from the DRI plant must match
            the required DRI input for the EAF, preventing material imbalances.

        These constraints collectively ensure proper material and energy flow within the steel
        production process, maintaining energy efficiency and operational feasibility.
        """

        # Constraint for direct hydrogen flow from Electrolyser to DRI plant
        @self.model.Constraint(self.model.time_steps)
        def direct_hydrogen_flow_constraint(m, t):
            """
            Ensures the direct hydrogen flow from the electrolyser to the DRI plant or storage.
            """
            if self.has_electrolyser:
                if self.has_h2storage:
                    return (
                        m.dsm_blocks["electrolyser"].hydrogen_out[t]
                        + m.dsm_blocks["hydrogen_buffer_storage"].discharge[t]
                        == m.dsm_blocks["dri_plant"].hydrogen_in[t]
                        + m.dsm_blocks["hydrogen_buffer_storage"].charge[t]
                    )
                else:
                    return (
                        m.dsm_blocks["electrolyser"].hydrogen_out[t]
                        == m.dsm_blocks["dri_plant"].hydrogen_in[t]
                    )
            else:
                # If no electrolyser, ensure DRI plant hydrogen input is as expected
                return m.dsm_blocks["dri_plant"].hydrogen_in[t] >= 0

        # Constraint for direct hydrogen flow from Electrolyser to dri plant
        @self.model.Constraint(self.model.time_steps)
        def direct_dri_flow_constraint(m, t):
            """
            Ensures the direct DRI flow from the DRI plant to the EAF or DRI storage.
            """
            # This constraint allows part of the dri produced by the dri plant to go directly to the dri storage
            # The actual amount should ensure that it does not exceed the capacity or demand of the EAF
            if self.has_dristorage:
                return (
                    m.dsm_blocks["dri_plant"].dri_output[t]
                    + m.dsm_blocks["dri_storage"].discharge[t]
                    == m.dsm_blocks["eaf"].dri_input[t]
                    + m.dsm_blocks["dri_storage"].charge[t]
                )
            else:
                return (
                    m.dsm_blocks["dri_plant"].dri_output[t]
                    == m.dsm_blocks["eaf"].dri_input[t]
                )

        # Constraint for material flow from dri plant to Electric Arc Furnace
        @self.model.Constraint(self.model.time_steps)
        def shaft_to_arc_furnace_material_flow_constraint(m, t):
            """
            Ensures the material flow from the DRI plant to the Electric Arc Furnace.
            """
            return (
                m.dsm_blocks["dri_plant"].dri_output[t]
                == m.dsm_blocks["eaf"].dri_input[t]
            )

    def define_constraints(self):
        """
        Defines key optimization constraints for the steel plant model, ensuring the proper
        operation of the production process and energy consumption.

        This function establishes the following constraints:

        1. **Steel Output Association Constraint**:
        - Ensures that the total steel output from the Electric Arc Furnace (EAF) across all
            time steps meets the required steel demand.
        - This constraint enforces a global balance between steel production and demand over
            the entire time horizon, rather than enforcing it at each individual time step.

        2. **Total Power Input Constraint**:
        - Ensures that the total power input to the steel plant equals the sum of the power
            consumption of all energy-intensive components, including the EAF, DRI plant,
            and electrolyzer (if present).
        - This constraint ensures that energy demand is correctly accounted for and used
            in the optimization process.

        3. **Variable Cost per Time Step Constraint**:
        - Calculates the total variable operating cost per time step based on the power
            consumption and operating costs of the EAF, DRI plant, and electrolyzer (if available).
        - This constraint is useful for cost optimization, as it ensures that the total
            variable cost is accurately computed for each time step.

        These constraints collectively ensure proper energy and material flow within the steel
        production process, enforcing both production targets and cost minimization strategies.
        """

        @self.model.Constraint(self.model.time_steps)
        def steel_output_association_constraint(m, t):
            """
            Ensures the steel output meets the steel demand across all time steps.

            This constraint sums the steel output from the Electric Arc Furnace (EAF) over all time steps
            and ensures that it equals the steel demand. This is useful when the steel demand is to be met
            by the total production over the entire time horizon.
            """
            return (
                sum(m.dsm_blocks["eaf"].steel_output[t] for t in m.time_steps)
                == m.steel_demand
            )

        # Constraint for total power input
        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            """
            Ensures the total power input is the sum of power inputs of all components.
            """
            power_input = (
                m.dsm_blocks["eaf"].power_in[t] + m.dsm_blocks["dri_plant"].power_in[t]
            )
            if self.has_electrolyser:
                power_input += m.dsm_blocks["electrolyser"].power_in[t]

            return m.total_power_input[t] == power_input

        # Constraint for variable cost per time step
        @self.model.Constraint(self.model.time_steps)
        def cost_per_time_step(m, t):
            """
            Calculates the variable cost per time step.
            """

            variable_cost = (
                m.dsm_blocks["eaf"].operating_cost[t]
                + m.dsm_blocks["dri_plant"].operating_cost[t]
            )
            if self.has_electrolyser:
                variable_cost += m.dsm_blocks["electrolyser"].operating_cost[t]

            return m.variable_cost[t] == variable_cost
