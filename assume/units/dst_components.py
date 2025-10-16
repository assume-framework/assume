# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo

logger = logging.getLogger(__name__)


class HeatPump:
    """
    A class to represent a generic heat pump unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a heat pump, such as power input, heat output, and operational limitations
    (like ramp rates and minimum operating times).

    Args:
        max_power (float): Maximum allowable power input to the heat pump.
        cop (float): Coefficient of performance of the heat pump, i.e., the ratio of heat output to power input.
        time_steps (list[int]): A list of time steps over which the heat pump operates.
        min_power (float, optional): Minimum allowable power input to the heat pump. Defaults to 0.0.
        ramp_up (float, optional): Maximum allowed increase in power input per time step. Defaults to `max_power` if not provided.
        ramp_down (float, optional): Maximum allowed decrease in power input per time step. Defaults to `max_power` if not provided.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the heat pump must operate once it starts. Defaults to 0 (no restriction).
        min_down_steps (int, optional): Minimum number of consecutive time steps the heat pump must remain off after being shut down. Defaults to 0 (no restriction).
    """

    def __init__(
        self,
        max_power: float,
        cop: float,
        time_steps: list[int],
        min_power: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.max_power = max_power
        self.cop = cop
        self.time_steps = time_steps
        self.min_power = min_power
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a heat pump block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_power`: The maximum allowable power input.
                - `min_power`: The minimum allowable power input.
                - `cop`: Coefficient of performance of the heat pump.
                - `operating_cost`: Operating cost at each time step per time step.
                - `ramp_up`: Maximum allowed increase in power per time step.
                - `ramp_down`: Maximum allowed decrease in power per time step.
                - `min_operating_steps`: Minimum number of consecutive time steps the heat pump must operate.
                - `min_down_steps`: Minimum number of consecutive time steps the heat pump must remain off.
                - `initial_operational_status`: The initial operational status of the heat pump (0 for off, 1 for on).

            - **Variables**:
                - `power_in[t]`: Power input to the heat pump at each time step `t` (continuous, non-negative).
                - `heat_out[t]`: Heat output of the heat pump at each time step `t` (continuous, non-negative).
                - `operational_status[t]` (optional): A binary variable indicating whether the heat pump is operational (1) or off (0) at each time step `t`.
                - `start_up[t]` (optional): A binary variable indicating whether the heat pump is starting up (1) or not (0) at each time step `t`.
                - `shut_down[t]` (optional): A binary variable indicating whether the heat pump is shutting down (1) or not (0) at each time step `t`.

            - **Constraints**:
                - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the heat pump is operational.
                - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input when the heat pump is operational.
                - `cop_constraint[t]`: Enforces the relationship between power input and heat output based on the coefficient of performance (COP).
                - `operating_cost_constraint[t]`: Calculates the operating cost based on the power input and electricity price.
                - `ramp_up_constraint[t]`: Limits the increase in power input from one time step to the next according to the ramp-up rate.
                - `ramp_down_constraint[t]`: Limits the decrease in power input from one time step to the next according to the ramp-down rate.
                - `min_operating_time_constraint[t]`: Ensures the heat pump operates for at least the specified minimum number of consecutive time steps.
                - `min_downtime_constraint[t]`: Ensures the heat pump remains off for at least the specified minimum number of consecutive time steps after shutdown.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the heat pump block will be added.

        Returns:
            pyo.Block: A Pyomo block representing the heat pump with variables and constraints.
        """

        # Define parameters
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.cop = pyo.Param(initialize=self.cop)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.heat_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Coefficient of performance (COP) constraint
        @model_block.Constraint(self.time_steps)
        def cop_constraint(b, t):
            return b.heat_out[t] == b.power_in[t] * b.cop

        # Operating costs
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        # Ramp-up constraint and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
        )

        # Define additional variables and constraints for startup/shutdown and operational status
        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.min_power > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


class HeatResistor:
    """
    A class to represent a generic ptH resistor/ heat resistor unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a heat pump, such as power input, heat output, and operational limitations
    (like ramp rates and minimum operating times).

    Args:
        max_power (float): Maximum allowable power input to the heat_resistor.
        efficiency (float): efficiency of the heat_resistor.
        time_steps (list[int]): A list of time steps over which the heat pump operates.
        min_power (float, optional): Minimum allowable power input to the heat pump. Defaults to 0.0.
        ramp_up (float, optional): Maximum allowed increase in power input per time step. Defaults to `max_power` if not provided.
        ramp_down (float, optional): Maximum allowed decrease in power input per time step. Defaults to `max_power` if not provided.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the heat pump must operate once it starts. Defaults to 0 (no restriction).
        min_down_steps (int, optional): Minimum number of consecutive time steps the heat pump must remain off after being shut down. Defaults to 0 (no restriction).
    """

    def __init__(
        self,
        max_power: float,
        efficiency: float,
        time_steps: list[int],
        min_power: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.max_power = max_power
        self.efficiency = efficiency
        self.time_steps = time_steps
        self.min_power = min_power
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a heat pump block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_power`: The maximum allowable power input.
                - `min_power`: The minimum allowable power input.
                - `efficiency`: Efficienbcy of the resistor.
                - `operating_cost`: Operating cost at each time step per time step.
                - `ramp_up`: Maximum allowed increase in power per time step.
                - `ramp_down`: Maximum allowed decrease in power per time step.
                - `min_operating_steps`: Minimum number of consecutive time steps the heat pump must operate.
                - `min_down_steps`: Minimum number of consecutive time steps the heat pump must remain off.
                - `initial_operational_status`: The initial operational status of the heat pump (0 for off, 1 for on).

            - **Variables**:
                - `power_in[t]`: Power input to the heat pump at each time step `t` (continuous, non-negative).
                - `heat_out[t]`: Heat output of the heat pump at each time step `t` (continuous, non-negative).
                - `operational_status[t]` (optional): A binary variable indicating whether the heat pump is operational (1) or off (0) at each time step `t`.
                - `start_up[t]` (optional): A binary variable indicating whether the heat pump is starting up (1) or not (0) at each time step `t`.
                - `shut_down[t]` (optional): A binary variable indicating whether the heat pump is shutting down (1) or not (0) at each time step `t`.

            - **Constraints**:
                - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the heat pump is operational.
                - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input when the heat pump is operational.
                - `efficiency_constraint[t]`: Enforces the relationship between power input and heat output based on the efficiency.
                - `operating_cost_constraint[t]`: Calculates the operating cost based on the power input and electricity price.
                - `ramp_up_constraint[t]`: Limits the increase in power input from one time step to the next according to the ramp-up rate.
                - `ramp_down_constraint[t]`: Limits the decrease in power input from one time step to the next according to the ramp-down rate.
                - `min_operating_time_constraint[t]`: Ensures the heat pump operates for at least the specified minimum number of consecutive time steps.
                - `min_downtime_constraint[t]`: Ensures the heat pump remains off for at least the specified minimum number of consecutive time steps after shutdown.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the heat pump block will be added.

        Returns:
            pyo.Block: A Pyomo block representing the p2H resistor with variables and constraints.
        """

        # Define parameters
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.efficiency = pyo.Param(initialize=self.efficiency)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.heat_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Coefficient of performance (COP) constraint
        @model_block.Constraint(self.time_steps)
        def efficiency_constraint_heat_resistor(b, t):
            return b.heat_out[t] == b.power_in[t] * b.efficiency

        # Operating costs
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        # Ramp-up constraint and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
        )

        # Define additional variables and constraints for startup/shutdown and operational status
        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.min_power > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


class Boiler:
    """
    A class to represent a generic boiler unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of a boiler, which can be either electric or natural gas-based, along with ramp rates and operational
    limitations.

    Args:
        max_power (float): Maximum allowable power input to the boiler.
        efficiency (float): Efficiency of the boiler, defined as the ratio of heat output to power input (or fuel input).
        time_steps (list[int]): A list of time steps over which the boiler operates.
        fuel_type (str, optional): Type of fuel used by the boiler ('electricity' or 'natural_gas'). Defaults to 'electricity'.
        min_power (float, optional): Minimum allowable power input to the boiler. Defaults to 0.0.
        ramp_up (float, optional): Maximum allowed increase in power input per time step. Defaults to `max_power` if not provided.
        ramp_down (float, optional): Maximum allowed decrease in power input per time step. Defaults to `max_power` if not provided.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the boiler must operate once started. Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive time steps the boiler must remain off after being shut down. Defaults to 0.
        initial_operational_status (int, optional): The initial operational status of the boiler (0 for off, 1 for on). Defaults to 1.
    """

    def __init__(
        self,
        max_power: float,
        efficiency: float,
        time_steps: list[int],
        fuel_type: str = "electricity",
        min_power: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.max_power = max_power
        self.efficiency = efficiency
        self.time_steps = time_steps
        self.fuel_type = fuel_type
        self.min_power = min_power
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

        if self.fuel_type not in ["electricity", "natural_gas", "hydrogen_gas"]:
            raise ValueError(
                "Unsupported fuel_type for a boiler. Choose 'electricity' or 'natural_gas' or 'hydrogen_gas'."
            )

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a boiler block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_power`: The maximum allowable power input.
                - `min_power`: The minimum allowable power input.
                - `efficiency`: Efficiency of the boiler.
                - `ramp_up`: Maximum allowed increase in power per time step.
                - `ramp_down`: Maximum allowed decrease in power per time step.
                - `min_operating_steps`: Minimum number of consecutive time steps the boiler must operate.
                - `min_down_steps`: Minimum number of consecutive time steps the boiler must remain off.
                - `initial_operational_status`: The initial operational status of the boiler (0 for off, 1 for on).

            - **Variables**:
                - `power_in[t]` (for electric boilers): Power input at each time step `t` (continuous, non-negative).
                - `natural_gas_in[t]` (for natural gas boilers): Natural gas input at each time step `t` (continuous, non-negative).
                - `heat_out[t]`: Heat output at each time step `t` (continuous, non-negative).
                - `operational_status[t]` (optional, for electric boilers): A binary variable indicating whether the boiler is operational (1) or off (0) at each time step `t`.
                - `start_up[t]` (optional, for electric boilers): A binary variable indicating whether the boiler is starting up (1) or not (0) at each time step `t`.
                - `shut_down[t]` (optional, for electric boilers): A binary variable indicating whether the boiler is shutting down (1) or not (0) at each time step `t`.

            - **Constraints**:
                - `min_power_constraint[t]` (for electric boilers): Ensures that the power input is at least the minimum power input when the boiler is operational.
                - `max_power_constraint[t]` (for electric boilers): Ensures that the power input does not exceed the maximum power input when the boiler is operational.
                - `efficiency_constraint[t]`: Enforces the relationship between input (power or natural gas) and heat output based on the boiler's efficiency.
                - `ramp_up_constraint[t]`: Limits the increase in power input from one time step to the next according to the ramp-up rate.
                - `ramp_down_constraint[t]`: Limits the decrease in power input from one time step to the next according to the ramp-down rate.
                - `min_operating_time_constraint[t]`: Ensures the boiler operates for at least the specified minimum number of consecutive time steps.
                - `min_downtime_constraint[t]`: Ensures the boiler remains off for at least the specified minimum number of consecutive time steps after shutdown.
                - `operating_cost_constraint[t]`: Calculates the operating cost based on the power input and electricity price (for electric boilers).

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the boiler block will be added.

        Returns:
            pyo.Block: A Pyomo block representing the boiler with variables and constraints.
        """

        # depending on the fuel type, check if the model has the price profile for the fuel
        if self.fuel_type == "electricity":
            if not hasattr(model, "electricity_price"):
                raise ValueError(
                    "Electric boiler requires an electricity price profile in the model."
                )
        elif self.fuel_type == "natural_gas":
            if not hasattr(model, "natural_gas_price"):
                raise ValueError(
                    "Natural gas boiler requires a natural gas price profile in the model."
                )
        elif self.fuel_type == "hydrogen_gas":
            if not hasattr(model, "hydrogen_gas_price"):
                raise ValueError(
                    "Hydrogen gas boiler requires a hydrogen gas price profile in the model."
                )

        # Define parameters
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.efficiency = pyo.Param(initialize=self.efficiency)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.natural_gas_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.hydrogen_gas_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )

        model_block.heat_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Efficiency constraint based on fuel type
        @model_block.Constraint(self.time_steps)
        def efficiency_constraint(b, t):
            if self.fuel_type == "electricity":
                return b.heat_out[t] == b.power_in[t] * b.efficiency
            elif self.fuel_type == "natural_gas":
                return b.heat_out[t] == b.natural_gas_in[t] * b.efficiency
            elif self.fuel_type == "hydrogen_gas":
                return b.heat_out[t] == b.hydrogen_gas_in[t] * b.efficiency
            else:
                raise ValueError("Unsupported fuel_type for constraint.")

        # Set unused fuel input variables to zero
        if self.fuel_type == "electricity":

            @model_block.Constraint(self.time_steps)
            def natural_gas_input_zero_constraint(b, t):
                return b.natural_gas_in[t] == 0

            @model_block.Constraint(self.time_steps)
            def hydrogen_input_zero_constraint(b, t):
                return b.hydrogen_gas_in[t] == 0

        elif self.fuel_type == "natural_gas":

            @model_block.Constraint(self.time_steps)
            def power_input_zero_constraint(b, t):
                return b.power_in[t] == 0

            @model_block.Constraint(self.time_steps)
            def hydrogen_input_zero_constraint(b, t):
                return b.hydrogen_gas_in[t] == 0

        elif self.fuel_type == "hydrogen_gas":

            @model_block.Constraint(self.time_steps)
            def power_input_zero_constraint(b, t):
                return b.power_in[t] == 0

            @model_block.Constraint(self.time_steps)
            def natural_gas_input_zero_constraint(b, t):
                return b.natural_gas_in[t] == 0

        # Operating cost constraint based on fuel type
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            if self.fuel_type == "electricity":
                return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]
            elif self.fuel_type == "natural_gas":
                return (
                    b.operating_cost[t]
                    == b.natural_gas_in[t] * model.natural_gas_price[t]
                )
            elif self.fuel_type == "hydrogen_gas":
                return (
                    b.operating_cost[t]
                    == b.hydrogen_gas_in[t] * model.hydrogen_gas_price[t]
                )

        # Ramp-up constraint and ramp-down constraints
        if self.fuel_type == "natural_gas":

            @model_block.Constraint(self.time_steps)
            def ramp_up_constraint(b, t):
                if t == self.time_steps.at(1):
                    return b.natural_gas_in[t] <= b.ramp_up
                return b.natural_gas_in[t] - b.natural_gas_in[t - 1] <= b.ramp_up

            @model_block.Constraint(self.time_steps)
            def ramp_down_constraint(b, t):
                if t == self.time_steps.at(1):
                    return b.natural_gas_in[t] <= b.ramp_down
                return b.natural_gas_in[t - 1] - b.natural_gas_in[t] <= b.ramp_down

        elif self.fuel_type == "hydrogen_gas":

            @model_block.Constraint(self.time_steps)
            def ramp_up_constraint(b, t):
                if t == self.time_steps.at(1):
                    return b.hydrogen_gas_in[t] <= b.ramp_up
                return b.hydrogen_gas_in[t] - b.hydrogen_gas_in[t - 1] <= b.ramp_up

            @model_block.Constraint(self.time_steps)
            def ramp_down_constraint(b, t):
                if t == self.time_steps.at(1):
                    return b.hydrogen_gas_in[t] <= b.ramp_down
                return b.hydrogen_gas_in[t - 1] - b.hydrogen_gas_in[t] <= b.ramp_down

        elif self.fuel_type == "electricity":
            add_ramping_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        # Define operational status variable if min operating time, downtime, or min_power is required (for electric boilers)
        if (
            self.min_operating_steps > 0
            or self.min_down_steps > 0
            or self.min_power > 0
        ):
            if self.fuel_type != "electricity":
                raise ValueError(
                    "Operational status constraints are only supported for electric boilers. Use 'electricity' as the fuel_type."
                )

            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


class GenericStorage:
    """
    A class to represent a generic storage unit (e.g., battery) in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a storage system, including charging, discharging, state of charge (SOC),
    ramp rates, and storage losses.

    Args:
        max_capacity (float): Maximum energy storage capacity of the storage unit.
        min_capacity (float, optional): Minimum allowable state of charge (SOC). Defaults to 0.0.
        max_power_charge (float, optional): Maximum charging power of the storage unit. Defaults to `max_capacity` if not provided.
        max_power_discharge (float, optional): Maximum discharging power of the storage unit. Defaults to `max_capacity` if not provided.
        efficiency_charge (float, optional): Efficiency of the charging process. Defaults to 1.0.
        efficiency_discharge (float, optional): Efficiency of the discharging process. Defaults to 1.0.
        initial_soc (float, optional): Initial state of charge as a fraction of `max_capacity`. Defaults to 1.0.
        ramp_up (float, optional): Maximum allowed increase in charging/discharging power per time step. Defaults to None (no ramp constraint).
        ramp_down (float, optional): Maximum allowed decrease in charging/discharging power per time step. Defaults to None (no ramp constraint).
        storage_loss_rate (float, optional): Fraction of energy lost per time step due to storage inefficiencies. Defaults to 0.0.
    """

    def __init__(
        self,
        max_capacity: float,
        time_steps: list[int],
        min_capacity: float = 0.0,
        max_power_charge: float | None = None,
        max_power_discharge: float | None = None,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        initial_soc: float = 1.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        storage_loss_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # check if initial_soc is within the bounds [0, 1] and fix it if not
        if initial_soc > 1:
            initial_soc /= max_capacity
            logger.warning(
                f"Initial SOC is greater than 1.0. Setting it to {initial_soc}."
            )

        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.time_steps = time_steps
        self.max_power_charge = (
            max_capacity if max_power_charge is None else max_power_charge
        )
        self.max_power_discharge = (
            max_capacity if max_power_discharge is None else max_power_discharge
        )
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.initial_soc = initial_soc * max_capacity
        self.ramp_up = max_power_charge if ramp_up is None else ramp_up
        self.ramp_down = max_power_charge if ramp_down is None else ramp_down
        self.storage_loss_rate = storage_loss_rate
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a generic storage block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_capacity`: Maximum capacity of the storage unit.
                - `min_capacity`: Minimum state of charge (SOC).
                - `max_power_charge`: Maximum charging power.
                - `max_power_discharge`: Maximum discharging power.
                - `efficiency_charge`: Charging efficiency.
                - `efficiency_discharge`: Discharging efficiency.
                - `initial_soc`: Initial state of charge.
                - `ramp_up`: Maximum allowed ramp-up rate for charging and discharging.
                - `ramp_down`: Maximum allowed ramp-down rate for charging and discharging.
                - `storage_loss_rate`: Fraction of energy lost during storage.

            - **Variables**:
                - `soc[t]`: State of charge (SOC) at each time step `t`.
                - `charge[t]`: Charging power at each time step `t`.
                - `discharge[t]`: Discharging power at each time step `t`.

            - **Constraints**:
                - `soc_balance_rule[t]`: Tracks SOC changes over time based on charging, discharging, and storage loss.
                - `charge_ramp_up_constraint[t]`: Limits the ramp-up rate for charging if specified.
                - `discharge_ramp_up_constraint[t]`: Limits the ramp-up rate for discharging if specified.
                - `charge_ramp_down_constraint[t]`: Limits the ramp-down rate for charging if specified.
                - `discharge_ramp_down_constraint[t]`: Limits the ramp-down rate for discharging if specified.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the storage block will be added.

        Returns:
            pyo.Block: A Pyomo block representing the storage system with variables and constraints.
        """

        # Define parameters
        model_block.max_capacity = pyo.Param(initialize=self.max_capacity)
        model_block.min_capacity = pyo.Param(initialize=self.min_capacity)
        model_block.max_power_charge = pyo.Param(initialize=self.max_power_charge)
        model_block.max_power_discharge = pyo.Param(initialize=self.max_power_discharge)
        model_block.efficiency_charge = pyo.Param(initialize=self.efficiency_charge)
        model_block.efficiency_discharge = pyo.Param(
            initialize=self.efficiency_discharge
        )
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.initial_soc = pyo.Param(
            initialize=self.initial_soc * self.max_capacity
        )
        model_block.storage_loss_rate = pyo.Param(initialize=self.storage_loss_rate)

        # Define variables
        model_block.soc = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(model_block.min_capacity, model_block.max_capacity),
            doc="State of Charge at each time step",
        )
        model_block.charge = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power_charge),
            doc="Charging power at each time step",
        )
        model_block.discharge = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power_discharge),
            doc="Discharging power at each time step",
        )
        # # add a binary variable to disallow discharging and charging at the same time
        # model_block.status = pyo.Var(self.time_steps, within=pyo.Binary)

        # # add a constraint that disallows discharging and charging at the same time
        # @model_block.Constraint(self.time_steps)
        # def max_charge_power_constraint_with_status(b, t):
        #     return b.charge[t] <= b.max_power_charge * b.status[t]

        # @model_block.Constraint(self.time_steps)
        # def max_discharge_power_constraint_with_status(b, t):
        #     return b.discharge[t] <= b.max_power_discharge * (1 - b.status[t])

        # Define SOC dynamics with energy loss and efficiency
        @model_block.Constraint(self.time_steps)
        def soc_balance_rule(b, t):
            if t == self.time_steps.at(1):
                prev_soc = b.initial_soc
            else:
                prev_soc = b.soc[t - 1]
            return b.soc[t] == (
                prev_soc
                + b.efficiency_charge * b.charge[t]
                - (1 / b.efficiency_discharge) * b.discharge[t]
                - b.storage_loss_rate * prev_soc
            )

        # Apply ramp-up constraints if ramp_up is specified
        @model_block.Constraint(self.time_steps)
        def charge_ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return b.charge[t] <= b.ramp_up
            return b.charge[t] - b.charge[t - 1] <= b.ramp_up

        @model_block.Constraint(self.time_steps)
        def discharge_ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return b.discharge[t] <= b.ramp_up
            return b.discharge[t] - b.discharge[t - 1] <= b.ramp_up

        # Apply ramp-down constraints if ramp_down is specified
        @model_block.Constraint(self.time_steps)
        def charge_ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return b.charge[t] <= b.ramp_down
            return b.charge[t - 1] - b.charge[t] <= b.ramp_down

        @model_block.Constraint(self.time_steps)
        def discharge_ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return b.discharge[t] <= b.ramp_down
            return b.discharge[t - 1] - b.discharge[t] <= b.ramp_down

        return model_block


class ThermalStorage(GenericStorage):
    """
    A flexible thermal storage class that extends GenericStorage to support short-term and long-term scheduling.

    This class enables control over when the storage can charge or discharge, based on a binary profile.
    It is useful in seasonal or scheduled operations like industrial heating cycles.

    - For 'short-term': behaves like GenericStorage without behavioral restrictions.
    - For 'long-term': follows a storage schedule (0: charge-only, 1: discharge-only).

    Args:
        storage_type (str): Type of storage behavior: 'short-term' or 'long-term'.
        storage_schedule_profile (pd.Series, optional): Binary schedule for discharge availability (only required for 'long-term').
        **kwargs: All other parameters are inherited from GenericStorage.
    """

    def __init__(
        self,
        storage_type: str = "short-term",
        storage_schedule_profile: pd.Series | None = None,
        # --- NEW: generator mode parameters (used only if storage_type == "short-term_with_generator")
        eta_electric: float = 0.97,  # electric heater efficiency
        max_power: float
        | None = None,  # cap on power_in; if None, derive from max_power_charge/eta
        **kwargs,
    ):
        """
        Initializes the thermal storage instance.

        Raises:
            ValueError: If `storage_type` is 'long-term' and no schedule is provided.
            ValueError: If `storage_type` is not one of ['short-term', 'long-term'].
        """
        super().__init__(**kwargs)

        self.storage_type = storage_type.lower()
        self.storage_schedule_profile = storage_schedule_profile

        if self.storage_type == "long-term" and self.storage_schedule_profile is None:
            raise ValueError(
                "storage_schedule_profile is required for 'long-term' storage_type."
            )

        if self.storage_type not in [
            "short-term",
            "long-term",
            "short-term_with_generator",
        ]:
            raise ValueError(
                "storage_type must be one of: 'short-term', 'long-term', or 'short-term_with_generator'."
            )

        # ---------------- generator-mode parameters ----------------
        self.eta_electric = eta_electric
        if self.storage_type == "short-term_with_generator":
            # derive max electric input if not specified
            if (
                max_power is None
                and hasattr(self, "max_power_charge")
                and self.max_power_charge is not None
            ):
                max_power = self.max_power_charge / max(1e-6, eta_electric)
            self.max_power = max_power or 0.0
        else:
            self.max_power = 0.0  # unused in non-generator modes

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds thermal storage constraints to the model based on storage type.

        - If 'long-term', restrict charge/discharge based on the binary schedule profile.
        - Otherwise, behaves identically to GenericStorage.

        Args:
            model (pyo.ConcreteModel): The Pyomo optimization model.
            model_block (pyo.Block): The block to which this storage is added.

        Returns:
            pyo.Block: Updated model block with thermal storage constraints.
        """
        model_block = super().add_to_model(model, model_block)

        # --- Params for generator mode
        model_block.eta_electric = pyo.Param(initialize=self.eta_electric)
        model_block.max_Pelec = pyo.Param(initialize=self.max_power)
        model_block.is_generator_mode = pyo.Param(
            initialize=(self.storage_type == "short-term_with_generator"),
            within=pyo.Boolean,
        )

        # --- Vars for generator & cost
        # Always define for plotting/aggregation; bind to 0 if not in generator mode
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_Pelec),
        )
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # --- Coupling & cost
        if self.storage_type == "short-term_with_generator":
            # electric heater converts electricity to stored heat (into 'charge')
            @model_block.Constraint(self.time_steps)
            def electric_heater_charge(b, t):
                return b.charge[t] == b.power_in[t] * b.eta_electric

            if not hasattr(model, "electricity_price"):
                raise ValueError(
                    "ThermalStorage in 'short-term_with_generator' mode requires electricity_price on the model."
                )

            @model_block.Constraint(self.time_steps)
            def storage_cost_rule(b, t):
                return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        else:
            # bind to zero if not using a generator (and keep all GenericStorage behavior)
            @model_block.Constraint(self.time_steps)
            def zero_power_in(b, t):
                return b.power_in[t] == 0.0

            @model_block.Constraint(self.time_steps)
            def storage_cost_rule(b, t):
                return b.operating_cost[t] == 0.0

        if self.storage_type == "long-term":

            @model_block.Constraint(self.time_steps)
            def availability_charge_constraint(b, t):
                return (
                    b.charge[t]
                    <= (1 - self.storage_schedule_profile.iat[t]) * b.max_power_charge
                )

            @model_block.Constraint(self.time_steps)
            def availability_discharge_constraint(b, t):
                return (
                    b.discharge[t]
                    <= self.storage_schedule_profile.iat[t] * b.max_power_discharge
                )

        return model_block


class PVPlant:
    """
    A class to represent a Photovoltaic (PV) power plant unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a PV plant, including availability profiles and predefined power output profiles.

    Args:
        max_power (float): The maximum power output of the PV unit.
        time_steps (list[int]): A list of time steps over which the PV operates.
        availability_profile (pd.Series | None, optional): A pandas Series indicating the PV's availability with time_steps as indices
            and binary values (1 available, 0 unavailable). Defaults to None.
        power_profile (pd.Series | None, optional): A predefined power output profile. If provided, the PV follows this profile instead of optimizing the power output. Defaults to None.
    """

    def __init__(
        self,
        max_power: float,
        time_steps: list[int],
        availability_profile: pd.Series | None = None,
        power_profile: pd.Series | None = None,
        **kwargs,
    ):
        super().__init__()

        # Initialize attributes
        self.max_power = max_power
        self.time_steps = time_steps
        self.availability_profile = availability_profile
        self.power_profile = power_profile

        # Validate that only one profile is provided (either availability_profile or power_profile)
        if availability_profile is not None and power_profile is not None:
            raise ValueError(
                "Provide either `availability_profile` or `power_profile` for the residential PV plant, not both."
            )
        elif availability_profile is None and power_profile is None:
            raise ValueError(
                "Provide `availability_profile` or `power_profile` for the residential PV plant."
            )

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a PV plant block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_power`: Maximum allowable power output.

            - **Variables**:
                - `power[t]`: Power output of the PV plant at each time step `t`.
                - `operating_cost[t]`: Operating cost at each time step.

            - **Constraints**:
                - `power_profile_constraint`: Ensures the PV follows a predefined power profile if provided.
                - `availability_pv_constraint`: Ensures the PV operates only during available periods.
                - `max_power_pv_constraint`: Ensures the power output of the PV unit does not exceed the maximum power limit.
                - `operating_cost_constraint_rule`: Calculates the operating cost based on the power output and electricity price.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the PV plant block will be added.

        Returns:
            pyo.Block: A Pyomo block representing the PV plant with variables and constraints.
        """

        # Define parameters
        model_block.max_power = pyo.Param(
            initialize=self.max_power, within=pyo.NonNegativeReals
        )

        # Define variables
        model_block.power = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Define constraints

        # Predefined power profile constraint
        if self.power_profile is not None:
            if len(self.time_steps) != len(self.power_profile.index):
                raise ValueError(
                    "The length of the `time_steps` list must match the length of the `power_profile` index."
                )

            @model_block.Constraint(self.time_steps)
            def power_profile_constraint(b, t):
                """
                Ensures the PV follows the predefined power profile.
                """
                return b.power[t] == self.power_profile.iat[t]

        # Availability profile constraints
        if self.availability_profile is not None:
            if len(self.time_steps) != len(self.availability_profile.index):
                raise ValueError(
                    "The length of the `time_steps` list must match the length of the `availability_profile` index."
                )

            @model_block.Constraint(self.time_steps)
            def availability_pv_constraint(b, t):
                """
                Ensures the PV operates only during available periods.
                """
                return b.power[t] <= self.availability_profile.iat[t] * b.max_power

        # Maximum power constraint (redundant due to variable bounds, included for clarity)
        @model_block.Constraint(self.time_steps)
        def max_power_pv_constraint(b, t):
            """
            Ensures the power output of the PV unit does not exceed the maximum power limit.
            """
            return b.power[t] <= b.max_power

        # Operating costs
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == b.power[t] * 0

        return model_block


class Electrolyser:
    """
    A class to represent an electrolyser unit used for hydrogen production through electrolysis.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of an electrolyser, including power input, hydrogen output, ramp rates, and operating times.

    Args:
        max_power (float): The rated power capacity of the electrolyser.
        efficiency (float): The efficiency of the electrolysis process (0-1).
        time_steps (list[int]): A list of time steps over which the electrolyser operates.
        min_power (float): The minimum power required for operation.
        ramp_up (float, optional): The maximum rate at which the electrolyser can increase its power output. Defaults to `max_power`.
        ramp_down (float, optional): The maximum rate at which the electrolyser can decrease its power output. Defaults to `max_power`.
        min_operating_steps (int, optional): The minimum number of steps the electrolyser must operate continuously. Defaults to 1.
        min_down_steps (int, optional): The minimum number of downtime steps required between operating cycles. Defaults to 1.
        initial_operational_status (int, optional): The initial operational status of the electrolyser (0 for off, 1 for on). Defaults to 1.
    """

    def __init__(
        self,
        max_power: float,
        efficiency: float,
        time_steps: list[int],
        min_power: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 1,
        min_down_steps: int = 1,
        initial_operational_status: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.max_power = max_power
        self.efficiency = efficiency
        self.time_steps = time_steps
        self.min_power = min_power
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds an electrolyser block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_power`: Maximum allowable power input.
                - `efficiency`: Efficiency of the electrolyser.
                - `min_power`: Minimum allowable power input.
                - `ramp_up`: Maximum ramp-up rate.
                - `ramp_down`: Maximum ramp-down rate.
                - `min_operating_steps`: Minimum operating time.
                - `min_down_steps`: Minimum downtime between operating cycles.
                - `initial_operational_status`: Initial operational status of the electrolyser.

            - **Variables**:
                - `power_in[t]`: Power input to the electrolyser at each time step `t`.
                - `hydrogen_out[t]`: Hydrogen output at each time step `t`.
                - `operating_cost[t]`: Operating cost at each time step `t`.
                - `operational_status[t]` (optional): Binary variable indicating whether the electrolyser is operational.
                - `start_up[t]` (optional): Binary variable indicating whether the electrolyser has started up at time `t`.
                - `shut_down[t]` (optional): Binary variable indicating whether the electrolyser has shut down at time `t`.

            - **Constraints**:
                - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the electrolyser is operational.
                - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input.
                - `hydrogen_production_constraint[t]`: Relates power input to hydrogen output based on efficiency.
                - `ramp_up_constraint[t]`: Limits the ramp-up rate of power input.
                - `ramp_down_constraint[t]`: Limits the ramp-down rate of power input.
                - `min_operating_time_constraint[t]`: Ensures the electrolyser operates for a minimum duration.
                - `min_downtime_constraint[t]`: Ensures the electrolyser remains off for a minimum duration between operations.
                - `operating_cost_constraint[t]`: Calculates the operating cost based on power input and electricity price.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the electrolyser block will be added.

        Returns:
            pyo.Block: A Pyomo block representing the electrolyser with variables and constraints.
        """

        # Define parameters
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.efficiency = pyo.Param(initialize=self.efficiency)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_block.hydrogen_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Efficiency constraint
        @model_block.Constraint(self.time_steps)
        def hydrogen_production_constraint_rule(b, t):
            return b.hydrogen_out[t] == b.power_in[t] * b.efficiency

        # Operating cost constraint
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        # Ramp-up constraint and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
        )

        # Define additional variables and constraints for startup/shutdown and operational status
        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.min_power > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


class DRIPlant:
    """
    A class to represent a DRI (Direct Reduced Iron) plant in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of a DRI plant, including power consumption, fuel consumption (hydrogen, natural gas, or both),
    iron ore input, and ramp rates.

    Args:
        specific_hydrogen_consumption (float): The specific hydrogen consumption of the DRI plant (in MWh per ton of DRI).
        specific_natural_gas_consumption (float): The specific natural gas consumption of the DRI plant (in MWh per ton of DRI).
        specific_electricity_consumption (float): The specific electricity consumption of the DRI plant (in MWh per ton of DRI).
        specific_iron_ore_consumption (float): The specific iron ore consumption of the DRI plant (in ton per ton of DRI).
        max_power (float): The rated power capacity of the DRI plant.
        min_power (float): The minimum power required for operation.
        fuel_type (str): The type of fuel used by the DRI plant ("hydrogen", "natural_gas", "both").
        time_steps (list[int]): A list of time steps over which the DRI plant operates.
        ramp_up (float, optional): The maximum rate at which the DRI plant can increase its power output.
        ramp_down (float, optional): The maximum rate at which the DRI plant can decrease its power output.
        min_operating_steps (int, optional): The minimum number of steps the DRI plant must operate continuously. Defaults to 0.
        min_down_steps (int, optional): The minimum number of downtime steps required between operating cycles. Defaults to 0.
        initial_operational_status (int, optional): The initial operational status of the DRI plant (0 for off, 1 for on). Defaults to 1.
        natural_gas_co2_factor (float, optional): The CO2 emission factor for natural gas (in ton/MWh). Defaults to 0.5.
    """

    def __init__(
        self,
        specific_hydrogen_consumption: float,
        specific_natural_gas_consumption: float,
        specific_electricity_consumption: float,
        specific_iron_ore_consumption: float,
        max_power: float,
        min_power: float,
        fuel_type: str,
        time_steps: list[int],
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        natural_gas_co2_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.specific_hydrogen_consumption = specific_hydrogen_consumption
        self.specific_natural_gas_consumption = specific_natural_gas_consumption
        self.specific_electricity_consumption = specific_electricity_consumption
        self.specific_iron_ore_consumption = specific_iron_ore_consumption
        self.natural_gas_co2_factor = natural_gas_co2_factor

        self.max_power = max_power
        self.min_power = min_power
        self.fuel_type = fuel_type
        self.time_steps = time_steps
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a DRI plant block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `specific_hydrogen_consumption`: Hydrogen consumption per ton of DRI.
                - `specific_natural_gas_consumption`: Natural gas consumption per ton of DRI.
                - `specific_electricity_consumption`: Electricity consumption per ton of DRI.
                - `specific_iron_ore_consumption`: Iron ore consumption per ton of DRI.
                - `max_power`: Maximum allowable power input.
                - `min_power`: Minimum allowable power input.
                - `ramp_up`: Maximum ramp-up rate.
                - `ramp_down`: Maximum ramp-down rate.
                - `min_operating_steps`: Minimum operating time.
                - `min_down_steps`: Minimum downtime between operating cycles.
                - `initial_operational_status`: Initial operational status of the DRI plant.
                - `natural_gas_co2_factor`: CO2 emission factor for natural gas.

            - **Variables**:
                - `power_in[t]`: Power input to the DRI plant at each time step `t`.
                - `dri_output[t]`: DRI output at each time step `t`.
                - `natural_gas_in[t]`: Natural gas input at each time step `t`.
                - `hydrogen_in[t]`: Hydrogen input at each time step `t`.
                - `iron_ore_in[t]`: Iron ore input at each time step `t`.
                - `operating_cost[t]`: Operating cost at each time step `t`.
                - `operational_status[t]` (optional): Binary variable indicating whether the DRI plant is operational.
                - `start_up[t]` (optional): Binary variable indicating whether the DRI plant has started up at time `t`.
                - `shut_down[t]` (optional): Binary variable indicating whether the DRI plant has shut down at time `t`.

            - **Constraints**:
                - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the DRI plant is operational.
                - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input.
                - `dri_output_constraint[t]`: Links DRI output to fuel (hydrogen or natural gas) consumption.
                - `electricity_consumption_constraint[t]`: Ensures that electricity consumption is proportional to DRI output.
                - `iron_ore_constraint[t]`: Links iron ore input to DRI output.
                - `ramp_up_constraint[t]`: Limits the ramp-up rate of power input.
                - `ramp_down_constraint[t]`: Limits the ramp-down rate of power input.
                - `min_operating_time_constraint[t]`: Ensures the DRI plant operates for a minimum duration.
                - `min_downtime_constraint[t]`: Ensures the DRI plant remains off for a minimum duration between operations.
                - `operating_cost_constraint[t]`: Calculates the operating cost based on fuel and electricity consumption.
                - `co2_emission_constraint[t]`: Calculates the CO2 emissions based on natural gas consumption.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the DRI plant component will be added.

        Returns:
            pyo.Block: A Pyomo block representing the DRI plant with variables and constraints.
        """

        # depending on the fuel type, check if the model has the price profile for the fuel
        if self.fuel_type in ["natural_gas", "both"]:
            if not hasattr(model, "natural_gas_price"):
                raise ValueError(
                    "DRI plant requires a natural gas price profile if 'natural_gas' is used as the fuel type."
                )
        elif self.fuel_type in ["hydrogen", "both"]:
            if not hasattr(model, "hydrogen_price"):
                raise ValueError(
                    "DRI plant requires a hydrogen price profile if 'hydrogen' is used as the fuel type."
                )

        # Define parameters
        model_block.specific_hydrogen_consumption = pyo.Param(
            initialize=self.specific_hydrogen_consumption
        )
        model_block.specific_natural_gas_consumption = pyo.Param(
            initialize=self.specific_natural_gas_consumption
        )
        model_block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        model_block.specific_iron_ore_consumption = pyo.Param(
            initialize=self.specific_iron_ore_consumption
        )
        model_block.natural_gas_co2_factor = pyo.Param(
            initialize=self.natural_gas_co2_factor
        )

        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.iron_ore_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.natural_gas_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.co2_emission = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.hydrogen_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.dri_output = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Fuel consumption constraint
        @model_block.Constraint(self.time_steps)
        def dri_output_constraint(b, t):
            if self.fuel_type == "hydrogen":
                return (
                    b.dri_output[t]
                    == b.hydrogen_in[t] / b.specific_hydrogen_consumption
                )
            elif self.fuel_type == "natural_gas":
                return (
                    b.dri_output[t]
                    == b.natural_gas_in[t] / b.specific_natural_gas_consumption
                )
            elif self.fuel_type == "both":
                return b.dri_output[t] == (
                    b.natural_gas_in[t] / b.specific_natural_gas_consumption
                ) + (b.hydrogen_in[t] / b.specific_hydrogen_consumption)

        # Add Constraints to Zero Unused Fuel Inputs**
        @model_block.Constraint(self.time_steps)
        def zero_unused_fuel_constraints(b, t):
            if self.fuel_type == "hydrogen":
                return b.natural_gas_in[t] == 0
            elif self.fuel_type == "natural_gas":
                return b.hydrogen_in[t] == 0
            elif self.fuel_type == "both":
                return pyo.Constraint.Skip  # No action needed
            else:
                raise ValueError(f"Unknown fuel_type '{self.fuel_type}' specified.")

        # Electricity consumption constraint
        @model_block.Constraint(self.time_steps)
        def electricity_consumption_constraint(b, t):
            return b.power_in[t] == b.dri_output[t] * b.specific_electricity_consumption

        # Iron ore consumption constraint
        @model_block.Constraint(self.time_steps)
        def iron_ore_constraint(b, t):
            return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

        # CO2 emissions
        @model_block.Constraint(self.time_steps)
        def co2_emission_constraint(b, t):
            return b.co2_emission[t] == b.natural_gas_in[t] * b.natural_gas_co2_factor

        # Operating cost constraint
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint(b, t):
            operating_cost = (
                b.power_in[t] * model.electricity_price[t]
                + b.iron_ore_in[t] * model.iron_ore_price[t]
                + b.co2_emission[t] * model.co2_price[t]
            )
            if self.fuel_type == "natural_gas":
                operating_cost += b.natural_gas_in[t] * model.natural_gas_price[t]
            elif self.fuel_type == "hydrogen":
                operating_cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            elif self.fuel_type == "both":
                operating_cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.hydrogen_in[t] * model.hydrogen_price[t]
                )

            return b.operating_cost[t] == operating_cost

        # Ramp-up constraint and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
        )

        # Define additional variables and constraints for startup/shutdown and operational status
        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.min_power > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


class ElectricArcFurnace:
    """
    A class to represent an Electric Arc Furnace (EAF) in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of an EAF, including power consumption, DRI input, lime demand, and ramp rates.

    Args:
        max_power (float): The rated power capacity of the electric arc furnace.
        min_power (float): The minimum power requirement of the electric arc furnace.
        specific_electricity_consumption (float): The specific electricity consumption of the electric arc furnace (in MWh per ton of steel produced).
        specific_dri_demand (float): The specific demand for Direct Reduced Iron (DRI) in the electric arc furnace (in tons per ton of steel produced).
        specific_lime_demand (float): The specific demand for lime in the electric arc furnace (in tons per ton of steel produced).
        lime_co2_factor (float): The CO2 emission factor for lime production (in ton/MWh).
        time_steps (list[int]): A list of time steps over which the EAF operates.
        ramp_up (float, optional): The ramp-up rate of the electric arc furnace. Defaults to `max_power`.
        ramp_down (float, optional): The ramp-down rate of the electric arc furnace. Defaults to `max_power`.
        min_operating_steps (int, optional): The minimum number of steps the EAF must operate continuously. Defaults to 0.
        min_down_steps (int, optional): The minimum number of downtime steps required between operating cycles. Defaults to 0.
        initial_operational_status (int, optional): The initial operational status of the EAF (0 for off, 1 for on). Defaults to 1.
    """

    def __init__(
        self,
        max_power: float,
        min_power: float,
        specific_electricity_consumption: float,
        specific_dri_demand: float,
        specific_lime_demand: float,
        lime_co2_factor: float,
        time_steps: list[int],
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.max_power = max_power
        self.min_power = min_power
        self.specific_electricity_consumption = specific_electricity_consumption
        self.specific_dri_demand = specific_dri_demand
        self.specific_lime_demand = specific_lime_demand
        self.lime_co2_factor = lime_co2_factor
        self.time_steps = time_steps
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds an EAF block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_power`: Maximum allowable power input.
                - `min_power`: Minimum allowable power input.
                - `specific_electricity_consumption`: Electricity consumption per ton of steel produced.
                - `specific_dri_demand`: DRI demand per ton of steel produced.
                - `specific_lime_demand`: Lime demand per ton of steel produced.
                - `lime_co2_factor`: CO2 emission factor for lime production.
                - `ramp_up`: Maximum ramp-up rate.
                - `ramp_down`: Maximum ramp-down rate.
                - `min_operating_steps`: Minimum operating time.
                - `min_down_steps`: Minimum downtime between operating cycles.
                - `initial_operational_status`: Initial operational status of the EAF.

            - **Variables**:
                - `power_in[t]`: Power input to the EAF at each time step `t`.
                - `dri_input[t]`: DRI input at each time step `t`.
                - `steel_output[t]`: Steel output at each time step `t`.
                - `operating_cost[t]`: Operating cost at each time step `t`.
                - `co2_emission[t]`: CO2 Emissions at each time step `t`.
                - `lime_demand[t]`: Lime demand at each time step `t`.
                - `operational_status[t]` (optional): Binary variable indicating whether the EAF is operational.
                - `start_up[t]` (optional): Binary variable indicating whether the EAF has started up at time `t`.
                - `shut_down[t]` (optional): Binary variable indicating whether the EAF has shut down at time `t`.

            - **Constraints**:
                - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the EAF is operational.
                - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input.
                - `steel_output_dri_relation_constraint[t]`: Links steel output to DRI input.
                - `steel_output_power_relation_constraint[t]`: Links steel output to power consumption.
                - `lime_demand_constraint[t]`: Links lime demand to steel output.
                - `co2_emission_constraint[t]`: Links CO2 emissions to lime demand.
                - `ramp_up_constraint[t]`: Limits the ramp-up rate of power input.
                - `ramp_down_constraint[t]`: Limits the ramp-down rate of power input.
                - `min_operating_time_constraint[t]`: Ensures the EAF operates for a minimum duration.
                - `min_down_time_constraint[t]`: Ensures the EAF remains off for a minimum duration between operations.
                - `operating_cost_constraint[t]`: Calculates the operating cost based on power input, CO2 emissions, and lime consumption.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the EAF component will be added.

        Returns:
            pyo.Block: A Pyomo block representing the EAF with variables and constraints.
        """

        # Define parameters
        model_block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        model_block.specific_dri_demand = pyo.Param(initialize=self.specific_dri_demand)
        model_block.specific_lime_demand = pyo.Param(
            initialize=self.specific_lime_demand
        )
        model_block.lime_co2_factor = pyo.Param(initialize=self.lime_co2_factor)

        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.dri_input = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.steel_output = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)
        model_block.co2_emission = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.lime_demand = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        # Steel output based on DRI input
        @model_block.Constraint(self.time_steps)
        def steel_output_dri_relation_constraint(b, t):
            return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

        # Steel output based on power consumption
        @model_block.Constraint(self.time_steps)
        def steel_output_power_relation_constraint(b, t):
            return (
                b.power_in[t] == b.steel_output[t] * b.specific_electricity_consumption
            )

        # Lime demand based on steel output
        @model_block.Constraint(self.time_steps)
        def lime_demand_constraint(b, t):
            return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

        # CO2 emissions based on lime demand
        @model_block.Constraint(self.time_steps)
        def co2_emission_constraint(b, t):
            return b.co2_emission[t] == b.lime_demand[t] * b.lime_co2_factor

        # Operating cost constraint
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint(b, t):
            return (
                b.operating_cost[t]
                == b.power_in[t] * model.electricity_price[t]
                + b.co2_emission[t] * model.co2_price[t]
                + b.lime_demand[t] * model.lime_price[t]
            )

        # Ramp-up constraint and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
        )

        # Define additional variables and constraints for startup/shutdown and operational status
        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.min_power > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


class GrindingMill:
    """
    A class to represent a generic grinding mill unit in an energy system model.

    This class handles both raw material milling and cement grinding by modeling electricity consumption,
    material flow, and operational constraints such as ramp rates, availability profiles, and maximum operating hours.

    Args:
        max_power (float): Maximum allowable power input to the mill.
        efficiency (float): Efficiency of the mill, defined as the ratio of material output to power input.
        time_steps (list[int]): A list of time steps over which the mill operates.
        max_operating_hours (int): Maximum number of operating hours per day.
        availability_profile (list[int] | None): Time series indicating the availability of the mill (1 for available, 0 for unavailable).
        ramp_up (float | None, optional): Maximum allowed increase in power input per time step. Defaults to `max_power` if not provided.
        ramp_down (float | None, optional): Maximum allowed decrease in power input per time step. Defaults to `max_power` if not provided.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the mill must operate once started. Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive time steps the mill must remain off after being shut down. Defaults to 0.
        initial_operational_status (int, optional): The initial operational status of the mill (0 for off, 1 for on). Defaults to 1.
        specific_electricity_consumption (float): Electricity consumption per ton of material output (MWh/tonne).

    Returns:
        pyo.Block: A Pyomo block representing the grinding mill with variables and constraints.
    """

    def __init__(
        self,
        max_power,
        min_power,
        efficiency,
        time_steps,
        max_operating_hours: int = 0,
        availability_profile="yes",
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps=0,
        min_down_steps=0,
        initial_operational_status=1,
        specific_electricity_consumption=0.0,
    ):
        super().__init__()

        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.time_steps = time_steps
        self.availability_profile = availability_profile
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.max_operating_hours = max_operating_hours
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.specific_electricity_consumption = specific_electricity_consumption

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        # Define parameters
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.efficiency = pyo.Param(initialize=self.efficiency)
        model_block.max_operating_hours = pyo.Param(initialize=self.max_operating_hours)
        model_block.grinder_availability_profile = pyo.Param(
            self.time_steps,
            initialize={t: 1 for t in self.time_steps},
        )
        model_block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_block.material_input = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.material_output = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.operating_hours = pyo.Var(self.time_steps, within=pyo.Binary)
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)

        # Constraints

        @model_block.Constraint(self.time_steps)
        def material_input_output_relation(b, t):
            return b.material_input[t] == b.material_output[t] / b.efficiency

        @model_block.Constraint(self.time_steps)
        def material_flow_constraint(b, t):
            return (
                b.power_in[t]
                == b.material_output[t] * b.specific_electricity_consumption
            )

        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        @model_block.Constraint(range(len(self.time_steps) // 24))
        def daily_max_operating_hours_constraint(b, d):
            time_indices = [idx for idx in self.time_steps if idx // 24 == d]
            return (
                sum(b.operating_hours[idx] for idx in time_indices)
                <= b.max_operating_hours
            )

        @model_block.Constraint(self.time_steps)
        def operation_hours_binding_constraint(b, t):
            big_M = self.max_power  # Use max_power as the scaling factor
            if self.availability_profile == "no":
                return (
                    b.power_in[t]
                    <= b.operating_hours[t] * big_M * b.grinder_availability_profile[t]
                )
            elif self.availability_profile == "yes":
                return (
                    b.power_in[t]
                    <= b.operating_hours[t]
                    * big_M
                    * model.grinder_availability_profile[t]
                )

        # Ramp-up constraint and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
        )

        # Define additional variables and constraints for startup/shutdown and operational status
        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.max_power > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
            )

        return model_block


# ---------------------------------------------------------------------------
# PREHEATER  (medium-T duty ~300–800 °C)
# ---------------------------------------------------------------------------
class Preheater:
    """
    Fuel-switchable preheater for cement raw meal.

    Modes:
      - fuel_type = "electricity":     heat_out = eta_electric * power_in
      - fuel_type = "fossil":          heat_out = eta_fossil  * (ng_in + coal_in)
      - fuel_type = "both":            heat_out = eta_elec*power_in + eta_fossil*(ng_in + coal_in)

    Fossil split:
      ng_in[t]   = fossil_ng_share * (ng_in[t] + coal_in[t])
      coal_in[t] = (1 - fossil_ng_share) * (ng_in[t] + coal_in[t])
    """

    def __init__(
        self,
        # sizing
        max_heat_out: float,  # [MW_th]
        specific_heat_demand: float,  # [MWh_th per t raw meal]
        time_steps: list[int],
        # operations
        fuel_type: str = "electricity",  # "electricity" | "fossil" | "both"
        eta_electric: float = 0.98,
        eta_fossil: float = 0.90,
        fossil_ng_share: float = 1.0,  # 1.0 => 100% NG, 0.0 => 100% coal
        max_power: float | None = None,  # optional cap for electric path [MW_e]
        ramp_up: float | None = None,  # [MW_th per step] on heat_out
        ramp_down: float | None = None,  # [MW_th per step] on heat_out
        # emissions
        ng_co2_factor: float = 0.202,  # [tCO2/MWh_LHV] ~nat. gas
        coal_co2_factor: float = 0.341,  # [tCO2/MWh_LHV] ~bituminous coal
        # on/off logic (kept simple; applies to heat_out)
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.time_steps = time_steps

        self.max_heat_out = max_heat_out
        self.specific_heat_demand = specific_heat_demand

        self.fuel_type = fuel_type.lower()
        assert self.fuel_type in ["electricity", "fossil", "both"]

        self.eta_electric = eta_electric
        self.eta_fossil = eta_fossil
        self.fossil_ng_share = fossil_ng_share

        self.max_power = (
            max_power
            if max_power is not None
            else max_heat_out / max(1e-6, eta_electric)
        )

        self.ramp_up = max_heat_out if ramp_up is None else ramp_up
        self.ramp_down = max_heat_out if ramp_down is None else ramp_down

        self.ng_co2_factor = ng_co2_factor
        self.coal_co2_factor = coal_co2_factor

        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        # --- Required price streams
        if self.fuel_type in ["fossil", "both"]:
            if not hasattr(model, "natural_gas_price"):
                raise ValueError(
                    "Preheater requires natural_gas_price for fossil operation."
                )
            if not hasattr(model, "coal_price"):
                raise ValueError("Preheater requires coal_price for fossil operation.")
        if self.fuel_type in ["electricity", "both"]:
            if not hasattr(model, "electricity_price"):
                raise ValueError(
                    "Preheater requires electricity_price for electric operation."
                )

        # --- Params ---
        model_block.max_heat_out = pyo.Param(initialize=self.max_heat_out)
        model_block.specific_heat_demand = pyo.Param(
            initialize=self.specific_heat_demand
        )

        model_block.eta_electric = pyo.Param(initialize=self.eta_electric)
        model_block.eta_fossil = pyo.Param(initialize=self.eta_fossil)
        model_block.fossil_ng_share = pyo.Param(
            initialize=self.fossil_ng_share, within=pyo.UnitInterval
        )

        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)

        model_block.ng_co2_factor = pyo.Param(initialize=self.ng_co2_factor)
        model_block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor)

        model_block.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_block.min_down_steps = pyo.Param(initialize=self.min_down_steps)
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # --- Variables ---
        model_block.heat_out = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_heat_out)
        )
        model_block.raw_meal_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        model_block.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_block.natural_gas_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.coal_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.external_heat_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)
        model_block.co2_emission = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        # Optional on/off status for heat_out (simple, analogous spirit to helpers)
        model_block.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

        # --- Core relationships ---
        @model_block.Constraint(self.time_steps)
        def throughput_from_heat(b, t):
            return b.raw_meal_out[t] == b.heat_out[t] / b.specific_heat_demand

        # heat balance
        @model_block.Constraint(self.time_steps)
        def heat_balance(b, t):
            # fuel/electric contribution
            if self.fuel_type == "electricity":
                gen = b.power_in[t] * b.eta_electric
            elif self.fuel_type == "fossil":
                gen = (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
            elif self.fuel_type == "both":
                gen = (
                    b.power_in[t] * b.eta_electric
                    + (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
                )
            else:
                # if you ever add H2 here later
                gen = b.hydrogen_in[t] * b.eta_fossil

            # total useful heat available to the preheater = generated + external waste heat
            return b.heat_out[t] == gen + b.external_heat_in[t]

        # Enforce unused inputs = 0 for pure modes (matches the pattern used in DRI/Boiler) :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
        if self.fuel_type == "electricity":

            @model_block.Constraint(self.time_steps)
            def zero_fossil_when_electric(b, t):
                return b.natural_gas_in[t] + b.coal_in[t] == 0
        elif self.fuel_type == "fossil":

            @model_block.Constraint(self.time_steps)
            def zero_electric_when_fossil(b, t):
                return b.power_in[t] == 0

        # Fossil split (active in fossil & both)
        if self.fuel_type in ["fossil", "both"]:

            @model_block.Constraint(self.time_steps)
            def fossil_split_ng(b, t):
                return b.natural_gas_in[t] == b.fossil_ng_share * (
                    b.natural_gas_in[t] + b.coal_in[t]
                )

            @model_block.Constraint(self.time_steps)
            def fossil_split_coal(b, t):
                return b.coal_in[t] == (1 - b.fossil_ng_share) * (
                    b.natural_gas_in[t] + b.coal_in[t]
                )

        # Simple on/off binding on heat_out
        @model_block.Constraint(self.time_steps)
        def max_heat_if_on(b, t):
            return b.heat_out[t] <= b.max_heat_out * b.operational_status[t]

        # Ramping on heat_out (works for any fuel path)
        @model_block.Constraint(self.time_steps)
        def heat_ramp_up(b, t):
            if t == self.time_steps.at(1):
                return b.heat_out[t] <= b.ramp_up
            return b.heat_out[t] - b.heat_out[t - 1] <= b.ramp_up

        @model_block.Constraint(self.time_steps)
        def heat_ramp_down(b, t):
            if t == self.time_steps.at(1):
                return b.heat_out[t] <= b.ramp_down
            return b.heat_out[t - 1] - b.heat_out[t] <= b.ramp_down

        # Emissions
        @model_block.Constraint(self.time_steps)
        def co2_emission_constraint(b, t):
            return (
                b.co2_emission[t]
                == b.natural_gas_in[t] * b.ng_co2_factor
                + b.coal_in[t] * b.coal_co2_factor
            )

        # Operating cost (mirrors DRI/Boiler cost assembly) :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint(b, t):
            cost = 0
            if self.fuel_type in ["electricity", "both"]:
                cost += b.power_in[t] * model.electricity_price[t]
            if self.fuel_type in ["fossil", "both"]:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                )
            # CO2 price always applies to fuel emissions
            cost += b.co2_emission[t] * model.co2_price[t]
            return b.operating_cost[t] == cost

        return model_block


# ---------------------------------------------------------------------------
# CALCINER  (high-T; process CO2; external_heat port)
# ---------------------------------------------------------------------------
class Calciner:
    """
    Fuel-switchable high-temperature calciner (~920–930 °C).

    Modes:
      - fuel_type = "electricity":
            heat_out = eta_electric * power_in
      - fuel_type = "fossil":           (NG + coal by fixed share)
            heat_out = eta_fossil * (ng_in + coal_in)
      - fuel_type = "both":
            heat_out = eta_electric * power_in + eta_fossil * (ng_in + coal_in)
      - fuel_type = "hydrogen":
            heat_out = eta_fossil * h2_in   (use eta_fossil as generic burner eff)

    Also accounts for process CO2 from calcination and energy CO2 for fossil fuels.
    """

    def __init__(
        self,
        # sizing
        max_heat_out: float,  # [MW_th] cap
        specific_heat_demand: float,  # [MWh_th per t_clinker] calcination heat
        time_steps: list[int],
        specific_electricity_aux: float = 0.0,  # [MWh_el per t_clinker] (ID fan etc.), optional
        # operations
        fuel_type: str = "electricity",  # "electricity" | "fossil" | "both" | "hydrogen"
        eta_electric: float = 0.95,
        eta_fossil: float = 0.90,
        fossil_ng_share: float = 1.0,  # 1.0 => 100% NG, 0.0 => 100% coal
        max_power: float | None = None,  # [MW_e] cap (if electric present)
        ramp_up: float | None = None,  # [MW_th/step] on heat_out
        ramp_down: float | None = None,  # [MW_th/step] on heat_out
        # emissions
        calcination_emission_factor: float = 0.525,  # [tCO2/t_clinker] typical order
        ng_co2_factor: float = 0.202,  # [tCO2/MWh_th]
        coal_co2_factor: float = 0.341,  # [tCO2/MWh_th]
        # on/off (bind to heat_out like we did for preheater)
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        name: str = "calciner",
    ):
        super().__init__()
        self.time_steps = time_steps
        self.max_heat_out = max_heat_out
        self.specific_heat_demand = specific_heat_demand
        self.specific_electricity_aux = specific_electricity_aux

        self.fuel_type = fuel_type.lower()
        assert self.fuel_type in ["electricity", "fossil", "both", "hydrogen"]

        self.eta_electric = eta_electric
        self.eta_fossil = eta_fossil
        self.fossil_ng_share = fossil_ng_share

        self.max_power = (
            max_power
            if max_power is not None
            else max_heat_out / max(1e-6, eta_electric)
        )
        self.rup = max_heat_out if ramp_up is None else ramp_up
        self.rdn = max_heat_out if ramp_down is None else ramp_down

        self.cf_proc = calcination_emission_factor
        self.ng_ef = ng_co2_factor
        self.coal_ef = coal_co2_factor

        self.min_up = min_operating_steps
        self.min_dn = min_down_steps
        self.init_on = initial_operational_status
        self.name = name

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        # price streams present?
        if self.fuel_type in ["electricity", "both"] and not hasattr(
            model, "electricity_price"
        ):
            raise ValueError(
                "Calciner requires electricity_price for electric operation."
            )
        if self.fuel_type in ["fossil", "both"] and (
            not hasattr(model, "natural_gas_price") or not hasattr(model, "coal_price")
        ):
            raise ValueError(
                "Calciner requires natural_gas_price and coal_price for fossil operation."
            )
        if self.fuel_type == "hydrogen" and not hasattr(model, "hydrogen_price"):
            raise ValueError("Calciner requires hydrogen_price for hydrogen operation.")

        # params
        model_block.max_heat_out = pyo.Param(initialize=self.max_heat_out)
        model_block.specific_heat_demand = pyo.Param(
            initialize=self.specific_heat_demand
        )
        model_block.specific_electricity_aux = pyo.Param(
            initialize=self.specific_electricity_aux
        )

        model_block.eta_electric = pyo.Param(initialize=self.eta_electric)
        model_block.eta_fossil = pyo.Param(initialize=self.eta_fossil)
        model_block.fossil_ng_share = pyo.Param(
            initialize=self.fossil_ng_share, within=pyo.UnitInterval
        )

        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.ramp_up = pyo.Param(initialize=self.rup)
        model_block.ramp_down = pyo.Param(initialize=self.rdn)

        model_block.cf_proc = pyo.Param(initialize=self.cf_proc)
        model_block.ng_co2_factor = pyo.Param(initialize=self.ng_ef)
        model_block.coal_co2_factor = pyo.Param(initialize=self.coal_ef)

        model_block.min_operating_steps = pyo.Param(initialize=self.min_up)
        model_block.min_down_steps = pyo.Param(initialize=self.min_dn)
        model_block.initial_operational_status = pyo.Param(initialize=self.init_on)

        # vars
        model_block.heat_out = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_heat_out)
        )
        model_block.effective_heat_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.clinker_out = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )  # downstream handle (t/h equivalent)
        model_block.fossil_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_block.natural_gas_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.coal_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.hydrogen_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)
        model_block.co2_proc = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.co2_energy = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

        # heat balance by mode
        @model_block.Constraint(self.time_steps)
        def heat_balance(b, t):
            if self.fuel_type == "electricity":
                return b.heat_out[t] == b.power_in[t] * b.eta_electric
            elif self.fuel_type == "fossil":
                return (
                    b.heat_out[t] == (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
                )
            elif self.fuel_type == "both":
                return (
                    b.heat_out[t]
                    == b.power_in[t] * b.eta_electric
                    + (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
                )
            elif self.fuel_type == "hydrogen":
                return b.heat_out[t] == b.hydrogen_in[t] * b.eta_fossil

        # zero unused inputs in pure modes
        if self.fuel_type == "electricity":

            @model_block.Constraint(self.time_steps)
            def zero_fossil_h2_when_electric(b, t):
                return b.natural_gas_in[t] + b.coal_in[t] + b.hydrogen_in[t] == 0

        elif self.fuel_type == "fossil":

            @model_block.Constraint(self.time_steps)
            def zero_electric_h2_when_fossil(b, t):
                return b.power_in[t] + b.hydrogen_in[t] == 0

            # fossil split
            @model_block.Constraint(self.time_steps)
            def fossil_split_ng(b, t):
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_split_coal(b, t):
                return b.coal_in[t] == (1 - b.fossil_ng_share) * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_sum_equals(b, t):
                return b.fossil_in[t] == b.natural_gas_in[t] + b.coal_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_balance(b, t):
                return b.heat_out[t] == b.fossil_in[t] * b.eta_fossil

        elif self.fuel_type == "both":

            @model_block.Constraint(self.time_steps)
            def fossil_split_ng(b, t):
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_split_coal(b, t):
                return b.coal_in[t] == (1 - b.fossil_ng_share) * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_sum_equals(b, t):
                return b.fossil_in[t] == b.natural_gas_in[t] + b.coal_in[t]

        elif self.fuel_type == "hydrogen":

            @model_block.Constraint(self.time_steps)
            def zero_electric_fossil_when_h2(b, t):
                return b.power_in[t] + b.natural_gas_in[t] + b.coal_in[t] == 0

        # throughput (link heat to clinker rate with specific demand)
        @model_block.Constraint(self.time_steps)
        def clinker_from_heat(b, t):
            # Always drive clinker by the effective heat input
            return b.clinker_out[t] == b.effective_heat_in[t] / b.specific_heat_demand

        # auxiliaries electricity (optional)
        if self.specific_electricity_aux > 0 and self.fuel_type in [
            "electricity",
            "both",
        ]:

            @model_block.Constraint(self.time_steps)
            def aux_power_min(b, t):
                return b.power_in[t] >= b.clinker_out[t] * self.specific_electricity_aux

        # on/off & ramping on heat_out
        @model_block.Constraint(self.time_steps)
        def max_heat_if_on(b, t):
            return b.heat_out[t] <= b.max_heat_out * b.operational_status[t]

        @model_block.Constraint(self.time_steps)
        def heat_ramp_up(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.heat_out[t] - b.heat_out[t - 1] <= b.ramp_up

        @model_block.Constraint(self.time_steps)
        def heat_ramp_down(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.heat_out[t - 1] - b.heat_out[t] <= b.ramp_down

        # emissions
        @model_block.Constraint(self.time_steps)
        def process_co2(b, t):
            return b.co2_proc[t] == b.clinker_out[t] * b.cf_proc

        @model_block.Constraint(self.time_steps)
        def energy_co2(b, t):
            # hydrogen & electricity assumed zero direct energy CO2 here (scope2 handled elsewhere if needed)
            return b.co2_energy[t] == (
                b.natural_gas_in[t] * b.ng_co2_factor + b.coal_in[t] * b.coal_co2_factor
            )

        # cost
        @model_block.Constraint(self.time_steps)
        def operating_cost_rule(b, t):
            cost = 0
            if self.fuel_type in ["electricity", "both"]:
                cost += b.power_in[t] * model.electricity_price[t]
            if self.fuel_type in ["fossil", "both"]:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                )
            if self.fuel_type == "hydrogen":
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]

            # auxiliary electricity (always charged)
            if self.specific_electricity_aux > 0:
                aux_MWh = b.clinker_out[t] * self.specific_electricity_aux
                cost += aux_MWh * model.electricity_price[t]
            # CO2 price on process + energy CO2
            cost += (b.co2_proc[t] + b.co2_energy[t]) * model.co2_price[t]
            return b.operating_cost[t] == cost

        return model_block


# ---------------------------------------------------------------------------
# ROTARY KILN  (~1450 °C; energy CO2 only; external_heat port)
# ---------------------------------------------------------------------------
class Kiln:
    """
    Fuel-switchable rotary kiln (~1450 °C) for final clinkerisation.

    Modes:
      - fuel_type = "electricity":
            heat_out = eta_electric * power_in
      - fuel_type = "fossil":           (NG + coal by fixed share)
            heat_out = eta_fossil * (ng_in + coal_in)
      - fuel_type = "both":
            heat_out = eta_electric * power_in + eta_fossil * (ng_in + coal_in)
      - fuel_type = "hydrogen":
            heat_out = eta_fossil * h2_in   (use eta_fossil as generic burner eff)

    Differs from calciner:
      - Higher specific heat demand (~3 MWh/t clinker).
      - Little/no process CO₂ (mainly energy CO₂).
    """

    def __init__(
        self,
        max_heat_out: float,  # [MW_th] cap
        specific_heat_demand: float,  # [MWh_th per t_clinker]
        time_steps: list[int],
        specific_electricity_aux: float = 0.0,  # [MWh_el per t_clinker], ID fans etc.
        fuel_type: str = "fossil",  # "electricity" | "fossil" | "both" | "hydrogen"
        eta_electric: float = 0.95,
        eta_fossil: float = 0.90,
        fossil_ng_share: float = 1.0,  # 1.0 => 100% NG, 0.0 => 100% coal
        max_power: float | None = None,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        # emissions (rotary kiln has negligible calcination CO₂)
        ng_co2_factor: float = 0.202,  # [tCO2/MWh_th]
        coal_co2_factor: float = 0.341,
        # on/off
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        name: str = "rotary_kiln",
    ):
        super().__init__()
        self.time_steps = time_steps
        self.max_heat_out = max_heat_out
        self.specific_heat_demand = specific_heat_demand
        self.specific_electricity_aux = specific_electricity_aux

        self.fuel_type = fuel_type.lower()
        assert self.fuel_type in ["electricity", "fossil", "both", "hydrogen"]

        self.eta_electric = eta_electric
        self.eta_fossil = eta_fossil
        self.fossil_ng_share = fossil_ng_share
        self.ng_ef = ng_co2_factor
        self.coal_ef = coal_co2_factor

        self.max_power = (
            max_power
            if max_power is not None
            else max_heat_out / max(1e-6, eta_electric)
        )
        self.rup = max_heat_out if ramp_up is None else ramp_up
        self.rdn = max_heat_out if ramp_down is None else ramp_down

        self.ng_ef = ng_co2_factor
        self.coal_ef = coal_co2_factor

        self.min_up = min_operating_steps
        self.min_dn = min_down_steps
        self.init_on = initial_operational_status
        self.name = name

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        # price streams present?
        if self.fuel_type in ["electricity", "both"] and not hasattr(
            model, "electricity_price"
        ):
            raise ValueError(
                "RotaryKiln requires electricity_price for electric operation."
            )
        if self.fuel_type in ["fossil", "both"] and (
            not hasattr(model, "natural_gas_price") or not hasattr(model, "coal_price")
        ):
            raise ValueError(
                "RotaryKiln requires natural_gas_price and coal_price for fossil operation."
            )
        if self.fuel_type == "hydrogen" and not hasattr(model, "hydrogen_price"):
            raise ValueError(
                "RotaryKiln requires hydrogen_price for hydrogen operation."
            )

        # params
        model_block.max_heat_out = pyo.Param(initialize=self.max_heat_out)
        model_block.specific_heat_demand = pyo.Param(
            initialize=self.specific_heat_demand
        )
        model_block.specific_electricity_aux = pyo.Param(
            initialize=self.specific_electricity_aux
        )
        model_block.ng_co2_factor = pyo.Param(initialize=self.ng_ef)
        model_block.coal_co2_factor = pyo.Param(initialize=self.coal_ef)

        model_block.eta_electric = pyo.Param(initialize=self.eta_electric)
        model_block.eta_fossil = pyo.Param(initialize=self.eta_fossil)
        model_block.fossil_ng_share = pyo.Param(
            initialize=self.fossil_ng_share, within=pyo.UnitInterval
        )

        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.ramp_up = pyo.Param(initialize=self.rup)
        model_block.ramp_down = pyo.Param(initialize=self.rdn)

        model_block.min_operating_steps = pyo.Param(initialize=self.min_up)
        model_block.min_down_steps = pyo.Param(initialize=self.min_dn)
        model_block.initial_operational_status = pyo.Param(initialize=self.init_on)

        # vars
        model_block.heat_out = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_heat_out)
        )
        model_block.clinker_out = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )  # final clinker [t/h]
        model_block.fossil_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_block.natural_gas_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.coal_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.hydrogen_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)
        model_block.co2_energy = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

        # heat balance
        @model_block.Constraint(self.time_steps)
        def heat_balance(b, t):
            if self.fuel_type == "electricity":
                return b.heat_out[t] == b.power_in[t] * b.eta_electric
            elif self.fuel_type == "fossil":
                return (
                    b.heat_out[t] == (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
                )
            elif self.fuel_type == "both":
                return (
                    b.heat_out[t]
                    == b.power_in[t] * b.eta_electric
                    + (b.natural_gas_in[t] + b.coal_in[t]) * b.eta_fossil
                )
            else:  # hydrogen
                return b.heat_out[t] == b.hydrogen_in[t] * b.eta_fossil

        # zero unused inputs
        if self.fuel_type == "electricity":

            @model_block.Constraint(self.time_steps)
            def zero_fossil_h2_when_electric(b, t):
                return b.natural_gas_in[t] + b.coal_in[t] + b.hydrogen_in[t] == 0
        elif self.fuel_type == "fossil":

            @model_block.Constraint(self.time_steps)
            def zero_electric_h2_when_fossil(b, t):
                return b.power_in[t] + b.hydrogen_in[t] == 0

            # fossil split
            @model_block.Constraint(self.time_steps)
            def fossil_split_ng(b, t):
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_split_coal(b, t):
                return b.coal_in[t] == (1 - b.fossil_ng_share) * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_balance(b, t):
                return b.heat_out[t] == b.fossil_in[t] * b.eta_fossil
        elif self.fuel_type == "both":

            @model_block.Constraint(self.time_steps)
            def fossil_split_ng(b, t):
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_split_coal(b, t):
                return b.coal_in[t] == (1 - b.fossil_ng_share) * b.fossil_in[t]
        elif self.fuel_type == "hydrogen":

            @model_block.Constraint(self.time_steps)
            def zero_electric_fossil_when_h2(b, t):
                return b.power_in[t] + b.natural_gas_in[t] + b.coal_in[t] == 0

        # throughput
        @model_block.Constraint(self.time_steps)
        def clinker_from_heat(b, t):
            return b.clinker_out[t] == b.heat_out[t] / b.specific_heat_demand

        # auxiliaries (if electric)
        if self.specific_electricity_aux > 0 and self.fuel_type in [
            "electricity",
            "both",
        ]:

            @model_block.Constraint(self.time_steps)
            def aux_power_min(b, t):
                return b.power_in[t] >= b.clinker_out[t] * (
                    self.specific_electricity_aux / 1000.0
                )

        # ramping
        @model_block.Constraint(self.time_steps)
        def max_heat_if_on(b, t):
            return b.heat_out[t] <= b.max_heat_out * b.operational_status[t]

        @model_block.Constraint(self.time_steps)
        def heat_ramp_up(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.heat_out[t] - b.heat_out[t - 1] <= b.ramp_up

        @model_block.Constraint(self.time_steps)
        def heat_ramp_down(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.heat_out[t - 1] - b.heat_out[t] <= b.ramp_down

        # emissions (only energy CO₂ here)
        @model_block.Constraint(self.time_steps)
        def energy_co2(b, t):
            return b.co2_energy[t] == (
                b.natural_gas_in[t] * b.ng_co2_factor + b.coal_in[t] * b.coal_co2_factor
            )

        # cost
        @model_block.Constraint(self.time_steps)
        def operating_cost_rule(b, t):
            cost = 0
            if self.fuel_type in ["electricity", "both"]:
                cost += b.power_in[t] * model.electricity_price[t]
            if self.fuel_type in ["fossil", "both"]:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                )
            if self.fuel_type == "hydrogen":
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]

            # auxiliary electricity
            if self.specific_electricity_aux > 0:
                aux_MWh = b.clinker_out[t] * (self.specific_electricity_aux / 1000.0)
                cost += aux_MWh * model.electricity_price[t]

            # CO₂ price (only energy)
            cost += b.co2_energy[t] * model.co2_price[t]
            return b.operating_cost[t] == cost

        return model_block


class ElectricVehicle(GenericStorage):
    """
    A class to represent an Electric Vehicle (EV) unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of an EV, including charging and discharging, battery storage limits, availability
    profiles, and ramp rates.

    Inherits from GenericStorage and adds EV-specific functionality, such as availability profiles
    and predefined charging profiles.

    Args:
        max_capacity (float): Maximum capacity of the EV battery.
        min_capacity (float): Minimum capacity of the EV battery.
        max_power_charge (float): Maximum allowable charging power.
        max_power_discharge (float): Maximum allowable discharging power. Defaults to 0 (no discharging allowed).
        availability_profile (pd.Series): A pandas Series indicating the EV's availability, where 1 means available and 0 means unavailable.
        time_steps (list[int]): A list of time steps over which the EV operates.
        efficiency_charge (float, optional): Charging efficiency of the EV. Defaults to 1.0.
        efficiency_discharge (float, optional): Discharging efficiency of the EV. Defaults to 1.0.
        initial_soc (float, optional): Initial state of charge (SOC) of the EV, represented as a fraction of `max_capacity`. Defaults to 1.0.
        ramp_up (float, optional): Maximum allowed increase in charging power per time step. Defaults to None (no ramp constraint).
        ramp_down (float, optional): Maximum allowed decrease in charging power per time step. Defaults to None (no ramp constraint).
        charging_profile (pd.Series | None, optional): A predefined charging profile. If provided, the EV follows this profile instead of optimizing the charge. Defaults to None.
    """

    def __init__(
        self,
        max_capacity: float,
        time_steps: list[int],
        availability_profile: pd.Series,
        max_power_charge: float,
        min_capacity: float = 0.0,
        max_power_discharge: float = 0,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        initial_soc: float = 1.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        charging_profile: pd.Series | None = None,
        storage_loss_rate: float = 0.0,
        **kwargs,
    ):
        # Call the parent class (GenericStorage) __init__ method
        super().__init__(
            max_capacity=max_capacity,
            time_steps=time_steps,
            min_capacity=min_capacity,
            max_power_charge=max_power_charge,
            max_power_discharge=max_power_discharge,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
            initial_soc=initial_soc,
            ramp_up=ramp_up,
            ramp_down=ramp_down,
            storage_loss_rate=storage_loss_rate,
            **kwargs,
        )

        # EV-specific attributes
        self.availability_profile = availability_profile
        self.charging_profile = charging_profile

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds an EV block to the Pyomo model, defining parameters, variables, and constraints.

        Pyomo Components:
            - **Parameters**:
                - `max_capacity`: Maximum battery capacity of the EV.
                - `min_capacity`: Minimum allowable battery capacity.
                - `max_power_charge`: Maximum charging power.
                - `max_power_discharge`: Maximum discharging power.
                - `efficiency_charge`: Charging efficiency.
                - `efficiency_discharge`: Discharging efficiency.

            - **Variables**:
                - `charge[t]`: Charging power input at each time step `t`.
                - `discharge[t]`: Discharging power output at each time step `t`.
                - `soc[t]`: State of charge (SOC) of the EV battery at each time step `t`.

            - **Constraints**:
                - `availability_constraints`: Ensures charging and discharging occur only during available periods.
                - `charging_profile_constraints`: Enforces predefined charging profiles if provided.
                - `soc_constraints`: Keeps SOC between `min_capacity` and `max_capacity`.
                - `ramp_constraints`: Limits ramp-up and ramp-down rates for charging.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the EV component will be added.

        Returns:
            pyo.Block: A Pyomo block representing the EV with variables and constraints.
        """

        # Call the parent class (GenericStorage) add_to_model method
        model_block = super().add_to_model(model, model_block)

        # Apply availability profile constraints if provided
        if self.availability_profile is not None:
            if len(self.availability_profile) != len(self.time_steps):
                raise ValueError(
                    "Length of `availability_profile` must match the number of `time_steps`."
                )

            @model_block.Constraint(self.time_steps)
            def discharge_availability_constraint(b, t):
                availability = self.availability_profile.iat[t]
                return b.discharge[t] <= availability * b.max_power_discharge

            @model_block.Constraint(self.time_steps)
            def charge_availability_constraint(b, t):
                availability = self.availability_profile.iat[t]
                return b.charge[t] <= availability * b.max_power_charge

        # Apply predefined charging profile constraints if provided
        if self.charging_profile is not None:
            if len(self.charging_profile) != len(self.time_steps):
                raise ValueError(
                    "Length of `charging_profile` must match the number of `time_steps`."
                )

            @model_block.Constraint(self.time_steps)
            def charging_profile_constraint(b, t):
                return b.charge[t] == self.charging_profile.iat[t]

        return model_block


class HydrogenBufferStorage(GenericStorage):
    """
    A class to represent a hydrogen storage unit in an energy system model.

    Inherits all the functionality from GenericStorage and can be extended in the future
    with hydrogen-specific constraints or attributes.

    Args:
        Inherits all attributes from the GenericStorage class.
    """

    def __init__(
        self,
        max_capacity: float,
        time_steps: list[int],
        min_capacity: float = 0.0,
        max_power_charge: float | None = None,
        max_power_discharge: float | None = None,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        initial_soc: float = 1.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        storage_loss_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            max_capacity=max_capacity,
            time_steps=time_steps,
            min_capacity=min_capacity,
            max_power_charge=max_power_charge,
            max_power_discharge=max_power_discharge,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
            initial_soc=initial_soc,
            ramp_up=ramp_up,
            ramp_down=ramp_down,
            storage_loss_rate=storage_loss_rate,
            **kwargs,
        )

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the hydrogen storage component. This method can be extended
        to add hydrogen-specific constraints or variables.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the hydrogen storage component will be added.

        Returns:
            pyo.Block: A Pyomo block representing the hydrogen storage with variables and constraints.
        """

        # Call the parent class (GenericStorage) add_to_model method
        model_block = super().add_to_model(model, model_block)

        # add a binary variable to disallow discharging and charging at the same time
        model_block.status = pyo.Var(self.time_steps, within=pyo.Binary)

        # add a constraint that disallows discharging and charging at the same time
        @model_block.Constraint(self.time_steps)
        def max_charge_power_constraint(b, t):
            return b.charge[t] <= b.max_power_charge * b.status[t]

        @model_block.Constraint(self.time_steps)
        def max_discharge_power_constraint(b, t):
            return b.discharge[t] <= b.max_power_discharge * (1 - b.status[t])

        # add further constraints or variables specific to hydrogen storage here

        return model_block


class SeasonalHydrogenStorage(GenericStorage):
    """
    A flexible hydrogen storage class that extends GenericStorage to support short-term and long-term scheduling.

    This class enables control over when the storage can charge or discharge, based on a binary profile.
    It is useful in seasonal or scheduled operations like industrial heating cycles.

    - For 'short-term': behaves like GenericStorage without behavioral restrictions.
    - For 'long-term': follows a storage schedule (0: charge-only, 1: discharge-only).

    Args:
        storage_type (str): Type of storage behavior: 'short-term' or 'long-term'.
        storage_schedule_profile (pd.Series, optional): Binary schedule for discharge availability (only required for 'long-term').
        **kwargs: All other parameters are inherited from GenericStorage.
    """

    def __init__(
        self,
        storage_type: str = "short-term",
        storage_schedule_profile: pd.Series | None = None,
        **kwargs,
    ):
        """
        Initializes the hydrogen storage instance.

        Raises:
            ValueError: If `storage_type` is 'long-term' and no schedule is provided.
            ValueError: If `storage_type` is not one of ['short-term', 'long-term'].
        """
        super().__init__(**kwargs)

        self.storage_type = storage_type.lower()
        self.storage_schedule_profile = storage_schedule_profile

        if self.storage_type == "long-term" and self.storage_schedule_profile is None:
            raise ValueError(
                "storage_schedule_profile is required for 'long-term' storage_type."
            )

        if self.storage_type not in ["short-term", "long-term"]:
            raise ValueError("storage_type must be either 'short-term' or 'long-term'.")

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds seasonal storage block to the Pyomo model, defining constraints based on seasonal operation.
        """
        model_block = super().add_to_model(model, model_block)

        if self.storage_type == "long-term":

            @model_block.Constraint(self.time_steps)
            def availability_charge_constraint(b, t):
                return (
                    b.charge[t]
                    <= (1 - self.storage_schedule_profile.iat[t]) * b.max_power_charge
                )

            @model_block.Constraint(self.time_steps)
            def availability_discharge_constraint(b, t):
                return (
                    b.discharge[t]
                    <= self.storage_schedule_profile.iat[t] * b.max_power_discharge
                )

        return model_block


class DRIStorage(GenericStorage):
    """
    A class to represent a Direct Reduced Iron (DRI) storage unit in an energy system model.

    Inherits all the functionality from GenericStorage and can be extended in the future
    with DRI-specific constraints or attributes.

    Args:
        Inherits all attributes from the GenericStorage class.
    """

    def __init__(
        self,
        max_capacity: float,
        time_steps: list[int],
        min_capacity: float = 0.0,
        max_power_charge: float | None = None,
        max_power_discharge: float | None = None,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        initial_soc: float = 1.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        storage_loss_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            max_capacity=max_capacity,
            time_steps=time_steps,
            min_capacity=min_capacity,
            max_power_charge=max_power_charge,
            max_power_discharge=max_power_discharge,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
            initial_soc=initial_soc,
            ramp_up=ramp_up,
            ramp_down=ramp_down,
            storage_loss_rate=storage_loss_rate,
            **kwargs,
        )

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the DRI storage component. This method can be extended
        in the future with DRI-specific constraints or variables.

        Args:
            model (pyo.ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.
            model_block (pyo.Block): A Pyomo Block object to which the DRI storage component will be added.

        Returns:
            pyo.Block: A Pyomo block representing the DRI storage with variables and constraints.
        """

        # Call the parent class (GenericStorage) add_to_model method
        model_block = super().add_to_model(model, model_block)

        # add a binary variable to disallow discharging and charging at the same time
        model_block.status = pyo.Var(self.time_steps, within=pyo.Binary)

        # add a constraint that disallows discharging and charging at the same time
        @model_block.Constraint(self.time_steps)
        def max_charge_power_constraint(b, t):
            return b.charge[t] <= b.max_power_charge * b.status[t]

        @model_block.Constraint(self.time_steps)
        def max_discharge_power_constraint(b, t):
            return b.discharge[t] <= b.max_power_discharge * (1 - b.status[t])

        # add further constraints or variables specific to DRI storage here

        return model_block


# Mapping of component type identifiers to their respective classes
demand_side_technologies: dict = {
    "electrolyser": Electrolyser,
    "hydrogen_buffer_storage": HydrogenBufferStorage,
    "hydrogen_seasonal_storage": SeasonalHydrogenStorage,
    "thermal_storage": ThermalStorage,
    "dri_plant": DRIPlant,
    "dri_storage": DRIStorage,
    "eaf": ElectricArcFurnace,
    "heat_pump": HeatPump,
    "heat_resistor": HeatResistor,
    "boiler": Boiler,
    "electric_vehicle": ElectricVehicle,
    "generic_storage": GenericStorage,
    "pv_plant": PVPlant,
    "cement_mill": GrindingMill,
    "preheater": Preheater,
    "calciner": Calciner,
    "kiln": Kiln,
}


def add_ramping_constraints(model_block, time_steps):
    # Ramp-up constraint
    @model_block.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        if t == time_steps.at(1):
            return b.power_in[t] <= b.ramp_up
        return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_block.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        if t == time_steps.at(1):
            return b.power_in[t] <= b.ramp_down
        return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

    return model_block


def add_min_up_down_time_constraints(model_block, time_steps):
    model_block.operational_status = pyo.Var(time_steps, within=pyo.Binary)

    # Power constraints based on operational status
    @model_block.Constraint(time_steps)
    def min_power_constraint(b, t):
        return b.power_in[t] >= b.min_power * b.operational_status[t]

    @model_block.Constraint(time_steps)
    def max_power_constraint(b, t):
        return b.power_in[t] <= b.max_power * b.operational_status[t]

    if model_block.min_operating_steps > 0 or model_block.min_down_steps > 0:
        model_block.start_up = pyo.Var(time_steps, within=pyo.Binary)
        model_block.shut_down = pyo.Var(time_steps, within=pyo.Binary)

        # State transition constraints
        @model_block.Constraint(time_steps)
        def state_transition_rule(b, t):
            if t == time_steps.at(1):
                return (
                    b.operational_status[t] - model_block.initial_operational_status
                    == b.start_up[t] - b.shut_down[t]
                )
            else:
                return (
                    b.operational_status[t] - b.operational_status[t - 1]
                    == b.start_up[t] - b.shut_down[t]
                )

        # Prevent simultaneous startup and shutdown
        @model_block.Constraint(time_steps)
        def prevent_simultaneous_startup_shutdown(b, t):
            return b.start_up[t] + b.shut_down[t] <= 1

        # Minimum operating time constraints
        if model_block.min_operating_steps > 0:
            # Start-up definition
            @model_block.Constraint(time_steps)
            def start_up_def_rule(b, t):
                if t == time_steps.at(1):
                    return (
                        b.start_up[t]
                        >= b.operational_status[t]
                        - model_block.initial_operational_status
                    )
                else:
                    return (
                        b.start_up[t]
                        >= b.operational_status[t] - b.operational_status[t - 1]
                    )

            @model_block.Constraint(time_steps)
            def min_operating_time_constraint(b, t):
                if t < model_block.min_operating_steps:
                    return pyo.Constraint.Skip
                return (
                    sum(
                        b.start_up[i]
                        for i in range(t - model_block.min_operating_steps + 1, t + 1)
                    )
                    <= b.operational_status[t]
                )

        # Minimum downtime constraints
        if model_block.min_down_steps > 0:
            # Shut-down definition
            @model_block.Constraint(time_steps)
            def shut_down_def_rule(b, t):
                if t == time_steps.at(1):
                    return (
                        b.shut_down[t]
                        >= model_block.initial_operational_status
                        - b.operational_status[t]
                    )
                else:
                    return (
                        b.shut_down[t]
                        >= b.operational_status[t - 1] - b.operational_status[t]
                    )

            @model_block.Constraint(time_steps)
            def min_downtime_constraint(b, t):
                if t < model_block.min_down_steps:
                    return pyo.Constraint.Skip
                return (
                    sum(
                        b.shut_down[i]
                        for i in range(t - model_block.min_down_steps + 1, t + 1)
                    )
                    <= 1 - b.operational_status[t]
                )

    return model_block
