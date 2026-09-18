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
            ramped=model_block.power_in,
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

        # Ramp-up constraint and ramp-down constraints. The first step has no
        # predecessor to decrease from, so ramp_down is skipped there rather than
        # also capping it (ramp_up already limits the rise from a standing start).
        if self.fuel_type == "natural_gas":

            @model_block.Constraint(self.time_steps)
            def ramp_up_constraint(b, t):
                if t == self.time_steps.at(1):
                    return b.natural_gas_in[t] <= b.ramp_up
                return b.natural_gas_in[t] - b.natural_gas_in[t - 1] <= b.ramp_up

            @model_block.Constraint(self.time_steps)
            def ramp_down_constraint(b, t):
                if t == self.time_steps.at(1):
                    return pyo.Constraint.Skip
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
                    return pyo.Constraint.Skip
                return b.hydrogen_gas_in[t - 1] - b.hydrogen_gas_in[t] <= b.ramp_down

        elif self.fuel_type == "electricity":
            add_ramping_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
                ramped=model_block.power_in,
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
        capacity (float): Energy storage capacity of the storage unit.
        min_soc (float, optional): Minimum allowable state of charge (SOC). Defaults to 0.0.
        max_soc (float, optional): Maximum allowable state of charge (SOC). Defaults to 1.0.
        max_power_charge (float, optional): Maximum charging power of the storage unit. Defaults to `capacity` if not provided.
        max_power_discharge (float, optional): Maximum discharging power of the storage unit. Defaults to `capacity` if not provided.
        efficiency_charge (float, optional): Efficiency of the charging process. Defaults to 1.0.
        efficiency_discharge (float, optional): Efficiency of the discharging process. Defaults to 1.0.
        initial_soc (float, optional): Initial state of charge as a fraction of `capacity`. Defaults to 1.0.
        ramp_up (float, optional): Maximum allowed increase in charging/discharging power per time step. Defaults to None (no ramp constraint).
        ramp_down (float, optional): Maximum allowed decrease in charging/discharging power per time step. Defaults to None (no ramp constraint).
        storage_loss_rate (float, optional): Fraction of energy lost per time step due to storage inefficiencies. Defaults to 0.0.
    """

    def __init__(
        self,
        capacity: float,
        time_steps: list[int],
        min_soc: float = 0.0,
        max_soc: float = 1.0,
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
            logger.warning(
                "Initial SOC is greater than 1.0 but SOC must be between 0 and 1."
            )
            raise ValueError("Initial SOC must be between 0 and 1.")

        self.capacity = capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.time_steps = time_steps
        self.max_power_charge = (
            capacity if max_power_charge is None else max_power_charge
        )
        self.max_power_discharge = (
            capacity if max_power_discharge is None else max_power_discharge
        )
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.initial_soc = initial_soc
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
                - `capacity`: Capacity of the storage unit.
                - `min_soc`: Minimum state of charge (SOC, between 0 and 1).
                - `max_soc`: Maximum state of charge (SOC, between 0 and 1).
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
        model_block.capacity = pyo.Param(initialize=self.capacity)
        model_block.min_soc = pyo.Param(initialize=self.min_soc)
        model_block.max_soc = pyo.Param(initialize=self.max_soc)
        model_block.max_power_charge = pyo.Param(initialize=self.max_power_charge)
        model_block.max_power_discharge = pyo.Param(initialize=self.max_power_discharge)
        model_block.efficiency_charge = pyo.Param(initialize=self.efficiency_charge)
        model_block.efficiency_discharge = pyo.Param(
            initialize=self.efficiency_discharge
        )
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.initial_soc = pyo.Param(initialize=self.initial_soc)
        model_block.storage_loss_rate = pyo.Param(initialize=self.storage_loss_rate)

        # Define variables
        model_block.soc = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(model_block.min_soc, model_block.max_soc),
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

        # Define SOC dynamics with energy loss and efficiency
        @model_block.Constraint(self.time_steps)
        def soc_balance_rule(b, t):
            if t == self.time_steps.at(1):
                prev_soc = b.initial_soc
            else:
                prev_soc = b.soc[t - 1]
            return b.soc[t] == (
                prev_soc
                + (
                    b.efficiency_charge * b.charge[t]
                    - (1 / b.efficiency_discharge) * b.discharge[t]
                    - b.storage_loss_rate * prev_soc * b.capacity
                )
                / b.capacity
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

        # Apply ramp-down constraints if ramp_down is specified. The first step has no
        # predecessor to decrease from, so it is left uncapped here rather than also
        # bounding it by ramp_down (ramp_up above already caps how fast it can rise
        # from a standing start).
        @model_block.Constraint(self.time_steps)
        def charge_ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.charge[t - 1] - b.charge[t] <= b.ramp_down

        @model_block.Constraint(self.time_steps)
        def discharge_ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.discharge[t - 1] - b.discharge[t] <= b.ramp_down

        return model_block


class ThermalStorage(GenericStorage):
    """
    A flexible thermal storage class that extends GenericStorage to support short-term and long-term scheduling.

    This class enables control over when the storage can charge or discharge, based on a binary profile.
    It is useful in seasonal or scheduled operations like industrial heating cycles.

    - For 'short-term': behaves like GenericStorage without behavioral restrictions.
    - For 'long-term': follows a storage schedule (0: charge-only, 1: discharge-only).
    - For 'short-term_with_generator': an electric heater charges the storage, making it
      a power-to-heat unit (E-TES). ``charge`` is then produced from ``power_in`` at
      ``eta_electric`` and priced at the electricity price.

    Args:
        storage_type (str): Type of storage behavior: 'short-term', 'long-term' or 'short-term_with_generator'.
        storage_schedule_profile (pd.Series, optional): Binary schedule for discharge availability (only required for 'long-term').
        eta_electric (float, optional): Efficiency of the electric heater (only used with a generator). Defaults to 0.97.
        max_power (float, optional): Cap on the electric heater power (MW). Defaults to ``max_power_charge / eta_electric``.
        **kwargs: All other parameters are inherited from GenericStorage.
    """

    supported_storage_types = (
        "short-term",
        "long-term",
        "short-term_with_generator",
    )

    def __init__(
        self,
        storage_type: str = "short-term",
        storage_schedule_profile: pd.Series | None = None,
        eta_electric: float = 0.97,
        max_power: float | None = None,
        **kwargs,
    ):
        """
        Initializes the thermal storage instance.

        Raises:
            ValueError: If `storage_type` is 'long-term' and no schedule is provided.
            ValueError: If `storage_type` is not a supported storage type.
        """
        super().__init__(**kwargs)

        self.storage_type = storage_type.lower()
        self.storage_schedule_profile = storage_schedule_profile

        if self.storage_type == "long-term" and self.storage_schedule_profile is None:
            raise ValueError(
                "storage_schedule_profile is required for 'long-term' storage_type."
            )

        if self.storage_type not in self.supported_storage_types:
            raise ValueError(
                "storage_type must be one of: "
                f"{', '.join(self.supported_storage_types)}."
            )

        self.eta_electric = eta_electric
        if self.storage_type == "short-term_with_generator":
            self.max_power = (
                self.max_power_charge / max(eta_electric, 1e-6)
                if max_power is None
                else max_power
            )
        else:
            # No electric heater: the power port exists but is fixed to zero
            self.max_power = 0.0

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds thermal storage constraints to the model based on storage type.

        - If 'long-term', restrict charge/discharge based on the binary schedule profile.
        - If 'short-term_with_generator', charge the storage with an electric heater.
        - Otherwise, behaves identically to GenericStorage.

        Args:
            model (pyo.ConcreteModel): The Pyomo optimization model.
            model_block (pyo.Block): The block to which this storage is added.

        Returns:
            pyo.Block: Updated model block with thermal storage constraints.
        """
        model_block = super().add_to_model(model, model_block)

        # The power port is always defined so plant-level load aggregation and
        # reporting can treat every thermal storage alike; without a generator it is
        # fixed to zero.
        model_block.eta_electric = pyo.Param(initialize=self.eta_electric)
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

        if self.storage_type == "short-term_with_generator":
            if not hasattr(model, "electricity_price"):
                raise ValueError(
                    "ThermalStorage with an electric heater requires an electricity "
                    "price profile in the model."
                )

            @model_block.Constraint(self.time_steps)
            def electric_heater_charge(b, t):
                return b.charge[t] == b.power_in[t] * b.eta_electric

            @model_block.Constraint(self.time_steps)
            def storage_operating_cost_constraint(b, t):
                return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        else:

            @model_block.Constraint(self.time_steps)
            def no_electric_heater(b, t):
                return b.power_in[t] == 0

            @model_block.Constraint(self.time_steps)
            def storage_operating_cost_constraint(b, t):
                return b.operating_cost[t] == 0

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
            return b.operating_cost[t] == b.power[t] * model.electricity_price[t]

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
        model_block.operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

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
            ramped=model_block.power_in,
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
        model_block.operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

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
            ramped=model_block.power_in,
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
        model_block.operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
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
            ramped=model_block.power_in,
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


class ElectricVehicle(GenericStorage):
    """
    Electric vehicle model for multi-asset applications.

    Extends GenericStorage with:
    - vehicle availability
    - optional predefined charging profile
    - transport energy use via `external_trip_distance`
    - unidirectional / bidirectional operation
    - binary non-simultaneity of charging and discharging

    Parameters
    ----------
    capacity : float
        Battery energy capacity [same energy unit as charge/discharge integration].
    time_steps : list[int]
        Optimisation time steps.
    availability_profile : pd.Series | None
        1 when vehicle is connected/available at the charging location, 0 otherwise.
    max_power_charge : float
        Maximum charging power.
    min_soc : float, default 0.0
        Minimum SOC in fraction of capacity.
    max_soc : float, default 1.0
        Maximum SOC in fraction of capacity.
    max_power_discharge : float, default 0.0
        Maximum discharging power.
    efficiency_charge : float, default 1.0
        Charging efficiency.
    efficiency_discharge : float, default 1.0
        Discharging efficiency.
    initial_soc : float, default 1.0
        Initial SOC in fraction of capacity.
    ramp_up : float | None
        Max increase in charge/discharge power between consecutive steps.
    ramp_down : float | None
        Max decrease in charge/discharge power between consecutive steps.
    charging_profile : pd.Series | None
        Optional fixed charging profile.
    storage_loss_rate : float, default 0.0
        Fractional storage loss per time step.
    mileage : float | None
        Specific electricity consumption for driving.
        Interpreted here as energy per unit of external_trip_distance.
        Example: if external_trip_distance is km, mileage should be MWh/km.
    power_flow_directionality : str, default "unidirectional"
        Either "unidirectional" or "bidirectional".
    """

    def __init__(
        self,
        capacity: float,
        time_steps: list[int],
        max_power_charge: float,
        availability_profile: pd.Series | None = None,
        min_soc: float = 0.0,
        max_soc: float = 1.0,
        max_power_discharge: float = 0.0,
        efficiency_charge: float = 1.0,
        efficiency_discharge: float = 1.0,
        initial_soc: float = 1.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        charging_profile: pd.Series | None = None,
        storage_loss_rate: float = 0.0,
        mileage: float | None = None,
        power_flow_directionality: str = "unidirectional",
        **kwargs,
    ):
        super().__init__(
            capacity=capacity,
            time_steps=time_steps,
            min_soc=min_soc,
            max_soc=max_soc,
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

        self.availability_profile = availability_profile
        self.charging_profile = charging_profile
        self.mileage = mileage
        self.power_flow_directionality = power_flow_directionality.lower()

        if self.power_flow_directionality not in ["unidirectional", "bidirectional"]:
            raise ValueError(
                "power_flow_directionality must be either 'unidirectional' or 'bidirectional'."
            )

        if availability_profile is not None and charging_profile is not None:
            raise ValueError(
                "Provide either `availability_profile` or `charging_profile`, not both."
            )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        model_block: pyo.Block,
        external_trip_distance: pyo.Param | None = None,
        external_trip_energy_consumption: pyo.Param | None = None,
    ) -> pyo.Block:
        """
        Add EV to Pyomo model.

        Parameters
        ----------
        model : pyo.ConcreteModel
        model_block : pyo.Block
        external_trip_distance : pyo.Param | None
            Exogenous trip distance per time step (Mode 1: distance-based).
            Used to calculate usage as: usage[t] = (1 - availability[t]) * external_trip_distance[t] * mileage
            Requires `mileage` to be provided.
        external_trip_energy_consumption : pyo.Param | None
            Exogenous trip energy consumption per time step (Mode 2: direct energy).
            Used to calculate usage as: usage[t] = (1 - availability[t]) * external_trip_energy_consumption[t]
            If both trip_distance and trip_energy_consumption are provided, trip_energy_consumption takes priority.
        """
        # Start from generic storage structure
        model_block = super().add_to_model(model, model_block)

        # Extra parameters
        model_block.mileage = pyo.Param(
            initialize=0.0 if self.mileage is None else self.mileage,
            mutable=False,
        )

        # Extra variables
        model_block.operating_cost = pyo.Var(self.time_steps, within=pyo.Reals)
        model_block.usage = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.status = pyo.Var(
            self.time_steps,
            within=pyo.Binary,
            doc="1 => charging mode, 0 => discharging mode",
        )

        # Replace the GenericStorage SOC equation with EV-specific one
        if hasattr(model_block, "soc_balance_rule"):
            model_block.del_component(model_block.soc_balance_rule)

        @model_block.Constraint(self.time_steps)
        def soc_balance_rule(b, t):
            if t == self.time_steps.at(1):
                prev_soc = b.initial_soc
            else:
                prev_soc = b.soc[t - 1]

            return b.soc[t] == (
                prev_soc
                + (
                    b.efficiency_charge * b.charge[t]
                    - (1 / b.efficiency_discharge) * b.discharge[t]
                    - b.storage_loss_rate * prev_soc * b.capacity
                    - b.usage[t]
                )
                / b.capacity
            )

        # No simultaneous charging and discharging
        @model_block.Constraint(self.time_steps)
        def charge_mode_constraint(b, t):
            return b.charge[t] <= b.max_power_charge * b.status[t]

        if self.power_flow_directionality == "bidirectional":

            @model_block.Constraint(self.time_steps)
            def discharge_mode_constraint(b, t):
                return b.discharge[t] <= b.max_power_discharge * (1 - b.status[t])

        else:

            @model_block.Constraint(self.time_steps)
            def unidirectional_discharge_constraint(b, t):
                return b.discharge[t] == 0

        # Availability logic
        if self.availability_profile is not None:
            if len(self.availability_profile) != len(self.time_steps):
                raise ValueError(
                    "Length of `availability_profile` must match number of `time_steps`."
                )

            @model_block.Constraint(self.time_steps)
            def charge_availability_constraint(b, t):
                availability = self.availability_profile.iat[t]
                return b.charge[t] <= availability * b.max_power_charge

            @model_block.Constraint(self.time_steps)
            def discharge_availability_constraint(b, t):
                availability = self.availability_profile.iat[t]
                if self.power_flow_directionality == "bidirectional":
                    return b.discharge[t] <= availability * b.max_power_discharge
                return b.discharge[t] == 0

            # Driving usage only when unavailable
            # Priority: trip_energy_consumption (Mode 2) > trip_distance (Mode 1)
            if external_trip_energy_consumption is not None:
                # Mode 2: Direct energy consumption for trip
                @model_block.Constraint(self.time_steps)
                def usage_constraint(b, t):
                    availability = self.availability_profile.iat[t]
                    return (
                        b.usage[t]
                        == (1 - availability) * external_trip_energy_consumption[t]
                    )

            elif external_trip_distance is not None:
                # Mode 1: Energy consumption from distance * mileage
                if self.mileage is None:
                    raise ValueError(
                        "`mileage` must be provided when `external_trip_distance` is used."
                    )

                @model_block.Constraint(self.time_steps)
                def usage_constraint(b, t):
                    availability = self.availability_profile.iat[t]
                    return b.usage[t] == (1 - availability) * (
                        external_trip_distance[t] * b.mileage
                    )

            else:

                @model_block.Constraint(self.time_steps)
                def usage_constraint(b, t):
                    return b.usage[t] == 0

        else:
            # No availability profile: no exogenous mobility depletion
            @model_block.Constraint(self.time_steps)
            def usage_constraint(b, t):
                return b.usage[t] == 0

        # Optional fixed charging profile
        if self.charging_profile is not None:
            if len(self.charging_profile) != len(self.time_steps):
                raise ValueError(
                    "Length of `charging_profile` must match number of `time_steps`."
                )

            @model_block.Constraint(self.time_steps)
            def charging_profile_constraint(b, t):
                return b.charge[t] == self.charging_profile.iat[t]

        # Operating costs: charging cost minus discharge revenue
        # If you want pure charging cost only, remove the second term.
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == (
                b.charge[t] * model.electricity_price[t]
                - b.discharge[t] * model.electricity_price[t]
            )

        return model_block


class ChargingStation:
    """
    Charging station model for EV fleets / depots.

    Supports:
    - unidirectional or bidirectional power flow
    - ramp constraints
    - optional availability profile
    - non-simultaneous charging/discharging in bidirectional mode

    Parameters
    ----------
    time_steps : list[int]
        Optimisation time steps.
    max_power : float
        Maximum charging/discharging power.
    min_power : float, default 0.0
        Minimum non-zero operating power, if used.
    ramp_up : float | None
        Max increase in power between steps.
    ramp_down : float | None
        Max decrease in power between steps.
    availability_profile : pd.Series | None
        Optional availability profile.
    power_flow_directionality : str, default "unidirectional"
        Either "unidirectional" or "bidirectional".
    """

    def __init__(
        self,
        time_steps: list[int],
        max_power: float,
        min_power: float = 0.0,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        availability_profile: pd.Series | None = None,
        power_flow_directionality: str = "unidirectional",
        **kwargs,
    ):
        self.time_steps = time_steps
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.availability_profile = availability_profile
        self.power_flow_directionality = power_flow_directionality.lower()
        self.kwargs = kwargs

        if self.power_flow_directionality not in ["unidirectional", "bidirectional"]:
            raise ValueError(
                "power_flow_directionality must be either 'unidirectional' or 'bidirectional'."
            )

    def add_to_model(
        self,
        model: pyo.ConcreteModel,
        model_block: pyo.Block,
    ) -> pyo.Block:
        # Parameters
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)

        # Variables
        # In this convention:
        # - charge: power drawn from grid to EVs
        # - discharge: power fed from EVs/depot back outward
        model_block.charge = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.discharge = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )

        if self.power_flow_directionality == "bidirectional":
            model_block.charging_mode = pyo.Var(self.time_steps, within=pyo.Binary)

            @model_block.Constraint(self.time_steps)
            def prevent_simultaneous_charge_discharge(b, t):
                return b.charge[t] <= b.max_power * b.charging_mode[t]

            @model_block.Constraint(self.time_steps)
            def discharge_only_when_not_charging_mode(b, t):
                return b.discharge[t] <= b.max_power * (1 - b.charging_mode[t])

        else:

            @model_block.Constraint(self.time_steps)
            def unidirectional_no_discharge(b, t):
                return b.discharge[t] == 0

        # Availability
        if self.availability_profile is not None:
            if len(self.availability_profile) != len(self.time_steps):
                raise ValueError(
                    "Length of `availability_profile` must match number of `time_steps`."
                )

            @model_block.Constraint(self.time_steps)
            def charge_availability_constraint(b, t):
                availability = self.availability_profile.iat[t]
                return b.charge[t] <= availability * b.max_power

            @model_block.Constraint(self.time_steps)
            def discharge_availability_constraint(b, t):
                availability = self.availability_profile.iat[t]
                return b.discharge[t] <= availability * b.max_power

        # Minimum power only if active
        # This is optional and only enforced in bidirectional mode where an activity binary exists.
        if self.min_power > 0 and self.power_flow_directionality == "bidirectional":

            @model_block.Constraint(self.time_steps)
            def min_charge_if_active(b, t):
                return b.charge[t] >= b.min_power * b.charging_mode[t]

            @model_block.Constraint(self.time_steps)
            def min_discharge_if_active(b, t):
                return b.discharge[t] >= b.min_power * (1 - b.charging_mode[t])

        # Ramping: charge
        @model_block.Constraint(self.time_steps)
        def charge_ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return b.charge[t] <= b.ramp_up
            return b.charge[t] - b.charge[t - 1] <= b.ramp_up

        # The first step has no predecessor to decrease from, so ramp_down is skipped
        # there rather than also capping it (ramp_up above already limits how fast it
        # can rise from a standing start).
        @model_block.Constraint(self.time_steps)
        def charge_ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.charge[t - 1] - b.charge[t] <= b.ramp_down

        # Ramping: discharge
        @model_block.Constraint(self.time_steps)
        def discharge_ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return b.discharge[t] <= b.ramp_up
            return b.discharge[t] - b.discharge[t - 1] <= b.ramp_up

        @model_block.Constraint(self.time_steps)
        def discharge_ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.discharge[t - 1] - b.discharge[t] <= b.ramp_down

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
        capacity: float,
        time_steps: list[int],
        min_soc: float = 0.0,
        max_soc: float = 1.0,
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
            capacity=capacity,
            time_steps=time_steps,
            min_soc=min_soc,
            max_soc=max_soc,
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
        Adds hydrogen storage constraints to the model based on storage type.

        - If 'long-term', restrict charge/discharge based on the binary schedule profile.
        - Otherwise, behaves identically to GenericStorage.

        Args:
            model (pyo.ConcreteModel): The Pyomo optimization model.
            model_block (pyo.Block): The block to which this storage is added.

        Returns:
            pyo.Block: Updated model block with hydrogen storage constraints.
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
        capacity: float,
        time_steps: list[int],
        min_soc: float = 0.0,
        max_soc: float = 1.0,
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
            capacity=capacity,
            time_steps=time_steps,
            min_soc=min_soc,
            max_soc=max_soc,
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


class GrindingMill:
    """
    A class to represent a generic grinding mill in a cement plant model.

    The same class covers raw material milling (limestone and additives to raw meal)
    and cement grinding (clinker and additives to cement), since both are purely
    electric grinding steps that differ only in their parameterisation.

    Args:
        max_power (float): Maximum allowable power input to the mill (MW).
        specific_electricity_consumption (float): Electricity consumption per tonne of
            material output (MWh/t).
        time_steps (list[int]): A list of time steps over which the mill operates.
        min_power (float, optional): Minimum allowable power input to the mill. Defaults to 0.0.
        efficiency (float, optional): Mass ratio of material output to material input.
            Defaults to 1.0 (no mass losses).
        ramp_up (float, optional): Maximum allowed increase in power input per time step.
            Defaults to `max_power`.
        ramp_down (float, optional): Maximum allowed decrease in power input per time step.
            Defaults to `max_power`.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the
            mill must operate once started. Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive time steps the mill
            must remain off after shutdown. Defaults to 0.
        initial_operational_status (int, optional): Operational status before the first
            time step (0 for off, 1 for on). Defaults to 1.
    """

    def __init__(
        self,
        max_power: float,
        specific_electricity_consumption: float,
        time_steps: list[int],
        min_power: float = 0.0,
        efficiency: float = 1.0,
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
        self.efficiency = efficiency
        self.specific_electricity_consumption = specific_electricity_consumption
        self.time_steps = time_steps
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = int(min_operating_steps)
        self.min_down_steps = int(min_down_steps)
        self.initial_operational_status = initial_operational_status
        self.kwargs = kwargs

        if specific_electricity_consumption <= 0:
            raise ValueError(
                "GrindingMill requires a positive specific_electricity_consumption."
            )
        if efficiency <= 0:
            raise ValueError("GrindingMill requires a positive efficiency.")

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds a grinding mill block to the Pyomo model, defining parameters, variables and constraints.

        Pyomo Components:
            - **Parameters**: `max_power`, `min_power`, `efficiency`,
              `specific_electricity_consumption`, `ramp_up`, `ramp_down`,
              `min_operating_steps`, `min_down_steps`, `initial_operational_status`.
            - **Variables**: `power_in[t]`, `material_input[t]`, `material_output[t]`,
              `operating_cost[t]`, and optionally `operational_status[t]` (binary).
            - **Constraints**: material input/output relation, electricity demand of the
              grinding step, operating cost, ramping, and minimum up/down times.

        Args:
            model (pyo.ConcreteModel): The optimisation model the block belongs to.
            model_block (pyo.Block): The block the mill is added to.

        Returns:
            pyo.Block: The block representing the grinding mill.
        """

        # Define parameters
        model_block.max_power = pyo.Param(initialize=self.max_power)
        model_block.min_power = pyo.Param(initialize=self.min_power)
        model_block.efficiency = pyo.Param(initialize=self.efficiency)
        model_block.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.min_operating_steps = pyo.Param(
            initialize=self.min_operating_steps, within=pyo.NonNegativeIntegers
        )
        model_block.min_down_steps = pyo.Param(
            initialize=self.min_down_steps, within=pyo.NonNegativeIntegers
        )
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

        # Define variables
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
        )
        model_block.material_input = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.material_output = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

        # Mass balance over the mill
        @model_block.Constraint(self.time_steps)
        def material_input_output_relation(b, t):
            return b.material_input[t] == b.material_output[t] / b.efficiency

        # Electricity demand of the grinding step
        @model_block.Constraint(self.time_steps)
        def material_flow_constraint(b, t):
            return (
                b.power_in[t]
                == b.material_output[t] * b.specific_electricity_consumption
            )

        # Operating costs
        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint_rule(b, t):
            return b.operating_cost[t] == b.power_in[t] * model.electricity_price[t]

        # Ramp-up and ramp-down constraints
        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
            ramped=model_block.power_in,
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


class ThermalProcessStage:
    """
    Base class for the fuel-switchable thermal stages of a cement kiln line.

    A stage converts an energy carrier into process heat, and its heat throughput sets
    the material throughput of the stage. Four firing modes are supported:

    - ``"electricity"``: ``heat_out = eta_electric × power_in``
    - ``"fossil"``: ``heat_out = eta_fossil × (natural_gas_in + coal_in)``, split by ``fossil_ng_share``
    - ``"both"``: the sum of the electric and the fossil contribution
    - ``"hydrogen"``: ``heat_out = eta_fossil × hydrogen_in`` (``eta_fossil`` acts as the burner efficiency)

    Auxiliary electricity (fans, drives, transport) is modelled separately from the
    electric heating path in ``aux_power``, so it is part of the plant load in every
    firing mode - not only when the stage is heated electrically.

    Args:
        max_heat_out (float): Maximum thermal output of the stage (MW_th).
        specific_heat_demand (float): Thermal energy per tonne of throughput (MWh_th/t).
        time_steps (list[int]): A list of time steps over which the stage operates.
        min_heat_out (float, optional): Minimum thermal output while the stage is on (MW_th). Defaults to 0.0.
        specific_electricity_aux (float, optional): Auxiliary electricity per tonne of throughput (MWh_el/t). Defaults to 0.0.
        fuel_type (str, optional): Firing mode, see above. Defaults to ``"electricity"``.
        eta_electric (float, optional): Efficiency of the electric heating path. Defaults to 0.95.
        eta_fossil (float, optional): Efficiency of the burner (fossil or hydrogen). Defaults to 0.90.
        fossil_ng_share (float, optional): Share of natural gas in the fossil fuel mix (1.0 = only gas, 0.0 = only coal). Defaults to 1.0.
        max_power (float, optional): Cap on the electric heating power (MW_el). Defaults to ``max_heat_out / eta_electric``.
        ramp_up (float, optional): Maximum increase in thermal output per time step. Defaults to `max_heat_out`.
        ramp_down (float, optional): Maximum decrease in thermal output per time step. Defaults to `max_heat_out`.
        ng_co2_factor (float, optional): Emissions of natural gas (t CO2/MWh_th). Defaults to 0.202.
        coal_co2_factor (float, optional): Emissions of coal (t CO2/MWh_th). Defaults to 0.341.
        min_operating_steps (int, optional): Minimum number of consecutive operating steps. Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive down steps. Defaults to 0.
        initial_operational_status (int, optional): Status before the first time step (0 for off, 1 for on). Defaults to 1.
        availability_profile (pd.Series | FastSeries | None, optional): Per-time-step
            availability of the stage (1 available, 0 unavailable). Defaults to None.
    """

    supported_fuel_types = ("electricity", "fossil", "both", "hydrogen")
    #: Name of the block variable holding the material throughput of the stage.
    throughput_name = "throughput"

    def __init__(
        self,
        max_heat_out: float,
        specific_heat_demand: float,
        time_steps: list[int],
        min_heat_out: float = 0.0,
        specific_electricity_aux: float = 0.0,
        fuel_type: str = "electricity",
        eta_electric: float = 0.95,
        eta_fossil: float = 0.90,
        fossil_ng_share: float = 1.0,
        max_power: float | None = None,
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        ng_co2_factor: float = 0.202,
        coal_co2_factor: float = 0.341,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        initial_operational_status: int = 1,
        availability_profile=None,
        **kwargs,
    ):
        super().__init__()

        self.fuel_type = str(fuel_type).lower()
        if self.fuel_type not in self.supported_fuel_types:
            raise ValueError(
                f"Unsupported fuel_type '{fuel_type}' for {type(self).__name__}. "
                f"Choose one of {', '.join(self.supported_fuel_types)}."
            )
        if specific_heat_demand <= 0:
            raise ValueError(
                f"{type(self).__name__} requires a positive specific_heat_demand."
            )
        if not 0 <= fossil_ng_share <= 1:
            raise ValueError("fossil_ng_share must be between 0 and 1.")

        self.time_steps = time_steps
        self.max_heat_out = max_heat_out
        self.min_heat_out = min_heat_out
        self.specific_heat_demand = specific_heat_demand
        self.specific_electricity_aux = specific_electricity_aux

        self.eta_electric = eta_electric
        self.eta_fossil = eta_fossil
        self.fossil_ng_share = fossil_ng_share

        self.max_power = (
            max_heat_out / max(eta_electric, 1e-6) if max_power is None else max_power
        )
        self.ramp_up = max_heat_out if ramp_up is None else ramp_up
        self.ramp_down = max_heat_out if ramp_down is None else ramp_down

        self.ng_co2_factor = ng_co2_factor
        self.coal_co2_factor = coal_co2_factor

        self.min_operating_steps = int(min_operating_steps)
        self.min_down_steps = int(min_down_steps)
        self.initial_operational_status = initial_operational_status
        self.availability_profile = sanitize_availability_profile(
            availability_profile, type(self).__name__
        )
        self.kwargs = kwargs

    @property
    def uses_electricity(self) -> bool:
        return self.fuel_type in ("electricity", "both")

    @property
    def uses_fossil(self) -> bool:
        return self.fuel_type in ("fossil", "both")

    def _check_prices(self, model: pyo.ConcreteModel) -> None:
        """Verify the plant model provides every price series this stage needs."""
        required = [
            "electricity_price",
            "co2_price",
        ]  # auxiliaries are electric; emission costs require co2_price
        if self.uses_fossil:
            required += ["natural_gas_price", "coal_price"]
        if self.fuel_type == "hydrogen":
            required += ["hydrogen_price"]
        for attr in required:
            if not hasattr(model, attr):
                raise ValueError(
                    f"{type(self).__name__} with fuel_type '{self.fuel_type}' requires "
                    f"a '{attr}' profile in the model."
                )

    def _add_common_parameters(self, model_block: pyo.Block) -> None:
        model_block.max_heat_out = pyo.Param(initialize=self.max_heat_out)
        model_block.min_heat_out = pyo.Param(initialize=self.min_heat_out)
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
        model_block.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_block.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_block.ng_co2_factor = pyo.Param(initialize=self.ng_co2_factor)
        model_block.coal_co2_factor = pyo.Param(initialize=self.coal_co2_factor)
        model_block.min_operating_steps = pyo.Param(
            initialize=self.min_operating_steps, within=pyo.NonNegativeIntegers
        )
        model_block.min_down_steps = pyo.Param(
            initialize=self.min_down_steps, within=pyo.NonNegativeIntegers
        )
        model_block.initial_operational_status = pyo.Param(
            initialize=self.initial_operational_status
        )

    def _add_common_variables(self, model_block: pyo.Block) -> None:
        model_block.heat_out = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_heat_out),
            doc="Useful process heat generated by the stage",
        )
        model_block.power_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, model_block.max_power),
            doc="Electric heating power",
        )
        model_block.aux_power = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            doc="Auxiliary electricity of fans, drives and transport",
        )
        model_block.natural_gas_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_block.coal_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.fossil_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.hydrogen_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.co2_energy = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_block.operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

    def _add_firing_constraints(self, model_block: pyo.Block) -> None:
        """Heat balance, fossil fuel split and zeroing of the unused carriers."""

        @model_block.Constraint(self.time_steps)
        def heat_balance(b, t):
            generated = 0
            if self.uses_electricity:
                generated += b.power_in[t] * b.eta_electric
            if self.uses_fossil:
                generated += b.fossil_in[t] * b.eta_fossil
            if self.fuel_type == "hydrogen":
                generated += b.hydrogen_in[t] * b.eta_fossil
            return b.heat_out[t] == generated + self._external_heat_expr(b, t)

        if self.uses_fossil:

            @model_block.Constraint(self.time_steps)
            def fossil_sum(b, t):
                return b.fossil_in[t] == b.natural_gas_in[t] + b.coal_in[t]

            @model_block.Constraint(self.time_steps)
            def fossil_split_ng(b, t):
                return b.natural_gas_in[t] == b.fossil_ng_share * b.fossil_in[t]

        else:

            @model_block.Constraint(self.time_steps)
            def no_fossil_input(b, t):
                return b.natural_gas_in[t] + b.coal_in[t] + b.fossil_in[t] == 0

        if not self.uses_electricity:

            @model_block.Constraint(self.time_steps)
            def no_electric_heating(b, t):
                return b.power_in[t] == 0

        if self.fuel_type != "hydrogen":

            @model_block.Constraint(self.time_steps)
            def no_hydrogen_input(b, t):
                return b.hydrogen_in[t] == 0

    def _external_heat_expr(self, model_block: pyo.Block, t: int):
        """Heat supplied from outside the stage (overridden where applicable)."""
        return 0

    def _add_aux_and_cost_constraints(self, model_block: pyo.Block, model) -> None:
        throughput = getattr(model_block, self.throughput_name)

        @model_block.Constraint(self.time_steps)
        def aux_power_constraint(b, t):
            return b.aux_power[t] == throughput[t] * b.specific_electricity_aux

        @model_block.Constraint(self.time_steps)
        def energy_co2_constraint(b, t):
            return (
                b.co2_energy[t]
                == b.natural_gas_in[t] * b.ng_co2_factor
                + b.coal_in[t] * b.coal_co2_factor
            )

        @model_block.Constraint(self.time_steps)
        def operating_cost_constraint(b, t):
            cost = (b.power_in[t] + b.aux_power[t]) * model.electricity_price[t]
            if self.uses_fossil:
                cost += (
                    b.natural_gas_in[t] * model.natural_gas_price[t]
                    + b.coal_in[t] * model.coal_price[t]
                )
            if self.fuel_type == "hydrogen":
                cost += b.hydrogen_in[t] * model.hydrogen_price[t]
            cost += self._emission_cost_expr(b, t, model)
            return b.operating_cost[t] == cost

    def _emission_cost_expr(self, model_block: pyo.Block, t: int, model):
        """CO2 cost of the stage; extended by stages with process emissions."""
        return model_block.co2_energy[t] * model.co2_price[t]

    def _add_operational_constraints(self, model_block: pyo.Block) -> None:
        """Availability, ramping and, where configured, unit commitment on the heat output."""
        if self.availability_profile is not None:

            @model_block.Constraint(self.time_steps)
            def availability_constraint(b, t):
                return (
                    b.heat_out[t]
                    <= availability_at(self.availability_profile, t) * b.max_heat_out
                )

        add_ramping_constraints(
            model_block=model_block,
            time_steps=self.time_steps,
            ramped=model_block.heat_out,
        )

        if (
            self.min_operating_steps > 1
            or self.min_down_steps > 1
            or self.min_heat_out > 0
        ):
            add_min_up_down_time_constraints(
                model_block=model_block,
                time_steps=self.time_steps,
                quantity="heat_out",
                min_quantity="min_heat_out",
                max_quantity="max_heat_out",
            )

    def add_to_model(
        self, model: pyo.ConcreteModel, model_block: pyo.Block
    ) -> pyo.Block:
        """
        Adds the thermal stage to the Pyomo model, defining parameters, variables and constraints.

        Pyomo Components:
            - **Parameters**: `max_heat_out`, `min_heat_out`, `specific_heat_demand`,
              `specific_electricity_aux`, `eta_electric`, `eta_fossil`, `fossil_ng_share`,
              `max_power`, `ramp_up`, `ramp_down`, `ng_co2_factor`, `coal_co2_factor`,
              `min_operating_steps`, `min_down_steps`, `initial_operational_status`.
            - **Variables**: `heat_out[t]`, `power_in[t]`, `aux_power[t]`,
              `natural_gas_in[t]`, `coal_in[t]`, `fossil_in[t]`, `hydrogen_in[t]`,
              `co2_energy[t]`, `operating_cost[t]`, the stage throughput, and optionally
              `operational_status[t]` / `start_up[t]` / `shut_down[t]` (binary).
            - **Constraints**: heat balance for the configured firing mode, fossil fuel
              split, zeroing of unused carriers, throughput from heat, auxiliary
              electricity, energy emissions, operating cost, ramping and minimum
              up/down times.

        Args:
            model (pyo.ConcreteModel): The optimisation model the block belongs to.
            model_block (pyo.Block): The block the stage is added to.

        Returns:
            pyo.Block: The block representing the thermal stage.
        """
        self._check_prices(model)
        self._add_common_parameters(model_block)
        self._add_stage_parameters(model_block)
        self._add_common_variables(model_block)
        self._add_stage_variables(model_block)
        self._add_firing_constraints(model_block)
        self._add_stage_constraints(model_block, model)
        self._add_aux_and_cost_constraints(model_block, model)
        self._add_operational_constraints(model_block)
        return model_block

    # Stage-specific hooks
    def _add_stage_parameters(self, model_block: pyo.Block) -> None:
        pass

    def _add_stage_variables(self, model_block: pyo.Block) -> None:
        pass

    def _add_stage_constraints(self, model_block: pyo.Block, model) -> None:
        pass


class Preheater(ThermalProcessStage):
    """
    A fuel-switchable preheater for cement raw meal (medium temperature, ~300-800 °C).

    The stage heats raw meal with electricity, natural gas, coal or hydrogen, and can
    additionally use waste heat recovered elsewhere in the plant. The recovered heat
    enters through ``external_heat_in``, which the plant unit links to its waste heat
    source (and fixes to zero when there is none).

    Args:
        See :class:`ThermalProcessStage`; ``specific_heat_demand`` is given per tonne of
        raw meal (MWh_th/t) and the throughput variable is ``raw_meal_out``.
    """

    throughput_name = "raw_meal_out"

    def _add_stage_variables(self, model_block: pyo.Block) -> None:
        model_block.raw_meal_out = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            doc="Preheated raw meal leaving the stage (t)",
        )
        model_block.external_heat_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            doc="Waste heat supplied from elsewhere in the plant (MWh_th)",
        )

    def _external_heat_expr(self, model_block: pyo.Block, t: int):
        return model_block.external_heat_in[t]

    def _add_stage_constraints(self, model_block: pyo.Block, model) -> None:
        @model_block.Constraint(self.time_steps)
        def throughput_from_heat(b, t):
            return b.raw_meal_out[t] == b.heat_out[t] / b.specific_heat_demand


class Calciner(ThermalProcessStage):
    """
    A fuel-switchable calciner for cement raw meal (high temperature, ~920-930 °C).

    Calcination releases process CO2 in proportion to the clinker produced, on top of
    the energy emissions of the fuel. The heat that actually drives calcination is
    ``effective_heat_in``, which the plant unit links to the burner output plus any
    thermal storage discharge - this is what lets an electric thermal storage displace
    burner heat in expensive hours.

    Args:
        calcination_emission_factor (float, optional): Process emissions per tonne of clinker (t CO2/t). Defaults to 0.525.
        Other arguments: see :class:`ThermalProcessStage`; ``specific_heat_demand`` is
        given per tonne of clinker (MWh_th/t) and the throughput variable is ``clinker_out``.
    """

    throughput_name = "clinker_out"

    def __init__(self, *args, calcination_emission_factor: float = 0.525, **kwargs):
        super().__init__(*args, **kwargs)
        self.calcination_emission_factor = calcination_emission_factor

    def _add_stage_parameters(self, model_block: pyo.Block) -> None:
        model_block.calcination_emission_factor = pyo.Param(
            initialize=self.calcination_emission_factor
        )

    def _add_stage_variables(self, model_block: pyo.Block) -> None:
        model_block.clinker_out = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            doc="Calcined material leaving the stage (t)",
        )
        model_block.effective_heat_in = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            doc="Heat driving calcination, incl. thermal storage discharge (MWh_th)",
        )
        model_block.co2_process = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

    def _add_stage_constraints(self, model_block: pyo.Block, model) -> None:
        @model_block.Constraint(self.time_steps)
        def throughput_from_heat(b, t):
            return b.clinker_out[t] == b.effective_heat_in[t] / b.specific_heat_demand

        @model_block.Constraint(self.time_steps)
        def process_co2_constraint(b, t):
            return b.co2_process[t] == b.clinker_out[t] * b.calcination_emission_factor

    def _emission_cost_expr(self, model_block: pyo.Block, t: int, model):
        return (
            model_block.co2_energy[t] + model_block.co2_process[t]
        ) * model.co2_price[t]


class Kiln(ThermalProcessStage):
    """
    A fuel-switchable rotary kiln for final clinkerisation (~1450 °C).

    Unlike the calciner the kiln has no process emissions - the calcination CO2 has
    already been released upstream - so only the energy emissions of its fuel are
    accounted for. Its thermal demand per tonne of clinker is correspondingly higher.

    Args:
        See :class:`ThermalProcessStage`; ``specific_heat_demand`` is given per tonne of
        clinker (MWh_th/t) and the throughput variable is ``clinker_out``.
    """

    throughput_name = "clinker_out"

    def _add_stage_variables(self, model_block: pyo.Block) -> None:
        model_block.clinker_out = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            doc="Clinker leaving the kiln (t)",
        )

    def _add_stage_constraints(self, model_block: pyo.Block, model) -> None:
        @model_block.Constraint(self.time_steps)
        def throughput_from_heat(b, t):
            return b.clinker_out[t] == b.heat_out[t] / b.specific_heat_demand


# Mapping of component type identifiers to their respective classes
demand_side_technologies: dict = {
    "electrolyser": Electrolyser,
    "hydrogen_buffer_storage": HydrogenBufferStorage,
    "hydrogen_seasonal_storage": SeasonalHydrogenStorage,
    "dri_plant": DRIPlant,
    "dri_storage": DRIStorage,
    "eaf": ElectricArcFurnace,
    "heat_pump": HeatPump,
    "boiler": Boiler,
    "electric_vehicle": ElectricVehicle,
    "charging_station": ChargingStation,
    "generic_storage": GenericStorage,
    "pv_plant": PVPlant,
    "thermal_storage": ThermalStorage,
    "raw_material_mill": GrindingMill,
    "cement_mill": GrindingMill,
    "preheater": Preheater,
    "calciner": Calciner,
    "kiln": Kiln,
}


def sanitize_availability_profile(availability_profile, component_name: str):
    """Return *availability_profile* if it can be indexed per time step, else ``None``.

    Availability is expected as a per-time-step series. Scalar flags (such as the
    ``"yes"``/``"no"`` strings sometimes present in DSM input files) carry no temporal
    information and are dropped with a warning rather than failing the run.
    """
    if availability_profile is None:
        return None
    if isinstance(availability_profile, str) or not hasattr(
        availability_profile, "__getitem__"
    ):
        logger.warning(
            "%s received a non-series availability_profile (%r); "
            "the availability constraint is skipped.",
            component_name,
            availability_profile,
        )
        return None
    return availability_profile


def availability_at(availability_profile, t: int) -> float:
    """Positional lookup of an availability value for local time step *t*."""
    if hasattr(availability_profile, "iat"):
        return float(availability_profile.iat[t])
    return float(availability_profile[t])


def add_ramping_constraints(model_block, time_steps, ramped):
    """Limit the change of *ramped* variable between consecutive time steps.

    *ramped* is the Pyomo Var on *model_block* the ramp limits apply to (e.g.
    ``model_block.power_in`` for electric components, ``model_block.heat_out`` for
    thermally driven ones).

    The first time step has no predecessor, so it is only capped by ``ramp_up`` (how
    fast the component can rise from an assumed standing start of zero). ``ramp_down``
    is skipped there rather than also capping the first value: there is no preceding
    value to decrease *from*, so a low ``ramp_down`` must not choke off a legitimately
    high first-step value the way it would if it were treated as a second, tighter
    absolute cap alongside ``ramp_up``.
    """

    # Ramp-up constraint
    @model_block.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        if t == time_steps.at(1):
            return ramped[t] <= b.ramp_up
        return ramped[t] - ramped[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_block.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        if t == time_steps.at(1):
            return pyo.Constraint.Skip
        return ramped[t - 1] - ramped[t] <= b.ramp_down

    return model_block


def add_min_up_down_time_constraints(
    model_block,
    time_steps,
    quantity: str = "power_in",
    min_quantity: str = "min_power",
    max_quantity: str = "max_power",
):
    """Add an on/off status with minimum up- and down-time to a component block.

    *quantity* names the block variable the status switches (``power_in`` for electric
    components, ``heat_out`` for thermally driven ones), bounded by the *min_quantity*
    and *max_quantity* parameters while the component is on.
    """
    model_block.operational_status = pyo.Var(time_steps, within=pyo.Binary)

    switched = getattr(model_block, quantity)
    min_level = getattr(model_block, min_quantity)
    max_level = getattr(model_block, max_quantity)

    # Output constraints based on operational status
    @model_block.Constraint(time_steps)
    def min_power_constraint(b, t):
        return switched[t] >= min_level * b.operational_status[t]

    @model_block.Constraint(time_steps)
    def max_power_constraint(b, t):
        return switched[t] <= max_level * b.operational_status[t]

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
