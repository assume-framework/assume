# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import pandas as pd
import pyomo.environ as pyo

from assume.common.base import BaseDSTComponent

logger = logging.getLogger(__name__)


class HeatPump(BaseDSTComponent):
    """
    A class to represent a generic heat pump unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a heat pump, such as power input, heat output, and operational limitations
    (like ramp rates and minimum operating times).

    Attributes
    ----------
        max_power (float): Maximum allowable power input to the heat pump.
        cop (float): Coefficient of performance of the heat pump, i.e., the ratio of heat output to power input.
        time_steps (list[int]): A list of time steps over which the heat pump operates.
        min_power (float, optional): Minimum allowable power input to the heat pump. Defaults to 0.0.
        ramp_up (float, optional): Maximum allowed increase in power input per time step. Defaults to `max_power` if not provided.
        ramp_down (float, optional): Maximum allowed decrease in power input per time step. Defaults to `max_power` if not provided.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the heat pump must operate once it starts. Defaults to 0 (no restriction).
        min_down_steps (int, optional): Minimum number of consecutive time steps the heat pump must remain off after being shut down. Defaults to 0 (no restriction).

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Adds a heat pump block to the Pyomo model, defining parameters, variables, and constraints.

            Pyomo Components:
                - **Parameters**:
                    - `max_power`: The maximum allowable power input.
                    - `min_power`: The minimum allowable power input.
                    - `cop`: Coefficient of performance of the heat pump.
                    - `ramp_up`: Maximum allowed increase in power per time step.
                    - `ramp_down`: Maximum allowed decrease in power per time step.
                    - `min_operating_steps`: Minimum number of consecutive time steps the heat pump must operate.
                    - `min_down_steps`: Minimum number of consecutive time steps the heat pump must remain off.

                - **Variables**:
                    - `power_in[t]`: Power input to the heat pump at each time step `t` (continuous, non-negative).
                    - `heat_out[t]`: Heat output of the heat pump at each time step `t` (continuous, non-negative).
                    - `operational_status[t]` (optional): A binary variable indicating whether the heat pump is operational (1) or off (0) at each time step `t`.

                - **Constraints**:
                    - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the heat pump is operational.
                    - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input when the heat pump is operational.
                    - `cop_constraint[t]`: Enforces the relationship between power input and heat output based on the coefficient of performance (COP).
                    - `ramp_up_constraint[t]`: Limits the increase in power input from one time step to the next according to the ramp-up rate.
                    - `ramp_down_constraint[t]`: Limits the decrease in power input from one time step to the next according to the ramp-down rate.
                    - `min_operating_time_constraint[t]`: Ensures the heat pump operates for at least the specified minimum number of consecutive time steps.
                    - `min_downtime_constraint[t]`: Ensures the heat pump remains off for at least the specified minimum number of consecutive time steps after shutdown.
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
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the heat pump component.

        Args:
            model (ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.

        Returns:
            pyo.Block: A Pyomo block representing the heat pump with variables and constraints.
        """

        # Define parameters
        model_part.max_power = pyo.Param(initialize=self.max_power)
        model_part.min_power = pyo.Param(initialize=self.min_power)
        model_part.cop = pyo.Param(initialize=self.cop)
        model_part.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_part.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_part.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_part.min_down_steps = pyo.Param(initialize=self.min_down_steps)

        # Define variables
        model_part.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_part.heat_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        # Define operational status variable if necessary
        if (
            self.min_operating_steps > 0
            or self.min_down_steps > 0
            or self.min_power > 0
        ):
            model_part.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

            @model_part.Constraint(self.time_steps)
            def min_power_constraint(b, t):
                return b.power_in[t] >= b.min_power * b.operational_status[t]

            @model_part.Constraint(self.time_steps)
            def max_power_constraint(b, t):
                return b.power_in[t] <= b.max_power * b.operational_status[t]

        # Coefficient of performance (COP) constraint
        @model_part.Constraint(self.time_steps)
        def cop_constraint(b, t):
            return b.heat_out[t] == b.power_in[t] * b.cop

        # Ramp-up constraint
        @model_part.Constraint(self.time_steps)
        def ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

        # Ramp-down constraint
        @model_part.Constraint(self.time_steps)
        def ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

        # Minimum operating time constraint
        if self.min_operating_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_operating_time_constraint(b, t):
                if t < self.min_operating_steps - 1:
                    return pyo.Constraint.Skip

                relevant_time_steps = range(t - self.min_operating_steps + 1, t + 1)
                return (
                    sum(b.operational_status[i] for i in relevant_time_steps)
                    >= self.min_operating_steps * b.operational_status[t]
                )

        # Minimum downtime constraint
        if self.min_down_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_downtime_constraint(b, t):
                if t < self.min_down_steps - 1:
                    return pyo.Constraint.Skip

                relevant_time_steps = range(t - self.min_down_steps + 1, t + 1)
                return sum(
                    1 - b.operational_status[i] for i in relevant_time_steps
                ) >= self.min_down_steps * (1 - b.operational_status[t])

        return model_part


class Boiler(BaseDSTComponent):
    """
    A class to represent a generic boiler unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of a boiler, which can be either electric or natural gas-based, along with ramp rates and operational
    limitations.

    Attributes
    ----------
        max_power (float): Maximum allowable power input to the boiler.
        efficiency (float): Efficiency of the boiler, defined as the ratio of heat output to power input (or fuel input).
        time_steps (list[int]): A list of time steps over which the boiler operates.
        fuel_type (str, optional): Type of fuel used by the boiler ('electricity' or 'natural_gas'). Defaults to 'electricity'.
        min_power (float, optional): Minimum allowable power input to the boiler. Defaults to 0.0.
        ramp_up (float, optional): Maximum allowed increase in power input per time step. Defaults to `max_power` if not provided.
        ramp_down (float, optional): Maximum allowed decrease in power input per time step. Defaults to `max_power` if not provided.
        min_operating_steps (int, optional): Minimum number of consecutive time steps the boiler must operate once started. Defaults to 0.
        min_down_steps (int, optional): Minimum number of consecutive time steps the boiler must remain off after being shut down. Defaults to 0.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Adds a boiler block to the Pyomo model, defining parameters, variables, and constraints.

            Pyomo Components:
                - **Parameters**:
                    - `max_power`: The maximum allowable power input.
                    - `min_power`: The minimum allowable power input.
                    - `efficiency`: Efficiency of the boiler.
                    - `ramp_up`: Maximum allowed increase in power per time step.
                    - `ramp_down`: Maximum allowed decrease in power per time step.

                - **Variables**:
                    - `power_in[t]` (for electric boilers): Power input at each time step `t` (continuous, non-negative).
                    - `natural_gas_in[t]` (for natural gas boilers): Natural gas input at each time step `t` (continuous, non-negative).
                    - `heat_out[t]`: Heat output at each time step `t` (continuous, non-negative).
                    - `operational_status[t]` (optional, for electric boilers): A binary variable indicating whether the boiler is operational (1) or off (0) at each time step `t`.

                - **Constraints**:
                    - `min_power_constraint[t]` (for electric boilers): Ensures that the power input is at least the minimum power input when the boiler is operational.
                    - `max_power_constraint[t]` (for electric boilers): Ensures that the power input does not exceed the maximum power input when the boiler is operational.
                    - `efficiency_constraint[t]`: Enforces the relationship between input (power or natural gas) and heat output based on the boiler's efficiency.
                    - `ramp_up_constraint[t]`: Limits the increase in power input from one time step to the next according to the ramp-up rate.
                    - `ramp_down_constraint[t]`: Limits the decrease in power input from one time step to the next according to the ramp-down rate.
                    - `min_operating_time_constraint[t]`: Ensures the boiler operates for at least the specified minimum number of consecutive time steps.
                    - `min_downtime_constraint[t]`: Ensures the boiler remains off for at least the specified minimum number of consecutive time steps after shutdown.
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
        self.fuel_type = fuel_type
        self.kwargs = kwargs

        if self.fuel_type not in ["electricity", "natural_gas"]:
            raise ValueError(
                "Unsupported fuel_type for a boiler. Choose 'electricity' or 'natural_gas'."
            )

    def add_to_model(
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the boiler component.

        Args:
            model (ConcreteModel): A Pyomo ConcreteModel object representing the optimization model.

        Returns:
            pyo.Block: A Pyomo block representing the boiler with variables and constraints.
        """

        # Define parameters
        model_part.max_power = pyo.Param(initialize=self.max_power)
        model_part.min_power = pyo.Param(initialize=self.min_power)
        model_part.efficiency = pyo.Param(initialize=self.efficiency)
        model_part.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_part.ramp_down = pyo.Param(initialize=self.ramp_down)

        # Define variables
        if self.fuel_type == "electricity":
            model_part.power_in = pyo.Var(
                self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
            )
        elif self.fuel_type == "natural_gas":
            model_part.natural_gas_in = pyo.Var(
                self.time_steps, within=pyo.NonNegativeReals
            )

        model_part.heat_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

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
            model_part.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

            @model_part.Constraint(self.time_steps)
            def min_power_constraint(b, t):
                return b.power_in[t] >= b.min_power * b.operational_status[t]

            @model_part.Constraint(self.time_steps)
            def max_power_constraint(b, t):
                return b.power_in[t] <= b.max_power * b.operational_status[t]

        # Efficiency constraint based on fuel type
        @model_part.Constraint(self.time_steps)
        def efficiency_constraint(b, t):
            if self.fuel_type == "electricity":
                return b.heat_out[t] == b.power_in[t] * b.efficiency
            elif self.fuel_type == "natural_gas":
                return b.heat_out[t] == b.natural_gas_in[t] * b.efficiency
            else:
                raise ValueError(
                    "Unsupported fuel_type. Choose 'electricity' or 'natural_gas'."
                )

        # Ramp-up constraint
        @model_part.Constraint(self.time_steps)
        def ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

        # Ramp-down constraint
        @model_part.Constraint(self.time_steps)
        def ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

        # Minimum operating time constraint
        if self.min_operating_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_operating_time_constraint(b, t):
                if t < self.min_operating_steps - 1:
                    return pyo.Constraint.Skip

                relevant_time_steps = range(t - self.min_operating_steps + 1, t + 1)
                return (
                    sum(b.operational_status[i] for i in relevant_time_steps)
                    >= self.min_operating_steps * b.operational_status[t]
                )

        # Minimum downtime constraint
        if self.min_down_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_downtime_constraint(b, t):
                if t < self.min_down_steps - 1:
                    return pyo.Constraint.Skip

                relevant_time_steps = range(t - self.min_down_steps + 1, t + 1)
                return sum(
                    1 - b.operational_status[i] for i in relevant_time_steps
                ) >= self.min_down_steps * (1 - b.operational_status[t])

        return model_part


class GenericStorage(BaseDSTComponent):
    """
    A class to represent a generic storage unit (e.g., battery) in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a storage system, including charging, discharging, state of charge (SOC),
    ramp rates, and storage losses.

    Attributes
    ----------
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

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
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
            logger.warning("Initial SOC is greater than 1.0. Setting it to 1.0.")
            initial_soc = 1.0

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
        self.initial_soc = initial_soc
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.storage_loss_rate = storage_loss_rate
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the storage component.

        Parameters
        ----------
        model : pyo.ConcreteModel
            A Pyomo ConcreteModel object representing the optimization model.

        Returns
        -------
        pyo.Block
            A Pyomo block representing the storage system with variables and constraints.
        """

        # Define parameters
        model_part.max_capacity = pyo.Param(initialize=self.max_capacity)
        model_part.min_capacity = pyo.Param(initialize=self.min_capacity)
        model_part.max_power_charge = pyo.Param(initialize=self.max_power_charge)
        model_part.max_power_discharge = pyo.Param(initialize=self.max_power_discharge)
        model_part.efficiency_charge = pyo.Param(initialize=self.efficiency_charge)
        model_part.efficiency_discharge = pyo.Param(
            initialize=self.efficiency_discharge
        )
        model_part.initial_soc = pyo.Param(
            initialize=self.initial_soc * self.max_capacity
        )
        model_part.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_part.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_part.storage_loss_rate = pyo.Param(initialize=self.storage_loss_rate)

        # Define variables
        model_part.soc = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(self.min_capacity, self.max_capacity),
            doc="State of Charge at each time step",
        )
        model_part.charge = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_power_charge),
            doc="Charging power at each time step",
        )
        model_part.discharge = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_power_discharge),
            doc="Discharging power at each time step",
        )

        # Define SOC dynamics with energy loss and efficiency
        @model_part.Constraint(self.time_steps)
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
        if self.ramp_up is not None:

            @model_part.Constraint(self.time_steps)
            def charge_ramp_up_constraint(b, t):
                if t == self.time_steps.at(1):
                    return pyo.Constraint.Skip
                return b.charge[t] - b.charge[t - 1] <= self.ramp_up

            @model_part.Constraint(self.time_steps)
            def discharge_ramp_up_constraint(b, t):
                if t == self.time_steps.at(1):
                    return pyo.Constraint.Skip
                return b.discharge[t] - b.discharge[t - 1] <= self.ramp_up

        # Apply ramp-down constraints if ramp_down is specified
        if self.ramp_down is not None:

            @model_part.Constraint(self.time_steps)
            def charge_ramp_down_constraint(b, t):
                if t == self.time_steps.at(1):
                    return pyo.Constraint.Skip
                return b.charge[t - 1] - b.charge[t] <= self.ramp_down

            @model_part.Constraint(self.time_steps)
            def discharge_ramp_down_constraint(b, t):
                if t == self.time_steps.at(1):
                    return pyo.Constraint.Skip
                return b.discharge[t - 1] - b.discharge[t] <= self.ramp_down

        return model_part


class PVPlant(BaseDSTComponent):
    """
    A class to represent a Photovoltaic (PV) power plant unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of a PV plant, including availability profiles and predefined power output profiles.

    Attributes
    ----------
        max_power (float): The maximum power output of the PV unit.
        time_steps (list[int]): A list of time steps over which the PV operates.
        availability_profile (pd.Series | None, optional): A pandas Series indicating the PV's availability with time_steps as indices
            and binary values (1 available, 0 unavailable). Defaults to None.
        power_profile (pd.Series | None, optional): A predefined power output profile. If provided, the PV follows this profile instead of optimizing the power output. Defaults to None.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Adds a PV plant block to the Pyomo model, defining parameters, variables, and constraints.

            Pyomo Components:
                - **Parameters**:
                    - `max_power`: Maximum allowable power output.

                - **Variables**:
                    - `power[t]`: Power output of the PV plant at each time step `t`.

                - **Constraints**:
                    - `power_profile_constraint`: Ensures the PV follows a predefined power profile if provided.
                    - `availability_pv_constraint`: Ensures the PV operates only during available periods.
                    - `max_power_pv_constraint`: Ensures the power output of the PV unit does not exceed the maximum power limit.
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
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the PV plant component.

        Parameters
        ----------
        model : pyo.ConcreteModel
            A Pyomo ConcreteModel object representing the optimization model.

        Returns
        -------
        pyo.Block
            A Pyomo block representing the PV plant with variables and constraints.
        """

        # Define parameters
        model_part.max_power = pyo.Param(
            initialize=self.max_power, within=pyo.NonNegativeReals
        )

        # Define variables
        model_part.power = pyo.Var(
            self.time_steps,
            within=pyo.NonNegativeReals,
            bounds=(0, self.max_power),
        )

        # Define constraints

        # Predefined power profile constraint
        if self.power_profile is not None:
            if not isinstance(self.power_profile, pd.Series):
                raise TypeError(
                    "Residential PV `power_profile` must be a pandas Series."
                )
            if not all(t in self.power_profile.index for t in self.time_steps):
                raise ValueError(
                    "All `time_steps` must be present in residential PV `power_profile` index."
                )

            @model_part.Constraint(self.time_steps)
            def power_profile_constraint(b, t):
                """
                Ensures the PV follows the predefined power profile.
                """
                return b.power[t] == self.power_profile[t]

        # Availability profile constraints
        if self.availability_profile is not None:
            if not isinstance(self.availability_profile, pd.Series):
                raise TypeError(
                    "Residential PV `availability_profile` must be a pandas Series."
                )
            if not all(t in self.availability_profile.index for t in self.time_steps):
                raise ValueError(
                    "All `time_steps` must be present in residential PV `availability_profile` index."
                )

            @model_part.Constraint(self.time_steps)
            def availability_pv_constraint(b, t):
                """
                Ensures the PV operates only during available periods.
                """
                return b.power[t] <= self.availability_profile[t] * b.max_power

        # Maximum power constraint (redundant due to variable bounds, included for clarity)
        @model_part.Constraint(self.time_steps)
        def max_power_pv_constraint(b, t):
            """
            Ensures the power output of the PV unit does not exceed the maximum power limit.
            """
            return b.power[t] <= b.max_power

        return model_part


class Electrolyser(BaseDSTComponent):
    """
    A class to represent an electrolyser unit used for hydrogen production through electrolysis.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of an electrolyser, including power input, hydrogen output, ramp rates, and operating times.

    Attributes
    ----------
        max_power (float): The rated power capacity of the electrolyser.
        efficiency (float): The efficiency of the electrolysis process (0-1).
        min_power (float): The minimum power required for operation.
        ramp_up (float, optional): The maximum rate at which the electrolyser can increase its power output. Defaults to `max_power`.
        ramp_down (float, optional): The maximum rate at which the electrolyser can decrease its power output. Defaults to `max_power`.
        min_operating_steps (int, optional): The minimum number of steps the electrolyser must operate continuously. Defaults to 0.
        min_down_steps (int, optional): The minimum number of downtime steps required between operating cycles. Defaults to 0.
        time_steps (list[int]): A list of time steps over which the electrolyser operates.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
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

                - **Variables**:
                    - `power_in[t]`: Power input to the electrolyser at each time step `t`.
                    - `hydrogen_out[t]`: Hydrogen output at each time step `t`.
                    - `electrolyser_operating_cost[t]`: Operating cost at each time step `t`.
                    - `operational_status[t]` (optional): Binary variable indicating whether the electrolyser is operational.

                - **Constraints**:
                    - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the electrolyser is operational.
                    - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input.
                    - `hydrogen_production_constraint[t]`: Relates power input to hydrogen output based on efficiency.
                    - `ramp_up_constraint[t]`: Limits the ramp-up rate of power input.
                    - `ramp_down_constraint[t]`: Limits the ramp-down rate of power input.
                    - `min_operating_time_constraint[t]`: Ensures the electrolyser operates for a minimum duration.
                    - `min_downtime_constraint[t]`: Ensures the electrolyser remains off for a minimum duration between operations.
                    - `operating_cost_with_el_price[t]`: Calculates the operating cost based on power input and electricity price.
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
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the electrolyser component.

        Parameters
        ----------
        model : pyo.ConcreteModel
            A Pyomo ConcreteModel object representing the optimization model.

        Returns
        -------
        pyo.Block
            A Pyomo block representing the electrolyser with variables and constraints.
        """

        # Define parameters
        model_part.max_power = pyo.Param(initialize=self.max_power)
        model_part.efficiency = pyo.Param(initialize=self.efficiency)
        model_part.min_power = pyo.Param(initialize=self.min_power)
        model_part.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_part.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_part.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_part.min_down_steps = pyo.Param(initialize=self.min_down_steps)

        # Define variables
        model_part.power_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_part.hydrogen_out = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.electrolyser_operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

        # Define operational status variable if needed
        if (
            self.min_operating_steps > 0
            or self.min_down_steps > 0
            or self.min_power > 0
        ):
            model_part.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

            @model_part.Constraint(self.time_steps)
            def min_power_constraint(b, t):
                return b.power_in[t] >= b.min_power * b.operational_status[t]

            @model_part.Constraint(self.time_steps)
            def max_power_constraint(b, t):
                return b.power_in[t] <= b.max_power * b.operational_status[t]

        # Efficiency constraint
        @model_part.Constraint(self.time_steps)
        def hydrogen_production_constraint(b, t):
            return b.power_in[t] == b.hydrogen_out[t] / b.efficiency

        # Ramp-up constraint
        @model_part.Constraint(self.time_steps)
        def ramp_up_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

        # Ramp-down constraint
        @model_part.Constraint(self.time_steps)
        def ramp_down_constraint(b, t):
            if t == self.time_steps.at(1):
                return pyo.Constraint.Skip
            return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

        # Minimum operating time constraint
        if self.min_operating_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_operating_time_constraint(b, t):
                if t < self.min_operating_steps - 1:
                    return pyo.Constraint.Skip
                relevant_time_steps = range(t - self.min_operating_steps + 1, t + 1)
                return (
                    sum(b.operational_status[i] for i in relevant_time_steps)
                    >= self.min_operating_steps * b.operational_status[t]
                )

        # Minimum downtime constraint
        if self.min_down_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_downtime_constraint(b, t):
                if t < self.min_down_steps - 1:
                    return pyo.Constraint.Skip
                relevant_time_steps = range(t - self.min_down_steps + 1, t + 1)
                return sum(
                    1 - b.operational_status[i] for i in relevant_time_steps
                ) >= self.min_down_steps * (1 - b.operational_status[t])

        # Operating cost constraint
        @model_part.Constraint(self.time_steps)
        def operating_cost_with_el_price(b, t):
            return (
                b.electrolyser_operating_cost[t]
                == b.power_in[t] * model.electricity_price[t]
            )

        return model_part


class DRIPlant(BaseDSTComponent):
    """
    A class to represent a DRI (Direct Reduced Iron) plant in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of a DRI plant, including power consumption, fuel consumption (hydrogen, natural gas, or both),
    iron ore input, and ramp rates.

    Attributes
    ----------
        specific_hydrogen_consumption (float): The specific hydrogen consumption of the DRI plant (in MWh per ton of DRI).
        specific_natural_gas_consumption (float): The specific natural gas consumption of the DRI plant (in MWh per ton of DRI).
        specific_electricity_consumption (float): The specific electricity consumption of the DRI plant (in MWh per ton of DRI).
        specific_iron_ore_consumption (float): The specific iron ore consumption of the DRI plant (in ton per ton of DRI).
        max_power (float): The rated power capacity of the DRI plant.
        min_power (float): The minimum power required for operation.
        fuel_type (str): The type of fuel used by the DRI plant ("hydrogen", "natural_gas", "both").
        ramp_up (float, optional): The maximum rate at which the DRI plant can increase its power output.
        ramp_down (float, optional): The maximum rate at which the DRI plant can decrease its power output.
        min_operating_steps (int, optional): The minimum number of steps the DRI plant must operate continuously. Defaults to 0.
        min_down_steps (int, optional): The minimum number of downtime steps required between operating cycles. Defaults to 0.
        time_steps (list[int]): A list of time steps over which the DRI plant operates.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Adds a DRI plant block to the Pyomo model, defining parameters, variables, and constraints.

            Pyomo Components:
            -----------------
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

                - **Variables**:
                    - `power_dri[t]`: Power input to the DRI plant at each time step `t`.
                    - `dri_output[t]`: DRI output at each time step `t`.
                    - `natural_gas_in[t]`: Natural gas input at each time step `t`.
                    - `hydrogen_in[t]`: Hydrogen input at each time step `t`.
                    - `iron_ore_in[t]`: Iron ore input at each time step `t`.
                    - `dri_operating_cost[t]`: Operating cost at each time step `t`.
                    - `operational_status[t]` (optional): Binary variable indicating whether the DRI plant is operational.

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
                    - `dri_operating_cost_constraint[t]`: Calculates the operating cost based on fuel and electricity consumption.
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
        **kwargs,
    ):
        super().__init__()

        self.specific_hydrogen_consumption = specific_hydrogen_consumption
        self.specific_natural_gas_consumption = specific_natural_gas_consumption
        self.specific_electricity_consumption = specific_electricity_consumption
        self.specific_iron_ore_consumption = specific_iron_ore_consumption
        self.max_power = max_power
        self.min_power = min_power
        self.fuel_type = fuel_type
        self.time_steps = time_steps
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the DRI plant component.

        Parameters
        ----------
        model : pyo.ConcreteModel
            A Pyomo ConcreteModel object representing the optimization model.

        Returns
        -------
        pyo.Block
            A Pyomo block representing the DRI plant with variables and constraints.
        """

        # Define parameters
        model_part.specific_hydrogen_consumption = pyo.Param(
            initialize=self.specific_hydrogen_consumption
        )
        model_part.specific_natural_gas_consumption = pyo.Param(
            initialize=self.specific_natural_gas_consumption
        )
        model_part.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        model_part.specific_iron_ore_consumption = pyo.Param(
            initialize=self.specific_iron_ore_consumption
        )
        model_part.max_power = pyo.Param(initialize=self.max_power)
        model_part.min_power = pyo.Param(initialize=self.min_power)
        model_part.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_part.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_part.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_part.min_down_steps = pyo.Param(initialize=self.min_down_steps)

        # Define variables
        model_part.power_dri = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_part.iron_ore_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.natural_gas_in = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_part.hydrogen_in = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.dri_output = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.dri_operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )

        # Define operational status variable if needed
        if (
            self.min_operating_steps > 0
            or self.min_down_steps > 0
            or self.min_power > 0
        ):
            model_part.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

            @model_part.Constraint(self.time_steps)
            def min_power_constraint(b, t):
                return b.power_dri[t] >= b.min_power * b.operational_status[t]

            @model_part.Constraint(self.time_steps)
            def max_power_constraint(b, t):
                return b.power_dri[t] <= b.max_power * b.operational_status[t]

        # Fuel consumption constraint
        @model_part.Constraint(self.time_steps)
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
                    b.hydrogen_in[t] / b.specific_hydrogen_consumption
                ) + (b.natural_gas_in[t] / b.specific_natural_gas_consumption)

        # Electricity consumption constraint
        @model_part.Constraint(self.time_steps)
        def electricity_consumption_constraint(b, t):
            return (
                b.power_dri[t] == b.dri_output[t] * b.specific_electricity_consumption
            )

        # Iron ore consumption constraint
        @model_part.Constraint(self.time_steps)
        def iron_ore_constraint(b, t):
            return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

        # Ramp-up constraint
        @model_part.Constraint(self.time_steps)
        def ramp_up_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_dri[t] - b.power_dri[t - 1] <= b.ramp_up

        # Ramp-down constraint
        @model_part.Constraint(self.time_steps)
        def ramp_down_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_dri[t - 1] - b.power_dri[t] <= b.ramp_down

        # Minimum operating time constraint
        if self.min_operating_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_operating_time_constraint(b, t):
                if t < self.min_operating_steps - 1:
                    return pyo.Constraint.Skip
                relevant_time_steps = range(t - self.min_operating_steps + 1, t + 1)
                return (
                    sum(b.operational_status[i] for i in relevant_time_steps)
                    >= self.min_operating_steps * b.operational_status[t]
                )

        # Minimum downtime constraint
        if self.min_down_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_downtime_constraint(b, t):
                if t < self.min_down_steps - 1:
                    return pyo.Constraint.Skip
                relevant_time_steps = range(t - self.min_down_steps + 1, t + 1)
                return sum(
                    1 - b.operational_status[i] for i in relevant_time_steps
                ) >= self.min_down_steps * (1 - b.operational_status[t])

        # Operating cost constraint
        @model_part.Constraint(self.time_steps)
        def dri_operating_cost_constraint(b, t):
            return (
                b.dri_operating_cost[t]
                == b.natural_gas_in[t] * model.natural_gas_price[t]
                + b.power_dri[t] * model.electricity_price[t]
                + b.iron_ore_in[t] * model.iron_ore_price
            )

        return model_part


class ElectricArcFurnace(BaseDSTComponent):
    """
    A class to represent an Electric Arc Furnace (EAF) in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model the behavior
    of an EAF, including power consumption, DRI input, lime demand, and ramp rates.

    Attributes
    ----------
        max_power (float): The rated power capacity of the electric arc furnace.
        min_power (float): The minimum power requirement of the electric arc furnace.
        specific_electricity_consumption (float): The specific electricity consumption of the electric arc furnace (in MWh per ton of steel produced).
        specific_dri_demand (float): The specific demand for Direct Reduced Iron (DRI) in the electric arc furnace (in tons per ton of steel produced).
        specific_lime_demand (float): The specific demand for lime in the electric arc furnace (in tons per ton of steel produced).
        ramp_up (float, optional): The ramp-up rate of the electric arc furnace. Defaults to `max_power`.
        ramp_down (float, optional): The ramp-down rate of the electric arc furnace. Defaults to `max_power`.
        min_operating_steps (int, optional): The minimum number of steps the EAF must operate continuously. Defaults to 0.
        min_down_steps (int, optional): The minimum number of downtime steps required between operating cycles. Defaults to 0.
        time_steps (list[int]): A list of time steps over which the EAF operates.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Adds an EAF block to the Pyomo model, defining parameters, variables, and constraints.

            Pyomo Components:
                - **Parameters**:
                    - `max_power`: Maximum allowable power input.
                    - `min_power`: Minimum allowable power input.
                    - `specific_electricity_consumption`: Electricity consumption per ton of steel produced.
                    - `specific_dri_demand`: DRI demand per ton of steel produced.
                    - `specific_lime_demand`: Lime demand per ton of steel produced.
                    - `ramp_up`: Maximum ramp-up rate.
                    - `ramp_down`: Maximum ramp-down rate.
                    - `min_operating_steps`: Minimum operating time.
                    - `min_down_steps`: Minimum downtime between operating cycles.

                - **Variables**:
                    - `power_eaf[t]`: Power input to the EAF at each time step `t`.
                    - `dri_input[t]`: DRI input at each time step `t`.
                    - `steel_output[t]`: Steel output at each time step `t`.
                    - `eaf_operating_cost[t]`: Operating cost at each time step `t`.
                    - `emission_eaf[t]`: Emissions at each time step `t`.
                    - `lime_demand[t]`: Lime demand at each time step `t`.
                    - `operational_status[t]` (optional): Binary variable indicating whether the EAF is operational.

                - **Constraints**:
                    - `min_power_constraint[t]`: Ensures that the power input is at least the minimum power input when the EAF is operational.
                    - `max_power_constraint[t]`: Ensures that the power input does not exceed the maximum power input.
                    - `steel_output_dri_relation[t]`: Links steel output to DRI input.
                    - `steel_output_power_relation[t]`: Links steel output to power consumption.
                    - `eaf_lime_demand[t]`: Links lime demand to steel output.
                    - `eaf_co2_emission[t]`: Links CO2 emissions to lime demand.
                    - `ramp_up_eaf_constraint[t]`: Limits the ramp-up rate of power input.
                    - `ramp_down_eaf_constraint[t]`: Limits the ramp-down rate of power input.
                    - `min_operating_time_constraint[t]`: Ensures the EAF operates for a minimum duration.
                    - `min_down_time_constraint[t]`: Ensures the EAF remains off for a minimum duration between operations.
                    - `eaf_operating_cost_constraint[t]`: Calculates the operating cost based on power input, CO2 emissions, and lime consumption.
    """

    def __init__(
        self,
        max_power: float,
        min_power: float,
        specific_electricity_consumption: float,
        specific_dri_demand: float,
        specific_lime_demand: float,
        time_steps: list[int],
        ramp_up: float | None = None,
        ramp_down: float | None = None,
        min_operating_steps: int = 0,
        min_down_steps: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.max_power = max_power
        self.min_power = min_power
        self.specific_electricity_consumption = specific_electricity_consumption
        self.specific_dri_demand = specific_dri_demand
        self.specific_lime_demand = specific_lime_demand
        self.time_steps = time_steps
        self.ramp_up = max_power if ramp_up is None else ramp_up
        self.ramp_down = max_power if ramp_down is None else ramp_down
        self.min_operating_steps = min_operating_steps
        self.min_down_steps = min_down_steps
        self.kwargs = kwargs

    def add_to_model(
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the EAF component.

        Parameters
        ----------
        model : pyo.ConcreteModel
            A Pyomo ConcreteModel object representing the optimization model.

        Returns
        -------
        pyo.Block
            A Pyomo block representing the EAF with variables and constraints.
        """

        # Define parameters
        model_part.max_power = pyo.Param(initialize=self.max_power)
        model_part.min_power = pyo.Param(initialize=self.min_power)
        model_part.specific_electricity_consumption = pyo.Param(
            initialize=self.specific_electricity_consumption
        )
        model_part.specific_dri_demand = pyo.Param(initialize=self.specific_dri_demand)
        model_part.specific_lime_demand = pyo.Param(
            initialize=self.specific_lime_demand
        )
        model_part.ramp_up = pyo.Param(initialize=self.ramp_up)
        model_part.ramp_down = pyo.Param(initialize=self.ramp_down)
        model_part.min_operating_steps = pyo.Param(initialize=self.min_operating_steps)
        model_part.min_down_steps = pyo.Param(initialize=self.min_down_steps)

        # Define variables
        model_part.power_eaf = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals, bounds=(0, self.max_power)
        )
        model_part.dri_input = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.steel_output = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.eaf_operating_cost = pyo.Var(
            self.time_steps, within=pyo.NonNegativeReals
        )
        model_part.emission_eaf = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)
        model_part.lime_demand = pyo.Var(self.time_steps, within=pyo.NonNegativeReals)

        # Define operational status variable if needed
        if (
            self.min_operating_steps > 0
            or self.min_down_steps > 0
            or self.min_power > 0
        ):
            model_part.operational_status = pyo.Var(self.time_steps, within=pyo.Binary)

            @model_part.Constraint(self.time_steps)
            def min_power_constraint(b, t):
                return b.power_eaf[t] >= b.min_power * b.operational_status[t]

            @model_part.Constraint(self.time_steps)
            def max_power_constraint(b, t):
                return b.power_eaf[t] <= b.max_power * b.operational_status[t]

        # Steel output based on DRI input
        @model_part.Constraint(self.time_steps)
        def steel_output_dri_relation(b, t):
            return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

        # Steel output based on power consumption
        @model_part.Constraint(self.time_steps)
        def steel_output_power_relation(b, t):
            return (
                b.power_eaf[t] == b.steel_output[t] * b.specific_electricity_consumption
            )

        # Lime demand based on steel output
        @model_part.Constraint(self.time_steps)
        def eaf_lime_demand(b, t):
            return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

        # CO2 emissions based on lime demand
        @model_part.Constraint(self.time_steps)
        def eaf_co2_emission(b, t):
            return b.emission_eaf[t] == b.lime_demand[t] * model.lime_co2_factor

        # Ramp-up constraint
        @model_part.Constraint(self.time_steps)
        def ramp_up_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_eaf[t] - b.power_eaf[t - 1] <= b.ramp_up

        # Ramp-down constraint
        @model_part.Constraint(self.time_steps)
        def ramp_down_eaf_constraint(b, t):
            if t == 0:
                return pyo.Constraint.Skip
            return b.power_eaf[t - 1] - b.power_eaf[t] <= b.ramp_down

        # Minimum operating time constraint
        if self.min_operating_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_operating_time_constraint(b, t):
                if t < self.min_operating_steps - 1:
                    return pyo.Constraint.Skip
                relevant_time_steps = range(t - self.min_operating_steps + 1, t + 1)
                return (
                    sum(b.operational_status[i] for i in relevant_time_steps)
                    >= self.min_operating_steps * b.operational_status[t]
                )

        # Minimum downtime constraint
        if self.min_down_steps > 0:

            @model_part.Constraint(self.time_steps)
            def min_down_time_constraint(b, t):
                if t < self.min_down_steps - 1:
                    return pyo.Constraint.Skip
                relevant_time_steps = range(t - self.min_down_steps + 1, t + 1)
                return sum(
                    1 - b.operational_status[i] for i in relevant_time_steps
                ) >= self.min_down_steps * (1 - b.operational_status[t])

        # Operating cost constraint
        @model_part.Constraint(self.time_steps)
        def eaf_operating_cost_constraint(b, t):
            return (
                b.eaf_operating_cost[t]
                == b.power_eaf[t] * model.electricity_price[t]
                + b.emission_eaf[t] * model.co2_price
                + b.lime_demand[t] * model.lime_price
            )

        return model_part


class ElectricVehicle(GenericStorage):
    """
    A class to represent an Electric Vehicle (EV) unit in an energy system model.

    The class encapsulates the parameters, variables, and constraints necessary to model
    the behavior of an EV, including charging and discharging, battery storage limits, availability
    profiles, and ramp rates.

    Inherits from GenericStorage and adds EV-specific functionality, such as availability profiles
    and predefined charging profiles.

    Attributes
    ----------
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

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
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
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        """
        Creates and returns a Pyomo Block for the Electric Vehicle (EV) component.

        Parameters
        ----------
        model : pyo.ConcreteModel
            A Pyomo ConcreteModel object representing the optimization model.

        Returns
        -------
        pyo.Block
            A Pyomo block representing the EV with variables and constraints.
        """

        # Call the parent class (GenericStorage) add_to_model method
        model_part = super().add_to_model(model, model_part)

        # Apply availability profile constraints if provided
        if self.availability_profile is not None:
            if not isinstance(self.availability_profile, pd.Series):
                raise TypeError("`availability_profile` must be a pandas Series.")
            if not all(t in self.availability_profile.index for t in self.time_steps):
                raise ValueError(
                    "All `time_steps` must be present in `availability_profile` index."
                )

            @model_part.Constraint(self.time_steps)
            def discharge_availability_constraint(b, t):
                availability = self.availability_profile[t]
                return b.discharge[t] <= availability * b.max_power_discharge

            @model_part.Constraint(self.time_steps)
            def charge_availability_constraint(b, t):
                availability = self.availability_profile[t]
                return b.charge[t] <= availability * b.max_power_charge

        # Apply predefined charging profile constraints if provided
        if self.charging_profile is not None:
            if not isinstance(self.charging_profile, pd.Series):
                raise TypeError("`charging_profile` must be a pandas Series.")
            if not all(t in self.charging_profile.index for t in self.time_steps):
                raise ValueError(
                    "All `time_steps` must be present in `charging_profile` index."
                )

            @model_part.Constraint(self.time_steps)
            def charging_profile_constraint(b, t):
                return b.charge[t] == self.charging_profile[t]

        return model_part


class HydrogenStorage(GenericStorage):
    """
    A class to represent a hydrogen storage unit in an energy system model.

    Inherits all the functionality from GenericStorage and can be extended in the future
    with hydrogen-specific constraints or attributes.

    Attributes
    ----------
        Inherits all attributes from the GenericStorage class.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Inherits from GenericStorage. Adds a hydrogen storage block to the Pyomo model,
            defining parameters, variables, and constraints.
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
        self, model: pyo.ConcreteModel, model_part: pyo.Block
    ) -> pyo.Block:
        # Call the parent class (GenericStorage) add_to_model method
        model_part = super().add_to_model(model, model_part)

        # add further constraints or variables specific to hydrogen storage here

        return model_part


class DRIStorage(GenericStorage):
    """
    A class to represent a Direct Reduced Iron (DRI) storage unit in an energy system model.

    Inherits all the functionality from GenericStorage and can be extended in the future
    with DRI-specific constraints or attributes.

    Attributes
    ----------
        Inherits all attributes from the GenericStorage class.

    Methods
    -------
        add_to_model(self, model: pyo.ConcreteModel, model_part: pyo.Block) -> pyo.Block:
            Inherits from GenericStorage. Adds a DRI storage block to the Pyomo model,
            defining parameters, variables, and constraints.
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
