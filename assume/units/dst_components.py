# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyomo.environ as pyo


def create_electrolyser(
    model,
    rated_power,
    min_power,
    ramp_up,
    ramp_down,
    min_operating_time,
    min_down_time,
    efficiency,
    time_steps,
    **kwargs,
):
    """
    Represents an electrolyser unit used for hydrogen production through electrolysis.

    Args:
        model (pyomo.ConcreteModel): The Pyomo model where the electrolyser unit will be added.
        id (str): Identifier for the electrolyser unit.
        rated_power (float): The rated power capacity of the electrolyser (in MW).
        min_power (float): The minimum power required for operation (in kW).
        ramp_up (float): The maximum rate at which the electrolyser can increase its power output (in MW/hr).
        ramp_down (float): The maximum rate at which the electrolyser can decrease its power output (in MW/hr).
        min_operating_time (float): The minimum duration the electrolyser must operate continuously (in hours).
        min_down_time (float): The minimum downtime required between operating cycles (in hours).
        efficiency (float): The efficiency of the electrolysis process.

    Constraints:
        power_upper_bound: Ensures the power input to the electrolyser does not exceed the rated power capacity.
        power_lower_bound: Ensures the power input to the electrolyser does not fall below the minimum required power.
        ramp_up_constraint: Limits the rate at which the power input to the electrolyser can increase.
        ramp_down_constraint: Limits the rate at which the power input to the electrolyser can decrease.
        min_operating_time_electrolyser_constraint: Ensures the electrolyser operates continuously for a minimum duration.
        min_downtime_electrolyser_constraint: Ensures the electrolyser has a minimum downtime between operating cycles.
        efficiency_constraint: Relates the power input to the hydrogen output based on the efficiency of the electrolysis process.
        operating_cost_with_el_price: Calculates the operating cost of the electrolyser based on the electricity price.

    """
    model_part = pyo.Block()
    model_part.rated_power = pyo.Param(initialize=rated_power)
    model_part.min_power = pyo.Param(initialize=min_power)
    model_part.ramp_up = pyo.Param(initialize=ramp_up)
    model_part.ramp_down = pyo.Param(initialize=ramp_down)
    model_part.min_operating_time = pyo.Param(initialize=min_operating_time)
    model_part.min_down_time = pyo.Param(initialize=min_down_time)
    model_part.efficiency = pyo.Param(initialize=efficiency)

    # define variables
    model_part.power_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.hydrogen_out = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.electrolyser_operating_cost = pyo.Var(
        time_steps, within=pyo.NonNegativeReals
    )

    # Power bounds constraints
    @model_part.Constraint(time_steps)
    def power_upper_bound(b, t):
        """
        Ensures that the power input to the electrolyser does not exceed its rated power capacity.

        """
        return b.power_in[t] <= b.rated_power

    @model_part.Constraint(time_steps)
    def power_lower_bound(b, t):
        """
        Ensures that the power input to the electrolyser does not fall below the minimum required power.

        """
        return b.power_in[t] >= b.min_power

    # Ramp-up constraint
    @model_part.Constraint(time_steps)
    def ramp_up_constraint(b, t):
        """
        Limits the rate at which the power input to the electrolyser can increase.

        """
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_in[t] - b.power_in[t - 1] <= b.ramp_up

    # Ramp-down constraint
    @model_part.Constraint(time_steps)
    def ramp_down_constraint(b, t):
        """
        Limits the rate at which the power input to the electrolyser can decrease.

        """
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_in[t - 1] - b.power_in[t] <= b.ramp_down

    @model_part.Constraint(time_steps)
    def min_operating_time_electrolyser_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip

        delta_t = t - (t - 1)
        min_operating_time_units = int(min_operating_time / delta_t)

        # Check for valid indexing
        if t < min_operating_time_units:
            if t - min_operating_time_units + 1 < 0:
                raise ValueError(
                    f"Invalid min_operating_time: {min_operating_time} exceeds available time steps. "
                    "Ensure min_operating_time is compatible with the available time steps."
                )
            return (
                sum(
                    b.power_in[i]
                    for i in range(t - min_operating_time_units + 1, t + 1)
                )
                >= min_operating_time_units * b.power_in[t]
            )
        else:
            return (
                sum(
                    b.power_in[i]
                    for i in range(t - min_operating_time_units + 1, t + 1)
                )
                >= min_operating_time_units * b.power_in[t]
            )

    @model_part.Constraint(time_steps)
    def min_downtime_electrolyser_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip

        delta_t = t - (t - 1)
        min_downtime_units = int(min_down_time / delta_t)

        # Check for valid indexing
        if t < min_downtime_units:
            if t - min_downtime_units < 0:
                raise ValueError(
                    f"Invalid min_down_time: {min_down_time} exceeds available time steps. "
                    "Ensure min_down_time is compatible with the available time steps."
                )
            return (
                sum(b.power_in[t - i] for i in range(min_downtime_units))
                >= min_downtime_units * b.power_in[t]
            )

        else:
            return (
                sum(b.hydrogen_out[t - i] for i in range(min_downtime_units))
                >= min_downtime_units * b.power_in[t]
            )

    # Efficiency constraint
    @model_part.Constraint(time_steps)
    def power_in_equation(b, t):
        """
        Relates the power input to the hydrogen output based on the efficiency of the electrolysis process.

        """
        return b.power_in[t] == b.hydrogen_out[t] / b.efficiency

    @model_part.Constraint(time_steps)
    def operating_cost_with_el_price(b, t):
        """
        Calculates the operating cost of the electrolyser based on the electricity price.

        """
        return (
            b.electrolyser_operating_cost[t]
            == b.power_in[t] * model.electricity_price[t]
        )

    return model_part


def create_driplant(
    model,
    specific_hydrogen_consumption,
    specific_natural_gas_consumption,
    specific_electricity_consumption,
    specific_iron_ore_consumption,
    rated_power,
    min_power,
    fuel_type,
    ramp_up,
    ramp_down,
    min_operating_time,
    min_down_time,
    time_steps,
    **kwargs,
):
    """
    Args:
        model (pyomo.ConcreteModel): The Pyomo model where the DRI plant will be added.
        id (str): Identifier for the DRI plant.
        specific_hydrogen_consumption (float): The specific hydrogen consumption of the DRI plant (in MWh per ton of DRI).
        specific_natural_gas_consumption (float): The specific natural gas consumption of the DRI plant (in MWh per ton of DRI).
        specific_electricity_consumption (float): The specific electricity consumption of the DRI plant (in MWh per ton of DRI).
        specific_iron_ore_consumption (float): The specific iron ore consumption of the DRI plant (in ton per ton of DRI).
        rated_power (float): The rated power capacity of the DRI plant (in MW).
        min_power (float): The minimum power required for operation (in MW).
        fuel_type (str): The type of fuel used by the DRI plant.
        ramp_up (float): The maximum rate at which the DRI plant can increase its power output (in MW/hr).
        ramp_down (float): The maximum rate at which the DRI plant can decrease its power output (in MW/hr).
        min_operating_time (float): The minimum duration the DRI plant must operate continuously (in hours).
        min_down_time (float): The minimum downtime required between operating cycles (in hours).

    Constraints:
        dri_power_lower_bound: Ensures that the power input to the DRI plant does not fall below the minimum power requirement.
        dri_power_upper_bound: Ensures that the power input to the DRI plant does not exceed the rated power capacity.
        dri_output_constraint: Relates the DRI output to the fuel inputs based on the specific consumption rates.
        dri_output_electricity_constraint: Relates the electricity input to the DRI output based on the specific electricity consumption.
        iron_ore_constraint: Relates the iron ore input to the DRI output based on the specific iron ore consumption.
        ramp_up_dri_constraint: Limits the rate at which the power input to the DRI plant can increase.
        ramp_down_dri_constraint: Limits the rate at which the power input to the DRI plant can decrease.
        min_operating_time_dri_constraint: Ensures that the DRI plant operates continuously for a minimum duration.
        min_down_time_dri_constraint: Ensures that the DRI plant has a minimum downtime between operating cycles.
        dri_operating_cost_constraint: Calculates the operating cost of the DRI plant based on inputs and prices.
    """
    model_part = pyo.Block()
    model_part.specific_hydrogen_consumption = pyo.Param(
        initialize=specific_hydrogen_consumption
    )
    model_part.specific_natural_gas_consumption = pyo.Param(
        initialize=specific_natural_gas_consumption
    )
    model_part.specific_electricity_consumption_dri = pyo.Param(
        initialize=specific_electricity_consumption
    )
    model_part.specific_iron_ore_consumption = pyo.Param(
        initialize=specific_iron_ore_consumption
    )
    # Flexibility parameters
    model_part.min_power_dri = pyo.Param(initialize=min_power)
    model_part.rated_power_dri = pyo.Param(initialize=rated_power)
    model_part.ramp_up_dri = pyo.Param(initialize=ramp_up)
    model_part.ramp_down_dri = pyo.Param(initialize=ramp_down)
    model_part.min_operating_time_dri = pyo.Param(initialize=min_operating_time)
    model_part.min_down_time_dri = pyo.Param(initialize=min_down_time)

    # define variables
    model_part.iron_ore_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.natural_gas_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.dri_operating_cost = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.hydrogen_in = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.dri_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.power_dri = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # define constraints
    @model_part.Constraint(time_steps)
    def dri_power_lower_bound(b, t):
        return b.power_dri[t] >= b.min_power_dri

    @model_part.Constraint(time_steps)
    def dri_power_upper_bound(b, t):
        return b.power_dri[t] <= b.rated_power_dri

    @model_part.Constraint(time_steps)
    def dri_output_constraint(b, t):
        if fuel_type == "hydrogen":
            return b.dri_output[t] == b.hydrogen_in[t] / b.specific_hydrogen_consumption
        elif fuel_type == "natural_gas":
            return (
                b.dri_output[t]
                == b.natural_gas_in[t] / b.specific_natural_gas_consumption
            )
        elif fuel_type == "both":
            return b.dri_output[t] == (
                b.hydrogen_in[t] / b.specific_hydrogen_consumption
            ) + (b.natural_gas_in[t] / b.specific_natural_gas_consumption)

    @model_part.Constraint(time_steps)
    def dri_output_electricity_constraint(b, t):
        return (
            b.power_dri[t] == b.dri_output[t] * b.specific_electricity_consumption_dri
        )

    @model_part.Constraint(time_steps)
    def iron_ore_constraint(b, t):
        return b.iron_ore_in[t] == b.dri_output[t] * b.specific_iron_ore_consumption

    # Flexibility constraints
    @model_part.Constraint(time_steps)
    def ramp_up_dri_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        else:
            return b.power_dri[t] - b.power_dri[t - 1] <= b.ramp_up_dri

    @model_part.Constraint(time_steps)
    def ramp_down_dri_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        else:
            return b.power_dri[t - 1] - b.power_dri[t] <= b.ramp_down_dri

    @model_part.Constraint(time_steps)
    def min_operating_time_dri_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip

        # Calculate the time step duration dynamically
        delta_t = t - (t - 1)
        # Convert minimum operating time to the time unit of your choice
        min_operating_time_units = int(min_operating_time / delta_t)

        # Check for valid indexing
        if t < min_operating_time_units:
            if t - min_operating_time_units + 1 < 0:
                raise ValueError(
                    f"Invalid min_operating_time: {min_operating_time} exceeds available time steps. "
                    "Ensure min_operating_time is compatible with the available time steps."
                )
            return (
                sum(
                    b.power_dri[i]
                    for i in range(t - min_operating_time_units + 1, t + 1)
                )
                >= min_operating_time_units * b.power_dri[t]
            )
        else:
            return (
                sum(
                    b.power_dri[i]
                    for i in range(t - min_operating_time_units + 1, t + 1)
                )
                >= min_operating_time_units * b.power_dri[t]
            )

    @model_part.Constraint(time_steps)
    def min_down_time_dri_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip

        # Calculate the time step duration dynamically
        delta_t = t - (t - 1)
        # Convert minimum downtime to the time unit of your choice
        min_downtime_units = int(min_down_time / delta_t)

        # Check for valid indexing
        if t < min_downtime_units:
            if t - min_downtime_units < 0:
                raise ValueError(
                    f"Invalid min_down_time: {min_down_time} exceeds available time steps. "
                    "Ensure min_down_time is compatible with the available time steps."
                )
            return (
                sum(b.power_dri[t - i] for i in range(min_downtime_units))
                >= min_downtime_units * b.power_dri[t]
            )

        else:
            return (
                sum(b.dri_output[t - i] for i in range(min_downtime_units))
                >= min_downtime_units * b.power_dri[t]
            )

    # Operational cost
    @model_part.Constraint(time_steps)
    def dri_operating_cost_constraint(b, t):
        # This constraint defines the steel output based on inputs and efficiency
        return (
            b.dri_operating_cost[t]
            == b.natural_gas_in[t] * model.natural_gas_price[t]
            + b.power_dri[t] * model.electricity_price[t]
            + b.iron_ore_in[t] * model.iron_ore_price
        )

    return model_part


def create_electric_arc_furnance(
    model: pyo.ConcreteModel,
    rated_power: float,
    min_power: float,
    specific_electricity_consumption: float,
    specific_dri_demand: float,
    specific_lime_demand: float,
    ramp_up: float,
    ramp_down: float,
    min_operating_time: int,
    min_down_time: int,
    time_steps: list,
    *args,
    **kwargs,
):
    """
    Adds the Electric Arc Furnace (EAF) to the Pyomo model.

        Represents an Electric Arc Furnace (EAF) in a steel production process.

    Args:
        rated_power: The rated power capacity of the electric arc furnace (in MW).
        min_power: The minimum power requirement of the electric arc furnace (in MW).
        specific_electricity_consumption: The specific electricity consumption of the electric arc furnace (in MWh per ton of steel produced).
        specific_dri_demand: The specific demand for Direct Reduced Iron (DRI) in the electric arc furnace (in tons per ton of steel produced).
        specific_lime_demand: The specific demand for lime in the electric arc furnace (in tons per ton of steel produced).
        ramp_up: The ramp-up rate of the electric arc furnace (in MW per hour).
        ramp_down: The ramp-down rate of the electric arc furnace (in MW per hour).
        min_operating_time: The minimum operating time requirement for the electric arc furnace (in hours).
        min_down_time: The minimum downtime requirement for the electric arc furnace (in hours).
        time_steps (list): List of time steps for which the model will be defined.

    Constraints:
        electricity_input_upper_bound: Limits the electricity input to the rated power capacity.
        electricity_input_lower_bound: Ensures the electricity input does not fall below the minimum power requirement.
        steel_output_dri_relation: Defines the steel output based on the DRI input and efficiency.
        steel_output_power_relation: Defines the steel output based on the electricity input and efficiency.
        eaf_lime_demand: Defines the lime demand based on the steel output and specific lime demand.
        eaf_co2_emission: Defines the CO2 emissions based on the lime demand and CO2 factor.
        ramp_up_eaf_constraint: Limits the rate at which the electricity input can increase.
        ramp_down_eaf_constraint: Limits the rate at which the electricity input can decrease.
        min_operating_time_eaf_constraint: Ensures the EAF operates continuously for a minimum duration.
        min_down_time_eaf_constraint: Ensures the EAF has a minimum downtime between operating cycles.
        eaf_operating_cost_cosntraint: Calculates the operating cost of the EAF based on inputs and prices.
    """
    model_part = pyo.Block()

    # create parameters
    model_part.rated_power_eaf = pyo.Param(initialize=rated_power)
    model_part.min_power_eaf = pyo.Param(initialize=min_power)
    model_part.specific_electricity_consumption_eaf = pyo.Param(
        initialize=specific_electricity_consumption
    )
    model_part.specific_dri_demand = pyo.Param(initialize=specific_dri_demand)
    model_part.specific_lime_demand = pyo.Param(initialize=specific_lime_demand)
    # Flexibility parameters
    model_part.ramp_up_eaf = pyo.Param(initialize=ramp_up)
    model_part.ramp_down_eaf = pyo.Param(initialize=ramp_down)
    model_part.min_operating_time_eaf = pyo.Param(initialize=min_operating_time)
    model_part.min_down_time_eaf = pyo.Param(initialize=min_down_time)

    # define pyomo variables
    model_part.power_eaf = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.dri_input = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.steel_output = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.eaf_operating_cost = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.emission_eaf = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.lime_demand = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # Power bounds constraints
    @model_part.Constraint(time_steps)
    def electricity_input_upper_bound(b, t):
        return b.power_eaf[t] <= b.rated_power_eaf

    @model_part.Constraint(time_steps)
    def electricity_input_lower_bound(b, t):
        return b.power_eaf[t] >= b.min_power_eaf

    # This constraint defines the steel output based on inputs and efficiency
    @model_part.Constraint(time_steps)
    def steel_output_dri_relation(b, t):
        return b.steel_output[t] == b.dri_input[t] / b.specific_dri_demand

    # This constraint defines the steel output based on inputs and efficiency
    @model_part.Constraint(time_steps)
    def steel_output_power_relation(b, t):
        return (
            b.power_eaf[t] == b.steel_output[t] * b.specific_electricity_consumption_eaf
        )

    @model_part.Constraint(time_steps)
    def eaf_lime_demand(b, t):
        return b.lime_demand[t] == b.steel_output[t] * b.specific_lime_demand

    @model_part.Constraint(time_steps)
    def eaf_co2_emission(b, t):
        return b.emission_eaf[t] == b.lime_demand[t] * model.lime_co2_factor

    # Flexibility constraints
    @model_part.Constraint(time_steps)
    def ramp_up_eaf_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_eaf[t] - b.power_eaf[t - 1] <= b.ramp_up_eaf

    @model_part.Constraint(time_steps)
    def ramp_down_eaf_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip
        return b.power_eaf[t - 1] - b.power_eaf[t] <= b.ramp_down_eaf

    @model_part.Constraint(time_steps)
    def min_operating_time_eaf_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip  # No constraint for the first time step

        # Calculate the time step duration dynamically
        delta_t = t - (t - 1)
        # Convert minimum operating time to the time unit of your choice
        min_operating_time_units = int(min_operating_time / delta_t)

        # Check for valid indexing
        if t < min_operating_time_units:
            if t - min_operating_time_units + 1 < 0:
                raise ValueError(
                    f"Invalid min_operating_time: {min_operating_time} exceeds available time steps. "
                    "Ensure min_operating_time is compatible with the available time steps."
                )
            return (
                sum(
                    b.steel_output[i]
                    for i in range(t - min_operating_time_units + 1, t + 1)
                )
                >= min_operating_time_units * b.steel_output[t]
            )
        else:
            return (
                sum(
                    b.steel_output[i]
                    for i in range(t - min_operating_time_units + 1, t + 1)
                )
                >= min_operating_time_units * b.steel_output[t]
            )

    @model_part.Constraint(time_steps)
    def min_down_time_eaf_constraint(b, t):
        if t == 0:
            return pyo.Constraint.Skip  # No constraint for the first time step

        # Calculate the time step duration dynamically
        delta_t = t - (t - 1)
        # Convert minimum downtime to the time unit of your choice
        min_downtime_units = int(min_down_time / delta_t)

        # Check for valid indexing
        if t < min_downtime_units:
            if t - min_downtime_units < 0:
                raise ValueError(
                    f"Invalid min_down_time: {min_down_time} exceeds available time steps. "
                    "Ensure min_down_time is compatible with the available time steps."
                )
            return (
                sum(b.steel_output[t - i] for i in range(min_downtime_units))
                >= min_downtime_units * b.steel_output[t]
            )
        else:
            return (
                sum(b.steel_output[t - i] for i in range(min_downtime_units))
                >= min_downtime_units * b.steel_output[t]
            )

    # Operational cost
    @model_part.Constraint(time_steps)
    def eaf_operating_cost_cosntraint(b, t):
        # This constraint defines the steel output based on inputs and efficiency
        return (
            b.eaf_operating_cost[t]
            == b.power_eaf[t] * model.electricity_price[t]
            + b.emission_eaf[t] * model.co2_price
            + b.lime_demand[t] * model.lime_price
        )

    return model_part


def create_hydrogen_storage(
    model,
    max_capacity,
    min_capacity,
    initial_soc,
    storage_loss_rate,
    charge_loss_rate,
    discharge_loss_rate,
    time_steps,
    **kwargs,
):
    """
    Represents a generic energy storage unit.

    Args:
        model: A Pyomo ConcreteModel object representing the optimization model.
        id (str): A unique identifier for the storage unit.
        max_capacity (float): The maximum storage capacity of the unit.
        min_capacity (float): The minimum storage capacity of the unit.
        initial_soc (float): The initial state of charge (SOC) of the storage unit.
        storage_loss_rate (float): The rate of energy loss due to storage inefficiency.
        charge_loss_rate (float): The rate of energy loss during charging.
        discharge_loss_rate (float): The rate of energy loss during discharging.
        **kwargs: Additional keyword arguments.

    Constraints:
        storage_min_capacity_constraint: Ensures the SOC of the storage unit stays above the minimum capacity.
        storage_max_capacity_constraint: Ensures the SOC of the storage unit stays below the maximum capacity.
        energy_in_max_capacity_constraint: Limits the charging of the storage unit to its maximum capacity.
        energy_out_max_capacity_constraint: Limits the discharging of the storage unit to its maximum capacity.
        energy_in_uniformity_constraint: Ensures uniformity in charging the storage unit.
        energy_out_uniformity_constraint: Ensures uniformity in discharging the storage unit.
        storage_capacity_change_constraint: Defines the change in SOC of the storage unit over time.
    """
    model_part = pyo.Block()
    # define parameters
    model_part.max_capacity = pyo.Param(initialize=max_capacity)
    model_part.min_capacity = pyo.Param(initialize=min_capacity)
    model_part.initial_soc = pyo.Param(initialize=initial_soc)
    model_part.storage_loss_rate = pyo.Param(initialize=storage_loss_rate)
    model_part.charge_loss_rate = pyo.Param(initialize=charge_loss_rate)
    model_part.discharge_loss_rate = pyo.Param(initialize=discharge_loss_rate)

    # define variables
    model_part.soc = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.uniformity_indicator = pyo.Var(time_steps, within=pyo.Binary)
    model_part.charge = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.discharge = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # define constraints
    """
    Ensures the SOC of the storage unit stays above the minimum capacity.
    """

    @model_part.Constraint(time_steps)
    def storage_min_capacity_constraint(b, t):
        return b.soc[t] >= b.min_capacity

    """
    Ensures the SOC of the storage unit stays below the maximum capacity.
    """

    @model_part.Constraint(time_steps)
    def storage_max_capacity_constraint(b, t):
        return b.soc[t] <= b.max_capacity

    """
    Limits the charging of the storage unit to its maximum capacity.
    """

    @model_part.Constraint(time_steps)
    def energy_in_max_capacity_constraint(b, t):
        return b.charge[t] <= b.max_capacity

    """
    Limits the discharging of the storage unit to its maximum capacity.
    """

    @model_part.Constraint(time_steps)
    def energy_out_max_capacity_constraint(b, t):
        return b.discharge[t] <= b.max_capacity

    """
    Ensures uniformity in charging the storage unit.
    """

    @model_part.Constraint(time_steps)
    def energy_in_uniformity_constraint(b, t):
        return b.charge[t] <= b.max_capacity * b.uniformity_indicator[t]

    """
    Ensures uniformity in discharging the storage unit.
    """

    @model_part.Constraint(time_steps)
    def energy_out_uniformity_constraint(b, t):
        return b.discharge[t] <= b.max_capacity * (1 - b.uniformity_indicator[t])

    """
    Defines the change in SOC of the storage unit over time.
    """

    @model_part.Constraint(time_steps)
    def storage_capacity_change_constraint(b, t):
        return b.soc[t] == (
            ((b.soc[t - 1] if t > 0 else b.initial_soc) * (1 - b.storage_loss_rate))
            + ((1 - b.charge_loss_rate) * b.charge[t])
            - ((1 + b.discharge_loss_rate) * b.discharge[t])
        )

    return model_part


def create_dristorage(
    model,
    max_capacity,
    min_capacity,
    initial_soc,
    storage_loss_rate,
    charge_loss_rate,
    discharge_loss_rate,
    time_steps,
    **kwargs,
):
    """
    Represents a Direct Reduced Iron (DRI) storage unit.

    Args:
        model: A Pyomo ConcreteModel object representing the optimization model.
        id: A unique identifier for the DRI storage unit.
        max_capacity: The maximum capacity of the DRI storage unit.
        min_capacity: The minimum capacity of the DRI storage unit.
        initial_soc: The initial state of charge (SOC) of the DRI storage unit.
        storage_loss_rate: The rate of DRI loss due to storage over time.
        charge_loss_rate: The rate of DRI loss during charging.
        discharge_loss_rate: The rate of DRI loss during discharging.

    Constraints:
        storage_min_capacity_dri_constraint: Ensures the SOC of the DRI storage unit stays above the minimum capacity.
        storage_max_capacity_dri_constraint: Ensures the SOC of the DRI storage unit stays below the maximum capacity.
        energy_in_max_capacity_dri_constraint: Limits the charging of the DRI storage unit to its maximum capacity.
        energy_out_max_capacity_dri_constraint: Limits the discharging of the DRI storage unit to its maximum capacity.
        energy_in_uniformity_dri_constraint: Ensures uniformity in charging the DRI storage unit.
        energy_out_uniformity_dri_constraint: Ensures uniformity in discharging the DRI storage unit.
        storage_capacity_change_dri_constraint: Defines the change in SOC of the DRI storage unit over time.
    """
    model_part = pyo.Block()
    model_part.max_capacity_dri = pyo.Param(initialize=max_capacity)
    model_part.min_capacity_dri = pyo.Param(initialize=min_capacity)
    model_part.initial_soc_dri = pyo.Param(initialize=initial_soc)
    model_part.storage_loss_rate_dri = pyo.Param(initialize=storage_loss_rate)
    model_part.charge_loss_rate_dri = pyo.Param(initialize=charge_loss_rate)
    model_part.discharge_loss_rate_dri = pyo.Param(initialize=discharge_loss_rate)

    # define variables
    model_part.soc_dri = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.uniformity_indicator_dri = pyo.Var(time_steps, within=pyo.Binary)

    # Define the variables for power and hydrogen
    model_part.charge_dri = pyo.Var(time_steps, within=pyo.NonNegativeReals)
    model_part.discharge_dri = pyo.Var(time_steps, within=pyo.NonNegativeReals)

    # define constraints

    @model_part.Constraint(time_steps)
    def storage_min_capacity_dri_constraint(b, t):
        """
        Ensures the SOC of the DRI storage unit stays above the minimum capacity.
        """
        return b.soc_dri[t] >= b.min_capacity_dri

    @model_part.Constraint(time_steps)
    def storage_max_capacity_dri_constraint(b, t):
        """
        Ensures the SOC of the DRI storage unit stays below the maximum capacity.
        """
        return b.soc_dri[t] <= b.max_capacity_dri

    @model_part.Constraint(time_steps)
    def energy_in_max_capacity_dri_constraint(b, t):
        """
        Limits the charging of the DRI storage unit to its maximum capacity.
        """
        return b.charge_dri[t] <= b.max_capacity_dri

    @model_part.Constraint(time_steps)
    def energy_out_max_capacity_dri_constraint(b, t):
        """
        Limits the discharging of the DRI storage unit to its maximum capacity.
        """
        return b.discharge_dri[t] <= b.max_capacity_dri

    @model_part.Constraint(time_steps)
    def energy_in_uniformity_dri_constraint(b, t):
        """
        Ensures uniformity in charging the DRI storage unit.
        """
        return b.charge_dri[t] <= b.max_capacity_dri * b.uniformity_indicator_dri[t]

    @model_part.Constraint(time_steps)
    def energy_out_uniformity_dri_constraint(b, t):
        """
        Ensures uniformity in discharging the DRI storage unit.
        """
        return b.discharge_dri[t] <= b.max_capacity_dri * (
            1 - b.uniformity_indicator_dri[t]
        )

    @model_part.Constraint(time_steps)
    def storage_capacity_change_dri_constraint(b, t):
        """
        Defines the change in SOC of the DRI storage unit over time.
        """
        return b.soc_dri[t] == (
            (
                (b.soc_dri[t - 1] if t > 0 else b.initial_soc_dri)
                * (1 - b.storage_loss_rate_dri)
            )
            + ((1 - b.charge_loss_rate_dri) * b.charge_dri[t])
            - ((1 + b.discharge_loss_rate_dri) * b.discharge_dri[t])
        )

    return model_part
