# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd


def determine_optimal_operation(demand_side_agent):
    """
    Function to determine the optimal operation of a demand side agent without considering flexibility.
    """
    optimal_operation = demand_side_agent.run_optimization()
    return optimal_operation


def calculate_reference_operational_state(demand_side_agent):
    """
    Function to calculate the reference operational state of a demand side agent based on the objective without flexibility.

    Args:
    - demand_side_agent: The demand side agent object representing the plant or facility.

    Returns:
    - reference_operational_state: Dictionary or object containing the reference operational state parameters.
    """
    reference_operational_state = demand_side_agent.calculate_operational_state()
    return reference_operational_state

def determine_flexibility(reference_curve, tolerance):
    min_flexibility = {}
    max_flexibility = {}
    for technology, curve_values in reference_curve.items():
        min_curve = [max(0, value - tolerance) for value in curve_values]
        min_flexibility[technology] = {
            "min_negative": min_curve,
            "min_positive": [
                max(0, value + tolerance) for value in curve_values
            ],  # Positive flexibility
        }
        max_flexibility[technology] = {
            "max_negative": curve_values,
            "max_positive": [max(0, value - tolerance) for value in curve_values],
        }
    return min_flexibility, max_flexibility


def calculate_deviation(
    reference_state, min_flexibility, max_flexibility, objective_function
):
    deviations = {}

    # Calculate costs or profit margins under reference scenario
    reference_costs = objective_function(reference_state)

    # Calculate costs or profit margins under flexibility measures
    for tech, state_values in reference_state.items():
        deviations[tech] = {}
        for timestep, _ in enumerate(state_values):
            # Calculate costs or profit margins under flexibility measures
            min_flex_value = min_flexibility[tech][timestep]
            max_flex_value = max_flexibility[tech][timestep]
            min_deviation = objective_function({tech: min_flex_value})
            max_deviation = objective_function({tech: max_flex_value})
            # Determine deviation from reference
            deviations[tech][timestep] = {
                "min_deviation": min_deviation - reference_costs[timestep],
                "max_deviation": max_deviation - reference_costs[timestep],
            }

    return deviations


def organize_results(deviations, min_flexibility, max_flexibility):
    # Initialize dataframes for positive and negative flexibility
    pos_flex_df = pd.DataFrame(
        columns=[
            "Technology",
            "Time Step",
            "Flexibility Capacity",
            "Power Output Adjustment",
        ]
    )
    neg_flex_df = pd.DataFrame(
        columns=[
            "Technology",
            "Time Step",
            "Flexibility Capacity",
            "Power Output Adjustment",
        ]
    )

    # Iterate over each technology and each time step
    for tech, tech_devs in deviations.items():
        for timestep, dev_values in tech_devs.items():
            # Extract the deviations for the current time step
            min_deviation = dev_values["min_deviation"]
            max_deviation = dev_values["max_deviation"]

            # If positive flexibility exists (ramp down), add it to the positive flexibility dataframe
            if min_deviation < 0:
                pos_flex_df = pos_flex_df.append(
                    {
                        "Technology": tech,
                        "Time Step": timestep,
                        "Flexibility Capacity": min_flexibility[tech][timestep],
                        "Power Output Adjustment": min_deviation,
                    },
                    ignore_index=True,
                )

            # If negative flexibility exists (ramp up), add it to the negative flexibility dataframe
            if max_deviation > 0:
                neg_flex_df = neg_flex_df.append(
                    {
                        "Technology": tech,
                        "Time Step": timestep,
                        "Flexibility Capacity": max_flexibility[tech][timestep],
                        "Power Output Adjustment": max_deviation,
                    },
                    ignore_index=True,
                )

    return pos_flex_df, neg_flex_df


def integrate_flexibility(simulation_model, pos_flex_df, neg_flex_df):
    # Iterate over positive flexibility dataframe and adjust power output accordingly
    for _, row in pos_flex_df.iterrows():
        tech = row["Technology"]
        timestep = row["Time Step"]
        power_adjustment = row["Power Output Adjustment"]

        # Adjust power output for the corresponding technology at the specified time step
        simulation_model.adjust_power_output(tech, timestep, power_adjustment)

    # Iterate over negative flexibility dataframe and adjust power output accordingly
    for _, row in neg_flex_df.iterrows():
        tech = row["Technology"]
        timestep = row["Time Step"]
        power_adjustment = row["Power Output Adjustment"]

        # Adjust power output for the corresponding technology at the specified time step
        simulation_model.adjust_power_output(tech, timestep, power_adjustment)

    # Return the modified simulation model
    return simulation_model


def calculate_deviation(
    reference_costs, reference_profit, flexibility_costs, flexibility_profit
):
    # Calculate the deviation in costs and profit margins
    cost_deviation = flexibility_costs - reference_costs
    profit_deviation = flexibility_profit - reference_profit

    # Return the calculated deviations
    return cost_deviation, profit_deviation


def construct_power_input_dataframes(reference_power_input, flexibility_power_input):
    # Convert the dictionaries to Pandas Series
    reference_series = pd.Series(reference_power_input)
    flexibility_series = pd.Series(flexibility_power_input)

    # Construct DataFrames
    reference_df = pd.DataFrame(
        {
            "Time Step": reference_series.index,
            "Reference Power Input": reference_series.values,
        }
    )
    flexibility_df = pd.DataFrame(
        {
            "Time Step": flexibility_series.index,
            "Flexibility Power Input": flexibility_series.values,
        }
    )

    return reference_df, flexibility_df


def calculate_flexibility_capacity(min_flexibility, max_flexibility):
    positive_flexibility_capacity = {}
    negative_flexibility_capacity = {}

    for timestep in min_flexibility:
        min_flex = min_flexibility[timestep]
        max_flex = max_flexibility[timestep]

        if min_flex < 0:
            negative_flexibility_capacity[timestep] = abs(min_flex)
        else:
            negative_flexibility_capacity[timestep] = 0

        if max_flex > 0:
            positive_flexibility_capacity[timestep] = max_flex
        else:
            positive_flexibility_capacity[timestep] = 0

    return positive_flexibility_capacity, negative_flexibility_capacity


def integrate_flexibility_measures(
    current_state,
    positive_flexibility_capacity,
    negative_flexibility_capacity,
    flexibility_objective,
):
    updated_state = {}

    # Adjust operational state based on flexibility objectives
    for tech, state_values in current_state.items():
        updated_state[tech] = {}
        for timestep, value in enumerate(state_values):
            if flexibility_objective[tech][timestep] == "minimize":
                # If objective is to minimize, decrease operational value by negative flexibility capacity
                updated_value = max(value - negative_flexibility_capacity[timestep], 0)
            elif flexibility_objective[tech][timestep] == "maximize":
                # If objective is to maximize, increase operational value by positive flexibility capacity
                updated_value = value + positive_flexibility_capacity[timestep]
            else:
                # If no specific objective, keep the operational value unchanged
                updated_value = value

            updated_state[tech][timestep] = updated_value

    return updated_state


def generate_power_input_dataframes(reference_state, updated_state):
    # Create DataFrame for reference operation
    reference_df = pd.DataFrame(reference_state)
    reference_df.index.name = "Time Step"
    reference_df.columns = [
        f"Reference_{tech}_Power_Input" for tech in reference_df.columns
    ]

    # Create DataFrame for operation with flexibility measures applied
    updated_df = pd.DataFrame(updated_state)
    updated_df.index.name = "Time Step"
    updated_df.columns = [f"Updated_{tech}_Power_Input" for tech in updated_df.columns]

    return reference_df, updated_df


def integrate_flexibility_measures(plant, min_flexibility, max_flexibility):
    """
    Integrate flexibility measures into the simulation by adjusting the operation of the plant.

    Parameters:
        plant: SteelPlant or similar object representing the plant.
        min_flexibility: Dictionary containing minimum flexibility measures for each technology.
        max_flexibility: Dictionary containing maximum flexibility measures for each technology.
    """
    # Iterate over time steps and adjust plant operation based on flexibility measures
    for t in plant.model.time_steps:
        for tech, component in plant.components.items():
            # Adjust operation based on minimum flexibility
            if tech in min_flexibility and t in min_flexibility[tech]:
                component.adjust_operation(min_flexibility[tech][t], direction="min")
            # Adjust operation based on maximum flexibility
            if tech in max_flexibility and t in max_flexibility[tech]:
                component.adjust_operation(max_flexibility[tech][t], direction="max")
