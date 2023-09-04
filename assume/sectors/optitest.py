from assume.sectors.building import Building
from pyomo.environ import *
from pyomo.environ import SolverFactory
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# Define some parameters for 5 unique time_steps
time_steps = pd.date_range("2021-01-01", periods=5, freq="H")
heating_demand = [20.0, 21.0, 22.0, 23.0, 24.0]
cooling_demand = [15.0, 14.0, 13.0, 12.0, 11.0]
electricity_price = [0.12, 0.13, 0.14, 0.15, 0.16]

# Create unit list
unit_parameters = {
    "HeatPump": {
        "id": "heatpump1",
        "technology": "HeatPumpTechnology",
        "max_power": 25,
        "min_power": 10,
        "ramp_up": 1,
        "ramp_down": -1,
        "cop": 2,
    },
    "AirConditioner": {
        "id": "airconditioner1",
        "technology": "AirConditioner",
        "max_power": 20,
        "min_power": 2,
        "ramp_up": 2,
        "ramp_down": 2,
        "cooling_factor": 0.8,
    },
}

# Just the names of the unit classes as strings
unit_list = ["HeatPump", "AirConditioner"]  #'AirConditioner'

# Create Building object
building = Building(
    id="building1",
    unit_operator="Operator1",
    technology="mixed",
    bidding_strategies={},
    index=time_steps,
    node="bus_1",
    time_steps=time_steps,
    unit_list=unit_list,
    unit_parameters=unit_parameters,
    objective="minimize_cost",
    heating_demand=heating_demand,
    cooling_demand=cooling_demand,
    electricity_price=electricity_price,
)

#######################################################################################
solver = SolverFactory("gurobi")  # Replace 'gurobi' with your solver of choice
results = solver.solve(building.model, tee=True)  # , tee=True
# print(results)
building.model.display()
print("Debugging: Start printing the entire model")
# building.model.pprint()
print("Debugging: End printing the entire model")


# Check solver status and termination condition
if (results.solver.status == SolverStatus.ok) and (
    results.solver.termination_condition == TerminationCondition.optimal
):
    print("The model was solved optimally.")

    # Display the Objective Function Value
    objective_value = building.model.obj_rule()
    print(f"The value of the objective function is {objective_value}.")

    # Initialize an empty list for each column
    time_step_list = []
    aggregated_power_in_list = []
    heatpump_power_in_list = []
    airconditioner_power_in_list = []

    # Populate the lists with data
    for t in building.model.time_steps:
        time_step_list.append(
            time_steps[t]
        )  # Assuming 'time_steps' is your original timestamp list
        aggregated_power_in_list.append(building.model.aggregated_power_in[t].value)
        heatpump_power_in_list.append(
            building.model.HeatPump.power_in["HeatPump", t].value
        )
        airconditioner_power_in_list.append(
            building.model.AirConditioner.power_in["AirConditioner", t].value
        )

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Time Step": time_step_list,
            "Aggregated Power In": aggregated_power_in_list,
            "Power In (HeatPump)": heatpump_power_in_list,
            "Power In (AirConditioner)": airconditioner_power_in_list,
        }
    )

    # Display the DataFrame
    pd.set_option("display.expand_frame_repr", False)
    print(df)


elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("The model is infeasible.")

else:
    print("Solver Status: ", results.solver.status)
    print("Termination Condition: ", results.solver.termination_condition)

print("Building optimization test with 5 unique time steps completed.")
