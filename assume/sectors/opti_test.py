from assume.units.heatpump import HeatPump
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
import logging
logger = logging.getLogger(__name__)


def run_optimization(heat_pump):
    model = ConcreteModel()
    # Create the optimization model
    heat_pump.create_model(model=heat_pump.model)

    # Solve the optimization problem using a solver
    # solver = SolverFactory('glpk')  # You can replace 'glpk' with any other solver supported by Pyomo
    # result = solver.solve(heat_pump.model)

    pyo.SolverFactory('gurobi').solve(heat_pump.model)  # .write()
    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(heat_pump.model)

    heat_pump.model.solutions.store_to(result)
    print(result)

    heat_pump.model.display()

    # Check if the optimization was successful
    if result.solver.termination_condition == TerminationCondition.optimal:
        logger.info("Optimization successful.")
        # Print the optimal power values
        for h in heat_pump.model.heat_pumps:
            for t in heat_pump.model.time_steps:
                power_out = heat_pump.model.power_out[h, t].value
                logger.info(f"Heat Pump {h} power output at time step {t}: {power_out:.2f}")
    else:
        logger.warning("Optimization did not converge to an optimal solution.")

    # Additional code to print parameter values, objective, and variable values
    print("Parameters before solving:")
    print("max_power:", heat_pump.model.max_power())
    print("min_power:", heat_pump.model.min_power())

    # Print the values of electricity_prices for each time step
    print("Electricity Prices:")
    for t in heat_pump.model.time_steps:
        print(f"Time step {t}: {heat_pump.model.electricity_prices[t]}")

    # Print the values of load_profile for each time step
    print("Load Profile:")
    for t in heat_pump.model.time_steps:
        print(f"Time step {t}: {heat_pump.model.load_profile[t]}")

    # Print objective value
    print("Objective Value:", heat_pump.model.obj())

    # Print variable values
    print("Variable Values:")
    for h in heat_pump.model.heat_pumps:
        for t in heat_pump.model.time_steps:
            power_out = heat_pump.model.power_out[h, t].value
            print(f"Heat Pump {h} power output at time step {t}: {power_out:.2f}")

    # After solving the model
    print("Parameters after solving:")
    print("max_power:", heat_pump.model.max_power())
    print("min_power:", heat_pump.model.min_power())

    # Print the values of electricity_prices after solving for each time step
    print("Electricity Prices after solving:")
    for t in heat_pump.model.time_steps:
        print(f"Time step {t}: {heat_pump.model.electricity_prices[t]}")

    # Print the values of load_profile after solving for each time step
    print("Load Profile after solving:")
    for t in heat_pump.model.time_steps:
        print(f"Time step {t}: {heat_pump.model.load_profile[t]}")


if __name__ == "__main__":
    # Create HeatPump instance and set parameters with data
    heat_pump = HeatPump(
        id="HP1",
        unit_operator="Operator1",
        technology="HeatPumpTechnology",
        sector="Sector1",
        bidding_strategies={},
        index=pd.DatetimeIndex(["2023-07-31 00:00:00", "2023-07-31 01:00:00", "2023-07-31 02:00:00"]),  # Specify your time steps here
    )
    heat_pump.add_model_parameters(
        time_steps=pd.DatetimeIndex(["2023-07-31 00:00:00", "2023-07-31 01:00:00", "2023-07-31 02:00:00"]),
        max_power=150,  # Replace with actual max_power data
        min_power=0,  # Replace with actual min_power data
        heat_pump_type="Type1",
        source_temp=pd.Series([15, 20, 20], index=pd.DatetimeIndex(["2023-07-31 00:00:00", "2023-07-31 01:00:00",
                                                                    "2023-07-31 02:00:00"])),
        sink_temp=pd.Series([35, 30, 30], index=pd.DatetimeIndex(["2023-07-31 00:00:00", "2023-07-31 01:00:00",
                                                                  "2023-07-31 02:00:00"])),
        electricity_prices=pd.Series([40, 45, 30], index=pd.DatetimeIndex(["2023-07-31 00:00:00", "2023-07-31 01:00:00",
                                                                       "2023-07-31 02:00:00"])),
        load_profile=pd.Series([0, 80, 100], index=pd.DatetimeIndex(["2023-07-31 00:00:00", "2023-07-31 01:00:00",
                                                                     "2023-07-31 02:00:00"])),
    )

    # Run the optimization
    run_optimization(heat_pump)