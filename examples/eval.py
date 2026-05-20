# packages needed: seaborn, plotly, kaleido


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from assume import World
from assume.scenario.loader_csv import load_scenario_folder

# assume module imports
#import examples.examples as examples




db_uri = "postgresql://assume:assume@localhost:5432/assume"
db = create_engine(db_uri)

input_path = "examples/inputs/"
scenario = "example_01a"
study_case = "base"

query = f"SELECT * FROM market_orders where simulation = '{scenario}_{study_case}'"
market_orders_df = pd.read_sql(query, db)
print("orders", market_orders_df.columns)


orders = market_orders_df[["start_time", "unit_id", "accepted_price", "accepted_volume", "price", "volume"]]

orders = orders.groupby(["start_time"])

for key, time_group in orders:
    print(time_group)

    ###### separate supply and demand ######
    demand = time_group[time_group["unit_id"] == "demand_EOM"]
    market_price = time_group.iloc[0]["accepted_price"]
    supply = time_group[time_group["unit_id"] != "demand_EOM"]
    supply_offers = supply[["price", "volume", "accepted_price", "accepted_volume"]].values

    ###### get aggregated supply per price ######
    print(supply_offers)
    supply_vol = []
    for offer in supply_offers[:, :2]:
        # offer = price, volume
        if len(supply_vol) == 0 or offer[0] > supply_vol[-1][0]:
            supply_vol.append([offer[0], offer[1]])
        elif offer[0] == supply_vol[-1][0]:
            supply_vol[-1][1] += offer[1]
    supply_vol = np.array(supply_vol)

    ###### get residual volumes (cumsum - demand) ######
    supply_vol[:, 1] = np.cumsum(supply_vol[:, 1]) - demand["volume"].abs().sum()
    print(supply_vol[:, 1])

    ###### get upper and lower brackets ######
    upper_ind = [i for i in range(len(supply_vol)) if supply_vol[i, 1] > 0]
    lower_ind = np.array([i for i in range(upper_ind[0])][::-1])
    #lower_ind = np.array([i for i in range(upper_ind[0] + 1)][::-1])
    #upper_ind = np.array([upper_ind[0] - 1] + upper_ind)

    lower_bracket = supply_vol[lower_ind]
    upper_bracket = supply_vol[upper_ind]
    print(lower_bracket)
    print(upper_bracket)

    ###### iterate through supply offers and get exploitability per unit ######
    for supply_offer in supply_offers:
        o_p, o_v, a_p, a_v = supply_offer

        ######iterate through residual volume brackets and collect solutions ######
        profits = []
        for (p, r_v) in upper_bracket:
            if r_v < 0:
                continue  # even with own volume there this price nivea does not fill demand
            elif r_v - o_v < 0:
                profits.append([p, ])
                continue  # own volume to little to meet demand




    break
# print(orders)

if False:
    world = World(database_uri=db_uri, export_csv_path="")

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario="example_01a",
        study_case="base",
    )