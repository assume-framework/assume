# packages needed: seaborn, plotly, kaleido

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from sqlalchemy import create_engine

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
    print(type(time_group))
    

    break
# print(orders)


