# %%
import itertools

import pandas as pd

folders = ["example_01a", "example_01b", "example_01c", "example_02", "example_03"]
files = ["demand_df.csv", "availability_df.csv", "fuel_prices_df.csv"]

year = 2019
index = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="15T")[:-1]

# %%
for folder, file in itertools.product(folders, files):
    try:
        df = pd.read_csv(f"inputs/{folder}/{file}", index_col=0)
        df = df.set_index(index)
        # rename index to datetime
        df.index.name = "datetime"
        df.to_csv(f"inputs/{folder}/{file}", index=True)
    except FileNotFoundError:
        print(f"File {file} not found in folder {folder}")
        pass
    except Exception as e:
        print(f"Error in file {file} in folder {folder}")
        print(e)
        pass


# %%
