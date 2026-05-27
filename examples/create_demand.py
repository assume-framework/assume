import pandas as pd
import numpy as np

df = pd.read_csv("examples/inputs/exploit_example/demand_df.csv", index_col=0)

days = 31
n = 24 * 4 * days
df = df[:n]
df["demand_EOM"] = np.ones(n) * 6900

df.to_csv("examples/inputs/exploit_example/demand_df.csv", )
