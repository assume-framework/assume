import pandas as pd

a = pd.read_csv("examples/inputs/example_03_naive/powerplant_units.csv", index_col=0)
a = a[[i for i in a.columns if i not in ["bidding_CRM_pos","bidding_CRM_neg", "1"]]]
a["bidding_EOM"] = ["powerplant_energy_naive" for i in range(len(a["bidding_EOM"]))]
a.to_csv("examples/inputs/example_03_naive/powerplant_units.csv", index=False)