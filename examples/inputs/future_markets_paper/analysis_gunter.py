# Module imports
import os

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# assume module imports
import examples.examples as examples


example = "case_study_2013"
db_uri = "postgresql://assume:assume@localhost:5432/assume"
inputs_dir = "examples/inputs"

scenario = examples.available_examples[example]["scenario"]
study_case = examples.available_examples[example]["study_case"]

# Set up the database connection
db = create_engine(db_uri)

query = f"SELECT * FROM unit_dispatch where simulation = '{scenario}_{study_case}'"
dispatch_df = pd.read_sql(query, db)
dispatch_df = dispatch_df.drop_duplicates(subset=["time", "unit"], keep="first")

dispatch_df = dispatch_df.sort_values("time")
dispatch_df.head()

query = f"SELECT * FROM market_meta where simulation = '{scenario}_{study_case}'"
market_meta_df = pd.read_sql(query, db)

market_meta_df = market_meta_df.sort_values("time")
market_meta_df.head()

query = f"""
SELECT 
    start_time, 
    unit_id, 
    AVG(accepted_price) AS avg_accepted_price
FROM market_orders
WHERE 
    simulation = '{scenario}_{study_case}'
    AND market_id = '{market_meta_df['market_id'].iloc[0]}'
    AND price = accepted_price
    AND volume >= accepted_volume
GROUP BY start_time, unit_id
ORDER BY start_time
"""
price_setters_df = pd.read_sql(query, db)

unit_mapping = {
    # Solar
    **{
        name: "solar"
        for name in [
            "solar_07_0",
            "solar_04_1",
            "solar_04_2",
            "solar_06_0",
            "solar_06_1",
            "solar_06_2",
            "solar_06_3",
            "solar_07_1",
            "solar_04_0",
            "solar_09_0",
            "solar_03_2",
            "solar_01_0",
            "solar_01_1",
            "solar_08",
            "solar_01_2",
            "solar_05",
            "solar_02",
            "solar_03_0",
            "solar_00",
            "solar_03_1",
            "solar_09_2",
            "solar_09_1",
        ]
    },
    # OCGT (Open Cycle Gas Turbine)
    **{
        name: "ocgt"
        for name in [
            "OCGT_04",
            "OCGT_03",
            "OCGT_02",
            "OCGT_00",
            "OCGT_05",
            "OCGT_01_0",
            "OCGT_01_2",
            "OCGT_01_3",
            "OCGT_01_4",
            "OCGT_09",
            "OCGT_08",
            "OCGT_07",
            "OCGT_06",
            "OCGT_01_1",
        ]
    },
    # Wind Onshore
    **{
        name: "wind_onshore"
        for name in [
            "wind_onshore_09",
            "wind_onshore_08",
            "wind_onshore_05",
            "wind_onshore_04",
            "wind_onshore_02",
            "wind_onshore_01",
            "wind_onshore_03_1",
            "wind_onshore_06_1",
            "wind_onshore_07_0",
            "wind_onshore_07_2",
            "wind_onshore_07_3",
            "wind_onshore_07_1",
            "wind_onshore_03_0",
            "wind_onshore_06_0",
            "wind_onshore_00_1",
            "wind_onshore_00_0",
            "wind_onshore_00_2",
        ]
    },
    # Wind Offshore
    **{
        name: "wind_offshore"
        for name in ["wind_offshore_07", "wind_offshore_06", "wind_offshore_00"]
    },
    # Battery
    **{
        name: "battery"
        for name in [
            "battery_06_1",
            "battery_06_0",
            "battery_04_2",
            "battery_04_1",
            "battery_04_0",
            "battery_03_2",
            "battery_03_1",
            "battery_03_0",
            "battery_06_2",
            "battery_01_1",
            "battery_01_0",
            "battery_00_1",
            "battery_00_0",
            "battery_09",
            "battery_08",
            "battery_07",
            "battery_05",
            "battery_02",
        ]
    },
    # Hydro (Pumped Storage and Run-of-River, based on names)
    **{
        name: "hydro"
        for name in [
            "Goldisthal",
            "Markersbach",
            "Wehr",
            "Waldeck II",
            "Säckingen",
            "Hohenwarte II",
            "Witznau",
            "Erzhausen",
            "Waldshut",
            "Langenprozelten",
            "Happurg",
            "Koepchenwerk Herdecke II",
            "Häusern",
            "Waldeck I",
            "Rönkhausen",
            "Geesthacht",
            "Tanzmühle - Rabenleite",
            "Wendefurth",
            "Hohenwarte I",
            "Leitzach I",
            "Bleiloch",
            "Glems",
            "Leitzach II",
            "Reisach - Rabenleite",
            "Schwarzenbachwerk",
        ]
    },
    # Demand (special category)
    "demand": "demand",
}

price_setters_df["unit_type"] = price_setters_df["unit_id"].map(unit_mapping)
unit_colors = {
    "solar": "#FFD700",  # Gold
    "ocgt": "#FF4500",  # OrangeRed
    "wind_onshore": "#1E90FF",  # DodgerBlue
    "wind_offshore": "#00CED1",  # DarkTurquoise
    "battery": "#8A2BE2",  # BlueViolet
    "hydro": "#32CD32",  # LimeGreen
    "demand": "#808080",  # Gray
}
# price duration curve with colors according to price setting unit type

price_setters_df.to_csv(os.path.join(inputs_dir, "price_setters.csv"), index=False)
# plt.figure(figsize=(8, 4))
# x = np.linspace(0, 100, len(price_setters_df))
# plt.plot(x, price_setters_df.sort_values("avg_accepted_price", ascending=False).reset_index(drop=True).avg_accepted_price,
#          color=[unit_colors[unit] for unit in price_setters_df.sort_values("avg_accepted_price", ascending=False).reset_index(drop=True).unit_type],
#          linewidth=2, marker='o', markersize=3)
# plt.xlabel("Percentage of Time [%]")
# plt.ylabel("Price [€/MWh]")
# plt.title("Electricity Price Duration Curve")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()
x = np.linspace(0, 100, len(price_setters_df))
sorted_df = price_setters_df.sort_values("avg_accepted_price", ascending=False).reset_index(drop=True)
ax.scatter(x, sorted_df.avg_accepted_price,
        color=[unit_colors[unit] for unit in sorted_df.unit_type],
        linewidth=2, marker='o', s=3)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlabel("Percentage of Time [%]")
ax.set_ylabel("Price [€/MWh]")
ax.set_title("Electricity Price Duration Curve")
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=unit, 
                               markerfacecolor=color, markersize=5) 
                               for unit, color in unit_colors.items()],
           title="Unit Type", loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()