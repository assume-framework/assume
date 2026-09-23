# SPDX-FileCopyrightText: ASSUME Developers
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the documented reduced German grid inputs for the 2019 scenario.

Run from the repository root with:
    uv run python examples/inputs/example_2019_update/build_redispatch_inputs.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def region(name: str, operator: str = "") -> str:
    text = f"{name} {operator}".lower()
    groups = {
        "north": (
            "hamburg", "bremen", "kiel", "lübeck", "lubeck", "emden",
            "wilhelmshaven", "brunsbüttel", "brunsbuttel", "stade", "rostock",
            "schwerin", "tennet", "swb", "enercity",
        ),
        "east": (
            "berlin", "leipzig", "dresden", "cottbus", "halle", "jena",
            "chemnitz", "magdeburg", "schkopau", "lippendorf", "boxberg",
            "jä nschwalde", "jänschwalde", "jaenschwalde", "schwarze pumpe",
            "leag", "envia", "lausitz", "grosskayna", "großkayna",
        ),
        "south": (
            "münchen", "munich", "bayern", "bavaria", "stuttgart", "mannheim",
            "karlsruhe", "ulm", "augsburg", "ingolstadt", "heilbronn", "isar",
            "neckar", "gundremmingen", "philippsburg", "enbw", "mainova",
        ),
        "west": (
            "köln", "cologne", "düsseldorf", "duesseldorf", "dortmund", "essen",
            "duisburg", "aachen", "bochum", "gelsenkirchen", "ruhr", "emsland",
            "rwe", "steag", "uniper", "rhein", "westfalen", "saar",
        ),
    }
    for node, terms in groups.items():
        if any(term in text for term in terms):
            return node
    return "west"


def main() -> None:
    plants = pd.read_csv(HERE / "powerplant_units.csv")
    plants["bidding_redispatch"] = "powerplant_energy_naive_redispatch"
    plants["node"] = [region(n, o) for n, o in zip(plants.name, plants.unit_operator)]
    renewable_nodes = {
        "Wind Onshore": "north", "Wind Offshore": "north", "Solar": "south",
        "Biomass": "west", "Hydro": "south",
    }
    plants.loc[plants.name.isin(renewable_nodes), "node"] = plants.name.map(renewable_nodes)
    plants.to_csv(HERE / "powerplant_units.csv", index=False)

    storage = pd.read_csv(HERE / "storage_units.csv")
    storage["bidding_redispatch"] = "storage_energy_heuristic_redispatch"
    storage["node"] = [region(n, o) for n, o in zip(storage.name, storage.unit_operator)]
    storage.to_csv(HERE / "storage_units.csv", index=False)

    exchange = pd.read_csv(HERE / "exchange_units.csv")
    exchange["bidding_redispatch"] = "exchange_energy_fixed_redispatch"
    exchange["node"] = exchange.name.str.split("_").str[-1].map(
        {"NL": "west", "BE": "west", "FR": "west", "LU": "west", "CH": "south", "AT": "south",
         "CZ": "east", "PL": "east", "DK": "north", "SE": "north"}
    )
    exchange.to_csv(HERE / "exchange_units.csv", index=False)

    demand = pd.read_csv(HERE / "demand_units.csv")
    shares = {"north": 0.16, "east": 0.16, "west": 0.38, "south": 0.30}
    existing_regional = demand.loc[demand.name.str.startswith("demand_EOM_")]
    if (demand.name == "demand_EOM").any():
        original = demand.loc[demand.name == "demand_EOM"].iloc[0].copy()
    else:
        # Allow the documented generator to be run again after it has already
        # replaced the national demand row with regional rows.
        original = existing_regional.iloc[0].copy()
        original["name"] = "demand_EOM"
        original["max_power"] = existing_regional["max_power"].sum()
    rows = []
    for node, share in shares.items():
        row = original.copy()
        row["name"] = f"demand_EOM_{node}"
        row["max_power"] *= share
        row["bidding_redispatch"] = "demand_energy_naive_redispatch"
        row["node"] = node
        rows.append(row)
    other = demand.loc[~demand.name.isin(["demand_EOM", *existing_regional.name])].copy()
    other["bidding_redispatch"] = ""
    other["node"] = "west"
    demand = pd.concat([pd.DataFrame(rows), other], ignore_index=True)
    demand.to_csv(HERE / "demand_units.csv", index=False)

    demand_ts = pd.read_csv(HERE / "demand_df.csv")
    regional_columns = [
        f"demand_EOM_{node}" for node in shares if f"demand_EOM_{node}" in demand_ts
    ]
    total = (
        demand_ts.pop("demand_EOM")
        if "demand_EOM" in demand_ts
        else demand_ts[regional_columns].sum(axis=1)
    )
    demand_ts = demand_ts.drop(columns=regional_columns, errors="ignore")
    for node, share in shares.items():
        demand_ts[f"demand_EOM_{node}"] = total * share
    demand_ts.to_csv(HERE / "demand_df.csv", index=False)

    buses = pd.DataFrame(
        {"v_nom": 380.0, "zone_id": "DE_LU", "x": [9.5, 12.0, 7.0, 10.5],
         "y": [53.5, 52.0, 51.0, 48.5]},
        index=pd.Index(["north", "east", "west", "south"], name="name"),
    )
    buses.to_csv(HERE / "buses.csv")
    lines = pd.DataFrame(
        [
            ("north_west", "north", "west", 12000.0, 1.0, 0.08, 0.008),
            ("north_east", "north", "east", 10000.0, 1.0, 0.08, 0.008),
            ("east_west", "east", "west", 8000.0, 1.0, 0.10, 0.010),
            ("east_south", "east", "south", 8000.0, 1.0, 0.10, 0.010),
            ("west_south", "west", "south", 14000.0, 1.0, 0.08, 0.008),
        ],
        columns=["line", "bus0", "bus1", "s_nom", "s_max_pu", "x", "r"],
    ).set_index("line")
    lines.to_csv(HERE / "lines.csv")


if __name__ == "__main__":
    main()
