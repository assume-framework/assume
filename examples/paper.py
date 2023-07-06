from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from assume import World

DB = "postgresql://assume:assume@localhost:5432/assume"


def run_cases():
    case_studies = [
        "ltm_case00",
        "ltm_case01",
        "ltm_case02",
        "ltm_case03",
        "ltm_case04",
        "ltm_case05",
        "ltm_case06",
        "ltm_case07",
        "ltm_case08",
        "ltm_case09",
        "ltm_case10",
    ]
    for case_study in case_studies:
        world = World(database_uri=DB)
        world.load_scenario(
            inputs_path="examples/inputs",
            scenario="example_02",
            study_case=case_study,
        )
        world.run()


def plot_and_save_results():
    engine = create_engine(DB)

    sql = """
    SELECT name,  simulation,
    sum(round(CAST(value AS numeric), 2))  FILTER (WHERE variable = 'total_cost') as total_cost,
    sum(round(CAST(value AS numeric), 2)*1000)  FILTER (WHERE variable = 'total_volume') as total_volume,
    sum(round(CAST(value AS numeric), 2))  FILTER (WHERE variable = 'avg_price') as average_cost
    FROM kpis
    where variable in ('total_cost', 'total_volume', 'avg_price')
    and simulation like 'ltm_case%%'
    --and simulation in ('ltm_case00', 'ltm_case07', 'ltm_case10')
    group by simulation, name ORDER BY simulation
    """
    # sql = "select * from kpis"
    df = pd.read_sql(sql, engine)
    df["total_volume"] /= 1e9
    df["total_cost"] /= 1e6
    savefig = partial(plt.savefig, transparent=False, bbox_inches="tight")

    eom = df[df["name"] == "EOM"]
    ltm = df[df["name"] == "LTM_OTC"].reset_index()
    ltm.loc[0, "average_cost"] = None
    xticks = list(ltm["simulation"])
    xlabels = [f"{i}%" for i in range(0, 101, 10)]
    plt.style.use("seaborn-v0_8")

    plt.bar(eom["simulation"], eom["total_cost"], label="EOM")
    plt.bar(ltm["simulation"], ltm["total_cost"], bottom=eom["total_cost"], label="LTM")
    plt.ylabel("total dispatch cost in million $€$ per market")
    plt.xticks(xticks, xlabels, rotation=55)
    plt.legend()
    savefig("overview-total_cost.pdf")
    savefig("overview-total_cost.svg")
    plt.clf()
    plt.bar(eom["simulation"], eom["total_volume"], label="EOM")
    plt.bar(
        ltm["simulation"], ltm["total_volume"], bottom=eom["total_volume"], label="LTM"
    )
    plt.xticks(xticks, xlabels, rotation=55)
    plt.ylabel("total traded volume per market in $TWh$ ")
    plt.xlabel("fraction of base load traded on LTM in percent")
    plt.legend()
    savefig("overview-total_volume.pdf")
    savefig("overview-total_volume.svg")
    plt.clf()
    plt.plot(eom["simulation"], eom["average_cost"], label="EOM")
    plt.plot(ltm["simulation"], ltm["average_cost"], label="LTM")
    plt.ylabel("average cost in $€/MWh$ for each scenario")
    plt.xticks(xticks, xlabels, rotation=55)
    plt.legend()
    savefig("overview-average_cost.pdf")
    savefig("overview-average_cost.png")
    savefig(Path("overview-average_cost.svg"))
    plt.clf()


if __name__ == "__main__":
    run_cases()
    plot_and_save_results()
