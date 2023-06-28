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
        # "ltm_case01",
        # "ltm_case02",
        # "ltm_case03",
        # "ltm_case04",
        # "ltm_case05",
        # "ltm_case06",
        # "ltm_case07",
        # "ltm_case08",
        # "ltm_case09",
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
    --and simulation like 'ltm_case%%'
    and simulation in ('ltm_case00', 'ltm_case10')
    group by simulation, name ORDER BY simulation
    """
    df = pd.read_sql(sql, engine)
    df["total_volume"] /= 1e9
    df["total_cost"] /= 1e6
    savefig = partial(plt.savefig, transparent=False, bbox_inches="tight")

    ## Data preparation
    eom = df[df["name"] == "EOM"]
    ltm = df[df["name"] == "LTM_OTC"].reset_index()
    # ltm.loc[0, "average_cost"] = None
    xticks = list(eom["simulation"])
    # xlabels = [f"{i}%" for i in range(0, 101, 10)]
    xlabels = ["EOM", "EOM + LTM"]
    plt.style.use("seaborn-v0_8")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Total Dispatch cost
    ax1.bar(eom["simulation"], eom["total_cost"], label="EOM")
    eom_ltm = eom[eom.simulation == "ltm_case10"]
    ax1.bar(
        ltm["simulation"],
        ltm["total_cost"],
        bottom=eom_ltm["total_cost"],
        label="LTM",
    )
    ax1.set_ylabel("Total dispatch cost \n per market [mill. $€$]")
    ax1.set_xticks(xticks, xlabels)
    ax1.legend()
    # Total Average Cost
    ax2.scatter(eom["simulation"], eom["average_cost"], label="EOM")
    ax2.scatter(ltm["simulation"], ltm["average_cost"], label="LTM")
    ax2.bar(eom["simulation"], eom["total_cost"] * 0)
    ax2.set_ylabel("Average cost \n for each scenario [$€/MWh$]")
    # ax2.set_xlabel("Fraction of base load traded on LTM in percent")
    ax2.set_xlabel("Selected electricity market design")
    ax2.set_xticks(xticks, xlabels)
    ax2.legend()
    savefig("overview-cost.pdf")
    savefig("overview-cost.png")
    savefig(Path("overview-cost.svg"))
    plt.clf()

    sql = """
SELECT
  time_bucket('60.000s',datetime) AS "time",
  sum(power) AS "market_dispatch",
  market_id,
  um.technology
FROM market_dispatch md
join unit_meta um on um.index = md.unit_id and um.simulation = md.simulation
WHERE
  md.simulation = 'ltm_case10'
GROUP BY 1, market_id, technology
ORDER BY technology, market_id desc, 1
    """

    df = pd.read_sql(sql, engine, index_col="time")
    # fig, ax = plt.subplots(figsize=(8,6))
    series = []
    for label, sub_df in df.groupby(["market_id", "technology"]):
        lab = "-".join(label)
        lab = lab.replace("LTM_OTC", "LTM")

        if "nuclear" not in lab and "coal" not in lab:
            continue
        group_sum = sub_df.market_dispatch.groupby("time").sum()
        group_sum.name = lab
        series.append(group_sum.resample("1h").ffill())

    ddf = pd.DataFrame(series)
    ddf = ddf.T.fillna(method="ffill")

    ddf = ddf[sorted(ddf.columns, reverse=True)]
    ddf = ddf.fillna(0)
    ddf /= 1e3
    base = ddf[ddf.columns[0]] * 0
    for col in ddf.columns:
        line = base + ddf[col]
        c = (0.3, 0.2, 0.6, 0.8) if "nuclear" in col else "g"
        alpha = 0.8 if "LTM" in col else 0.6
        plt.fill_between(line.index, line, base, alpha=alpha, label=col, color=c)
        base += ddf[col]
    plt.ylabel("Hourly dispatch power [$GW$]")
    plt.xlabel("Datetime")
    plt.xticks(rotation=25)
    plt.legend()
    savefig(Path("overview-dispatch.svg"))
    savefig(Path("overview-dispatch.pdf"))
    savefig(Path("overview-dispatch.png"))
    plt.clf()


if __name__ == "__main__":
    # run_cases()
    plot_and_save_results()
