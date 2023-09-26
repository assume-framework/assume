# %%
import logging
import os

from assume import World, load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = "examples/outputs"

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
    "small": {"scenario": "example_01a", "study_case": "base"},
    "small_dam": {"scenario": "example_01a", "study_case": "dam"},
    "small_with_opt_clearing": {
        "scenario": "example_01a",
        "study_case": "dam_with_complex_clearing",
    },
    "small_with_BB": {
        "scenario": "example_01e",
        "study_case": "dam_with_complex_clearing",
    },
    "small_with_vre": {"scenario": "example_01b", "study_case": "base"},
    "small_with_vre_and_storage": {
        "scenario": "example_01c",
        "study_case": "eom_only",
    },
    "small_with_vre_and_storage_and_complex_clearing": {
        "scenario": "example_01c",
        "study_case": "dam_with_complex_opt_clearing",
    },
    "small_with_crm": {
        "scenario": "example_01c",
        "study_case": "eom_and_crm",
    },
    "large_2019_eom": {"scenario": "example_02", "study_case": "base_case_2019"},
    "large_2019_eom_crm": {
        "scenario": "example_02",
        "study_case": "eom_crm_case_2019",
    },
    "large_2019_day_ahead": {
        "scenario": "example_02",
        "study_case": "dam_case_2019",
    },
    "rl": {"scenario": "example_01_rl", "study_case": "base"},
    "uc_clearing": {
        "scenario": "example_01_uc",
        "study_case": "dam_with_uc_clearing",
    },
    "uc_clearing_with_rl": {
        "scenario": "example_01_uc",
        "study_case": "dam_with_uc_clearing_rl",
    },
}

# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    data_format = "timescale"  # "local_db" or "timescale"
    example = "uc_clearing_with_rl"

    if data_format == "local_db":
        db_uri = f"sqlite:///./examples/local_db/assume_db_{example}.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario=availabe_examples[example]["scenario"],
        study_case=availabe_examples[example]["study_case"],
    )

    # to add a custom unit type, you need to import it
    # and add it to the world as follows:
    # from custom_unit import CustomUnit
    # world.unit_types["custom_unit"] = CustomUnit

    # you can also add custom bidding strategies as follows:
    # from custom_strategy import CustomStrategy
    # world.bidding_types["custom_strategy"] = CustomStrategy

    # next you need to load and add the custom units to the scenario
    # from assume import load_custom_units
    # load_custom_units(
    #     world,
    #     inputs_path="examples/inputs",
    #     scenario=availabe_examples[example]["scenario"],
    #     file_name="custom_units",
    #     unit_type="custom_unit",
    # )

    if world.learning_config.get("learning_mode", False):
        # run learning if learning mode is enabled
        run_learning(
            world,
            inputs_path="examples/inputs",
            scenario=availabe_examples[example]["scenario"],
            study_case=availabe_examples[example]["study_case"],
        )

    world.run()
    # %%
    import pandas as pd
    import plotly.express as px

    cashflows = pd.read_csv(
        "outputs/example_01_uc_dam_with_uc_clearing_rl/unit_dispatch.csv",
        index_col=0,
        parse_dates=True,
    )
    cashflows = cashflows.loc["2019-03-01"]

    profits = pd.DataFrame(index=cashflows.index.unique())
    # group by unit and iterate to get profit
    for unit, df in cashflows.groupby("unit"):
        profit = df["energy_cashflow"] - df["energy_marginal_costs"]
        profits[f"{unit}_profit"] = profit
        # profits[f"{unit}_mc"] = (round(df["energy_marginal_costs"]/1000,0))
        # profits[f"{unit}_cf"] = (round(df["energy_cashflow"]/1000,0))

    # print total profit per unit
    print(profits.sum(axis=0))

    # delete demand_EOM_profit
    profits = profits.drop(columns=["demand_EOM_profit"])

    # using plotly plot total profits per unit
    fig = px.bar(
        profits.sum(axis=0),
        title="Total profit per unit",
        labels={"index": "Unit", "Profit": "Profit [k€]"},
    )
    # renamy axis to profit in kEUR
    fig.update_yaxes(title_text="Profit [€]")
    # remove legend
    fig.update_layout(showlegend=False)
    fig.show()

# %%
