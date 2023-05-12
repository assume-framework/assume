import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil import rrule as rr
from mango import Role
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)


class WriteOutput(Role):
    def __init__(
        self,
        simulation_id: str,
        start: datetime,
        end: datetime,
        db_engine=None,
        export_csv_path: str = "",
        save_frequency_hours: int | None = None,
    ):
        super().__init__()

        # store needed date
        self.simulation_id = simulation_id
        self.save_frequency_hours: int = save_frequency_hours or 1

        # make directory if not already present
        self.export_csv_path = export_csv_path
        if self.export_csv_path:
            self.p = Path(self.export_csv_path)
            self.p.mkdir(parents=True, exist_ok=True)
        self.db = db_engine

        # contruct all timeframe under which hourly values are written to excel and db
        self.delay = (end - start).total_seconds()
        self.start = start
        self.end = end

        # initalizes dfs for storing and writing asynchron
        self.write_dfs: dict[str, pd.DataFrame] = {
            "power_plant_dispatch": pd.DataFrame(),
            "market_meta": pd.DataFrame(),
            "market_orders": pd.DataFrame(),
        }

        if self.export_csv_path:
            # Check id data for this simulation id is already present and delete it if so
            logger.info(
                f"deleting all data with the id {self.simulation_id} if this simulation was previously run"
            )

            # Loop through all Excel files in the directory
            for file_name in os.listdir(self.export_csv_path):
                # Load the Excel file into a pandas dataframe
                file_path = os.path.join(self.export_csv_path, file_name)
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Filter the dataframe based on the specified column and condition
                    df = df[df["simulation"] != self.simulation_id]

                    # Save the updated dataframe back to the original Excel file
                    df.to_csv(file_path, index=False)

        # Loop throuph all database tabels
        # Get list of table names in database
        if self.db is not None:
            table_names = inspect(self.db.bind).get_table_names()

            # Iterate through each table
            for table_name in table_names:
                # Read table into Pandas DataFrame
                query = text(
                    f"delete from {table_name} where simulation = '{self.simulation_id}'"
                )
                with self.db() as db:
                    db.execute(query)

    def setup(self):
        self.context.subscribe_message(
            self,
            self.handle_message,
            lambda content, meta: content.get("context") == "write_results",
        )

        recurrency_task = rr.rrule(
            rr.HOURLY,
            interval=self.save_frequency_hours,
            dtstart=self.start,
            until=self.end,
        )
        self.context.schedule_recurrent_task(self.store_dfs, recurrency_task)

    def handle_message(self, content, meta):
        if not isinstance(content, dict):
            return False

        if content.get("type") == "store_order_book":
            self.write_market_orders(content.get("data"), content.get("sender"))

        elif content.get("type") == "store_market_results":
            self.write_market_results(content.get("data"))

        elif content.get("type") == "store_units":
            self.write_units_definition(content.get("unit_type"), content.get("data"))

        elif content.get("type") == "store_dispatch":
            self.write_dispatch_plan(
                content.get("unit"),
                content.get("unit_id"),
                content.get("capacity"),
                content.get("timestamp"),
            )

    def write_market_results(self, market_meta):
        df = pd.DataFrame.from_dict(market_meta)
        df["simulation"] = self.simulation_id
        self.write_dfs["market_meta"] = pd.concat(
            [self.write_dfs["market_meta"], df], axis=0
        )

    async def store_dfs(self):
        for table, df in self.write_dfs.items():
            if df.empty:
                continue
            df.reset_index()
            if self.export_csv_path:
                data_path = self.p.joinpath(f"{table}.csv")
                df.to_csv(data_path, mode="a", header=not data_path.exists())

            if self.db is not None:
                df.to_sql(table, self.db.bind, if_exists="append")

    def write_market_orders(self, market_result, market_name):
        df = pd.DataFrame.from_dict(market_result)
        del df["only_hours"]
        del df["agent_id"]
        df["simulation"] = self.simulation_id
        df["market_name"] = market_name
        self.write_dfs["market_orders"] = pd.concat(
            [self.write_dfs["market_orders"], df], axis=0
        )

    def write_units_definition(self, unit_type, unit_params):
        if unit_type == "power_plant":
            df = pd.DataFrame([unit_params])
            df["simulation"] = self.simulation_id
            df = df[
                [
                    "simulation",
                    "technology",
                    "fuel_type",
                    "emission_factor",
                    "max_power",
                    "min_power",
                    "efficiency",
                    "unit_operator",
                ]
            ]
            df["max_power"] = max(df["max_power"])
            df["min_power"] = max(df["min_power"])

            table_name = "unit_meta"

        elif unit_type == "demand":
            del unit_params["bidding_strategies"]

            df = pd.DataFrame.from_dict(unit_params)
            df["type"] = unit_type
            df.reset_index(inplace=True)
            df = df.rename(columns={"level_0": "", "index": "Timestamp"})
            # sql does not like Timestamp or other types of values
            # df=df.astype(str)
            # df['Timestamp']=df['Timestamp'].astype(float)
            df["simulation"] = self.simulation_id
            # df['volume']=df["volume"].max()

            table_name = "demand_meta"
        else:
            logger.info(f"unknown {unit_type} is not exported")
            return False

        if self.export_csv_path:
            p = Path(self.export_csv_path)
            market_data_path = p.joinpath(f"{table_name}.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
        if self.db is not None and not df.empty:
            df.to_sql(table_name, self.db.bind, if_exists="append")

    def write_dispatch_plan(self, unit, unit_id, total_power_output, current_time):
        """
        Writes the planned dispatch of the units after the market clearing to a csv and db
        In the case that we have no portfolio optimisation this equals the bids.
        """

        df = pd.DataFrame.from_dict(total_power_output)
        df.rename(columns={df.columns[0]: "power"}, inplace=True)
        df["unit"] = unit
        # sql does not liek tuples, so conversion necessary
        df["unit_id"] = unit_id
        df["timestamp"] = current_time
        df["simulation"] = self.simulation_id
        self.write_dfs["power_plant_dispatch"] = pd.concat(
            [self.write_dfs["power_plant_dispatch"], df], axis=0
        )
