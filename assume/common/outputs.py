import logging
import os
import shutil
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
        """
        Initializes an instance of the WriteOutput class.

        Args:
            simulation_id (str): The ID of the simulation as a unique calssifier.
            start (datetime): The start datetime of the simulation run.
            end (datetime): The end datetime of the simulation run.
            db_engine (optional): The database engine. Defaults to None.
            export_csv_path (str, optional): The path for exporting CSV files, no path results in not writing the csv. Defaults to "".
            save_frequency_hours (int | None, optional): The frequency in hours for storeing data in the db and/or csv files. Defaults to None.
        """

        super().__init__()

        # store needed date
        self.simulation_id = simulation_id
        self.save_frequency_hours: int = save_frequency_hours or 1

        # make directory if not already present
        self.export_csv_path = export_csv_path
        if self.export_csv_path:
            self.p = Path(self.export_csv_path, simulation_id)
            if self.p.exists():
                shutil.rmtree(self.p)
            self.p.mkdir(parents=True)
        self.db = db_engine

        # contruct all timeframe under which hourly values are written to excel and db
        self.delay = (end - start).total_seconds()
        self.start = start
        self.end = end

        # initalizes dfs for storing and writing asynchron
        self.write_dfs: dict[str, pd.DataFrame] = {
            "unit_dispatch": pd.DataFrame(),
            "market_meta": pd.DataFrame(),
            "market_orders": pd.DataFrame(),
        }

        # Loop throuph all database tabels
        # Get list of table names in database
        if self.db is not None:
            table_names = inspect(self.db.bind).get_table_names()
            # Iterate through each table
            for table_name in table_names:
                with self.db() as db:
                    # Read table into Pandas DataFrame
                    query = text(
                        f"delete from {table_name} where simulation = '{self.simulation_id}'"
                    )
                    rowcount = db.execute(query).rowcount
                    # has to be done manually with raw queries
                    db.commit()
                    logger.debug("deleted %s rows from %s", rowcount, table_name)

    def setup(self):
        """
        Sets up the WriteOutput instance by subscribing to messages and scheduling recurrent tasks of storing the data.
        """

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
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta: The metadata associated with the message. (not needed yet)
        """

        if content.get("type") == "store_order_book":
            self.write_market_orders(content.get("data"), content.get("sender"))

        elif content.get("type") == "store_market_results":
            self.write_market_results(content.get("data"))

        elif content.get("type") == "store_units":
            self.write_units_definition(content.get("unit_type"), content.get("data"))

        elif content.get("type") == "store_dispatch":
            self.write_dispatch_plan(content.get("data"))

    def write_market_results(self, market_meta):
        """
        Writes market results to the corresponding data frame.

        Args:
            market_meta: The market metadata, which includes the clearing price and volume.
        """

        df = pd.DataFrame(market_meta)
        df["simulation"] = self.simulation_id
        self.write_dfs["market_meta"] = pd.concat(
            [self.write_dfs["market_meta"], df], axis=0
        )

    async def store_dfs(self):
        """
        Stores the data frames to CSV files and the database.
        Is scheduled as a recurrent task based on the frequency.
        """

        for table in self.write_dfs.keys():
            df = self.write_dfs[table]
            if df.empty:
                continue
            df.reset_index()
            if self.export_csv_path:
                data_path = self.p.joinpath(f"{table}.csv")
                df.to_csv(data_path, mode="a", header=not data_path.exists())

            if self.db is not None:
                df.to_sql(table, self.db.bind, if_exists="append")
            self.write_dfs[table] = pd.DataFrame()

    def write_market_orders(self, market_result, market_name):
        """
        Writes market orders to the corresponding data frame.
        Append new data until it is wirtten to db and csv with store_df function.

        Args:
            market_result: The market result including all orders.
            market_name: The name of the market.
        """

        df = pd.DataFrame.from_records(market_result, index="start_time")
        del df["only_hours"]
        del df["agent_id"]
        df["simulation"] = self.simulation_id
        df["market_name"] = market_name
        self.write_dfs["market_orders"] = pd.concat(
            [self.write_dfs["market_orders"], df], axis=0
        )

    def write_units_definition(self, unit_type, unit_params):
        """
        Writes unit definitions to the corresponding data frame and directly store it in db and csv.
        Since that is only done once, no need for recurrent sheduling arises.

        Args:
            unit_type (str): The type of the unit.
            unit_params: The parameters of the unit.
        """

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
            df["min_power"] = min(df["min_power"])

            table_name = "unit_meta"

        elif unit_type == "demand":
            del unit_params["bidding_strategies"]

            df = pd.DataFrame(unit_params)
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
            market_data_path = self.p.joinpath(f"{table_name}.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
        if self.db is not None and not df.empty:
            df.to_sql(table_name, self.db.bind, if_exists="append")

    def write_dispatch_plan(self, data):
        """
        Writes the planned dispatch of the units after the market clearing to a csv and db
        In the case that we have no portfolio optimisation this equals the bids.

        Args:
            data: The records to be put into the table.
            Formatted like, "datetime, power, market_id, bid_id"
        """
        df = pd.DataFrame(data, columns=["datetime", "power", "market_id", "bid_id"])
        df["simulation"] = self.simulation_id
        self.write_dfs["unit_dispatch"] = pd.concat(
            [self.write_dfs["unit_dispatch"], df], axis=0
        )

    async def on_stop(self):
        """
        This function makes it possible to calculate Key Performance Indicators
        """

        # insert left records into db
        await self.store_dfs()
        queries = {
            "avg_price_mw": f"select name, avg(price) as avg_price from market_meta where simulation = '{self.simulation_id}' group by name",
            "total_cost": f"select name, sum(price*demand_volume) as total_cost from market_meta where simulation = '{self.simulation_id}' group by name",
            "total_volume": f"select name, sum(demand_volume) as total_volume from market_meta where simulation = '{self.simulation_id}' group by name",
            "capacity_factor": f"select bid_id as name, market_id, avg(power/max_power) as capacity_factor from unit_dispatch ud join unit_meta um on ud.bid_id = um.\"index\" and ud.simulation=um.simulation where um.simulation = '{self.simulation_id}' group by name, market_id",
        }
        dfs = []
        for value, query in queries.items():
            df = pd.read_sql(query, self.db.bind)
            dfs.append(df.melt(id_vars=["name"]))
        df = pd.concat(dfs)
        df.reset_index()
        df["simulation"] = self.simulation_id
        if self.export_csv_path:
            kpi_data_path = self.p.joinpath("kpis.csv")
            df.to_csv(kpi_data_path, mode="a", header=not kpi_data_path.exists())
        if self.db is not None and not df.empty:
            df.to_sql("kpis", self.db.bind, if_exists="append")
