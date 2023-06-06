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
        self.start = start
        self.end = end

        # initalizes dfs for storing and writing asynchron
        self.write_dfs: dict[str, []] = {
            "unit_dispatch": [],
            "market_dispatch": [],
            "market_meta": [],
            "market_orders": [],
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
            freq=rr.HOURLY,
            interval=self.save_frequency_hours,
            dtstart=self.start,
            until=self.end,
            cache=True,
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

        elif content.get("type") == "market_dispatch":
            self.write_market_dispatch(content.get("data"))

        elif content.get("type") == "unit_dispatch":
            self.write_unit_dispatch(content.get("data"))

    def write_market_results(self, market_meta):
        """
        Writes market results to the corresponding data frame.

        Args:
            market_meta: The market metadata, which includes the clearing price and volume.
        """

        df = pd.DataFrame(market_meta)
        if df.empty:
            return
        df["simulation"] = self.simulation_id
        self.write_dfs["market_meta"].append(df)

    async def store_dfs(self):
        """
        Stores the data frames to CSV files and the database.
        Is scheduled as a recurrent task based on the frequency.
        """

        for table in self.write_dfs.keys():
            if len(self.write_dfs[table]) == 0:
                continue
            df = pd.concat(self.write_dfs[table], axis=0)
            df.reset_index()
            if df.empty:
                continue
            if self.export_csv_path:
                data_path = self.p.joinpath(f"{table}.csv")
                df.to_csv(data_path, mode="a", header=not data_path.exists())

            if self.db is not None:
                df.to_sql(table, self.db.bind, if_exists="append")
            self.write_dfs[table] = []

    def write_market_orders(self, market_orders, market_id):
        """
        Writes market orders to the corresponding data frame.
        Append new data until it is written to db and csv with store_df function.

        Args:
            market_result: The market result including all orders.
            market_id: The name of the market.
        """
        # check if market results list is empty and skip the funktion and raise a warning
        if not market_orders:
            return
        df = pd.DataFrame.from_records(market_orders, index="start_time")
        del df["only_hours"]
        del df["agent_id"]
        df["simulation"] = self.simulation_id
        df["market_id"] = market_id
        self.write_dfs["market_orders"].append(df)

    def write_units_definition(self, unit_type, unit):
        """
        Writes unit definitions to the corresponding data frame and directly store it in db and csv.
        Since that is only done once, no need for recurrent sheduling arises.

        Args:
            unit_type (str): The type of the unit.
            unit_params: The parameters of the unit.
        """

        if unit_type == "power_plant":
            unit_info = {
                unit.id: {
                    "simulation": self.simulation_id,
                    "unit_type": unit_type,
                    "technology": unit.technology,
                    "max_power": unit.max_power,
                    "min_power": unit.min_power,
                    "emission_factor": unit.emission_factor,
                    "efficiency": unit.efficiency,
                    "unit_operator": unit.unit_operator,
                }
            }

            df = pd.DataFrame(unit_info).T

            table_name = "unit_meta"

        elif unit_type == "storage_unit":
            unit_info = {
                unit.id: {
                    "simulation": self.simulation_id,
                    "unit_type": unit_type,
                    "technology": unit.technology,
                    "max_power_charge": unit.max_power_charge,
                    "max_power_discharge": unit.max_power_discharge,
                    "min_power_charge": unit.min_power_charge,
                    "min_power_discharge": unit.min_power_discharge,
                    "efficiency_charge": unit.efficiency_discharge,
                    "unit_operator": unit.unit_operator,
                }
            }
            
            df = pd.DataFrame(unit_info).T

            table_name = "storage_meta"

        elif unit_type == "demand":
            unit_info = {
                unit.id: {
                    "simulation": self.simulation_id,
                    "unit_type": unit_type,
                    "technology": unit.technology,
                    "max_power": unit.max_power,
                    "min_power": unit.min_power,
                    "unit_operator": unit.unit_operator,
                }
            }

            df = pd.DataFrame(unit_info).T

            table_name = "demand_meta"
        else:
            logger.info(f"unknown {unit_type} is not exported")
            return False

        if self.export_csv_path:
            market_data_path = self.p.joinpath(f"{table_name}.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
        if self.db is not None and not df.empty:
            df.to_sql(table_name, self.db.bind, if_exists="append")

    def write_market_dispatch(self, data):
        """
        Writes the planned dispatch of the units after the market clearing to a csv and db
        In the case that we have no portfolio optimisation this equals the resulting bids.

        Args:
            data: The records to be put into the table.
            Formatted like, "datetime, power, market_id, unit_id"
        """
        df = pd.DataFrame(data, columns=["datetime", "power", "market_id", "unit_id"])
        df["simulation"] = self.simulation_id
        self.write_dfs["market_dispatch"].append(df)

    def write_unit_dispatch(self, data):
        """
        Writes the actual dispatch of the units to a csv and db

        Args:
            data: The records to be put into the table.
            Formatted like, "datetime, power, market_id, unit_id"
        """
        data["simulation"] = self.simulation_id
        self.write_dfs["unit_dispatch"].append(data)

    async def on_stop(self):
        """
        This function makes it possible to calculate Key Performance Indicators
        """

        # insert left records into db
        await self.store_dfs()
        queries = [
            f"select market_id as name, avg(price) as avg_price from market_meta where simulation = '{self.simulation_id}' group by market_id",
            f"select market_id as name, sum(price*demand_volume) as total_cost from market_meta where simulation = '{self.simulation_id}' group by market_id",
            f"select market_id as name, sum(demand_volume) as total_volume from market_meta where simulation = '{self.simulation_id}' group by market_id",
            f"select unit_id as name, market_id, avg(power/max_power) as capacity_factor from market_dispatch ud join unit_meta um on ud.unit_id = um.\"index\" and ud.simulation=um.simulation where um.simulation = '{self.simulation_id}' group by name, market_id",
        ]
        dfs = []
        for query in queries:
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
