# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import rrule as rr
from mango import Role
from pandas.api.types import is_numeric_dtype
from sqlalchemy import inspect, text
from sqlalchemy.exc import DataError, OperationalError, ProgrammingError

logger = logging.getLogger(__name__)

from assume.common.market_objects import MetaDict
from assume.common.utils import separate_orders


class WriteOutput(Role):
    """
    Initializes an instance of the WriteOutput class.

    Args:
        simulation_id (str): The ID of the simulation as a unique classifier.
        start (datetime.datetime): The start datetime of the simulation run.
        end (datetime.datetime): The end datetime of the simulation run.
        db_engine: The database engine. Defaults to None.
        export_csv_path (str, optional): The path for exporting CSV files, no path results in not writing the csv. Defaults to "".
        save_frequency_hours (int): The frequency in hours for storing data in the db and/or csv files. Defaults to None.
        learning_mode (bool, optional): Indicates if the simulation is in learning mode. Defaults to False.
        evaluation_mode (bool, optional): Indicates if the simulation is in evaluation mode. Defaults to False.
    """

    def __init__(
        self,
        simulation_id: str,
        start: datetime,
        end: datetime,
        db_engine=None,
        export_csv_path: str = "",
        save_frequency_hours: int = None,
        learning_mode: bool = False,
        evaluation_mode: bool = False,
    ):
        super().__init__()

        # store needed date
        self.simulation_id = simulation_id
        self.save_frequency_hours = save_frequency_hours or (end - start).days * 24

        # make directory if not already present
        self.export_csv_path = export_csv_path
        if self.export_csv_path:
            self.p = Path(self.export_csv_path, simulation_id)
            shutil.rmtree(self.p, ignore_errors=True)
            self.p.mkdir(parents=True)

        self.db = db_engine

        self.learning_mode = learning_mode
        self.evaluation_mode = evaluation_mode

        # get episode number if in learning or evaluation mode
        self.episode = None
        if self.learning_mode or self.evaluation_mode:
            episode = self.simulation_id.split("_")[-1]
            if episode.isdigit():
                self.episode = int(episode)

            # check if episode=0 and delete all similar runs
            if self.episode == 0:
                self.del_similar_runs()

        # contruct all timeframe under which hourly values are written to excel and db
        self.start = start
        self.end = end
        # initalizes dfs for storing and writing asynchron
        self.write_dfs: dict = defaultdict(list)

        if self.db is not None:
            self.delete_db_scenario(self.simulation_id)

    def delete_db_scenario(self, simulation_id: str):
        """
        Deletes all data from the database for the given simulation id.

        Args:
            simulation_id (str): The ID of the simulation as a unique classifier.
        """

        # Loop throuph all database tables
        # Get list of table names in database
        table_names = inspect(self.db).get_table_names()
        # Iterate through each table
        for table_name in table_names:
            try:
                with self.db.begin() as db:
                    # create index on table
                    query = text(
                        f'create index if not exists "{table_name}_scenario" on "{table_name}" (simulation)'
                    )
                    db.execute(query)

                    query = text(
                        f"delete from \"{table_name}\" where simulation = '{simulation_id}'"
                    )
                    rowcount = db.execute(query).rowcount
                    # has to be done manually with raw queries
                    db.commit()
                    logger.debug("deleted %s rows from %s", rowcount, table_name)
            except Exception as e:
                logger.error(
                    f"could not clear old scenarios from table {table_name} - {e}"
                )

    def del_similar_runs(self):
        """
        Deletes all similar runs from the database based on the simulation ID. This ensures that we overwrite simulations results when restarting one. Please note that a simulation which you also want to keep need to be assigned anew ID.
        """
        query = text("select distinct simulation from rl_params")

        try:
            with self.db.begin() as db:
                simulations = db.execute(query).fetchall()
        except Exception:
            simulations = []
        simulations = [s[0] for s in simulations]

        for simulation_id in simulations:
            # delete all simulation_id which are similar to my simulation_id
            if simulation_id.startswith(self.simulation_id[:-1]):
                self.delete_db_scenario(simulation_id)

    def setup(self):
        """
        Sets up the WriteOutput instance by subscribing to messages and scheduling recurrent tasks of storing the data.
        """
        super().setup()

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

    def handle_message(self, content: dict, meta: MetaDict):
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta (MetaDict): The metadata associated with the message.
        """

        if content.get("type") == "store_order_book":
            self.write_market_orders(content.get("data"), content.get("market_id"))

        elif content.get("type") == "store_market_results":
            self.write_market_results(content.get("data"))

        elif content.get("type") == "store_units":
            self.write_units_definition(content.get("data"))

        elif content.get("type") == "market_dispatch":
            self.write_market_dispatch(content.get("data"))

        elif content.get("type") == "unit_dispatch":
            self.write_unit_dispatch(content.get("data"))

        elif content.get("type") == "rl_learning_params":
            self.write_rl_params(content.get("data"))

    def write_rl_params(self, rl_params: dict):
        """
        Writes the RL parameters such as reward, regret, and profit to the corresponding data frame.

        Args:
            rl_params (dict): The RL parameters.
        """

        df = pd.DataFrame.from_records(rl_params, index="datetime")
        if df.empty:
            return

        df["simulation"] = self.simulation_id
        df["learning_mode"] = self.learning_mode
        df["evaluation_mode"] = self.evaluation_mode
        df["episode"] = self.episode

        self.write_dfs["rl_params"].append(df)

    def write_market_results(self, market_meta: dict):
        """
        Writes market results to the corresponding data frame.

        Args:
            market_meta (dict): The market metadata, which includes the clearing price and volume.
        """

        df = pd.DataFrame(market_meta)
        if df.empty:
            return
        df["simulation"] = self.simulation_id
        self.write_dfs["market_meta"].append(df)

    async def store_dfs(self):
        """
        Stores the data frames to CSV files and the database. Is scheduled as a recurrent task based on the frequency.
        """

        for table in self.write_dfs.keys():
            if len(self.write_dfs[table]) == 0:
                continue

            df = pd.concat(self.write_dfs[table], axis=0)
            df.reset_index()
            if df.empty:
                continue

            df = df.apply(self.check_for_tensors)

            if self.export_csv_path:
                data_path = self.p.joinpath(f"{table}.csv")
                df.to_csv(data_path, mode="a", header=not data_path.exists())

            if self.db is not None:
                try:
                    with self.db.begin() as db:
                        df.to_sql(table, db, if_exists="append")
                except (ProgrammingError, OperationalError, DataError):
                    self.check_columns(table, df)
                    # now try again
                    with self.db.begin() as db:
                        df.to_sql(table, db, if_exists="append")

            self.write_dfs[table] = []

    def check_columns(self, table: str, df: pd.DataFrame):
        """
        Checks and adds columns to the database table if necessary.

        Args:
            table (str): The name of the database table.
            df (pd.DataFrame): The DataFrame to be checked.
        """
        with self.db.begin() as db:
            # Read table into Pandas DataFrame
            query = f"select * from {table} where 1=0"
            db_columns = pd.read_sql(query, db).columns
        for column in df.columns:
            if column not in db_columns:
                try:
                    # TODO this only works for float and text
                    column_type = "float" if is_numeric_dtype(df[column]) else "text"
                    query = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                    with self.db.begin() as db:
                        db.execute(text(query))
                except Exception:
                    logger.exception("Error converting column")

    def check_for_tensors(self, data: pd.Series):
        """
        Checks if the data contains tensors and converts them to floats.

        Args:
            data (pd.Series): The data to be checked.
        """
        try:
            import torch as th

            if data.map(lambda x: isinstance(x, th.Tensor)).any():
                for i, value in enumerate(data):
                    if isinstance(value, th.Tensor):
                        data.iat[i] = value.item()
        except ImportError:
            pass

        return data

    def write_market_orders(self, market_orders: any, market_id: str):
        """
        Writes market orders to the corresponding data frame.

        Args:
            market_orders (any): The market orders.
            market_id (str): The id of the market.
        """
        # check if market results list is empty and skip the funktion and raise a warning
        if not market_orders:
            return

        market_orders = separate_orders(market_orders)
        df = pd.DataFrame.from_records(market_orders, index="start_time")
        if "eligible_lambda" in df.columns:
            df["eligible_lambda"] = df["eligible_lambda"].apply(lambda x: x.__name__)
        if "evaluation_frequency" in df.columns:
            df["evaluation_frequency"] = df["evaluation_frequency"].apply(
                lambda x: repr(x)
            )

        del df["only_hours"]
        del df["agent_id"]

        df["simulation"] = self.simulation_id
        df["market_id"] = market_id

        self.write_dfs["market_orders"].append(df)

    def write_units_definition(self, unit_info: dict):
        """
        Writes unit definitions to the corresponding data frame and directly stores it in the database and CSV.

        Args:
            unit_info (dict): The unit information.
        """

        table_name = unit_info["unit_type"] + "_meta"

        if table_name is None:
            logger.info(f"unknown {unit_info['unit_type']} is not exported")
            return False
        del unit_info["unit_type"]
        unit_info["simulation"] = self.simulation_id
        u_info = {unit_info["id"]: unit_info}
        del unit_info["id"]

        self.write_dfs[table_name].append(pd.DataFrame(u_info).T)

    def write_market_dispatch(self, data: any):
        """
        Writes the planned dispatch of the units after the market clearing to a CSV and database.

        Args:
            data (any): The records to be put into the table. Formatted like, "datetime, power, market_id, unit_id".
        """
        df = pd.DataFrame(data, columns=["datetime", "power", "market_id", "unit_id"])
        if not df.empty:
            df["simulation"] = self.simulation_id
            self.write_dfs["market_dispatch"].append(df)

    def write_unit_dispatch(self, data: any):
        """
        Writes the actual dispatch of the units to a CSV and database.

        Args:
            data (any): The records to be put into the table. Formatted like, "datetime, power, market_id, unit_id".
        """
        data["simulation"] = self.simulation_id
        self.write_dfs["unit_dispatch"].append(data)

    async def on_stop(self):
        """
        This function makes it possible to calculate Key Performance Indicators.
        It is called when the simulation is finished. It collects average price, total cost, total volume and capacity factors
        and uses them to calculate the KPIs. The KPIs are then stored in the database and CSV files.
        """
        await super().on_stop()

        # insert left records into db
        await self.store_dfs()

        if self.db is None:
            return

        queries = [
            f"select 'avg_price' as variable, market_id as ident, avg(price) as value from market_meta where simulation = '{self.simulation_id}' group by market_id",
            f"select 'total_cost' as variable, market_id as ident, sum(price*demand_volume_energy) as value from market_meta where simulation = '{self.simulation_id}' group by market_id",
            f"select 'total_volume' as variable, market_id as ident, sum(demand_volume_energy) as value from market_meta where simulation = '{self.simulation_id}' group by market_id",
            f"select 'capacity_factor' as variable, market_id as ident, avg(power/max_power) as value from market_dispatch ud join power_plant_meta um on ud.unit_id = um.\"index\" and ud.simulation=um.simulation where um.simulation = '{self.simulation_id}' group by variable, market_id",
        ]
        if self.episode:
            queries.extend(
                [
                    f"SELECT 'sum_reward' as variable, simulation as ident, sum(reward) as value FROM rl_params WHERE episode='{self.episode}' AND simulation='{self.simulation_id}' GROUP BY simulation",
                    f"SELECT 'sum_regret' as variable, simulation as ident, sum(regret) as value FROM rl_params WHERE episode='{self.episode}' AND simulation='{self.simulation_id}' GROUP BY simulation",
                    f"SELECT 'sum_profit' as variable, simulation as ident, sum(profit) as value FROM rl_params WHERE episode='{self.episode}' AND simulation='{self.simulation_id}' GROUP BY simulation",
                ]
            )

        dfs = []
        for query in queries:
            try:
                df = pd.read_sql(query, self.db)
            except (ProgrammingError, OperationalError, DataError):
                continue
            except Exception as e:
                logger.error("could not read query: %s", e)
                continue

            dfs.append(df)

        # remove all empty dataframes
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return

        df = pd.concat(dfs)
        df.reset_index()
        df["simulation"] = self.simulation_id

        if self.export_csv_path:
            kpi_data_path = self.p.joinpath("kpis.csv")
            df.to_csv(
                kpi_data_path,
                mode="a",
                header=not kpi_data_path.exists(),
                index=None,
            )

        if self.db is not None and not df.empty:
            with self.db.begin() as db:
                df.to_sql("kpis", self.db, if_exists="append", index=None)

    def get_sum_reward(self):
        """
        Retrieves the total reward for each learning unit.

        Returns:
            np.array: The total reward for each learning unit.
        """
        query = text(
            f"select unit, SUM(reward) FROM rl_params where simulation='{self.simulation_id}' GROUP BY unit"
        )

        with self.db.begin() as db:
            rewards_by_unit = db.execute(query).fetchall()

        # convert into a numpy array
        rewards_by_unit = [r[1] for r in rewards_by_unit]
        rewards_by_unit = np.array(rewards_by_unit)

        return rewards_by_unit
