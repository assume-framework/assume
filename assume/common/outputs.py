# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import shutil
from collections import defaultdict
from datetime import datetime
from io import StringIO
from multiprocessing import Lock
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from dateutil import rrule as rr
from mango import Role
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from psycopg2.errors import UndefinedColumn
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import (
    DataError,
    NoSuchTableError,
    OperationalError,
    ProgrammingError,
)

from assume.common.market_objects import MetaDict
from assume.common.utils import (
    calculate_content_size,
    convert_tensors,
    separate_orders,
)

logger = logging.getLogger(__name__)

# time-series (partitioned) tables
PARTITIONED_TABLES = [
    "market_meta",
    "market_dispatch",
    "market_orders",
    "unit_dispatch",
    "rl_params",
    "grid_flows",
    "kpis",
]

# all tables that carry a `simulation` column, whether static or hypertable
ALL_SIM_TABLES = PARTITIONED_TABLES + [
    "power_plant_meta",
    "storage_meta",
    "demand_meta",
    "exchange_meta",
    "rl_meta",
]


class OutputDef(TypedDict):
    name: str
    value: str
    from_table: str


class WriteOutput(Role):
    """
    Initializes an instance of the WriteOutput class.

    Args:
        simulation_id (str): The ID of the simulation as a unique classifier.
        start (datetime.datetime): The start datetime of the simulation run.
        end (datetime.datetime): The end datetime of the simulation run.
        db_uri: The uri of the database engine. Defaults to ''.
        export_csv_path (str, optional): The path for exporting CSV files, no path results in not writing the csv. Defaults to "".
        save_frequency_hours (int): The frequency in hours for storing data in the db and/or csv files. Defaults to 48 hours.
        outputs_buffer_size_mb (int, optional): The maximum storage size (in MB) for storing output data before saving it. Defaults to 300 MB.
        learning_mode (bool, optional): Indicates if the simulation is in learning mode. Defaults to False.
        evaluation_mode (bool, optional): Indicates if the simulation is in evaluation mode. Defaults to False.
        additional_kpis (dict[str, OutputDef], optional): makes it possible to define additional kpis evaluated
    """

    def __init__(
        self,
        simulation_id: str,
        start: datetime,
        end: datetime,
        db_uri="",
        export_csv_path: str = "",
        save_frequency_hours: int = 48,
        outputs_buffer_size_mb: int = 300,
        learning_mode: bool = False,
        evaluation_mode: bool = False,
        episode: int = None,
        eval_episode: int = None,
        additional_kpis: dict[str, OutputDef] = {},
    ):
        super().__init__()

        # store needed date
        self.simulation_id = simulation_id
        self.save_frequency_hours = save_frequency_hours
        logger.debug("saving results every %s hours", self.save_frequency_hours)

        # make directory if not already present
        if export_csv_path:
            self.export_csv_path = Path(export_csv_path, simulation_id)
            shutil.rmtree(self.export_csv_path, ignore_errors=True)
            self.export_csv_path.mkdir(parents=True)
        else:
            self.export_csv_path = None

        self.db = None
        self.db_uri = db_uri

        self.learning_mode = learning_mode
        self.evaluation_mode = evaluation_mode

        # get episode number if in learning or evaluation mode
        self.episode = episode
        self.eval_episode = eval_episode

        # construct all timeframe under which hourly values are written to excel and db
        self.start = start
        self.end = end

        self.outputs_buffer_size_bytes = outputs_buffer_size_mb * 1024 * 1024
        self.current_dfs_size_bytes = 0

        # initializes dfs for storing and writing asynchronous
        self.write_buffers: dict = defaultdict(list)
        self.locks = defaultdict(lambda: Lock())

        self._column_cache: dict[str, list[str]] = {}

        self.kpi_defs: dict[str, OutputDef] = {
            "avg_price": {
                "value": "avg(price)",
                "from_table": "market_meta",
            },
            "total_cost": {
                "value": "sum(price*demand_volume_energy)",
                "from_table": "market_meta",
            },
            "total_volume": {
                "value": "sum(demand_volume_energy)",
                "from_table": "market_meta",
            },
            "capacity_factor": {
                "value": "avg(power/max_power)",
                "from_table": 'market_dispatch ud join power_plant_meta um on ud.unit_id = um."index" and ud.simulation=um.simulation',
                "group_bys": ["market_id", "variable"],
            },
        }
        self.kpi_defs.update(additional_kpis)

        # add rl_meta if in learning or evaluation mode
        if self.learning_mode or self.evaluation_mode:
            # Add rl_meta entry to write_buffers
            self.write_buffers["rl_meta"] = [
                {
                    "simulation": self.simulation_id,
                    "evaluation_mode": self.evaluation_mode,
                    "learning_mode": self.learning_mode,
                    "episode": self.episode,
                    "eval_episode": self.eval_episode,
                }
            ]

    def setup(self):
        """
        Sets up the WriteOutput instance by subscribing to messages and scheduling recurrent tasks of storing the data.
        """
        super().setup()

        self.context.subscribe_message(
            self,
            self.handle_output_message,
            lambda content, meta: content.get("context") == "write_results",
        )

    def on_ready(self):
        """
        Called once when the agent/container starts.
        Sets up the database connection (per‐simulation SQLite or Postgres),
        tears down any old data, creates partitions if needed, and schedules
        the periodic store_dfs task.
        """
        # 1) Initialize the engine from URI and an inspector
        if self.db_uri:
            self.db = create_engine(self.db_uri)
            self._inspector = inspect(self.db)

        # 2) Clear out previous data
        if self.db is not None:
            dialect = self.db.dialect.name

            if dialect == "sqlite":
                # SQLite fallback: DELETE all rows for this simulation
                self._purge_simulation_data_sqlite(self.simulation_id)

            elif dialect == "postgresql":
                # Postgres/Timescale: drop & recreate partitions
                self._purge_simulation_data_postgresql(self.simulation_id)
                self._create_partitions(self.simulation_id)

        if self.save_frequency_hours is not None:
            recurrency_task = rr.rrule(
                freq=rr.HOURLY,
                interval=self.save_frequency_hours,
                dtstart=self.start,
                until=self.end,
                cache=True,
            )
            self.context.schedule_recurrent_task(
                self.store_dfs,
                recurrency_task,
                src="no_wait",
            )

    def handle_output_message(self, content: dict, meta: MetaDict):
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta (MetaDict): The metadata associated with the message.
        """
        content_data = content.get("data")
        content_type = content.get("type")
        market_id = content.get("market_id")

        if content_data is None or len(content_data) == 0:
            return

        if content_type in [
            "market_meta",
            "market_dispatch",
            "unit_dispatch",
            "rl_params",
            "rl_critic_params",
        ]:
            # these can be processed as a single dataframe
            self.write_buffers[content_type].extend(content_data)
        elif content_type == "store_units":
            table_name = content_data["unit_type"] + "_meta"
            self.write_buffers[table_name].append(content_data)

        elif content_type == "grid_flows":
            # these need to be converted to df individually
            self.write_buffers[content_type].append(content_data)
        elif content_type in ["market_orders", "grid_topology"]:
            # here we need an additional market_id
            self.write_buffers[content_type].append((content_data, market_id))

        # keep track of the memory usage of the data
        self.current_dfs_size_bytes += calculate_content_size(content_data)
        # if the current size is larger than self.outputs_buffer_size_bytes, store the data
        if self.current_dfs_size_bytes > self.outputs_buffer_size_bytes:
            logger.debug("storing output data due to size limit")
            self.context.schedule_instant_task(coroutine=self.store_dfs())

    def convert_rl_params(self, rl_params: list[dict]):
        """
        Convert the RL parameters such as reward, regret, and profit to a dataframe.

        Args:
            rl_params (dict): The RL parameters.
        """

        df = pd.DataFrame.from_records(rl_params, index="datetime")
        df["simulation"] = self.simulation_id
        df["evaluation_mode"] = self.evaluation_mode
        df["episode"] = self.episode if not self.evaluation_mode else self.eval_episode
        # Add missing rl_critic_params columns in case of initial_exploration
        required_columns = [
            "critic_loss",
            "total_grad_norm",
            "max_grad_norm",
            "learning_rate",
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def convert_market_results(self, market_results: list[dict]):
        """
        Convert market results to a dataframe.

        Args:
            market_meta (dict): The market metadata, which includes the clearing price and volume.
        """
        if len(market_results) == 0:
            return

        df = pd.DataFrame(market_results)
        df["simulation"] = self.simulation_id
        return df

    def convert_market_orders(self, market_orders: any, market_id: str):
        """
        Convert market orders to a dataframe.

        Args:
            market_orders (any): The market orders.
            market_id (str): The id of the market.
        """
        # Check if market orders are empty and exit early
        if not market_orders:
            return

        # Separate orders outside of lock to reduce locking time
        market_orders = separate_orders(market_orders)

        # Construct DataFrame and perform vectorized operations
        df = pd.DataFrame.from_records(market_orders, index="start_time")

        # Replace lambda functions with vectorized operations
        if "eligible_lambda" in df.columns:
            df["eligible_lambda"] = df["eligible_lambda"].map(
                lambda x: getattr(x, "__name__", None)
            )
        if "evaluation_frequency" in df.columns:
            df["evaluation_frequency"] = df["evaluation_frequency"].astype(str)

        # Remove unnecessary columns (use a list to minimize deletion calls)
        df.drop(
            columns=["only_hours", "agent_addr", "contractor_addr"],
            inplace=True,
            errors="ignore",
        )

        # Add missing columns with defaults
        for col in ["bid_type", "node"]:
            if col not in df.columns:
                df[col] = None

        # Add constant columns
        df["simulation"] = self.simulation_id
        df["market_id"] = market_id

        # Append to the shared DataFrame within lock
        return df

    def convert_units_definition(self, unit_info: dict):
        """
        Convert unit definitions to a dataframe.

        Args:
            unit_info (dict): The unit information.
        """
        info = dict(unit_info)
        info.pop("unit_type", None)
        row_id = info.pop("id", None)
        info["simulation"] = self.simulation_id

        return pd.DataFrame([info], index=[row_id])

    def convert_market_dispatch(self, market_dispatch: list[dict]):
        """
        Convert the planned dispatch of the units to a DataFrame.

        Args:
            data (any): The records to be put into the table. Formatted like, "datetime, power, market_id, unit_id".
        """

        df = pd.DataFrame(
            market_dispatch,
            columns=["datetime", "power", "market_id", "unit_id"],
        )
        if not df.empty:
            df["simulation"] = self.simulation_id
        return df

    def convert_unit_dispatch(self, unit_dispatch: list[dict]):
        """
        Convert the actual dispatch of the units to a DataFrame.

        Args:
            unit_dispatch (list): A list of dictionaries containing unit dispatch data.
                                Each dictionary includes arrays for multiple values (e.g., power, costs) and other metadata.
        """

        # Flatten and expand the arrays in `unit_dispatch` into a list of records for DataFrame construction
        records = []
        for dispatch in unit_dispatch:
            time_values = dispatch["time"]
            num_records = len(time_values)

            # Create a record for each time step, expanding array-based fields
            for i in range(num_records):
                record = {
                    key: (value[i] if isinstance(value, (list | np.ndarray)) else value)
                    for key, value in dispatch.items()
                }
                record["time"] = time_values[i]
                records.append(record)

        # Convert the list of records into a DataFrame
        data = pd.DataFrame.from_records(records)

        # Set the index and add the simulation ID
        data.set_index("time", inplace=True)
        data["simulation"] = self.simulation_id

        return data

    def convert_flows(self, data: dict[tuple[datetime, str], float]):
        """
        Convert the flows of the grid results into a dataframe.

        Args:
            data: The records to be put into the table. Formatted like, "(datetime, line), flow" if generated by pyomo or df if it comes from pypsa.
        """
        # Daten in ein DataFrame umwandeln depending on the data format which differs when different solver are used
        # transformation done here to avoid adapting format during clearing

        # if data is dataframe
        if isinstance(data, pd.DataFrame):
            df = data

        # if data is list
        elif isinstance(data, list):
            df = pd.DataFrame.from_dict(data)
        elif isinstance(data, dict):
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame.from_dict(
                data, orient="index", columns=["flow"]
            ).reset_index()
            # Split the 'index' column into 'timestamp' and 'line'
            df[["datetime", "line"]] = pd.DataFrame(
                df["index"].tolist(), index=df.index
            )
            # Rename the columns
            df = df.drop(columns=["index"])

            # set timestamp to index
            df.set_index("datetime", inplace=True)

        df["simulation"] = self.simulation_id

        return df

    async def store_dfs(self):
        """
        Stores the data frames to CSV files and the database. Is scheduled as a recurrent task based on the frequency.
        """
        if not self.db and not self.export_csv_path:
            return

        # If both rl_critic_params and rl_params exist, merge them before uploading to db
        if (
            "rl_params" in self.write_buffers
            and "rl_critic_params" in self.write_buffers
        ):
            df1 = pd.DataFrame(self.write_buffers["rl_params"])
            df2 = pd.DataFrame(self.write_buffers["rl_critic_params"])
            merged_df = pd.merge(df1, df2, how="outer")
            merged_list = merged_df.to_dict("records")
            self.write_buffers["rl_params"] = merged_list
            del self.write_buffers["rl_critic_params"]
        # elif only rl_critic_params exist, rename them to rl_params
        elif "rl_critic_params" in self.write_buffers:
            self.write_buffers["rl_params"] = self.write_buffers["rl_critic_params"]
            del self.write_buffers["rl_critic_params"]

        for table, data_list in self.write_buffers.items():
            if len(data_list) == 0:
                continue
            df = None
            with self.locks[table]:
                if table == "grid_topology":
                    for grid_data, market_id in data_list:
                        self.store_grid(grid_data, market_id)
                    data_list.clear()
                    continue

                match table:
                    case "market_meta":
                        df = self.convert_market_results(data_list)
                    case "market_dispatch":
                        df = self.convert_market_dispatch(data_list)
                    case "unit_dispatch":
                        df = self.convert_unit_dispatch(data_list)
                    case "rl_params":
                        df = self.convert_rl_params(data_list)
                    case "rl_meta":
                        df = pd.DataFrame(data_list)
                    case "grid_flows":
                        dfs = []
                        for data in data_list:
                            df = self.convert_flows(data)
                            dfs.append(df)
                        df = pd.concat(dfs, axis=0, join="outer")
                    case "market_orders":
                        dfs = []
                        for market_data, market_id in data_list:
                            df = self.convert_market_orders(market_data, market_id)
                            dfs.append(df)
                        df = pd.concat(dfs, axis=0, join="outer")
                    case _:
                        # store_units has the name of the units_meta
                        dfs = []
                        for data in data_list:
                            df = self.convert_units_definition(data)
                            dfs.append(df)
                        df = pd.concat(dfs, axis=0, join="outer")
                data_list.clear()

            if df is None or df.empty:
                continue

            # check and correct dtypes
            if table == "rl_params":
                df = df.apply(convert_tensors)

            float_cols = df.select_dtypes(include=["float64"]).columns
            if len(float_cols):
                df[float_cols] = df[float_cols].astype("float32")

            # export to csv if path is set
            if self.export_csv_path:
                data_path = self.export_csv_path / f"{table}.csv"
                df.to_csv(
                    data_path,
                    mode="a",
                    header=not data_path.exists(),
                    float_format="%.5g",
                )

            # store to db if db is set
            if self.db is not None:
                try:
                    self._copy_df_to_db(table, df)
                except NoSuchTableError:
                    # if the table does not exist, create it
                    self._ensure_table(table, df)
                    # now try again
                    self._copy_df_to_db(table, df)
                except (ProgrammingError, OperationalError, DataError):
                    self._check_columns(table, df)
                    # now try again
                    self._copy_df_to_db(table, df)
                except Exception as e:
                    logger.error("could not write to db: %s", e)
                    continue

        self.current_dfs_size_bytes = 0

    def store_grid(
        self,
        grid: dict[str, pd.DataFrame],
        market_id: str,
    ):
        """
        Stores the grid data to the database.
        This is done once at the beginning for every agent which takes care of a grid.
        """
        if self.db is None:
            return

        with self.db.begin() as db:
            try:
                db.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            except Exception:
                logger.info("tried writing grid data to non postGIS database")
                return

        # Check if 'x' and 'y' columns are in the buses DataFrame
        if "x" in grid["buses"].columns and "y" in grid["buses"].columns:
            grid["buses"]["wkt_srid_4326"] = grid["buses"].agg(
                "SRID=4326;POINT ({0[x]} {0[y]})".format, axis=1
            )
            translate_point_dict = grid["buses"]["wkt_srid_4326"].to_dict()
            translate_dict = grid["buses"].agg("{0[x]} {0[y]}".format, axis=1).to_dict()

            def create_line(row):
                return f"SRID=4326;LINESTRING ({translate_dict[row['bus0']]}, {translate_dict[row['bus1']]})"

            # Apply the function to each row
            grid["lines"]["wkt_srid_4326"] = grid["lines"].apply(create_line, axis=1)

            grid_col = "node" if "node" in grid["generators"].columns else "bus"
            grid["generators"]["wkt_srid_4326"] = grid["generators"][grid_col].apply(
                translate_point_dict.get
            )
            grid_col = (
                "node" if "node" in grid["loads"].columns else "bus"
            )  # TODO: anschauen ob da die loads drauf sind
            grid["loads"]["wkt_srid_4326"] = grid["loads"][grid_col].apply(
                translate_point_dict.get
            )
        else:
            logger.warning(
                "Missing 'x' and/or 'y' columns in 'buses' DataFrame. The grid data will not be stored in the dataframe."
            )

        for table, df in grid.items():
            geo_table = f"{table}_geo"
            if df.empty:
                continue
            df["simulation"] = self.simulation_id
            df.reset_index()
            df.columns = df.columns.str.lower()

            try:
                with self.db.begin() as db:
                    df.to_sql(geo_table, db, if_exists="append")
            except (ProgrammingError, OperationalError, DataError, UndefinedColumn):
                # if a column is missing, check and try again
                self._check_columns(geo_table, df)
                # now try again
                with self.db.begin() as db:
                    df.to_sql(geo_table, db, if_exists="append")

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

        queries = []
        for variable, kpi_def in self.kpi_defs.items():
            group_bys = ",".join(kpi_def.get("group_bys", ["market_id"]))
            queries.append(
                f"select '{variable}' as variable, market_id as ident, {kpi_def['value']} as value from {kpi_def['from_table']} where simulation = '{self.simulation_id}' group by {group_bys}"
            )

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
        dfs = [df for df in dfs if not df.empty and df["value"].notna().all()]
        if not dfs:
            return

        df = pd.concat(dfs)
        df.reset_index()
        df["simulation"] = self.simulation_id

        if self.export_csv_path:
            kpi_data_path = self.export_csv_path / "kpis.csv"
            df.to_csv(
                kpi_data_path,
                mode="a",
                header=not kpi_data_path.exists(),
                index=None,
                float_format="%.5g",
            )

        if self.db is not None and not df.empty:
            with self.db.begin() as db:
                df.to_sql("kpis", db, if_exists="append", index=None)

    def get_sum_reward(self, episode: int, evaluation_mode=True):
        """
        Retrieves the total reward for each learning unit.

        Returns:
            np.array: The total reward for each learning unit.
        """
        query = text(
            f"SELECT unit, SUM(reward) FROM rl_params "
            f"WHERE simulation='{self.simulation_id}' "
            f"AND evaluation_mode={evaluation_mode} "
            f"AND episode={episode} "
            f"GROUP BY unit"
        )
        if self.db is None:
            return []

        with self.db.begin() as db:
            rewards_by_unit = db.execute(query).fetchall()

        # convert into a numpy array
        rewards_by_unit = [r[1] for r in rewards_by_unit]
        rewards_by_unit = np.array(rewards_by_unit)

        return rewards_by_unit

    def _get_columns(self, table: str) -> list[str]:
        if table not in self._column_cache:
            self._column_cache[table] = [
                c["name"] for c in self._inspector.get_columns(table)
            ]
        return self._column_cache[table]

    def _copy_df_to_db(self, table: str, df: pd.DataFrame):
        # ---- SQLite fallback ----
        if self.db.dialect.name == "sqlite":
            # simple append via pandas
            df.to_sql(
                name=table,
                con=self.db,
                if_exists="append",
                index=df.index.name is not None,
                index_label=df.index.name,
                method="multi",  # batches INSERTs
            )
            return

        # 1) If there’s a named index (e.g. time/datetime), turn it into a real column
        if df.index.name:
            df = df.reset_index()

        # 2) Introspect the exact column order from Postgres
        cols = self._get_columns(table)

        # 3) Reindex the DataFrame to those columns (missing → NaN)
        df = df.reindex(columns=cols)

        # 4) Serialize CSV with header
        with StringIO() as buffer:
            df.to_csv(buffer, index=False, header=True)
            buffer.seek(0)

            # 5) Bulk‐load via COPY, matching on header
            col_list = ",".join(cols)
            sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
            raw_connection = self.db.raw_connection()
            try:
                cursor = raw_connection.cursor()
                cursor.copy_expert(sql, buffer)
                raw_connection.commit()
            finally:
                cursor.close()

    def _create_partitions(self, simulation_id: str):
        """Create one child‐partition per parent table for this simulation_id."""
        with self.db.begin() as conn:
            for table in PARTITIONED_TABLES:
                partition = f"{table}_{simulation_id}"
                sql = f"""
                CREATE TABLE IF NOT EXISTS {partition}
                  PARTITION OF {table}
                  FOR VALUES IN ('{simulation_id}');
                """
                conn.execute(text(sql))
                logger.debug("created partition %s", partition)

    def _ensure_table(self, table: str, df: pd.DataFrame):
        """
        If `table` doesn’t exist in the DB yet, create it using df.head(0).to_sql(),
        so that future copies will succeed.
        """
        if not self._inspector.has_table(table):
            # Use zero‐row to create the right columns & types
            df.head(0).to_sql(
                name=table,
                con=self.db,
                if_exists="append",  # create if missing, then do nothing
                index=bool(df.index.name),
                method=None,
            )

    def _check_columns(self, table: str, df: pd.DataFrame, index: bool = True):
        """
        Checks and adds columns to the database table if necessary.

        Args:
            table (str): The name of the database table.
            df (pandas.DataFrame): The DataFrame to be checked.
        """
        with self.db.begin() as db:
            # Read table into Pandas DataFrame
            query = f"select * from {table} where 1=0"
            db_columns = pd.read_sql(query, db).columns

        for column in df.columns:
            if column.lower() not in db_columns:
                try:
                    # TODO this only works for float and text
                    if is_bool_dtype(df[column]):
                        column_type = "boolean"
                    elif is_numeric_dtype(df[column]):
                        column_type = "float"
                    else:
                        column_type = "text"
                    query = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                    with self.db.begin() as db:
                        db.execute(text(query))
                except Exception:
                    logger.exception("Error converting column")

        if index and df.index.name:
            df.index.name = df.index.name.lower()
            if df.index.name in db_columns:
                return
            column_type = "float" if is_numeric_dtype(df.index) else "text"
            query = f"ALTER TABLE {table} ADD COLUMN {df.index.name} {column_type}"
            with self.db.begin() as db:
                db.execute(text(query))

    def _purge_simulation_data_sqlite(self, simulation_id: str):
        """
        SQLite‐only fallback for clearing out old simulation data.
        Deletes rows from every table where simulation = simulation_id,
        but skips any tables that don’t yet exist.
        """
        with self.db.begin() as conn:
            for table in ALL_SIM_TABLES:
                if not self._inspector.has_table(table):
                    logger.debug("Skipping purge: table %s does not exist", table)
                    continue

                sql = text(f'DELETE FROM "{table}" WHERE simulation = :sim')
                res = conn.execute(sql, {"sim": simulation_id})
                logger.debug("Deleted %s rows from %s", res.rowcount, table)

    def _purge_simulation_data_postgresql(self, simulation_id: str):
        """
        TimescaleDB/Postgres cleanup:
        1) Drop any simulation-specific partitions.
        2) Delete all rows in the static tables for that simulation.
        """
        with self.db.begin() as conn:
            # 1) drop per-simulation partitions
            for tbl in PARTITIONED_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {tbl}_{simulation_id};"))

            # 2) delete from the static tables
            static_tables = [t for t in ALL_SIM_TABLES if t not in PARTITIONED_TABLES]
            for tbl in static_tables:
                conn.execute(
                    text(f"DELETE FROM {tbl} WHERE simulation = :sim;"),
                    {"sim": simulation_id},
                )


class DatabaseMaintenance:
    """
    Utility class for managing simulation data in a partitioned TimescaleDB setup.

    Supports:
      - Listing all simulation IDs.
      - Dropping specific simulation partitions.
      - Dropping all simulations (or all except exclusions).

    Tables partitioned by simulation: market_meta, market_dispatch, unit_dispatch,
    rl_params, grid_flows, kpis. Other tables are dropped via DELETE if needed.
    """

    def __init__(self, db_uri: str):
        self.db = create_engine(db_uri)
        self._inspector = inspect(self.db)
        self._table_cache = None

    def _get_tables(self) -> list[str]:
        # List only parent tables (exclude partition children)
        if self._table_cache is None:
            all_tables = self._inspector.get_table_names()
            # exclude system table
            tables = [t for t in all_tables if t != "spatial_ref_sys"]
            # only include parent tables, not partitions of form '<parent>_<sim>'
            parent_tables = []
            for t in tables:
                if any(t.startswith(f"{p}_") for p in PARTITIONED_TABLES):
                    continue
                parent_tables.append(t)
            self._table_cache = parent_tables
        return self._table_cache

    def get_simulation_ids(self) -> list[str]:
        ids = set()
        for table in self._get_tables():
            # skip tables without simulation column
            cols = [c["name"] for c in self._inspector.get_columns(table)]
            if "simulation" not in cols:
                continue
            query = text(f"SELECT DISTINCT simulation FROM {table}")
            with self.db.begin() as conn:
                for (sim,) in conn.execute(query):
                    if sim:
                        ids.add(sim)
        return sorted(ids)

    def delete_simulations(self, simulation_ids: list[str]) -> None:
        if not simulation_ids:
            logger.info("No simulations specified to delete.")
            return
        with self.db.begin() as conn:
            # Drop partitions for partitioned tables
            for table in PARTITIONED_TABLES:
                for sim in simulation_ids:
                    partition = f"{table}_{sim}"
                    conn.execute(text(f"DROP TABLE IF EXISTS {partition};"))
                    logger.debug("Dropped partition %s", partition)
            # For non-partitioned tables, do DELETE
            non_part = set(self._get_tables()) - set(PARTITIONED_TABLES)
            for table in non_part:
                # ensure simulation index
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {table}_simulation_idx"
                        f" ON {table}(simulation)"
                    )
                )
                del_q = text(f"DELETE FROM {table} WHERE simulation = ANY(:s)")
                res = conn.execute(del_q, {"s": simulation_ids})
                logger.debug("Deleted %s rows from %s", res.rowcount, table)

    def delete_all_simulations(self, exclude: list[str] = None) -> None:
        with self.db.begin() as conn:
            # Partitioned: drop partitions
            for table in PARTITIONED_TABLES:
                # list existing partitions
                parts = [
                    p
                    for p in self._inspector.get_table_names()
                    if p.startswith(f"{table}_")
                ]
                to_drop = []
                if exclude:
                    keep = {f"{table}_{s}" for s in exclude}
                    to_drop = [p for p in parts if p not in keep]
                else:
                    to_drop = parts
                for partition in to_drop:
                    conn.execute(text(f"DROP TABLE IF EXISTS {partition};"))
                    logger.debug("Dropped partition %s", partition)
            # Non-partitioned: delete or exclude
            non_part = set(self._get_tables()) - set(PARTITIONED_TABLES)
            for table in non_part:
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {table}_simulation_idx"
                        f" ON {table}(simulation)"
                    )
                )
                if exclude:
                    q = text(f"DELETE FROM {table} WHERE simulation NOT IN :ex")
                    res = conn.execute(q, {"ex": exclude})
                else:
                    res = conn.execute(text(f"DELETE FROM {table}"))
                logger.debug("Deleted %s rows from %s", res.rowcount, table)
