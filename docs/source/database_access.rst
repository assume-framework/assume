.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

##############################
Accessing the Output Database
##############################

After running a simulation, ASSUME stores all results in a PostgreSQL/TimescaleDB database.
This page explains how to connect to the database and query simulation results using Python,
the VSCode PostgreSQL extension, or the PyCharm Database tool.

For an overview of the database tables and columns, see :doc:`outputs`.

Prerequisites
=============

Before accessing the database, ensure that the TimescaleDB container is running:

.. code-block:: bash

   docker compose up -d

The default connection parameters are:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Host
     - ``localhost``
   * - Port
     - ``5432``
   * - Database
     - ``assume``
   * - Username
     - ``assume``
   * - Password
     - ``assume``

This results in the connection URI: ``postgresql://assume:assume@localhost:5432/assume``


Connecting via Python
=====================

Use ``sqlalchemy`` and ``pandas`` to connect to the database and query results as DataFrames.

Setting Up the Connection
-------------------------

.. code-block:: python

   from sqlalchemy import create_engine
   import pandas as pd

   db_uri = "postgresql://assume:assume@localhost:5432/assume"
   db = create_engine(db_uri)

Listing Available Simulations
-----------------------------

Each simulation run is identified by a unique simulation ID (composed of ``scenario_studycase``).
To list all available simulation IDs:

.. code-block:: python

   query = "SELECT DISTINCT simulation FROM market_meta;"
   simulations = pd.read_sql(query, db)
   print(simulations)

Querying Market Results
-----------------------

To retrieve market clearing data (prices, volumes) for a specific simulation:

.. code-block:: python

   scenario = "example_01"
   study_case = "base"
   simulation = f"{scenario}_{study_case}"

   query = f"SELECT * FROM market_meta WHERE simulation = '{simulation}';"
   market_meta_df = pd.read_sql(query, db)
   market_meta_df = market_meta_df.sort_values("time")

   print(market_meta_df[["time", "price", "demand_volume", "supply_volume"]])

Connecting via VSCode
=====================

The `PostgreSQL extension <https://marketplace.visualstudio.com/items?itemName=ckolkman.vscode-postgres>`_
for Visual Studio Code allows you to browse and query the database directly from the editor.

1. Install the **PostgreSQL** extension (by Chris Kolkman) from the VSCode marketplace.
2. Open the Command Palette (``Ctrl+Shift+P``) and select **PostgreSQL: Add Connection**.
3. Enter the connection parameters:

   - **Host**: ``localhost``
   - **Port**: ``5432``
   - **Database**: ``assume``
   - **Username**: ``assume``
   - **Password**: ``assume``
   - **Use SSL**: No

4. Once connected, you can browse tables in the PostgreSQL Explorer sidebar.
5. Right-click a table to select rows, or open a new SQL editor to run custom queries.

Example query in the SQL editor:

.. code-block:: sql

   SELECT time, price, demand_volume, supply_volume
   FROM market_meta
   WHERE simulation = 'example_01_base'
   ORDER BY time;


Connecting via PyCharm
======================

PyCharm Professional includes a built-in Database tool that supports PostgreSQL.

1. Open **View > Tool Windows > Database**.
2. Click the **+** icon and select **Data Source > PostgreSQL**.
3. If prompted, download the PostgreSQL JDBC driver.
4. Enter the connection parameters:

   - **Host**: ``localhost``
   - **Port**: ``5432``
   - **Database**: ``assume``
   - **User**: ``assume``
   - **Password**: ``assume``

5. Click **Test Connection** to verify, then **OK**.
6. The database tables appear in the Database tool window. Double-click a table to view its contents.
7. Right-click the data source and select **New > Console** to open a SQL console for custom queries.
