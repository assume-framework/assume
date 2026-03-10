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

The `SQLTools extension <https://marketplace.visualstudio.com/items?itemName=mtxr.sqltools>`_
for Visual Studio Code supports both PostgreSQL and SQLite, allowing you to browse and query
the database directly from the editor.

1. Install the **SQLTools** extension (by Matheus Teixeira) from the VSCode marketplace.
2. Install the appropriate driver extension:

   - For PostgreSQL: `SQLTools PostgreSQL/Cockroach Driver <https://marketplace.visualstudio.com/items?itemName=mtxr.sqltools-driver-pg>`_
   - For SQLite: `SQLTools SQLite Driver <https://marketplace.visualstudio.com/items?itemName=mtxr.sqltools-driver-sqlite>`_

3. After installing, VSCode may show pop-ups asking to install additional dependencies or to
   reload the window. Accept these prompts to complete the setup.
4. Open the Command Palette (``Ctrl+Shift+P``) and select **SQLTools Management: Add New Connection**.

5. **For PostgreSQL:** Select **PostgreSQL** as the driver and enter the connection parameters:

   - **Connection Name**: ``ASSUME``
   - **Server**: ``localhost``
   - **Port**: ``5432``
   - **Database**: ``assume``
   - **Username**: ``assume``
   - **Password**: ``assume``

   **For SQLite:** Select **SQLite** as the driver. If prompted to install the SQLite binary or
   Node dependencies, accept the installation and reload the window when asked. Enter the
   connection parameters:

   - **Connection Name**: ``ASSUME Local``
   - **Database File**: path to your SQLite file, e.g. ``./examples/local_db/assume_db.db``

6. Click **Test Connection** to verify, then **Save Connection**.

.. note::
   The ASSUME database uses the ``public`` schema. TimescaleDB creates additional internal
   schemas (e.g. ``timescaledb_information``, ``_timescaledb_internal``). These can be ignored.
   If the SQLTools explorer shows multiple schemas, only browse tables under **public**.

Browsing and Querying
----------------------

Once connected, you can browse tables in the SQLTools sidebar.
Click a table to preview its data, or open a new SQL editor to run custom queries.

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
6. In the **Schemas** tab, ensure only the ``public`` schema is checked. Uncheck any
   TimescaleDB internal schemas (e.g. ``timescaledb_information``, ``_timescaledb_internal``)
   to keep the database explorer clean.
7. The database tables appear in the Database tool window. Double-click a table to view its contents.
8. Right-click the data source and select **New > Console** to open a SQL console for custom queries.
