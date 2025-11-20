.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

Scenario Loader
===============

For compatibility with other simulation tools, ASSUME provides a variety of scenario loaders. These are:

- :ref:`csv` - File based scenarios (most flexible)
- :ref:`amiris` - used to create comparative studies
- :ref:`oeds` - create scenarios with the Open Energy Data Server
- :ref:`pypsa_loader_doc` - create scenarios from imported PyPSA networks


The possibilities and a short usage guide of the different scenario loaders are explained below:


.. _csv:

CSV
---

The CSV loader is the default scenario loader for ASSUME. Everything is configured through a ``config.yaml`` file, which describes a market design and references the input series of the agents, as well as the bidding strategies used.

It is introduced in :doc:`this example </examples/02_automated_run_example>`, where a small simulation is created from scratch.

If you already have an existing csv scenario, you can load it using the ASSUME CLI like:

``assume -c tiny -s example_01a --db-uri postgresql://assume:assume@localhost:5432/assume``

.. _amiris:

AMIRIS
------

The AMIRIS loader can be used to run examples configured for usage with the energy market simulation tool `AMIRIS by the DLR <https://gitlab.com/dlr-ve/esy/amiris/amiris>`_.

.. code-block:: python

    from assume import World
    from assume.scenario import load_amiris

    # To download some amiris examples run:
    # git clone https://gitlab.com/dlr-ve/esy/amiris/examples.git amiris-examples
    # next to the assume folder
    base_path = f"../amiris-examples/Germany2019"

    # Read the scenario from this base path
    amiris_scenario = read_amiris_yaml(base_path)

    # Configure where to write the output
    db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # Create a simulation world
    world = World(database_uri=db_uri)

    default_strategy = {mc.market_id: "powerplant_energy_naive" for mc in marketdesign}
    default_demand_strategy = {mc.market_id: "demand_energy_naive" for mc in marketdesign}

    bidding_strategies = {
        "hard coal": default_strategy,
        "lignite": default_strategy,
        "oil": default_strategy,
        "gas": default_strategy,
        "biomass": default_strategy,
        "nuclear": default_strategy,
        "wind": default_strategy,
        "solar": default_strategy,
        "demand": default_demand_strategy,
    }

    # Let the loader add everything to the world
    world.loop.run_until_complete(
        load_amiris(
            world,
            "amiris",
            scenario,
            base_path,
        )
    )

    # Run the scenario
    world.run()

This makes it possible to compare or validate results from AMIRIS.
If you want to adjust the scenario or change bidding strategies, you currently have to adjust the amiris loader accordingly,
as it currently does not use reinforcement learning or different bidding strategies at all.
It tries to resemble the behavior of AMIRIS in the best way possible.
As AMIRIS currently only supports a single market design (with different support mechanisms), the market design can not be adjusted.
For more information consult the methods documentation :py:meth:`assume.scenario.loader_amiris.load_amiris`.

.. _oeds:

OEDS
----

`The Open-Energy-Data-Server <https://github.com/NOWUM/open-energy-data-server/>`_ is a tool that facilitates the aggregation of open research data in a way that allows for easy reuse and structured work. It includes data from the `Marktstammdatenregister of Germany <https://www.marktstammdatenregister.de/MaStR/Datendownload>`_, `ENTSO-E <https://transparency.entsoe.eu/>`_, and weather datasets, making it versatile for modeling different localized scenarios.

Once you have an Open-Energy-Data-Server running, you can query data for various scenarios and interactively compare your simulation results with the actual data recorded by ENTSO-E using Grafana.

The main configuration required for the Open-Energy-Data-Server involves specifying the `NUTS areas <https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics>`_ that should be simulated, as well as a marketdesign.
An example configuration of how this can be used is shown here:

.. code-block:: python

    # where to write the simulation output to - can also be the oeds
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    # adjust to your institute's database server
    infra_uri = "postgresql://readonly:readonly@myoeds-server:5432"

    # you can also just use ["DE"] for a simulation of germany with single agents per generation technology
    nuts_config = ["DE1", "DEA", "DEB", "DEC", "DED", "DEE", "DEF"]

    # define a marketdesign which can be used for the simulation
    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=24, dtstart=start, until=end),
            timedelta(hours=1),
            "pay_as_clear",
            [MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
            additional_fields=["block_id", "link", "exclusive_id"],
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
        )
    ]

    default_strategy = {mc.market_id: "powerplant_energy_naive" for mc in marketdesign}
    default_demand_strategy = {mc.market_id: "demand_energy_naive" for mc in marketdesign}

    bidding_strategies = {
        "hard coal": default_strategy,
        "lignite": default_strategy,
        "oil": default_strategy,
        "gas": default_strategy,
        "biomass": default_strategy,
        "nuclear": default_strategy,
        "wind": default_strategy,
        "solar": default_strategy,
        "demand": default_demand_strategy,
    }

    # load the dataset from the database
    load_oeds(world, "oeds_mastr_simulation", "my_studycase", infra_uri, marketdesign, nuts_config)

    # Run the scenario
    world.run()

This creates operators each per NUTS areas and creates a single EOM market, just as the `DMAS simulation <https://github.com/NOWUM/dmas/>`_ from FH Aachen.
For more information consult the methods documentation :py:meth:`assume.scenario.loader_oeds.load_oeds`.

.. _pypsa_loader_doc:

PyPSA
-----

The `PyPSA <https://github.com/PyPSA/pypsa/>`_ loader can be used to load a scenario from a configured PyPSA network.

The components for `generators`, `loads`, `buses`, `lines`, `storage_operators` and so on have to be configured.
Operation values have to be given through the `generators_t` and `loads_t` param of the pypsa network.

It makes it possible to load for example from PyPSA CSV files using :py:meth:`pypsa.Network.import_from_csv_folder`

An example can be seen from the pypsa scigrid case:

.. code-block:: python

    from assume.scenario.loader_pypsa import load_pypsa
    from assume import World
    import pypsa

    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    network = pypsa.examples.scigrid_de(from_master=True)
    start = network.snapshots[0]
    end = network.snapshots[-1]
    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=1, dtstart=start, until=end),
            timedelta(hours=1),
            "redispatch",
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["node"],
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
        )
    ]

    bidding_strategies = {
        "hard coal": "powerplant_energy_naive_redispatch",
        "lignite": "powerplant_energy_naive_redispatch",
        "oil": "powerplant_energy_naive_redispatch",
        "gas": "powerplant_energy_naive_redispatch",
        "biomass": "powerplant_energy_naive_redispatch",
        "nuclear": "powerplant_energy_naive_redispatch",
        "wind": "powerplant_energy_naive_redispatch",
        "solar": "powerplant_energy_naive_redispatch",
        "demand": "demand_energy_naive_redispatch",
    }
    load_pypsa(world, scenario, study_case, network, marketdesign, bidding_strategies)
    load_pypsa(world, "world_pypsa", "scigrid_de", network, marketdesign)

    world.run()

You can also create and use your own existing scenarios in pypsa format to convert these into a market simulation too.

For more information consult the methods documentation :py:meth:`assume.scenario.loader_pypsa.load_pypsa`.
