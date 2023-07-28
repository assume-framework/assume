# ASSUME

**ASSUME** is an open-source tool-box for an agent-based simulation
of the European electricity and later on energy markets with a main
focus on the German market setup. As a newly developed open-source
model its priority is to ensure usability and customisability
for a variety of users and use cases in the energy system modelling community.

The unique feature of the ASSUME tool-box is the integration of **deep reinforcement
learning** methods into the behavioural strategies of market agents.
The model comes with different predefined agent representations for the demand and
generation side that can be used as plug-and-play modules,
which facilitate reinforcement of learning strategies.
This setup enables a research of new market designs and dynamics in the energy markets.

The project is developed by [developers](https://assume.readthedocs.io/en/latest/developers.html) from INATECH at University of Freiburg, IISM at KIT, Fraunhofer ISI, FH Aachen.
The project ASSUME is funded by the Federal Ministry for Economic
Affairs and Energy (BMWK).

Documentation
=============

[Documentation](https://assume.readthedocs.io/en/latest/)

[Installation](https://assume.readthedocs.io/en/latest/installation.html)


Installation
============

Using conda
-----------

First clone the repository:

```
git clone https://github.com/assume-framework/assume.git
```

Next, navigate to the cloned directory:

```
cd $where you cloned the repo$
```

Now, create a conda environment:

```
conda env create -f environment.yml
```

Afterwards, activate the environment:
```
conda activate assume-framework
```

After these steps you can also run the example simulation:

```
python examples/example_01a_sqlite.py
```

You can also use the cli to run simulations:

```
assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume"
```

If you need help using the CLI run `assume -h`


Please note that if you have python running on windows that you need to alter the same_process binary in world to True (to be changed).


Development
-----------

install pre-commit

```
pip install pre-commit
pre-commit install
```

Use Learning Capabilities
---------------------------------
If you want to use the reinforcement learning strategies of Assume please make sure to install torch on top of the requirements listed in the requirements.txt. If you Ressources do not have a GPU available use the CPU only version of torch, which can be installed like this:

pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Access to database and dashboards
---------------------------------
To save the simulation results to a database and be able to analyze them using Grafan dashboards, install the docker container:

```
docker compose up
```

This will start a container for timescaledb and grafana with preconfigured grafana dashboard.

Quick Start
-----------

To run an exemplar simulation without database and grafana, run the following command:
```
    python examples/example_01a_sqlite.py
```

If you have docker installed, you can run the following two commands:
```
    docker compose up -d
    python examples/example_01a_timescale.py
```

Afterwards you can access the Dashboard on `http://localhost:3000`

Licence
=======

Copyright 2022-2023 [ASSUME developers](https://assume.readthedocs.io/en/latest/developers.html)

ASSUME is licensed under the [GNU Affero General Public License v3.0](./LICENSE)
