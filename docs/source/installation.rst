.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

################
 Installation
################

You can install ASSUME using pip or docker. Choose the appropriate
installation method based on your needs.

Using Learning Capabilities
---------------------------

If you intend to use the reinforcement learning capabilities of
ASSUME and train your agents, make sure to install Torch. Detailed
installation instructions can be found `here <https://pytorch.org/get-started/locally/>`_.

Using Pip/Python
================

In general most people use an environment manager for python.
Here, Install instructions for Conda and Venv are given.

If it is your first time with Python, we recommend `conda
<https://docs.conda.io/en/latest/miniconda.html>`_ as easy-to-use package managers. It is
available for Windows, Mac OS X and GNU/Linux.

With Conda
----------

After installing conda, you can create an environment for assume like::

    conda create -n assume python=3.12

accept the installed packages and activate the newly created environment::

    conda activate assume

Now you can proceed with the assume installation using pip below.

With Venv
-----------------

Venvs are the python native way of handling environments using the same python version.
To activate a venv, you have to be in the same folder where you created it.

Using venvs, we can create a new environment::

    python -m venv venv

And activate it::

    source venv/bin/activate

Now you can proceed with the assume installation using pip below.


Install using pip
-----------------

To install the core package::

    pip install assume-framework

To install with learning capabilities::

    pip install assume-framework[learning]

To install with optimization capabilities::

    pip install assume-framework[optimize]

To install with testing capabilities::

    pip install assume-framework[test]

Install Tab-Completion
----------------------

ASSUME uses `argcomplete` for argument completion on the CLI.

On Windows, one needs to run:

`register-python-argcomplete --shell powershell assume | Out-String | Invoke-Expression`

in the used conda environment, to install tab completions.

On Bash or zsh (Linux and Mac) run the following in the correct conda environment with assume and argcomplete installed:

`eval "$(register-python-argcomplete assume)"`


Install using Docker
=========================================

After installing `Docker Engine <https://docs.docker.com/engine/install/>`_ you can run::
    docker compose up -d assume_db grafana
    docker compose run simulation -s example_01a -db "postgresql://assume:assume@assume_db:5432/assume"

Now you can see the simulation results directly on grafana at `http://localhost:3000`

Timescale Database and Grafana Dashboards
=========================================

ASSUME supports writing into Databases supported by `SQLAlchemy <https://docs.sqlalchemy.org/en/latest/dialects/index.html>`_ and CSV files.

If you want to benefit from a supported database and integrated
Grafana dashboards for scenario analysis, you can use the provided
Docker Compose file.

Follow these steps:

1. Clone the repository and navigate to its directory::

    git clone https://github.com/assume-framework/assume.git
    cd assume

2. Start the database and Grafana using the following command::

    docker compose up -d

This will launch a container for TimescaleDB and Grafana with
preconfigured dashboards for analysis. You can access the Grafana
dashboards at `http://localhost:3000`. Here are some exampels of the general grafana dashboard.

.. image:: ./img/Grafana_General.png
   :width: 100%
   :alt: Grafana Dashboard

.. image:: ./img/Grafana_General_2.png
   :width: 100%
   :alt: Grafana Dashboard 2
