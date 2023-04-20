################
 Installation
################


Getting Python
==============

If it is your first time with Python, we recommend `conda
<https://docs.conda.io/en/latest/miniconda.html>`_ as easy-to-use package managers. It is 
available for Windows, Mac OS X and GNU/Linux.


Get started
===========

Using conda
-----------

First clone the repository::

    git clone https://github.com/assume-framework/assume.git


Next, navigate to the cloned directory::

    cd $where you cloned the repo$

Now, create a conda environment::

    conda env create -f environment.yml

Afterwards, activate the environment::
    
    conda activate assume-framework

After these steps you can also run the example simulation::
    
    python examples/example_01.py

Access to database and dashboards
--------------------------------
To save the simulation results to a database and be able to analyze them using Grafan dashboards, 
install the docker container::

    docker compose up --build

This will start a container for timescaledb and grafana with preconfigured grafana dashboard.
