################
 Installation
################

Getting Python
==============

If it is your first time with Python, we recommend `conda
<https://docs.conda.io/en/latest/miniconda.html>`_ as easy-to-use package managers. It is
available for Windows, Mac OS X and GNU/Linux.

Installation
============

You can install ASSUME using pip. Choose the appropriate
installation method based on your needs:

Using pip
---------

To install the core package::

    pip install assume-framework

To install with testing capabilities::

    pip install assume-framework[test]

Timescale Database and Grafana Dashboards
-----------------------------------------

If you want to benefit from a supported database and integrated
Grafana dashboards for scenario analysis, you can use the provided
Docker Compose file.

Follow these steps:

1. Clone the repository and navigate to its directory::

    git clone https://github.com/assume-framework/assume.git
    cd assume

2. Start the database and Grafana using the following command::

    docker-compose up -d

This will launch a container for TimescaleDB and Grafana with
preconfigured dashboards for analysis. You can access the Grafana
dashboards at `http://localhost:3000`.

Using Learning Capabilities
---------------------------

If you intend to use the reinforcement learning capabilities of
ASSUME and train your agents, make sure to install Torch. Detailed
installation instructions can be found `here <https://pytorch.org/get-started/locally/>`_.
