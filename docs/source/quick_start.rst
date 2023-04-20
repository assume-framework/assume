###########################
Quick Start
###########################

For installation instructions see :doc:`installation`.

To run an exemplar simulation without database and grafana, run the following command::

    python examples/example_01/example_01.py


If you have also built the docker container, run the following command::

    docker compose up -d
    python examples/example_02/example_02.py

Afterwards you can access the Dashboard on `http://localhost:3000`