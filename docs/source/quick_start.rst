.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

###########################
Quick Start
###########################

For installation instructions see :doc:`installation`.

Exploring Examples
==================

To explore the provided examples, follow these steps:

1. Clone the repository and navigate to its directory:

.. code-block:: bash

   git clone https://github.com/assume-framework/assume.git
   cd assume

2. Run a simulation:

   There are three ways to run a simulation:

   - Local:

   .. code-block:: bash

      python examples/examples.py

   - Using the provided Docker setup:

     If you have installed Docker and set up the Docker Compose file previously, you can select 'timescale' in ``examples.py`` before running the simulation. This will save the simulation results in a Timescale database, and you can access the Dashboard at http://localhost:3000.

   - Using the CLI to run simulations:

   .. code-block:: bash

      assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume"

     For additional CLI options, run ``assume -h``.


Running tests
=============

Install the testing packages after checking out the repo::

    pip install -e .[test]

Run pytest with coverage to run all tests and produce a coverage report::

    pytest --cov

All tests should pass locally.
If they are not working, the Continuous Integration (CI) pipeline will fail.

Building Docs
=============

Create the Docs environment::

    conda env create -f environment_docs.yaml

Then you can build the docs using the Makefile in the docs directory::

    cd docs
    make html

Finally, to serve the build directory locally, run::

    python -m http.server --directory build/html

Now you can visit http://localhost:8000 to see the working docs locally.
