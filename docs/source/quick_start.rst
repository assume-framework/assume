###########################
Quick Start
###########################

For installation instructions see :doc:`installation`.

To run an exemplar simulation without database and grafana, run the following command::

    python examples/example_01_sqlite.py


If you have also built the docker container, run the following command::

    docker compose up -d
    python examples/example_01_timescale.py

Afterwards you can access the Dashboard on `http://localhost:3000`


Using the command line interface (CLI)
======================================

Instead of using one of the written examples, you can also use the CLI::

    assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume" -l DEBUG

This will run the example_01b and write the output into the local PostgreSQL database.
All Debug logs will be written (to the file assume.log).
To take a look at options which can be used for this command run::

    assume --help


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

    conda env create -f environment_docs.yml

Then you can build the docs using the Makefile in the docs directory::

    cd docs
    make html

Finally, to serve the build directory locally, run::

    python -m http.server --directory build/html

Now you can visit http://localhost:8000 to see the working docs locally
