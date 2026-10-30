.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

Assume CLI
==========

Submodules
----------

Command Line Interface
-------------------------

There is a command line interface available to assume which makes it easy to run simulations from the CLI.
It takes an input folder relative from the place it was run from.

The following command line parameters are available in the CLI:

.. argparse::
   :filename: ../../assume_cli/cli.py
   :func: create_parser
   :prog: assume

Verifying a scenario folder
---------------------------

Before running a simulation, ``assume --verify <dir>`` checks that a scenario folder
(the folder containing ``config.yaml`` and the csv files) can be loaded into ASSUME.
Nothing is simulated and nothing is written to the folder.

.. code-block:: bash

   assume --verify examples/inputs/example_01a
   assume --verify examples/inputs/example_01a -c tiny   # only one study case

The verification runs in two stages:

1. **Static checks** report *all* problems at once, each with the file and the location:

   - required files exist (``config.yaml``, ``powerplant_units.csv``, ``demand_units.csv``, and files named in the config),
   - ``config.yaml`` and every study case declare the required keys, have no unknown keys (typos are suggested)
     and the values have the right type (number, string, bool, duration, ...),
   - csv files are readable, unit names are unique, the required unit columns exist and numeric columns contain numbers,
   - bidding strategies and market mechanisms exist,
   - time series cover the simulation horizon and match the units (e.g. ``exchange_import`` columns for exchange units).

   The expected unit columns and types are read from the unit classes and ``learning_config`` from ``LearningConfig``,
   so they always match the code.

2. **Dry-run load**: if there are no errors, each study case is loaded into a ``World`` exactly like a simulation does, but not run.

Errors mean the scenario cannot be loaded or would silently ignore a file, and make the command exit with status 1.
Warnings point to likely mistakes (e.g. an ignored column) and keep the exit status at 0.
