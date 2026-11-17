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

Market timing
-------------

Before running a simulation, the timing of the markets of a study case can be shown and checked with ``assume-market``.
It renders when each market opens, is cleared and delivers, and prints the timing issues which are also checked when a simulation starts.
It exits with code 1 if the timing contains errors.

.. code-block:: console

   assume-market examples/inputs/example_01a/config.yaml --case-study dam

.. argparse::
   :filename: ../../assume_cli/cli.py
   :func: create_market_parser
   :prog: assume-market
