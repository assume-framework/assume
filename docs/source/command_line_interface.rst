.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

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
