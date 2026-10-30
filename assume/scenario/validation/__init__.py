# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Verification of scenario folders, exposed on the command line as ``assume --verify <dir>``.

A scenario folder is verified in two stages:

1. **Static checks** read ``config.yaml`` and all referenced csv files *without* building
   a simulation. They report every problem found (not just the first one) with the file
   and the location of the problem:

   * required files and config keys exist, unknown keys/columns are flagged (with a hint
     for typos),
   * values have the expected type (number, string, bool, ...),
   * csv files are parseable, unit names are unique, required unit columns are present,
     and time series cover the simulation horizon.

   The expected columns and types of the unit files are read from the ``__init__``
   signatures of the unit classes, and those of ``learning_config`` from
   :class:`~assume.common.base.LearningConfig`. They therefore cannot drift from the code.

2. **Dry-run load.** If the static checks found no errors, every study case is loaded into a
   :class:`~assume.world.World` exactly like the simulation would do it, but never run.
   This catches everything the static checks do not know about (e.g. a strategy that
   rejects its parameters) and is the final answer to "can this be loaded into ASSUME".

The package is organised by what is checked:

* ``verifier``: :class:`ScenarioVerifier` runs the stages, ``verify_scenario_folder`` is the entry point,
* ``config``: :class:`ConfigChecker` for ``config.yaml`` (keys, types, markets, learning),
* ``tables``: one :class:`~assume.scenario.validation.tables.Table` class per kind of csv file,
  and :class:`~assume.scenario.validation.tables.CaseFiles` for how the files of a study case fit together,
* ``folder``: :class:`ScenarioFolder`, reading the files like the loader does,
* ``schema``, ``report``, ``utils``: declared keys and files, the result, and small helpers.
"""

from assume.scenario.validation.report import Issue, ValidationReport
from assume.scenario.validation.verifier import verify_scenario_folder

__all__ = ["Issue", "ValidationReport", "verify_scenario_folder"]
