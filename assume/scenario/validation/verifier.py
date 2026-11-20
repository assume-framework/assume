# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import cast

from assume.scenario.validation.config import ConfigChecker, StudyCase
from assume.scenario.validation.folder import ScenarioFolder
from assume.scenario.validation.grid import GridChecker
from assume.scenario.validation.report import ValidationReport
from assume.scenario.validation.tables import BiddingTable, CaseFiles
from assume.scenario.validation.utils import quiet_logging


class ScenarioVerifier:
    """Verifies that all study cases of a scenario folder can be loaded into ASSUME."""

    def __init__(self, scenario_dir: str | Path):
        self.folder = ScenarioFolder(scenario_dir)
        self.checker = ConfigChecker(self.folder)

    def run(self, dry_run: bool = True) -> ValidationReport:
        """
        Run the static checks and, if they found no errors, the dry-run load.

        Args:
            dry_run (bool, optional): Whether to load each study case into a ``World``. Defaults to True.
        """
        report = self.folder.report
        cases = self.checker.load_cases()
        report.cases = [case.name for case in cases]
        self._check_static(cases)
        if dry_run and report.ok:
            for case in cases:
                self._dry_run(case)
        return report

    def _check_static(self, cases: list[StudyCase]) -> None:
        for case in cases:
            self.checker.check(case)
        files = [CaseFiles(self.folder, case) for case in cases]

        # unit and operator files are shared between study cases: check each file once.
        # Their bidding columns may refer to the markets of any study case.
        markets = set().union(*(case.market_ids for case in cases))
        shared: dict[Path, BiddingTable] = {}
        for case_files in files:
            for table in case_files.units.values():
                shared.setdefault(table.path, table)
        for table in shared.values():
            table.check(markets)

        # time series depend on the time line, so they are checked per study case
        for case_files in files:
            case_files.check()
        GridChecker(self.folder, cases).check()
        self.folder.check_unused_files([case.config for case in cases])

    def _dry_run(self, case: StudyCase) -> None:
        """Load a study case into a World exactly like a simulation would, but do not run it."""
        from assume.scenario.loader_csv import (
            load_config_and_create_forecaster,
            setup_world,
        )
        from assume.world import World

        path = self.folder.path
        world = World(log_level="WARNING")
        try:
            with quiet_logging():
                scenario_data = load_config_and_create_forecaster(
                    str(path.parent), path.name, case.name
                )
                # verifying must not write files into the scenario folder
                config = cast(dict[str, object], scenario_data["config"])
                config["save_forecasts"] = False
                world.scenario_data = scenario_data
                setup_world(world)
        except Exception as e:
            case.reporter("dry-run").error(
                f"cannot be loaded into ASSUME: {type(e).__name__}: {e}"
            )
        finally:
            if not world.loop.is_closed():
                world.loop.close()


def verify_scenario_folder(
    scenario_dir: str | Path, dry_run: bool = True
) -> ValidationReport:
    """
    Verify that all study cases of a scenario folder (``config.yaml`` and csv files) can be loaded.

    All study cases are verified together, as they share the csv files of the folder: a file is
    only unused if no study case reads it, and a ``bidding_<market>`` column may refer to the
    market of any study case.

    Args:
        scenario_dir (str | Path): The folder of the scenario, e.g. ``examples/inputs/example_01a``.
        dry_run (bool, optional): Also load each case into a ``World`` after the static checks passed.
            Defaults to True.

    Returns:
        ValidationReport: All errors and warnings found; ``report.ok`` is True if there are no errors.
    """
    return ScenarioVerifier(scenario_dir).run(dry_run=dry_run)
