# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import cast

from assume.scenario.validation.config import ConfigChecker, StudyCase
from assume.scenario.validation.folder import ScenarioFolder
from assume.scenario.validation.report import ValidationReport
from assume.scenario.validation.tables import CaseFiles
from assume.scenario.validation.utils import quiet_logging


class ScenarioVerifier:
    """Verifies that the study cases of a scenario folder can be loaded into ASSUME."""

    def __init__(self, scenario_dir: str | Path, study_case: str = ""):
        self.folder = ScenarioFolder(scenario_dir)
        self.checker = ConfigChecker(self.folder)
        self.only_case = study_case

    def run(self, dry_run: bool = True) -> ValidationReport:
        """
        Run the static checks and, if they found no errors, the dry-run load.

        Args:
            dry_run (bool, optional): Whether to load each study case into a ``World``. Defaults to True.
        """
        report = self.folder.report
        cases = self.checker.load_cases(self.only_case)
        report.cases = [case.name for case in cases]
        self._check_static(cases)
        if dry_run and report.ok:
            for case in cases:
                self._dry_run(case)
        return report

    def _check_static(self, cases: list[StudyCase]) -> None:
        # csv columns may refer to markets of any study case, as the files are shared between them
        markets = set().union(*(case.market_ids for case in cases))
        for case in cases:
            self.checker.check(case)
            CaseFiles(self.folder, case).check(markets)

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
    scenario_dir: str | Path, study_case: str = "", dry_run: bool = True
) -> ValidationReport:
    """
    Verify that a scenario folder (containing ``config.yaml`` and csv files) can be loaded.

    Args:
        scenario_dir (str | Path): The folder of the scenario, e.g. ``examples/inputs/example_01a``.
        study_case (str, optional): Only verify this study case. Defaults to all cases.
        dry_run (bool, optional): Also load each case into a ``World`` after the static checks passed.
            Defaults to True.

    Returns:
        ValidationReport: All errors and warnings found; ``report.ok`` is True if there are no errors.
    """
    return ScenarioVerifier(scenario_dir, study_case).run(dry_run=dry_run)
