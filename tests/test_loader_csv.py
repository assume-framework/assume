# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

from assume.scenario.loader_csv import load_config_and_create_forecaster


def test_csv_loader_validation():
    with pytest.raises(
        ValueError,
        match="min_power and max_power must both be either negative or positive",
    ):
        load_config_and_create_forecaster(
            inputs_path="tests/fixtures", scenario="invalid_units", study_case="base"
        )
    with pytest.raises(
        ValueError, match="No power plant or no demand units were provided!"
    ):
        load_config_and_create_forecaster(
            inputs_path="tests/fixtures", scenario="missing_units", study_case="base"
        )
