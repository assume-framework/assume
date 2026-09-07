# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest
import yaml

from examples.inputs.example_lv_tariff.generate_inputs import generate
from examples.inputs.example_lv_tariff.run_experiments import HERE, run_suite


def test_fleet_growth_preserves_existing_profiles(tmp_path):
    generate(tmp_path / "small", 2)
    generate(tmp_path / "large", 5)
    small = pd.read_csv(tmp_path / "small/forecasts_df.csv")
    large = pd.read_csv(tmp_path / "large/forecasts_df.csv")
    pd.testing.assert_frame_equal(small, large[small.columns])


@pytest.mark.parametrize("v2g", [True, False])
def test_suite_settlement_and_shared_limits_with_multiple_aggregators(tmp_path, v2g):
    config = yaml.safe_load((HERE / "experiments.yaml").read_text())
    config.update(n_evs=3, n_aggregators=2, v2g=v2g)
    result = run_suite(config, tmp_path / "run").set_index("case")
    assert len(result) == 9
    assert result.at["independent", "energy_cost_eur"] == pytest.approx(
        result.at["aggregated", "energy_cost_eur"]
    )
    assert result.at["constrained", "overload_hours"] == 0
    assert result.at["constrained", "energy_cost_delta_vs_aggregated_eur"] >= -1e-6
    assert (
        result.at["ex_post_peak", "energy_cost_eur"]
        == result.at["aggregated", "energy_cost_eur"]
    )
    assert result.at["ex_post_peak", "grid_fee_eur"] > 0
    for case in result.index:
        bills = pd.read_csv(tmp_path / "run" / case / "operators.csv")
        assert bills.total_cost_eur.sum() == pytest.approx(
            result.at[case, "total_cost_eur"]
        )
        if not v2g:
            assert bills.export_mwh.sum() == pytest.approx(0)
    assert result.at["independent", "n_operators"] == 3
    assert result.at["aggregated", "n_operators"] == 2
    with pytest.raises(FileExistsError):
        run_suite(config, tmp_path / "run")
