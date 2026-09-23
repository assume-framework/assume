# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from assume.common.exceptions import AssumeException
from assume.common.forecaster import UnitForecaster
from assume.scenario.entsoe_helper.client import EntsoeInterface
from assume.scenario.entsoe_helper.fuel_prices import InstratFuelPrices
from assume.scenario.entsoe_helper.mappings import (
    DEFAULT_CO2_PRICE_EUR_T,
    PSR_TO_ASSUME,
    block_price_factors,
    interpolate_block_prices,
    split_capacity_blocks,
)
from assume.scenario.loader_entsoe import (
    _add_blocked_units,
    _add_storage_units,
    _add_variable_unit,
    _resolve_co2_prices,
    load_entsoe,
)


@pytest.fixture
def hourly_index():
    return pd.date_range("2024-01-01", "2024-01-02 23:00", freq="h")


@pytest.fixture
def mock_entsoe_data(hourly_index):
    demand = pd.Series(50_000.0, index=hourly_index, name="Actual Load")
    generation = pd.DataFrame(
        {
            "Solar": 5_000.0,
            "Wind Onshore": 8_000.0,
            "Fossil Gas": 15_000.0,
            "Nuclear": 10_000.0,
            "Hydro Pumped Storage": 2_000.0,
            "Other": 500.0,
        },
        index=hourly_index,
    )
    capacity = pd.Series(
        {
            "Solar": 80_000.0,
            "Wind Onshore": 60_000.0,
            "Fossil Gas": 2_000.0,
            "Nuclear": 12_000.0,
            "Hydro Pumped Storage": 4_000.0,
            "Other": 1_000.0,
        }
    )
    return demand, generation, capacity


def test_load_entsoe_requires_api_key():
    world = MagicMock()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AssumeException, match="API key missing"):
            load_entsoe(
                world,
                "entsoe_test",
                "DE_2024",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                ["DE"],
                [],
                {"demand": {}},
                use_instrat_fuel_prices=False,
            )


def test_aggregate_raises_for_unmapped_psr(hourly_index):
    generation = pd.DataFrame({"Unknown Fuel": 100.0}, index=hourly_index)
    capacity = pd.Series({"Unknown Fuel": 100.0})
    with pytest.raises(AssumeException, match="Unmapped ENTSO-E production type"):
        EntsoeInterface.aggregate_by_technology(capacity, generation)


def test_split_capacity_blocks():
    assert split_capacity_blocks(950, 400) == [400, 400, 150]
    assert split_capacity_blocks(400, 400) == [400]
    assert split_capacity_blocks(0, 400) == []


def test_interpolate_block_prices():
    assert interpolate_block_prices(1, 30, 20) == [30]
    assert interpolate_block_prices(3, 30, 20) == [30, 25, 20]


def test_block_price_factors():
    factors = block_price_factors(3, 30, 20)
    assert factors[0] > factors[-1]
    assert pytest.approx(sum(factors) / len(factors), rel=1e-6) == 1.0


def test_instrat_co2_price_parsing(hourly_index):
    payload = StringIO(
        '[{"date":"2024-01-01T00:00:00","price":80.0},'
        '{"date":"2024-01-02T00:00:00","price":82.0}]'
    )
    df = pd.read_json(payload).set_index("date")
    df.index = df.index.tz_localize(None)
    series = df["price"].resample("D").bfill()
    assert series.iloc[0] == 80.0


@patch.object(InstratFuelPrices, "_download")
def test_get_fuel_prices(mock_download, hourly_index):
    mock_download.side_effect = [
        pd.DataFrame({"pscmi1_pln_per_gj": [10.0, 11.0]}, index=hourly_index[:2]),
        pd.DataFrame({"price": [100.0, 110.0]}, index=hourly_index[:2]),
        pd.DataFrame({"price": [80.0, 82.0]}, index=hourly_index[:2]),
    ]

    with patch.object(
        InstratFuelPrices,
        "_pln_to_eur",
        return_value=pd.Series(0.23, index=hourly_index[:2]),
    ):
        prices = InstratFuelPrices().get_fuel_prices(
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            hourly_index,
            use_cache=False,
        )

    assert set(prices) == {"hard coal", "lignite", "gas", "co2"}
    assert len(prices["gas"]) == len(hourly_index)
    assert prices["hard coal"].isna().sum() == 0


def test_get_fuel_prices_handles_empty_coal_cache(hourly_index, tmp_path):
    cache_dir = tmp_path / "instrat"
    period_dir = cache_dir / "20240101_20240102"
    period_dir.mkdir(parents=True)
    (period_dir / "coal.csv").write_text("date,hard coal\n2024-01-01,\n")
    (period_dir / "gas.csv").write_text("date,gas\n2024-01-01,30.0\n2024-01-02,31.0\n")
    (period_dir / "co2.csv").write_text("date,co2\n2024-01-01,80.0\n2024-01-02,81.0\n")

    client = InstratFuelPrices(cache_dir=cache_dir)
    with patch.object(client, "_download") as mock_download:
        prices = client.get_fuel_prices(
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            hourly_index,
            use_cache=True,
        )

    mock_download.assert_not_called()
    assert prices["hard coal"].isna().sum() == 0
    assert prices["lignite"].isna().sum() == 0


def test_aggregate_by_technology(mock_entsoe_data, hourly_index):
    _, generation, capacity = mock_entsoe_data
    aggregated = EntsoeInterface.aggregate_by_technology(capacity, generation)

    assert "solar" in aggregated
    assert "wind_onshore" in aggregated
    assert "gas" in aggregated
    assert "nuclear" in aggregated
    assert "hydro_storage" in aggregated
    assert "other" in aggregated
    assert aggregated["gas"]["capacity_mw"] == 2_000.0
    assert len(aggregated["solar"]["generation_mw"]) == len(hourly_index)


def test_aggregate_uses_generation_peak_when_capacity_missing(hourly_index):
    generation = pd.DataFrame({"Other": 100.0}, index=hourly_index)
    capacity = pd.Series({"Other": 0.0})
    aggregated = EntsoeInterface.aggregate_by_technology(capacity, generation)
    assert aggregated["other"]["capacity_mw"] == 100.0


def test_variable_unit_uses_installed_capacity_and_availability(hourly_index):
    world = MagicMock()
    gen_series = pd.Series(5_000.0, index=hourly_index)
    mapping = PSR_TO_ASSUME["Solar"]

    _add_variable_unit(
        world,
        "DE",
        "solar",
        mapping,
        total_capacity=80_000.0,
        gen_series=gen_series,
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"solar": {"EOM": "powerplant_energy_naive"}},
    )

    unit_params = world.add_unit.call_args[0][3]
    forecaster = world.add_unit.call_args[0][4]
    assert unit_params["max_power"] == 80_000.0
    assert forecaster.availability.iloc[0] == pytest.approx(5_000.0 / 80_000.0)


def test_resolve_co2_prices_uses_api_or_fallback(hourly_index):
    fallback = _resolve_co2_prices(hourly_index, {})
    assert (fallback == DEFAULT_CO2_PRICE_EUR_T).all()

    api_co2 = pd.Series(82.5, index=hourly_index, name="co2")
    resolved = _resolve_co2_prices(hourly_index, {"co2": api_co2})
    assert resolved.iloc[0] == 82.5


def test_blocked_units_use_installed_capacity_not_generation_peak(hourly_index):
    world = MagicMock()
    gen_series = pd.Series(3_000.0, index=hourly_index)
    mapping = PSR_TO_ASSUME["Hydro Water Reservoir"]

    _add_blocked_units(
        world,
        "DE",
        "hydro",
        mapping,
        total_capacity=10_000.0,
        gen_series=gen_series,
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"hydro": {"EOM": "powerplant_energy_naive"}},
        block_sizes_mw={"hydro": 300.0},
        fuel_price_ranges={"hydro": (0.4, 0.1)},
        api_fuel_prices={},
        co2_prices=pd.Series(70.0, index=hourly_index, name="co2"),
    )

    max_powers = [call[0][3]["max_power"] for call in world.add_unit.call_args_list]
    assert sum(max_powers) == pytest.approx(10_000.0)
    assert max(max_powers) == 300.0
    assert all(
        call[0][4].availability.iloc[0] == 1.0 for call in world.add_unit.call_args_list
    )


def test_thermal_units_bid_installed_capacity(hourly_index):
    world = MagicMock()
    gen_series = pd.Series(15_000.0, index=hourly_index)
    mapping = PSR_TO_ASSUME["Fossil Gas"]

    co2_prices = pd.Series(75.0, index=hourly_index, name="co2")

    _add_blocked_units(
        world,
        "DE",
        "gas",
        mapping,
        total_capacity=2_000.0,
        gen_series=gen_series,
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"gas": {"EOM": "powerplant_energy_naive"}},
        block_sizes_mw={"gas": 400.0},
        fuel_price_ranges={"gas": (32.0, 22.0)},
        api_fuel_prices={},
        co2_prices=co2_prices,
    )

    gas_units = [
        call
        for call in world.add_unit.call_args_list
        if call[0][0].startswith("generation_DE_gas_")
    ]
    assert len(gas_units) == 5
    for call in gas_units:
        unit_params = call[0][3]
        forecaster = call[0][4]
        assert unit_params["max_power"] in {400.0, 200.0}
        assert unit_params["emission_factor"] == 0.201
        assert unit_params["fuel_type"] == "gas"
        assert forecaster.availability.iloc[0] == 1.0
        assert "co2" in forecaster.fuel_prices
        assert forecaster.fuel_prices["co2"].iloc[0] == 75.0


def test_oil_units_use_shared_co2_price(hourly_index):
    world = MagicMock()
    gen_series = pd.Series(500.0, index=hourly_index)
    mapping = PSR_TO_ASSUME["Fossil Oil"]
    co2_prices = pd.Series(80.0, index=hourly_index, name="co2")

    _add_blocked_units(
        world,
        "DE",
        "oil",
        mapping,
        total_capacity=400.0,
        gen_series=gen_series,
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"oil": {"EOM": "powerplant_energy_naive"}},
        block_sizes_mw={"oil": 200.0},
        fuel_price_ranges={"oil": (25.0, 18.0)},
        api_fuel_prices={},
        co2_prices=co2_prices,
    )

    forecaster = world.add_unit.call_args[0][4]
    assert forecaster.fuel_prices["co2"].iloc[0] == 80.0


def test_storage_units_use_installed_capacity(hourly_index):
    world = MagicMock()
    mapping = PSR_TO_ASSUME["Hydro Pumped Storage"]

    _add_storage_units(
        world,
        "DE",
        "hydro_storage",
        mapping,
        total_capacity=1_000.0,
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"storage": {"EOM": "storage_energy_heuristic_flexable"}},
        block_sizes_mw={"hydro_storage": 250.0},
    )

    storage_units = [
        call
        for call in world.add_unit.call_args_list
        if call[0][0].startswith("storage_DE_hydro_storage_")
    ]
    assert len(storage_units) == 4
    unit_params = storage_units[0][0][3]
    forecaster = storage_units[0][0][4]
    assert unit_params["max_power_discharge"] == 250.0
    assert unit_params["max_power_charge"] == -250.0
    assert unit_params["capacity"] == 250.0 * 8.0
    assert unit_params["initial_soc"] == 0.5
    assert unit_params["additional_cost_charge"] == 0.28
    assert isinstance(forecaster, UnitForecaster)


def test_load_entsoe_builds_world(mock_entsoe_data, hourly_index):
    demand, generation, capacity = mock_entsoe_data
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2, 23, 0)

    mock_interface = MagicMock()
    mock_interface.get_country_demand.return_value = demand
    mock_interface.get_country_generation.return_value = generation
    mock_interface.get_installed_capacity.return_value = capacity
    mock_interface.aggregate_by_technology.return_value = (
        EntsoeInterface.aggregate_by_technology(capacity, generation)
    )

    world = MagicMock()
    marketdesign = []

    bidding_strategies = {
        "demand": {"EOM": "demand_energy_naive"},
        "solar": {"EOM": "powerplant_energy_naive"},
        "wind": {"EOM": "powerplant_energy_naive"},
        "gas": {"EOM": "powerplant_energy_naive"},
        "nuclear": {"EOM": "powerplant_energy_naive"},
        "biomass": {"EOM": "powerplant_energy_naive"},
        "storage": {"EOM": "storage_energy_heuristic_flexable"},
    }

    with (
        patch(
            "assume.scenario.loader_entsoe.EntsoeInterface",
            return_value=mock_interface,
        ),
        patch(
            "assume.scenario.loader_entsoe.InstratFuelPrices.get_fuel_prices",
            return_value={},
        ),
    ):
        load_entsoe(
            world,
            "entsoe_test",
            "DE_2024",
            start,
            end,
            ["DE"],
            marketdesign,
            bidding_strategies,
            api_key="test-key",
        )

    world.setup.assert_called_once()
    world.add_market_operator.assert_called_once()
    world.add_unit_operator.assert_any_call("demand_DE")
    world.add_unit_operator.assert_any_call("generation_DE")
    assert world.add_unit.call_count > 5
    world.init_forecasts.assert_called_once()
