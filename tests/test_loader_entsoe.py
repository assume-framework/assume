# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from assume.common.exceptions import AssumeException
from assume.common.forecaster import UnitForecaster
from assume.scenario.entsoe_helper.client import EntsoeInterface, _flatten_columns
from assume.scenario.entsoe_helper.fuel_prices import (
    _CO2_FALLBACK_EUR_T,
    _COAL_FALLBACK_EUR_MWH,
    _GAS_FALLBACK_EUR_MWH,
    _PLN_EUR_FALLBACK,
    InstratFuelPrices,
)
from assume.scenario.entsoe_helper.mappings import (
    DEFAULT_CO2_PRICE_EUR_T,
    DEFAULT_EFFICIENCIES,
    DEFAULT_STORAGE_ADDITIONAL_COST,
    DEFAULT_STORAGE_EFFICIENCY_CHARGE,
    DEFAULT_STORAGE_HOURS,
    PSR_TO_ASSUME,
    block_price_factors,
    default_storage_params,
    interpolate_block_prices,
    split_capacity_blocks,
)
from assume.scenario.loader_entsoe import (
    _add_blocked_units,
    _add_storage_units,
    _add_variable_unit,
    _resolve_co2_prices,
    _total_generation,
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
        efficiencies=DEFAULT_EFFICIENCIES,
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
        efficiencies=DEFAULT_EFFICIENCIES,
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
        assert unit_params["efficiency"] == DEFAULT_EFFICIENCIES["gas"]
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
        efficiencies=DEFAULT_EFFICIENCIES,
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
        storage=default_storage_params(),
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
    assert unit_params["capacity"] == 250.0 * DEFAULT_STORAGE_HOURS
    assert unit_params["initial_soc"] == 0.5
    assert unit_params["additional_cost_charge"] == DEFAULT_STORAGE_ADDITIONAL_COST
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


def test_total_generation_sums_all_technologies(mock_entsoe_data, hourly_index):
    _, generation, capacity = mock_entsoe_data
    technologies = EntsoeInterface.aggregate_by_technology(capacity, generation)
    total = _total_generation(technologies, hourly_index)
    expected = sum(
        t["generation_mw"].reindex(hourly_index).fillna(0)
        for t in technologies.values()
    )
    assert total.equals(expected.clip(lower=0))
    assert (total >= 0).all()


def test_load_entsoe_rejects_unknown_demand_proxy():
    with pytest.raises(AssumeException, match="demand_proxy"):
        load_entsoe(
            MagicMock(),
            "s",
            "c",
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            ["DE"],
            [],
            {},
            api_key="k",
            demand_proxy="bogus",
        )


def test_aggregate_maps_pondage_and_poundage_hydropower():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    capacity = pd.Series(
        {
            "Hydro Run-of-river and pondage": 100.0,
            "Hydro Run-of-river and poundage": 50.0,
        }
    )
    generation = pd.DataFrame(
        {
            "Hydro Run-of-river and pondage": 10.0,
            "Hydro Run-of-river and poundage": 5.0,
        },
        index=index,
    )
    aggregated = EntsoeInterface.aggregate_by_technology(capacity, generation)
    assert aggregated["hydro"]["capacity_mw"] == 150.0
    assert aggregated["hydro"]["generation_mw"].iloc[0] == 15.0


def test_flatten_columns_collapses_duplicates():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    data = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=index,
        columns=pd.MultiIndex.from_tuples([("Wind", "A"), ("Wind", "B")]),
    )
    flat = _flatten_columns(data)
    assert list(flat.columns) == ["Wind"]
    assert flat["Wind"].tolist() == [3.0, 7.0]


def test_get_country_demand_handles_series_and_single_column(tmp_path, hourly_index):
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2, 23, 0)
    load = pd.Series(range(48), index=hourly_index, dtype=float, name="Actual Load")

    responses = (
        load,
        pd.DataFrame({"Load": load.values}, index=hourly_index),
        pd.DataFrame({"A": load.values, "B": load.values}, index=hourly_index),
    )
    for i, response in enumerate(responses):
        iface = EntsoeInterface.__new__(EntsoeInterface)
        iface.client = MagicMock()
        iface.cache_dir = tmp_path / f"cache_{i}"
        iface.client.query_load.return_value = response
        if i < 2:
            demand = iface.get_country_demand(start, end, "DE")
            assert isinstance(demand, pd.Series)
            assert len(demand) == 48
            assert demand.iloc[0] == 0.0
            assert demand.iloc[47] == 47.0
        else:
            with pytest.raises(AssumeException, match="Unexpected ENTSO-E load format"):
                iface.get_country_demand(start, end, "DE")


def test_generation_cache_single_column_stays_dataframe(tmp_path, hourly_index):
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2, 23, 0)
    iface = EntsoeInterface.__new__(EntsoeInterface)
    iface.cache_dir = tmp_path
    cache_path = iface._cache_path("DE", start, end, "generation")
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame({"Solar": range(48)}, index=hourly_index, dtype=float).to_csv(
        cache_path
    )

    generation = iface.get_country_generation(start, end, "DE", use_cache=True)
    assert isinstance(generation, pd.DataFrame)
    assert list(generation.columns) == ["Solar"]
    assert len(generation) == 48


def test_get_installed_capacity_selects_january_row(tmp_path):
    iface = EntsoeInterface.__new__(EntsoeInterface)
    iface.client = MagicMock()
    iface.cache_dir = tmp_path
    capacity = pd.DataFrame(
        {"Fossil Gas": [1000.0, 2000.0]},
        index=pd.to_datetime(["2023-12-01", "2024-01-01"]).tz_localize("UTC"),
    )
    iface.client.query_installed_generation_capacity.return_value = capacity
    row = iface.get_installed_capacity(datetime(2024, 1, 1), datetime(2024, 1, 2), "DE")
    assert isinstance(row, pd.Series)
    assert row["Fossil Gas"] == 2000.0


def test_get_installed_capacity_without_datetime_index(tmp_path):
    iface = EntsoeInterface.__new__(EntsoeInterface)
    iface.client = MagicMock()
    iface.cache_dir = tmp_path
    capacity = pd.DataFrame({"Fossil Gas": [2000.0]}, index=["2024-01-01"])
    iface.client.query_installed_generation_capacity.return_value = capacity
    row = iface.get_installed_capacity(datetime(2024, 1, 1), datetime(2024, 1, 2), "DE")
    assert isinstance(row, pd.Series)
    assert row["Fossil Gas"] == 2000.0


def test_coal_price_lookback_covers_short_window(tmp_path, hourly_index):
    client = InstratFuelPrices(cache_dir=tmp_path)
    calls: list[tuple[str, datetime, datetime]] = []
    monthly = pd.DataFrame(
        {"pscmi1_pln_per_gj": [40.0]}, index=pd.to_datetime(["2021-03-01"])
    )

    def fake_download(url, start, end):
        calls.append((url, start, end))
        if "coal" in url:
            return monthly
        return pd.DataFrame({"price": [100.0]}, index=pd.to_datetime(["2021-03-01"]))

    with (
        patch.object(client, "_download", side_effect=fake_download),
        patch.object(
            InstratFuelPrices,
            "_pln_to_eur",
            return_value=pd.Series(0.25, index=monthly.index),
        ),
    ):
        prices = client.get_fuel_prices(
            datetime(2021, 3, 2),
            datetime(2021, 3, 3),
            hourly_index,
            use_cache=False,
        )

    coal_url, coal_start, _ = calls[0]
    assert "coal" in coal_url
    assert coal_start <= datetime(2021, 3, 2) - pd.Timedelta(days=366)
    for key in ("hard coal", "lignite", "gas", "co2"):
        assert prices[key].isna().sum() == 0
    assert prices["hard coal"].iloc[0] == pytest.approx(36.0)
    assert prices["gas"].iloc[0] == pytest.approx(25.0)


def test_fuel_prices_fallback_on_empty_download(tmp_path, hourly_index):
    client = InstratFuelPrices(cache_dir=tmp_path)
    with patch.object(InstratFuelPrices, "_download", return_value=pd.DataFrame()):
        prices = client.get_fuel_prices(
            datetime(2021, 3, 2),
            datetime(2021, 3, 3),
            hourly_index,
            use_cache=False,
        )
    for key in ("hard coal", "lignite", "gas", "co2"):
        assert prices[key].isna().sum() == 0
    assert prices["hard coal"].iloc[0] == _COAL_FALLBACK_EUR_MWH
    assert prices["gas"].iloc[0] == _GAS_FALLBACK_EUR_MWH
    assert prices["co2"].iloc[0] == _CO2_FALLBACK_EUR_T


def test_download_returns_empty_on_request_failure():
    import requests

    with patch(
        "assume.scenario.entsoe_helper.fuel_prices.requests.get",
        side_effect=requests.ConnectionError("dns failure"),
    ):
        df = InstratFuelPrices._download(
            "https://example.invalid", datetime(2021, 1, 1), datetime(2021, 1, 2)
        )
    assert df.empty


def test_fuel_prices_fallback_is_not_cached(tmp_path, hourly_index):
    client = InstratFuelPrices(cache_dir=tmp_path)
    with patch.object(InstratFuelPrices, "_download", return_value=pd.DataFrame()):
        client.get_fuel_prices(
            datetime(2021, 3, 2), datetime(2021, 3, 3), hourly_index, use_cache=True
        )
    assert not any(tmp_path.rglob("*.csv"))


def test_pln_to_eur_handles_yfinance_shapes(monkeypatch):
    index = pd.date_range("2024-01-01", "2024-01-03", freq="D")

    def fake_yf(result):
        module = MagicMock(name="yfinance")
        module.download.return_value = result
        monkeypatch.setitem(sys.modules, "yfinance", module)

    series_close = pd.DataFrame(
        {"Close": [4.1, 4.2]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )
    frame_close = pd.DataFrame(
        [[4.1, 90.0], [4.2, 91.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=pd.MultiIndex.from_tuples(
            [("Close", "PLNEUR=X"), ("Volume", "PLNEUR=X")]
        ),
    )

    fake_yf(series_close)
    fx = InstratFuelPrices._pln_to_eur(index)
    assert fx.tolist() == [4.1, 4.2, 4.2]

    fake_yf(frame_close)
    fx = InstratFuelPrices._pln_to_eur(index)
    assert fx.tolist() == [4.1, 4.2, 4.2]

    for empty_result in (None, pd.DataFrame()):
        fake_yf(empty_result)
        fx = InstratFuelPrices._pln_to_eur(index)
        assert (fx == _PLN_EUR_FALLBACK).all()


def test_pln_to_eur_queries_inclusive_end_for_single_date(monkeypatch):
    index = pd.DatetimeIndex(["2024-01-15"])
    module = MagicMock(name="yfinance")
    module.download.return_value = pd.DataFrame(
        {"Close": [4.3]}, index=pd.to_datetime(["2024-01-15"])
    )
    monkeypatch.setitem(sys.modules, "yfinance", module)

    fx = InstratFuelPrices._pln_to_eur(index)

    kwargs = module.download.call_args.kwargs
    assert kwargs["start"] == "2024-01-15"
    assert kwargs["end"] == "2024-01-16"
    assert fx.tolist() == [4.3]


def test_demand_unit_max_power_matches_negative_forecaster(hourly_index):
    demand = pd.Series(-50_000.0, index=hourly_index, name="Actual Load")
    generation = pd.DataFrame({"Solar": 5_000.0}, index=hourly_index)
    capacity = pd.Series({"Solar": 80_000.0})

    mock_interface = MagicMock()
    mock_interface.get_country_demand.return_value = demand
    mock_interface.get_country_generation.return_value = generation
    mock_interface.get_installed_capacity.return_value = capacity
    mock_interface.aggregate_by_technology.return_value = (
        EntsoeInterface.aggregate_by_technology(capacity, generation)
    )

    world = MagicMock()
    bidding_strategies = {
        "demand": {"EOM": "demand_energy_naive"},
        "solar": {"EOM": "powerplant_energy_naive"},
    }

    with (
        patch(
            "assume.scenario.loader_entsoe.EntsoeInterface", return_value=mock_interface
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
            datetime(2024, 1, 1),
            datetime(2024, 1, 2, 23, 0),
            ["DE"],
            [],
            bidding_strategies,
            api_key="test-key",
        )

    demand_calls = [
        call for call in world.add_unit.call_args_list if call[0][0] == "demand_DE"
    ]
    assert len(demand_calls) == 1
    unit_params = demand_calls[0][0][3]
    forecaster = demand_calls[0][0][4]
    assert unit_params["max_power"] == -50_000.0
    assert (forecaster.demand == -50_000.0).all()


def test_storage_units_use_custom_defaults(hourly_index):
    world = MagicMock()
    mapping = PSR_TO_ASSUME["Hydro Pumped Storage"]
    storage = default_storage_params()
    storage.update(
        {
            "hours": 12.0,
            "initial_soc": 0.3,
            "max_soc": 0.9,
            "min_soc": 0.1,
            "efficiency_discharge": 0.8,
        }
    )

    _add_storage_units(
        world,
        "DE",
        "hydro_storage",
        mapping,
        total_capacity=250.0,
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"storage": {"EOM": "storage_energy_heuristic_flexable"}},
        block_sizes_mw={"hydro_storage": 250.0},
        storage=storage,
    )

    unit_params = world.add_unit.call_args[0][3]
    assert unit_params["capacity"] == 250.0 * 12.0
    assert unit_params["initial_soc"] == 0.3
    assert unit_params["max_soc"] == 0.9
    assert unit_params["min_soc"] == 0.1
    assert unit_params["efficiency_discharge"] == 0.8
    assert unit_params["efficiency_charge"] == DEFAULT_STORAGE_EFFICIENCY_CHARGE


def test_blocked_non_thermal_units_have_no_efficiency(hourly_index):
    world = MagicMock()
    mapping = PSR_TO_ASSUME["Hydro Water Reservoir"]

    _add_blocked_units(
        world,
        "DE",
        "hydro",
        mapping,
        total_capacity=300.0,
        gen_series=pd.Series(300.0, index=hourly_index),
        index=hourly_index,
        location=(51.16, 10.45),
        bidding_strategies={"hydro": {"EOM": "powerplant_energy_naive"}},
        block_sizes_mw={"hydro": 300.0},
        fuel_price_ranges={"hydro": (0.4, 0.1)},
        api_fuel_prices={},
        co2_prices=pd.Series(70.0, index=hourly_index, name="co2"),
        efficiencies=DEFAULT_EFFICIENCIES,
    )

    assert "efficiency" not in world.add_unit.call_args[0][3]


def test_to_naive_index_converts_to_utc():
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="Europe/Berlin")
    series = pd.Series([1.0, 2.0], index=index)
    naive = EntsoeInterface._to_naive_index(series)
    assert naive.index.tz is None
    # 00:00 CET is 23:00 UTC of the previous day
    assert naive.index[0] == pd.Timestamp("2023-12-31 23:00")


def test_demand_is_utc_across_spring_dst(tmp_path):
    iface = EntsoeInterface.__new__(EntsoeInterface)
    iface.client = MagicMock()
    iface.cache_dir = tmp_path
    local = pd.date_range(
        "2024-03-30 12:00", "2024-04-01 12:00", freq="h", tz="Europe/Berlin"
    )
    iface.client.query_load.return_value = pd.Series(1.0, index=local)
    start, end = datetime(2024, 3, 30, 11), datetime(2024, 4, 1, 10)
    demand = iface.get_country_demand(start, end, "DE", use_cache=False)
    assert demand.index.is_unique
    assert (
        demand.index.freq is not None
        or demand.index.to_series().diff().dropna().eq(pd.Timedelta("1h")).all()
    )
    assert demand.index[0] == pd.Timestamp("2024-03-30 11:00")


def test_single_row_caches_stay_series(tmp_path):
    iface = EntsoeInterface.__new__(EntsoeInterface)
    iface.cache_dir = tmp_path
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 1)
    demand_path = iface._cache_path("DE", start, end, "demand")
    demand_path.parent.mkdir(parents=True)
    pd.Series([5.0], index=pd.to_datetime(["2024-01-01"])).rename_axis("t").to_csv(
        demand_path
    )
    demand = iface.get_country_demand(start, end, "DE", use_cache=True)
    assert isinstance(demand, pd.Series)
    assert len(demand) == 1

    cap_path = iface._cache_path("DE", start, end, "capacity")
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series({"Solar": 10.0}).to_csv(cap_path)
    capacity = iface.get_installed_capacity(start, end, "DE", use_cache=True)
    assert isinstance(capacity, pd.Series)
    assert capacity["Solar"] == 10.0


def test_instrat_download_sorts_descending_data():
    resp = MagicMock()
    resp.text = (
        '[{"date":"2024-01-02T00:00:00Z","v":2},{"date":"2024-01-01T00:00:00Z","v":1}]'
    )
    with patch("assume.scenario.entsoe_helper.fuel_prices.requests.get") as get:
        get.return_value = resp
        df = InstratFuelPrices._download(
            "http://x", datetime(2024, 1, 1), datetime(2024, 1, 2)
        )
    assert df.index.is_monotonic_increasing
