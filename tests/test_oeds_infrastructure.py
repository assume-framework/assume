# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from assume.scenario.oeds.infrastructure import InfrastructureInterface
from assume.scenario.oeds.static import fuel_translation, mastr_fuel_type


def test_mastr_fuel_type_mappings():
    assert mastr_fuel_type["lignite"] == "Braunkohle"
    assert mastr_fuel_type["hard coal"] == "Steinkohle"
    assert mastr_fuel_type["gas"] == "Erdgas"
    assert mastr_fuel_type["oil"] == "Mineralölprodukte"
    assert mastr_fuel_type["nuclear"] == "Kernenergie"

    for ger, eng in fuel_translation.items():
        if eng in mastr_fuel_type:
            assert mastr_fuel_type[eng] in fuel_translation


@pytest.fixture
def mock_infrastructure():
    with patch.object(InfrastructureInterface, "__init__", return_value=None):
        interface = InfrastructureInterface("test", "dummy_uri")
        interface.plz_nuts = pd.DataFrame(
            {"latitude": [50.0], "longitude": [6.0], "nuts3": ["DEA2D"]},
            index=[52353],
        )
        interface.databases = {"mastr": MagicMock(), "nuts": MagicMock()}
        interface.energietraeger_translated = mastr_fuel_type
        interface.mastr_generation_codes = {}
        return interface


def test_set_default_params(mock_infrastructure):
    df = pd.DataFrame(
        {
            "maxPower": [10000.0],
            "turbineTyp": ["Dampfturbine"],
            "startDate": [pd.to_datetime("2010-01-01")],
            "endDate": [pd.to_datetime("2040-01-01")],
            "generatorID": [1],
        }
    )
    result = mock_infrastructure.set_default_params(df)

    assert result["minPower"].iloc[0] == 5000.0
    assert result["ramp_up"].iloc[0] == 1000.0
    assert result["turbineTyp"].iloc[0] == "Dampfturbine"
    assert result["type"].iloc[0] == 2000


def test_get_power_plant_in_area_queries(mock_infrastructure):
    from datetime import datetime

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=pd.DataFrame()) as mock_read_sql:
        # Gas with created_before and stopped_after date filters
        mock_infrastructure.get_power_plant_in_area(
            area=52353,
            fuel_type="gas",
            created_before=datetime(2020, 1, 1),
            stopped_after=datetime(2023, 1, 1),
        )
        query_gas = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Erdgas'" in query_gas
        assert 'FROM "combustion_extended" ev' in query_gas
        assert "AND ev.\"Inbetriebnahmedatum\" < '2020-01-01T00:00:00'" in query_gas
        assert (
            'AND (ev."DatumEndgueltigeStilllegung" IS NULL OR ev."DatumEndgueltigeStilllegung"  > \'2023-01-01T00:00:00\')'
            in query_gas
        )

        # Lignite with string postal code
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area="52353", fuel_type="lignite")
        query_lignite = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Braunkohle'" in query_lignite

        # Hard Coal
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area=52353, fuel_type="hard coal")
        query_hard_coal = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Steinkohle'" in query_hard_coal

        # Oil
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area=52353, fuel_type="oil")
        query_oil = mock_read_sql.call_args[0][0]
        assert "ev.\"Energietraeger\" = 'Mineralölprodukte'" in query_oil

        # Nuclear
        mock_read_sql.reset_mock()
        mock_infrastructure.get_power_plant_in_area(area=52353, fuel_type="nuclear")
        query_nuclear = mock_read_sql.call_args[0][0]
        assert 'FROM "nuclear_extended" ev' in query_nuclear


def test_get_power_plant_in_area_cchp_parameters(mock_infrastructure):
    # Test that CCHP units receive 'gas_combined' technical parameters instead of standard 'gas'
    cchp_df = pd.DataFrame(
        {
            "unitID": ["SEE1001", "SEE1002"],
            "fuel": ["gas", "gas"],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "startDate": [pd.to_datetime("2010-01-01"), pd.to_datetime("2010-01-01")],
            "endDate": [pd.to_datetime("2040-01-01"), pd.to_datetime("2040-01-01")],
            "maxPower": [20000.0, 30000.0],
            "turbineTyp": [
                "Closed Cycle Heat Power",
                "Kondensationsmaschine ohne Entnahme",
            ],
            "generatorID": [10, 0],
            "kwkPowerTherm": [5000.0, 0.0],
            "kwkPowerElec": [4000.0, 0.0],
            "combination": [1, 0],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=cchp_df):
        result = mock_infrastructure.get_power_plant_in_area(
            area=52353, fuel_type="gas"
        )

        # The aggregated CCHP row should have fuel == 'gas_combined'
        cchp_row = result[result["fuel"] == "gas_combined"].iloc[0]
        assert cchp_row["fuel"] == "gas_combined"
        # Ramp up for gas_combined (2000) is 4% per min -> 4 * 60 / 100 = 2.4 * maxPower
        # For standard gas (2000), ramp up is 12% per min -> 12 * 60 / 100 = 7.2 * maxPower
        assert cchp_row["ramp_up"] == pytest.approx(
            cchp_row["maxPower"] * 4.0 * 60 / 100
        )


def test_asset_queries_with_string_plz(mock_infrastructure):
    # Verify string postal codes work across all asset query methods without 'invalid plz code' exception
    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=pd.DataFrame()):
        assert mock_infrastructure.get_solar_systems_in_area(area="52353").empty
        assert mock_infrastructure.get_wind_turbines_in_area(area="52353").empty
        assert mock_infrastructure.get_biomass_systems_in_area(area="52353").empty
        assert mock_infrastructure.get_run_river_systems_in_area(area="52353").empty
        assert mock_infrastructure.get_water_storage_systems(area="52353") == []
        assert mock_infrastructure.get_solar_storage_systems_in_area(area="52353").empty


def test_get_lat_lon_area(mock_infrastructure):
    # NUTS3 string area
    lat, lon = mock_infrastructure.get_lat_lon_area("DEA2D")
    assert (lat, lon) == (50.0, 6.0)

    # Integer postal code
    lat, lon = mock_infrastructure.get_lat_lon_area(52353)
    assert (lat, lon) == (50.0, 6.0)

    # String postal code
    lat, lon = mock_infrastructure.get_lat_lon_area("52353")
    assert (lat, lon) == (50.0, 6.0)


def test_get_solar_storage_systems_in_area_stopped_after(mock_infrastructure):
    from datetime import datetime

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=pd.DataFrame()) as mock_read_sql:
        cutoff = datetime(2023, 1, 1)
        mock_infrastructure.get_solar_storage_systems_in_area(
            area=52353, stopped_after=cutoff
        )

        mock_read_sql.assert_called_once()
        query = mock_read_sql.call_args[0][0]
        assert (
            'AND (so."DatumEndgueltigeStilllegung" IS NULL OR so."DatumEndgueltigeStilllegung" > \'2023-01-01T00:00:00\')'
            in query
        )


def test_get_solar_systems_in_area_power_cap_threshold(mock_infrastructure):
    # A system with 120 kWp installed capacity and 70% limit (84 kWp after limit)
    # should be classified as > 100 kWp for EEG threshold (power_cap evaluated on installed capacity)
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE12345"],
            "maxPower": [120.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Süd"],
            "limited": ["Ja, auf 70%"],
            "ownConsumption": [0],
            "tiltCode": ["21 - 40 Grad"],
            "startDate": [pd.to_datetime("2018-01-01")],
            "eeg": [None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

        # maxPower should be limited to 120 * 0.7 = 84.0
        assert result["maxPower"].iloc[0] == pytest.approx(84.0)
        # demandP should be in W: 84.0 * 1000 = 84000.0
        assert result["demandP"].iloc[0] == pytest.approx(84000.0)
        # eeg should be set to 0 (direct marketing) because installed capacity was 120 > 100
        assert result["eeg"].iloc[0] == 0
