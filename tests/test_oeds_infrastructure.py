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


def test_operational_status_filter_in_queries(mock_infrastructure):
    # the removed "EinheitBetriebsstatus >= 35" filter must be replaced by a
    # string status filter so that "In Planung" units are not counted as
    # built capacity by callers that are not time-bounded
    import re

    from assume.scenario.oeds.infrastructure import MASTR_OPERATIONAL_STATUS

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    OPERATIONAL = f"IN {MASTR_OPERATIONAL_STATUS}"
    expected_clauses = {
        "solar": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "wind": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "biomass": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "hydro": f'"EinheitBetriebsstatus" {OPERATIONAL}',
        "solar_storage": f'so."EinheitBetriebsstatus" {OPERATIONAL}',
        "storage": "spe.\"EinheitBetriebsstatus\" = 'In Betrieb'",
        "power_plant": f'ev."EinheitBetriebsstatus" {OPERATIONAL}',
    }
    calls = [
        ("solar", lambda: mock_infrastructure.get_solar_systems_in_area(area=52353)),
        ("wind", lambda: mock_infrastructure.get_wind_turbines_in_area(area=52353)),
        (
            "biomass",
            lambda: mock_infrastructure.get_biomass_systems_in_area(area=52353),
        ),
        (
            "hydro",
            lambda: mock_infrastructure.get_run_river_systems_in_area(area=52353),
        ),
        (
            "solar_storage",
            lambda: mock_infrastructure.get_solar_storage_systems_in_area(area=52353),
        ),
        ("storage", lambda: mock_infrastructure.get_water_storage_systems(area=52353)),
        (
            "power_plant",
            lambda: mock_infrastructure.get_power_plant_in_area(
                area=52353, fuel_type="gas"
            ),
        ),
    ]
    for label, call in calls:
        with (
            patch("pandas.read_sql", return_value=pd.DataFrame()) as mr1,
            patch("pandas.read_sql_query", return_value=pd.DataFrame()) as mr2,
        ):
            call()
        query = (mr1.call_args or mr2.call_args)[0][0]
        assert expected_clauses[label] in query, (
            label,
            re.sub(r"\s+", " ", query)[-200:],
        )


def test_area_none_queries_without_postal_code_filter(mock_infrastructure):
    # area=None selects all units, so no postal code filter may be applied;
    # a postal code area keeps the filter
    postal_code, lat, lon = mock_infrastructure.resolve_area(None)
    assert postal_code is None
    assert (lat, lon) == (50.0, 6.0)

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    getters = [
        mock_infrastructure.get_solar_systems_in_area,
        mock_infrastructure.get_wind_turbines_in_area,
        mock_infrastructure.get_biomass_systems_in_area,
        mock_infrastructure.get_run_river_systems_in_area,
        mock_infrastructure.get_solar_storage_systems_in_area,
        mock_infrastructure.get_water_storage_systems,
        mock_infrastructure.get_power_plant_in_area,
    ]
    for getter in getters:
        for area, filtered in [(None, False), (52353, True)]:
            with (
                patch("pandas.read_sql", return_value=pd.DataFrame()) as mr1,
                patch("pandas.read_sql_query", return_value=pd.DataFrame()) as mr2,
            ):
                getter(area=area)
            query = (mr1.call_args or mr2.call_args)[0][0]
            assert ("Postleitzahl\" in ('52353')" in query) == filtered, (
                getter.__name__,
                area,
            )


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


def test_map_mastr_orientation_codes(caplog):
    from assume.scenario.oeds.infrastructure import map_mastr_codes
    from assume.scenario.oeds.static import mastr_solar_azimuth, mastr_solar_tilt

    tilt_codes = pd.Series(
        [
            "Nachgeführt",
            "5 - 20 Grad",
            "unter 5 Grad (horizontal)",
            # MaStR carries a trailing space on this code
            "90 Grad (vertikal) ",
            None,
            "unbekannt",
        ]
    )
    with caplog.at_level("WARNING"):
        tilt = map_mastr_codes(tilt_codes, mastr_solar_tilt, "30")
    # tracked units keep the default tilt instead of the 180° azimuth value
    assert tilt.tolist() == ["30", "13", "3", "90", "30", "30"]
    # missing codes are silent, unknown codes are reported
    assert "unbekannt" in caplog.text

    azimuth_codes = pd.Series(["nachgeführt", "Süd-West", None])
    azimuth = map_mastr_codes(azimuth_codes, mastr_solar_azimuth, "180")
    assert azimuth.tolist() == ["180", "225", "180"]


def test_get_solar_systems_in_area_rejects_unknown_solar_type(mock_infrastructure):
    with pytest.raises(ValueError, match="water"):
        mock_infrastructure.get_solar_systems_in_area(area=52353, solar_type="water")


def test_get_solar_systems_in_area_own_consumption_fallback(mock_infrastructure):
    # Rooftop PV commissioned after 2013 with NULL ownConsumption defaults to 1 (own consumption).
    # Rooftop PV commissioned <= 2013 with NULL ownConsumption defaults to 0 (grid feed-in).
    # Free-area / other PV with NULL ownConsumption defaults to 0 regardless of date (demand is unknown).
    # Explicit ownConsumption settings must still be respected across all types.
    raw_df_rooftop = pd.DataFrame(
        {
            "unitID": ["SEE1", "SEE2", "SEE3"],
            "maxPower": [10.0, 10.0, 10.0],
            "lon": [6.0, 6.0, 6.0],
            "lat": [50.0, 50.0, 50.0],
            "plzCode": ["52353", "52353", "52353"],
            "azimuthCode": ["Süd", "Süd", "Süd"],
            "limited": ["Nein", "Nein", "Nein"],
            "ownConsumption": [None, None, "Volleinspeisung"],
            "tiltCode": ["30 Grad", "30 Grad", "30 Grad"],
            "startDate": [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2012-01-01"),
                pd.to_datetime("2020-01-01"),
            ],
            "eeg": [1, 1, 1],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df_rooftop):
        res_rooftop = mock_infrastructure.get_solar_systems_in_area(
            area=52353, solar_type="roof_top"
        )
    assert res_rooftop["ownConsumption"].tolist() == [1, 0, 0]

    raw_df_free = pd.DataFrame(
        {
            "unitID": ["SEE4", "SEE5"],
            "maxPower": [100.0, 100.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "azimuthCode": ["Süd", "Süd"],
            "limited": ["Nein", "Nein"],
            "ownConsumption": [None, "Teileinspeisung (einschließlich Eigenverbrauch)"],
            "tiltCode": ["30 Grad", "30 Grad"],
            "startDate": [pd.to_datetime("2020-01-01"), pd.to_datetime("2020-01-01")],
            "eeg": [1, 1],
        }
    )

    with patch("pandas.read_sql", return_value=raw_df_free):
        res_free = mock_infrastructure.get_solar_systems_in_area(
            area=52353, solar_type="free_area"
        )
    assert res_free["ownConsumption"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "fuel_type, mastr_label",
    [
        ("lignite", "Braunkohle"),
        ("hard coal", "Steinkohle"),
        ("nuclear", "Kernenergie"),
    ],
)
def test_get_power_plant_in_area_uses_fuel_parameters(
    mock_infrastructure, fuel_type, mastr_label
):
    # MaStR returns German fuel labels; the technical parameters must still
    # be looked up by the internal fuel name, not fall back to gas_combined
    from assume.scenario.oeds.static import technical_parameter

    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE1001"],
            "fuel": [mastr_label],
            "lon": [6.0],
            "lat": [50.0],
            "startDate": [pd.to_datetime("2010-01-01")],
            "endDate": [pd.to_datetime("2040-01-01")],
            "maxPower": [500000.0],
            "turbineTyp": ["Kondensationsmaschine ohne Entnahme"],
            "generatorID": [0],
        }
    )
    if fuel_type != "nuclear":
        raw_df["kwkPowerTherm"] = [0.0]
        raw_df["kwkPowerElec"] = [0.0]
        raw_df["combination"] = [None]

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_power_plant_in_area(
            area=52353, fuel_type=fuel_type
        )

    params = technical_parameter[fuel_type][2000]
    row = result.iloc[0]
    assert row["fuel"] == fuel_type
    assert row["eta"] == pytest.approx(params["eta"] / 100)
    assert row["minPower"] == pytest.approx(500000.0 * params["minPower"] / 100)


def test_get_solar_systems_in_area_returns_serializable_columns(
    mock_infrastructure, tmp_path
):
    # InanspruchnahmeZahlungNachEeg is boolean and DatumEndgueltigeStilllegung is
    # NULL for units still standing, so both came back as object columns mixing
    # types, which pandas refuses to write to parquet. Callers caching the fleet
    # for an offline run need the frame to round-trip.
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE10001", "SEE10002"],
            "maxPower": [30.0, 3000.0],
            "acPower": [28.0, 2800.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "azimuthCode": ["Süd", "Süd"],
            "limited": ["Nein", None],
            "ownConsumption": [0, 0],
            "tiltCode": ["21 - 40 Grad", "21 - 40 Grad"],
            "startDate": ["2019-01-01", "2019-01-01"],
            "endDate": [None, "2023-06-01"],
            "eeg": [True, None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

    assert result["eeg"].tolist() == [1, 0]
    assert pd.api.types.is_integer_dtype(result["eeg"])
    assert pd.api.types.is_datetime64_any_dtype(result["startDate"])
    assert pd.api.types.is_datetime64_any_dtype(result["endDate"])
    assert result["endDate"].isna().tolist() == [True, False]

    path = tmp_path / "solar.parquet"
    result.to_parquet(path)
    assert len(pd.read_parquet(path)) == len(result)


def test_get_water_storage_systems_matches_name_substring(mock_infrastructure):
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE1", "SEE2"],
            "locationID": ["SLO1", "SLO2"],
            "storageID": ["SPE1", "SPE2"],
            "name": ["PSW Markersbach 1", None],
            "startDate": ["2018-01-01", "2018-01-01"],
            "PMinus_max": [250000.0, 100000.0],
            "VMax": [None, 0.0],
            "PPlus_max": [250000.0, 100000.0],
            "lon": [12.0, 12.0],
            "lat": [50.0, 50.0],
        }
    )
    with patch("pandas.read_sql", return_value=raw_df):
        storages = mock_infrastructure.get_water_storage_systems(area=52353)

    assert len(storages) == 1
    # Markersbach is mapped to 4018 MWh -> 4018000 kWh
    assert storages[0]["capacity"] == 4018 * 1e3
    assert storages[0]["unitID"] == "SPE1"
