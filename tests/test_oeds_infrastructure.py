# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import numpy as np
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

        # The standalone gas unit (combination=0) should retain fuel == 'gas'
        gas_row = result[result["unitID"] == "SEE1002"].iloc[0]
        assert gas_row["fuel"] == "gas"
        assert gas_row["ramp_up"] == pytest.approx(
            gas_row["maxPower"] * 12.0 * 60 / 100
        )

        # No phantom rows or NaNs created
        assert len(result) == 2
        assert not result["unitID"].isna().any()


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


def test_resolve_area_rejects_invalid_inputs(mock_infrastructure):
    with pytest.raises(ValueError, match="invalid areas"):
        mock_infrastructure.resolve_area("DE_NONEXISTENT")

    with pytest.raises(ValueError, match="invalid plz code"):
        mock_infrastructure.resolve_area(99999)

    with pytest.raises(ValueError, match="invalid plz code"):
        mock_infrastructure.resolve_area("99999")


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
        assert (
            'AND (spe."DatumEndgueltigeStilllegung" IS NULL OR spe."DatumEndgueltigeStilllegung" > \'2023-01-01T00:00:00\')'
            in query
        )
        assert 'spe."EinheitBetriebsstatus" IN' in query


def test_get_solar_storage_systems_in_area_battery_dates_and_filters(
    mock_infrastructure,
):
    # Tests that batStartDate and batEndDate are parsed and reflect the later start
    # and earlier end of solar and storage units.
    from datetime import datetime

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    # Check created_before filter in query
    with patch("pandas.read_sql", return_value=pd.DataFrame()) as mock_read_sql:
        mock_infrastructure.get_solar_storage_systems_in_area(
            area=52353, created_before=datetime(2023, 1, 1)
        )
        query = mock_read_sql.call_args[0][0]
        assert "AND so.\"Inbetriebnahmedatum\" < '2023-01-01T00:00:00'" in query
        assert "AND spe.\"Inbetriebnahmedatum\" < '2023-01-01T00:00:00'" in query

    # Check date bounding: later start and earlier end
    raw_df = pd.DataFrame(
        {
            "unitID": ["SLO1", "SLO2"],
            "unitMastrID": ["SSE1", "SSE2"],
            "maxPower": [10.0, 10.0],
            "batPower": [5.0, 5.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "azimuthCode": ["Süd", "Süd"],
            "limited": ["Nein", "Nein"],
            "ownConsumption": ["Teileinspeisung", "Teileinspeisung"],
            "tiltCode": ["30 Grad", "30 Grad"],
            "startDate": ["2015-01-01", "2020-01-01"],
            "endDate": ["2035-01-01", "2030-01-01"],
            "batStartDate": ["2020-06-01", "2018-01-01"],  # retrofitted vs existing
            "batEndDate": ["2040-01-01", "2028-01-01"],
            "VMax": [8.0, 8.0],
        }
    )

    with patch("pandas.read_sql", return_value=raw_df):
        res = mock_infrastructure.get_solar_storage_systems_in_area(area=52353)

    # Unit 1: solar started 2015, battery started 2020 -> battery availability starts 2020-06-01
    # Unit 1: solar ends 2035, battery ends 2040 -> battery availability ends 2035-01-01 (earlier)
    assert res["batStartDate"].iloc[0] == pd.Timestamp("2020-06-01")
    assert res["batEndDate"].iloc[0] == pd.Timestamp("2035-01-01")

    # Unit 2: solar started 2020, battery started 2018 -> battery availability starts 2020-01-01
    # Unit 2: solar ends 2030, battery ends 2028 -> battery availability ends 2028-01-01 (earlier)
    assert res["batStartDate"].iloc[1] == pd.Timestamp("2020-01-01")
    assert res["batEndDate"].iloc[1] == pd.Timestamp("2028-01-01")


def test_get_solar_systems_in_area_power_cap_threshold(mock_infrastructure):
    # A system with 120 kWp installed capacity and a 70% feed-in limit
    # keeps its installed capacity in maxPower (the limit is applied as a
    # peak clip in get_solar_series) and must still qualify for direct
    # marketing (installed 120 kWp > 100 kWp threshold).
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

        # maxPower keeps the installed (pre-limit) capacity
        assert result["maxPower"].iloc[0] == pytest.approx(120.0)
        # demandP reflects the clipped peak: 120 * 0.7 * 1000 = 84000.0 W
        assert result["demandP"].iloc[0] == pytest.approx(84000.0)
        # limit_factor is carried for get_solar_series
        assert result["limit_factor"].iloc[0] == 0.7
        # eeg should be set to 0 (direct marketing) because installed capacity was 120 > 100
        assert result["eeg"].iloc[0] == 0


def test_get_solar_systems_in_area_missing_ac_power_falls_back_to_dc(
    mock_infrastructure,
):
    # units without Nettonennleistung must not turn demandP or the clip NaN
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE1", "SEE2"],
            "maxPower": [10.0, 10.0],
            "acPower": [None, 8.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "azimuthCode": ["Süd", "Süd"],
            "limited": [None, None],
            "ownConsumption": [None, None],
            "tiltCode": [None, None],
            "startDate": [pd.to_datetime("2019-01-01")] * 2,
            "eeg": [None, None],
        }
    )
    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

    assert result["acPower"].tolist() == [10.0, 8.0]
    assert not result["demandP"].isna().any()


def test_get_solar_systems_in_area_ost_west_split(mock_infrastructure):
    # An Ost-West double-row unit is split into a south-independent east
    # (90) and west (270) orientation at half capacity each, so total
    # capacity is preserved and the groupby in get_solar_series produces
    # two separate orientations.
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE10001", "SEE10002"],
            "maxPower": [30.0, 30.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "azimuthCode": ["Ost-West", "Süd"],
            "limited": ["Nein", "Nein"],
            "ownConsumption": [0, 0],
            "tiltCode": ["21 - 40 Grad", "21 - 40 Grad"],
            "startDate": [pd.to_datetime("2019-01-01"), pd.to_datetime("2019-01-01")],
            "eeg": [None, 0],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

        # total capacity unchanged
        assert result["maxPower"].sum() == pytest.approx(60.0)
        # Ost-West unit produced one east and one west row at half capacity
        ew = result[(result["azimuth"] == "90") | (result["azimuth"] == "270")]
        assert ew["maxPower"].tolist() == [15.0, 15.0]
        assert result[result["azimuth"] == "180"]["maxPower"].iloc[0] == 30.0


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


def test_get_solar_systems_in_area_ost_west_split_after_size_rules(
    mock_infrastructure,
):
    # a 50 kWp Ost-West unit must get the 70% feed-in limit for units
    # above 30 kWp before it is split into two 25 kWp rows
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE10001"],
            "maxPower": [50.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Ost-West"],
            "limited": [None],
            "ownConsumption": [None],
            "tiltCode": [None],
            "startDate": [pd.to_datetime("2019-01-01")],
            "eeg": [None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

    assert result["maxPower"].tolist() == [25.0, 25.0]
    assert result["limit_factor"].tolist() == [0.7, 0.7]
    assert result["tilt"].tolist() == ["30", "30"]


def test_get_solar_storage_ost_west_split_keeps_battery_totals(mock_infrastructure):
    # the Ost-West split must not double the battery of a PV+battery unit,
    # get_solar_series sums batPower per orientation group
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEL10001"],
            "unitMastrID": ["SSE10001"],
            "maxPower": [10.0],
            "batPower": [5.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Ost-West"],
            "limited": ["Nein"],
            "ownConsumption": ["Teileinspeisung (einschließlich Eigenverbrauch)"],
            "tiltCode": ["21 - 40 Grad"],
            "startDate": ["2020-01-01"],
            "VMax": [8.0],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_storage_systems_in_area(area=52353)

    assert sorted(result["azimuth"]) == ["270", "90"]
    assert result["maxPower"].sum() == pytest.approx(10.0)
    assert result["batPower"].sum() == pytest.approx(5.0)
    assert result["VMax"].sum() == pytest.approx(8.0)


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


def test_get_solar_systems_in_area_mixed_pv_fallbacks(mock_infrastructure):
    # Mixed query (solar_type=None) must apply the rooftop fallbacks to rooftop units
    # (Gebäudesolaranlage) and the general fallbacks to other units (Freiflächensolaranlage).
    raw_mixed = pd.DataFrame(
        {
            "unitID": ["SEE_ROOF_1", "SEE_ROOF_2", "SEE_FREE_1", "SEE_FREE_2"],
            "maxPower": [40.0, 40.0, 40.0, 40.0],
            "lon": [6.0, 6.0, 6.0, 6.0],
            "lat": [50.0, 50.0, 50.0, 50.0],
            "plzCode": ["52353", "52353", "52353", "52353"],
            "azimuthCode": ["Süd", "Süd", "Süd", "Süd"],
            "limited": [None, None, None, None],
            "ownConsumption": [None, None, None, None],
            "tiltCode": ["30 Grad", "30 Grad", "30 Grad", "30 Grad"],
            "startDate": [
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2012-01-01"),
                pd.to_datetime("2020-01-01"),
                pd.to_datetime("2012-01-01"),
            ],
            "eeg": [1, 1, 1, 1],
            "solar_type": [
                "Gebäudesolaranlage",
                "Gebäudesolaranlage",
                "Freiflächensolaranlage",
                "Freiflächensolaranlage",
            ],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql", return_value=raw_mixed):
        result_mixed = mock_infrastructure.get_solar_systems_in_area(
            area=52353, solar_type=None
        )

    # Rooftop units: 2020 gets ownConsumption=1, 2012 gets ownConsumption=0
    # Free-area units: both get ownConsumption=0
    assert result_mixed["ownConsumption"].tolist() == [1, 0, 0, 0]

    # Rooftop units: 2020 (>2012 and >30kW) gets limited="Ja, auf 70%", 2012 gets "Nein"
    # Free-area units: both get "Nein" (unlimited)
    assert result_mixed["limit_factor"].tolist() == [0.7, 1.0, 1.0, 1.0]
    assert result_mixed["limited"].tolist() == ["Ja, auf 70%", "Nein", "Nein", "Nein"]


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


def test_get_solar_series_applies_feed_in_limit_as_peak_clip():
    # A limited system must keep its rated capacity in maxPower but its
    # series must be clipped at limit_factor * rated, not scaled by it.
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21", periods=3, freq="h", tz="Europe/Berlin")
    weather_df = pd.DataFrame(
        {
            "zenith": [93.0, 10.0, 93.0],
            "azimuth": [180.0, 180.0, 180.0],
            "dni": [0.0, 900.0, 0.0],
            "ghi": [50.0, 450.0, 50.0],
            "dhi": np.zeros(3),
        },
        index=index,
    )

    rated = 100.0
    unlimited = pd.DataFrame({"azimuth": ["180"], "tilt": ["30"], "maxPower": [rated]})
    solar_open, _ = get_solar_series(unlimited, weather_df, inverter_efficiency=1.0)
    open_kw = solar_open.to_numpy()
    assert open_kw.max() > 0

    # MaStR semantics: output is clipped at limit_factor * rated power
    factor = 0.8
    expected = np.minimum(open_kw, factor * rated)

    systems = pd.DataFrame(
        {
            "azimuth": ["180"],
            "tilt": ["30"],
            "maxPower": [rated],
            "limit_factor": [factor],
        }
    )
    solar, _ = get_solar_series(systems, weather_df, inverter_efficiency=1.0)
    clipped_kw = solar.to_numpy()
    # clipped, never scaled: the shape below the clip level is unchanged,
    # the peak sits exactly at the clip level instead of the natural peak
    assert not np.allclose(clipped_kw, open_kw * factor)
    assert np.allclose(clipped_kw, expected)
    assert np.allclose(clipped_kw.max(), factor * rated)


def test_clipped_feed_in_matches_per_unit_clip():
    from assume.scenario.oeds.infrastructure import clipped_feed_in

    rng = np.random.default_rng(0)
    installed = rng.uniform(1, 100, 50)
    installed[3] = 0.0  # units without capacity are ignored
    cap = installed * rng.uniform(0.4, 1.0, 50)
    share = np.array([0.0, 0.3, 0.6, 0.85, 1.0, 1.1])

    expected = np.minimum(np.outer(share, installed), cap).sum(axis=1)
    assert np.allclose(clipped_feed_in(share, installed, cap), expected)


def test_get_solar_series_clips_units_at_ac_capacity():
    # maxPower is the installed DC capacity; units with a smaller inverter
    # (acPower) or a feed-in limit are clipped individually, even if they
    # share an orientation group
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21", periods=3, freq="h", tz="Europe/Berlin")
    weather_df = pd.DataFrame(
        {
            "zenith": [93.0, 10.0, 93.0],
            "azimuth": [180.0, 180.0, 180.0],
            "dni": [0.0, 900.0, 0.0],
            "ghi": [50.0, 450.0, 50.0],
            "dhi": np.zeros(3),
        },
        index=index,
    )
    unit = pd.DataFrame({"azimuth": ["180"], "tilt": ["30"], "maxPower": [100.0]})
    open_kw = get_solar_series(unit, weather_df)[0].to_numpy()

    systems = pd.DataFrame(
        {
            "azimuth": ["180", "180"],
            "tilt": ["30", "30"],
            "maxPower": [100.0, 100.0],
            "acPower": [60.0, 100.0],
            "limit_factor": [1.0, 0.7],
        }
    )
    solar, _ = get_solar_series(systems, weather_df)
    expected = np.minimum(open_kw, 60.0) + np.minimum(open_kw, 70.0)
    assert open_kw.max() > 70.0
    assert np.allclose(solar.to_numpy(), expected)


def test_wind_hub_height_and_diameter_are_filled_in_series(mock_infrastructure):
    # the query returns the raw MaStR values; missing hub heights and rotor
    # diameters are only filled for the simulation
    from assume.scenario.oeds.infrastructure import get_wind_series

    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE1", "SEE2"],
            "maxPower": [2000.0, 2000.0],
            "lon": [6.0, 6.0],
            "lat": [50.0, 50.0],
            "plzCode": ["52353", "52353"],
            "height": [100.0, None],
            "diameter": [None, 90.0],
            "generatorID": [None, None],
        }
    )

    mock_conn = MagicMock()
    mock_infrastructure.databases[
        "mastr"
    ].connect.return_value.__enter__.return_value = mock_conn

    with patch("pandas.read_sql_query", return_value=raw_df):
        turbines = mock_infrastructure.get_wind_turbines_in_area(area=52353)
    assert turbines["height"].isna().tolist() == [False, True]
    assert turbines["diameter"].isna().tolist() == [True, False]

    index = pd.date_range("2020-01-01", periods=3, freq="h")
    weather_df = pd.DataFrame(
        {"temp_air": [280.0] * 3, "wind_speed": [3.0, 8.0, 12.0]}, index=index
    )
    filled = turbines.fillna({"height": 100.0, "diameter": 90.0})
    assert np.allclose(
        get_wind_series(turbines, weather_df), get_wind_series(filled, weather_df)
    )
    assert get_wind_series(turbines, weather_df).max() > 0


def test_get_solar_series_applies_losses_and_aging_before_clip():
    # DC output is reduced by the inverter efficiency and by linear aging
    # at the middle of the simulated period, the AC capacity clips afterwards
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21 11:00", periods=3, freq="h")
    weather_df = pd.DataFrame(
        {
            "zenith": [93.0, 10.0, 93.0],
            "azimuth": [180.0, 180.0, 180.0],
            "dni": [0.0, 900.0, 0.0],
            "ghi": [50.0, 450.0, 50.0],
            "dhi": np.zeros(3),
        },
        index=index,
    )
    unit = pd.DataFrame({"azimuth": ["180"], "tilt": ["30"], "maxPower": [100.0]})
    lossless = get_solar_series(unit, weather_df, inverter_efficiency=1.0)[0]

    # commissioned ten years before the middle of the period
    aged = unit.assign(startDate=pd.Timestamp("2010-06-21 12:00"))
    solar = get_solar_series(aged, weather_df, inverter_efficiency=0.9)[0]
    expected_factor = 0.9 * (1 - 0.005 * 3653 / 365.25)
    assert np.allclose(solar, lossless * expected_factor)

    # units commissioned later in the period do not gain from negative aging
    new = unit.assign(startDate=pd.Timestamp("2020-06-21 12:00"))
    solar = get_solar_series(new, weather_df, inverter_efficiency=0.9)[0]
    assert np.allclose(solar.iloc[1:], lossless.iloc[1:] * 0.9)

    # the AC capacity clips the output after losses
    capped = unit.assign(acPower=60.0)
    solar = get_solar_series(capped, weather_df, inverter_efficiency=0.9)[0]
    assert np.allclose(solar, np.minimum(lossless * 0.9, 60.0))
    assert (lossless * 0.9).max() > 60.0

    # missing startDate is treated as unaged without producing NaNs
    missing_start = pd.DataFrame(
        [
            {"azimuth": "180", "tilt": "30", "maxPower": 100.0, "startDate": None},
            {
                "azimuth": "180",
                "tilt": "30",
                "maxPower": 100.0,
                "startDate": pd.Timestamp("2010-06-21 12:00"),
            },
        ]
    )
    solar_mixed = get_solar_series(missing_start, weather_df, inverter_efficiency=0.9)[
        0
    ]
    assert not solar_mixed.isna().any()
    expected_mixed = lossless * 0.9 + lossless * expected_factor
    assert np.allclose(solar_mixed, expected_mixed)


def test_series_only_count_units_while_operating():
    # a fleet growing within the simulated period must not be applied to
    # the whole period: units produce from startDate until endDate only
    from assume.scenario.oeds.infrastructure import get_solar_series, get_wind_series

    index = pd.date_range("2020-06-21 10:00", periods=5, freq="h")
    weather_df = pd.DataFrame(
        {
            "zenith": [30.0, 20.0, 10.0, 20.0, 30.0],
            "azimuth": [180.0] * 5,
            "dni": [700.0, 800.0, 900.0, 800.0, 700.0],
            "ghi": [500.0, 550.0, 600.0, 550.0, 500.0],
            "dhi": [100.0] * 5,
            "temp_air": [290.0] * 5,
            "wind_speed": [8.0] * 5,
        },
        index=index,
    )
    unit = {"azimuth": "180", "tilt": "30", "maxPower": 100.0, "batPower": 10.0}
    always = pd.DataFrame([unit])
    solar_always, battery_always = get_solar_series(always, weather_df)

    systems = pd.DataFrame(
        [
            # commissioned before the period, decommissioned at 12:00
            {
                **unit,
                "startDate": pd.Timestamp("2019-01-01"),
                "endDate": pd.Timestamp("2020-06-21 12:00"),
            },
            # commissioned at 12:00 within the period
            {**unit, "startDate": pd.Timestamp("2020-06-21 12:00"), "endDate": None},
            # commissioned after the period
            {**unit, "startDate": pd.Timestamp("2021-01-01"), "endDate": None},
        ]
    )
    solar, battery = get_solar_series(systems, weather_df, aging_rate=0)
    # exactly one of the three units operates at any time
    assert np.allclose(solar, solar_always)
    assert np.allclose(battery, battery_always)

    turbine = {"maxPower": 2000.0, "height": 100.0, "diameter": 90.0}
    wind_always = get_wind_series(pd.DataFrame([turbine]), weather_df)
    turbines = pd.DataFrame(
        [
            {
                **turbine,
                "startDate": pd.Timestamp("2020-06-21 12:00"),
                "endDate": pd.Timestamp("2050-01-01"),
            },
            {
                **turbine,
                "startDate": pd.Timestamp("2018-01-01"),
                "endDate": pd.Timestamp("2020-06-21 13:00"),
            },
        ]
    )
    wind = get_wind_series(turbines, weather_df)
    expected = wind_always * np.array([1, 1, 2, 1, 1])
    assert np.allclose(wind, expected)


def test_get_solar_series_retrofitted_battery_availability():
    # Retrofitted battery commissioned after the PV unit must not have its
    # capacity available before the battery commissioning date.
    from assume.scenario.oeds.infrastructure import get_solar_series

    index = pd.date_range("2020-06-21 10:00", periods=5, freq="h")
    weather_df = pd.DataFrame(
        {
            "zenith": [30.0, 20.0, 10.0, 20.0, 30.0],
            "azimuth": [180.0] * 5,
            "dni": [700.0, 800.0, 900.0, 800.0, 700.0],
            "ghi": [500.0, 550.0, 600.0, 550.0, 500.0],
            "dhi": [100.0] * 5,
            "temp_air": [290.0] * 5,
            "wind_speed": [8.0] * 5,
        },
        index=index,
    )
    # PV commissioned before the period (2018), retrofitted battery commissioned at 12:00 within the period
    system = pd.DataFrame(
        [
            {
                "azimuth": "180",
                "tilt": "30",
                "maxPower": 100.0,
                "batPower": 10.0,
                "startDate": pd.Timestamp("2018-01-01"),
                "endDate": None,
                "batStartDate": pd.Timestamp("2020-06-21 12:00"),
                "batEndDate": None,
            }
        ]
    )

    solar, battery = get_solar_series(system, weather_df, aging_rate=0)

    # Solar produces for all 5 timesteps (commissioned in 2018)
    assert (solar > 0).all()

    # Battery is 0 before 12:00, and 10.0 at and after 12:00
    expected_battery = pd.Series([0.0, 0.0, 10.0, 10.0, 10.0], index=index)
    assert np.allclose(battery, expected_battery)


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


def test_finish_solar_systems_handles_unknown_limited_codes(mock_infrastructure):
    raw_df = pd.DataFrame(
        {
            "unitID": ["SEE1"],
            "maxPower": [10.0],
            "acPower": [9.0],
            "lon": [6.0],
            "lat": [50.0],
            "plzCode": ["52353"],
            "azimuthCode": ["Süd"],
            "limited": ["UnknownLimitCode"],
            "ownConsumption": [0],
            "tiltCode": ["21 - 40 Grad"],
            "startDate": ["2019-01-01"],
            "endDate": [None],
            "eeg": [True],
        }
    )
    with patch("pandas.read_sql", return_value=raw_df):
        result = mock_infrastructure.get_solar_systems_in_area(area=52353)

    assert result["limit_factor"].iloc[0] == 1.0
