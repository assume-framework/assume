# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import holidays
import numpy as np
import pandas as pd
from demandlib.bdew import ElecSlp
from pvlib.irradiance import erbs
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from sqlalchemy import create_engine
from tqdm import tqdm

# !pip install git+https://github.com/wind-python/windpowerlib@dev
from windpowerlib import ModelChain, WindTurbine

from assume.scenario.oeds.static import (
    fuel_translation,
    mastr_solar_azimuth,
    mastr_solar_codes,
    mastr_storage,
    technical_parameter,
)

WEATHER_PARAMS_ECMWF = [
    "temp_air",
    "ghi",
    "wind_speed",
]


class InfrastructureInterface:
    def __init__(
        self,
        name,
        db_server_uri,
        structure_databases=("mastr", "oep", "windmodel", "nuts", "scigrid", "weather"),
    ):
        self.databases = {}
        for db in structure_databases:
            self.databases[db] = create_engine(
                f"{db_server_uri}/{db}",
                connect_args={"application_name": name},
            )
        self.setup()

    def setup(self):
        with self.databases["nuts"].connect() as conn:
            self.plz_nuts = pd.read_sql_query(
                "select code, nuts3, longitude, latitude from plz",
                conn,
                index_col="code",
            )

        query = 'select kw."Id", "Wert", "Name" from "Katalogwerte" kw join "Katalogkategorien" kk on kw."KatalogKategorieId"=kk."Id"'
        with self.databases["mastr"].connect() as conn:
            katalogwerte = pd.read_sql_query(query, conn, index_col="Id")

        energietraeger = katalogwerte[
            katalogwerte["Name"].str.contains("Energieträger")
        ]
        del energietraeger["Name"]
        energietraeger = energietraeger.to_dict()["Wert"]

        self.energietraeger_translated = {
            fuel_translation.get(y, "unknown"): x for x, y in energietraeger.items()
        }

        verbrennungsanlagen = katalogwerte[
            katalogwerte["Name"] == "TechnologieVerbrennungsanlagen"
        ]
        kernkraft = katalogwerte[katalogwerte["Name"] == "TechnologieKernkraft"]
        verbrennungsanlagen = pd.concat([verbrennungsanlagen, kernkraft])
        del verbrennungsanlagen["Name"]
        self.mastr_generation_codes = verbrennungsanlagen.to_dict()["Wert"]
        # solarlage = katalogwerte[katalogwerte["Name"] == "SolarLage"]

        windlage = katalogwerte[katalogwerte["Name"] == "WindLage"]
        del windlage["Name"]
        windlage = windlage.to_dict()["Wert"]
        windlage_translation = {
            "Windkraft an Land": "on_shore",
            "Windkraft auf See": "off_shore",
        }

        self.mastr_wind_type = {
            windlage_translation.get(y, "unknown"): x for x, y in windlage.items()
        }

        windhersteller = katalogwerte[katalogwerte["Name"] == "WindHersteller"]
        del windhersteller["Name"]
        self.windhersteller = windhersteller.to_dict()["Wert"]

    def get_lat_lon(self, plz):
        if not isinstance(plz, int):
            raise ValueError(f"invalid plz {plz}")
        latitude, longitude = self.plz_nuts.loc[plz, ["latitude", "longitude"]]
        return latitude, longitude

    def get_lat_lon_area(self, area):
        if not area.startswith("DE"):
            return self.get_lat_lon(area)
        plz_codes = self.get_plz_codes(area)
        lat_lons = [self.get_lat_lon(plz) for plz in plz_codes]
        lat, lon = np.array(lat_lons).mean(axis=0)
        return lat, lon

    def get_plz_codes(self, area):
        plzs = self.plz_nuts["nuts3"].str.startswith(area)
        return list(self.plz_nuts.loc[plzs].index)

    def aggregate_cchps(self, df):
        # CCHP Power Plant with Combination
        cchps = df[df["combination"] == 1]
        new_cchps = []
        # aggregate with generatorID
        for genID in cchps["generatorID"].unique():
            if genID != 0:
                cchp = cchps[cchps["generatorID"] == genID]
                cchp.index = range(len(cchp))
                cchp.at[0, "maxPower"] = sum(cchp["maxPower"])
                cchp.at[0, "kwkPowerTherm"] = sum(cchp["kwkPowerTherm"])
                cchp.at[0, "kwkPowerElec"] = sum(cchp["kwkPowerElec"])
                cchp.at[0, "turbineTyp"] = "Closed Cycle Heat Power"
                cchp.at[0, "fuel"] = "gas_combined"
                new_cchps.append(
                    cchp.loc[0, cchp.columns]
                )  # only append the aggregated row!
            else:
                cchp = cchps[cchps["generatorID"] == 0]
                cchp.at[0, "turbineTyp"] = "Closed Cycle Heat Power"
                cchp.at[0, "fuel"] = "gas_combined"
                for line in range(len(cchp)):
                    new_cchps.append(cchp.iloc[line])  # append all rows

        # combine the gas turbines without combination flag with the new created
        df = pd.concat([df[df["combination"] == 0], pd.DataFrame(new_cchps)])
        df.index = range(len(df))

        # check the gas turbines with non set combination flag but turbine = typ Closed Cycle Heat Power
        for line, row in df.iterrows():
            if all(
                [
                    row["combination"] == 0,
                    row["turbineTyp"] == "Closed Cycle Heat Power",
                    row["fuel"] == "gas",
                ]
            ):
                df.at[line, "fuel"] = "gas_combined"

        return df

    def set_default_params(self, df: pd.DataFrame):
        # df["maxPower"] is Rated Power [kW]
        df["minPower"] = df["maxPower"] * 0.5  # MinPower = 1/2 MaxPower
        df["P0"] = df["minPower"] + 0.1
        df["ramp_up"] = 0.1 * df["maxPower"]  # 10% Change per hour
        df["ramp_down"] = 0.1 * df["maxPower"]  # 10% Change per hour
        df["min_down_time"] = 5  # default min stop time 5h
        df["min_operating_time"] = 5  # default min run time 5h
        df["on"] = 1  # on counter --> Plant is on till 1 hour
        df["off"] = 0  # off counter --> Plant is on NOT off
        df["eta"] = 0.3  # efficiency
        df["chi"] = 1.0  # emission factor [t/MWh therm]
        df["start_cost"] = 100 * df["maxPower"]  # starting cost [€/kW Rated]

        df["turbineTyp"] = df["turbineTyp"].replace(self.mastr_generation_codes)

        df["startDate"] = df["startDate"].fillna(pd.to_datetime("2005-05-05"))
        df["startDate"] = pd.to_datetime(df["startDate"])
        if "combination" in df.columns:  # if no combination flag is set, set it to 0
            # 0 if None, else 1
            df["combination"] = df["combination"].notna().astype(int)
        else:  # Add column for nuclear power plants
            df["combination"] = 0

        type_years = np.asarray([0, 2000, 2024])  # technical setting typ
        df["type"] = [type_years[type_years < x.year][-1] for x in df["startDate"]]
        df["generatorID"] = df["generatorID"].fillna(0)
        if "kwkPowerTherm" in df.columns:
            df["kwkPowerTherm"] = df["kwkPowerTherm"].fillna(0)
        else:
            df["kwkPowerTherm"] = 0

        if "kwkPowerElec" in df.columns:
            df["kwkPowerElec"] = df["kwkPowerElec"].fillna(0)
        else:
            df["kwkPowerElec"] = 0
        return df

    def get_power_plant_in_area(self, area=52353, fuel_type="lignite"):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)

        query = f"""
            SELECT ev."EinheitMastrNummer" as "unitID",
            ev."Energietraeger" as "fuel",
            COALESCE(ev."Laengengrad", {longitude}) as "lon",
            COALESCE(ev."Breitengrad", {latitude}) as "lat",
            COALESCE(ev."Inbetriebnahmedatum", '2010-01-01') as "startDate",
            ev."Nettonennleistung" as "maxPower",
            COALESCE(ev."Technologie", 839) as "turbineTyp",
            ev."GenMastrNummer" as "generatorID"
            """
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"
        if fuel_type != "nuclear":
            query += f"""
                ,
                kwk."ThermischeNutzleistung" as "kwkPowerTherm",
                kwk."ElektrischeKwkLeistung" as "kwkPowerElec",
                ev."AnlageIstImKombibetrieb" as "combination"
                FROM "EinheitenVerbrennung" ev
                LEFT JOIN "AnlagenKwk" kwk ON kwk."KwkMastrNummer" = ev."KwkMastrNummer"
                WHERE ev."Postleitzahl" in {plz_codes_str}
                AND ev."Energietraeger" = {self.energietraeger_translated[fuel_type]}
                AND ev."Nettonennleistung" > 5000 AND ev."EinheitBetriebsstatus" = 35
                AND ev."ArtDerStilllegung" isnull;
                """
        else:
            query += f"""
                FROM "EinheitenKernkraft" ev
                WHERE ev."Postleitzahl" in {plz_codes_str}
                """

        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return df

        df["fuel"] = fuel_type  # current fuel typ
        df = self.set_default_params(df)

        # for all gas turbines check if they are used in a combination of gas and steam turbine
        if fuel_type == "gas":
            df = self.aggregate_cchps(df)

        # TODO this should not happen
        df = df[~df.type.isna()]

        for line, row in df.iterrows():
            type_year = row["type"]
            if fuel_type in technical_parameter:
                tech_params = technical_parameter[fuel_type][type_year]
            else:
                tech_params = technical_parameter["gas_combined"][0]

            df.at[line, "minPower"] = (
                df.at[line, "maxPower"] * tech_params.get("minPower", 0) / 100
            )
            df.at[line, "P0"] = df.at[line, "minPower"]
            df.at[line, "ramp_up"] = np.round(
                df.at[line, "maxPower"] * tech_params["ramp_up"] * 60 / 100,
                2,
            )
            df.at[line, "ramp_down"] = np.round(
                df.at[line, "maxPower"] * tech_params["ramp_down"] * 60 / 100,
                2,
            )
            df.at[line, "eta"] = tech_params["eta"] / 100  # convert to percentage
            df.at[line, "chi"] = (
                tech_params["chi"] / 1e3
            )  # [t CO2/MWh therm.] -> [t CO2/kWh therm.]
            df.at[line, "min_down_time"] = tech_params["min_down_time"]
            df.at[line, "min_operating_time"] = tech_params["min_operating_time"]

            start_cost = tech_params["start_cost"] / 1e3  # [€/MW] -> [€/kW]
            df.at[line, "start_cost"] = (
                df.at[line, "maxPower"] * start_cost * 2
            )  # multiply by 2 to respect heat demand

        return df

    def get_solar_systems_in_area(self, area=520, solar_type="roof_top"):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"

        query = (
            f'SELECT "EinheitMastrNummer" as "unitID", '
            f'"Nettonennleistung" as "maxPower", '
            f'COALESCE("Laengengrad", {longitude}) as "lon", '
            f'COALESCE("Breitengrad", {latitude}) as "lat", '
            f'COALESCE("Hauptausrichtung", 699) as "azimuthCode", '
            f'"Leistungsbegrenzung" as "limited", '
            f'"Einspeisungsart" as "ownConsumption", '
            f'COALESCE("HauptausrichtungNeigungswinkel", 809) as "tiltCode", '
            f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate",'
            f'"InanspruchnahmeZahlungNachEeg" as "eeg" '
            f'FROM "EinheitenSolar" '
            f'INNER JOIN "AnlagenEegSolar" ON "EinheitMastrNummer" = "VerknuepfteEinheitenMastrNummern" '
            f'WHERE "Postleitzahl" in {plz_codes_str} '
            f'AND "Lage" = {mastr_solar_codes[solar_type]} '
            f'AND "EinheitBetriebsstatus" = 35;'
        )

        # Get Data from Postgres
        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)
        # If the response Dataframe is not empty set technical parameter
        if df.empty:
            return df

        # all PVs with are implemented in 2018
        df["startDate"] = pd.to_datetime(df["startDate"])
        # all PVs with nan are south oriented assets
        df["azimuth"] = [
            mastr_solar_azimuth[str(code)] for code in df["azimuthCode"].to_numpy(int)
        ]
        del df["azimuthCode"]
        # all PVs with nan have a tilt angle of 30°
        df["tilt"] = [
            mastr_solar_azimuth[str(code)] for code in df["tiltCode"].to_numpy(int)
        ]
        del df["tiltCode"]
        if solar_type == "roof_top":
            # all PVs with nan and startDate > 2013 have ownConsumption
            missing_values = df["ownConsumption"].isna()
            deadline = [date.year > 2013 for date in df["startDate"]]
            own_consumption = [
                all([missing_values[i], deadline[i]])
                for i in range(len(missing_values))
            ]
            df.loc[own_consumption, "ownConsumption"] = 1
            grid_use = [
                all([missing_values[i], not deadline[i]])
                for i in range(len(missing_values))
            ]
            df.loc[grid_use, "ownConsumption"] = 0
            df["ownConsumption"] = df["ownConsumption"].replace(689, 1)
            df["ownConsumption"] = df["ownConsumption"].replace(688, 0)
            # assumption "regenerative Energiesysteme":
            # a year has 1000 hours peak
            df["demandP"] = df["maxPower"] * 1e3
        elif solar_type == "free_area" or solar_type == "other":
            # set own consumption for solar power plant mounted PVs to 0, because the demand is unknown
            df["ownConsumption"] = 0
        if solar_type == "roof_top":
            # all PVs with nan and startDate > 2012 and maxPower > 30 kWp are limited to 70%
            missing_values = df["limited"].isna()
            power_cap = df["maxPower"] > 30
            deadline = [date.year > 2012 for date in df["startDate"]]
            limited = [
                all([missing_values[i], deadline[i], power_cap[i]])
                for i in range(len(missing_values))
            ]
            df.loc[limited, "limited"] = 803
            # rest nans have no limitation
            df["limited"] = df["limited"].fillna(802)
            df["limited"] = [
                mastr_solar_azimuth[str(code)] for code in df["limited"].to_numpy(int)
            ]
        if solar_type == "free_area" or solar_type == "other":
            # TODO: Check restrictions for solar power plant
            # nans have no limitation
            df["limited"] = df["limited"].fillna(802)
            df["limited"] = [
                mastr_solar_azimuth[str(code)] for code in df["limited"].to_numpy(int)
            ]
        # all PVs with nan and startDate > 2016 and maxPower > 100 kWp have direct marketing
        missing_values = df["eeg"].isna()
        power_cap = df["maxPower"] > 100
        deadline = [date.year > 2016 for date in df["startDate"]]
        eeg = [
            all([missing_values[i], deadline[i], power_cap[i]])
            for i in range(len(missing_values))
        ]
        df.loc[eeg, "eeg"] = 0
        # rest nans are eeg assets and are managed by the tso
        df["eeg"] = df["eeg"].replace(np.nan, 0)
        return df

    def get_wind_turbines_in_area(self, area=520, wind_type="on_shore"):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"

        query = (
            f'SELECT "EinheitMastrNummer" as "unitID", '
            f'"Nettonennleistung" as "maxPower", '
            f'COALESCE("Laengengrad", {longitude}) as "lon", '
            f'COALESCE("Breitengrad", {latitude}) as "lat", '
            f'"Typenbezeichnung" as "typ", '
            f'COALESCE("Hersteller", -1) as "manufacturer", '
            f'"Nabenhoehe" as "height", '
            f'"Rotordurchmesser" as "diameter", '
            f'"ClusterNordsee" as "nordicSea", '
            f'"ClusterOstsee" as "balticSea", '
            f'"GenMastrNummer" as "generatorID", '
            f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate" '
            f'FROM "EinheitenWind" '
            f'WHERE "EinheitBetriebsstatus" = 35 '
            f'AND "Lage" = {self.mastr_wind_type[wind_type]}'
        )
        if wind_type == "on_shore":
            query += f' AND "Postleitzahl" in {plz_codes_str};'

        # Get Data from Postgres
        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)
        # If the response Dataframe is not empty set technical parameter
        if df.empty:
            return df
        # all WEA with nan set hight to mean value
        df["height"] = df["height"].fillna(df["height"].mean())
        # all WEA with nan set hight to mean diameter
        df["diameter"] = df["diameter"].fillna(df["diameter"].mean())
        # all WEA with na are on shore and not allocated to a sea cluster
        df["nordicSea"] = df["nordicSea"].fillna(0)
        df["balticSea"] = df["balticSea"].fillna(0)
        # get name of manufacturer
        df["manufacturer"] = df["manufacturer"].replace(self.windhersteller)
        # try to find the correct type TODO: Check Pattern of new turbines
        # df['typ'] = [str(typ).replace(' ', '').replace('-', '').upper() for typ in df['typ']]
        # df['typ'] = [None if re.search(self.pattern_wind, typ) is None else re.search(self.pattern_wind, typ).group()
        #             for typ in df['typ']]
        # df['typ'] = df['typ'].replace('', 'default')
        # set tag for wind farms
        wind_farm_prefix = f"{area}0F"
        df["windFarm"] = "x"
        counter = 0
        for genId in df["generatorID"].unique():
            if genId is not None and len(df[df["generatorID"] == genId]) > 1:
                windFarm = df[df["generatorID"] == genId]
                for line, row in windFarm.iterrows():
                    df.at[line, "windFarm"] = f"{wind_farm_prefix}{counter}"
                counter += 1
        return df

    def get_biomass_systems_in_area(self, area=520):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"

        # TODO: Add more Parameters, if the model get more complex
        query = (
            f'SELECT "EinheitMastrNummer" as "unitID", '
            f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate", '
            f'"Nettonennleistung" as "maxPower", '
            f'COALESCE("Laengengrad", {longitude}) as "lon", '
            f'COALESCE("Breitengrad", {latitude}) as "lat" '
            f'FROM "EinheitenBiomasse"'
            f'WHERE "Postleitzahl" in {plz_codes_str} AND'
            f'"EinheitBetriebsstatus" = 35 ;'
        )

        # Get Data from Postgres
        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)
        # If the response Dataframe is not empty set technical parameter
        return df

    def get_run_river_systems_in_area(self, area=520):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"

        query = (
            f'SELECT "EinheitMastrNummer" as "unitID", '
            f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate", '
            f'"Nettonennleistung" as "maxPower", '
            f'COALESCE("Laengengrad", {longitude}) as "lon", '
            f'COALESCE("Breitengrad", {latitude}) as "lat" '
            f'FROM "EinheitenWasser" '
            f'WHERE "Postleitzahl"::int in {plz_codes_str} AND '
            f'"EinheitBetriebsstatus" = 35 AND "ArtDerWasserkraftanlage" = 890'
        )

        # Get Data from Postgres
        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)

        return df

    def get_water_storage_systems(self, area=800):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"

        query = (
            f'SELECT "EinheitMastrNummer" as "unitID", '
            f'"LokationMastrNummer" as "locationID", '
            f'"SpeMastrNummer" as "storageID", '
            f'"NameStromerzeugungseinheit" as "name", '
            f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate", '
            f'"Nettonennleistung" as "PMinus_max", '
            f'"NutzbareSpeicherkapazitaet" as "VMax", '
            f'"PumpbetriebLeistungsaufnahme" as "PPlus_max", '
            f'COALESCE("Laengengrad", {longitude}) as "lon", '
            f'COALESCE("Breitengrad", {latitude}) as "lat" '
            f'FROM "EinheitenStromSpeicher"'
            f'LEFT JOIN "AnlagenStromSpeicher" ON "EinheitMastrNummer" = "VerknuepfteEinheitenMastrNummern" '
            f'WHERE "Postleitzahl"::int in {plz_codes_str} AND '
            f'"EinheitBetriebsstatus" = 35 AND "Technologie" = 1537 AND "EinheitSystemstatus"=472 AND "Land"=84 '
            f'AND "Nettonennleistung" > 500'
        )
        # print(query)
        # Get Data from Postgres
        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)

        # If the response Dataframe is not empty set technical parameter
        if df.empty:
            return df

        print(df["name"])
        # set charge and discharge power
        df["PPlus_max"] = df["PPlus_max"].fillna(
            df["PMinus_max"]
        )  # fill na with Rated Power
        # df['PMinus_max'] = 0                                        # set min to zero
        # df['PPlus_max'] = 0                                         # set min to zero

        # fill nan values with default from wiki
        df["VMax"] = df["VMax"].fillna(0)
        df["VMax"] = df["VMax"]
        for index, row in df[df["VMax"] == 0].iterrows():
            # storage_volumes is in [MWh]
            df.at[index, "VMax"] = mastr_storage.get(row["name"], 0) * 1e3

        storages = []
        for id_ in df["storageID"].unique():
            data = df[df["storageID"] == id_]
            storage = {
                "unitID": id_,
                "startDate": pd.to_datetime(data["startDate"].to_numpy()[0]),
                "PMinus_max": data["PMinus_max"].sum(),
                "PPlus_max": data["PPlus_max"].sum(),
                "VMax": data["VMax"].to_numpy()[0],
                "VMin": 0,
                "V0": data["VMax"].to_numpy()[0] / 2,
                "lat": data["lat"].to_numpy()[0],
                "lon": data["lon"].to_numpy()[0],
                "eta_plus": 0.88,
                "eta_minus": 0.92,
            }
            # https://energie.ch/pumpspeicherkraftwerk/
            if storage["VMax"] > 0:
                storages.append(storage)
        return df

    def get_demand_in_area(self, area):
        if area == "DE91C":
            # nuts areas changed: https://en.wikipedia.org/wiki/NUTS_statistical_regions_of_Germany#Older_Version
            # upstream issue: https://github.com/openego/data_processing/issues/379
            DE915 = self.get_demand_in_area("DE915")
            DE919 = self.get_demand_in_area("DE919")
            return DE915 + DE919
        elif area == "DEB1C":
            return self.get_demand_in_area("DEB16")
        elif area == "DEB1D":
            return self.get_demand_in_area("DEB19")
        query = f"""select sum(sector_consumption_residential) as household, sum(sector_consumption_retail) as business,
                sum(sector_consumption_industrial) as industry, sum(sector_consumption_agricultural) as agriculture
                from demand where version='v0.4.5' and nuts LIKE '{area}%%'
                """
        with self.databases["oep"].connect() as conn:
            df = pd.read_sql(query, conn)
        # returned in GWh
        return df * 1e3  # convert to MWh

    def get_solar_storage_systems_in_area(self, area):
        if isinstance(area, str) and area.startswith("DE"):
            plz_codes = self.get_plz_codes(area)
            if not plz_codes:
                raise Exception("invalid areas")
        else:
            plz_codes = [area]

        for plz in plz_codes:
            if plz not in self.plz_nuts.index:
                raise Exception("invalid plz code")

        longitude, latitude = self.get_lat_lon_area(area)
        plz_codes_str = ", ".join([str(x) for x in plz_codes])
        plz_codes_str = f"({plz_codes_str})"

        query = (
            f'SELECT spe."LokationMastrNummer" as "unitID", '
            f'so."Nettonennleistung" as "maxPower", '
            f'spe."Nettonennleistung" as "batPower", '
            f'COALESCE(so."Laengengrad", {longitude}) as "lon", '
            f'COALESCE(so."Breitengrad", {latitude}) as "lat", '
            f'COALESCE(so."Hauptausrichtung", 699) as "azimuthCode", '
            f'COALESCE(so."Leistungsbegrenzung", 802) as "limited", '
            f'COALESCE(so."Einspeisungsart", 689) as "ownConsumption", '
            f'COALESCE(so."HauptausrichtungNeigungswinkel", 809) as "tiltCode", '
            f'COALESCE(so."Inbetriebnahmedatum", \'2018-01-01\') as "startDate", '
            f'an."NutzbareSpeicherkapazitaet" as "VMax" '
            f'FROM "EinheitenStromSpeicher" spe '
            f'INNER JOIN "EinheitenSolar" so ON spe."LokationMastrNummer" = so."LokationMastrNummer" '
            f'INNER JOIN "AnlagenStromSpeicher" an ON spe."SpeMastrNummer" = an."MastrNummer"'
            f'WHERE so."Postleitzahl" in {plz_codes_str} '
            f'AND so."EinheitBetriebsstatus" = 35;'
        )

        # Get Data from Postgres
        with self.databases["mastr"].connect() as conn:
            df = pd.read_sql(query, conn)

        # If the response Dataframe is not empty set technical parameter
        if df.empty:
            return df

        df["VMax"] = df["VMax"].fillna(10)
        df["ownConsumption"] = df["ownConsumption"].replace(689, 1)
        df["ownConsumption"] = df["ownConsumption"].replace(688, 0)
        df["limited"] = [
            mastr_solar_azimuth[str(code)] for code in df["limited"].to_numpy(int)
        ]

        # all PVs with nan are south oriented assets
        df["azimuth"] = [
            mastr_solar_azimuth[str(code)] for code in df["azimuthCode"].to_numpy(int)
        ]
        del df["azimuthCode"]
        # all PVs with nan have a tilt angle of 30°
        df["tilt"] = [
            mastr_solar_azimuth[str(code)] for code in df["tiltCode"].to_numpy(int)
        ]
        del df["tiltCode"]
        # assumption "regenerative Energiesysteme":
        # a year has 1000 hours peak
        df["demandP"] = df["maxPower"] * 1e3

        df["eta"] = 0.96
        df["V0"] = 0
        return df

    def get_demand_series_in_area(self, area, year=2019):
        demand = self.get_demand_in_area(area=area)
        demand = demand.T.to_dict()[0]
        ann_el_demand_per_sector = {
            "g0": demand["business"],
            # "h0": demand["household"],
            "h0_dyn": demand["household"],
            "l0": demand["agriculture"],
            "g3": demand["industry"],
        }
        # d = holidays.DE(subdiv='NW', years=year)
        holi = holidays.DE(years=year)
        e_slp = ElecSlp(year, holidays=holi)
        return e_slp.get_profile(ann_el_demand_per_sector)

    def get_weather_param(
        self,
        params: str | list[str],
        start: datetime,
        end: datetime,
        area: str = "",
    ):
        if isinstance(params, str):
            params = [params]
        params = [f"avg({p}) as {p}" for p in params]
        selection = ", ".join(params)
        query = f"SELECT time, {selection} FROM ecmwf_eu  WHERE time BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'"
        if area is not None:
            query += f" AND nuts_id LIKE '{area.upper()}%%'"
        query += "group by time"
        with self.databases["weather"].connect() as connection:
            return pd.read_sql_query(query, connection, index_col="time")

    def get_renewables_series_in_area(
        self, area: str | int, start: datetime, end: datetime
    ):
        # prepare weather
        longitude, latitude = self.get_lat_lon_area(area)
        location = Location(latitude, longitude, tz="Europe/Berlin")
        date_range = pd.date_range(start=start, end=end, freq="h")
        sun_position = location.get_solarposition(date_range)
        weather_df = self.get_weather_param(
            WEATHER_PARAMS_ECMWF,
            start,
            end,
            area,
        )
        # convert from J/m^2 to Wh/m^2
        weather_df["ghi"] /= 3600
        weather_df["zenith"] = sun_position["zenith"]
        weather_df["azimuth"] = sun_position["azimuth"]
        # calculate ghi and zenith
        calculated = erbs(
            np.array(weather_df["ghi"]),
            np.array(weather_df["zenith"]),
            date_range,
        )
        weather_df["dni"] = calculated["dni"]
        weather_df["dhi"] = calculated["dhi"]

        # load systems from mastr
        solar = self.get_solar_storage_systems_in_area(area)
        solar_sys = self.get_solar_systems_in_area(area)
        wind = self.get_wind_turbines_in_area(area)
        solar_series, bat_power = get_solar_series(solar_sys, weather_df)
        solar_series_sys, bat_power_sys = get_solar_series(solar, weather_df)
        wind_power = get_wind_series(wind, weather_df)
        solar_power = solar_series + solar_series_sys
        # conversion kW -> MW
        return solar_power / 1e3, wind_power / 1e3

    def get_grid_nodes(self):
        # get scigrid
        return {}

    def get_grid_edges(self):
        return {}


def get_wind_series(wind_systems: pd.DataFrame, weather_df: pd.DataFrame):
    data = [
        0.2 * np.ones(len(weather_df.index)),
        weather_df["temp_air"],
        weather_df["wind_speed"],
    ]
    columns = [["roughness_length", "temperature", "wind_speed"], [0, 2, 10]]
    ww = pd.DataFrame(
        np.asarray(data).T,
        index=weather_df.index,
        columns=columns,
    )

    wt = WindTurbine(82, turbine_type="E-82/2300")
    # todo get wind turbine types from database
    wind_power = pd.Series()
    std_curve = wt.power_curve
    std_curve["value"] = std_curve["value"] / wt.power_curve["value"].max()
    for line, row in tqdm(wind_systems.iterrows(), total=len(wind_systems)):
        max_power = row["maxPower"] * 1e3
        diameter = float(row["diameter"])
        height = float(row["height"])
        if height <= 0:
            # weird fix
            height = max_power / 20

        # small wind systems have no diameter set
        if diameter <= 0:
            diameter = height
        if diameter / 2 > height:
            diameter = height
        p_curve = std_curve.copy()
        p_curve["value"] = std_curve["value"] * max_power
        wt = WindTurbine(
            hub_height=height,
            rotor_diameter=diameter,
            nominal_power=max_power,
            power_curve=p_curve,
        )
        mc = ModelChain(wt).run_model(ww)
        wpower = mc.power_output / 1e3  # [W] -> [kW]
        wind_power += wpower
    return wind_power


def get_solar_series(solar_systems: pd.DataFrame, weather_df: pd.DataFrame):
    systems = []
    solar_power = pd.Series()
    battery_power = pd.Series()
    if solar_systems.empty:
        return solar_power, battery_power
    for info, group in tqdm(solar_systems.groupby(["azimuth", "tilt"])):
        azimuth = int(info[0])
        tilt = int(info[1])
        maxPower = group["maxPower"].sum()  # in kW

        if "batPower" in group.columns:
            battery_power += group["batPower"].sum()
        system = PVSystem(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            module_parameters={"pdc0": maxPower},
        )
        systems.append(system)

        ir = system.get_irradiance(
            solar_zenith=weather_df["zenith"],
            solar_azimuth=weather_df["azimuth"],
            dni=weather_df["dni"],
            ghi=weather_df["ghi"],
            dhi=weather_df["dhi"],
        )
        solar_power += ir["poa_global"] * maxPower
    solar_power /= 1e3  # W -> kW
    return solar_power, battery_power


def get_pwp_agents(interface, areas):
    pwp_agents = []
    for area in areas:
        print(area)
        plants = False
        for fuel in ["lignite", "gas", "oil", "hard coal", "nuclear"]:
            df = interface.get_power_plant_in_area(area=area, fuel_type=fuel)
            if not df.empty:
                plants = True
                break
        if plants:
            pwp_agents.append(area)
    return pwp_agents


def get_res_agents(interface, areas):
    res_agents = []
    for area in areas:
        print(area)
        wind = interface.get_wind_turbines_in_area(area=area)
        solar = interface.get_solar_storage_systems_in_area(area=area)
        bio = interface.get_biomass_systems_in_area(area=area)
        water = interface.get_run_river_systems_in_area(area=area)
        if any([not wind.empty, not solar.empty, not bio.empty, not water.empty]):
            res_agents.append(area)
    return res_agents


def get_storage_agents(interface, areas):
    str_agents = []
    for area in areas:
        print(area)
        str = interface.get_water_storage_systems(area)
        if str.empty:
            continue
        # print(str['name'])
        if any(str["PMinus_max"] > 1) and any(str["VMax"] > 1):
            print(f"add {area}")
            str_agents.append(area)
    return str_agents


def get_dem_agents(areas):
    dem_agents = []
    for area in areas:
        dem_agents.append(area)
    return dem_agents


if __name__ == "__main__":
    import json
    import os

    x = os.getenv("INFRASTRUCTURE_SOURCE", "timescale.nowum.fh-aachen.de:5432")
    y = os.getenv("INFRASTRUCTURE_LOGIN", "readonly:readonly")
    uri = f"postgresql://{y}@{x}"
    interface = InfrastructureInterface("test", uri)
    interface.get_plz_codes("DEF")
    interface.get_lat_lon(52379)
    # x = interface.get_power_plant_in_area(area='DEA2D', fuel_type='gas')
    # y = interface.get_water_storage_systems(area=415)
    # z = interface.get_solar_storage_systems_in_area(area=415)
    # a = interface.get_run_river_systems_in_area(area='DE111')
    areas = interface.plz_nuts["nuts3"].unique()

    create_agents = True
    if create_agents:
        agents = {}
        agents["dem"] = get_dem_agents(areas)
        agents["res"] = get_res_agents(interface, areas)
        agents["str"] = get_storage_agents(interface, areas)
        agents["pwp"] = get_pwp_agents(interface, areas)
        with open("../agents.json", "w") as f:
            json.dump(agents, f, indent=2)
    else:
        with open("../agents.json", "r") as f:
            agents = json.load(f)

    dem = interface.get_demand_in_area(area="DE91C")
    solar = interface.get_solar_storage_systems_in_area("DE7")
    lignite = interface.get_power_plant_in_area("DE", fuel_type="lignite")

    ## test DEM from NUTS2 vs NUTS3:
    level_3 = agents["dem"]
    level_2 = list({a[0 : 2 + 2] for a in level_3})
    level_1 = list({a[0 : 2 + 1] for a in level_3})

    summe = 0
    for a in level_1:
        print(a)
        dem_a = interface.get_demand_in_area(area=a)
        summe += dem_a
    print(summe)

    for a in level_2:
        print(a)
        l3_a = [ag for ag in level_3 if ag.startswith(a)]

        dem_a = interface.get_demand_in_area(area=a)
        print(dem_a)

        for area in l3_a:
            df = interface.get_demand_in_area(area=area)
            print(area, df)
            dem_a -= df.fillna(0)

        print(dem_a)
        assert (dem_a < 1e-10).all().all()

    ## infra tests
    year = 2020
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    index = pd.date_range(
        start=start,
        end=end + timedelta(hours=24),
        freq="h",
    )
    database = os.getenv("INFRASTRUCTURE_SOURCE", "timescale.nowum.fh-aachen.de:5432")
    login = os.getenv("INFRASTRUCTURE_LOGIN", "readonly:readonly")
    infra_uri = f"postgresql://{login}@{database}"
    infra_interface = InfrastructureInterface("test", infra_uri)

    solar = infra_interface.get_solar_storage_systems_in_area("DE123")
    solar_sys = infra_interface.get_solar_systems_in_area("DE127")
    wind = infra_interface.get_wind_turbines_in_area("DE7")
    weather_df = infra_interface.get_weather_param(
        WEATHER_PARAMS_ECMWF, start, end, "DEA"
    )
    weather_df_de = infra_interface.get_weather_param(
        WEATHER_PARAMS_ECMWF, start, end, "DE"
    )
    demand = infra_interface.get_demand_series_in_area("DEA", year)
    solar, wind = infra_interface.get_renewables_series_in_area("DE12", start, end)
