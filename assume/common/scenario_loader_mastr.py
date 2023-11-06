import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

geo_information = gpd.read_file(r"./interfaces/data/NUTS_EU.shp")
plz_nuts = pd.read_csv(r"./interfaces/data/plz_to_nuts.csv", sep=";")
plz_nuts["CODE"] = [int(str_.replace("'", "")) for str_ in plz_nuts["CODE"]]
plz_nuts["NUTS3"] = [str_.replace("'", "") for str_ in plz_nuts["NUTS3"]]

mastr_codes_fossil = pd.read_csv(
    r"./interfaces/data/mastr_codes_fossil.csv", index_col=0
)
mastr_codes_solar = pd.read_csv(r"./interfaces/data/mastr_codes_solar.csv", index_col=0)
mastr_codes_wind = pd.read_csv(r"./interfaces/data/mastr_codes_wind.csv", index_col=0)
storage_volumes = pd.read_csv(
    r"./interfaces/data/technical_parameter_storage.csv", index_col=0
)


def get_lon_lat(area):
    point = (
        geo_information[geo_information["NUTS_ID"] == area]["geometry"]
        .values[0]
        .centroid
    )
    longitude, latitude = point.coords.xy
    return longitude[0], latitude[0]


def get_plz_codes(area):
    return plz_nuts.loc[plz_nuts["NUTS3"].str.startswith(area), "CODE"].to_numpy()


class InfrastructureInterface:
    def __init__(
        self, name, db_server_uri, structure_databases=("mastr", "oep", "windmodel")
    ):
        self.database_mastr = create_engine(
            f"{db_server_uri}/{structure_databases[0]}",
            connect_args={"application_name": name},
        )
        self.database_oep = create_engine(
            f"{db_server_uri}/{structure_databases[1]}",
            connect_args={"application_name": name},
        )
        self.database_wind = create_engine(
            f"{db_server_uri}/{structure_databases[2]}",
            connect_args={"application_name": name},
        )

    def get_power_plant_in_area(self, area=520, fuel_type="lignite"):
        data_frames = []
        if area in geo_information["NUTS_ID"].values:
            longitude, latitude = get_lon_lat(area)
            plz_codes = get_plz_codes(area)

            for plz_code in plz_codes:
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
                if fuel_type != "nuclear":
                    query += f"""
                        ,
                        kwk."ThermischeNutzleistung" as "kwkPowerTherm",
                        kwk."ElektrischeKwkLeistung" as "kwkPowerElec",
                        ev."AnlageIstImKombibetrieb" as "combination"
                        FROM "EinheitenVerbrennung" ev
                        LEFT JOIN "AnlagenKwk" kwk ON kwk."KwkMastrNummer" = ev."KwkMastrNummer"
                        WHERE ev."Postleitzahl" = {plz_code}
                        AND ev."Energietraeger" = {mastr_codes_fossil.loc[fuel_type, "value"]}
                        AND ev."Nettonennleistung" > 5000 AND ev."EinheitBetriebsstatus" = 35
                        AND ev."ArtDerStilllegung" isnull;
                        """
                else:
                    query += f"""
                             FROM "EinheitenKernkraft" ev
                             WHERE ev."Postleitzahl" = {plz_code}
                             """

                df = pd.read_sql(query, self.database_mastr)

                if not df.empty:
                    type_years = np.asarray([0, 2000, 2018])  # technical setting typ
                    df["fuel"] = fuel_type  # current fuel typ
                    df["maxPower"] = df["maxPower"]  # Rated Power [kW]
                    df["minPower"] = df["maxPower"] * 0.5  # MinPower = 1/2 MaxPower
                    df["P0"] = df["minPower"] + 0.1
                    df["gradP"] = 0.1 * df["maxPower"]  # 10% Change per hour
                    df["gradM"] = 0.1 * df["maxPower"]  # 10% Change per hour
                    df["stopTime"] = 5  # default stop time 5h
                    df["runTime"] = 5  # default run time 5h
                    df["on"] = 1  # on counter --> Plant is on till 1 hour
                    df["off"] = 0  # off counter --> Plant is on NOT off
                    df["eta"] = 0.3  # efficiency
                    df["chi"] = 1.0  # emission factor [t/MWh therm]
                    df["startCost"] = 100 * df["maxPower"]  # starting cost [€/kW Rated]
                    df["turbineTyp"] = [
                        mastr_codes_fossil.loc[str(x), "value"]  # convert int to string
                        if x is not None
                        else None
                        for x in df["turbineTyp"].to_numpy(int)
                    ]
                    df["startDate"] = [
                        pd.to_datetime(x)
                        if x is not None  # convert to timestamp and set default to 2005
                        else pd.to_datetime("2005-05-05")
                        for x in df["startDate"]
                    ]
                    if (
                        "combination" in df.columns
                    ):  # if no combination flag is set, set it to 0
                        df["combination"] = [
                            0 if x == 0 or x is None else 1 for x in df["combination"]
                        ]
                    else:  # Add column for nuclear power plants
                        df["combination"] = 0
                    df["type"] = [
                        type_years[type_years < x.year][-1] for x in df["startDate"]
                    ]
                    df["generatorID"] = [
                        id_ if id_ is not None else 0  # set default generatorID to 0
                        for id_ in df["generatorID"]
                    ]
                    if "kwkPowerTherm" in df.columns:
                        df["kwkPowerTherm"] = df["kwkPowerTherm"].fillna(0)
                        df["kwkPowerTherm"] = df["kwkPowerTherm"]
                    else:
                        df["kwkPowerTherm"] = 0
                    if "kwkPowerElec" in df.columns:
                        df["kwkPowerElec"] = df["kwkPowerElec"].fillna(0)
                        df["kwkPowerElec"] = df["kwkPowerElec"]
                    else:
                        df["kwkPowerElec"] = 0

                    # for all gas turbines check if they are used in a combination of gas and steam turbine
                    if fuel_type == "gas":
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
                                cchp["turbineTyp"] = "Closed Cycle Heat Power"
                                cchp["fuel"] = "gas_combined"
                                for line in range(len(cchp)):
                                    new_cchps.append(cchp.iloc[line])  # append all rows

                        # combine the gas turbines without combination flag with the new created
                        df = pd.concat(
                            [df[df["combination"] == 0], pd.DataFrame(new_cchps)]
                        )
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

                    # Set technical parameter corresponding to the type (0, 2000, 2018)
                    technical_parameter = pd.read_csv(
                        rf"./interfaces/data/technical_parameter_{fuel_type}.csv",
                        sep=";",
                        decimal=",",
                        index_col=0,
                    )
                    # start cost given in [€/MW]
                    # chi in [t CO2/MWh therm.]

                    for line, row in df.iterrows():
                        start_cost = (
                            technical_parameter.at[row["type"], "startCost"] / 1e3
                        )  # [€/MW] -> [€/kW]
                        df.at[line, "minPower"] = (
                            df.at[line, "maxPower"]
                            * technical_parameter.at[row["type"], "minPower"]
                            / 100
                        )
                        df.at[line, "P0"] = df.at[line, "minPower"]
                        df.at[line, "gradP"] = np.round(
                            df.at[line, "maxPower"]
                            * technical_parameter.at[row["type"], "gradP"]
                            * 60
                            / 100,
                            2,
                        )
                        df.at[line, "gradM"] = np.round(
                            df.at[line, "maxPower"]
                            * technical_parameter.at[row["type"], "gradM"]
                            * 60
                            / 100,
                            2,
                        )
                        df.at[line, "eta"] = (
                            technical_parameter.at[row["type"], "eta"] / 100
                        )  # convert to percentage
                        df.at[line, "chi"] = (
                            technical_parameter.at[row["type"], "chi"] / 1e3
                        )  # [t CO2/MWh therm.] -> [t CO2/kWh therm.]
                        df.at[line, "stopTime"] = technical_parameter.at[
                            row["type"], "stopTime"
                        ]
                        df.at[line, "runTime"] = technical_parameter.at[
                            row["type"], "runTime"
                        ]
                        df.at[line, "startCost"] = (
                            df.at[line, "maxPower"] * start_cost * 2
                        )  # multiply by 2 to respect heat demand

                    data_fraxmes.append(df)

        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

    def get_solar_systems_in_area(self, area=520, solar_type="roof_top"):
        data_frames = []
        if area in geo_information["NUTS_ID"].values:
            longitude, latitude = get_lon_lat(area)
            plz_codes = get_plz_codes(area)

            for plz_code in plz_codes:
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
                    f'WHERE "Postleitzahl" = {plz_code} '
                    f'AND "Lage" = {mastr_codes_solar.loc[solar_type, "value"]} '
                    f'AND "EinheitBetriebsstatus" = 35;'
                )

                # Get Data from Postgres
                df = pd.read_sql(query, self.database_mastr)
                # If the response Dataframe is not empty set technical parameter
                if not df.empty:
                    # all PVs with are implemented in 2018
                    df["startDate"] = pd.to_datetime(
                        df["startDate"], infer_datetime_format=True
                    )
                    # all PVs with nan are south oriented assets
                    df["azimuth"] = [
                        mastr_codes_solar.loc[str(code), "value"]
                        for code in df["azimuthCode"].to_numpy(int)
                    ]
                    del df["azimuthCode"]
                    # all PVs with nan have a tilt angle of 30°
                    df["tilt"] = [
                        mastr_codes_solar.loc[str(code), "value"]
                        for code in df["tiltCode"].to_numpy(int)
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
                            mastr_codes_solar.loc[str(code), "value"]
                            for code in df["limited"].to_numpy(int)
                        ]
                    if solar_type == "free_area" or solar_type == "other":
                        # TODO: Check restrictions for solar power plant
                        # nans have no limitation
                        df["limited"] = df["limited"].fillna(802)
                        df["limited"] = [
                            mastr_codes_solar.loc[str(code), "value"]
                            for code in df["limited"].to_numpy(int)
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

                    data_frames.append(df)

        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

    def get_wind_turbines_in_area(self, area=520, wind_type="on_shore"):
        data_frames = []

        if area in geo_information["NUTS_ID"].values:
            longitude, latitude = get_lon_lat(area)
            plz_codes = get_plz_codes(area)

            for plz_code in plz_codes:
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
                    f'AND "Lage" = {mastr_codes_wind.loc[wind_type, "value"]}'
                )
                if wind_type == "on_shore":
                    query += f' AND "Postleitzahl" = {plz_code};'

                # Get Data from Postgres
                df = pd.read_sql(query, self.database_mastr)
                # If the response Dataframe is not empty set technical parameter
                if not df.empty:
                    # all WEA with nan set hight to mean value
                    df["height"] = df["height"].fillna(df["height"].mean())
                    # all WEA with nan set hight to mean diameter
                    df["diameter"] = df["diameter"].fillna(df["diameter"].mean())
                    # all WEA with na are on shore and not allocated to a sea cluster
                    df["nordicSea"] = df["nordicSea"].fillna(0)
                    df["balticSea"] = df["balticSea"].fillna(0)
                    # get name of manufacturer
                    df["manufacturer"] = [
                        mastr_codes_wind.loc[str(x), "value"]
                        for x in df["manufacturer"].to_numpy(int)
                    ]
                    # try to find the correct type TODO: Check Pattern of new turbines
                    # df['typ'] = [str(typ).replace(' ', '').replace('-', '').upper() for typ in df['typ']]
                    # df['typ'] = [None if re.search(self.pattern_wind, typ) is None else re.search(self.pattern_wind, typ).group()
                    #             for typ in df['typ']]
                    # df['typ'] = df['typ'].replace('', 'default')
                    # set tag for wind farms
                    wind_farm_prefix = f"{area * 10}F"
                    df["windFarm"] = "x"
                    counter = 0
                    for genId in df["generatorID"].unique():
                        if (
                            genId is not None
                            and len(df[df["generatorID"] == genId]) > 1
                        ):
                            windFarm = df[df["generatorID"] == genId]
                            for line, row in windFarm.iterrows():
                                df.at[line, "windFarm"] = f"{wind_farm_prefix}{counter}"
                            counter += 1
                    data_frames.append(df)

        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

    def get_biomass_systems_in_area(self, area=520):
        data_frames = []

        if area in geo_information["NUTS_ID"].values:
            longitude, latitude = get_lon_lat(area)
            plz_codes = get_plz_codes(area)

            for plz_code in plz_codes:
                # TODO: Add more Parameters, if the model get more complex
                query = (
                    f'SELECT "EinheitMastrNummer" as "unitID", '
                    f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate", '
                    f'"Nettonennleistung" as "maxPower", '
                    f'COALESCE("Laengengrad", {longitude}) as "lon", '
                    f'COALESCE("Breitengrad", {latitude}) as "lat" '
                    f'FROM "EinheitenBiomasse"'
                    f'WHERE "Postleitzahl" = {plz_code} AND'
                    f'"EinheitBetriebsstatus" = 35 ;'
                )

                # Get Data from Postgres
                df = pd.read_sql(query, self.database_mastr)
                # If the response Dataframe is not empty set technical parameter
                if not df.empty:
                    data_frames.append(df)
        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

    def get_run_river_systems_in_area(self, area=520):
        data_frames = []

        if area in geo_information["NUTS_ID"].values:
            plz_codes = get_plz_codes(area)
            longitude, latitude = get_lon_lat(area)

            for plz_code in plz_codes:
                query = (
                    f'SELECT "EinheitMastrNummer" as "unitID", '
                    f'COALESCE("Inbetriebnahmedatum", \'2018-01-01\') as "startDate", '
                    f'"Nettonennleistung" as "maxPower", '
                    f'COALESCE("Laengengrad", {longitude}) as "lon", '
                    f'COALESCE("Breitengrad", {latitude}) as "lat" '
                    f'FROM "EinheitenWasser"'
                    f'WHERE "Postleitzahl" = {plz_code} AND '
                    f'"EinheitBetriebsstatus" = 35 AND "ArtDerWasserkraftanlage" = 890'
                )

                # Get Data from Postgres
                df = pd.read_sql(query, self.database_mastr)
                # If the response Dataframe is not empty set technical parameter
                if not df.empty:
                    data_frames.append(df)

        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

    def get_water_storage_systems(self, area=800):
        data_frames = []
        if area in geo_information["NUTS_ID"].values:
            longitude, latitude = get_lon_lat(area)
            plz_codes = get_plz_codes(area)

            for plz_code in plz_codes:
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
                    f'WHERE "Postleitzahl" = {plz_code} AND '
                    f'"EinheitBetriebsstatus" = 35 AND "Technologie" = 1537 AND "EinheitSystemstatus"=472 AND "Land"=84 '
                    f'AND "Nettonennleistung" > 500'
                )
                # print(query)
                # Get Data from Postgres
                df = pd.read_sql(query, self.database_mastr)

                # If the response Dataframe is not empty set technical parameter
                if not df.empty:
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
                        # TODO validate that storage_volumes is in [kW]
                        for key, value in storage_volumes.iterrows():
                            if key in row["name"]:
                                df.at[index, "VMax"] = value["volume"] * 1e3

                    storages = []
                    for id_ in df["storageID"].unique():
                        data = df[df["storageID"] == id_]
                        storage = {
                            "unitID": id_,
                            "startDate": pd.to_datetime(
                                data["startDate"].to_numpy()[0]
                            ),
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
                    data_frames.append(pd.DataFrame(storages))

        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

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
        df = pd.read_sql(query, self.database_oep)
        return df

    def get_solar_storage_systems_in_area(self, area):
        data_frames = []
        if area in geo_information["NUTS_ID"].values:
            longitude, latitude = get_lon_lat(area)
            plz_codes = get_plz_codes(area)

            for plz_code in plz_codes:
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
                    f'WHERE so."Postleitzahl" = {plz_code} '
                    f'AND so."EinheitBetriebsstatus" = 35;'
                )

                # Get Data from Postgres
                df = pd.read_sql(query, self.database_mastr)
                # If the response Dataframe is not empty set technical parameter
                if not df.empty:
                    df["VMax"] = df["VMax"].fillna(10)
                    df["ownConsumption"] = df["ownConsumption"].replace(689, 1)
                    df["ownConsumption"] = df["ownConsumption"].replace(688, 0)
                    df["limited"] = [
                        mastr_codes_solar.loc[str(code), "value"]
                        for code in df["limited"].to_numpy(int)
                    ]

                    # all PVs with nan are south oriented assets
                    df["azimuth"] = [
                        mastr_codes_solar.loc[str(code), "value"]
                        for code in df["azimuthCode"].to_numpy(int)
                    ]
                    del df["azimuthCode"]
                    # all PVs with nan have a tilt angle of 30°
                    df["tilt"] = [
                        mastr_codes_solar.loc[str(code), "value"]
                        for code in df["tiltCode"].to_numpy(int)
                    ]
                    del df["tiltCode"]
                    # assumption "regenerative Energiesysteme":
                    # a year has 1000 hours peak
                    df["demandP"] = df["maxPower"] * 1e3

                    df["eta"] = 0.96
                    df["V0"] = 0

                    data_frames.append(df)

        return pd.concat(data_frames) if len(data_frames) else pd.DataFrame()

    def get_grid_nodes(self):
        return {}

    def get_grid_edges(self):
        return {}


def get_pwp_agents(interface, areas):
    pwp_agents = []
    for area in areas:
        print(area)
        plants = False
        for fuel in ["lignite", "gas", "coal", "nuclear"]:
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
        plants = False
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
        if not str.empty:
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
    import os

    x = os.getenv("INFRASTRUCTURE_SOURCE", "10.13.10.55:4321")
    y = os.getenv("INFRASTRUCTURE_LOGIN", "readonly:readonly")
    uri = f"postgresql://{y}@{x}"
    interface = InfrastructureInterface("test", uri)
    # x = interface.get_power_plant_in_area(area='DEA2D', fuel_type='gas')
    # y = interface.get_water_storage_systems(area=415)
    # z = interface.get_solar_storage_systems_in_area(area=415)
    # a = interface.get_run_river_systems_in_area(area='DE111')
    areas = np.unique(plz_nuts["NUTS3"].to_numpy())

    import json

    create_agents = False
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
    # solar = interface.get_solar_storage_systems_in_area('DE7')

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
