# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta

import dateutil.rrule as rr
import pandas as pd
import yaml
from yamlinclude import YamlIncludeConstructor

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.world import World

logger = logging.getLogger(__name__)

translate_clearing = {
    "SAME_SHARES": "pay_as_clear",
    "FIRST_COME_FIRST_SERVE": "pay_as_bid",
    "RANDOMIZE": "not_implemented_yet",
}
translate_fuel_type = {
    "NUCLEAR": "nuclear",
    "LIGNITE": "lignite",
    "HARD_COAL": "hard coal",
    "NATURAL_GAS": "natural gas",
    "OIL": "oil",
    "HYDROGEN": "hydrogen",
    "Biogas": "biomass",
    "PV": "solar",
    "WindOn": "wind_onshore",
    "WindOff": "wind_offshore",
    "RunOfRiver": "hydro",
    "Other": "hydro,",
}


def read_csv(base_path, filename):
    return pd.read_csv(
        base_path + "/" + filename,
        date_format="%Y-%m-%d_%H:%M:%S",
        sep=";",
        header=None,
        names=["time", "load"],
        index_col="time",
    )["load"]


def get_send_receive_msgs_per_id(agent_id, contracts_config: list):
    sends = []
    receives = []
    for contracts in contracts_config:
        for contract in contracts["Contracts"]:
            # sends
            if isinstance(contract["SenderId"], list):
                if agent_id in contract["SenderId"]:
                    sends.append(contract)
            elif agent_id == contract["SenderId"]:
                sends.append(contract)
            # receives
            if isinstance(contract["ReceiverId"], list):
                if agent_id in contract["ReceiverId"]:
                    receives.append(contract)
            elif agent_id == contract["ReceiverId"]:
                receives.append(contract)
    return sends, receives


def get_matching_send_one_or_multi(agent_id: int, contract: dict):
    assert isinstance(contract["SenderId"], list)
    # if the receiver is only one - use it
    if not isinstance(contract["ReceiverId"], list):
        return contract["ReceiverId"]

    # else we need to find the matching index
    idx = contract["SenderId"].index(agent_id)
    return contract["ReceiverId"][idx]


def interpolate_blocksizes(
    installed_power: float,
    block_size_in_mw: float,
    min_eff: float,
    max_eff: float,
    min_markup: float,
    max_markup: float,
):
    """
    This method interpolates efficiencies and markups for a given powerplant
    The fist powerplant receives the highest markup and lowest efficiency.
    The last one has lowest markup and highest efficiency.
    """
    full_blocks = int(installed_power // block_size_in_mw)
    block_sizes = [block_size_in_mw for i in range(full_blocks)]
    last_block = installed_power % block_size_in_mw
    if last_block > 0:
        block_sizes.append(last_block)

    # interpolate
    gradient_markup = (max_markup - min_markup) / (len(block_sizes) - 1)
    gradient_eff = (max_eff - min_eff) / (len(block_sizes) - 1)
    markups = []
    efficiencies = []
    for i, power in enumerate(block_sizes):
        # markup is high to low
        # efficiency is low to high
        markups.append(max_markup - i * gradient_markup)
        efficiencies.append(min_eff + i * gradient_eff)
    return list(zip(block_sizes, markups, efficiencies))


def add_agent_to_world(
    agent: dict,
    world: World,
    prices: dict,
    contracts: list,
    base_path: str,
    markups: dict = {},
):
    match agent["Type"]:
        case "EnergyExchange" | "DayAheadMarketSingleZone":
            market_config = MarketConfig(
                name=f"Market_{agent['Id']}",
                opening_hours=rr.rrule(
                    rr.HOURLY, interval=1, dtstart=world.start, until=world.end
                ),
                opening_duration=timedelta(hours=1),
                market_mechanism=translate_clearing[
                    agent["Attributes"]["DistributionMethod"]
                ],
                market_products=[
                    MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))
                ],
                maximum_bid_volume=1e6,
            )
            world.add_market_operator(f"Market_{agent['Id']}")
            world.add_market(f"Market_{agent['Id']}", market_config)
        case "CarbonMarket":
            co2_price = agent["Attributes"]["Co2Prices"]
            if isinstance(co2_price, str):
                price_series = read_csv(base_path, co2_price)
                co2_price = price_series.reindex(world.index).ffill().fillna(0)
            prices["co2"] = co2_price
        case "FuelsMarket":
            # fill prices for forecaster
            for fuel in agent["Attributes"]["FuelPrices"]:
                fuel_type = translate_fuel_type[fuel["FuelType"]]
                price = fuel["Price"]
                if isinstance(fuel["Price"], str):
                    price_series = read_csv(base_path, fuel["Price"])
                    price_series.index = price_series.index.round("h")
                    if not price_series.index.is_unique:
                        price_series = price_series.groupby(level=0).last()
                    price = price_series.reindex(world.index).ffill()
                prices[fuel_type] = price * fuel["ConversionFactor"]
        case "DemandTrader":
            world.add_unit_operator(agent["Id"])

            for i, load in enumerate(agent["Attributes"]["Loads"]):
                demand_series = -read_csv(base_path, load["DemandSeries"])
                world.add_unit(
                    f"demand_{agent['Id']}_{i}",
                    "demand",
                    agent["Id"],
                    {
                        "min_power": 0,
                        "max_power": 100000,
                        "bidding_strategies": {"energy": "naive"},
                        "technology": "demand",
                        "price": load["ValueOfLostLoad"],
                    },
                    NaiveForecast(world.index, demand=demand_series),
                )

        case "StorageTrader":
            operator_id = f"Operator_{agent['Id']}"
            world.add_unit_operator(operator_id)
            device = agent["Attributes"]["Device"]
            strategy = agent["Attributes"]["Strategy"]
            if strategy["StrategistType"] != "SINGLE_AGENT_MIN_SYSTEM_COST":
                logger.warning(f"unknown strategy for storage trader: {strategy}")

            forecast_price = prices.get("co2", 20)
            # TODO forecast should be calculated using calculate_EOM_price_forecast
            forecast = NaiveForecast(
                world.index,
                availability=1,
                co2_price=prices.get("co2", 2),
                # price_forecast is used for price_EOM
                price_forecast=forecast_price,
            )

            max_volume = device["EnergyToPowerRatio"] * device["InstalledPowerInMW"]
            initial_soc = 100 * device["InitialEnergyLevelInMWH"] / max_volume
            # TODO device["SelfDischargeRatePerHour"]
            world.add_unit(
                f"StorageTrader_{agent['Id']}",
                "storage",
                operator_id,
                {
                    "max_power_charge": device["InstalledPowerInMW"],
                    "max_power_discharge": device["InstalledPowerInMW"],
                    "efficiency_charge": device["ChargingEfficiency"],
                    "efficiency_discharge": device["DischargingEfficiency"],
                    "initial_soc": initial_soc,
                    "max_volume": max_volume,
                    "bidding_strategies": {"energy": "flexable_eom_storage"},
                    "technology": "hydro",  # PSPP? Pump-Storage Power Plant
                    "emission_factor": 0,
                },
                forecast,
            )

        case "RenewableTrader":
            world.add_unit_operator(f"Operator_{agent['Id']}")
            # send, receive = get_send_receive_msgs_per_id(agent["Id"], contracts)
        case "NoSupportTrader":
            # does not get support - just trades renewables
            # has a ShareOfRevenues (how much of the profit he keeps)
            world.add_unit_operator(f"Operator_{agent['Id']}")
        case "SystemOperatorTrader":
            world.add_unit_operator(f"Operator_{agent['Id']}")

        case "ConventionalPlantOperator":
            world.add_unit_operator(f"Operator_{agent['Id']}")
        case "ConventionalTrader":
            world.add_unit_operator(f"Operator_{agent['Id']}")
            min_markup = agent["Attributes"]["minMarkup"]
            max_markup = agent["Attributes"]["maxMarkup"]
            markups[agent["Id"]] = (min_markup, max_markup)
        case "PredefinedPlantBuilder":
            # this is the actual powerplant
            prototype = agent["Attributes"]["Prototype"]
            attr = agent["Attributes"]
            send, receive = get_send_receive_msgs_per_id(agent["Id"], contracts)
            raw_operator_id = get_matching_send_one_or_multi(agent["Id"], send[0])
            send_t, receive_t = get_send_receive_msgs_per_id(raw_operator_id, contracts)
            raw_trader_id = get_matching_send_one_or_multi(raw_operator_id, send_t[2])
            operator_id = f"Operator_{raw_operator_id}"
            fuel_price = prices.get(translate_fuel_type[prototype["FuelType"]], 0)
            fuel_price += prototype.get("OpexVarInEURperMWH", 0)
            # TODO CyclingCostInEURperMW
            # costs due to plant start up
            forecast = NaiveForecast(
                world.index,
                availability=prototype["PlannedAvailability"],
                fuel_price=fuel_price,
                co2_price=prices.get("co2", 2),
            )
            # TODO UnplannedAvailabilityFactor is not respected

            min_markup, max_markup = markups.get(raw_trader_id, (0, 0))
            # Amiris interpolates blocks linearly
            interpolated_values = interpolate_blocksizes(
                attr["InstalledPowerInMW"],
                attr["BlockSizeInMW"],
                attr["Efficiency"]["Minimal"],
                attr["Efficiency"]["Maximal"],
                min_markup,
                max_markup,
            )
            # add a unit for each block
            for i, values in enumerate(interpolated_values):
                power, markup, efficiency = values
                world.add_unit(
                    f"PredefinedPlantBuilder_{agent['Id']}_{i}",
                    "power_plant",
                    operator_id,
                    {
                        # AMIRIS does not have min_power
                        "min_power": 0,
                        "max_power": power,
                        "fixed_cost": markup,
                        "bidding_strategies": {"energy": "naive"},
                        "technology": translate_fuel_type[prototype["FuelType"]],
                        "fuel_type": translate_fuel_type[prototype["FuelType"]],
                        "emission_factor": prototype["SpecificCo2EmissionsInTperMWH"],
                        "efficiency": efficiency,
                    },
                    forecast,
                )
        case "VariableRenewableOperator" | "Biogas":
            send, receive = get_send_receive_msgs_per_id(agent["Id"], contracts)
            operator_id = get_matching_send_one_or_multi(agent["Id"], send[0])
            operator_id = f"Operator_{operator_id}"
            attr = agent["Attributes"]
            availability = attr.get("YieldProfile", attr.get("DispatchTimeSeries"))
            max_power = attr["InstalledPowerInMW"]
            if isinstance(availability, str):
                dispatch_profile = read_csv(base_path, availability)
                availability = dispatch_profile.reindex(world.index).ffill().fillna(0)

                if availability.max() > 1:
                    scale_value = availability.max()
                    availability /= scale_value
                    max_power *= scale_value
                    # availability above 1 does not make sense

            fuel_price = prices.get(translate_fuel_type[attr["EnergyCarrier"]], 0)
            fuel_price += attr.get("OpexVarInEURperMWH", 0)
            forecast = NaiveForecast(
                world.index,
                availability=availability,
                fuel_price=fuel_price,
                co2_price=prices.get("co2", 0),
            )
            # TODO attr["SupportInstrument"] and
            world.add_unit(
                f"VariableRenewableOperator_{agent['Id']}",
                "power_plant",
                operator_id,
                {
                    "min_power": 0,
                    "max_power": max_power,
                    "bidding_strategies": {"energy": "naive"},
                    "technology": translate_fuel_type[attr["EnergyCarrier"]],
                    "fuel_type": translate_fuel_type[attr["EnergyCarrier"]],
                    "emission_factor": 0,
                    "efficiency": 1,
                },
                forecast,
            )


def read_amiris_yaml(base_path):
    YamlIncludeConstructor.add_to_loader_class(
        loader_class=yaml.FullLoader, base_dir=base_path
    )

    with open(base_path + "/scenario.yaml", "rb") as f:
        amiris_scenario = yaml.load(f, Loader=yaml.FullLoader)
    return amiris_scenario


async def load_amiris_async(
    world: World,
    scenario: str,
    study_case: str,
    base_path: str,
):
    amiris_scenario = read_amiris_yaml(base_path)
    # DeliveryIntervalInSteps = 3600
    # In practice - this seems to be a fixed number in AMIRIS
    if study_case.lower() == "simple":
        print("is simple - adjusting start time")
        amiris_scenario["GeneralProperties"]["Simulation"][
            "StartTime"
        ] = "2020-12-31_23:58:00"

    start = amiris_scenario["GeneralProperties"]["Simulation"]["StartTime"]
    start = pd.to_datetime(start, format="%Y-%m-%d_%H:%M:%S")
    end = amiris_scenario["GeneralProperties"]["Simulation"]["StopTime"]
    end = pd.to_datetime(end, format="%Y-%m-%d_%H:%M:%S")
    # AMIRIS caveat: start and end is always two minutes before actual start
    start += timedelta(minutes=2)
    sim_id = f"{scenario}_{study_case}"
    save_interval = amiris_scenario["GeneralProperties"]["Output"]["Interval"] // 2
    prices = {}
    index = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_interval,
        simulation_id=sim_id,
        index=index,
    )
    # helper dict to map trader markups/markdowns to powerplants
    markups = {}
    keyorder = [
        "EnergyExchange",
        "DayAheadMarketSingleZone",
        "CarbonMarket",
        "FuelsMarket",
        "DemandTrader",
        "StorageTrader",
        "RenewableTrader",
        "NoSupportTrader",
        "ConventionalTrader",
        "SystemOperatorTrader",
        "ConventionalPlantOperator",
        "PredefinedPlantBuilder",
        "VariableRenewableOperator",
        "Biogas",
        "MeritOrderForecaster",
        "SupportPolicy",
    ]
    agents_sorted = sorted(
        amiris_scenario["Agents"], key=lambda agent: keyorder.index((agent["Type"]))
    )
    for agent in agents_sorted:
        add_agent_to_world(
            agent,
            world,
            prices,
            amiris_scenario["Contracts"],
            base_path,
            markups,
        )
    # calculate market price before simulation
    world


if __name__ == "__main__":
    # To use this with amiris run:
    # git clone https://gitlab.com/dlr-ve/esy/amiris/examples.git amiris-examples
    # next to the assume folder
    scenario = "Simple"  # Germany2019 or Austria2019 or Simple
    base_path = f"../amiris-examples/{scenario}/"
    amiris_scenario = read_amiris_yaml(base_path)
    sends, receives = get_send_receive_msgs_per_id(
        1000,
        amiris_scenario["Contracts"],
    )

    demand_agent = amiris_scenario["Agents"][3]
    demand_series = read_csv(
        base_path, demand_agent["Attributes"]["Loads"][0]["DemandSeries"]
    )

    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    world.loop.run_until_complete(
        load_amiris_async(
            world,
            "amiris",
            scenario,
            base_path,
        )
    )
    print(f"did load {scenario} - now simulating")
    world.run()
