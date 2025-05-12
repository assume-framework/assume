# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
import logging
from datetime import timedelta

import dateutil.rrule as rr
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta as rd
from yaml_include import Constructor

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies.extended import SupportStrategy
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
    "Other": "other",
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


def get_send_receive_msgs_per_id(agent_id: int, contracts_config: list[dict]):
    """
    AMIRIS contract conversion function which finds the list of ids which receive or send a message from/to the agent with agent_id.

    Args:
        agent_id (int): the agent id to which the contracts are found
        contracts_config (list[dict]): whole contracts dict read from yaml

    Returns:
        tuple: A tuple containing the following:
            - list: A list containing the ids of sending agents.
            - list: A list containing the ids of receiving agents
    """

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
    supports: dict = {},
    index: pd.DatetimeIndex = None,
):
    """
    Adds an agent from a amiris agent definition to the ASSUME world.
    It should be called in load_amiris, which loads agents in the correct order.

    Args:
        agent (dict): AMIRIS agent dict
        world (World): ASSUME world to add the agent to
        prices (dict): prices read from amiris scenario beforehand
        contracts (list): contracts read from the amiris scenario beforehand
        base_path (str): base path to load profile csv files from
        markups (dict, optional): markups read from former agents. Defaults to {}.
    """
    strategies = {m: "flexable_eom" for m in list(world.markets.keys())}
    storage_strategies = {m: "flexable_eom_storage" for m in list(world.markets.keys())}
    demand_strategies = {m: "naive_eom" for m in list(world.markets.keys())}
    match agent["Type"]:
        case "SupportPolicy":
            support_data = agent["Attributes"]["SetSupportData"]
            supports |= {x.pop("PolicySet"): x for x in support_data}
            world.add_unit_operator(agent["Id"])

            for name, support in supports.items():
                contract = list(support.keys())[0]
                value = list(support[contract].values())[0]
                # TODO
                world.add_unit(
                    f"{name}_{agent['Id']}",
                    "demand",
                    agent["Id"],
                    {
                        "min_power": 0,
                        "max_power": 100000,
                        "bidding_strategies": {
                            "energy": "support",
                            "financial_support": "support",
                        },
                        "technology": "demand",
                        "price": value,
                    },
                    NaiveForecast(index, demand=100000),
                )
        case "EnergyExchange" | "DayAheadMarketSingleZone":
            clearing_section = agent["Attributes"].get("Clearing", agent["Attributes"])
            market_config = MarketConfig(
                market_id=f"Market_{agent['Id']}",
                opening_hours=rr.rrule(
                    rr.HOURLY, interval=24, dtstart=world.start, until=world.end
                ),
                opening_duration=timedelta(hours=1),
                market_mechanism=translate_clearing[
                    clearing_section["DistributionMethod"]
                ],
                market_products=[
                    MarketProduct(timedelta(hours=1), 24, timedelta(hours=0))
                ],
                maximum_bid_volume=1e6,
            )
            world.add_market_operator(f"Market_{agent['Id']}")
            world.add_market(f"Market_{agent['Id']}", market_config)

            if supports:
                support_config = MarketConfig(
                    name=f"SupportMarket_{agent['Id']}",
                    opening_hours=rr.rrule(
                        rr.YEARLY, dtstart=world.start, until=world.end
                    ),
                    opening_duration=timedelta(hours=1),
                    market_mechanism="pay_as_bid_contract",
                    market_products=[
                        MarketProduct(rd(months=12), 1, timedelta(hours=1))
                    ],
                    additional_fields=[
                        "sender_id",
                        "contract",  # one of MPVAR, MPFIX, CFD
                        "eligible_lambda",
                        "evaluation_frequency",  # monthly
                    ],
                    product_type="financial_support",
                    supports_get_unmatched=True,
                    maximum_bid_volume=1e6,
                )
                world.add_market_operator(f"SupportMarket_{agent['Id']}")
                world.add_market(f"SupportMarket_{agent['Id']}", support_config)
        case "CarbonMarket":
            co2_price = agent["Attributes"]["Co2Prices"]
            if isinstance(co2_price, str):
                price_series = read_csv(base_path, co2_price)
                co2_price = price_series.reindex(index).ffill().fillna(0)
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
                    price = price_series.reindex(index).ffill()
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
                        "bidding_strategies": demand_strategies,
                        "technology": "demand",
                        "price": load["ValueOfLostLoad"],
                    },
                    # demand_series might contain more values than index
                    NaiveForecast(index, demand=demand_series[: len(index)]),
                )

        case "StorageTrader":
            operator_id = f"Operator_{agent['Id']}"
            world.add_unit_operator(operator_id)
            device = agent["Attributes"]["Device"]
            strategy = agent["Attributes"]["Strategy"]
            if strategy["StrategistType"] not in [
                "SINGLE_AGENT_MIN_SYSTEM_COST",
                "SINGLE_AGENT_MAX_PROFIT",
            ]:
                logger.warning(f"unknown strategy for storage trader: {strategy}")

            forecast_price = prices.get("co2", 20)
            # TODO forecast should be calculated using calculate_EOM_price_forecast
            forecast = NaiveForecast(
                index,
                availability=1,
                co2_price=prices.get("co2", 2),
                # price_forecast is used for price_EOM
                price_forecast=forecast_price,
            )

            max_soc = device["EnergyToPowerRatio"] * device["InstalledPowerInMW"]
            initial_soc = device["InitialEnergyLevelInMWH"]
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
                    "max_soc": max_soc,
                    "bidding_strategies": storage_strategies,
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
            # can also have a ForecastError
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
            # this is the actual powerplant/PlantBuilder
            prototype = agent["Attributes"]["Prototype"]
            attr = agent["Attributes"]
            # first get send and receives for our PlantBuilder
            send, receive = get_send_receive_msgs_per_id(agent["Id"], contracts)
            # the first multi send includes message from us to our operator/portfolio
            raw_operator_id = get_matching_send_one_or_multi(agent["Id"], send[0])
            # we need to find send and receive for the raw operator too
            send_t, receive_t = get_send_receive_msgs_per_id(raw_operator_id, contracts)
            # the third entry here is the multi send to the actual trader
            raw_trader_id = get_matching_send_one_or_multi(raw_operator_id, send_t[2])
            operator_id = f"Operator_{raw_operator_id}"
            fuel_price = prices.get(translate_fuel_type[prototype["FuelType"]], 0)
            fuel_price += prototype.get("OpexVarInEURperMWH", 0)
            # TODO CyclingCostInEURperMW
            # costs due to plant start up
            availability = prototype["PlannedAvailability"]
            if isinstance(availability, str):
                availability = read_csv(base_path, availability)
                availability = availability.reindex(index).ffill()
            availability *= prototype.get("UnplannedAvailabilityFactor", 1)

            forecast = NaiveForecast(
                index,
                availability=availability,
                fuel_price=fuel_price,
                co2_price=prices.get("co2", 2),
            )
            # TODO UnplannedAvailabilityFactor is not respected

            # we get the markups from the trader id:
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
                        "additional_cost": markup,
                        "bidding_strategies": strategies,
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
                availability = dispatch_profile.reindex(index).ffill().fillna(0)

                if availability.max() > 1:
                    scale_value = availability.max()
                    availability /= scale_value
                    max_power *= scale_value
                    # availability above 1 does not make sense

            fuel_price = prices.get(translate_fuel_type[attr["EnergyCarrier"]], 0)
            fuel_price += attr.get("OpexVarInEURperMWH", 0)
            forecast = NaiveForecast(
                index,
                availability=availability,
                fuel_price=fuel_price,
                co2_price=prices.get("co2", 0),
            )
            support_instrument = attr.get("SupportInstrument")
            support_conf = supports.get(attr.get("Set"))
            bidding_params = {}
            if support_instrument and support_conf:
                for market in world.markets.keys():
                    if "SupportMarket" in market:
                        strategies[market] = "support"
                strategies["financial_support"] = "support"
                if support_instrument == "FIT":
                    conf_key = "TsFit"
                elif support_instrument in ["CFD", "MPVAR"]:
                    conf_key = "Lcoe"
                else:
                    conf_key = "Premium"
                value = support_conf[support_instrument][conf_key]
                bidding_params["contract_types"] = support_instrument
                bidding_params["support_value"] = value
                # ASSUME evaluates contracts on a monthly schedule
                bidding_params["evaluation_frequency"] = rr.MONTHLY

            world.add_unit(
                f"VariableRenewableOperator_{agent['Id']}",
                "power_plant",
                operator_id,
                {
                    "min_power": 0,
                    "max_power": max_power,
                    "bidding_strategies": strategies,
                    "technology": translate_fuel_type[attr["EnergyCarrier"]],
                    "fuel_type": translate_fuel_type[attr["EnergyCarrier"]],
                    "emission_factor": 0,
                    "efficiency": 1,
                    "bidding_params": bidding_params,
                },
                forecast,
            )


def read_amiris_yaml(base_path):
    yaml.add_constructor("!include", Constructor(base_dir=base_path))

    with open(base_path + "/scenario.yaml", "rb") as f:
        amiris_scenario = yaml.load(f, Loader=yaml.FullLoader)
    return amiris_scenario


def load_amiris(
    world: World,
    scenario: str,
    study_case: str,
    base_path: str,
):
    """
    Loads an Amiris scenario.
    Markups and markdowns are handled by linearly interpolating the agents volume.
    This mimics the behavior of the way it is done in AMIRIS.

    Args:
        world (World): the ASSUME world
        scenario (str): the scenario name
        study_case (str): study case to define
        base_path (str): base path from where to load the amrisi scenario
    """
    amiris_scenario = read_amiris_yaml(base_path)
    # DeliveryIntervalInSteps = 3600
    # In practice - this seems to be a fixed number in AMIRIS
    simulation = amiris_scenario["GeneralProperties"]["Simulation"]
    start = pd.to_datetime(simulation["StartTime"], format="%Y-%m-%d_%H:%M:%S")
    if calendar.isleap(start.year):
        # AMIRIS does not considerate leap years
        start += timedelta(days=1)
    end = pd.to_datetime(simulation["StopTime"], format="%Y-%m-%d_%H:%M:%S")
    # AMIRIS caveat: start and end is always two minutes before actual start
    start += timedelta(minutes=2)
    end += timedelta(minutes=2)
    simulation_id = f"{scenario}_{study_case}"
    prices = {}
    index = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    world.bidding_strategies["support"] = SupportStrategy
    world.setup(
        start=start,
        end=end,
        simulation_id=simulation_id,
    )
    # helper dict to map trader markups/markdowns to powerplants
    markups = {}
    supports = {}
    keyorder = [
        "EnergyExchange",
        "DayAheadMarketSingleZone",
        "CarbonMarket",
        "FuelsMarket",
        "SupportPolicy",
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
    ]
    agents_sorted = sorted(
        amiris_scenario["Agents"], key=lambda agent: keyorder.index(agent["Type"])
    )
    for agent in agents_sorted:
        add_agent_to_world(
            agent,
            world,
            prices,
            amiris_scenario["Contracts"],
            base_path,
            markups,
            supports,
            index,
        )
    # calculate market price before simulation
    world


if __name__ == "__main__":
    # To use this with amiris run:
    # git clone https://gitlab.com/dlr-ve/esy/amiris/examples.git amiris-examples
    # next to the assume folder
    scenario = "Germany2019"  # Germany2019 or Austria2019 or Simple
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
    load_amiris(
        world,
        "amiris",
        scenario.lower(),
        base_path,
    )
    logger.info(f"did load {scenario} - now simulating")
    world.run()
