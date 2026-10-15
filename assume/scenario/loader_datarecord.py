# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pandas as pd

from assume.common.forecaster import (
    DemandForecaster,
    PowerplantForecaster,
    UnitForecaster,
)
from assume.common.market_objects import MarketConfig

if TYPE_CHECKING:
    from assume.world import World

logger = logging.getLogger(__name__)


def _extract_connection_map(record: Any) -> dict[str, str]:
    """Map each entity to its connected bus/node."""
    conn_frame = record.groups.get("connection")
    if conn_frame is None:
        return {}
    df = conn_frame.collect().to_pandas()
    if "entity" in df.columns and "bus" in df.columns:
        return dict(zip(df["entity"].astype(str), df["bus"].astype(str), strict=False))
    return {}


def _build_attribute_map(record: Any, attr_name: str) -> dict[str, pd.Series]:
    """Map entity names to their time series for `attr_name` in one pass."""
    frame = record.attributes.get(attr_name)
    if frame is None:
        return {}
    df = frame.collect().to_pandas()
    if "entity" not in df.columns or "value" not in df.columns:
        return {}
    time_col = "snapshot" if "snapshot" in df.columns else "timestep"
    if time_col not in df.columns:
        return {}
    return {
        str(entity): sub.set_index(time_col)["value"]
        for entity, sub in df.groupby("entity")
    }


def _extract_attribute_series(
    record: Any, attr_name: str, entity_name: str
) -> pd.Series | None:
    """Extract a time series for a given attribute and entity."""
    return _build_attribute_map(record, attr_name).get(entity_name)


def load_datarecord(
    world: World,
    record: Any,
    scenario: str = "datarecord_scenario",
    study_case: str = "base",
    marketdesign: list[MarketConfig] | None = None,
    bidding_strategies: dict[str, dict[str, Any]] | None = None,
    save_frequency_hours: int = 4,
) -> None:
    """Initialize an ASSUME scenario from a datarecord.

    Parameters
    ----------
    world : World
        The ASSUME World instance to configure.
    record : RecordLike
        The resolved datarecord.
    scenario : str, default "datarecord_scenario"
        Name of the scenario.
    study_case : str, default "base"
        Name of the study case.
    marketdesign : list of MarketConfig, optional
        Market configurations to attach to the market operator.
    bidding_strategies : dict, optional
        Bidding strategy configurations for units.
    save_frequency_hours : int, default 4
        Output persistence interval in hours.
    """
    if isinstance(record, str) and not hasattr(record, "dims"):
        scenario_str = record
        study_case_str = str(scenario)
        actual_record = study_case
        scenario = scenario_str
        study_case = study_case_str
        record = actual_record

    time_frame = record.dims.get("snapshot") or record.dims.get("timestep")
    if time_frame is None:
        msg = "record must declare a 'snapshot' or 'timestep' dimension for ASSUME"
        raise ValueError(msg)

    time_df = time_frame.collect().to_pandas()
    time_col = time_df.columns[0]
    index = pd.DatetimeIndex(time_df[time_col])
    start = index[0]
    end = index[-1]
    simulation_id = f"{scenario}_{study_case}"
    logger.info("loading scenario %s", simulation_id)

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=simulation_id,
    )

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    if marketdesign is None:
        from datetime import timedelta

        from dateutil import rrule as rr

        from assume.common.market_objects import MarketProduct

        marketdesign = [
            MarketConfig(
                "EOM",
                rr.rrule(rr.HOURLY, interval=1, dtstart=start, until=end),
                timedelta(hours=1),
                "pay_as_clear",
                [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
                additional_fields=["node", "max_power", "min_power"],
            )
        ]

    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    if bidding_strategies is None:
        default_powerplant = {
            mc.market_id: "powerplant_energy_naive" for mc in marketdesign
        }
        default_demand = {mc.market_id: "demand_energy_naive" for mc in marketdesign}
        default_storage = {
            mc.market_id: "storage_energy_heuristic_flexable" for mc in marketdesign
        }
        strategies = {
            "power_plant": defaultdict(lambda: default_powerplant),
            "demand": defaultdict(lambda: default_demand),
            "storage": defaultdict(lambda: default_storage),
        }
    else:
        strategies = bidding_strategies

    connections = _extract_connection_map(record)
    p_max_map = _build_attribute_map(record, "p_max_pu")
    mc_map = _build_attribute_map(record, "marginal_cost")
    p_set_map = _build_attribute_map(record, "p_set")

    # Generators / Power Plants
    gen_type = "power_plant" if "power_plant" in record.entity_types else "Generator"
    if gen_type in record.entity_types:
        world.add_unit_operator("powerplant_operator")
        generators_df = record.entity_types[gen_type].collect().to_pandas()
        for _, gen in generators_df.iterrows():
            name = str(gen.get("entity", gen.name))
            node = connections.get(name, str(gen.get("bus", "node_0")))
            max_power = float(gen.get("p_nom", gen.get("max_power", 1000.0)))
            carrier = str(gen.get("carrier", gen.get("fuel_type", "gas")))
            min_power = float(gen.get("min_power", 0.0))
            efficiency = float(gen.get("efficiency", 1.0))
            ramp_up = float(
                gen.get("ramp_limit_start_up", gen.get("ramp_up", max_power))
            )
            ramp_down = float(
                gen.get("ramp_limit_shut_down", gen.get("ramp_down", max_power))
            )
            min_up = int(gen.get("min_up_time", gen.get("min_operating_time", 1)))
            min_down = int(gen.get("min_down_time", 1))

            av_series = p_max_map.get(name)
            availability = av_series if av_series is not None else 1.0
            mc_series = mc_map.get(name)
            mc_value = float(gen.get("marginal_cost", 50.0))
            fuel_prices = (
                {carrier: mc_series} if mc_series is not None else {carrier: mc_value}
            )

            world.add_unit(
                name,
                "power_plant",
                "powerplant_operator",
                {
                    "min_power": min_power,
                    "max_power": max_power,
                    "bidding_strategies": strategies["power_plant"][name],
                    "technology": str(gen.get("technology", "conventional")),
                    "node": node,
                    "efficiency": efficiency,
                    "fuel_type": carrier,
                    "ramp_up": ramp_up,
                    "ramp_down": ramp_down,
                    "min_operating_time": min_up,
                    "min_down_time": min_down,
                },
                PowerplantForecaster(
                    index,
                    fuel_prices=fuel_prices,
                    availability=availability,
                ),
            )

    # Loads / Demands
    load_type = "demand" if "demand" in record.entity_types else "Load"
    if load_type in record.entity_types:
        world.add_unit_operator("demand_operator")
        loads_df = record.entity_types[load_type].collect().to_pandas()
        for _, load in loads_df.iterrows():
            name = str(load.get("entity", load.name))
            node = connections.get(name, str(load.get("bus", "node_0")))
            load_series = p_set_map.get(name)
            if load_series is None:
                continue
            abs_max = float(abs(load_series).max())
            world.add_unit(
                name,
                "demand",
                "demand_operator",
                {
                    "min_power": 0.0,
                    "max_power": -abs_max,
                    "bidding_strategies": strategies["demand"][name],
                    "technology": "demand",
                    "node": node,
                    "price": float(load.get("price", 1000.0)),
                },
                DemandForecaster(
                    index,
                    demand=-abs(load_series),
                ),
            )

    # Storage Units
    storage_type = "storage" if "storage" in record.entity_types else "StorageUnit"
    if storage_type in record.entity_types:
        world.add_unit_operator("storage_operator")
        storages_df = record.entity_types[storage_type].collect().to_pandas()
        for _, storage in storages_df.iterrows():
            name = str(storage.get("entity", storage.name))
            node = connections.get(name, str(storage.get("bus", "node_0")))
            p_nom = float(
                storage.get("p_nom", storage.get("max_power_discharge", 100.0))
            )
            max_hours = float(storage.get("max_hours", 4.0))
            capacity = float(storage.get("capacity", p_nom * max_hours))
            charge_power = float(storage.get("max_power_charge", p_nom))
            discharge_power = float(storage.get("max_power_discharge", p_nom))
            world.add_unit(
                name,
                "storage",
                "storage_operator",
                {
                    "max_power_charge": -abs(charge_power),
                    "max_power_discharge": abs(discharge_power),
                    "efficiency_charge": float(
                        storage.get(
                            "efficiency_store", storage.get("efficiency_charge", 0.9)
                        )
                    ),
                    "efficiency_discharge": float(
                        storage.get(
                            "efficiency_dispatch",
                            storage.get("efficiency_discharge", 0.9),
                        )
                    ),
                    "initial_soc": float(
                        storage.get(
                            "state_of_charge_initial", storage.get("initial_soc", 0.5)
                        )
                    ),
                    "capacity": capacity,
                    "bidding_strategies": strategies["storage"][name],
                    "technology": str(
                        storage.get("carrier", storage.get("technology", "battery"))
                    ),
                    "node": node,
                },
                UnitForecaster(index),
            )

    world.init_forecasts()
