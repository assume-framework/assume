# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from assume.units import BaseUnit

if TYPE_CHECKING:
    from assume.world import World

logger = logging.getLogger(__name__)


def export_to_folder(
    world: "World",
    scenario_save_path: str | Path = "scenario_exports",
    study_case: str = "base",
) -> None:
    """
    Export the current world setup to a CSV-based scenario folder.

    Creates a scenario folder with config.yaml and CSV files compatible with
    the CSV loader (loader_csv.py). The exported scenario can be adjusted, loaded and
    re-run using load_scenario_folder().

    Args:
        world: The World instance to export.
        scenario_save_path: Path where the scenario folder will be created.
            Defaults to "scenario_exports", which creates a subfolder with simulation_id under scenario_exports/.
        study_case: Name of the study case to use in config.yaml. Defaults to "base"

    Raises:
        ValueError: If world.setup() has not been called or required attributes are missing.
    """
    _validate_export_preconditions(world)

    # Create path: {scenario_save_path}/{simulation_id}/
    scenario_path = Path(scenario_save_path) / world.simulation_id
    scenario_path.mkdir(parents=True, exist_ok=True)

    _export_config(world, scenario_path, study_case)
    _export_grid(world, scenario_path)
    _export_units(world, scenario_path)
    _export_time_series(world, scenario_path)

    logger.info(f"Scenario exported to {scenario_path.resolve()}")


def _validate_export_preconditions(world: "World") -> None:
    """Check that world has required data for export."""
    required_attrs = [
        "start",
        "end",
        "simulation_id",
        "markets",
        "units",
        "unit_operators",
    ]
    missing = [
        attr
        for attr in required_attrs
        if not hasattr(world, attr) or getattr(world, attr) is None
    ]

    if missing:
        raise ValueError(
            f"World is not properly set up for export. Missing attributes: {missing}. "
            "Please ensure world.setup() has been called before exporting."
        )


def _export_config(world: "World", scenario_path: Path, study_case: str) -> None:
    """Export configuration to config.yaml."""
    config = _build_config_dict(world, study_case)
    config_path = scenario_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {study_case: config},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _build_config_dict(world: "World", study_case: str) -> dict:
    """Build configuration dictionary for export."""
    config = {
        "start_date": world.start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": world.end.strftime("%Y-%m-%d %H:%M:%S"),
        "simulation_id": world.simulation_id,
    }

    # Only include time_step if we can infer it from units
    if time_step := _infer_time_step_or_none(world):
        config["time_step"] = time_step

    # Only include seed if it was explicitly set in scenario_data
    if "seed" in world.scenario_data.get("config", {}):
        seed_value = world.scenario_data["config"]["seed"]
        config["seed"] = (
            int(seed_value)
            if isinstance(seed_value, str) and seed_value.isdigit()
            else seed_value
        )

    # Only include save_frequency_hours if it was explicitly set in scenario_data
    if "save_frequency_hours" in world.scenario_data.get("config", {}):
        save_freq = world.scenario_data["config"]["save_frequency_hours"]
        config["save_frequency_hours"] = (
            int(save_freq)
            if isinstance(save_freq, str) and save_freq.isdigit()
            else save_freq
        )

    config["markets_config"] = _serialize_markets_config(world)

    # Only include bidding_strategy_params if it's not empty
    if world.bidding_params:
        config["bidding_strategy_params"] = world.bidding_params

    if world.learning_config:
        # Convert learning_config to a clean dictionary without internal attributes
        learning_dict = {}
        for key, value in world.learning_config.__dict__.items():
            if not key.startswith("_") and not callable(value):
                learning_dict[key] = value
        if learning_dict:
            config["learning_config"] = learning_dict

    return config


def _infer_time_step(world: "World") -> str:
    """Infer time step from unit forecasters as pandas frequency string, or default to '1h'."""
    if result := _infer_time_step_or_none(world):
        return result
    return "1h"


def _infer_time_step_or_none(world: "World") -> str | None:
    """Infer time step from unit forecasters as pandas frequency string, or return None if not inferable."""
    if world.units:
        first_unit = next(iter(world.units.values()))
        if hasattr(first_unit.forecaster, "index") and first_unit.forecaster.index:
            freq = first_unit.forecaster.index.freq
            if freq:
                # Convert timedelta to pandas frequency string
                if isinstance(freq, timedelta):
                    total_seconds = int(freq.total_seconds())
                    if total_seconds == 3600:
                        return "1h"
                    elif total_seconds == 1800:
                        return "30min"
                    elif total_seconds == 60:
                        return "1min"
                    else:
                        return (
                            f"{total_seconds // 3600}h"
                            if total_seconds % 3600 == 0
                            else f"{total_seconds // 60}min"
                        )
                return str(freq)
    return None


def _timedelta_to_frequency_str(td: timedelta) -> str:
    """Convert timedelta to pandas frequency string."""
    DURATION_FACTORS = [
        (86400, "d"),
        (3600, "h"),
        (60, "m"),
        (1, "s"),
    ]
    total_seconds = int(td.total_seconds())

    for factor, short_suffix in DURATION_FACTORS:
        if total_seconds % factor == 0:
            value = total_seconds // factor
            return f"{value}{short_suffix}"

    return str(td)


def _serialize_markets_config(world: "World") -> dict:
    """Serialize market configurations for export."""
    markets_config = {}
    for market_id, market_config in world.markets.items():
        market_dict = {}

        # Extract start and end from opening_hours if available
        if hasattr(market_config.opening_hours, "_dtstart"):
            market_dict["start_date"] = market_config.opening_hours._dtstart.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        if hasattr(market_config.opening_hours, "_until"):
            market_dict["end_date"] = market_config.opening_hours._until.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # Only include opening_frequency if we can determine it
        if opening_freq := _rrule_to_frequency_str(market_config.opening_hours):
            market_dict["opening_frequency"] = opening_freq

        # Always include these core fields
        market_dict["opening_duration"] = _timedelta_to_frequency_str(
            market_config.opening_duration
        )
        market_dict["market_mechanism"] = market_config.market_mechanism

        # Products
        if market_config.market_products:
            market_dict["products"] = [
                {
                    "duration": _timedelta_to_frequency_str(p.duration),
                    "count": p.count,
                    "first_delivery": _timedelta_to_frequency_str(p.first_delivery),
                }
                for p in market_config.market_products
            ]

        market_dict["product_type"] = market_config.product_type
        # Use values directly as they are already proper types (float | None)
        market_dict["maximum_bid_volume"] = market_config.maximum_bid_volume
        market_dict["maximum_bid_price"] = market_config.maximum_bid_price
        market_dict["minimum_bid_price"] = market_config.minimum_bid_price
        market_dict["volume_unit"] = market_config.volume_unit
        market_dict["price_unit"] = market_config.price_unit

        # Only include if not None or empty
        if market_config.additional_fields:
            market_dict["additional_fields"] = market_config.additional_fields

        # Ensure boolean is properly typed for YAML serialization
        supports_unmatched = market_config.supports_get_unmatched
        if isinstance(supports_unmatched, str):
            supports_unmatched = supports_unmatched.lower() == "true"
        market_dict["supports_get_unmatched"] = bool(supports_unmatched)

        # Only include param_dict if not empty
        if market_config.param_dict:
            # Handle grid_data separately - replace with network_path
            param_dict = {}
            has_grid_data = "grid_data" in market_config.param_dict
            for k, v in market_config.param_dict.items():
                if k == "grid_data":
                    continue  # Skip grid_data as it's handled by _export_grid

                # Convert to native Python types for clean YAML serialization
                if isinstance(v, bool):
                    param_dict[k] = v
                elif isinstance(v, (int, float)):
                    param_dict[k] = v
                else:
                    param_dict[k] = str(v) if v is not None else v

            # Add network_path if grid_data was present
            if has_grid_data and param_dict:
                param_dict["network_path"] = "."

            if param_dict:
                market_dict["param_dict"] = param_dict

        # Find operator for this market
        for op_id, op in world.market_operators.items():
            if market_config in op.markets:
                market_dict["operator"] = op_id
                break

        markets_config[market_id] = market_dict

    return markets_config


def _rrule_to_frequency_str(rrule) -> str | None:
    """Convert rrule to frequency string compatible with convert_to_rrule_freq, or return None."""
    freq = rrule._freq
    interval = rrule._interval

    if freq == 4:  # HOURLY
        return f"{interval}h"
    elif freq == 3:  # DAILY
        return f"{interval}d"
    elif freq == 2:  # WEEKLY
        return f"{interval}w"
    elif freq == 1:  # MONTHLY
        return f"{interval}m"
    elif freq == 0:  # YEARLY
        return f"{interval}y"
    else:
        return None


def _export_units(world: "World", scenario_path: Path) -> None:
    """Export all units to CSV files."""
    _export_powerplant_units(world, scenario_path)
    _export_demand_units(world, scenario_path)
    _export_storage_units(world, scenario_path)
    _export_exchange_units(world, scenario_path)
    _export_dsm_units(world, scenario_path)


def _export_powerplant_units(world: "World", scenario_path: Path) -> None:
    """Export power plant units to CSV."""
    powerplants = [u for u in world.units.values() if type(u).__name__ == "PowerPlant"]
    if not powerplants:
        return

    data = []
    for unit in powerplants:
        row = _unit_to_dict(world, unit)
        data.append(row)

    df = pd.DataFrame(data).set_index("name")
    if not df.empty:
        df.to_csv(scenario_path / "powerplant_units.csv", index=True)


def _export_demand_units(world: "World", scenario_path: Path) -> None:
    """Export demand units to CSV."""
    demands = [u for u in world.units.values() if type(u).__name__ == "Demand"]
    if not demands:
        return

    data = []
    for unit in demands:
        row = _unit_to_dict(world, unit)
        data.append(row)

    df = pd.DataFrame(data).set_index("name")
    if not df.empty:
        df.to_csv(scenario_path / "demand_units.csv", index=True)


def _export_storage_units(world: "World", scenario_path: Path) -> None:
    """Export storage units to CSV."""
    storages = [u for u in world.units.values() if type(u).__name__ == "Storage"]
    if not storages:
        return

    data = []
    for unit in storages:
        row = _unit_to_dict(world, unit)
        data.append(row)

    df = pd.DataFrame(data).set_index("name")
    if not df.empty:
        df.to_csv(scenario_path / "storage_units.csv", index=True)


def _export_exchange_units(world: "World", scenario_path: Path) -> None:
    """Export exchange units to CSV."""
    exchanges = [u for u in world.units.values() if type(u).__name__ == "Exchange"]
    if not exchanges:
        return

    data = []
    for unit in exchanges:
        row = _unit_to_dict(world, unit)
        data.append(row)

    df = pd.DataFrame(data).set_index("name")
    if not df.empty:
        df.to_csv(scenario_path / "exchange_units.csv", index=True)


def _export_dsm_units(world: "World", scenario_path: Path) -> None:
    """Export DSM units (Building, SteelPlant, etc.) to CSV."""
    building_units = [u for u in world.units.values() if type(u).__name__ == "Building"]
    steel_units = [u for u in world.units.values() if type(u).__name__ == "SteelPlant"]
    hydrogen_units = [
        u for u in world.units.values() if type(u).__name__ == "HydrogenPlant"
    ]
    steam_units = [u for u in world.units.values() if type(u).__name__ == "SteamPlant"]

    if building_units:
        _export_units_to_csv(
            world, building_units, scenario_path, "residential_dsm_units"
        )
    if steel_units:
        _export_units_to_csv(world, steel_units, scenario_path, "industrial_dsm_units")
    if hydrogen_units or steam_units:
        combined = hydrogen_units + steam_units
        _export_units_to_csv(world, combined, scenario_path, "industrial_dsm_units")


def _export_units_to_csv(
    world: "World", units: list, scenario_path: Path, filename: str
) -> None:
    """Helper to export a list of units to CSV."""
    data = []
    for unit in units:
        row = _unit_to_dict(world, unit)
        data.append(row)

    df = pd.DataFrame(data).set_index("name")
    if not df.empty:
        df.to_csv(scenario_path / f"{filename}.csv", index=True)


def _unit_to_dict(world: "World", unit: BaseUnit) -> dict:
    """Convert a unit to a dictionary for CSV export using dynamic attribute extraction."""
    # Start with as_dict() as base - gets standard attributes
    unit_dict = unit.as_dict()

    # Attributes to skip (internal/non-serializable or handled separately)
    SKIP_ATTRS = {
        "forecaster",
        "index",
        "outputs",
        "avg_op_time",
        "total_op_time",
        "bidding_strategies",
        "unit_type",
        "marginal_cost",
        "price",  # unit_type handled separately
    }

    # Dynamically add all missing attributes from unit and its bases
    for attr in dir(unit):
        if attr.startswith("_") or attr in SKIP_ATTRS:
            continue
        if attr not in unit_dict:
            try:
                value = getattr(unit, attr)
                # Skip callable attributes (methods) and None values
                if callable(value) or value is None:
                    continue
                # Format location specially as "lat,lng" string
                if attr == "location" and value:
                    unit_dict["location"] = f"{value[0]},{value[1]}"
                else:
                    unit_dict[attr] = value
            except (AttributeError, TypeError):
                # Skip attributes that can't be accessed
                pass

    # Add bidding strategies (handle separately for consistent formatting)
    unit_type_name = type(unit).__name__.lower()
    for market_id, strategy in unit.bidding_strategies.items():
        if not strategy:
            unit_dict[f"bidding_{market_id}"] = ""
            continue
        strategy_cls = strategy.__class__
        best_key = ""
        for key, cls in world.bidding_strategies.items():
            if cls == strategy_cls:
                if not best_key:
                    best_key = key
                    continue

                key_pref = key.startswith(f"{unit_type_name}_")
                best_pref = best_key.startswith(f"{unit_type_name}_")

                if key_pref and not best_pref:
                    best_key = key
                    continue
                if best_pref and not key_pref:
                    continue

                if key_pref == best_pref:
                    market_lower = market_id.lower()
                    key_has_market = market_lower in key
                    best_has_market = market_lower in best_key

                    if key_has_market and not best_has_market:
                        best_key = key
                        continue
                    if best_has_market and not key_has_market:
                        continue

                    if len(key) < len(best_key):
                        best_key = key
        unit_dict[f"bidding_{market_id}"] = (
            best_key if best_key else strategy_cls.__name__
        )

    # Add forecast algorithms
    if hasattr(unit.forecaster, "forecast_algorithms"):
        for key, value in unit.forecaster.forecast_algorithms.items():
            unit_dict[f"forecast_{key}"] = value

    # Rename id to name for CSV format
    unit_dict["name"] = unit_dict.pop("id")

    # Convert negative demand/charge powers back to positive for user-friendly export
    if unit_type_name == "demand":
        if "min_power" in unit_dict:
            unit_dict["min_power"] = abs(unit_dict["min_power"])
        if "max_power" in unit_dict:
            unit_dict["max_power"] = abs(unit_dict["max_power"])
    elif unit_type_name == "storage":
        if "max_power_charge" in unit_dict:
            unit_dict["max_power_charge"] = abs(unit_dict["max_power_charge"])
        if "min_power_charge" in unit_dict:
            unit_dict["min_power_charge"] = abs(unit_dict["min_power_charge"])

    return unit_dict


def _export_time_series(world: "World", scenario_path: Path) -> None:
    """Export time series data to CSV files."""
    _export_demand_df(world, scenario_path)
    _export_availability_df(world, scenario_path)
    _export_fuel_prices_df(world, scenario_path)
    _export_forecasts_df(world, scenario_path)
    _export_exchanges_df(world, scenario_path)


def _export_demand_df(world: "World", scenario_path: Path) -> None:
    """Export demand time series."""
    demand_units = [u for u in world.units.values() if type(u).__name__ == "Demand"]
    if not demand_units:
        return

    series_dict = {}
    for unit in demand_units:
        if hasattr(unit.forecaster, "demand"):
            # Use the forecaster's index as the datetime index
            series_dict[unit.id] = -unit.forecaster.demand

    if series_dict:
        df = pd.DataFrame(
            series_dict, index=unit.forecaster.index.as_datetimeindex()
        ).rename_axis("datetime")
        df.to_csv(scenario_path / "demand_df.csv", index=True)


def _export_availability_df(world: "World", scenario_path: Path) -> None:
    """Export availability time series."""
    series_dict = {}
    for unit in world.units.values():
        if hasattr(unit.forecaster, "availability"):
            series_dict[unit.id] = unit.forecaster.availability

    if series_dict:
        df = pd.DataFrame(
            series_dict, index=unit.forecaster.index.as_datetimeindex()
        ).rename_axis("datetime")
        df.to_csv(scenario_path / "availability_df.csv", index=True)


def _export_fuel_prices_df(world: "World", scenario_path: Path) -> None:
    """Export fuel prices time series."""
    all_fuel_prices = {}

    for unit in world.units.values():
        if hasattr(unit.forecaster, "fuel_prices") and unit.forecaster.fuel_prices:
            for fuel, series in unit.forecaster.fuel_prices.items():
                if fuel not in all_fuel_prices:
                    all_fuel_prices[fuel] = series

    if all_fuel_prices:
        df = pd.DataFrame(
            all_fuel_prices, index=unit.forecaster.index.as_datetimeindex()
        ).rename_axis("datetime")
        df.to_csv(scenario_path / "fuel_prices_df.csv", index=True)


def _export_forecasts_df(world: "World", scenario_path: Path) -> None:
    """Export the given forecasts (price, residual_load, ...) the scenario was loaded with.

    Forecasts that were calculated by a forecast algorithm are not exported: the
    algorithm is stored per unit instead (see ``_unit_to_dict``) and recomputes them
    on load.
    """
    forecasts_df = world.scenario_data.get("forecasts_df")
    if forecasts_df is None or forecasts_df.empty:
        return

    forecasts_df.rename_axis("datetime").to_csv(
        scenario_path / "forecasts_df.csv", index=True
    )


def _export_exchanges_df(world: "World", scenario_path: Path) -> None:
    """Export exchange volume time series."""
    exchange_units = [u for u in world.units.values() if type(u).__name__ == "Exchange"]
    if not exchange_units:
        return

    series_dict = {}
    for unit in exchange_units:
        forecaster = unit.forecaster
        if (
            hasattr(forecaster, "volume_export")
            and forecaster.volume_export is not None
        ):
            series_dict[f"{unit.id}_export"] = forecaster.volume_export
        if (
            hasattr(forecaster, "volume_import")
            and forecaster.volume_import is not None
        ):
            series_dict[f"{unit.id}_import"] = forecaster.volume_import

    if series_dict:
        df = pd.DataFrame(
            series_dict, index=forecaster.index.as_datetimeindex()
        ).rename_axis("datetime")
        df.to_csv(scenario_path / "exchanges_df.csv", index=True)


def _export_grid(world: "World", scenario_path: Path) -> None:
    """Export grid data (buses and lines) to CSV files if available in market configurations."""
    # Look for grid_data in market configurations
    grid_data = None
    for market_config in world.markets.values():
        if grid_data := market_config.param_dict.get("grid_data"):
            break

    if not grid_data:
        return

    # Export only buses and lines (generators/loads/storage are already exported via _export_units)
    grid_components = [("buses", "buses.csv"), ("lines", "lines.csv")]

    for component_name, filename in grid_components:
        if (
            component_df := grid_data.get(component_name)
        ) is not None and not component_df.empty:
            component_df.to_csv(scenario_path / filename, index=True)
