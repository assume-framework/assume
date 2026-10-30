# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""What a scenario folder may contain, as far as it is not derived from the code itself."""

# keys of a study case in config.yaml that are read by the loader (besides file names)
REQUIRED_CASE_KEYS = ("start_date", "end_date", "time_step", "markets_config")
OPTIONAL_CASE_KEYS = {
    "learning_config": dict,
    "seed": int | None,
    "simulation_id": str,
    "save_frequency_hours": int | None,
    "bidding_strategy_params": dict,
    "forecast_algorithms": dict,
    "use_forecasts_df": bool,
    "save_forecasts": bool,
    "forecast_save_file": str,
}

# csv files that can be renamed (or disabled with `null`) via a key of the same name in the config
UNIT_FILES = {
    "powerplant_units": "power_plant",
    "storage_units": "storage",
    "demand_units": "demand",
    "exchange_units": "exchange",
}
DSM_FILES = ("industrial_dsm_units", "residential_dsm_units")
REQUIRED_UNIT_FILES = ("powerplant_units", "demand_units")
TIMESERIES_FILES = (
    "demand_df",
    "exchanges_df",
    "availability_df",
    "fuel_prices_df",
    "forecasts_df",
)
FILE_KEYS = (*UNIT_FILES, *DSM_FILES, "unit_operators", *TIMESERIES_FILES)

MARKET_KEYS = {
    "operator": str,
    "product_type": str,
    "products": list,
    "opening_frequency": str,
    "opening_duration": str,
    "market_mechanism": str,
    "start_date": None,
    "end_date": None,
    "maximum_bid_volume": float | None,
    "maximum_bid_price": float | None,
    "minimum_bid_price": float,
    "max_gradient": float | None,
    "volume_unit": str,
    "volume_tick": float | None,
    "price_unit": str,
    "price_tick": float | None,
    "additional_fields": list,
    "supports_get_unmatched": bool,
    "param_dict": dict,
}
REQUIRED_MARKET_KEYS = (
    "operator",
    "products",
    "opening_frequency",
    "opening_duration",
    "market_mechanism",
)
PRODUCT_KEYS = ("duration", "count", "first_delivery")
