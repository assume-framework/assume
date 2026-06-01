# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from assume.common.exceptions import AssumeException
from assume.scenario.entsoe_helper.mappings import PSR_TO_ASSUME

logger = logging.getLogger(__name__)


def _require_entsoe_client():
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise AssumeException(
            "entsoe-py is required for the ENTSO-E loader. "
            "Install it with: pip install 'assume-framework[entsoe]'"
        ) from exc
    return EntsoePandasClient


def _flatten_columns(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if not isinstance(data, pd.DataFrame):
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.map(str)
    return data


class EntsoeInterface:
    """Fetch country-level ENTSO-E load, generation and capacity data."""

    def __init__(self, api_key: str, cache_dir: Path | None = None):
        EntsoePandasClient = _require_entsoe_client()
        self.client = EntsoePandasClient(api_key=api_key)
        self.cache_dir = cache_dir or Path.home() / ".assume" / "entsoe"

    def _cache_path(
        self,
        country: str,
        start: datetime,
        end: datetime,
        dataset: str,
    ) -> Path:
        country = country.upper()
        if dataset == "capacity":
            return self.cache_dir / f"{country}_{start.year}" / f"{dataset}.csv"
        period = f"{start:%Y%m%d}_{end:%Y%m%d}"
        return self.cache_dir / f"{country}_{period}" / f"{dataset}.csv"

    def _read_cache(
        self, path: Path, parse_dates: bool = True
    ) -> pd.Series | pd.DataFrame | None:
        if not path.is_file():
            return None
        logger.info(f"using cached ENTSO-E data from {path}")
        data = pd.read_csv(path, index_col=0, parse_dates=parse_dates)
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            return data.squeeze()
        return data

    def _write_cache(self, path: Path, data: pd.Series | pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path)

    @staticmethod
    def _ensure_unique_index(
        data: pd.Series | pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        if data.index.is_unique:
            return data
        return data.groupby(level=0).mean()

    @staticmethod
    def _to_naive_index(
        data: pd.Series | pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)
        return data

    @staticmethod
    def _slice_to_period(
        data: pd.Series | pd.DataFrame,
        start: datetime,
        end: datetime,
    ) -> pd.Series | pd.DataFrame:
        return data.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    @staticmethod
    def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def get_country_demand(
        self,
        start: datetime,
        end: datetime,
        country: str,
        use_cache: bool = True,
    ) -> pd.Series:
        country = country.upper()
        cache_path = self._cache_path(country, start, end, "demand")
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return self._slice_to_period(
                    self._ensure_unique_index(
                        self._to_naive_index(cached)
                    ),
                    start,
                    end,
                )

        logger.info(f"querying ENTSO-E load for {country}")
        start_ts = self._to_utc_timestamp(start)
        end_ts = self._to_utc_timestamp(end)
        load = _flatten_columns(
            self.client.query_load(country, start=start_ts, end=end_ts)
        )
        demand = load["Actual Load"].resample("h").mean()
        demand = self._ensure_unique_index(self._to_naive_index(demand))
        if use_cache:
            self._write_cache(cache_path, demand)
        return self._slice_to_period(demand, start, end)

    def get_country_generation(
        self,
        start: datetime,
        end: datetime,
        country: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        country = country.upper()
        cache_path = self._cache_path(country, start, end, "generation")
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return self._slice_to_period(
                    self._ensure_unique_index(
                        self._to_naive_index(cached)
                    ),
                    start,
                    end,
                )

        logger.info(f"querying ENTSO-E generation for {country}")
        start_ts = self._to_utc_timestamp(start)
        end_ts = self._to_utc_timestamp(end)
        generation = self.client.query_generation(
            country, start=start_ts, end=end_ts, nett=True
        )
        if isinstance(generation, pd.Series):
            generation = generation.to_frame()
        generation = _flatten_columns(generation)
        generation = generation.resample("h").mean()
        generation = self._ensure_unique_index(self._to_naive_index(generation))
        if use_cache:
            self._write_cache(cache_path, generation)
        return self._slice_to_period(generation, start, end)

    def get_installed_capacity(
        self,
        start: datetime,
        end: datetime,
        country: str,
        use_cache: bool = True,
    ) -> pd.Series:
        country = country.upper()
        cache_path = self._cache_path(country, start, end, "capacity")
        if use_cache:
            cached = self._read_cache(cache_path, parse_dates=False)
            if cached is not None:
                return cached

        logger.info(f"querying ENTSO-E installed capacity for {country}")
        start_ts = self._to_utc_timestamp(start)
        end_ts = self._to_utc_timestamp(end)
        capacity = self.client.query_installed_generation_capacity(
            country, start=start_ts, end=end_ts
        )
        if isinstance(capacity, pd.Series):
            capacity = capacity.to_frame().T
        capacity = _flatten_columns(capacity)
        if capacity.empty:
            raise AssumeException(
                f"No installed capacity data returned for {country} in {start.year}"
            )

        index_tz = capacity.index.tz
        target = pd.Timestamp(start.year, 1, 1, tz=index_tz)
        if target not in capacity.index:
            capacity_row = capacity.iloc[-1]
        else:
            capacity_row = capacity.loc[target]

        if isinstance(capacity_row, pd.DataFrame):
            capacity_row = capacity_row.iloc[0]

        capacity_row = capacity_row.fillna(0)
        capacity_row.index = capacity_row.index.map(str)
        if use_cache:
            self._write_cache(cache_path, capacity_row)
        return capacity_row

    @staticmethod
    def aggregate_by_technology(
        capacity: pd.Series,
        generation: pd.DataFrame,
    ) -> dict[str, dict]:
        capacity = capacity.fillna(0)
        capacity.index = capacity.index.map(str)
        generation = generation.fillna(0)
        generation.columns = generation.columns.map(str)

        aggregated: dict[str, dict] = {}
        all_psr_types = set(capacity.index) | set(generation.columns)

        for psr_name in sorted(all_psr_types):
            mapping = PSR_TO_ASSUME.get(psr_name)
            if mapping is None:
                raise AssumeException(
                    f"Unmapped ENTSO-E production type '{psr_name}'. "
                    "Extend PSR_TO_ASSUME in "
                    "assume/scenario/entsoe_helper/mappings.py"
                )

            cap_mw = float(capacity.get(psr_name, 0.0))
            gen_series = (
                generation[psr_name]
                if psr_name in generation.columns
                else pd.Series(0.0, index=generation.index)
            )
            peak_gen = float(gen_series.max())

            if cap_mw <= 0 and peak_gen > 0:
                cap_mw = peak_gen
                logger.info(
                    f"using peak generation {cap_mw:.1f} MW as capacity for {psr_name}"
                )

            if cap_mw <= 0 and peak_gen <= 0:
                continue

            tech = mapping.technology
            if tech not in aggregated:
                aggregated[tech] = {
                    "mapping": mapping,
                    "capacity_mw": cap_mw,
                    "generation_mw": gen_series,
                }
            else:
                aggregated[tech]["capacity_mw"] += cap_mw
                aggregated[tech]["generation_mw"] = (
                    aggregated[tech]["generation_mw"] + gen_series
                )

        return aggregated
