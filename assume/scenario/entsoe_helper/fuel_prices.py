# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import io
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from assume.scenario.entsoe_helper.mappings import (
    DEFAULT_CO2_PRICE_EUR_T,
    DEFAULT_COAL_PRICE_EUR_MWH,
    DEFAULT_GAS_PRICE_EUR_MWH,
)

logger = logging.getLogger(__name__)

EU_ETS_URL = "https://energy-api.instrat.pl/api/prices/co2"
COAL_URL = "https://energy-api.instrat.pl/api/coal/pscmi_1"
GAS_URL = "https://energy-api.instrat.pl/api/prices/gas_price_rdn_daily"
USER_AGENT = "Mozilla/5.0 (compatible; ASSUME/1.0; +https://assume-project.de/)"

GJ_TO_KWH = 1e6 / 3600

# Fallback prices (EUR/MWh thermal, EUR/tCO2 for CO2) used when instrat.pl
# returns no usable data, so loading does not fail hard on data gaps. Sourced
# from Kost et al., Fraunhofer ISE (2024), see mappings.py.
_COAL_FALLBACK_EUR_MWH = DEFAULT_COAL_PRICE_EUR_MWH
_GAS_FALLBACK_EUR_MWH = DEFAULT_GAS_PRICE_EUR_MWH
_CO2_FALLBACK_EUR_T = DEFAULT_CO2_PRICE_EUR_T

# Mean 2024 PLNEUR=X rate on yfinance (0.2319, about 4.3 PLN per EUR), matching
# the 2024 basis of the fallback prices; used only when yfinance returns no FX
# data at all (e.g. no network access).
_PLN_EUR_FALLBACK = 0.232

# How far to query before the simulation start for each instrat.pl series.
# Coal is only published monthly, so a simulation window shorter than one
# month must look back over a full year to still find a usable observation.
# Gas is published daily and CO2 on trading days (gaps of up to 6 days in
# 2023), so one week always contains an observation.
INSTRAT_LOOKBACK_DAYS = {"coal": 366, "gas": 7, "co2": 7}


def _sanitize_daily_prices(
    series: pd.Series, name: str, fallback: float | None = None
) -> pd.Series:
    """Drop invalid values and ensure the series is usable for reindexing."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if not numeric.empty:
        numeric.name = name
        return numeric

    fill = fallback if fallback is not None else _COAL_FALLBACK_EUR_MWH
    logger.warning(
        "No valid %s prices returned from instrat.pl; using fallback %.1f €/MWh",
        name,
        fill,
    )
    anchor = series.index[0] if len(series.index) else pd.Timestamp("2024-01-01")
    return pd.Series(fill, index=[anchor], name=name)


def _to_hourly_prices(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Expand daily prices to the simulation index without leading NaNs."""
    daily = series.resample("D").ffill().bfill()
    hourly = daily.reindex(index, method="ffill").bfill().ffill()
    if hourly.isna().any():
        fill_value = hourly.dropna().iloc[0]
        hourly = hourly.fillna(fill_value)
    return hourly


class InstratFuelPrices:
    """Fetch coal, gas and EU ETS prices from energy.instrat.pl."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".assume" / "instrat_pl"

    def _cache_path(self, start: datetime, end: datetime, dataset: str) -> Path:
        period = f"{start:%Y%m%d}_{end:%Y%m%d}"
        return self.cache_dir / period / f"{dataset}.csv"

    def _read_cache(self, path: Path) -> pd.Series | None:
        if not path.is_file():
            return None
        logger.info(f"using cached instrat_pl data from {path}")
        data = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]
        return data

    def _write_cache(self, path: Path, data: pd.Series) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path)

    @staticmethod
    def _download(url: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "date_from": start.strftime("%d-%m-%YT%H:%M:%SZ"),
            "date_to": end.strftime("%d-%m-%YT%H:%M:%SZ"),
        }
        headers = {"User-Agent": USER_AGENT}
        try:
            # request timeout in seconds
            response = requests.get(url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            df = pd.read_json(io.StringIO(response.text))
        except (requests.RequestException, ValueError) as exc:
            # callers treat an empty frame as "use the fallback price"
            logger.warning("instrat.pl request for %s failed: %s", url, exc)
            return pd.DataFrame()
        if df.empty:
            logger.warning("instrat.pl returned no data for %s", url)
            return df
        if "date" not in df.columns:
            logger.warning("instrat.pl response for %s has no date column", url)
            return pd.DataFrame()
        df = df.set_index("date")
        df.index = df.index.tz_localize(None)
        return df

    @staticmethod
    def _pln_to_eur(index: pd.DatetimeIndex) -> pd.Series:
        """Return the PLN->EUR rate on the given dates (yfinance PLNEUR=X)."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for instrat_pl fuel prices. "
                "Install with: pip install 'assume-framework[entsoe]'"
            ) from exc

        start = index[0].strftime("%Y-%m-%d")
        end = index[-1].strftime("%Y-%m-%d")
        # yfinance treats `end` as exclusive; a single-date index (start == end)
        # would otherwise return an empty frame and trigger the constant fallback.
        query_end = (index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download("PLNEUR=X", start=start, end=query_end, progress=False)
        if data is None or data.empty:
            close: pd.Series | pd.DataFrame | None = None
        else:
            close = data["Close"]

        # Depending on the yfinance version, ["Close"] is a Series (single
        # ticker) or a DataFrame (MultiIndex columns); unify to a Series.
        if isinstance(close, pd.DataFrame):
            ticker = "PLNEUR=X"
            if ticker in close.columns:
                close = close[ticker]
            elif ticker in close.columns.get_level_values(-1):
                close = close.loc[:, ticker]
            else:
                close = close.iloc[:, 0]

        if close is None or close.empty:
            logger.warning(
                "No PLN/EUR FX data returned by yfinance; falling back to a "
                "constant rate of %.2f EUR/PLN",
                _PLN_EUR_FALLBACK,
            )
            return pd.Series(_PLN_EUR_FALLBACK, index=index, name="PLNEUR")

        pln_eur = close.reindex(index).ffill().bfill()
        if pln_eur.isna().all():
            logger.warning(
                "PLN/EUR FX data from yfinance does not cover %s - %s; "
                "falling back to a constant rate of %.2f EUR/PLN",
                start,
                end,
                _PLN_EUR_FALLBACK,
            )
            return pd.Series(_PLN_EUR_FALLBACK, index=index, name="PLNEUR")
        return pln_eur

    def get_co2_price(
        self,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> pd.Series:
        """Return EU ETS price in €/tCO2."""
        cache_path = self._cache_path(start, end, "co2")
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        query_start = start - timedelta(days=INSTRAT_LOOKBACK_DAYS["co2"])
        df = self._download(EU_ETS_URL, query_start, end)
        if not df.empty:
            series = _sanitize_daily_prices(df["price"], "co2", _CO2_FALLBACK_EUR_T)
        else:
            series = self._empty_fallback(start, _CO2_FALLBACK_EUR_T, "co2")
        series = series.resample("D").ffill().bfill()
        # never cache fallback prices: they may stem from a transient failure
        if use_cache and not df.empty:
            self._write_cache(cache_path, series)
        return series

    def get_coal_price(
        self,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> pd.Series:
        """Return steam coal price in €/MWh thermal."""
        cache_path = self._cache_path(start, end, "coal")
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                series = _sanitize_daily_prices(cached, "hard coal")
                return series.resample("D").ffill().bfill()

        # coal prices are monthly: look back a full year so that windows
        # shorter than one month still contain an observation
        query_start = start - timedelta(days=INSTRAT_LOOKBACK_DAYS["coal"])
        coal_data = self._download(COAL_URL, query_start, end)
        if not coal_data.empty:
            pln_eur = self._pln_to_eur(coal_data.index)
            steam_coal_eur_per_gj = coal_data["pscmi1_pln_per_gj"] * pln_eur
            # EUR/GJ -> EUR/MWh thermal (1 GJ = 277.8 kWh)
            series = _sanitize_daily_prices(
                steam_coal_eur_per_gj / GJ_TO_KWH * 1e3,
                "hard coal",
                _COAL_FALLBACK_EUR_MWH,
            )
        else:
            series = self._empty_fallback(start, _COAL_FALLBACK_EUR_MWH, "hard coal")
        series = series.resample("D").ffill().bfill()
        # never cache fallback prices: they may stem from a transient failure
        if use_cache and not coal_data.empty:
            self._write_cache(cache_path, series)
        return series

    def get_gas_price(
        self,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> pd.Series:
        """Return gas price in €/MWh thermal."""
        cache_path = self._cache_path(start, end, "gas")
        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        query_start = start - timedelta(days=INSTRAT_LOOKBACK_DAYS["gas"])
        gas_data = self._download(GAS_URL, query_start, end)
        if gas_data.empty:
            series = self._empty_fallback(start, _GAS_FALLBACK_EUR_MWH, "gas")
        else:
            pln_eur = self._pln_to_eur(gas_data.index)
            series = _sanitize_daily_prices(
                gas_data["price"] * pln_eur, "gas", _GAS_FALLBACK_EUR_MWH
            )
        series = series.resample("D").ffill().bfill()
        # never cache fallback prices: they may stem from a transient failure
        if use_cache and not gas_data.empty:
            self._write_cache(cache_path, series)
        return series

    @staticmethod
    def _empty_fallback(start: datetime, value: float, name: str) -> pd.Series:
        """Single-point fallback series when instrat.pl returned no data."""
        logger.warning(
            "instrat.pl returned no data for %s; using fallback %.1f", name, value
        )
        return pd.Series(value, index=[pd.Timestamp(start).normalize()], name=name)

    def get_fuel_prices(
        self,
        start: datetime,
        end: datetime,
        index: pd.DatetimeIndex,
        use_cache: bool = True,
    ) -> dict[str, pd.Series]:
        """
        Return hourly fuel price series for the simulation index.

        Coal and gas come from instrat_pl; lignite reuses the coal series.
        """
        coal = self.get_coal_price(start, end, use_cache=use_cache)
        gas = self.get_gas_price(start, end, use_cache=use_cache)
        co2 = self.get_co2_price(start, end, use_cache=use_cache)

        coal = _to_hourly_prices(coal, index)
        gas = _to_hourly_prices(gas, index)
        co2 = _to_hourly_prices(co2, index)

        return {
            "hard coal": coal,
            "lignite": coal,
            "gas": gas,
            "co2": co2,
        }
