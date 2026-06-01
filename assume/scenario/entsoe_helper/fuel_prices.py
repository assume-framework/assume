# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import io
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EU_ETS_URL = "https://energy-api.instrat.pl/api/prices/co2"
COAL_URL = "https://energy-api.instrat.pl/api/coal/pscmi_1"
GAS_URL = "https://energy-api.instrat.pl/api/prices/gas_price_rdn_daily"
USER_AGENT = (
    "Mozilla/5.0 (compatible; ASSUME/1.0; +https://assume-project.de/)"
)

GJ_TO_KWH = 1e6 / 3600

# €/MWh thermal – fallback when instrat returns no usable coal data
_COAL_FALLBACK_EUR_MWH = 18.0


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
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        df = pd.read_json(io.StringIO(response.text))
        df = df.set_index("date")
        df.index = df.index.tz_localize(None)
        return df

    @staticmethod
    def _pln_to_eur(index: pd.DatetimeIndex) -> pd.Series:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for instrat_pl fuel prices. "
                "Install with: pip install 'assume-framework[entsoe]'"
            ) from exc

        start = index[0].strftime("%Y-%m-%d")
        end = index[-1].strftime("%Y-%m-%d")
        pln_eur = yf.download("PLNEUR=X", start=start, end=end, progress=False)[
            "Close"
        ]
        if isinstance(pln_eur.columns, pd.MultiIndex):
            pln_eur = pln_eur["PLNEUR=X"]
        else:
            pln_eur = pln_eur.squeeze()
        pln_eur = pln_eur.reindex(index).ffill().bfill()
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

        df = self._download(EU_ETS_URL, start, end)
        series = _sanitize_daily_prices(df["price"], "co2", fallback=80.0)
        series = series.resample("D").ffill().bfill()
        if use_cache:
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

        coal_data = self._download(COAL_URL, start, end)
        pln_eur = self._pln_to_eur(coal_data.index)
        steam_coal_eur_per_gj = coal_data["pscmi1_pln_per_gj"] * pln_eur
        series = _sanitize_daily_prices(
            steam_coal_eur_per_gj / GJ_TO_KWH * 1e3, "hard coal"
        )
        series = series.resample("D").ffill().bfill()
        if use_cache and not series.empty:
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

        gas_data = self._download(GAS_URL, start, end)
        pln_eur = self._pln_to_eur(gas_data.index)
        series = _sanitize_daily_prices(gas_data["price"] * pln_eur, "gas", fallback=30.0)
        series = series.resample("D").ffill().bfill()
        if use_cache:
            self._write_cache(cache_path, series)
        return series

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
