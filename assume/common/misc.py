import logging
import os

import dateutil.rrule as rr
import pandas as pd

from assume.common import MarketConfig, MarketProduct

freq_map = {"h": rr.HOURLY, "m": rr.MINUTELY}

logger = logging.getLogger(__name__)


def load_file(
    path: str,
    config: dict,
    file_name: str,
    index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    df = None

    if file_name in config:
        file_path = f"{path}/{config[file_name]}"
    else:
        file_path = f"{path}/{file_name}.csv"

    try:
        df = pd.read_csv(file_path, index_col=0, encoding="Latin-1")
        if index is not None:
            if len(df.index) != len(index):
                logger.warning(
                    f"Length of index does not match length of {file_name} dataframe. Attempting to resample."
                )
                df = attempt_resample(df, index)
            else:
                df.index = index

        for col in df:
            # check if the column is of dtype int
            if df[col].dtype == "int":
                # convert the column to float
                df[col] = df[col].astype(float)

        return df

    except FileNotFoundError as e:
        logger.warning(f"File {file_name} not found. Returning None")


def attempt_resample(
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(df.index) > len(index):
        temp_index = pd.date_range(start=index[0], end=index[-1], periods=len(df))
        df.index = temp_index
        df = df.resample(index.freq).mean()
        logger.info("Resampling successful.")
    elif len(df.index) < index.freq:
        raise ValueError("Index length mismatch. Upsampling not supported.")

    return df


def convert_to_rrule_freq(string):
    freq = freq_map[string[-1]]
    interval = int(string[:-1])
    return freq, interval


def make_market_config(
    id,
    market_params,
    start,
    end,
):
    freq, interval = convert_to_rrule_freq(market_params["opening_frequency"])

    market_config = MarketConfig(
        name=id,
        market_products=[
            MarketProduct(
                duration=pd.Timedelta(market_params["duration"]),
                count=market_params["count"],
                first_delivery=pd.Timedelta(market_params["first_delivery"]),
            )
        ],
        opening_hours=rr.rrule(
            freq=freq,
            interval=interval,
            dtstart=start,
            until=end,
            cache=True,
        ),
        opening_duration=pd.Timedelta(market_params["opening_duration"]),
        maximum_gradient=market_params.get("max_gradient"),
        volume_unit=market_params.get("volume_unit"),
        volume_tick=market_params.get("volume_tick"),
        maximum_volume=market_params["maximum_volume"],
        price_tick=market_params.get("price_tick"),
        price_unit=market_params["price_unit"],
        market_mechanism=market_params["market_mechanism"],
    )

    return market_config
