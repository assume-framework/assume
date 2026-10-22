# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.fast_pandas import FastIndex, FastSeries

t0 = datetime(2024, 1, 1)
H = timedelta(hours=1)


@pytest.fixture
def series() -> FastSeries:
    """A 6-hour hourly FastSeries with data [0, 1, 2, 3, 4, 5]."""
    return FastSeries(FastIndex(start=t0, periods=6, freq="1h"), np.arange(6.0))


def test_at_before_index_start_raises(series):
    """A timestamp before the index start should raise KeyError/IndexError, not wrap around."""
    with pytest.raises((KeyError, IndexError)):
        _ = series.at[t0 - H]


def test_loc_slice_start_before_index_start_is_clipped(series):
    """A slice starting before the index start should be clipped like pandas, giving [0, 1, 2]."""
    result = np.asarray(series.loc[t0 - 2 * H : t0 + 2 * H])
    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))


def test_loc_slice_stop_before_index_start_is_empty(series):
    """A slice whose stop is before the index start should be empty."""
    result = np.asarray(series.loc[: t0 - 2 * H])
    assert result.size == 0


def test_contains_ignores_frequency_alignment(series):
    """A timestamp not aligned to the index frequency should not be reported as contained."""
    assert (t0 + timedelta(minutes=30)) not in series.index


@pytest.mark.parametrize("minutes", [30, 40])
def test_at_unaligned_is_not_rounded_badly(series, minutes):
    """An unaligned scalar lookup should raise KeyError/ValueError or return the 01:00 value (1.0)."""
    assert series.at[t0 + timedelta(minutes=minutes)] == 1.0


def test_loc_slice_start_round_up_matches_pandas(series):
    """A slice starting at 00:40 should start at 01:00 and give [1, 2, 3, 4, 5] like pandas."""
    result = np.asarray(series.loc[t0 + timedelta(minutes=40) :])
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_get_date_list_round_up_start_is_next_hour(series):
    """get_date_list from 00:40 should begin at 01:00, the next grid point."""
    first = series.index.get_date_list(t0 + timedelta(minutes=40))[0]
    assert first == t0 + H


def test_loc_slice_stop_round_down_matches_pandas(series):
    """A slice stopping at 00:40 should end at 00:00 and give [0] (floor, not round)."""
    result = np.asarray(series.loc[: t0 + timedelta(minutes=40)])
    np.testing.assert_array_equal(result, np.array([0.0]))


@pytest.mark.parametrize("freq", ["m", "W", "M", "MS", "ME"])
def test_index_rejects_calendar_frequencies(freq):
    """Calendar frequencies have no fixed step and should raise ValueError."""
    with pytest.raises(ValueError):
        FastIndex(start=t0, periods=3, freq=freq)


def test_slicing_fastseries_even():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 5)
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)
    b = start + timedelta(hours=1)
    e = start + timedelta(hours=4)
    result = fs[b:e]

    datelist = fs.index.get_date_list(b, e)
    series = pd.Series(0, index=pd.date_range(start, end, freq="h"))
    assert list(series[b:e].index) == datelist
    assert len(series[b:e]) == len(fs[b:e])


def test_slicing_fastseries_uneven():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 5)
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)
    b = start + timedelta(seconds=1)
    e = start + timedelta(hours=4, seconds=1)
    result = fs[b:e]

    datelist = fs.index.get_date_list(b, e)
    series = pd.Series(0, index=pd.date_range(start, end, freq="h"))
    assert list(series[b:e].index) == datelist
    assert len(series[b:e]) == len(fs[b:e])


def test_index_end_and_periods_are_equivalent():
    """Building with end= or periods= should give the same index."""
    by_end = FastIndex(start=t0, end=t0 + 5 * H, freq="1h")
    by_periods = FastIndex(start=t0, periods=6, freq="1h")
    assert len(by_end) == len(by_periods) == 6
    assert by_end.end == by_periods.end == t0 + 5 * H
    assert by_end.get_date_list() == by_periods.get_date_list()


def test_index_requires_end_or_periods():
    """Omitting both end and periods should raise ValueError."""
    with pytest.raises(ValueError):
        FastIndex(start=t0)


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("1h", H),
        ("h", H),
        ("15min", timedelta(minutes=15)),
        ("min", timedelta(minutes=1)),
        ("D", 24 * H),
        ("24h", 24 * H),
        (H, H),
    ],
)
def test_index_parses_frequency(freq, expected):
    """Timedeltas and fixed offset aliases should be parsed to a timedelta."""
    assert FastIndex(start=t0, periods=2, freq=freq).freq == expected


def test_index_rejects_invalid_frequency_type():
    """A frequency that is neither str nor timedelta should raise TypeError."""
    with pytest.raises(TypeError):
        FastIndex(start=t0, periods=2, freq=3600)


def test_index_accepts_string_start():
    """A string start should be converted to a datetime."""
    assert FastIndex(start="2024-01-01", periods=2, freq="1h").start == t0


def test_index_integer_access(series):
    """Integer access should return dates, including negative indices."""
    index = series.index
    assert index[0] == t0
    assert index[2] == t0 + 2 * H
    assert index[-1] == t0 + 5 * H
    with pytest.raises(IndexError):
        index[6]


def test_index_slicing_by_int_and_datetime(series):
    """Integer and datetime slices should return the matching dates (datetime stop inclusive)."""
    index = series.index
    assert index[1:3] == [t0 + H, t0 + 2 * H]
    assert index[t0 + H : t0 + 2 * H] == [t0 + H, t0 + 2 * H]


def test_index_contains_aligned_dates(series):
    """Aligned dates inside the range are contained, dates outside are not."""
    assert t0 in series.index
    assert t0 + 5 * H in series.index
    assert t0 + 6 * H not in series.index
    assert t0 - H not in series.index


def test_index_get_date_list_subrange(series):
    """get_date_list(start, end) should return the inclusive sub-range."""
    assert series.index.get_date_list(t0 + H, t0 + 3 * H) == [
        t0 + H,
        t0 + 2 * H,
        t0 + 3 * H,
    ]


def test_index_pandas_round_trip():
    """as_datetimeindex and from_pandas_index should round-trip an hourly index."""
    pd_index = pd.date_range(t0, periods=6, freq="h")
    index = FastIndex.from_pandas_index(pd_index)
    assert len(index) == 6
    assert index.freq == H
    assert list(index.as_datetimeindex()) == list(pd_index)


def test_series_datetime_and_string_lookup(series):
    """Aligned datetime and string keys should return the stored value."""
    assert series.at[t0 + 2 * H] == 2.0
    assert series[t0 + 3 * H] == 3.0
    assert series["2024-01-01 04:00"] == 4.0


def test_series_datetime_slice_is_inclusive(series):
    """A datetime slice should include both its start and stop, like pandas."""
    np.testing.assert_array_equal(series.loc[t0 + H : t0 + 3 * H], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(series.loc[: t0 + H], [0.0, 1.0])
    np.testing.assert_array_equal(series.loc[t0 + 4 * H :], [4.0, 5.0])
