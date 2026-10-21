# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from assume.common.fast_pandas import FastIndex, FastSeries

t0 = datetime(2024, 1, 1)
H = timedelta(hours=1)


@pytest.fixture
def index() -> FastIndex:
    """A 6-hour hourly FastIndex starting at t0."""
    return FastIndex(start=t0, periods=6, freq="1h")


@pytest.fixture
def series(index) -> FastSeries:
    """A FastSeries on `index` with data [0, 1, 2, 3, 4, 5]."""
    return FastSeries(index, np.arange(6.0))


def test_fillna_uses_value(series):
    """fillna(9) should replace NaN with 9 at the NaN position."""
    series.iloc[1] = np.nan
    assert list(series.fillna(9).data) == [0, 9, 2, 3, 4, 5]


def test_fillna_does_not_mutate_original(series):
    """fillna should return a new series and leave the original NaN untouched."""
    series.iloc[1] = np.nan
    series.fillna(0)
    assert np.isnan(series.data[1])


def test_integer_slice_assignment_is_positional(series):
    """series[2:4] = 7 should only overwrite positions 2 and 3, negative bounds count from the end."""
    series[2:4] = 7
    assert list(series.data) == [0, 1, 7, 7, 4, 5]
    series[-2:] = [40.0, 50.0]
    assert list(series.data) == [0, 1, 7, 7, 40, 50]


def test_integer_slice_read_is_positional(series):
    """series[1:3] and series[-2:] should slice by position, like python and pandas."""
    assert list(series[1:3]) == [1, 2]
    assert list(series[-2:]) == [4, 5]


def test_string_slice_read_is_rejected(series):
    """String slices are not supported; [] slices take datetimes only."""
    with pytest.raises(TypeError):
        _ = series["2024-01-01 01:00":"2024-01-01 02:00"]


def test_scalar_assignment_with_full_list_key(series):
    """Assigning a scalar via a full-length date list should set every element."""
    series[series.index.get_date_list()] = 5.0
    assert list(series.data) == [5.0] * 6


def test_full_int_list_assignment_keeps_float_dtype(series):
    """After assigning ints over the full index, later float writes should keep their fraction."""
    series[series.index.get_date_list()] = [1, 2, 3, 4, 5, 6]
    series.at[t0] = 1.5
    assert series.at[t0] == 1.5
    assert series.dtype == np.float64


def test_constructor_rejects_wrong_length_values(index):
    """Constructing with 3 values for a 6-period index should raise ValueError."""
    with pytest.raises(ValueError):
        FastSeries(index, [1.0, 2.0, 3.0])


def test_constructor_rejects_mismatched_pandas_series(index):
    """A 15-minute pandas Series on an hourly index should raise or produce length 6."""
    quarter_hourly = pd.Series(1.0, index=pd.date_range(t0, periods=24, freq="15min"))
    try:
        result = FastSeries(index, quarter_hourly)
    except ValueError:
        return
    assert len(result.data) == len(index)


def test_constructor_scalar_numpy_int_broadcasts(index):
    """A numpy integer scalar should broadcast to a 1-d array of the index length."""
    series = FastSeries(index, np.int64(3))
    assert series.data.shape == (len(index),)


def test_arithmetic_accepts_numpy_scalars(series):
    """Adding a numpy int64 or multiplying by a numpy float32 should work elementwise."""
    assert list((series + np.int64(1)).data) == [1, 2, 3, 4, 5, 6]
    assert list((series * np.float32(2)).data) == [0, 2, 4, 6, 8, 10]


def test_create_date_range():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 5)
    n = 1000
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)

    t = time.time()
    for i in range(n):
        index = FastIndex(start, end, freq="1h")
        fs = FastSeries(index)
    res = time.time() - t

    t = time.time()
    for i in range(n):
        q_pd = pd.date_range(start, end, freq="1h")
    res_pd = time.time() - t
    # this is sometimes faster, sometimes not
    # as a lot of objects are created
    assert res < res_pd + 0.1

    new_end = datetime(2020, 1, 1, 3)

    # check that slicing is faster
    t = time.time()
    for i in range(n):
        q_slice = fs.loc[start:new_end]
    res_slice = time.time() - t

    series = pd.Series(0, index=q_pd)

    t = time.time()
    for i in range(n):
        q_pd_slice = series.loc[start:new_end]
    res_slice_pd = time.time() - t
    # more than at least factor 4
    assert res_slice < res_slice_pd / 4

    # check that setting items is faster:
    t = time.time()
    for i in range(n):
        fs.at[start] = 1
    res_slice = time.time() - t

    series = pd.Series(0, index=q_pd)

    t = time.time()
    for i in range(n):
        series.at[start] = 1
    res_slice_pd = time.time() - t
    # more than at least factor 4
    assert res_slice < res_slice_pd / 4

    # check that setting slices is faster
    t = time.time()
    for i in range(n):
        fs.loc[start:new_end] = 17
    res_slice = time.time() - t

    series = pd.Series(0, index=q_pd)

    t = time.time()
    for i in range(n):
        series.loc[start:new_end] = 17
    res_slice_pd = time.time() - t
    # more than at least factor 4
    assert res_slice < res_slice_pd / 4

    se = pd.Series(0.0, index=fs.index.get_date_list())
    se.loc[start]

    series.loc[new_end] = 33

    fs[new_end] = 33
    new = series.loc[start:new_end][::-1]
    assert new.iloc[0] == 33
    new = fs.loc[start:new_end][::-1]
    assert new[0] == 33
    fs.data
    fs.index._get_idx_from_date(start)
    fs.index._get_idx_from_date(new_end)
    fs.data[0:4]


def test_convert_pd():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 5)
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)

    df = fs.as_df()
    assert isinstance(df, pd.DataFrame)


def test_set_list():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 2, 1, 5)
    n = 1000
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)

    dr = pd.date_range(start, end=datetime(2020, 1, 3, 5))
    dr = pd.Series(dr)

    series = pd.Series(0, index=pd.date_range(start, end, freq="1h"))

    t = time.time()
    for i in range(n):
        result_pd = fs[dr]
    res_fds = time.time() - t
    f_series = series > 1
    series[f_series] = 4
    fs.data[f_series] = 4

    # accessing lists or series elements is also faster
    # check getting list or series
    t = time.time()
    for i in range(n):
        result = series[dr]
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd

    # check setting list or series with single value
    t = time.time()
    for i in range(n):
        fs[dr] = 3
    res_fds = time.time() - t

    t = time.time()
    for i in range(n):
        series[dr] = 3
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd

    # check setting list or series with a series
    d_new = pd.Series(dr.index)

    t = time.time()
    for i in range(n):
        fs[dr] = d_new
    res_fds = time.time() - t

    t = time.time()
    for i in range(n):
        series[dr] = d_new
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd

    # check setting list or series with a list
    d_new = list(d_new)

    t = time.time()
    for i in range(n):
        fs[dr] = d_new
    res_fds = time.time() - t

    t = time.time()
    for i in range(n):
        series[dr] = d_new
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd


def test_window_edge_cases():
    # ── setup ────────────────────────────────
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 4)
    idx = FastIndex(start, end, freq="1h")
    fs = FastSeries(idx, value=np.arange(len(idx)))  # data = [0,1,2,3,4]

    # ── 1. forward, no wrap ───────────────────
    out = fs.window(center=1, length=3, direction="forward")
    assert np.array_equal(out, np.array([1, 2, 3]))

    # ── 2. forward, wrap ──────────────────────
    # raw positions = [3,4,5,6] → wrapped = [3,4,0,1]
    out = fs.window(center=3, length=4, direction="forward")
    assert np.array_equal(out, np.array([3, 4, 0, 1]))

    # ── 3. backward, no wrap ──────────────────
    # raw positions = [1,2,3]
    out = fs.window(center=3, length=3, direction="backward")
    assert np.array_equal(out, np.array([1, 2, 3]))

    # ── 4. backward, wrap ─────────────────────
    # raw positions = [-1,0,1] → wrapped = [4,0,1]
    out = fs.window(center=1, length=3, direction="backward")
    assert np.array_equal(out, np.array([4, 0, 1]))

    # ── 5. datetime center equivalence ────────
    dt_center = start + timedelta(hours=2)
    out_int = fs.window(center=2, length=3, direction="forward")
    out_dt = fs.window(center=dt_center, length=3, direction="forward")
    assert np.array_equal(out_int, out_dt)

    # ── 6. invalid direction raises ───────────
    with pytest.raises(ValueError):
        fs.window(center=0, length=5, direction="sideways")

    # ── 7. full-cycle rotations ───────────────
    N = len(fs)
    # full window from 0 is original array
    full0 = fs.window(center=0, length=N, direction="forward")
    assert np.array_equal(full0, fs.data)
    # full window from 2 is roll by -2
    full2 = fs.window(center=2, length=N, direction="forward")
    assert np.array_equal(full2, np.roll(fs.data, -2))

    # ── 8. non-mutation guarantee ─────────────
    before = fs.data.copy()
    _ = fs.window(center=1, length=4)
    assert np.array_equal(fs.data, before)


def test_constructor_scalar_and_array(index):
    """A scalar is broadcast; an array of matching length is stored as float64."""
    assert list(FastSeries(index).data) == [0.0] * 6
    assert list(FastSeries(index, 2.5).data) == [2.5] * 6
    data = FastSeries(index, [1, 2, 3, 4, 5, 6], name="x")
    assert data.dtype == np.float64
    assert data.name == "x"
    assert len(data) == 6


def test_constructor_rejects_non_fastindex():
    """The index must be a FastIndex."""
    with pytest.raises(TypeError):
        FastSeries(pd.date_range(t0, periods=6, freq="h"))


def test_setitem_datetime_and_slice(series):
    """Setting by datetime, string and datetime slice should only change the targeted values."""
    series.at[t0] = 10.0
    series["2024-01-01 01:00"] = 11.0
    series.loc[t0 + 3 * H : t0 + 4 * H] = 0.0
    assert list(series.data) == [10.0, 11.0, 2.0, 0.0, 0.0, 5.0]


def test_setitem_slice_with_array(series):
    """Assigning an array of matching length to a datetime slice should work, a wrong length should raise."""
    series.loc[t0 + H : t0 + 2 * H] = np.array([7.0, 8.0])
    assert list(series.data) == [0.0, 7.0, 8.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        series.loc[t0 + H : t0 + 2 * H] = np.array([1.0, 2.0, 3.0])


def test_list_get_and_set(series):
    """A partial list of dates should read and write the matching positions."""
    keys = [t0 + H, t0 + 3 * H]
    np.testing.assert_array_equal(series[keys], [1.0, 3.0])
    series[keys] = [10.0, 30.0]
    assert list(series.data) == [0.0, 10.0, 2.0, 30.0, 4.0, 5.0]
    series[keys] = -1.0
    assert list(series.data) == [0.0, -1.0, 2.0, -1.0, 4.0, 5.0]


def test_arithmetic_with_scalars_and_series(series):
    """Arithmetic with scalars and aligned series returns new FastSeries and leaves inputs unchanged."""
    other = FastSeries(series.index, 2.0)
    assert list((series + 1).data) == [1, 2, 3, 4, 5, 6]
    assert list((series - other).data) == [-2, -1, 0, 1, 2, 3]
    assert list((series * other).data) == [0, 2, 4, 6, 8, 10]
    assert list((series / 2).data) == [0, 0.5, 1, 1.5, 2, 2.5]
    assert list((10 - series).data) == [10, 9, 8, 7, 6, 5]
    assert list((2 * series).data) == [0, 2, 4, 6, 8, 10]
    assert list((1 + series).data) == [1, 2, 3, 4, 5, 6]
    assert isinstance(series + other, FastSeries)
    assert list(series.data) == [0, 1, 2, 3, 4, 5]


def test_arithmetic_with_misaligned_series_raises(series):
    """Arithmetic between series with different indices should raise ValueError."""
    other = FastSeries(FastIndex(start=t0, periods=3, freq="1h"), 1.0)
    with pytest.raises(ValueError):
        series + other


def test_inplace_arithmetic(series):
    """In-place operators should update the series itself."""
    original = series
    series += 1
    series *= 2
    series -= 2
    series /= 2
    assert series is original
    assert list(series.data) == [0, 1, 2, 3, 4, 5]


def test_unary_and_comparison_operators(series):
    """Negation, abs and comparisons should work elementwise."""
    negative = -series
    assert list(negative.data) == [0, -1, -2, -3, -4, -5]
    assert list(abs(negative).data) == [0, 1, 2, 3, 4, 5]
    assert list(series > 2) == [False, False, False, True, True, True]
    assert list(series <= 1) == [True, True, False, False, False, False]
    assert list(series == 3) == [False, False, False, True, False, False]
    assert list(series != 3) == [True, True, True, False, True, True]


def test_aggregations(series):
    """Aggregation methods should match numpy on the data."""
    assert series.sum() == 15
    assert series.mean() == 2.5
    assert series.min() == 0
    assert series.max() == 5
    assert series.median() == 2.5
    assert series.std() == pytest.approx(np.arange(6.0).std())


def test_copy_is_independent(series):
    """Changing a copy (shallow or deep) should not change the original."""
    for copied in (series.copy(), series.copy(deep=True)):
        copied.at[t0] = 99.0
        assert series.at[t0] == 0.0
        assert copied.index is series.index


def test_fillna_without_nan_and_ones_like(series):
    """fillna on a series without NaN keeps the data; ones_like returns ones."""
    assert list(series.fillna(1).data) == [0, 1, 2, 3, 4, 5]
    assert list(series.ones_like().data) == [1.0] * 6


def test_iteration(series):
    """Iterating a series should yield its values."""
    assert list(series) == [0, 1, 2, 3, 4, 5]


def test_pandas_round_trip(series):
    """as_pd_series and from_pandas_series should round-trip the data and dates."""
    pd_series = series.as_pd_series(name="values")
    assert list(pd_series.index) == series.index.get_date_list()
    assert list(pd_series.values) == list(series.data)
    assert pd_series.name == "values"

    restored = FastSeries.from_pandas_series(pd_series)
    assert restored.name == "values"
    assert restored.freq == H
    assert list(restored.data) == list(series.data)


def test_as_pd_series_and_as_df_subrange(series):
    """as_pd_series and as_df should honour start and end."""
    sub = series.as_pd_series(start=t0 + H, end=t0 + 2 * H)
    assert list(sub.values) == [1.0, 2.0]
    assert list(sub.index) == [t0 + H, t0 + 2 * H]
    df = series.as_df(name="col", start=t0 + H, end=t0 + 2 * H)
    assert list(df.columns) == ["col"]
    assert list(df["col"]) == [1.0, 2.0]


def test_from_pandas_series_rejects_empty_and_irregular():
    """Empty series and series without inferable frequency should raise ValueError."""
    with pytest.raises(ValueError):
        FastSeries.from_pandas_series(pd.Series(dtype=float))
    irregular = pd.Series(1.0, index=pd.to_datetime([t0, t0 + H, t0 + 3 * H]))
    with pytest.raises(ValueError):
        FastSeries.from_pandas_series(irregular)


def test_unsupported_key_type_raises(series):
    """Keys of unsupported types should raise TypeError."""
    with pytest.raises(TypeError):
        series[1.5]
    with pytest.raises(TypeError):
        series[1.5] = 0.0
