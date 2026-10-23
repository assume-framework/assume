# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np
import pytest

from assume.common.fast_pandas import FastIndex, FastSeries, TensorFastSeries

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


@pytest.fixture
def tensor_series(index) -> TensorFastSeries:
    """A TensorFastSeries on `index` filled with tensor(0.0); skipped without torch."""
    pytest.importorskip("torch")
    return TensorFastSeries(index, 0.0)


def test_iloc_and_iat_access(series):
    """iloc and iat should read and write by position."""
    assert series.iloc[2] == 2.0
    assert series.iat[3] == 3.0
    np.testing.assert_array_equal(series.iloc[1:3], [1.0, 2.0])
    np.testing.assert_array_equal(series.iloc[-2:], [4.0, 5.0])
    np.testing.assert_array_equal(series.iloc[::2], [0.0, 2.0, 4.0])

    series.iloc[0] = 10.0
    series.iat[1] = 11.0
    series.iloc[4:] = [40.0, 50.0]
    assert list(series.data) == [10.0, 11.0, 2.0, 3.0, 40.0, 50.0]


def test_iloc_out_of_bounds_and_bad_types(series):
    """Out-of-range positions raise IndexError, unsupported keys raise TypeError or ValueError."""
    with pytest.raises(IndexError):
        series.iloc[6]
    with pytest.raises(IndexError):
        series.iloc[6] = 1.0
    with pytest.raises(TypeError):
        series.iloc[t0]
    with pytest.raises(TypeError):
        series.iat[1:2]
    with pytest.raises(ValueError):
        series.iloc[0:2] = [1.0, 2.0, 3.0]


def test_loc_and_at_delegate_to_series(series):
    """loc and at should read and write by date."""
    series.loc[t0 + H] = 9.0
    series.at[t0 + 2 * H] = 8.0
    assert series.loc[t0 + H] == 9.0
    assert series.at[t0 + 2 * H] == 8.0


def test_tensor_series_get_and_set(tensor_series):
    """TensorFastSeries should store clones and support int, datetime and slice access."""
    import torch as th

    value = th.tensor(3.0)
    tensor_series[1] = value
    tensor_series[t0 + 2 * H] = th.tensor(4.0)
    tensor_series[3:5] = [th.tensor(5.0), th.tensor(6.0)]
    value += 1

    assert float(tensor_series[1]) == 3.0
    assert float(tensor_series[t0 + 2 * H]) == 4.0
    assert [float(t) for t in tensor_series[3:5]] == [5.0, 6.0]
    assert len(tensor_series) == 6


def test_tensor_series_init_and_copy(index):
    """TensorFastSeries can start empty or from a tensor; deep copies are independent."""
    th = pytest.importorskip("torch")
    empty = TensorFastSeries(index)
    assert empty[0] is None

    filled = TensorFastSeries(index, th.tensor([1.0, 2.0]))
    deep = filled.copy(deep=True)
    deep[0].add_(1)
    assert filled[0].tolist() == [1.0, 2.0]

    with pytest.raises(TypeError):
        TensorFastSeries(index, "bad")
