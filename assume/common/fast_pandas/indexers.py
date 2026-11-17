# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from assume.common.fast_pandas.series import FastSeries


class FastSeriesLocIndexer:
    def __init__(self, series: FastSeries):
        self._series = series

    def __getitem__(
        self, item: datetime | slice | list | pd.Index | pd.Series | np.ndarray | str
    ):
        """
        Retrieve item(s) using label-based indexing.

        Parameters:
            item (datetime | slice | list | pd.Index | pd.Series | np.ndarray | str): The label(s) to retrieve.

        Returns:
            float | np.ndarray: The retrieved value(s).
        """
        return self._series.__getitem__(item)

    def __setitem__(
        self,
        item: datetime | slice | list | pd.Index | pd.Series | np.ndarray | str,
        value: float | np.ndarray,
    ):
        """
        Assign value(s) using label-based indexing.

        Parameters:
            item (datetime | slice | list | pd.Index | pd.Series | np.ndarray | str): The label(s) to set.
            value (float | np.ndarray): The value(s) to assign.
        """
        self._series.__setitem__(item, value)


class FastSeriesILocIndexer:
    def __init__(self, series: FastSeries):
        self._series = series

    def __getitem__(self, item: int | slice) -> float | np.ndarray:
        """
        Retrieve item(s) using integer-based indexing.

        Parameters:
            item (int | slice): The integer index or slice.

        Returns:
            float | np.ndarray: The retrieved value(s).

        Raises:
            IndexError: If the index is out of bounds.
            TypeError: If the index type is unsupported.
        """
        if isinstance(item, int):
            if item < 0 or item >= len(self._series):
                raise IndexError(
                    f"Index {item} is out of bounds for series of length {len(self._series)}"
                )
            return self._series._data[item]

        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or len(self._series)
            step = item.step or 1

            if start < 0:
                start += len(self._series)
            if stop < 0:
                stop += len(self._series)

            start = max(0, start)
            stop = min(len(self._series), stop)

            return self._series._data[start:stop:step]

        else:
            raise TypeError(
                f"Unsupported index type for iloc: {type(item)}. Must be int or slice."
            )

    def __setitem__(self, item: int | slice, value: float | np.ndarray):
        """
        Assign value(s) using integer-based indexing.

        Parameters:
            item (int | slice): The integer index or slice.
            value (float | np.ndarray): The value(s) to assign.

        Raises:
            IndexError: If the index is out of bounds.
            TypeError: If the index type is unsupported.
            ValueError: If the length of the value does not match the slice length.
        """
        if isinstance(item, int):
            if item < 0 or item >= len(self._series):
                raise IndexError(
                    f"Index {item} is out of bounds for series of length {len(self._series)}"
                )
            self._series._data[item] = value

        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or len(self._series)
            step = item.step or 1

            if start < 0:
                start += len(self._series)
            if stop < 0:
                stop += len(self._series)

            start = max(0, start)
            stop = min(len(self._series), stop)

            # Assign the values
            if np.isscalar(value) or len(self._series._data[start:stop:step]) == len(
                value
            ):
                self._series._data[start:stop:step] = value
            else:
                raise ValueError(
                    f"Length of value ({len(value)}) does not match the length of the slice "
                    f"({len(self._series._data[start:stop:step])})."
                )
        else:
            raise TypeError(
                f"Unsupported index type for iloc: {type(item)}. Must be int or slice."
            )


class FastSeriesAtIndexer:
    def __init__(self, series: FastSeries):
        self._series = series

    def __getitem__(self, item):
        """
        Retrieve a single item using label-based indexing.

        Parameters:
            item (datetime | str): The label.

        Returns:
            float: The retrieved value.
        """
        return self._series[item]

    def __setitem__(self, item, value: float):
        """
        Assign a value using label-based indexing.

        Parameters:
            item (datetime | str): The label.
            value (float): The value to assign.
        """
        self._series[item] = value


class FastSeriesIatIndexer:
    def __init__(self, series: FastSeries):
        self._series = series

    def __getitem__(self, item: int) -> float:
        """
        Retrieve a single item using integer-based indexing.

        Parameters:
            item (int): The integer index.

        Returns:
            float: The retrieved value.

        Raises:
            IndexError: If the index is out of bounds.
            TypeError: If the index is not an integer.
        """
        if not isinstance(item, int):
            raise TypeError(
                f"iat only supports single integer indices, got {type(item)}"
            )
        return self._series.iloc[item]

    def __setitem__(self, item: int, value: float):
        """
        Assign a value using integer-based indexing.

        Parameters:
            item (int): The integer index.
            value (float): The value to assign.

        Raises:
            IndexError: If the index is out of bounds.
            TypeError: If the index is not an integer.
        """
        if not isinstance(item, int):
            raise TypeError(
                f"iat only supports single integer indices, got {type(item)}"
            )
        self._series.iloc[item] = value
