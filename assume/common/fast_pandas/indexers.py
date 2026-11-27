# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from assume.common.fast_pandas.series import FastSeries


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
