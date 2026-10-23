# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from assume.common.fast_pandas.index import _ALIGNMENT_EPSILON_SECONDS, FastIndex
from assume.common.fast_pandas.indexers import (
    FastSeriesAtIndexer,
    FastSeriesIatIndexer,
    FastSeriesILocIndexer,
    FastSeriesLocIndexer,
)


class FastSeries:
    """
    A fast, memory-efficient replacement for pandas Series with a FastIndex.

    This class leverages NumPy arrays for data storage to enhance performance
    during market simulations. It supports lazy initialization, vectorized
    operations, and partial compatibility with pandas Series for ease of use.

    Attributes:
        index (FastIndex): The datetime-based index for the series.
        data (np.ndarray): The underlying NumPy array storing series values.
        name (str): The name of the series.
    """

    def __init__(
        self, index: FastIndex, value: float | np.ndarray | list = 0.0, name: str = ""
    ):
        """
        Initialize the FastSeries.

        Parameters:
            index (FastIndex): The datetime index.
            value (float | np.ndarray, optional): Initial value(s) for the data. Defaults to 0.0.
            name (str, optional): Name of the series. Defaults to an empty string.
        """
        # check that the index is a FastIndex
        if not isinstance(index, FastIndex):
            raise TypeError("In FastSeries, index must be a FastIndex object.")

        self._index = index
        self._name = name

        if isinstance(value, pd.Series) and is_datetime64_any_dtype(value.index):
            value = value[self.start : self.end]

        count = len(self.index)  # Use index length directly
        data = np.array(value, dtype=np.float64)
        if data.ndim == 0:
            data = np.full(count, data, dtype=np.float64)
        elif data.shape != (count,):
            raise ValueError(
                f"Length of values ({len(data)}) does not match length of index ({count})."
            )
        self._data = data

    @property
    def index(self) -> FastIndex:
        """Get the FastIndex of the series."""
        return self._index

    @property
    def start(self) -> datetime:
        """Get the start datetime of the series."""
        return self._index.start

    @property
    def end(self) -> datetime:
        """Get the end datetime of the series."""
        return self._index.end

    @property
    def freq(self) -> timedelta:
        """Get the frequency of the series as a timedelta."""
        return self._index.freq

    @property
    def freq_seconds(self) -> float:
        """Get the frequency of the series in total seconds."""
        return self._index.freq_seconds

    @property
    def name(self) -> str:
        """Get the name of the series."""
        return self._name

    @property
    def data(self) -> np.ndarray:
        """
        Access the underlying data array.

        Returns:
            np.ndarray: The data array.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """
        Set the underlying data array.

        Parameters:
            value (np.ndarray): The new data array.
        """
        if np.ndim(value) == 0 or len(value) != len(self.index):
            raise ValueError("Data length must match index length.")
        self._data = value

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the series.

        Returns:
            np.dtype: The data type of the underlying NumPy array.
        """
        return self.data.dtype

    @property
    def loc(self):
        """
        Label-based indexing property.

        Returns:
            FastSeriesLocIndexer: Indexer for label-based access.
        """
        return FastSeriesLocIndexer(self)

    @property
    def iloc(self):
        """
        Integer-based indexing property.

        Returns:
            FastSeriesILocIndexer: Indexer for integer-based access.
        """
        return FastSeriesILocIndexer(self)

    @property
    def at(self):
        """
        Label-based single-item access property.

        Returns:
            FastSeriesAtIndexer: Indexer for label-based single-element access.
        """
        return FastSeriesAtIndexer(self)

    @property
    def iat(self):
        """
        Integer-based single-item access property.

        Returns:
            FastSeriesIatIndexer: Indexer for integer-based single-element access.
        """
        return FastSeriesIatIndexer(self)

    def fillna(self, value):
        result = self.copy()
        result.data = np.where(np.isnan(self.data), value, self.data)
        return result

    def ones_like(self):
        result = self.copy()
        result.data = np.ones_like(self.data)
        return result

    def __getitem__(
        self, item: datetime | slice | list | pd.Index | pd.Series | np.ndarray | str
    ):
        """
        Retrieve item(s) from the series using datetime or label-based indexing.

        Parameters:
            item (datetime | slice | list | pd.Index | pd.Series | np.ndarray | str):
                The key(s) to retrieve.

        Returns:
            float | np.ndarray: The retrieved value(s).

        Raises:
            TypeError: If the index type is unsupported.
            KeyError: If dates are outside of the index range or not aligned with the frequency.
        """
        if isinstance(item, slice):
            return self.data[self._data_slice(item)]

        elif isinstance(
            item, (list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series)
        ):
            # Handle list-like datetime-based inputs
            dates = self._convert_to_datetime_array(item)
            delta_seconds = np.array(
                [(d - self.index.start).total_seconds() for d in dates]
            )
            indices = (delta_seconds / self.index.freq_seconds).round().astype(int)
            remainders = delta_seconds % self.index.freq_seconds

            if not np.all(
                (remainders <= _ALIGNMENT_EPSILON_SECONDS)
                | (self.index.freq_seconds - remainders <= _ALIGNMENT_EPSILON_SECONDS)
            ):
                raise KeyError(
                    "One or more dates are not aligned with the index frequency."
                )
            if np.any((indices < 0) | (indices >= len(self.data))):
                raise KeyError("One or more dates are outside of the index range.")
            return self.data[indices]

        elif isinstance(item, str):
            # Handle string input
            date = pd.to_datetime(item).to_pydatetime()
            return self.data[self.index.get_position(date)]

        elif isinstance(item, datetime):
            # Handle datetime input
            return self.data[self.index.get_position(item)]

        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be datetime, slice, list, "
                "pandas Index, NumPy array, pandas Series, or string."
            )

    def __setitem__(
        self,
        item: datetime | slice | list | pd.Index | pd.Series | np.ndarray | str,
        value: float | np.ndarray,
    ):
        """
        Assign value(s) to item(s) in the series.

        Parameters:
            item (datetime | slice | list | pd.Index | pd.Series | np.ndarray | str):
                The key(s) to set.
            value (float | np.ndarray): The value(s) to assign.

        Raises:
            TypeError: If the index type is unsupported.
            ValueError: If lengths of indices and values do not match.
            KeyError: If dates are outside of the index range or not aligned with the frequency.
        """
        if isinstance(item, slice):
            data_slice = self._data_slice(item)

            # Assign values to the slice
            slice_length = len(self.data[data_slice])
            if np.isscalar(value) or slice_length == len(value):
                self.data[data_slice] = value
            else:
                raise ValueError(
                    f"Length of values ({len(value)}) does not match slice length ({slice_length})."
                )

        elif isinstance(
            item, (list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series)
        ):
            if (
                len(item) == len(self.data)
                and item[0] == self.index.start
                and item[-1] == self.index.end
            ):
                # assign in place to keep the float dtype and broadcast scalars
                self.data[:] = value
            else:
                if isinstance(value, pd.Series):
                    for idx, i in enumerate(item):
                        start = self.index.get_position(i)
                        self.data[start] = value.iloc[idx]
                elif isinstance(value, list | np.ndarray):
                    for idx, i in enumerate(item):
                        start = self.index.get_position(i)
                        self.data[start] = value[idx]
                else:
                    for i in item:
                        start = self.index.get_position(i)
                        self.data[start] = value

        elif isinstance(item, datetime | str):
            # Handle single datetime or string
            date = (
                pd.to_datetime(item).to_pydatetime() if isinstance(item, str) else item
            )
            idx = self.index.get_position(date)
            self.data[idx] = value

        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be datetime, slice, list, "
                "pandas Index, NumPy array, pandas Series, or string."
            )

    def _data_slice(self, item: slice) -> slice:
        """
        Convert a slice on the series into a slice on the data array.

        Datetime bounds are inclusive and clipped to the index range, like label-based
        slicing in pandas. Integer bounds are positional, like python slicing.

        Raises:
            TypeError: If the bounds are neither datetimes nor integers, or are mixed.
        """
        bounds = [bound for bound in (item.start, item.stop) if bound is not None]
        if all(isinstance(bound, datetime) for bound in bounds):
            start_idx = (
                max(0, self.index._get_idx_from_date(item.start))
                if item.start is not None
                else 0
            )
            stop_idx = (
                max(0, self.index._get_idx_from_date(item.stop, round_up=False) + 1)
                if item.stop is not None
                else len(self.data)
            )
            return slice(start_idx, stop_idx, item.step)
        if all(isinstance(bound, int | np.integer) for bound in bounds):
            return item
        raise TypeError(
            "FastSeries slices must use datetime or integer bounds, got "
            f"{type(item.start)} and {type(item.stop)}."
        )

    def __add__(self, other: int | float | np.ndarray):
        return self._arithmetic_operation(other, "add")

    def __sub__(self, other: int | float | np.ndarray):
        return self._arithmetic_operation(other, "sub")

    def __mul__(self, other: int | float | np.ndarray):
        return self._arithmetic_operation(other, "mul")

    def __truediv__(self, other: int | float | np.ndarray):
        return self._arithmetic_operation(other, "truediv")

    # Support for in-place operations
    def __iadd__(self, other: int | float | np.ndarray):
        self.data = self.__add__(other).data
        return self

    def __isub__(self, other: int | float | np.ndarray):
        self.data = self.__sub__(other).data
        return self

    def __imul__(self, other: int | float | np.ndarray):
        self.data = self.__mul__(other).data
        return self

    def __itruediv__(self, other: int | float | np.ndarray):
        self.data = self.__truediv__(other).data
        return self

    def __neg__(self):
        """
        Negate all values in the series.

        Returns:
            FastSeries: A new FastSeries with negated values.
        """
        result = self.copy()
        result.data = -self.data
        return result

    def __abs__(self):
        result = self.copy()
        result.data = abs(self.data)
        return result

    # Reverse Arithmetic Operations
    def __radd__(self, other: int | float | np.ndarray):
        return self.__add__(other)

    def __rsub__(self, other: int | float | np.ndarray):
        result = self.copy()
        result.data = other - self.data
        return result

    def __rmul__(self, other: int | float | np.ndarray):
        return self.__mul__(other)

    def __rtruediv__(self, other: int | float | np.ndarray):
        result = self.copy()
        result.data = other / self.data
        return result

    # Comparison Operations
    def __gt__(self, other: int | float | np.ndarray) -> np.ndarray:
        """
        Greater than comparison.

        Parameters:
            other (int | float | np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is greater.
        """
        return self.data > other

    def __lt__(self, other: int | float | np.ndarray) -> np.ndarray:
        """
        Less than comparison.

        Parameters:
            other (int | float | np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is less.
        """
        return self.data < other

    def __ge__(self, other: int | float | np.ndarray) -> np.ndarray:
        """
        Greater than or equal to comparison.

        Parameters:
            other (int | float | np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is greater or equal.
        """
        return self.data >= other

    def __le__(self, other: int | float | np.ndarray) -> np.ndarray:
        """
        Less than or equal to comparison.

        Parameters:
            other (int | float | np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is less or equal.
        """
        return self.data <= other

    def __eq__(self, other: int | float | np.ndarray) -> np.ndarray:
        """
        Equality comparison.

        Parameters:
            other (int | float | np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is equal.
        """
        return self.data == other

    def __ne__(self, other: int | float | np.ndarray) -> np.ndarray:
        """
        Not equal comparison.

        Parameters:
            other (int | float | np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is not equal.
        """
        return self.data != other

    def __len__(self) -> int:
        """
        Get the number of data points in the series.

        Returns:
            int: The length of the series.
        """
        return len(self.data)

    def __repr__(self, preview_length: int = 3) -> str:
        """
        Official string representation of the FastSeries, showing metadata and a sample of data.

        Parameters:
            preview_length (int, optional): Number of entries to show from the start and end. Defaults to 3.

        Returns:
            str: String representation of the FastSeries.
        """
        repr_string = f"FastSeries(name='{self.name}', start={self.start}, end={self.end}, freq='{self.freq}', dtype={self.dtype})\n\nData Preview:\n"
        if len(self) == 0:
            return repr_string + "[Empty Series]"

        if len(self.index.get_date_list()) <= 2 * preview_length:
            preview_str = "\n".join(
                f"{date}: {value}"
                for date, value in zip(self.index.get_date_list(), self.data)
            )
        else:
            first_dates = self.index.get_date_list()[:preview_length]
            last_dates = self.index.get_date_list()[-preview_length:]
            first_str = "\n".join(
                f"{date}: {value}"
                for date, value in zip(first_dates, self.data[:preview_length])
            )
            last_str = "\n".join(
                f"{date}: {value}"
                for date, value in zip(last_dates, self.data[-preview_length:])
            )
            preview_str = first_str + "\n...\n" + last_str

        return f"{repr_string}{preview_str}"

    def __str__(self) -> str:
        """
        Informal string representation of the FastSeries, identical to __repr__.

        Returns:
            str: String representation of the FastSeries.
        """
        return self.__repr__()

    # Aggregation Methods
    def mean(self) -> float:
        """
        Calculate the mean of the series.

        Returns:
            float: The mean value.
        """
        return self.data.mean()

    def sum(self) -> float:
        """
        Calculate the sum of the series.

        Returns:
            float: The sum of all values.
        """
        return self.data.sum()

    def min(self) -> float:
        """
        Find the minimum value in the series.

        Returns:
            float: The minimum value.
        """
        return self.data.min()

    def max(self) -> float:
        """
        Find the maximum value in the series.

        Returns:
            float: The maximum value.
        """
        return self.data.max()

    def std(self) -> float:
        """
        Calculate the standard deviation of the series.

        Returns:
            float: The standard deviation.
        """
        return self.data.std()

    def median(self) -> float:
        """
        Calculate the median of the series.

        Returns:
            float: The median value.
        """
        return np.median(self.data)

    def copy(self, deep: bool = False):
        """
        Create a copy of the FastSeries.

        Parameters:
            deep (bool, optional): If True, perform a deep copy of the data array. Defaults to False.

        Returns:
            FastSeries: A new FastSeries instance with copied data and metadata.
        """
        copied_data = self._data.copy() if deep else self._data.view()
        return FastSeries(
            index=self.index,
            value=copied_data,
            name=self.name,
        )

    def window(
        self, center: int | datetime, length: int, direction: str = "forward"
    ) -> np.ndarray:
        """
        Extract a fixed-length window, wrapping around the ends.

        Parameters
        ----------
        center : int or datetime
            If int: treated as position in the array (can be negative or beyond len).
            If datetime: converted to its integer index.
        length : int
            Number of points to return.
        direction : {'forward','backward'}, default 'forward'
            Whether to go from center → center+length or center-length → center.
        """
        # 1) compute the "raw" center index
        if isinstance(center, datetime):
            raw_center = self.index._get_idx_from_date(center)
        else:
            raw_center = center

        # 2) build the raw positions (may lie outside [0, len))
        if direction == "forward":
            raw_ix = np.arange(raw_center, raw_center + length)
        elif direction == "backward":
            raw_ix = np.arange(raw_center - length + 1, raw_center + 1)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

        # 3) wrap them into [0, len) and take
        wrapped = raw_ix % len(self._data)

        return np.take(self._data, wrapped, mode="wrap")

    def as_df(
        self, name: str = None, start: datetime = None, end: datetime = None
    ) -> pd.DataFrame:
        """
        Convert the FastSeries to a pandas DataFrame.

        Parameters:
            name (str | None): Name of the DataFrame column. Defaults to None.
            start (datetime | None): Start datetime for the DataFrame. Defaults to None.
            end (datetime | None): End datetime for the DataFrame. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame representation of the series.
        """
        data_slice = self[start:end]
        index = pd.to_datetime(self.index.get_date_list(start, end))
        return pd.DataFrame(
            data_slice, index=index, columns=[name if name else self.name]
        )

    def as_pd_series(
        self, name: str = None, start: datetime = None, end: datetime = None
    ) -> pd.Series:
        """
        Convert the FastSeries to a pandas Series.

        Parameters:
            name (str | None): Name of the Series. Defaults to None.
            start (datetime | None): Start datetime for the Series. Defaults to None.
            end (datetime | None): End datetime for the Series. Defaults to None.

        Returns:
            pd.Series: pandas Series representation of the FastSeries.
        """
        # Slice the data within the specified range
        data_slice = self[start:end]
        # Generate the corresponding index
        index = pd.to_datetime(self.index.get_date_list(start, end))
        # Create and return the pandas Series
        return pd.Series(data_slice, index=index, name=name if name else self.name)

    @staticmethod
    def from_pandas_series(series: pd.Series):
        """
        Create a FastSeries from a pandas Series.

        Parameters:
            series (pd.Series): The pandas Series to convert.

        Returns:
            FastSeries: The converted FastSeries object.

        Raises:
            ValueError: If the series has fewer than two index entries to infer frequency or frequency cannot be inferred.
        """
        if series.empty:
            raise ValueError("Cannot create FastSeries from an empty pandas Series.")

        freq = pd.infer_freq(series.index)
        if freq is None:
            raise ValueError("Cannot infer frequency from the series index.")

        if freq.isalpha():  # Ensure numeric format for frequency
            freq = f"1{freq}"

        index = FastIndex(
            start=series.index[0].to_pydatetime(),
            end=series.index[-1].to_pydatetime(),
            freq=freq,
        )
        return FastSeries(
            index=index,
            value=series.values,
            name=series.name or "",
        )

    def __iter__(self):
        """
        Make FastSeries iterable by iterating over the stored data.

        Yields:
            The elements of the series data (e.g., float, tensor).
        """
        return iter(self._data)

    # Helper method to check index alignment
    def _index_aligned_with(self, other: "FastSeries") -> bool:
        """
        Check if this series is aligned with another FastSeries.

        Parameters:
            other (FastSeries): The other series to check alignment with.

        Returns:
            bool: True if the indices match, False otherwise.
        """
        aligned = (
            self.start == other.start
            and self.end == other.end
            and self.freq == other.freq
            and len(self.data) == len(other.data)
        )
        if not aligned:
            print(  # Replace with logging if needed
                f"Indices are not aligned:\n"
                f"Self: start={self.start}, end={self.end}, freq={self.freq}, len={len(self.data)}\n"
                f"Other: start={other.start}, end={other.end}, freq={other.freq}, len={len(other.data)}"
            )
        return aligned

    def _convert_to_datetime_array(
        self, item: list | pd.Index | pd.Series | np.ndarray
    ) -> np.ndarray:
        """
        Convert input to a NumPy array of datetime objects.

        Parameters:
            item (list | pd.Index | pd.Series | np.ndarray):
                A collection of datetimes (e.g., list, pandas Index, Series, or NumPy array).

        Returns:
            np.ndarray: Array of datetime objects.

        Raises:
            ValueError: If the input cannot be converted to datetime.
        """
        try:
            if isinstance(item, pd.Series):
                if is_datetime64_any_dtype(item.index):
                    item = item.index
                else:
                    item = item.values

            return pd.to_datetime(item).to_pydatetime()
        except Exception as e:
            raise ValueError(
                f"Cannot convert {type(item)} to a NumPy array of datetime objects. Ensure the input is "
                f"a list, pandas Index, Series, or NumPy array of datetimes. Original error: {e}"
            )

    # Helper for arithmetic operations
    def _arithmetic_operation(self, other: int | float | np.ndarray, op: str):
        """
        Perform an arithmetic operation on the series.

        Parameters:
            other (int | float | np.ndarray | FastSeries): The value(s) to operate on.
            op (str): The operation to perform ('add', 'sub', 'mul', 'truediv').

        Returns:
            FastSeries: A new FastSeries with the result of the operation.

        Raises:
            ValueError: If the indices do not align for FastSeries.
            TypeError: If the `other` type is unsupported.
        """
        result = self.copy()
        if isinstance(other, int | float | np.number | np.ndarray):
            result.data = getattr(self.data, f"__{op}__")(other)
        elif isinstance(other, FastSeries):
            if self._index_aligned_with(other):
                result.data = getattr(self.data, f"__{op}__")(other.data)
            else:
                raise ValueError(f"Cannot perform {op}: Series indices do not match")
        else:
            raise TypeError(f"Unsupported type for {op}: {type(other)}")
        return result
