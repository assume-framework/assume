# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

try:
    import torch as th
except ImportError:
    th = None


class FastIndex:
    """
    A fast, memory-efficient datetime index similar to pandas DatetimeIndex.

    This class manages a range of datetime objects with a specified frequency,
    providing efficient indexing, slicing (both integer and datetime-based),
    and membership checking with alignment tolerance.
    """

    def __init__(
        self,
        start: datetime | str,
        end: datetime | str = None,
        freq: timedelta | str = timedelta(hours=1),
        periods: int = None,
    ):
        """
        Initialize the FastIndex.

        Parameters:
            start (datetime | str): The start datetime or its string representation.
            end (datetime | str, optional): The end datetime or its string representation. Defaults to None.
            freq (timedelta | str, optional): The frequency of the index. Can be a timedelta or pandas-style string.
                                               Defaults to timedelta(hours=1).
            periods (int, optional): Number of periods in the index. Either `end` or `periods` must be provided.
        """
        self._start = self._convert_to_datetime(start)
        if end is None and periods is None:
            raise ValueError("Either 'end' or 'periods' must be specified")

        self._freq = self._parse_frequency(freq)
        self._freq_seconds = self._freq.total_seconds()

        if periods is not None:
            self._end = self._start + (periods - 1) * self._freq
            self._count = periods
        else:
            self._end = self._convert_to_datetime(end)
            total_seconds = (self._end - self._start).total_seconds()
            self._count = int(np.floor(total_seconds / self._freq_seconds)) + 1

        self._tolerance_seconds = 1
        self._date_list = None  # Lazy-loaded

    @property
    def start(self) -> datetime:
        """Get the start datetime of the index."""
        return self._start

    @property
    def end(self) -> datetime:
        """Get the end datetime of the index."""
        return self._end

    @property
    def freq(self) -> timedelta:
        """Get the frequency of the index as a timedelta."""
        return self._freq

    @property
    def freq_seconds(self) -> float:
        """Get the frequency of the index in total seconds."""
        return self._freq_seconds

    @property
    def tolerance_seconds(self) -> int:
        """Get the tolerance in seconds for date alignment."""
        return self._tolerance_seconds

    def __getitem__(self, item: int | slice):
        """
        Retrieve datetime(s) based on the specified index or slice.

        Parameters:
            item (int | slice): Index or slice to retrieve. Slices can use integers or datetime values.

        Returns:
            datetime | FastIndex: A single datetime object or a new FastIndex for the sliced range.

        Raises:
            IndexError: If an integer index is out of range.
            TypeError: If `item` is not an integer or slice.
            ValueError: If slicing results in an empty range.
        """
        if self._date_list is None:
            self.get_date_list()

        if isinstance(item, int):
            if item < 0:
                item += len(self._date_list)
            if item < 0 or item >= len(self._date_list):
                raise IndexError("Index out of range")
            return self._date_list[item]

        elif isinstance(item, slice):
            start_idx = (
                self._get_idx_from_date(item.start)
                if isinstance(item.start, datetime)
                else item.start or 0
            )
            stop_idx = (
                self._get_idx_from_date(item.stop, round_up=False) + 1
                if isinstance(item.stop, datetime)
                else item.stop or len(self._date_list)
            )
            step = item.step or 1

            if isinstance(start_idx, int) and start_idx < 0:
                start_idx += len(self._date_list)
            if isinstance(stop_idx, int) and stop_idx < 0:
                stop_idx += len(self._date_list)

            sliced_dates = self._date_list[start_idx:stop_idx:step]
            if not sliced_dates:
                return []

            return sliced_dates

        else:
            raise TypeError("Index must be an integer or a slice")

    def __contains__(self, date: datetime) -> bool:
        """
        Check if a datetime is within the index range and aligned with the frequency.

        Parameters:
            date (datetime.datetime): The datetime to check.

        Returns:
            bool: True if the datetime is in the index range and aligned; False otherwise.
        """
        if self.start > date or self.end < date:
            return False
        try:
            self._get_idx_from_date(date)
            return True
        except ValueError:
            return False

    def __len__(self) -> int:
        """Return the number of datetime points in the index."""
        return self._count

    def __repr__(self) -> str:
        """Return a string representation of the FastIndex, including metadata and a date preview."""
        preview_length = 3  # Show first and last 3 dates
        date_list = self.get_date_list()

        def format_dates(date_range, date_format="%Y-%m-%d %H:%M:%S"):
            return ", ".join(date.strftime(date_format) for date in date_range)

        if len(date_list) <= 2 * preview_length:
            preview_str = format_dates(date_list)
        else:
            preview_str = format_dates(date_list[:preview_length]) + ", ..., "
            preview_str += format_dates(date_list[-preview_length:])

        metadata = (
            f"FastIndex(start={self.start}, end={self.end}, "
            f"freq='{self.freq}', dtype=datetime64[ns])"
        )
        return f"{metadata}\nDates Preview: [{preview_str}]"

    def __str__(self) -> str:
        """Return an informal string representation of the FastIndex."""
        return self.__repr__()

    @lru_cache(maxsize=1000)
    def get_date_list(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> list[datetime]:
        """
        Generate a list of datetime objects within the specified range.

        Parameters:
            start (datetime | None, optional): Start datetime for the subset. Defaults to the beginning of the index.
            end (datetime | None, optional): End datetime for the subset. Defaults to the end of the index.

        Returns:
            list[datetime]: A list of datetime objects representing the specified range.
        """
        if self._date_list is None:
            total_dates = np.arange(self._count) * self._freq_seconds
            self._date_list = [self._start + timedelta(seconds=s) for s in total_dates]

        start_idx = self._get_idx_from_date(start or self.start)
        end_idx = self._get_idx_from_date(end or self.end, round_up=False) + 1
        return self._date_list[start_idx:end_idx]

    def as_datetimeindex(self) -> pd.DatetimeIndex:
        """
        Convert the FastIndex to a pandas DatetimeIndex.

        Returns:
            pd.DatetimeIndex: A pandas DatetimeIndex representing the FastIndex.
        """
        # Retrieve the datetime range using get_date_list
        datetimes = self.get_date_list()
        # Convert to pandas DatetimeIndex
        return pd.DatetimeIndex(pd.to_datetime(datetimes), name="FastIndex")

    @lru_cache(maxsize=1000)
    def _get_idx_from_date(self, date: datetime, round_up: bool = True) -> int:
        """
        Convert a datetime to its corresponding index in the range.

        Parameters:
            date (datetime.datetime): The datetime to convert.

        Returns:
            int: The index of the datetime in the index range.

        Raises:
            KeyError: If the input `date` is None.
            ValueError: If the `date` is not aligned with the frequency within tolerance.
        """
        if date is None:
            raise KeyError("Date cannot be None. Please provide a valid datetime.")

        delta_seconds = (date - self.start).total_seconds()
        remainder = delta_seconds % self.freq_seconds

        if round_up and remainder > 0:
            # if there is a large remainder, we need to add to return the value of the next date as begin
            delta_seconds += self.freq_seconds

        return round(delta_seconds / self.freq_seconds)

    @staticmethod
    def _convert_to_datetime(value: datetime | str) -> datetime:
        """Convert input to datetime if it's not already."""
        return value if isinstance(value, datetime) else pd.to_datetime(value)

    @staticmethod
    def _parse_frequency(freq: timedelta | str) -> timedelta:
        """
        Parse a frequency input into a timedelta.

        Parameters:
            freq (timedelta | str): Frequency in timedelta or pandas-style string format.

        Returns:
            timedelta: The parsed frequency.

        Raises:
            TypeError: If the input type is not supported.
            ValueError: If the string format is invalid.
        """
        if isinstance(freq, timedelta):
            return freq
        if isinstance(freq, str):
            try:
                if freq.isalpha():
                    freq = f"1{freq}"
                return pd.to_timedelta(freq)
            except ValueError as e:
                raise ValueError(f"Invalid frequency string: {freq}. Error: {e}")
        raise TypeError("Frequency must be a string or timedelta")


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
        self, index: FastIndex, value: float | np.ndarray = 0.0, name: str = ""
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
        self._data = (
            np.full(count, value, dtype=np.float64)
            if isinstance(value, int | float)
            else np.array(value, dtype=np.float64)
        )

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
        if value.shape[0] != len(self.index):
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
            ValueError: If dates are not aligned within tolerance.
        """
        if isinstance(item, slice):
            # Handle slicing with datetime start/stop
            start_idx = (
                self.index._get_idx_from_date(item.start)
                if item.start is not None
                else 0
            )
            stop_idx = (
                self.index._get_idx_from_date(item.stop, round_up=False) + 1
                if item.stop is not None
                else len(self.data)
            )
            return self.data[start_idx:stop_idx]

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

            if not np.all(remainders <= self.index.tolerance_seconds):
                raise ValueError(
                    "One or more dates are not aligned with the index frequency."
                )
            return self.data[indices]

        elif isinstance(item, str):
            # Handle string input
            date = pd.to_datetime(item).to_pydatetime()
            return self.data[self.index._get_idx_from_date(date)]

        elif isinstance(item, datetime):
            # Handle datetime input
            return self.data[self.index._get_idx_from_date(item)]

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
            ValueError: If lengths of indices and values do not match or dates are not aligned within tolerance.
        """
        if isinstance(item, slice):
            # Handle slicing
            start_idx = (
                self.index._get_idx_from_date(item.start)
                if isinstance(item.start, datetime)
                else (
                    len(self.data) + item.start
                    if item.start is not None and item.start < 0
                    else 0
                )
            )
            stop_idx = (
                self.index._get_idx_from_date(item.stop, round_up=False) + 1
                if isinstance(item.stop, datetime)
                else (
                    len(self.data) + item.stop
                    if item.stop is not None and item.stop < 0
                    else len(self.data)
                )
            )

            # Assign values to the slice
            if np.isscalar(value) or len(self.data[start_idx:stop_idx]) == len(value):
                self.data[start_idx:stop_idx] = value
            else:
                raise ValueError(
                    f"Length of values ({len(value)}) does not match slice length ({stop_idx - start_idx})."
                )

        elif isinstance(
            item, (list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series)
        ):
            if (
                len(item) == len(self.data)
                and item[0] == self.index.start
                and item[-1] == self.index.end
            ):
                self.data = np.array(value)
            else:
                if isinstance(value, pd.Series):
                    for idx, i in enumerate(item):
                        start = self.index._get_idx_from_date(i)
                        self.data[start] = value.iloc[idx]
                elif isinstance(value, list | np.ndarray):
                    for idx, i in enumerate(item):
                        start = self.index._get_idx_from_date(i)
                        self.data[start] = value[idx]
                else:
                    for i in item:
                        start = self.index._get_idx_from_date(i)
                        self.data[start] = value

        elif isinstance(item, datetime | str):
            # Handle single datetime or string
            date = (
                pd.to_datetime(item).to_pydatetime() if isinstance(item, str) else item
            )
            idx = self.index._get_idx_from_date(date)
            self.data[idx] = value

        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be datetime, slice, list, "
                "pandas Index, NumPy array, pandas Series, or string."
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
        if isinstance(other, int | float | np.ndarray):
            result.data = getattr(self.data, f"__{op}__")(other)
        elif isinstance(other, FastSeries):
            if self._index_aligned_with(other):
                result.data = getattr(self.data, f"__{op}__")(other.data)
            else:
                raise ValueError(f"Cannot perform {op}: Series indices do not match")
        else:
            raise TypeError(f"Unsupported type for {op}: {type(other)}")
        return result


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


class TensorFastSeries(FastSeries):
    """
    A specialized version of FastSeries designed to handle tensors.
    """

    def __init__(self, index: FastIndex, value=None, name: str = ""):
        """
        Initialize a TensorFastSeries.

        Parameters:
            index (FastIndex): The index for the series.
            value (torch.Tensor | float | None): The initial value to populate the series.
                If a scalar (e.g., 0.0) is provided, it will be converted to a tensor.
                Defaults to None.
            name (str, optional): The name of the series. Defaults to "".
        """
        super().__init__(index=index, value=None, name=name)

        # Ensure _data is initialized to hold tensors
        if value is None:
            self._data = [None for _ in range(len(index))]
        elif isinstance(value, th.Tensor):
            self._data = [value.clone() for _ in range(len(index))]
        elif isinstance(value, (int | float)):
            self._data = [th.tensor(value) for _ in range(len(index))]
        else:
            raise TypeError(
                f"Unsupported value type: {type(value)}. Must be torch.Tensor, float, or int."
            )

    def __setitem__(self, item: int | datetime | slice, value):
        """
        Assign tensor value(s) to item(s) in the series.

        Parameters:
            item (int | datetime | slice): The index or slice.
            value (th.Tensor): The tensor value(s) to assign.
        """
        if isinstance(item, int):
            if item < 0 or item >= len(self._data):
                raise IndexError(
                    f"Index {item} is out of bounds for series of length {len(self._data)}"
                )
            self._data[item] = value.clone()
        elif isinstance(item, slice):
            start_idx = item.start or 0
            stop_idx = item.stop or len(self._data)
            step = item.step or 1
            slice_length = len(range(start_idx, stop_idx, step))

            if len(value) != slice_length:
                raise ValueError(
                    f"Length of value ({len(value)}) does not match the length of the slice ({slice_length})."
                )
            for i, idx in enumerate(range(start_idx, stop_idx, step)):
                self._data[idx] = value[i].clone()
        elif isinstance(item, datetime):
            idx = self.index._get_idx_from_date(item)
            self._data[idx] = value.clone()
        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be int, slice, or datetime."
            )

    def __getitem__(self, item: int | datetime | slice):
        """
        Retrieve tensor(s) from the series.

        Parameters:
            item (int | datetime | slice): The index or slice.

        Returns:
            th.Tensor | list[th.Tensor]: The retrieved tensor(s).
        """
        if isinstance(item, int):
            if item < 0 or item >= len(self._data):
                raise IndexError(
                    f"Index {item} is out of bounds for series of length {len(self._data)}"
                )
            return self._data[item]
        elif isinstance(item, slice):
            start_idx = item.start or 0
            stop_idx = item.stop or len(self._data)
            step = item.step or 1
            return [self._data[i] for i in range(start_idx, stop_idx, step)]
        elif isinstance(item, datetime):
            idx = self.index._get_idx_from_date(item)
            return self._data[idx]
        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be int, slice, or datetime."
            )

    def copy(self, deep: bool = False):
        """
        Create a copy of the TensorFastSeries.

        Parameters:
            deep (bool): If True, perform a deep copy. Defaults to False.

        Returns:
            TensorFastSeries: A new instance with copied data.
        """
        if deep:
            copied_data = [
                tensor.clone() if tensor is not None else None for tensor in self._data
            ]
        else:
            copied_data = self._data[:]

        return TensorFastSeries(
            index=self._index,
            value=None,  # We'll manually set _data below
            name=self._name,
        )._set_data(copied_data)

    def _set_data(self, data):
        """
        Helper method to set data during initialization.

        Parameters:
            data (list[th.Tensor]): The data to set.

        Returns:
            TensorFastSeries: The modified instance.
        """
        self._data = data
        return self

    def __repr__(self) -> str:
        """
        Return a string representation of the TensorFastSeries.

        Returns:
            str: A string describing the series.
        """
        preview_length = 3  # Number of items to preview from the start and end
        total_length = len(self._data)

        if total_length == 0:
            return f"TensorFastSeries(name='{self._name}', length=0, data=[])"

        # Preview a subset of the data
        start_preview = self._data[:preview_length]
        end_preview = (
            self._data[-preview_length:] if total_length > preview_length else []
        )

        preview = (
            start_preview
            + (["..."] if total_length > 2 * preview_length else [])
            + end_preview
        )

        return (
            f"TensorFastSeries(name='{self._name}', length={total_length}, "
            f"data={preview})"
        )

    def __str__(self) -> str:
        """
        Informal string representation of the FastSeries, identical to __repr__.

        Returns:
            str: String representation of the FastSeries.
        """
        return self.__repr__()
