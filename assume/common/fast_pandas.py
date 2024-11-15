# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd


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

    @lru_cache(maxsize=100)
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

        start_idx = self.get_idx_from_date(start or self.start)
        end_idx = self.get_idx_from_date(end or self.end) + 1
        return self._date_list[start_idx:end_idx]

    @lru_cache(maxsize=1000)
    def get_idx_from_date(self, date: datetime) -> int:
        """
        Convert a datetime to its corresponding index in the range.

        Parameters:
            date (datetime): The datetime to convert.

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

        if remainder > self.tolerance_seconds and remainder < (
            self.freq_seconds - self.tolerance_seconds
        ):
            raise ValueError(
                f"Date {date} is not aligned with frequency {self.freq_seconds} seconds. "
                f"Allowed tolerance: {self.tolerance_seconds} seconds."
            )

        return round(delta_seconds / self.freq_seconds)

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
                self.get_idx_from_date(item.start)
                if isinstance(item.start, datetime)
                else item.start or 0
            )
            stop_idx = (
                self.get_idx_from_date(item.stop) + 1
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
                raise ValueError("Slice resulted in an empty range")

            return FastIndex(
                start=sliced_dates[0], end=sliced_dates[-1], freq=self._freq
            )

        else:
            raise TypeError("Index must be an integer or a slice")

    def __contains__(self, date: datetime) -> bool:
        """
        Check if a datetime is within the index range and aligned with the frequency.

        Parameters:
            date (datetime): The datetime to check.

        Returns:
            bool: True if the datetime is in the index range and aligned; False otherwise.
        """
        if self.start > date or self.end < date:
            return False
        try:
            self.get_idx_from_date(date)
            return True
        except ValueError:
            return False

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

    def __len__(self) -> int:
        """Return the number of datetime points in the index."""
        return self._count

    def __repr__(self) -> str:
        """Return a string representation of the FastIndex, including metadata and a date preview."""
        preview_length = 3  # Show first and last 3 dates
        dates_preview = (
            self.get_date_list()[:preview_length]
            + self.get_date_list()[-preview_length:]
        )
        preview_str = ", ".join(
            date.strftime("%Y-%m-%d %H:%M:%S") for date in dates_preview
        )

        metadata = (
            f"FastIndex(start={self.start}, end={self.end}, "
            f"freq='{self.freq}', dtype=datetime64[ns])"
        )
        return f"{metadata}\nDates Preview: [{preview_str}]{'...' if len(self) > 2 * preview_length else ''}"

    def __str__(self) -> str:
        """Return an informal string representation of the FastIndex."""
        return self.__repr__()

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


class FastSeries:
    """
    A fast, memory-efficient replacement for pandas Series with a FastIndex.

    This class leverages NumPy arrays for data storage to enhance performance
    during market simulations. It supports lazy initialization, vectorized
    operations, and partial compatibility with pandas Series for ease of use.
    """

    def __init__(self, index: FastIndex, value=0.0, name: str = ""):
        """
        Initialize the FastSeries.

        Parameters:
            index (FastIndex): The datetime index.
            value (scalar or array-like, optional): Initial value(s) for the data.
            name (str, optional): Name of the series.
        """
        self._index = index
        self._data = None  # Private attribute for data
        self.loc = self  # Allow adjusting loc as well
        self.at = self
        self._name = name

        self.init_data(value)

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
    def tolerance_seconds(self) -> int:
        """Get the tolerance in seconds for date alignment."""
        return self._index.tolerance_seconds

    @property
    def name(self) -> str:
        """Get the name of the series."""
        return self._name

    @property
    def data(self) -> np.ndarray:
        """
        Access the underlying data array with lazy initialization.

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
        self._data = value

    def init_data(self, value=None):
        """
        Initialize the data array if it's not already initialized.

        Parameters:
            value (scalar or array-like, optional): Initial value(s) for the data.
        """
        if self._data is None:
            self._data = self.get_numpy_date_range(
                self.index.start, self.index.end, self.index.freq, value
            )

    def __getitem__(self, item):
        """
        Retrieve item(s) from the series.

        Parameters:
            item (datetime, slice, list, pd.Index, pd.Series, np.ndarray, str): The key(s) to retrieve.

        Returns:
            float or np.ndarray: The retrieved value(s).

        Raises:
            TypeError: If the index type is unsupported.
            ValueError: If dates are not aligned within tolerance.
        """

        if isinstance(item, slice):
            # Get the starting index
            start_idx = (
                max(0, self.index.get_idx_from_date(item.start))
                if item.start and item.start >= self.index.start
                else 0
            )

            # Adjust stop_idx to include the single point when start and stop are the same
            if item.start == item.stop:
                stop_idx = start_idx + 1
            else:
                stop_idx = (
                    min(len(self.data), self.index.get_idx_from_date(item.stop) + 1)
                    if item.stop and item.stop <= self.index.end
                    else len(self.data)
                )

            # Return a new FastSeries for the sliced data
            sliced_data = self.data[start_idx:stop_idx]
            return sliced_data

        elif isinstance(
            item, list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series
        ):
            # Extract dates from the item
            dates = item.index if isinstance(item, pd.Series) else item
            # Convert to NumPy array of datetime objects
            dates = pd.to_datetime(dates).to_pydatetime()
            # Vectorized calculation of delta seconds
            delta_seconds = np.array(
                [(d - self.index.start).total_seconds() for d in dates]
            )
            indices = delta_seconds / self.index.freq_seconds
            remainders = delta_seconds % self.index.freq_seconds
            # Check if all remainders are within tolerance
            if not np.all(remainders <= self.index.tolerance_seconds):
                raise ValueError(
                    "One or more dates are not aligned with frequency within tolerance."
                )
            indices = indices.astype(int)
            return self.data[indices]

        elif isinstance(item, str):
            # Attempt to parse string to datetime
            date = pd.to_datetime(item).to_pydatetime()
            return self.data[self.index.get_idx_from_date(date)]

        elif isinstance(item, datetime):
            return self.data[self.index.get_idx_from_date(item)]

        else:
            raise TypeError(f"Unsupported index type: {type(item)}")

    def __setitem__(self, item, value):
        """
        Assign value(s) to item(s) in the series.

        Parameters:
            item (datetime, slice, list, pd.Index, pd.Series, np.ndarray, str): The key(s) to set.
            value (float or array-like): The value(s) to assign.

        Raises:
            TypeError: If the index type is unsupported.
            ValueError: If lengths of indices and values do not match or dates are not aligned within tolerance.
        """

        if isinstance(item, slice):
            # Handle slicing, including negative indices
            start_idx = (
                self.index.get_idx_from_date(item.start)
                if isinstance(item.start, datetime)
                else (
                    len(self.data) + item.start
                    if item.start is not None and item.start < 0
                    else item.start or 0
                )
            )
            stop_idx = (
                self.index.get_idx_from_date(item.stop) + 1
                if isinstance(item.stop, datetime)
                else (
                    len(self.data) + item.stop
                    if item.stop is not None and item.stop < 0
                    else len(self.data)
                )
            )
            self.data[start_idx:stop_idx] = value

        elif isinstance(
            item, list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series
        ):
            # Extract dates from the item
            dates = item.index if isinstance(item, pd.Series) else item
            dates = pd.to_datetime(dates).to_pydatetime()
            # Vectorized calculation of delta seconds
            delta_seconds = np.array(
                [(d - self.index.start).total_seconds() for d in dates]
            )
            indices = delta_seconds / self.index.freq_seconds
            remainders = delta_seconds % self.index.freq_seconds
            if not np.all(remainders <= self.index.tolerance_seconds):
                raise ValueError(
                    "One or more dates are not aligned with frequency within tolerance."
                )
            indices = indices.astype(int)

            if isinstance(item, pd.Series):
                values = item.values
                if len(indices) != len(values):
                    raise ValueError(
                        f"Length of values ({len(values)}) does not match number of indices ({len(indices)})."
                    )
                self.data[indices] = values
            else:
                if np.isscalar(value):
                    self.data[indices] = value
                else:
                    value = np.asarray(value)
                    if value.shape[0] != len(indices):
                        raise ValueError(
                            f"Length of value array ({value.shape[0]}) does not match number of indices ({len(indices)})."
                        )
                    self.data[indices] = value
        else:
            # Assume item is a single datetime or string
            if isinstance(item, datetime):
                idx = self.index.get_idx_from_date(item)
            else:
                date = pd.to_datetime(item).to_pydatetime()
                idx = self.index.get_idx_from_date(date)
            self.data[idx] = value

    def get_numpy_date_range(
        self, start: datetime, end: datetime, freq: timedelta, value=None
    ) -> np.ndarray:
        """
        Create a NumPy array filled with a specific value for the date range.

        Parameters:
            start (datetime): Start datetime.
            end (datetime): End datetime.
            freq (timedelta): Frequency.
            value (scalar, optional): Value to fill the array with.

        Returns:
            np.ndarray: NumPy array with the specified values.
        """
        if value is None:
            value = 0.0
        count = self.index.get_idx_from_date(end) + 1
        return np.full(count, value, dtype=np.float64)

    def as_df(
        self, name: str = None, start: datetime = None, end: datetime = None
    ) -> pd.DataFrame:
        """
        Convert the FastSeries to a pandas DataFrame.

        Parameters:
            name (str, optional): Name of the DataFrame column.
            start (datetime, optional): Start datetime for the DataFrame.
            end (datetime, optional): End datetime for the DataFrame.

        Returns:
            pd.DataFrame: DataFrame representation of the series.
        """
        data_slice = self[start:end]
        index = pd.to_datetime(self.index.get_date_list(start, end))
        return pd.DataFrame(
            data_slice, index=index, columns=[name if name else self.name]
        )

    @staticmethod
    def from_series(series: pd.Series) -> "FastSeries":
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

        # Infer the frequency from the index
        freq = pd.infer_freq(series.index)
        if freq is None:
            raise ValueError("Cannot infer frequency from the series index.")

        # Ensure freq is in a format that includes a number
        if freq.isalpha():  # If freq is something like "D" or "M" without a number
            freq = f"1{freq}"  # Prepend '1' to make it a valid timedelta string

        index = FastIndex(
            start=series.index[0].to_pydatetime(),
            end=series.index[-1].to_pydatetime(),
            freq=freq,
        )
        return FastSeries(
            index=index,
            value=series.values,
            name=series.name,
        )

    # Arithmetic Operations in FastSeries
    def __add__(self, other):
        """
        Add a scalar, array, or another FastSeries to this series.

        Parameters:
            other (float, np.ndarray, or FastSeries): The value(s) to add.

        Returns:
            FastSeries: The result of the addition.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data + other
        elif isinstance(other, FastSeries):  # Handle another FastSeries
            if self.index_aligned_with(other):
                result.data = self.data + other.data
            else:
                raise ValueError("Series indices do not match for addition")
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        return result

    def __sub__(self, other):
        """
        Subtract a scalar, array, or another FastSeries from this series.

        Parameters:
            other (float, np.ndarray, or FastSeries): The value(s) to subtract.

        Returns:
            FastSeries: The result of the subtraction.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data - other
        elif isinstance(other, FastSeries):
            if self.index_aligned_with(other):
                result.data = self.data - other.data
            else:
                raise ValueError("Series indices do not match for subtraction")
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return result

    def __truediv__(self, other):
        """
        Divide this series by a scalar, array, or another FastSeries.

        Parameters:
            other (float, np.ndarray, or FastSeries): The value(s) to divide by.

        Returns:
            FastSeries: The result of the division.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data / other
        elif isinstance(other, FastSeries):
            if self.index_aligned_with(other):
                result.data = self.data / other.data
            else:
                raise ValueError("Series indices do not match for division")
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        return result

    def __mul__(self, other):
        """
        Multiply this series by a scalar, array, or another FastSeries.

        Parameters:
            other (float, np.ndarray, or FastSeries): The value(s) to multiply by.

        Returns:
            FastSeries: The result of the multiplication.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data * other
        elif isinstance(other, FastSeries):
            if self.index_aligned_with(other):
                result.data = self.data * other.data
            else:
                raise ValueError("Series indices do not match for multiplication")
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
        return result

    def __neg__(self):
        """
        Negate all values in the series.

        Returns:
            FastSeries: A new FastSeries with negated values.
        """
        result = self.copy()
        result.data = -self.data
        return result

    # Helper method to check index alignment
    def index_aligned_with(self, other):
        """
        Check if this series is aligned with another FastSeries.

        Parameters:
            other (FastSeries): The other series to check alignment with.

        Returns:
            bool: True if the indices match, False otherwise.
        """
        return (
            self.start == other.start
            and self.end == other.end
            and self.freq == other.freq
            and len(self.data) == len(other.data)
        )

    # Reverse Arithmetic Operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        result = self.copy()
        result.data = other - self.data
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    # In-place Arithmetic Operations
    def __iadd__(self, other):
        self.data += other
        return self

    def __isub__(self, other):
        self.data -= other
        return self

    def __imul__(self, other):
        self.data *= other
        return self

    def __itruediv__(self, other):
        self.data /= other
        return self

    # Comparison Operations
    def __gt__(self, other):
        """
        Greater than comparison.

        Parameters:
            other (float or np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is greater.
        """
        return self.data > other

    def __lt__(self, other):
        """
        Less than comparison.

        Parameters:
            other (float or np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is less.
        """
        return self.data < other

    def __ge__(self, other):
        """
        Greater than or equal to comparison.

        Parameters:
            other (float or np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is greater or equal.
        """
        return self.data >= other

    def __le__(self, other):
        """
        Less than or equal to comparison.

        Parameters:
            other (float or np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is less or equal.
        """
        return self.data <= other

    def __eq__(self, other):
        """
        Equality comparison.

        Parameters:
            other (float or np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is equal.
        """
        return self.data == other

    def __ne__(self, other):
        """
        Not equal comparison.

        Parameters:
            other (float or np.ndarray): The value(s) to compare against.

        Returns:
            np.ndarray: Boolean array where True indicates the series value is not equal.
        """
        return self.data != other

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

    # Copy Methods
    def copy(self, deep: bool = False) -> "FastSeries":
        """
        Create a copy of the FastSeries.

        Parameters:
            deep (bool, optional): If True, perform a deep copy. Defaults to False.

        Returns:
            FastSeries: The copied FastSeries.
        """
        if deep:
            copied_data = self._data.copy()
        else:
            copied_data = self._data.view()
        return FastSeries(
            index=self.index,
            value=copied_data,
            name=self.name,
        )

    def __len__(self) -> int:
        """
        Get the length of the series.

        Returns:
            int: Number of data points.
        """
        return self.index._count

    def __repr__(self):
        """
        Official string representation of the FastSeries, showing key metadata and sample data.
        """
        # Show a small preview of the data (similar to pandas Series)
        preview_length = 10  # number of elements to preview
        data_preview = self.data[:preview_length]
        dates_preview = self.index.get_date_list()[:preview_length]

        # Format preview display
        preview_str = "\n".join(
            f"{date}: {value}" for date, value in zip(dates_preview, data_preview)
        )

        # Metadata summary
        metadata = (
            f"FastSeries(name='{self.name}', start={self.start}, end={self.end}, "
            f"freq='{self.freq}', dtype={self.dtype})"
        )

        # Return full string
        return f"{metadata}\n\nData Preview:\n{preview_str}\n..."

    def __str__(self):
        """
        Informal string representation of the FastSeries, similar to __repr__.
        """
        return self.__repr__()

    @property
    def dtype(self):
        """
        Get the data type of the series.

        Returns:
            dtype: The data type of the underlying NumPy array.
        """
        return self.data.dtype if self._data is not None else None

    @staticmethod
    def make_series(index_series, value=0.0, name: str = ""):
        """
        Create a FastSeries using another FastSeries as the index,
        initializing all data points with the specified value.

        Parameters:
            index_series (FastSeries): The FastSeries to use as the index.
            value (scalar or array-like, optional): The value(s) to initialize the data with. Defaults to 0.0.
            name (str, optional): The name of the new series. Defaults to an empty string.

        Returns:
            FastSeries: A new FastSeries with the same index as `index_series` and data initialized to `value`.
        """
        return FastSeries(
            index=index_series.index,
            value=value,
            name=name if name else index_series.name,
        )
