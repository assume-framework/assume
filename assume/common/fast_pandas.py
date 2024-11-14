# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class FastDatetimeIndex:
    """
    A fast, memory-efficient datetime index similar to pandas DatetimeIndex.

    This class manages a range of datetime objects with a specified frequency.
    It provides methods for indexing, slicing, and checking membership with
    tolerance for alignment.
    """

    def __init__(
        self,
        start: datetime,
        end: datetime = None,
        freq: str = "1h",
        periods: int = None,
    ):
        """
        Initialize the FastDatetimeIndex.

        Parameters:
            start (datetime): Start datetime.
            end (datetime, optional): End datetime.
            freq (str or timedelta, optional): Frequency of the index. Defaults to "1h".
            periods (int, optional): Number of periods to generate. Defaults to None.
        """
        self._start = start if isinstance(start, datetime) else pd.to_datetime(start)

        # Ensure freq is a timedelta
        if isinstance(freq, str):
            # Ensure that frequency string includes a numeric value if necessary
            if freq.isalpha():  # If freq is something like "H" or "D" without a number
                freq = f"1{freq}"  # Prepend '1' to make it a valid timedelta string
            self._freq = pd.to_timedelta(freq)
        elif isinstance(freq, timedelta):
            self._freq = freq
        else:
            raise TypeError("Frequency must be a string or timedelta")

        if end is None and periods is None:
            raise ValueError("Either 'end' or 'periods' must be specified")

        # Calculate the end date based on the number of periods
        if periods is not None:
            self._end = self._start + (periods - 1) * self._freq
        else:
            self._end = end if isinstance(end, datetime) else pd.to_datetime(end)

        self._freq_seconds = self._freq.total_seconds()  # Precompute total seconds
        self._tolerance_seconds = 1  # Tolerance of 1 second

        # Precompute the total number of periods
        self._count = self.get_idx_from_date(self._end) + 1
        # Generate and store the full date range as a list of datetime objects
        self._date_list = self.get_date_list()

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

    def get_date_list(self, start: datetime = None, end: datetime = None) -> list:
        """
        Generate a list of datetime objects within the specified range.

        Parameters:
            start (datetime, optional): Start datetime.
            end (datetime, optional): End datetime.

        Returns:
            list of datetime: The list of datetime indices.
        """
        start = max(start, self.start) if start else self.start
        end = min(end, self.end) if end else self.end
        start_idx = self.get_idx_from_date(start)
        end_idx = self.get_idx_from_date(end) + 1
        date_range = [
            self.start + timedelta(seconds=i * self.freq_seconds)
            for i in range(start_idx, end_idx)
        ]
        return date_range

    def get_idx_from_date(self, date: datetime) -> int:
        """
        Convert a datetime to its corresponding index.

        Parameters:
            date (datetime): The datetime to convert.

        Returns:
            int: The corresponding index.

        Raises:
            KeyError: If the date is None.
            ValueError: If the date is not aligned with the frequency within tolerance.
        """
        if date is None:
            raise KeyError("Date cannot be None")
        delta_seconds = (date - self.start).total_seconds()
        idx = delta_seconds / self.freq_seconds
        remainder = delta_seconds % self.freq_seconds
        if remainder > self.tolerance_seconds:
            raise ValueError(
                f"Date {date} is not aligned with frequency {self.freq} within tolerance of {self.tolerance_seconds} second(s)."
            )
        return int(idx)

    def __getitem__(self, item):
        """
        Retrieve datetime(s) based on the specified index or slice.

        Parameters:
            item (int, slice, datetime): The index or slice to retrieve.

        Returns:
            datetime or FastDatetimeIndex: A single datetime object or a new FastDatetimeIndex.
        """
        if isinstance(item, int):
            # Return a specific datetime at the position `item`
            return self._date_list[item]
        elif isinstance(item, slice):
            # Handle slice of date range
            start_date = item.start if item.start else self._date_list[0]
            end_date = item.stop if item.stop else self._date_list[-1]
            sliced_dates = self.get_date_list(start=start_date, end=end_date)
            # Return a new FastDatetimeIndex for the sliced range
            return FastDatetimeIndex(
                start=sliced_dates[0], end=sliced_dates[-1], freq=self.freq
            )
        else:
            raise TypeError("Index must be an integer or slice")

    def __contains__(self, date: datetime) -> bool:
        """
        Check if a datetime is within the index and aligned with the frequency within tolerance.

        Parameters:
            date (datetime): The datetime to check.

        Returns:
            bool: True if contained and aligned within tolerance, False otherwise.
        """
        if self.start > date or self.end < date:
            return False
        try:
            self.get_idx_from_date(date)
            return True
        except ValueError:
            return False

    def __len__(self) -> int:
        """
        Get the number of datetime points in the index.

        Returns:
            int: Number of datetime points.
        """
        return self._count

    def __repr__(self):
        """
        Official string representation of the FastDatetimeIndex, showing key metadata and sample dates.
        """
        # Show a small preview of the dates (similar to pandas DatetimeIndex)
        preview_length = 10  # number of elements to preview
        dates_preview = self.get_date_list()[:preview_length]

        # Format preview display
        preview_str = ", ".join(
            date.strftime("%Y-%m-%d %H:%M:%S") for date in dates_preview
        )

        # Metadata summary
        metadata = (
            f"FastDatetimeIndex(start={self.start}, end={self.end}, "
            f"freq='{self.freq}', dtype=datetime64[ns])"
        )

        # Return full string
        return f"{metadata}\nDates Preview: [{preview_str}]{'...' if len(self) > preview_length else ''}"

    def __str__(self):
        """
        Informal string representation of the FastDatetimeIndex, similar to __repr__.
        """
        return self.__repr__()


class FastDatetimeSeries:
    """
    A fast, memory-efficient replacement for pandas Series with a FastDatetimeIndex.

    This class leverages NumPy arrays for data storage to enhance performance
    during market simulations. It supports lazy initialization, vectorized
    operations, and partial compatibility with pandas Series for ease of use.
    """

    def __init__(self, index: FastDatetimeIndex, value=None, name: str = ""):
        """
        Initialize the FastDatetimeSeries.

        Parameters:
            index (FastDatetimeIndex): The datetime index.
            value (scalar or array-like, optional): Initial value(s) for the data.
            name (str, optional): Name of the series.
        """
        self._index = index
        self._data = None  # Private attribute for data
        self.loc = self  # Allow adjusting loc as well
        self.at = self
        self._name = name

        if value is not None:
            self.init_data(value)

    @property
    def index(self) -> FastDatetimeIndex:
        """Get the FastDatetimeIndex of the series."""
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
        if self._data is None:
            self.init_data()
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

            # Return a new FastDatetimeSeries for the sliced data
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
        Convert the FastDatetimeSeries to a pandas DataFrame.

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
    def from_series(series: pd.Series) -> "FastDatetimeSeries":
        """
        Create a FastDatetimeSeries from a pandas Series.

        Parameters:
            series (pd.Series): The pandas Series to convert.

        Returns:
            FastDatetimeSeries: The converted FastDatetimeSeries object.

        Raises:
            ValueError: If the series has fewer than two index entries to infer frequency or frequency cannot be inferred.
        """
        if series.empty:
            raise ValueError(
                "Cannot create FastDatetimeSeries from an empty pandas Series."
            )

        # Infer the frequency from the index
        freq = pd.infer_freq(series.index)
        if freq is None:
            raise ValueError("Cannot infer frequency from the series index.")

        # Ensure freq is in a format that includes a number
        if freq.isalpha():  # If freq is something like "D" or "M" without a number
            freq = f"1{freq}"  # Prepend '1' to make it a valid timedelta string

        index = FastDatetimeIndex(
            start=series.index[0].to_pydatetime(),
            end=series.index[-1].to_pydatetime(),
            freq=freq,
        )
        return FastDatetimeSeries(
            index=index,
            value=series.values,
            name=series.name,
        )

    # Arithmetic Operations in FastDatetimeSeries
    def __add__(self, other):
        """
        Add a scalar, array, or another FastDatetimeSeries to this series.

        Parameters:
            other (float, np.ndarray, or FastDatetimeSeries): The value(s) to add.

        Returns:
            FastDatetimeSeries: The result of the addition.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data + other
        elif isinstance(other, FastDatetimeSeries):  # Handle another FastDatetimeSeries
            if self.index_aligned_with(other):
                result.data = self.data + other.data
            else:
                raise ValueError("Series indices do not match for addition")
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        return result

    def __sub__(self, other):
        """
        Subtract a scalar, array, or another FastDatetimeSeries from this series.

        Parameters:
            other (float, np.ndarray, or FastDatetimeSeries): The value(s) to subtract.

        Returns:
            FastDatetimeSeries: The result of the subtraction.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data - other
        elif isinstance(other, FastDatetimeSeries):
            if self.index_aligned_with(other):
                result.data = self.data - other.data
            else:
                raise ValueError("Series indices do not match for subtraction")
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return result

    def __truediv__(self, other):
        """
        Divide this series by a scalar, array, or another FastDatetimeSeries.

        Parameters:
            other (float, np.ndarray, or FastDatetimeSeries): The value(s) to divide by.

        Returns:
            FastDatetimeSeries: The result of the division.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data / other
        elif isinstance(other, FastDatetimeSeries):
            if self.index_aligned_with(other):
                result.data = self.data / other.data
            else:
                raise ValueError("Series indices do not match for division")
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        return result

    def __mul__(self, other):
        """
        Multiply this series by a scalar, array, or another FastDatetimeSeries.

        Parameters:
            other (float, np.ndarray, or FastDatetimeSeries): The value(s) to multiply by.

        Returns:
            FastDatetimeSeries: The result of the multiplication.
        """
        result = self.copy()
        if isinstance(other, int | float | np.ndarray):
            result.data = self.data * other
        elif isinstance(other, FastDatetimeSeries):
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
            FastDatetimeSeries: A new FastDatetimeSeries with negated values.
        """
        result = self.copy()
        result.data = -self.data
        return result

    # Helper method to check index alignment
    def index_aligned_with(self, other):
        """
        Check if this series is aligned with another FastDatetimeSeries.

        Parameters:
            other (FastDatetimeSeries): The other series to check alignment with.

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
    def copy(self, deep: bool = False) -> "FastDatetimeSeries":
        """
        Create a copy of the FastDatetimeSeries.

        Parameters:
            deep (bool, optional): If True, perform a deep copy. Defaults to False.

        Returns:
            FastDatetimeSeries: The copied FastDatetimeSeries.
        """
        if deep:
            copied_data = deepcopy(self._data) if self._data is not None else None
        else:
            copied_data = self._data.copy() if self._data is not None else None
        return FastDatetimeSeries(
            index=self.index,
            value=copied_data,
            name=self.name,
        )

    def deepcopy(self) -> "FastDatetimeSeries":
        """
        Create a deep copy of the FastDatetimeSeries.

        Returns:
            FastDatetimeSeries: The deep-copied FastDatetimeSeries.
        """
        return self.copy(deep=True)

    def copy_empty(self, value: float = 0.0, name: str = "") -> "FastDatetimeSeries":
        """
        Create a new FastDatetimeSeries with the same time index but with data initialized to a specified value.

        Parameters:
            value (float, optional): The value to initialize the data array with. Defaults to 0.0.
            name (str, optional): The name of the new series. Defaults to an empty string.

        Returns:
            FastDatetimeSeries: A new instance with initialized data.
        """
        return FastDatetimeSeries(
            index=self.index,
            value=value,
            name=name if name else self.name,
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
        Official string representation of the FastDatetimeSeries, showing key metadata and sample data.
        """
        # Initialize data if it's not already done
        self.init_data()

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
            f"FastDatetimeSeries(name='{self.name}', start={self.start}, end={self.end}, "
            f"freq='{self.freq}', dtype={self.dtype})"
        )

        # Return full string
        return f"{metadata}\n\nData Preview:\n{preview_str}\n..."

    def __str__(self):
        """
        Informal string representation of the FastDatetimeSeries, similar to __repr__.
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
        Create a FastDatetimeSeries using another FastDatetimeSeries as the index,
        initializing all data points with the specified value.

        Parameters:
            index_series (FastDatetimeSeries): The FastDatetimeSeries to use as the index.
            value (scalar or array-like, optional): The value(s) to initialize the data with. Defaults to 0.0.
            name (str, optional): The name of the new series. Defaults to an empty string.

        Returns:
            FastDatetimeSeries: A new FastDatetimeSeries with the same index as `index_series` and data initialized to `value`.
        """
        return FastDatetimeSeries(
            index=index_series.index,
            value=value,
            name=name if name else index_series.name,
        )
