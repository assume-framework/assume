# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class FastDatetimeSeries:
    """
    A fast, memory-efficient replacement for pandas Series with datetime indices.

    This class leverages NumPy arrays for data storage to enhance performance
    during market simulations. It supports lazy initialization, vectorized
    operations, and partial compatibility with pandas Series for ease of use.
    """

    def __init__(
        self, start: datetime, end: datetime, freq: str, value=None, name: str = ""
    ):
        """
        Initialize the FastDatetimeSeries.

        Parameters:
            start (datetime): Start datetime.
            end (datetime): End datetime.
            freq (str): Frequency string (e.g., '1T' for 1 minute).
            value (scalar or array-like, optional): Initial value(s) for the data.
            name (str, optional): Name of the series.
        """
        self._start = start
        self._end = end
        self._freq = pd.to_timedelta(freq)
        self._freq_seconds = (
            self._freq.total_seconds()
        )  # Precompute total seconds for faster calculations
        self._tolerance_seconds = 1  # Tolerance of 1 second
        self._data = None  # Private attribute for data
        self.loc = self  # Allow adjusting loc as well
        self.at = self
        self._name = name

        if value is not None:
            self.init_data(value)

    @property
    def start(self) -> datetime:
        """Get the start datetime of the series."""
        return self._start

    @property
    def end(self) -> datetime:
        """Get the end datetime of the series."""
        return self._end

    @property
    def freq(self) -> timedelta:
        """Get the frequency of the series as a timedelta."""
        return self._freq

    @property
    def freq_seconds(self) -> float:
        """Get the frequency of the series in total seconds."""
        return self._freq_seconds

    @property
    def tolerance_seconds(self) -> int:
        """Get the tolerance in seconds for date alignment."""
        return self._tolerance_seconds

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
                self.start, self.end, self.freq, value
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
            start_idx = self.get_idx_from_date(item.start) if item.start else 0
            stop_idx = (
                self.get_idx_from_date(item.stop) + 1 if item.stop else len(self.data)
            )
            return self.data[start_idx:stop_idx]

        elif isinstance(
            item, list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series
        ):
            # Extract dates from the item
            dates = item.index if isinstance(item, pd.Series) else item
            # Convert to NumPy array of datetime objects
            dates = pd.to_datetime(dates).to_pydatetime()
            # Vectorized calculation of delta seconds
            delta_seconds = np.array([(d - self.start).total_seconds() for d in dates])
            indices = delta_seconds / self.freq_seconds
            remainders = delta_seconds % self.freq_seconds
            # Check if all remainders are within tolerance
            if not np.all(remainders <= self.tolerance_seconds):
                raise ValueError(
                    "One or more dates are not aligned with frequency within tolerance."
                )
            indices = indices.astype(int)
            return self.data[indices]

        elif isinstance(item, str):
            # Attempt to parse string to datetime
            date = pd.to_datetime(item).to_pydatetime()
            return self.data[self.get_idx_from_date(date)]

        elif isinstance(item, datetime):
            return self.data[self.get_idx_from_date(item)]

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
            start_idx = self.get_idx_from_date(item.start) if item.start else 0
            stop_idx = (
                self.get_idx_from_date(item.stop) + 1 if item.stop else len(self.data)
            )
            self.data[start_idx:stop_idx] = value

        elif isinstance(
            item, list | pd.Index | pd.DatetimeIndex | np.ndarray | pd.Series
        ):
            # Extract dates from the item
            dates = item.index if isinstance(item, pd.Series) else item
            # Convert to NumPy array of datetime objects
            dates = pd.to_datetime(dates).to_pydatetime()
            # Vectorized calculation of delta seconds
            delta_seconds = np.array([(d - self.start).total_seconds() for d in dates])
            indices = delta_seconds / self.freq_seconds
            remainders = delta_seconds % self.freq_seconds
            # Check if all remainders are within tolerance
            if not np.all(remainders <= self.tolerance_seconds):
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
            idx = (
                self.get_idx_from_date(item)
                if isinstance(item, datetime)
                else self.get_idx_from_date(pd.to_datetime(item).to_pydatetime())
            )
            self.data[idx] = value

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
        Convert a datetime to its corresponding index in the data array.

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
        count = self.get_idx_from_date(end) + 1
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
        index = pd.to_datetime(self.get_date_list(start, end))
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

        return FastDatetimeSeries(
            start=series.index[0].to_pydatetime(),
            end=series.index[-1].to_pydatetime(),
            freq=freq,
            value=series.values,
            name=series.name,
        )

    def __len__(self) -> int:
        """
        Get the length of the series.

        Returns:
            int: Number of data points.
        """
        return len(self.data)

    # Arithmetic Operations
    def __add__(self, other):
        """
        Add a scalar or array to the series.

        Parameters:
            other (float or np.ndarray): The value(s) to add.

        Returns:
            np.ndarray: The result of the addition.
        """
        return self.data + other

    def __sub__(self, other):
        """
        Subtract a scalar or array from the series.

        Parameters:
            other (float or np.ndarray): The value(s) to subtract.

        Returns:
            np.ndarray: The result of the subtraction.
        """
        return self.data - other

    def __truediv__(self, other: float):
        """
        Divide the series by a scalar.

        Parameters:
            other (float): The scalar to divide by.

        Returns:
            np.ndarray: The result of the division.
        """
        return self.data / other

    def __mul__(self, other: float):
        """
        Multiply the series by a scalar.

        Parameters:
            other (float): The scalar to multiply by.

        Returns:
            np.ndarray: The result of the multiplication.
        """
        return self.data * other

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

    # Properties
    @property
    def index(self) -> list:
        """
        Get the index of the series as a list of datetime objects.

        Returns:
            list of datetime: The index of the series.
        """
        return self.get_date_list()

    @property
    def dtype(self):
        """
        Get the data type of the series.

        Returns:
            dtype: The data type of the underlying NumPy array.
        """
        return self.data.dtype if self._data is not None else None

    def __contains__(self, other: datetime) -> bool:
        """
        Check if a datetime is within the series and aligned with the frequency within tolerance.

        Parameters:
            other (datetime): The datetime to check.

        Returns:
            bool: True if contained and aligned within tolerance, False otherwise.
        """
        if self.start > other or self.end < other:
            return False
        try:
            self.get_idx_from_date(other)
            return True
        except ValueError:
            return False

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
            start=self.start,
            end=self.end,
            freq=self.freq,
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
        return FastDatetimeSeries(self.start, self.end, self.freq, value, name)

    def __repr__(self):
        """
        Official string representation of the FastDatetimeSeries, showing key metadata and sample data.
        """
        # Initialize data if it's not already done
        self.init_data()

        # Show a small preview of the data (similar to pandas Series)
        preview_length = 10  # number of elements to preview
        data_preview = self.data[:preview_length]
        dates_preview = pd.to_datetime(self.get_date_list())[:preview_length]

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
            start=index_series.start,
            end=index_series.end,
            freq=index_series.freq,
            value=value,
            name=name if name else index_series.name,
        )
