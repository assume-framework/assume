# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Day, Tick

# tolerance for floating point errors when mapping dates onto the index grid
_ALIGNMENT_EPSILON_SECONDS = 1e-6


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
            freq (timedelta | str, optional): The fixed step of the index, as timedelta or pandas offset alias
                (e.g. "15min", "h" or "D"). Calendar frequencies such as "W" or "MS" are not supported.
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
            # datetime bounds are clipped to the index range like in pandas
            start_idx = (
                max(0, self._get_idx_from_date(item.start))
                if isinstance(item.start, datetime)
                else item.start or 0
            )
            stop_idx = (
                max(0, self._get_idx_from_date(item.stop, round_up=False) + 1)
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
        return self._is_aligned(date)

    def _is_aligned(self, date: datetime) -> bool:
        """Check whether a datetime lies exactly on the index grid."""
        delta_seconds = (date - self.start).total_seconds()
        remainder = delta_seconds % self.freq_seconds
        return (
            remainder <= _ALIGNMENT_EPSILON_SECONDS
            or self.freq_seconds - remainder <= _ALIGNMENT_EPSILON_SECONDS
        )

    def get_position(self, date: datetime) -> int:
        """
        Get the position of a single datetime in the index.

        Like label-based lookups in pandas, the datetime has to match an index date exactly.

        Raises:
            KeyError: If the datetime lies outside the index range or is not aligned with the frequency.
        """
        if not self._is_aligned(date):
            raise KeyError(
                f"{date} is not aligned with the index frequency {self.freq}"
            )
        idx = self._get_idx_from_date(date)
        if idx < 0 or idx >= self._count:
            raise KeyError(
                f"{date} is outside of the index range {self.start} - {self.end}"
            )
        return idx

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

        start_idx = max(0, self._get_idx_from_date(start or self.start))
        end_idx = max(0, self._get_idx_from_date(end or self.end, round_up=False) + 1)
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

        The result is not bounds-checked and can lie outside of the index range.

        Parameters:
            date (datetime.datetime): The datetime to convert.
            round_up (bool): For dates between two index dates, return the next date (True)
                or the previous date (False).

        Returns:
            int: The index of the datetime in the index range.

        Raises:
            KeyError: If the input `date` is None.
        """
        if date is None:
            raise KeyError("Date cannot be None. Please provide a valid datetime.")

        delta_seconds = (date - self.start).total_seconds()
        idx, remainder = divmod(delta_seconds, self.freq_seconds)
        idx = int(idx)

        if remainder <= _ALIGNMENT_EPSILON_SECONDS:
            return idx
        if self.freq_seconds - remainder <= _ALIGNMENT_EPSILON_SECONDS:
            return idx + 1
        return idx + 1 if round_up else idx

    def align_up(self, date: datetime) -> datetime:
        """
        Rounds ``date`` up to the first step of the index at or after it.

        The result may lie beyond the end of the index.

        Parameters:
            date (datetime.datetime): The datetime to align.

        Returns:
            datetime.datetime: The first step at or after ``date``.
        """
        return self.start + self._get_idx_from_date(date) * self.freq

    @staticmethod
    def _convert_to_datetime(value: datetime | str) -> datetime:
        """Convert input to datetime if it's not already."""
        return value if isinstance(value, datetime) else pd.to_datetime(value)

    @staticmethod
    def _parse_frequency(freq: timedelta | str) -> timedelta:
        """
        Parse a frequency input into a timedelta.

        Parameters:
            freq (timedelta | str): Frequency as timedelta or fixed pandas offset alias (e.g. "15min", "h", "D").

        Returns:
            timedelta: The parsed frequency.

        Raises:
            TypeError: If the input type is not supported.
            ValueError: If the string format is invalid or not a fixed frequency (e.g. monthly).
        """
        if isinstance(freq, timedelta):
            return freq
        if isinstance(freq, str):
            try:
                offset = to_offset(freq)
            except ValueError as e:
                raise ValueError(f"Invalid frequency string: {freq}. Error: {e}")
            # Day is no Tick in pandas>=3, but has a fixed length for naive datetimes
            if not isinstance(offset, Tick | Day):
                raise ValueError(
                    f"Frequency {freq} is not fixed, use e.g. '15min', 'h' or 'D'."
                )
            return pd.Timedelta(offset.nanos, unit="ns")
        raise TypeError("Frequency must be a string or timedelta")

    @staticmethod
    def from_pandas_index(idx: pd.DatetimeIndex | pd.Series):
        return FastIndex(start=idx[0], end=idx[-1], freq=pd.infer_freq(idx))
