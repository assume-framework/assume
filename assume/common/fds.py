# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd



def freq_to_timedelta(freq: str) -> timedelta:
    return timedelta(minutes=pd.tseries.frequencies.to_offset(freq).nanos / 1e9 // 60)


class FastDatetimeSeries:
    def __init__(self, start, end, freq, value=None, name=""):
        self.start = start
        self.end = end
        self.freq = freq_to_timedelta(freq)
        self.data = None
        self.loc = self  # allow adjusting loc as well
        self.at = self
        if value is not None:
            self.init_data(value)
        # the name is not actually used other than for the compatibility of pandas import/export
        self.name = name

    def init_data(self, value=None):
        if self.data is None:
            self.data = self.get_numpy_date_range(self.start, self.end, self.freq, value)

    def __getitem__(self, item: slice):
        self.init_data()
        if isinstance(item, slice):
            start = self.get_idx_from_date(item.start)
            # stop should be inclusive
            # to comply with pandas behavior
            stop = self.get_idx_from_date(item.stop) + 1
            return self.data[start:stop]
        else:
            start = self.get_idx_from_date(item)
            if start >= len(self.data):
                return 0
            else:
                return self.data[start]

    def __setitem__(self, item, value):
        self.init_data()
        if isinstance(item, slice):
            start = self.get_idx_from_date(item.start)
            stop = self.get_idx_from_date(item.stop)
            self.data[start:stop] = value
        else:
            start = self.get_idx_from_date(item)
            self.data[start] = value

    def get_date_list(self, start = None, end = None):
        self.init_data()
        if start is None or start < self.start:
            start = self.start
        if end is None or end > self.end:
            end = self.end
        # takes next start value at most
        start_idx = self.get_idx_from_date(start)
        hour_count = self.get_idx_from_date(end) +1 - start_idx
        
        start = self.start + start_idx * self.freq

        return [start + i * self.freq for i in range(hour_count)]

    def get_idx_from_date(self, date: datetime):
        if not date: 
            return None
        idx = (date - self.start) / self.freq
        # todo check if modulo is 0 - else this is not a valid multiple of freq
        #if idx < 0:
        #    raise ValueError("date %s is before start %s", date, self.start)
        return math.ceil(idx)

    def get_numpy_date_range(self, start, end, freq, value=None):
        """
        Get the index of a date range.
        """
        if value is None:
            value = 0.0
        hour_count = self.get_idx_from_date(end) +1
        return np.full(hour_count, value)

    def as_df(self, name, start = None, end = None):
        self.init_data()
        return pd.DataFrame(self[start:end], index=self.get_date_list(start,end), columns=[self.name])

    @staticmethod
    def from_series(series):
        if series.index.freq:
            freq = series.index.freq
        else:
            freq = series.index[1] - series.index[0]
        return FastDatetimeSeries(series.index[0], series.index[-1], freq, series.values, series.name)
    
    def __truediv__(self, other: float):
        self.init_data()
        return self.data / other

    def __mul__(self, other: float):
        self.init_data()
        return self.data * other

    def __len__(self):
        self.init_data()
        return len(self.data)

    def copy_empty(self, value=0.0, name=""):
        return FastDatetimeSeries(self.start, self.end, self.freq, value, name)