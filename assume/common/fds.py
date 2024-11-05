# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas._libs.tslibs import to_offset


def freq_to_timedelta(freq: str) -> timedelta:
    return timedelta(minutes=to_offset(freq).nanos / 1e9 // 60)


class FastDatetimeSeries:
    def __init__(self, start, end, freq, value=0.0, name=""):
        self.start = start
        self.end = end
        self.freq = freq_to_timedelta(freq)
        self.data = None
        self.loc = self  # allow adjusting loc as well
        self.at = self
        self.value = value
        self.name = name

    def init_data(self):
        if self.data is None:
            self.data = self.date_range_idx(self.start, self.end, self.freq, self.value)

    def __getitem__(self, item: slice):
        self.init_data()
        if isinstance(item, slice):
            start = self.idx_from_date(item.start)
            # stop should be inclusive
            # to comply with pandas behavior
            stop = self.idx_from_date(item.stop) + 1
            return self.data[start:stop]
        else:
            start = self.idx_from_date(item)
            if start >= len(self.data):
                return 0
            else:
                return self.data[start]

    def __setitem__(self, item, value):
        self.init_data()
        if isinstance(item, slice):
            start = self.idx_from_date(item.start)
            stop = self.idx_from_date(item.stop)
            self.data[start:stop] = value
        else:
            start = self.idx_from_date(item)
            self.data[start] = value

    def dt_index(self, start = None, end = None):
        self.init_data()
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        hour_count = self.idx_from_date(end) + 1

        return [start + i * self.freq for i in range(hour_count)]

    def idx_from_date(self, date: datetime):
        if not date: 
            return None
        idx = (date - self.start) // self.freq
        # todo check if modulo is 0 - else this is not a valid multiple of freq
        #if idx < 0:
        #    raise ValueError("date %s is before start %s", date, self.start)
        return idx

    def date_range_idx(self, start, end, freq, value=0):
        """
        Get the index of a date range.
        """
        hour_count = self.idx_from_date(end) +1
        return np.full(hour_count, value)

    def as_df(self, name, start = None, end = None):
        self.init_data()
        return pd.DataFrame(self[start:end], index=self.dt_index(start,end), columns=[self.name])

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