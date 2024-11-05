# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np
from pandas._libs.tslibs import to_offset


def freq_to_timedelta(freq: str) -> timedelta:
    return timedelta(minutes=to_offset(freq).nanos / 1e9 // 60)


class FastDatetimeSeries:
    def __init__(self, start, end, freq, value=0):
        self.start = start
        self.end = end
        self.freq = freq_to_timedelta(freq)
        self.data = None
        self.loc = self  # allow adjusting loc as well
        self.at = self
        self.value = value

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

    def dt_index(self):
        self.init_data()
        hour_count = self.idx_from_date(self.end)
        return [self.start + i * self.freq for i in range(hour_count)]

    def idx_from_date(self, date: datetime):
        if not date: return None
        return (date - self.start) // self.freq

    def date_range_idx(self, start, end, freq, value=0):
        """
        Get the index of a date range.
        """
        hour_count = self.idx_from_date(end)
        return np.full(hour_count, value)
