# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
from assume.common.forecast_algorithms import custom_lru_cache

def test_custom_lru_cache_basic():
    calls = 0

    @custom_lru_cache
    def add(a, b):
        nonlocal calls
        calls += 1
        return a + b

    assert add(1, 2) == 3
    assert calls == 1
    assert add(1, 2) == 3
    assert calls == 1
    assert add(2, 3) == 5
    assert calls == 2

def test_custom_lru_cache_unhashable():
    calls = 0

    @custom_lru_cache
    def get_len(lst):
        nonlocal calls
        calls += 1
        return len(lst)

    l1 = [1, 2, 3]
    l2 = [1, 2, 3]

    assert get_len(l1) == 3
    assert calls == 1
    assert get_len(l1) == 3
    assert calls == 1  # Hit cache because same id
    assert get_len(l2) == 3
    assert calls == 2  # Miss cache because different id

def test_custom_lru_cache_maxsize():
    calls = 0

    @custom_lru_cache(maxsize=1)
    def add(a, b):
        nonlocal calls
        calls += 1
        return a + b

    assert add(1, 2) == 3
    assert calls == 1
    assert add(2, 3) == 5
    assert calls == 2
    assert add(1, 2) == 3
    assert calls == 3  # Cache evicted due to maxsize=1
