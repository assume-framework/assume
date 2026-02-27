# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from assume.common.forecast_algorithms import custom_lru_cache


def test_custom_lru_cache_basic():
    @custom_lru_cache
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add.cache_info().hits == 0 and add.cache_info().misses == 1
    assert add(1, 2) == 3
    assert add.cache_info().hits == 1 and add.cache_info().misses == 1
    assert add(2, 3) == 5
    assert add.cache_info().hits == 1 and add.cache_info().misses == 2


def test_custom_lru_cache_unhashable():
    @custom_lru_cache
    def get_len(lst):
        return len(lst)

    l1 = [1, 2, 3]
    l2 = [1, 2, 3]

    assert get_len(l1) == 3
    assert get_len.cache_info().hits == 0 and get_len.cache_info().misses == 1
    assert get_len(l1) == 3
    assert get_len.cache_info().hits == 1 and get_len.cache_info().misses == 1
    assert get_len(l2) == 3
    assert get_len.cache_info().hits == 1 and get_len.cache_info().misses == 2


def test_custom_lru_cache_maxsize():
    @custom_lru_cache(maxsize=1)
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add.cache_info().hits == 0 and add.cache_info().misses == 1
    assert add(2, 3) == 5
    assert add.cache_info().hits == 0 and add.cache_info().misses == 2
    assert add(1, 2) == 3
    assert add.cache_info().hits == 0 and add.cache_info().misses == 3
