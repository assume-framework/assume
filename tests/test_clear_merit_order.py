# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import numpy as np

from assume.markets.clearing_algorithms.simple import clear_merit_order


def test_horizontal_overlap_maximises_volume():
    # gen 100@0, 100@10 and dem 60@20, 60@10, 60@0 overlap at 10 between 100 and 120 MW
    price, supply, demand = clear_merit_order(
        supply_price=[0, 10],
        supply_volume=[100, 100],
        demand_price=[20, 10, 0],
        demand_volume=[60, 60, 60],
    )
    assert price == 10
    assert supply.tolist() == [100, 20]
    assert demand.tolist() == [60, 60, 0]


def test_results_in_input_order():
    price, supply, demand = clear_merit_order(
        supply_price=[50, 10, 30],
        supply_volume=[100, 100, 100],
        demand_price=[1000],
        demand_volume=[150],
    )
    assert price == 30
    assert supply.tolist() == [0, 100, 50]
    assert demand.tolist() == [150]


def test_partially_accepted_demand_sets_price():
    # the supply at 50 is too expensive for the second demand bid, which is partially accepted
    price, supply, demand = clear_merit_order(
        supply_price=[10, 50],
        supply_volume=[100, 100],
        demand_price=[100, 40],
        demand_volume=[60, 60],
    )
    assert price == 40
    assert supply.tolist() == [100, 0]
    assert demand.tolist() == [60, 40]


def test_supply_exhausted_with_demand_left_over():
    # the supply is exhausted exactly with the first demand bid, the second one is not served
    price, supply, demand = clear_merit_order(
        supply_price=[10],
        supply_volume=[100],
        demand_price=[100, 40],
        demand_volume=[100, 60],
    )
    assert price == 100
    assert demand.tolist() == [100, 0]


def test_unserved_demand_does_not_set_price():
    # the second demand bid is not accepted at all and the supply is not exhausted
    price, supply, demand = clear_merit_order(
        supply_price=[10, 50],
        supply_volume=[100, 100],
        demand_price=[100, 40],
        demand_volume=[100, 60],
    )
    assert price == 10
    assert supply.tolist() == [100, 0]
    assert demand.tolist() == [100, 0]


def test_no_trade_and_empty_sides():
    assert clear_merit_order([50], [100], [10], [100])[0] == 0
    price, supply, demand = clear_merit_order([], [], [10], [100])
    assert price == 0 and len(supply) == 0 and demand.tolist() == [0]
    price, supply, demand = clear_merit_order([10], [100], [], [])
    assert price == 0 and supply.tolist() == [0] and len(demand) == 0


def test_float_residue_is_tolerated():
    # 0.1 + 0.2 != 0.3 in floating point; the demand bid is still fully accepted,
    # so the next (unserved) demand bid must not count as partially accepted
    price, supply, demand = clear_merit_order(
        supply_price=[10, 20, 90],
        supply_volume=[0.1, 0.2, 1],
        demand_price=[100, 50],
        demand_volume=[0.3, 1],
    )
    assert price == 20
    assert demand.tolist() == [0.3, 0]
    assert supply.tolist() == [0.1, 0.2, 0]


def test_zero_volume_bids_are_ignored():
    price, supply, demand = clear_merit_order(
        supply_price=[0, 10],
        supply_volume=[0, 100],
        demand_price=[100],
        demand_volume=[50],
    )
    assert price == 10
    assert supply.tolist() == [0, 50]


def test_random_tie_breaking_is_seedable():
    def clear(seed):
        return clear_merit_order(
            supply_price=[10, 10, 10, 10],
            supply_volume=[100, 100, 100, 100],
            demand_price=[100],
            demand_volume=[150],
            rng=np.random.default_rng(seed),
        )[1].tolist()

    assert clear(1) == clear(1)
    outcomes = {tuple(clear(seed)) for seed in range(20)}
    assert len(outcomes) > 1
    assert all(sum(o) == 150 for o in outcomes)
    # without rng, ties keep the input order
    assert clear_merit_order([10, 10], [100, 100], [100], [150])[1].tolist() == [
        100,
        50,
    ]
