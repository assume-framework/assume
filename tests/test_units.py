# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pandas as pd

from assume.common.base import SupportsMinMax, SupportsMinMaxCharge
from assume.common.forecasts import NaiveForecast


def test_minmax():
    index = pd.date_range("2022-01-01", periods=24, freq="h")
    forecaster = NaiveForecast(index, availability=1, price_forecast=50)

    mm = SupportsMinMax(
        id="Test",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        forecaster=forecaster,
        node="empty",
    )

    mm.ramp_down = 200
    mm.ramp_up = 400
    mm.max_power = 1000
    mm.min_power = 200

    # stay turned off
    assert mm.calculate_ramp(op_time=1, previous_power=0, power=0, current_power=0) == 0
    # ramp up only possible to min_power
    assert (
        mm.calculate_ramp(op_time=1, previous_power=0, power=190, current_power=0)
        == 200
    )
    # ramp up 400
    assert (
        mm.calculate_ramp(op_time=1, previous_power=0, power=1200, current_power=0)
        == 400
    )
    # should not ramp up, if max_power already sold
    assert (
        mm.calculate_ramp(op_time=1, previous_power=0, power=800, current_power=400)
        == 0
    )
    # ramp up to 800
    assert (
        mm.calculate_ramp(op_time=1, previous_power=400, power=1200, current_power=0)
        == 800
    )
    # ramp up to max_power
    assert (
        mm.calculate_ramp(op_time=1, previous_power=800, power=1200, current_power=0)
        == mm.max_power
    )
    # can't sell more if already sold
    assert (
        mm.calculate_ramp(op_time=1, previous_power=800, power=1200, current_power=1000)
        == 0
    )

    # reduce output
    assert (
        mm.calculate_ramp(op_time=1, previous_power=1000, power=800, current_power=0)
        == 800
    )
    # use float
    assert (
        mm.calculate_ramp(op_time=1, previous_power=800, power=753.2, current_power=0)
        == 753.2
    )
    # check ramp down constraint
    assert (
        mm.calculate_ramp(op_time=1, previous_power=800, power=500, current_power=0)
        == 600
    )

    assert (
        mm.calculate_ramp(op_time=1, previous_power=800, power=500, current_power=0)
        == 600
    )
    # check min_operating_time constraint
    mm.min_operating_time = 2
    mm.outputs["energy"][mm.index[0]] = 0
    mm.outputs["energy"][mm.index[1]] = 200
    op_time = mm.get_operation_time(mm.index[2])

    assert op_time == 1
    assert (
        mm.calculate_ramp(op_time=op_time, previous_power=200, power=0, current_power=0)
        == 200
    )
    assert (
        mm.calculate_ramp(
            op_time=op_time, previous_power=200, power=400, current_power=0
        )
        == 400
    )

    # check min_down_time constraint
    mm.min_down_time = 2
    mm.outputs["energy"][mm.index[2]] = 0
    op_time = mm.get_operation_time(mm.index[3])

    assert op_time == -1
    assert (
        mm.calculate_ramp(op_time=op_time, previous_power=0, power=400, current_power=0)
        == 0
    )
    assert (
        mm.calculate_ramp(op_time=op_time, previous_power=0, power=0, current_power=0)
        == 0
    )


def test_minmaxcharge():
    index = pd.date_range("2022-01-01", periods=24, freq="h")
    forecaster = NaiveForecast(index, availability=1, price_forecast=50)

    mmc = SupportsMinMaxCharge(
        id="Test",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        forecaster=forecaster,
        node="empty",
    )

    mmc.ramp_down_charge = -100
    mmc.ramp_down_discharge = 100
    mmc.ramp_up_charge = -100
    mmc.ramp_up_discharge = 100
    mmc.max_power_charge = -1000
    mmc.max_power_discharge = 1000
    mmc.min_power_charge = 0
    mmc.min_power_discharge = 0

    # stay turned off
    assert (
        mmc.calculate_ramp_charge(previous_power=0, power_charge=0, current_power=0)
        == 0
    )

    # stay turned off
    assert (
        mmc.calculate_ramp_discharge(
            previous_power=0, power_discharge=0, current_power=0
        )
        == 0
    )

    # calculate ramping
    # should be previous_power + 100
    assert mmc.calculate_ramp_discharge(0, 992, 0) == 100
    assert mmc.calculate_ramp_discharge(-10, 992, 0) == 100

    assert mmc.calculate_ramp_charge(-10, -200, 0) == -110


def test_minmaxcharge_unconstrained():
    index = pd.date_range("2022-01-01", periods=24, freq="h")
    forecaster = NaiveForecast(index, availability=1, price_forecast=50)

    mmc = SupportsMinMaxCharge(
        id="Test",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        forecaster=forecaster,
        node="empty",
    )

    # 1. wenn ramp is undefined, it should not create constraints
    # 2. all maximum/minimum and ramp values should be given as positive values

    mmc.max_power_charge = -1000  # MW
    mmc.max_power_discharge = 1000  # MW
    # ramp constraints
    mmc.ramp_up_charge = -2000
    mmc.ramp_down_charge = -2000
    mmc.ramp_up_discharge = 2000
    mmc.ramp_down_discharge = 2000
    # min power
    mmc.min_power_charge = -0
    mmc.min_power_discharge = 0

    # calculate ramping
    # should be previous_power + max_power
    assert mmc.calculate_ramp_discharge(0, 992, 0) == 992
    assert mmc.calculate_ramp_discharge(-10, 992, 0) == 992

    assert mmc.calculate_ramp_charge(0, -992, 0) == -992
    assert mmc.calculate_ramp_charge(-10, -992, 0) == -992


def test_minmax_operationtime():
    index = pd.date_range(
        start=datetime(2023, 7, 1), end=datetime(2023, 7, 2), freq="1h"
    )
    forecaster = NaiveForecast(index, availability=1, price_forecast=50)

    mm = SupportsMinMax(
        id="Test",
        unit_operator="TestOperator",
        technology="TestTechnology",
        bidding_strategies={},
        forecaster=forecaster,
        node="empty",
    )

    mm.outputs["energy"] += 500
    mm.min_down_time = 4
    mm.min_operating_time = 4
    runtime = mm.get_operation_time(datetime(2023, 7, 2))
    assert runtime > 0

    mm.outputs["energy"][-4:] = 0
    runtime = mm.get_operation_time(datetime(2023, 7, 2))
    assert runtime == -3

    mm.outputs["energy"][-2:] = 1000
    runtime = mm.get_operation_time(datetime(2023, 7, 2))
    assert runtime == 1

    mm.outputs["energy"][:] = 400
    mm.outputs["energy"][-1:] = 0
    runtime = mm.get_operation_time(datetime(2023, 7, 2))
    assert runtime == 4
