import pandas as pd

from assume.common.base import SupportsMinMaxCharge
from assume.units.storage import Storage


def test_storage():
    storage_unit = Storage(
        "Test_Storage",
        "TestOperator",
        "TestTechnolog",
        {},
        max_power_charge=-100,
        max_power_discharge=100,
        max_SOC=1000,
    )

    storage_unit.efficiency_charge = 0.9
    storage_unit.efficiency_discharge = 0.95

    start = pd.Timestamp("2019-01-01 00:00")
    end = pd.Timestamp("2019-01-01 01:00")

    storage_unit.index = pd.date_range(start, end, freq="1H")
    storage_unit.reset()

    # if no variable costs given, marginal costs should be 0
    assert storage_unit.calc_marginal_cost(timestep=start, discharge=True) == 0
    assert storage_unit.calc_marginal_cost(timestep=start, discharge=False) == 0

    storage_unit.outputs["energy"][start] = 10
    storage_unit.outputs["capacity_neg"][start] = -50
    storage_unit.outputs["capacity_pos"][start] = 30

    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )
    assert min_power_charge[0] == -40
    assert max_power_charge[0] == -60

    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    assert max_power_discharge[0] == 60

    storage_unit.current_SOC = 50
    min_power_discharge, max_power_discharge = storage_unit.calculate_min_max_discharge(
        start, end
    )
    assert min_power_discharge[0] == 40
    assert max_power_discharge[0] == round(50 * storage_unit.efficiency_discharge, 3)

    storage_unit.current_SOC = 950
    min_power_charge, max_power_charge = storage_unit.calculate_min_max_charge(
        start, end
    )
    assert min_power_charge[0] == -40
    assert max_power_charge[0] == round(-50 / storage_unit.efficiency_charge, 3)
