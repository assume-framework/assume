# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseUnit, BaseDSTComponent
from assume.units.demand import Demand
from assume.units.powerplant import PowerPlant
from assume.units.storage import Storage

unit_types: dict[str, BaseUnit] = {
    "power_plant": PowerPlant,
    "demand": Demand,
    "storage": Storage,
}
demand_side_components: dict[str, BaseDSTComponent] = {}

try:
    from assume.units.steel_plant import SteelPlant

    from assume.units.dst_components import (
        DRIPlant,
        GenericStorage,
        HydrogenStorage,
        DRIStorage,
        ElectricArcFurnace,
        Electrolyser,
        HeatPump,
        Boiler,
        ElectricVehicle,
        PVPlant,
    )

    unit_types["steel_plant"] = SteelPlant

    # Mapping of component type identifiers to their respective classes
    demand_side_components: dict[str, BaseDSTComponent] = {
        "electrolyser": Electrolyser,
        "hydrogen_storage": HydrogenStorage,
        "dri_plant": DRIPlant,
        "dri_storage": DRIStorage,
        "eaf": ElectricArcFurnace,
        "heat_pump": HeatPump,
        "boiler": Boiler,
        "electric_vehicle": ElectricVehicle,
        "generic_storage": GenericStorage,
        "pv_plant": PVPlant,
    }

except ImportError:
    pass
