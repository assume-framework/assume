# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseUnit
from assume.units.demand import Demand
from assume.units.powerplant import PowerPlant
from assume.units.storage import Storage

unit_types: dict[str, BaseUnit] = {
    "power_plant": PowerPlant,
    "demand": Demand,
    "storage": Storage,
}

try:
    from assume.units.steel_plant import SteelPlant

    unit_types["steel_plant"] = SteelPlant
except ImportError:
    pass
