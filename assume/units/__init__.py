# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseUnit
from assume.units.demand import Demand
from assume.units.powerplant import PowerPlant
from assume.units.storage import Storage
from assume.units.steel_plant import SteelPlant
from assume.units.hydrogen_plant import HydrogenPlant
from assume.units.dst_components import demand_side_technologies

unit_types: dict[str, BaseUnit] = {
    "power_plant": PowerPlant,
    "demand": Demand,
    "storage": Storage,
    "steel_plant": SteelPlant,
    "hydrogen_plant": HydrogenPlant,
}
