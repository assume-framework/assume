# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from importlib.metadata import version

from assume.common import MarketConfig, MarketProduct
from assume.scenario.loader_csv import (
    load_custom_units,
    load_scenario_folder,
    run_learning,
)
from assume.world import World

__version__ = version("assume-framework")

__author__ = "ASSUME Developers: Nick Harder, Kim Miskiw, Florian Maurer, Manish Khanra"
__copyright__ = "AGPL-3.0 License"
