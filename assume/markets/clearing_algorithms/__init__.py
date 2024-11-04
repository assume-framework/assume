# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import Orderbook
from assume.markets.base_market import MarketRole

from .all_or_nothing import PayAsBidAonRole, PayAsClearAonRole
from .contracts import PayAsBidContractRole
from .simple import PayAsBidRole, PayAsClearRole, PayAsBidBuildingRole, PayAsClearBuildingRole
from .complex_clearing import ComplexClearingRole
from .complex_clearing_dmas import ComplexDmasClearingRole

clearing_mechanisms: dict[str, MarketRole] = {
    "pay_as_clear": PayAsClearRole,
    "pay_as_clear_building": PayAsBidBuildingRole,
    "pay_as_bid": PayAsBidRole,
    "pay_as_bid_building": PayAsBidBuildingRole,
    "pay_as_bid_aon": PayAsBidAonRole,
    "pay_as_clear_aon": PayAsClearAonRole,
    "pay_as_bid_contract": PayAsBidContractRole,
    "pay_as_clear_complex": ComplexClearingRole,
    "pay_as_clear_complex_dmas": ComplexDmasClearingRole,
}

# try importing pypsa if it is installed
try:
    from .nodal_pricing import NodalMarketRole
    from .redispatch import RedispatchMarketRole

    clearing_mechanisms["redispatch"] = RedispatchMarketRole
    clearing_mechanisms["nodal"] = NodalMarketRole

except ImportError:
    pass
