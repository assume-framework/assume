# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import Orderbook
from assume.markets.base_market import MarketRole

from .contracts import PayAsBidContractRole
from .simple import PayAsBidRole, PayAsClearRole
from .complex_clearing import ComplexClearingRole
from .complex_clearing_dmas import ComplexDmasClearingRole

clearing_mechanisms: dict[str, MarketRole] = {
    "pay_as_clear": PayAsClearRole,
    "pay_as_bid": PayAsBidRole,
    "pay_as_bid_contract": PayAsBidContractRole,
    "complex_clearing": ComplexClearingRole,
    "pay_as_clear_complex_dmas": ComplexDmasClearingRole,
}

# try importing pypsa if it is installed
try:
    from .redispatch import RedispatchMarketRole

    clearing_mechanisms["redispatch"] = RedispatchMarketRole
except ImportError:
    pass
