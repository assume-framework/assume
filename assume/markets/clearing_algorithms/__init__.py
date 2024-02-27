# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import Orderbook
from assume.markets.base_market import MarketRole

from .all_or_nothing import PayAsBidAonRole, PayAsClearAonRole
from .simple import PayAsBidRole, PayAsClearRole
from .contracts import PayAsBidContractRole

clearing_mechanisms: dict[str, MarketRole] = {
    "pay_as_clear": PayAsClearRole,
    "pay_as_bid": PayAsBidRole,
    "pay_as_bid_aon": PayAsBidAonRole,
    "pay_as_clear_aon": PayAsClearAonRole,
    "pay_as_bid_contract": PayAsBidContractRole,
}

# try importing pypsa if it is installed
try:
    from .complex_clearing import ComplexClearingRole
    from .complex_clearing_dmas import ComplexDmasClearingRole
    from .nodal_pricing import NodalPyomoMarketRole
    from .redispatch import RedispatchMarketRole

    clearing_mechanisms["pay_as_clear_complex"] = ComplexClearingRole
    clearing_mechanisms["pay_as_clear_complex_dmas"] = ComplexDmasClearingRole
    clearing_mechanisms["redispatch"] = RedispatchMarketRole
    clearing_mechanisms["nodal_pricing"] = NodalPyomoMarketRole

except ImportError:
    pass
