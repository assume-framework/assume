# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import Orderbook
from assume.markets.base_market import MarketRole

from .all_or_nothing import PayAsBidAonRole, PayAsClearAonRole
from .complex_clearing import ComplexClearingRole
from .complex_clearing_dmas import ComplexDmasClearingRole
from .nodal_pricing import NodalPyomoMarketRole
from .simple import PayAsBidRole, PayAsClearRole

clearing_mechanisms: dict[str, MarketRole] = {
    "pay_as_clear": PayAsClearRole,
    "pay_as_bid": PayAsBidRole,
    "pay_as_clear_complex": ComplexClearingRole,
    "pay_as_clear_complex_dmas": ComplexDmasClearingRole,
    "pay_as_bid_aon": PayAsBidAonRole,
    "pay_as_clear_aon": PayAsClearAonRole,
    "nodal_pricing": NodalPyomoMarketRole,
}
