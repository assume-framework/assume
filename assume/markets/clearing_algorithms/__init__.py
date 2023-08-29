from assume.common.market_objects import MarketMechanism, Orderbook
from assume.markets.base_market import MarketRole

from .all_or_nothing import pay_as_bid_aon, pay_as_clear_aon
from .complex_clearing import pay_as_clear_complex
from .complex_clearing_dmas import complex_clearing_dmas
from .nodal_pricing import nodal_pricing_pyomo
from .simple import PayAsBidRole, PayAsClearRole


class MarketMechanismRole(MarketRole, MarketMechanism):
    def __init__(self, marketconfig):
        super().__init__(marketconfig)
        self.legacy_mechanism = legacy_mechanism

    def clear(orderbook: Orderbook) -> (Orderbook, Orderbook, list[dict]):
        self.legacy_mechanism(self, orderbook)


clearing_mechanisms: dict[str, MarketRole] = {
    "pay_as_clear": PayAsClearRole,
    "pay_as_bid": PayAsBidRole,
    # "pay_as_bid_all_or_nothing": MarketMechanismRole(pay_as_bid_aon),
    # "pay_as_clear_all_or_nothing": MarketMechanismRole(pay_as_clear_aon),
    # "nodal_pricing_pyomo": MarketMechanismRole(nodal_pricing_pyomo),
    # "pay_as_clear_complex": MarketMechanismRole(pay_as_clear_complex),
    # "pay_as_clear_complex_dmas": complex_clearing_dmas,
}
